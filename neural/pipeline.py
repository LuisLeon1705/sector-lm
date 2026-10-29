from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

from neural.data import read_jsonl
from neural.model import Config, GPT, SwitchGPT, build_models, count_parameters, copy_shared_base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "round5"
CACHE_DIR = ROOT / "data" / "cache"
TOK_PATH = OUT / "tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPECTED_CHARS = 795700
N_EXPERTS = 12
MICRO_BATCH = 8
CONTEXT = 512
ACCUMULATION = 4
EFFECTIVE_BATCH_TOKENS = MICRO_BATCH * CONTEXT * ACCUMULATION
# Expert ids after merging: Rust ids 0..7 stay, AsyncConcurrency (8) joins GeneralLocal, 9..12 shift down.
DOMAIN_NAMES = [
    "ScientificNumeric", "MachineLearning", "WebBackend", "NetworkingCloud", "TestingQa",
    "DataStorageFormats", "SystemOsCli", "RuntimeInspectPkg", "GuiGraphicsMedia",
    "TextCryptoMath", "StdlibCoreTypes", "GeneralLocal",
]


def make_progress(total, label):
    return tqdm(total=total, desc=label, unit="tokens", dynamic_ncols=False, file=os.sys.stdout,
                disable=False, mininterval=2.0, miniters=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")


def train_tokenizer(train_path: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=16384, min_frequency=2, special_tokens=["<unk>", "<bos>"])
    nrows = sum(1 for _ in read_jsonl(train_path))

    def corpus():
        bar = tqdm(total=nrows, desc="tokenizer corpus", unit="files", file=os.sys.stdout, disable=False, mininterval=2.0, miniters=100)
        chars = 0
        start = time.monotonic()
        for row in read_jsonl(train_path):
            content = row["content"]
            chars += len(content)
            bar.update(1)
            elapsed = max(time.monotonic() - start, .001)
            if bar.n % 500 == 0:
                bar.set_postfix(tokens_per_s=f"{chars / elapsed:.0f}", eta_s=f"{(nrows-bar.n)*elapsed/max(bar.n,1):.0f}", refresh=False)
            yield content
        bar.close()

    tok.train_from_iterator(corpus(), trainer=trainer, length=nrows)
    tok.save(str(TOK_PATH))
    return tok


def load_or_train_tokenizer(train_path):
    if TOK_PATH.exists():
        return Tokenizer.from_file(str(TOK_PATH))
    return train_tokenizer(train_path)


def encode_corpus(path: Path, tok: Tokenizer, domain_file: Path, retain_records=True):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"cache_{path.stem}_{'records' if retain_records else 'train'}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("cache_schema") == 2 and cached.get("tokenizer_size") == tok.get_vocab_size() and cached.get("source_size") == path.stat().st_size:
            return cached["seqs"], cached["metadata"], cached["chars"]
    domains = json.loads(domain_file.read_text(encoding="utf-8"))
    rows = list(read_jsonl(path))
    seqs, metadata = [], []
    chars = 0
    bar = tqdm(total=len(rows), desc=f"encode {path.name}", unit="files", file=os.sys.stdout, disable=False, mininterval=2.0, miniters=100)
    started = time.monotonic()
    for i, row in enumerate(rows):
        content = row["content"]
        enc = tok.encode(content, add_special_tokens=False)
        ids = enc.ids
        if not ids:
            continue
        domain = int(domains.get(str(i), 12))
        # Rust merges source id 8 into general id 12; compact 0..7,9..12 into exactly 12 expert IDs.
        domain = 11 if domain == 8 else (domain - 1 if domain > 8 else domain)
        seqs.append((torch.tensor(ids, dtype=torch.int32), domain))
        if retain_records:
            metadata.append({"text": content, "ids": ids, "offsets": enc.offsets,
                             "domain": domain, "index": i})
        chars += len(content)
        bar.update(1)
        elapsed = max(time.monotonic() - started, .001)
        if bar.n % 500 == 0:
            encoded_tokens = sum(len(s) for s, _ in seqs)
            bar.set_postfix(tokens_per_s=f"{encoded_tokens/elapsed:.0f}", eta_s=f"{(len(rows)-bar.n)*elapsed/max(bar.n,1):.0f}", refresh=False)
    bar.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache_schema": 2, "tokenizer_size": tok.get_vocab_size(), "source_size": path.stat().st_size,
                "seqs": seqs, "metadata": metadata, "chars": chars}, cache_path)
    return seqs, metadata, chars


BOS_ID = 1  # "<bos>" is the second special token of the BPE trainer


def get_batch(seqs, batch_size, context, device, domain_filter=None):
    pool = [s for s in seqs if len(s[0]) > 1 and (domain_filter is None or s[1] == domain_filter)]
    if not pool:
        pool = [s for s in seqs if len(s[0]) > 1]
    xs, ys, dms = [], [], []
    valid_count = 0
    for _ in range(batch_size):
        s, source_domain = random.choice(pool)
        # Each file is read as <bos> + tokens, the same way evaluation starts a file.
        start = random.randrange(max(1, len(s) + 1 - context))
        if start == 0:
            chunk = torch.cat((s.new_tensor([BOS_ID]), s[:context])).to(torch.long)
        else:
            chunk = s[start - 1:start + context].to(torch.long)
        if len(chunk) < 2:
            chunk = torch.cat((chunk, chunk[-1:]))
        x = torch.zeros(context, dtype=torch.long)
        y = torch.full((context,), -100, dtype=torch.long)
        n = min(context, len(chunk) - 1)
        x[:n] = chunk[:n]
        y[:n] = chunk[1:n + 1]
        valid_count += n
        xs.append(x)
        ys.append(y)
        dms.append(source_domain)
    return torch.stack(xs).to(device), torch.stack(ys).to(device), torch.tensor(dms, device=device), valid_count


def limit_valid_targets(targets, remaining):
    """Keep only the next valid labels; padding is always ignore_index and unbudgeted."""
    flat = targets.reshape(-1)
    valid = torch.nonzero(flat != -100, as_tuple=False).flatten()
    keep = min(int(valid.numel()), int(remaining))
    if keep < valid.numel():
        flat[valid[keep:]] = -100
    return keep


def save_atomic(state, path):
    """Write to a temporary file and rename it, so a power cut never leaves a half-written checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def lr_multiplier(step, total_steps):
    warmup = 500
    if step < warmup:
        return max((step + 1) / warmup, 1e-5)
    span = max(total_steps - warmup, 1)
    progress = min(max((step - warmup) / span, 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_steps(model, seqs, steps, token_budget, label, save_path, adapter_mode=False, expert_id=None,
                seconds=None, base_only=False, lr_total_tokens=None, init_optimizer_path=None):
    model.to(DEVICE).train()
    micro_batch = 8
    accumulation = EFFECTIVE_BATCH_TOKENS // (micro_batch * CONTEXT)
    if adapter_mode:
        model.adapter_trainable()
        params = [p for p in model.parameters() if p.requires_grad]
    elif base_only:
        for p in model.parameters():
            p.requires_grad_(True)
        for p in model.experts.parameters():
            p.requires_grad_(False)
        params = [p for name, p in model.named_parameters() if not name.startswith("experts.")]
    else:
        params = list(model.parameters())
    base_params, lora_router_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("experts.") or ".router." in name:
            lora_router_params.append(param)
        else:
            base_params.append(param)
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": 1e-3})
    if lora_router_params:
        groups.append({"params": lora_router_params, "lr": 1e-3})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = math.ceil((lr_total_tokens or token_budget or max(steps, 1)*MICRO_BATCH*CONTEXT) / EFFECTIVE_BATCH_TOKENS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_multiplier(step, total_steps))
    nominal_total = token_budget or max(steps, 1) * micro_batch * CONTEXT
    pbar = make_progress(nominal_total, label)
    start = time.monotonic()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    done = 0
    step = 0
    checkpoint = Path(save_path)
    if init_optimizer_path and not checkpoint.exists():
        init_state = torch.load(init_optimizer_path, map_location=DEVICE, weights_only=False)
        opt.load_state_dict(init_state["optimizer"])
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            opt.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        done = int(state.get("tokens", 0))
        pbar.update(min(done, nominal_total))
    while (done < nominal_total) if token_budget else (step < steps or seconds is not None):
        if seconds and time.monotonic() - start >= seconds:
            break
        microbatches = []
        valid_remaining = max(0, nominal_total - done) if token_budget else EFFECTIVE_BATCH_TOKENS
        accum_target = min(EFFECTIVE_BATCH_TOKENS, valid_remaining)
        total_valid = 0
        attempts = 0
        while total_valid < accum_target and valid_remaining > 0 and attempts < accumulation * 8:
            attempts += 1
            x, y, domains, valid_count = get_batch(seqs, micro_batch, CONTEXT, DEVICE, expert_id if adapter_mode else None)
            if valid_count > valid_remaining:
                valid_count = limit_valid_targets(y, valid_remaining)
            microbatches.append((x, y, domains, valid_count))
            total_valid += valid_count
            valid_remaining -= valid_count
        if total_valid == 0:
            break
        opt.zero_grad(set_to_none=True)
        loss_values = []
        for x, y, domains, valid_count in microbatches:
            if not valid_count:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                if isinstance(model, SwitchGPT):
                    logits = model(x)
                    lm_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum")
                    loss = lm_sum / total_valid + 0.1 * model.last_router_aux / len(microbatches)
                else:
                    domain_arg = expert_id if adapter_mode else None
                    logits = model(x, expert_id=domain_arg, use_expert=not base_only)
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum") / total_valid
            loss.backward()
            loss_values.append(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        scheduler.step()
        done += total_valid
        step += 1
        pbar.update(min(total_valid, max(0, nominal_total - pbar.n)))
        elapsed = max(time.monotonic() - start, .001)
        loss_value = torch.stack(loss_values).sum().item()
        pbar.set_postfix(loss=f"{loss_value:.3f}", tokens_per_s=f"{done/elapsed:.0f}", eta_s=f"{max(0,nominal_total-done)/(done/elapsed):.0f}", lr=f"{opt.param_groups[0]['lr']:.2g}", refresh=False)
        if step % 100 == 0:
            save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(), "tokens": done}, checkpoint)
    pbar.close()
    save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(), "tokens": done}, checkpoint)
    elapsed = time.monotonic() - start
    return {"tokens": done, "optimizer_steps": step, "seconds": elapsed,
            "tokens_per_second": done / max(elapsed, 1e-9),
            "gpu_hours": elapsed / 3600 if DEVICE == "cuda" else 0.0,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if DEVICE == "cuda" else 0}


def char_categories(text):
    cats = ["format" if c.isspace() else "code" for c in text]
    try:
        import io, tokenize
        lines = text.splitlines(keepends=True)
        starts, acc = [], 0
        for line in lines:
            starts.append(acc); acc += len(line)
        for item in tokenize.generate_tokens(io.StringIO(text).readline):
            if item.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            (r0, c0), (r1, c1) = item.start, item.end
            a = starts[r0-1] + c0
            b = starts[r1-1] + c1
            cat = "comments" if item.type == tokenize.COMMENT else "strings"
            cats[a:b] = [cat] * max(0, min(b, len(cats)) - a)
    except (tokenize.TokenError, IndentationError):
        pass
    return cats


def eval_file(model, ids, offsets, text, expert_id=None, base_only=False, wrong_expert=None):
    model.eval()
    cats = char_categories(text)
    sums = Counter()
    chars = Counter()
    bos = BOS_ID
    total_nats = 0.0
    covered = 0  # byte-level BPE can split one character across tokens; count each character once
    context, stride = 512, 256
    for target_start in range(0, len(ids), stride):
        targets = ids[target_start:target_start + stride]
        history = ids[max(0, target_start - 256):target_start]
        # A single <bos> marks the file start, as in training; no run of padding tokens.
        prefix = ([bos] if target_start <= 256 else []) + history
        inp = prefix + targets[:-1]
        x = torch.tensor(inp, dtype=torch.long, device=DEVICE)[None, :]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            if isinstance(model, SwitchGPT):
                logits = model(x)
            else:
                route = wrong_expert if wrong_expert is not None else expert_id
                logits = model(x, expert_id=route, use_expert=not base_only)
            start_pos = len(prefix) - 1
            target_logits = logits[0, start_pos:start_pos + len(targets)].float()
            target = torch.tensor(targets, dtype=torch.long, device=DEVICE)
            nats = F.cross_entropy(target_logits, target, reduction="none").cpu().tolist()
        for j, (loss, (a, b)) in enumerate(zip(nats, offsets[target_start:target_start + len(targets)])):
            # Token gets one category: majority character coverage; deterministic priority on ties.
            seg = cats[a:b]
            if not seg:
                cat = "code"
            else:
                counts = Counter(seg)
                cat = max(("code", "comments", "strings", "format"), key=lambda c: (counts[c], ("code", "comments", "strings", "format").index(c)))
            width = max(0, b - max(a, covered))
            covered = max(covered, b)
            sums[cat] += loss
            chars[cat] += width
            total_nats += loss
    return {"nats": total_nats, "by_type_nats": dict(sums), "by_type_chars": dict(chars)}


def evaluate(model, records, label, mode, wrong_domains=None):
    totals = Counter(); chars = Counter(); all_nats = 0.0
    bar = tqdm(total=sum(len(r["ids"]) for r in records), desc=label, unit="tokens", file=os.sys.stdout, disable=False, mininterval=2.0, miniters=100)
    start = time.monotonic(); n_done = 0
    for rec in records:
        eid = rec["domain"]
        if mode == "base":
            result = eval_file(model, rec["ids"], rec["offsets"], rec["text"], base_only=True)
        elif wrong_domains is not None:
            result = eval_file(model, rec["ids"], rec["offsets"], rec["text"],
                               wrong_expert=wrong_domains[str(rec["index"])])
        else:
            result = eval_file(model, rec["ids"], rec["offsets"], rec["text"], expert_id=eid)
        all_nats += result["nats"]
        totals.update(result["by_type_nats"]); chars.update(result["by_type_chars"])
        n_done += len(rec["ids"]); bar.update(len(rec["ids"]))
        elapsed = max(time.monotonic()-start, .001)
        bar.set_postfix(tokens_per_s=f"{n_done/elapsed:.0f}", eta_s=f"{(bar.total-bar.n)*elapsed/max(bar.n,1):.0f}", refresh=False)
    bar.close()
    denom = sum(chars.values())
    if denom != EXPECTED_CHARS:
        raise RuntimeError(f"evaluation character coverage {denom} != {EXPECTED_CHARS}")
    return {"nats": all_nats, "characters": denom, "ppc": math.exp(all_nats/denom),
            "nats_per_character_by_type": {k: totals[k]/max(chars[k],1) for k in ("code","comments","strings","format")},
            "type_characters": dict(chars)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prepare", "smoke", "full", "eval"], default="prepare")
    ap.add_argument("--train-file", type=Path, default=ROOT/"data/round5/train.jsonl")
    ap.add_argument("--valid-file", type=Path, default=ROOT/"data/valid.jsonl")
    ap.add_argument("--test-file", type=Path, default=ROOT/"data/test.jsonl")
    ap.add_argument("--domain-dir", type=Path, default=ROOT/"data/round5_domains")
    ap.add_argument("--smoke-seconds", type=float, default=120)
    ap.add_argument("--models", default="D1,D1+,D2,E1,M1", help="comma-separated model subset, useful for resuming smoke runs")
    ap.add_argument("--epochs", type=float, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_or_train_tokenizer(args.train_file)
    train_seqs, train_meta, _ = encode_corpus(args.train_file, tok, args.domain_dir/"train_domains.json", retain_records=False)
    test_seqs, test_meta, test_chars = encode_corpus(args.test_file, tok, args.domain_dir/"test_domains.json")
    test_meta = test_meta[:100]
    if sum(len(r["text"]) for r in test_meta) != EXPECTED_CHARS:
        raise RuntimeError(f"expected {EXPECTED_CHARS} chars in first 100 test files")
    models = build_models(vocab_size=tok.get_vocab_size(), experts=N_EXPERTS)
    models["D1+"] = copy.deepcopy(models["D1"])
    counts = {k: count_parameters(v) for k,v in models.items()}
    router_params = sum(p.numel() for block in models["M1"].blocks for p in block.router.parameters())
    total_train_tokens = sum(len(s) for s, _ in train_seqs)
    print(json.dumps({"device": DEVICE, "train_tokens": total_train_tokens, "test_files": len(test_meta),
                      "test_characters": sum(len(r["text"]) for r in test_meta), "parameters": counts}, indent=2), flush=True)
    if args.mode == "prepare":
        return
    stats = {}
    if args.mode == "smoke":
        smoke_stats_path = OUT/"smoke_timing_round5c3.json"
        stats = json.loads(smoke_stats_path.read_text(encoding="utf-8")) if smoke_stats_path.exists() else {}
        selected_models = [name for name in args.models.split(",") if name]
        for name in selected_models:
            model = models[name]
            if name in ("E1", "D1+"):
                d1_smoke_ckpt = OUT/"checkpoints"/"round5c4"/"smoke_D1.pt"
                if not d1_smoke_ckpt.exists():
                    d1_smoke_ckpt = OUT/"checkpoints"/"round5c3"/"smoke_D1.pt"
                if d1_smoke_ckpt.exists():
                    models["D1"].load_state_dict(torch.load(d1_smoke_ckpt, map_location="cpu", weights_only=False)["model"])
                copy_shared_base(models["D1"], model)
            ckpt_root = OUT/"checkpoints"/"round5c4"
            if name == "E1":
                counts_domains = Counter(domain for _, domain in train_seqs)
                eid = counts_domains.most_common(1)[0][0]
                stats[name] = train_steps(model, train_seqs, 0, 10_000_000, f"E1 adapter {eid} smoke",
                    ckpt_root/"smoke_E1.pt", adapter_mode=True, expert_id=eid, seconds=args.smoke_seconds)
                stats[name]["smoke_domain"] = eid
            else:
                init_opt = ckpt_root/"smoke_D1.pt" if name == "D1+" else None
                stats[name] = train_steps(model, train_seqs, 0, 20_000_000, f"{name} smoke",
                    ckpt_root/f"smoke_{name.replace('+','plus')}.pt", seconds=args.smoke_seconds,
                    init_optimizer_path=init_opt)
        full_tokens = total_train_tokens
        estimates = {}
        for name, measured in stats.items():
            rate = measured["tokens_per_second"]
            tokens = full_tokens if name in ("D1+", "E1") else full_tokens*args.epochs
            estimates[name] = {"full_training_tokens": int(tokens), "measured_smoke_tokens_per_second": rate,
                               "estimated_gpu_hours": tokens / max(rate, 1) / 3600,
                               "smoke_peak_gpu_memory_bytes": measured.get("peak_gpu_memory_bytes")}
        total_hours = sum(v["estimated_gpu_hours"] for v in estimates.values())
        estimate = {"assumed_epochs": args.epochs, "adapter_epochs": 1,
                    "estimate_method": "full budgets divided by each corrected smoke throughput; D1+ adds N train tokens and E1 trains N adapter tokens on the D1 checkpoint; excludes eval/data prep",
                    "parameters": counts,
                    "router_parameters_M1": router_params,
                    "active_parameters_per_token": {"D1": counts["D1"], "D1+": counts["D1"], "D2": counts["D2"],
                        "E1": counts["D1"] + (counts["E1"]-counts["D1"])//N_EXPERTS,
                        "M1": counts["D1"] + (counts["M1"]-counts["D1"]-router_params)//N_EXPERTS + router_params},
                    "per_model": estimates, "total_sequential_gpu_hours": total_hours}
        (OUT/"full_training_estimate.json").write_text(json.dumps(estimate, indent=2), encoding="utf-8")
        print("FULL TRAINING ESTIMATE (full run not started):", json.dumps(estimate, indent=2), flush=True)
        (OUT/"smoke_timing.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2), flush=True)
        return
    if args.mode == "full":
        budget = int(total_train_tokens * args.epochs)
        ckpt_root = OUT/"checkpoints"/"round5c4"
        stats = {}
        stats["D1"] = train_steps(models["D1"], train_seqs, 0, budget, "D1", ckpt_root/"D1.pt", lr_total_tokens=budget)
        copy_shared_base(models["D1"], models["D1+"])
        stats["D1+"] = train_steps(models["D1+"], train_seqs, 0, total_train_tokens, "D1+ continuation", ckpt_root/"D1plus.pt",
            lr_total_tokens=total_train_tokens, init_optimizer_path=ckpt_root/"D1.pt")
        copy_shared_base(models["D1"], models["E1"])
        per_domain = Counter()
        for sequence, domain in train_seqs:
            per_domain[domain] += len(sequence)
        adapters = {}
        domain_ids = list(range(N_EXPERTS))
        remainder = total_train_tokens
        for i, eid in enumerate(domain_ids):
            share = remainder if i == len(domain_ids)-1 else int(total_train_tokens * per_domain[eid]/sum(per_domain.values()))
            remainder -= share
            adapters[str(eid)] = train_steps(models["E1"], train_seqs, 0, share, f"E1 adapter {eid}",
                ckpt_root/f"E1_domain_{eid}.pt", adapter_mode=True, expert_id=eid, lr_total_tokens=share)
        save_atomic({"model": models["E1"].state_dict(), "tokens": total_train_tokens}, ckpt_root/"E1_all.pt")
        stats["E1"] = {"adapters": adapters, "tokens": sum(item["tokens"] for item in adapters.values()),
                       "shared_base_checkpoint": str(ckpt_root/"D1.pt")}
        for name in ("D2", "M1"):
            stats[name] = train_steps(models[name], train_seqs, 0, budget, name, ckpt_root/f"{name}.pt", lr_total_tokens=budget)
        (OUT/"training_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    elif args.mode == "eval":
        pass
    # Evaluate checkpoints if present; no test-time finetuning.
    evals = {}
    for name in ("D1", "D1+", "D2", "E1", "M1"):
        ckpt_name = {"D1+":"D1plus", "E1":"E1_all"}.get(name, name)
        path = OUT/"checkpoints"/"round5c4"/f"{ckpt_name}.pt"
        if not path.exists():
            continue
        state = torch.load(path, map_location=DEVICE, weights_only=False)
        model = models[name].to(DEVICE); model.load_state_dict(state["model"])
        if name == "E1":
            rng = random.Random(2025)
            wrong_domains = {}
            for rec in test_meta:
                wrong_domains[str(rec["index"])] = rng.choice([d for d in range(N_EXPERTS) if d != rec["domain"]])
            evals[name] = {"correct_adapter": evaluate(model, test_meta, "E1 correct", "adapted"),
                           "base_only": evaluate(model, test_meta, "E1 base", "base"),
                           "wrong_adapter": evaluate(model, test_meta, "E1 wrong", "adapted", wrong_domains=wrong_domains)}
        else:
            evals[name] = evaluate(model, test_meta, f"{name} eval", "adapted")
    training_stats_path = OUT/"training_stats.json"
    training_stats = json.loads(training_stats_path.read_text(encoding="utf-8")) if training_stats_path.exists() else {}
    def aggregate_training(name):
        record = training_stats.get(name)
        if not record:
            return {}
        parts = list(record["adapters"].values()) if name == "E1" else [record]
        if name == "E1":
            return {"training_tokens": sum(p.get("tokens",0) for p in parts),
                    "training_seconds": sum(p.get("seconds",0) for p in parts),
                    "training_tokens_per_second": sum(p.get("tokens",0) for p in parts)/max(sum(p.get("seconds",0) for p in parts),1e-9),
                    "gpu_hours": sum(p.get("gpu_hours",0) for p in parts),
                    "peak_gpu_memory_bytes": max((p.get("peak_gpu_memory_bytes",0) for p in parts),default=0),
                    "shared_base_from": record.get("shared_base_checkpoint")}
        tokens = sum(p.get("tokens", 0) for p in parts)
        seconds = sum(p.get("seconds", 0) for p in parts)
        return {"training_tokens": tokens, "training_seconds": seconds,
                "training_tokens_per_second": tokens/max(seconds, 1e-9),
                "gpu_hours": sum(p.get("gpu_hours", 0) for p in parts),
                "peak_gpu_memory_bytes": max((p.get("peak_gpu_memory_bytes", 0) for p in parts), default=0)}
    model_outputs = {}
    for name, val in evals.items():
        active = counts[name] if name in ("D1", "D1+", "D2") else counts["D1"] + (counts["E1"]-counts["D1"])//N_EXPERTS
        if name == "M1":
            active += router_params
        model_outputs[name] = {"model": name, "parameters_total": counts[name], "parameters_active_per_token": active,
                               **aggregate_training(name), "evaluation": val}
        (OUT/f"{name}.json").write_text(json.dumps(model_outputs[name], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps({"parameters_total": counts,
        "parameters_active_per_token": {k:v["parameters_active_per_token"] for k,v in model_outputs.items()},
        "models": model_outputs, "evaluation": evals,
        "reference_ppc": {"BPE_monolithic": 3.279, "Sector_LM_A": 3.603, "tiny_starcoder_py": 1.296}}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
