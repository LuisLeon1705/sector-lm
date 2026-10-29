import math
import json
import subprocess
from pathlib import Path

import torch

from neural.model import Config, GPT, SwitchGPT, build_models, count_parameters, copy_shared_base
from neural.pipeline import BOS_ID, get_batch, limit_valid_targets
from neural.router import route_from_import_tokens


def ppc_from_nats(total_nats: float, characters: int) -> float:
    if characters != 795700:
        raise ValueError(f"expected the fixed 795700-character benchmark, got {characters}")
    return math.exp(total_nats / characters)


def test_ppc_uses_fixed_test_character_count():
    assert ppc_from_nats(795700 * math.log(3.279), 795700) == pytest.approx(3.279)
    try:
        ppc_from_nats(10, 100)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong benchmark denominator was accepted")


def test_router_uses_import_tokens_only():
    rules = {"torch": (1, 3), "sklearn": (1, 3), "flask": (2, 3), "os": (6, 1)}
    # This interface has no source/body argument; a body-only mention cannot affect routing.
    assert route_from_import_tokens(["os", "flask"], rules) == 2
    assert route_from_import_tokens(["os"], rules) == 6
    assert route_from_import_tokens([], rules) == 12


def test_rust_domain_router_ignores_non_import_body(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "two_files.jsonl"
    contents = [
        "import torch\n\ndef f():\n    return 1\n",
        "import torch\n\ndef f():\n    flask()\n    flask()\n    flask()\n    flask()\n",
    ]
    source.write_text("".join(json.dumps({"content": s}) + "\n" for s in contents), encoding="utf-8")
    output = tmp_path / "domains"
    subprocess.run([
        "cargo", "run", "--quiet", "--release", "--bin", "dump_domains", "--",
        "--train-file", str(source), "--valid-file", str(source), "--test-file", str(source),
        "--output-dir", str(output),
    ], cwd=root, check=True, capture_output=True, text=True)
    labels = json.loads((output / "train_domains.json").read_text(encoding="utf-8"))
    assert labels["0"] == labels["1"] == 1


def test_adapter_training_keeps_base_weights_frozen():
    torch.manual_seed(7)
    model = GPT(Config(vocab_size=64, width=24, layers=1, heads=4, context=8, rank=4, experts=1), with_experts=True)
    model.adapter_trainable()
    base_before = {name: p.detach().clone() for name, p in model.named_parameters() if not name.startswith("experts.")}
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    x = torch.randint(0, 64, (1, 8))
    loss = model(x, expert_id=0).float().mean()
    loss.backward()
    opt.step()
    for name, p in model.named_parameters():
        if name in base_before:
            assert torch.equal(p, base_before[name]), name


def test_model_parameter_budgets():
    models = build_models(vocab_size=16384, experts=12)
    counts = {name: count_parameters(model) for name, model in models.items()}
    assert 14_000_000 <= counts["D1"] <= 16_000_000
    assert abs(counts["D2"] - counts["E1"]) / counts["E1"] < 0.03
    assert models["E1"].cfg.experts == models["M1"].cfg.experts == 12
    router_parameters = sum(p.numel() for block in models["M1"].blocks for p in block.router.parameters())
    assert counts["M1"] == counts["E1"] + router_parameters
    assert models["D2"].cfg.width > models["D1"].cfg.width


def test_e1_dense_base_copies_d1_checkpoint_exactly():
    models = build_models(vocab_size=64, experts=12)
    with torch.no_grad():
        models["D1"].token_embedding.weight.fill_(0.125)
    copied = copy_shared_base(models["D1"], models["E1"])
    assert copied > 0
    assert torch.equal(models["D1"].token_embedding.weight, models["E1"].token_embedding.weight)
    assert torch.equal(models["D1"].blocks[0].qkv.weight, models["E1"].blocks[0].qkv.weight)


def test_switch_router_learns_and_routes_same_token_by_context():
    torch.manual_seed(114)
    model = SwitchGPT(Config(vocab_size=32, width=24, layers=2, heads=4, context=8, rank=4, experts=3))
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = [block.router.weight.detach().clone() for block in model.blocks]
    for step in range(8):
        x = torch.tensor([[1 + step, 2, 5, 7], [3, 4 + step, 5, 7]])
        y = torch.tensor([[2, 5, 7, 8], [4 + step, 5, 7, 9]])
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), y.reshape(-1)) + 0.1 * model.last_router_aux
        loss.backward()
        optim.step()
    assert any(not torch.equal(old, block.router.weight) for old, block in zip(before, model.blocks))
    contexts = torch.tensor([[i, (i+3) % 32, 5] for i in range(32)])
    model(contexts)
    # Block 2 receives context-enriched hidden states from block 1.
    last_position_routes = model.last_routes[-1][:, -1]
    assert torch.unique(last_position_routes).numel() > 1
    assert len(model.last_routes) == len(model.blocks)
    model.zero_grad(set_to_none=True)
    lm_logits = model(contexts)
    torch.nn.functional.cross_entropy(lm_logits.reshape(-1, 32), torch.full((contexts.numel(),), 6)).backward()
    assert model.blocks[-1].router.weight.grad.abs().sum().item() > 0


def test_padding_labels_are_ignored_and_unbudgeted():
    seqs = [(torch.tensor([1, 2, 3], dtype=torch.int32), 0)]
    _, labels, _, valid = get_batch(seqs, batch_size=1, context=8, device="cpu")
    # The file is read as <bos> 1 2 3, so all three tokens are targets.
    assert labels.tolist()[0][:3] == [1, 2, 3]
    assert labels.tolist()[0][3:] == [-100] * 5
    assert valid == 3
    assert limit_valid_targets(labels, 1) == 1
    assert int((labels != -100).sum()) == 1


import pytest


def test_initial_logits_are_small_with_tied_embeddings():
    torch.manual_seed(0)
    model = build_models(vocab_size=16384, experts=12)["D1"]
    logits = model(torch.randint(0, 16384, (2, 64)))
    assert model.token_embedding.weight.std().item() < 0.05
    assert logits.std().item() < 1.0


def test_a_short_file_is_trained_after_one_bos():
    seq = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    x, y, _, valid = get_batch([(seq, 0)], 1, 512, "cpu")
    assert x[0, :4].tolist() == [BOS_ID, 10, 11, 12]
    assert y[0, :4].tolist() == [10, 11, 12, 13]
    assert valid == 4


def test_evaluation_counts_each_character_once_when_bpe_splits_it():
    from tokenizers import Tokenizer
    from neural import pipeline
    tok = Tokenizer.from_file(str(Path(__file__).resolve().parents[2] / "results/round5/tokenizer.json"))
    text = "s = '😀ñ€'  # día 😀\n" * 60
    enc = tok.encode(text, add_special_tokens=False)
    assert len(enc.ids) > 300  # covers the first window (<bos> only) and a later one
    model = GPT(Config(vocab_size=tok.get_vocab_size(), width=32, heads=2, layers=1)).to(pipeline.DEVICE)
    result = pipeline.eval_file(model, enc.ids, enc.offsets, text)
    assert sum(result["by_type_chars"].values()) == len(text)
