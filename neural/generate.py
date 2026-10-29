"""Generate Python code with a trained round-5 model.

    python -m neural.generate --prompt "def fibonacci(n):"
    python -m neural.generate                      # interactive: type code, end with an empty line
    python -m neural.generate --model Parallax-E --domain WebBackend --prompt "from flask import Flask"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from neural.model import SwitchGPT, build_models
from neural.pipeline import DOMAIN_NAMES as DOMAINS

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "results/round5/checkpoints/round5c4"
TOKENIZER = ROOT / "results/round5/tokenizer.json"
CONTEXT = 512
BOS_ID = 1
CHECKPOINTS = {"D1": "D1", "D1+": "D1plus", "D2": "D2", "E1": "E1_all", "M1": "M1"}
ARCH = {"D1": "D1", "D1+": "D1", "D2": "D2", "E1": "E1", "M1": "M1"}
# Family names used in the docs; each maps to a code above.
NAMES = {"Parallax-15M": "D1+", "Parallax-D15M": "D1", "Parallax-D27M": "D2", "Parallax-E": "E1", "Parallax-M": "M1"}


def load(name: str, device: str):
    tok = Tokenizer.from_file(str(TOKENIZER))
    model = build_models(vocab_size=tok.get_vocab_size(), experts=len(DOMAINS))[ARCH[name]]
    state = torch.load(CKPT_DIR / f"{CHECKPOINTS[name]}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return tok, model.to(device).eval()


@torch.inference_mode()
def generate(model, tok, prompt: str, max_new: int, temperature: float, top_k: int, expert_id, device: str):
    ids = [BOS_ID] + tok.encode(prompt, add_special_tokens=False).ids
    shown = tok.decode(ids[1:])
    for _ in range(max_new):
        x = torch.tensor(ids[-CONTEXT:], device=device)[None]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            if isinstance(model, SwitchGPT):
                logits = model(x)
            else:
                logits = model(x, expert_id=expert_id, use_expert=expert_id is not None)
        logits = logits[0, -1].float()
        logits[:2] = float("-inf")  # never emit <unk> or <bos>
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                cutoff = torch.topk(logits, top_k).values[-1]
                logits[logits < cutoff] = float("-inf")
            nxt = int(torch.multinomial(F.softmax(logits, -1), 1))
        ids.append(nxt)
        text = tok.decode(ids[1:])
        if not text.endswith("�"):  # wait until a character split across byte tokens is complete
            sys.stdout.write(text[len(shown):])
            sys.stdout.flush()
            shown = text
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(NAMES) + list(CHECKPOINTS), default="Parallax-15M")
    ap.add_argument("--domain", choices=DOMAINS, default="GeneralLocal", help="adapter used by E1")
    ap.add_argument("--prompt", help="code to continue; without it, an interactive session starts")
    ap.add_argument("--max-new", type=int, default=200, help="tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = always the most likely token")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    label = args.model
    args.model = NAMES.get(args.model, args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load(args.model, device)
    expert_id = DOMAINS.index(args.domain) if args.model == "E1" else None
    gen = lambda p: generate(model, tok, p, args.max_new, args.temperature, args.top_k, expert_id, device)

    if args.prompt is not None:
        gen(args.prompt.replace("\\n", "\n"))
        return
    print(f"{label} listo en {device}. Escribe código y termina con una línea vacía (Ctrl+C para salir).")
    while True:
        try:
            lines = []
            while True:
                line = input("... " if lines else ">>> ")
                if not line and lines:
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print()
            return
        print("-" * 60)
        gen("\n".join(lines) + "\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
