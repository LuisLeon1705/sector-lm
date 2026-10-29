import argparse
import datetime
import json
import math
import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

def main():
    parser = argparse.ArgumentParser(description="Evaluate tiny_starcoder_py with sliding window on the exact same test files")
    parser.add_argument("--test-file", type=str, default="data/test.jsonl")
    parser.add_argument("--model-id", type=str, default="bigcode/tiny_starcoder_py")
    parser.add_argument("--max-files", type=int, default=100, help="Number of test files")
    parser.add_argument("--window-size", type=int, default=2048, help="Context window size")
    parser.add_argument("--stride", type=int, default=1024, help="Sliding window stride")
    parser.add_argument("--output", type=str, default="results/starcoder_metrics.json")
    args = parser.parse_args()

    command_executed = " ".join(sys.argv)
    eval_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print(f"Loading {args.model_id} on GPU/CUDA...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    test_docs = []
    total_chars_exact = 0
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                content = json.loads(line)["content"]
                test_docs.append(content)
                total_chars_exact += len(content)
            if len(test_docs) >= args.max_files:
                break

    print(f"Loaded {len(test_docs)} test files for evaluation (Exact Total Characters: {total_chars_exact}).")

    total_tokens_evaluated = 0
    total_log_prob = 0.0
    top1_correct = 0
    top5_correct = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for doc_idx, content in enumerate(test_docs):
            enc = tokenizer(content, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            seq_len = input_ids.shape[1]

            if seq_len < 2:
                continue

            # Sliding window over the full file
            # When seq_len <= window_size, a single forward pass evaluates the entire file
            # When seq_len > window_size, windows of size `window_size` slide with `stride`
            prev_end_loc = 0
            for begin_loc in range(0, seq_len - 1, args.stride):
                end_loc = min(begin_loc + args.window_size, seq_len)
                window_ids = input_ids[:, begin_loc:end_loc]

                outputs = model(window_ids)
                logits = outputs.logits  # [1, window_len, vocab_size]

                # Position within window to start predicting
                # For first window (begin_loc == 0), start predicting from token 0 (predicts token 1)
                # For subsequent windows, only predict tokens not yet scored in previous windows
                trg_start = 0 if begin_loc == 0 else (prev_end_loc - begin_loc - 1)
                trg_end = window_ids.shape[1] - 1

                if trg_start >= trg_end:
                    continue

                shift_logits = logits[0, trg_start:trg_end, :]  # [num_eval, vocab]
                shift_labels = window_ids[0, trg_start + 1:trg_end + 1]  # [num_eval]

                log_probs = torch.log_softmax(shift_logits, dim=-1)
                target_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

                _, top5_indices = torch.topk(shift_logits, k=5, dim=-1)
                top1_preds = top5_indices[:, 0]

                is_top1 = (top1_preds == shift_labels).sum().item()
                is_top5 = (top5_indices == shift_labels.unsqueeze(1)).any(dim=-1).sum().item()

                num_eval = shift_labels.shape[0]
                top1_correct += is_top1
                top5_correct += is_top5
                total_tokens_evaluated += num_eval
                total_log_prob += target_log_probs.sum().item()

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            if (doc_idx + 1) % 20 == 0 or (doc_idx + 1) == len(test_docs):
                print(f"Processed {doc_idx + 1}/{len(test_docs)} files ({total_tokens_evaluated} tokens evaluated)...")

    elapsed = time.perf_counter() - start_time

    top1_acc = top1_correct / total_tokens_evaluated if total_tokens_evaluated > 0 else 0.0
    top5_acc = top5_correct / total_tokens_evaluated if total_tokens_evaluated > 0 else 0.0
    # Perplexity per character: exp(- total_log_prob / total_chars_exact)
    ppc = math.exp(- total_log_prob / total_chars_exact) if total_chars_exact > 0 else 1.0
    tok_per_sec = total_tokens_evaluated / elapsed if elapsed > 0 else 0.0

    if device == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    metrics = {
        "model_name": "bigcode/tiny_starcoder_py (164M)",
        "evaluation_timestamp": eval_timestamp,
        "command_executed": command_executed,
        "evaluation_strategy": "Sliding window (window=2048, stride=1024) over 100 complete test files",
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "perplexity_per_char": ppc,
        "memory_mb": mem_mb,
        "tokens_per_sec": tok_per_sec,
        "total_tokens_evaluated": total_tokens_evaluated,
        "total_chars_evaluated": total_chars_exact,
        "elapsed_seconds": elapsed,
    }

    print("\n=== tiny_starcoder_py Final Measured Metrics ===")
    print(json.dumps(metrics, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.output}")

if __name__ == "__main__":
    main()
