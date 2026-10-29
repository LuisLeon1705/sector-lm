import argparse
import hashlib
import json
import os
import sys
from datasets import load_dataset
import tree_sitter_python
from tree_sitter import Language, Parser
from tqdm import tqdm

def main():
    parser_arg = argparse.ArgumentParser(description="Prepare CodeParrot dataset splits")
    parser_arg.add_argument("--train-limit", type=int, default=20000, help="Number of clean train files")
    parser_arg.add_argument("--valid-limit", type=int, default=1000, help="Number of clean validation files")
    parser_arg.add_argument("--test-limit", type=int, default=1000, help="Number of clean test files")
    parser_arg.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser_arg.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    py_lang = Language(tree_sitter_python.language())
    ts_parser = Parser(py_lang)

    def is_valid_python(content: str) -> bool:
        if not content or len(content.strip()) < 10:
            return False
        try:
            tree = ts_parser.parse(content.encode("utf-8"))
            return not tree.root_node.has_error
        except Exception:
            return False

    print("=== Step 1: Streaming codeparrot-clean-valid for Validation and Test splits ===")
    ds_valid_stream = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)

    valid_items = []
    test_items = []
    test_hashes = set()
    total_valid_scanned = 0
    discarded_valid = 0

    pbar = tqdm(total=args.valid_limit + args.test_limit, desc="Valid + Test files")
    for item in ds_valid_stream:
        total_valid_scanned += 1
        content = item.get("content", "")
        if not is_valid_python(content):
            discarded_valid += 1
            continue

        h = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if len(valid_items) < args.valid_limit:
            valid_items.append({"content": content, "sha256": h})
            pbar.update(1)
        elif len(test_items) < args.test_limit:
            test_items.append({"content": content, "sha256": h})
            test_hashes.add(h)
            pbar.update(1)
        else:
            break

    pbar.close()
    print(f"Collected {len(valid_items)} valid files, {len(test_items)} test files (discarded {discarded_valid} invalid syntax)")

    print(f"\n=== Step 2: Streaming codeparrot-clean for Training (limit {args.train_limit}) ===")
    ds_train_stream = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

    train_items = []
    train_hashes = set()
    leaked_found = 0
    discarded_train = 0
    total_train_scanned = 0

    pbar_train = tqdm(total=args.train_limit, desc="Train files")
    for item in ds_train_stream:
        total_train_scanned += 1
        content = item.get("content", "")
        if not is_valid_python(content):
            discarded_train += 1
            continue

        h = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Strict check: Test file must NEVER appear in training
        if h in test_hashes:
            leaked_found += 1
            print(f"LEAKAGE PREVENTED: Training file with hash {h} matched test set and was strictly discarded!")
            continue

        if h in train_hashes:
            # Skip duplicates within train
            continue

        train_hashes.add(h)
        train_items.append({"content": content, "sha256": h})
        pbar_train.update(1)

        if len(train_items) >= args.train_limit:
            break

    pbar_train.close()
    print(f"Collected {len(train_items)} train files (discarded {discarded_train} syntax errors, prevented {leaked_found} test leaks)")

    # Save to disk
    train_file = os.path.join(args.output_dir, "train.jsonl")
    valid_file = os.path.join(args.output_dir, "valid.jsonl")
    test_file = os.path.join(args.output_dir, "test.jsonl")
    report_file = os.path.join(args.output_dir, "leakage_report.json")

    print("\nSaving splits to JSONL...")
    with open(train_file, "w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it) + "\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for it in valid_items:
            f.write(json.dumps(it) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for it in test_items:
            f.write(json.dumps(it) + "\n")

    report = {
        "train_count": len(train_items),
        "valid_count": len(valid_items),
        "test_count": len(test_items),
        "test_hashes_count": len(test_hashes),
        "leaks_prevented": leaked_found,
        "discarded_syntax_train": discarded_train,
        "discarded_syntax_valid": discarded_valid,
        "train_scanned": total_train_scanned,
        "valid_scanned": total_valid_scanned,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Done! Leakage report saved to {report_file}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
