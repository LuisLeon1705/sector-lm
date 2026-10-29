from __future__ import annotations

import hashlib
import json
import os
import time
import unicodedata
import argparse
from pathlib import Path

DATA = Path("data")
ROUND = DATA / "round5"
TARGET_TOTAL_BYTES = int(1.5 * 1024**3)


def normalize_content(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def normalized_hash(text: str) -> str:
    return hashlib.sha256(normalize_content(text).encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true", help="prepare only from data/train.jsonl")
    parser.add_argument("--target-total-gib", type=float, default=1.5)
    args = parser.parse_args()
    target_total_bytes = int(args.target_total_gib * 1024**3)
    ROUND.mkdir(parents=True, exist_ok=True)
    valid_hashes = {normalized_hash(row["content"]) for row in read_jsonl(DATA / "valid.jsonl")}
    test_hashes = {normalized_hash(row["content"]) for row in read_jsonl(DATA / "test.jsonl")}
    protected = valid_hashes | test_hashes
    train_path = DATA / "train.jsonl"
    source_bytes = train_path.stat().st_size + (DATA / "valid.jsonl").stat().st_size + (DATA / "test.jsonl").stat().st_size
    train_hashes: set[str] = set()
    rows: list[str] = []
    bytes_written = 0
    duplicates = 0
    for row in read_jsonl(train_path):
        h = normalized_hash(row["content"])
        if h in protected or h in train_hashes:
            duplicates += 1
            continue
        train_hashes.add(h)
        encoded = json.dumps({"content": row["content"], "sha256": h}, ensure_ascii=False) + "\n"
        rows.append(encoded)
        bytes_written += len(encoded.encode("utf-8"))

    started = time.monotonic()
    last_report = started
    extra_rows = 0
    extra_bytes = 0
    with (ROUND / "train.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for encoded in rows:
            out.write(encoded)
        if not args.no_download and source_bytes + bytes_written < target_total_bytes:
            from datasets import load_dataset
            dataset = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
            for item in dataset:
                if source_bytes + bytes_written >= target_total_bytes:
                    break
                content = item.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    continue
                h = normalized_hash(content)
                if h in protected or h in train_hashes:
                    duplicates += 1
                    continue
                train_hashes.add(h)
                encoded = json.dumps({"content": content, "sha256": h}, ensure_ascii=False) + "\n"
                out.write(encoded)
                line_bytes = len(encoded.encode("utf-8"))
                bytes_written += line_bytes
                extra_bytes += line_bytes
                extra_rows += 1
                now = time.monotonic()
                if now - last_report >= 10:
                    elapsed = max(now - started, 0.001)
                    rate = extra_bytes / elapsed
                    remaining = max(0, target_total_bytes - source_bytes - bytes_written)
                    eta = remaining / max(rate, 1)
                    print(f"streaming CodeParrot: {extra_rows} files, {extra_bytes/1024**2:.1f} MiB, {extra_rows/elapsed:.1f} files/s, {rate/1024**2:.2f} MiB/s, ETA {eta/60:.1f} min", flush=True)
                    last_report = now

    test_rows = sum(1 for _ in read_jsonl(DATA / "test.jsonl"))
    valid_rows = sum(1 for _ in read_jsonl(DATA / "valid.jsonl"))
    report_path = DATA / "leakage_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report["round5"] = {
        "normalization": "Unicode NFC and CRLF/CR to LF before SHA-256",
        "train_files": len(train_hashes),
        "extra_streamed_files": extra_rows,
        "extra_streamed_bytes": extra_bytes,
        "download_skipped": args.no_download,
        "duplicates_removed": duplicates,
        "valid_files_checked": valid_rows,
        "test_files_checked": test_rows,
        "normalized_train_valid_test_overlap": 0,
        "total_dataset_bytes": source_bytes + bytes_written,
        "train_file": str(ROUND / "train.jsonl"),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["round5"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
