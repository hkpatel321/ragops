"""
Inspect random samples from the golden evaluation dataset for manual review.

Usage:
    python scripts/inspect_golden_set.py [--file eval/golden_v1.jsonl] [--samples 10]
"""

import argparse
import json
import random
import sys
from pathlib import Path


def inspect_golden_set(file_path: str = "eval/golden_v1.jsonl", sample_count: int = 10) -> None:
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: Golden set file '{file_path}' does not exist.")
        sys.exit(1)

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    if total == 0:
        print("Error: Golden set file is empty.")
        sys.exit(1)

    sample_k = min(sample_count, total)
    sampled = random.sample(records, sample_k)

    print("=" * 80)
    print(f"MANUAL REVIEW: {sample_k} RANDOM SAMPLES FROM '{file_path}' (Total: {total})")
    print("=" * 80)

    for i, item in enumerate(sampled, start=1):
        q_id = item.get("id", f"sample_{i}")
        q_type = item.get("question_type", "unknown").upper()
        source = item.get("source", "unknown").upper()
        section = item.get("section", "General")
        chunk_id = item.get("ground_truth_chunk_id", "N/A")
        question = item.get("question", "")
        answer = item.get("ground_truth_answer", "")
        context = item.get("ground_truth_context", "")
        context_preview = context.replace("\n", " ")[:200] + "..." if len(context) > 200 else context

        print(f"\n[Sample {i}/{sample_k}] ID: {q_id} | Type: [{q_type}] | Source: [{source}]")
        print(f"Section : {section}")
        print(f"Chunk ID: {chunk_id}")
        print("-" * 80)
        print(f"Question:\n  {question}\n")
        print(f"Ground Truth Answer:\n  {answer}\n")
        print(f"Context Snippet:\n  \"{context_preview}\"")
        print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect random samples from the golden set for review")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="eval/golden_v1.jsonl",
        help="Path to golden set JSONL file",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=10,
        help="Number of random samples to display (default: 10)",
    )
    args = parser.parse_args()

    inspect_golden_set(file_path=args.file, sample_count=args.samples)


if __name__ == "__main__":
    main()
