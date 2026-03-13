#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split


INSTRUCTION = "You are a content safety judge. Given a user question and assistant response, output a JSON object with keys label, reason, evidence, confidence. Use label in {safe, unsafe}. Evidence must quote exact spans from the input."


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_example(row: dict) -> dict:
    output_obj = {
        "label": row["label"],
        "reason": row["reason"],
        "evidence": row["evidence"],
        "confidence": row.get("confidence", 0.5),
    }
    model_input = f"Question:\n{row['question']}\n\nResponse:\n{row['response']}"
    return {
        "id": row["id"],
        "instruction": INSTRUCTION,
        "input": model_input,
        "output": json.dumps(output_obj, ensure_ascii=False),
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": model_input},
            {"role": "assistant", "content": json.dumps(output_obj, ensure_ascii=False)},
        ],
        "gold_label": row["label"],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT JSONL data from filtered pseudo labels.")
    parser.add_argument("--input_path", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "pseudo_filtered.jsonl")
    parser.add_argument("--train_output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "sft_train.jsonl")
    parser.add_argument("--dev_output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "sft_dev.jsonl")
    parser.add_argument("--dev_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [build_example(row) for row in read_jsonl(args.input_path)]
    if not rows:
        raise ValueError("No filtered pseudo-labeled rows found. Run script 05 first.")

    random.seed(args.seed)
    train_rows, dev_rows = train_test_split(rows, test_size=args.dev_size, random_state=args.seed)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.dev_output, dev_rows)

    print(json.dumps({"train": len(train_rows), "dev": len(dev_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
