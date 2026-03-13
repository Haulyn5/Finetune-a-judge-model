#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


LABEL_MAP = {
    "harmful": "unsafe",
    "unharmful": "safe",
    "safe": "safe",
    "unsafe": "unsafe",
}


def normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    return LABEL_MAP.get(str(value).strip().lower())


def convert_records(dataset, source_split: str, label_field: str) -> list[dict]:
    rows = []
    for idx, row in enumerate(dataset):
        question = (row.get("prompt") or row.get("question") or "").strip()
        response = (row.get("response") or "").strip()
        label = normalize_label(row.get(label_field))
        if not question or not response or label is None:
            continue
        sample_id = row.get("id") or f"{source_split}-{idx}"
        rows.append(
            {
                "id": str(sample_id),
                "question": question,
                "response": response,
                "label": label,
                "split": source_split,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized binary safety classification JSONL files.")
    parser.add_argument("--dataset_name", default="allenai/wildguardmix")
    parser.add_argument("--train_config", default="wildguardtrain")
    parser.add_argument("--test_config", default="wildguardtest")
    parser.add_argument("--label_field", default="response_harm_label")
    parser.add_argument("--dev_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=Path(__file__).resolve().parents[1] / "data" / "processed", type=Path)
    args = parser.parse_args()

    train_ds = load_dataset(args.dataset_name, args.train_config)
    test_ds = load_dataset(args.dataset_name, args.test_config)

    train_split = train_ds[next(iter(train_ds.keys()))]
    test_split = test_ds[next(iter(test_ds.keys()))]

    train_rows = convert_records(train_split, "train", args.label_field)
    test_rows = convert_records(test_split, "test", args.label_field)

    if not train_rows:
        raise ValueError("No usable training rows were produced. Check dataset fields and label mapping.")
    if not test_rows:
        raise ValueError("No usable test rows were produced. Check dataset fields and label mapping.")

    train_df = pd.DataFrame(train_rows)
    train_part, dev_part = train_test_split(
        train_df,
        test_size=args.dev_size,
        random_state=args.seed,
        stratify=train_df["label"],
    )

    train_out = train_part.to_dict("records")
    dev_out = dev_part.to_dict("records")
    test_out = pd.DataFrame(test_rows).to_dict("records")

    for row in dev_out:
        row["split"] = "dev"

    write_jsonl(args.output_dir / "train.jsonl", train_out)
    write_jsonl(args.output_dir / "dev.jsonl", dev_out)
    write_jsonl(args.output_dir / "test.jsonl", test_out)

    summary = {
        "train": len(train_out),
        "dev": len(dev_out),
        "test": len(test_out),
        "labels": sorted({row["label"] for row in train_out + dev_out + test_out}),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
