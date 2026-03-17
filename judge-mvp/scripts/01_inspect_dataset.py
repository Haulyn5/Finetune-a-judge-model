#!/usr/bin/env python3
"""Inspect the raw WildGuardMix dataset before any normalization.

Pipeline step:
    01 / 08

Goal:
    Help a learner confirm the real dataset schema before writing any
    preprocessing logic. This script is intentionally read-only.

Inputs:
    - Hugging Face dataset configs such as ``wildguardtrain`` and ``wildguardtest``.

Outputs:
    - Prints split names, feature names, and a few sample rows to stdout.
    - Prints a JSON summary at the end for easier inspection or logging.

Key assumptions:
    - The dataset can be loaded from Hugging Face with the provided config names.
    - We inspect the source dataset first so downstream scripts do not rely on
      guessed field names.
"""

import argparse
import json

from datasets import load_dataset


def inspect_config(dataset_name: str, config_name: str, preview_rows: int) -> dict:
    """Load one dataset config and return a compact inspection summary.

    Args:
        dataset_name: Hugging Face dataset identifier.
        config_name: Dataset config to inspect.
        preview_rows: Maximum number of rows to print and keep in the summary.

    Returns:
        A JSON-serializable dictionary containing split name, row count,
        feature names, and preview rows.
    """
    ds = load_dataset(dataset_name, config_name)
    split_name = next(iter(ds.keys()))
    split_ds = ds[split_name]

    print(f"\n=== config={config_name} split={split_name} ===")
    print(f"rows={len(split_ds)}")
    print("features=")
    print(split_ds.features)

    preview = []
    for idx in range(min(preview_rows, len(split_ds))):
        row = split_ds[idx]
        preview.append(row)
        print(f"sample[{idx}]=")
        print(json.dumps(row, ensure_ascii=False, indent=2, default=str))

    return {
        "split": split_name,
        "rows": len(split_ds),
        "features": list(split_ds.features.keys()),
        "preview": preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the raw allenai/wildguardmix dataset.")
    parser.add_argument("--dataset_name", default="allenai/wildguardmix")
    parser.add_argument("--configs", nargs="*", default=["wildguardtrain", "wildguardtest"])
    parser.add_argument("--preview_rows", type=int, default=3)
    args = parser.parse_args()

    summary = {}
    for config_name in args.configs:
        summary[config_name] = inspect_config(args.dataset_name, config_name, args.preview_rows)

    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
