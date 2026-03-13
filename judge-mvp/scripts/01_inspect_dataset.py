#!/usr/bin/env python3
import argparse
import json
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the allenai/wildguardmix dataset.")
    parser.add_argument("--dataset_name", default="allenai/wildguardmix")
    parser.add_argument("--configs", nargs="*", default=["wildguardtrain", "wildguardtest"])
    parser.add_argument("--preview_rows", type=int, default=3)
    args = parser.parse_args()

    summary = {}
    for config_name in args.configs:
        ds = load_dataset(args.dataset_name, config_name)
        split_name = next(iter(ds.keys()))
        split_ds = ds[split_name]
        print(f"\n=== config={config_name} split={split_name} ===")
        print(f"rows={len(split_ds)}")
        print("features=")
        print(split_ds.features)
        preview = []
        for idx in range(min(args.preview_rows, len(split_ds))):
            row = split_ds[idx]
            preview.append(row)
            print(f"sample[{idx}]=")
            print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
        summary[config_name] = {
            "split": split_name,
            "rows": len(split_ds),
            "features": list(split_ds.features.keys()),
            "preview": preview,
        }

    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
