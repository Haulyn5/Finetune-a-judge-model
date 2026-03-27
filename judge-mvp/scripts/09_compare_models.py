#!/usr/bin/env python3
"""Compare baseline, base-model, and LoRA-model evaluation results.

Pipeline step:
    09 / 09

Goal:
    Build the only multi-model analysis layer on top of step-08-compatible
    prediction files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _eval_common import FLAT_METRIC_ORDER, VALID_LABELS, evaluate_prediction_file, read_jsonl, validate_reference_rows, write_json

PAIRWISE_ORDER = [
    ("qwen_base", "baseline", "qwen_base_minus_baseline"),
    ("qwen_lora", "baseline", "qwen_lora_minus_baseline"),
    ("qwen_lora", "qwen_base", "qwen_lora_minus_qwen_base"),
]
MODEL_ORDER = ("baseline", "qwen_base", "qwen_lora")


def compute_pairwise_delta(results: dict) -> dict:
    deltas = {}
    for left_name, right_name, delta_name in PAIRWISE_ORDER:
        if left_name not in results or right_name not in results:
            continue
        deltas[delta_name] = {
            metric: results[left_name]["flat_metrics"][metric] - results[right_name]["flat_metrics"][metric]
            for metric in FLAT_METRIC_ORDER
        }
    return deltas


def rank_models(results: dict) -> dict:
    metric_preferences = {
        "accuracy": True,
        "macro_f1": True,
        "unsafe_precision": True,
        "unsafe_recall": True,
        "unsafe_f1": True,
        "overblock_rate": False,
        "raw_json_parse_success_rate": True,
        "raw_json_parse_failure_rate": False,
        "fallback_usable_rate": True,
        "json_valid_rate": True,
        "evidence_hit_rate": True,
        "reason_label_consistency": True,
    }
    ranking = {}
    for metric, larger_is_better in metric_preferences.items():
        ordered = sorted(
            ((name, result["flat_metrics"][metric]) for name, result in results.items()),
            key=lambda item: item[1],
            reverse=larger_is_better,
        )
        ranking[metric] = ordered
    return ranking


def build_takeaways(pairwise_delta: dict, ranking: dict) -> list[str]:
    takeaways = []
    macro_f1_winner, macro_f1_value = ranking["macro_f1"][0]
    unsafe_f1_winner, unsafe_f1_value = ranking["unsafe_f1"][0]
    parse_winner, parse_value = ranking["raw_json_parse_success_rate"][0]
    takeaways.append(f"Best overall macro_f1 run is {macro_f1_winner} ({macro_f1_value:.4f}).")
    takeaways.append(f"Best unsafe_f1 run is {unsafe_f1_winner} ({unsafe_f1_value:.4f}).")
    takeaways.append(f"Best strict structured-output reliability is {parse_winner} ({parse_value:.4f} raw_json_parse_success_rate).")
    for delta_name in ("qwen_base_minus_baseline", "qwen_lora_minus_baseline", "qwen_lora_minus_qwen_base"):
        delta = pairwise_delta.get(delta_name)
        if not delta:
            continue
        takeaways.append(
            f"{delta_name}: macro_f1 {delta['macro_f1']:+.4f}, unsafe_f1 {delta['unsafe_f1']:+.4f}, overblock_rate {delta['overblock_rate']:+.4f}."
        )
    return takeaways


def build_markdown(reference_file: Path, input_files: dict, results: dict, pairwise_delta: dict, ranking: dict, takeaways: list[str]) -> str:
    lines = ["# Step 09 Comparison Analysis", "", "## Inputs", ""]
    lines.append(f"- Reference file: `{reference_file}`")
    for model_name in MODEL_ORDER:
        lines.append(f"- {model_name}: `{input_files[model_name]}`")

    lines.extend(["", "## Metrics", ""])
    lines.append("| Model | " + " | ".join(FLAT_METRIC_ORDER) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(FLAT_METRIC_ORDER)) + " |")
    for model_name in MODEL_ORDER:
        metrics = results[model_name]["flat_metrics"]
        values = [f"{metrics[key]:.4f}" for key in FLAT_METRIC_ORDER]
        lines.append("| " + model_name + " | " + " | ".join(values) + " |")

    lines.extend(["", "## Confusion Matrices", ""])
    for model_name in MODEL_ORDER:
        confusion = results[model_name]["summary"]["confusion"]
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| gold \\ pred | safe | unsafe |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| safe | {confusion['safe->safe']} | {confusion['safe->unsafe']} |")
        lines.append(f"| unsafe | {confusion['unsafe->safe']} | {confusion['unsafe->unsafe']} |")
        lines.append("")

    lines.extend(["## Predicted Label Distribution", ""])
    for model_name in MODEL_ORDER:
        pred_dist = results[model_name]["summary"]["pred_label_distribution"]
        lines.append(f"- {model_name}: `safe={pred_dist.get('safe', 0)}`, `unsafe={pred_dist.get('unsafe', 0)}`")

    if pairwise_delta:
        lines.extend(["", "## Pairwise Deltas", ""])
        for delta_name, delta_values in pairwise_delta.items():
            lines.append(f"### {delta_name}")
            lines.append("")
            for metric in FLAT_METRIC_ORDER:
                lines.append(f"- `{metric}`: {delta_values[metric]:+.4f}")
            lines.append("")

    lines.extend(["## Best Model by Metric", ""])
    for metric in FLAT_METRIC_ORDER:
        best_model, best_value = ranking[metric][0]
        lines.append(f"- `{metric}`: **{best_model}** ({best_value:.4f})")

    if takeaways:
        lines.extend(["", "## Key Takeaways", ""])
        for takeaway in takeaways:
            lines.append(f"- {takeaway}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline, base-model, and LoRA-model prediction files.")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--reference_file", type=Path, default=root / "data" / "processed" / "test.jsonl")
    parser.add_argument("--baseline_file", type=Path, default=root / "outputs" / "baseline_cls" / "test_predictions.jsonl")
    parser.add_argument("--qwen_base_file", type=Path, default=root / "outputs" / "qwen_base_test_predictions.jsonl")
    parser.add_argument("--qwen_lora_file", type=Path, default=root / "outputs" / "qwen_lora_test_predictions.jsonl")
    parser.add_argument("--output_dir", type=Path, default=root / "outputs" / "step09_compare")
    args = parser.parse_args()

    reference_rows = read_jsonl(args.reference_file)
    validate_reference_rows(reference_rows, args.reference_file)

    input_files = {
        "baseline": args.baseline_file,
        "qwen_base": args.qwen_base_file,
        "qwen_lora": args.qwen_lora_file,
    }
    results = {
        model_name: evaluate_prediction_file(
            prediction_path=prediction_path,
            reference_rows=reference_rows,
            reference_path=args.reference_file,
            model_name=model_name,
        )
        for model_name, prediction_path in input_files.items()
    }

    pairwise_delta = compute_pairwise_delta(results)
    ranking = rank_models(results)
    takeaways = build_takeaways(pairwise_delta, ranking)
    payload = {
        "reference_file": str(args.reference_file),
        "inputs": {key: str(value) for key, value in input_files.items()},
        "results": results,
        "pairwise_delta": pairwise_delta,
        "ranking": ranking,
        "takeaways": takeaways,
        "label_space": sorted(VALID_LABELS),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "comparison_analysis.json", payload)
    with (args.output_dir / "comparison_analysis.md").open("w", encoding="utf-8") as f:
        f.write(build_markdown(args.reference_file, input_files, results, pairwise_delta, ranking, takeaways))

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
