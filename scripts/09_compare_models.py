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

from _eval_common import VALID_LABELS, evaluate_prediction_file, read_jsonl, validate_reference_rows, write_json

PAIRWISE_ORDER = [
    ("qwen_base", "baseline", "qwen_base_minus_baseline"),
    ("qwen_lora", "baseline", "qwen_lora_minus_baseline"),
    ("qwen_lora", "qwen_base", "qwen_lora_minus_qwen_base"),
]
MODEL_ORDER = ("baseline", "qwen_base", "qwen_lora")
JSON_ANALYSIS_MODELS = ("qwen_base", "qwen_lora")
STEP09_METRIC_ORDER = [
    "accuracy",
    "macro_f1",
    "unsafe_precision",
    "unsafe_recall",
    "unsafe_f1",
    "raw_json_parse_success_rate",
    "raw_json_parse_failure_rate",
    "fallback_usable_rate",
    "evidence_hit_rate",
    "reason_label_consistency",
]


def safe_json_loads(text) -> dict | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_json_block(text: str) -> dict | None:
    parsed = safe_json_loads(text)
    if parsed is not None:
        return parsed
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return safe_json_loads(text[start : end + 1])


def analyze_generated_json_parseability(prediction_path: Path) -> dict:
    rows = read_jsonl(prediction_path)
    total = len(rows)
    direct_successes = 0
    extracted_successes = 0
    prediction_json_populated = 0

    for row in rows:
        raw_prediction = row.get("prediction")
        parsed_direct = safe_json_loads(raw_prediction)
        parsed_extracted = extract_json_block(raw_prediction)
        parsed_saved = row.get("prediction_json")

        if parsed_direct is not None:
            direct_successes += 1
        if parsed_extracted is not None:
            extracted_successes += 1
        if isinstance(parsed_saved, dict) and parsed_saved:
            prediction_json_populated += 1

    return {
        "num_rows": total,
        "direct_json_parse_success_count": direct_successes,
        "direct_json_parse_success_rate": direct_successes / total if total else 0.0,
        "extract_json_block_success_count": extracted_successes,
        "extract_json_block_success_rate": extracted_successes / total if total else 0.0,
        "prediction_json_populated_count": prediction_json_populated,
        "prediction_json_populated_rate": prediction_json_populated / total if total else 0.0,
    }


def compute_pairwise_delta(results: dict) -> dict:
    deltas = {}
    for left_name, right_name, delta_name in PAIRWISE_ORDER:
        if left_name not in results or right_name not in results:
            continue
        deltas[delta_name] = {
            metric: results[left_name]["flat_metrics"][metric] - results[right_name]["flat_metrics"][metric]
            for metric in STEP09_METRIC_ORDER
        }
    return deltas


def rank_models(results: dict) -> dict:
    metric_preferences = {
        "accuracy": True,
        "macro_f1": True,
        "unsafe_precision": True,
        "unsafe_recall": True,
        "unsafe_f1": True,
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


def build_takeaways(pairwise_delta: dict, ranking: dict, json_parse_analysis: dict) -> list[str]:
    takeaways = []
    macro_f1_winner, macro_f1_value = ranking["macro_f1"][0]
    unsafe_f1_winner, unsafe_f1_value = ranking["unsafe_f1"][0]
    parse_winner, parse_value = ranking["raw_json_parse_success_rate"][0]
    takeaways.append(f"Best overall macro_f1 run is {macro_f1_winner} ({macro_f1_value:.4f}).")
    takeaways.append(f"Best unsafe_f1 run is {unsafe_f1_winner} ({unsafe_f1_value:.4f}).")
    takeaways.append(f"Best strict structured-output reliability is {parse_winner} ({parse_value:.4f} raw_json_parse_success_rate).")
    for model_name in JSON_ANALYSIS_MODELS:
        analysis = json_parse_analysis.get(model_name)
        if not analysis:
            continue
        takeaways.append(
            f"{model_name} raw output JSON parseability: direct {analysis['direct_json_parse_success_rate']:.4f}, "
            f"step08_extract_json_block {analysis['extract_json_block_success_rate']:.4f}."
        )
    for delta_name in ("qwen_base_minus_baseline", "qwen_lora_minus_baseline", "qwen_lora_minus_qwen_base"):
        delta = pairwise_delta.get(delta_name)
        if not delta:
            continue
        takeaways.append(
            f"{delta_name}: macro_f1 {delta['macro_f1']:+.4f}, unsafe_f1 {delta['unsafe_f1']:+.4f}."
        )
    return takeaways


def build_markdown(
    reference_file: Path,
    input_files: dict,
    results: dict,
    pairwise_delta: dict,
    ranking: dict,
    takeaways: list[str],
    json_parse_analysis: dict,
) -> str:
    lines = ["# Step 09 Comparison Analysis", "", "## Inputs", ""]
    lines.append(f"- Reference file: `{reference_file}`")
    for model_name in MODEL_ORDER:
        lines.append(f"- {model_name}: `{input_files[model_name]}`")

    lines.extend(["", "## Metrics", ""])
    lines.append("| Model | " + " | ".join(STEP09_METRIC_ORDER) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(STEP09_METRIC_ORDER)) + " |")
    for model_name in MODEL_ORDER:
        metrics = results[model_name]["flat_metrics"]
        values = [f"{metrics[key]:.4f}" for key in STEP09_METRIC_ORDER]
        lines.append("| " + model_name + " | " + " | ".join(values) + " |")

    if json_parse_analysis:
        lines.extend(["", "## Step-08 JSON Parseability Analysis", ""])
        lines.append(
            "| Model | Direct JSON Parse Rate | Direct Success / Total | extract_json_block Parse Rate | extract_json_block Success / Total | prediction_json Populated Rate |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for model_name in JSON_ANALYSIS_MODELS:
            analysis = json_parse_analysis.get(model_name)
            if not analysis:
                continue
            lines.append(
                "| "
                + model_name
                + " | "
                + f"{analysis['direct_json_parse_success_rate']:.4f}"
                + " | "
                + f"{analysis['direct_json_parse_success_count']}/{analysis['num_rows']}"
                + " | "
                + f"{analysis['extract_json_block_success_rate']:.4f}"
                + " | "
                + f"{analysis['extract_json_block_success_count']}/{analysis['num_rows']}"
                + " | "
                + f"{analysis['prediction_json_populated_rate']:.4f}"
                + " |"
            )
        lines.extend(
            [
                "",
                "- `Direct JSON Parse Rate`: raw `prediction` can be parsed by `json.loads` directly.",
                "- `extract_json_block Parse Rate`: raw `prediction` can be parsed by the same `extract_json_block` fallback used in step 08.",
                "- `prediction_json Populated Rate`: the saved `prediction_json` field is a non-empty dict.",
            ]
        )

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
            for metric in STEP09_METRIC_ORDER:
                lines.append(f"- `{metric}`: {delta_values[metric]:+.4f}")
            lines.append("")

    lines.extend(["## Best Model by Metric", ""])
    for metric in STEP09_METRIC_ORDER:
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
    json_parse_analysis = {
        model_name: analyze_generated_json_parseability(input_files[model_name])
        for model_name in JSON_ANALYSIS_MODELS
        if model_name in input_files
    }

    pairwise_delta = compute_pairwise_delta(results)
    ranking = rank_models(results)
    takeaways = build_takeaways(pairwise_delta, ranking, json_parse_analysis)
    payload = {
        "reference_file": str(args.reference_file),
        "inputs": {key: str(value) for key, value in input_files.items()},
        "results": results,
        "json_parse_analysis": json_parse_analysis,
        "pairwise_delta": pairwise_delta,
        "ranking": ranking,
        "takeaways": takeaways,
        "label_space": sorted(VALID_LABELS),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "comparison_analysis.json", payload)
    with (args.output_dir / "comparison_analysis.md").open("w", encoding="utf-8") as f:
        f.write(build_markdown(args.reference_file, input_files, results, pairwise_delta, ranking, takeaways, json_parse_analysis))

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
