#!/usr/bin/env python3
"""Filter teacher pseudo labels into higher-quality SFT supervision.

Pipeline step:
    05 / 08

Goal:
    Keep only pseudo labels that are structurally usable and broadly consistent
    with the binary ground-truth label from step 02. This quality filter matters
    because noisy rationale data can hurt the main structured judge much more
    than it hurts a simple classifier.

Inputs:
    - ``data/interim/pseudo_raw.jsonl`` from step 04.

Outputs:
    - ``data/processed/pseudo_filtered.jsonl``

Key assumptions:
    - Teacher label should agree with the gold binary label.
    - Evidence should be directly grounded in the source text.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_DROPPED_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "interim" / "pseudo_dropped.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    """Read raw pseudo-labeled rows from disk."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evidence_matches(text: str, evidence_list: list[str]) -> bool:
    """Check whether every evidence span appears in the source text.

    This grounding check is intentionally strict because evidence that cannot be
    found in the source is not useful as supervision for a judge model.
    """
    lowered = text.lower()
    return all(ev.strip() and ev.lower() in lowered for ev in evidence_list)


def evaluate_row(row: dict, min_reason_chars: int, max_reason_chars: int) -> tuple[bool, str | None, dict]:
    """Return keep/drop decision with normalized teacher fields."""
    teacher = row.get("teacher_output") or {}
    label = str(teacher.get("label", "")).strip().lower()
    reason = str(teacher.get("reason", "")).strip()
    evidence = teacher.get("evidence") or []
    normalized = {
        "teacher_label": label,
        "teacher_reason": reason,
        "teacher_evidence": evidence,
    }

    if label != row["label"]:
        return False, "label_mismatch", normalized
    if not isinstance(evidence, list) or len(evidence) < 1:
        return False, "missing_evidence", normalized

    normalized_evidence = [str(x).strip() for x in evidence]
    normalized["teacher_evidence"] = normalized_evidence
    source_text = f"{row['question']}\n{row['response']}"
    if not evidence_matches(source_text, normalized_evidence):
        return False, "evidence_not_grounded", normalized
    if not (min_reason_chars <= len(reason) <= max_reason_chars):
        return False, "reason_length_out_of_range", normalized

    return True, None, normalized


def build_dropped_row(row: dict, drop_reason: str, normalized: dict, min_reason_chars: int, max_reason_chars: int) -> dict:
    """Preserve original row fields while annotating why filtering dropped it."""
    return {
        **row,
        **normalized,
        "drop_reason": drop_reason,
        "dropped_at_step": "05_filter_pseudo_labels",
        "filter_config": {
            "min_reason_chars": min_reason_chars,
            "max_reason_chars": max_reason_chars,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter pseudo-labeled rationale data for the main Qwen SFT path.")
    parser.add_argument("--input_path", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "interim" / "pseudo_raw.jsonl")
    parser.add_argument("--output_path", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "pseudo_filtered.jsonl")
    parser.add_argument("--label_mismatch_output_path", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "pseudo_label_mismatch.jsonl")
    parser.add_argument(
        "--dropped_output_path",
        type=Path,
        default=DEFAULT_DROPPED_OUTPUT_PATH,
        help="Optional JSONL path for all dropped rows so step 04 can rerun only the rejected subset.",
    )
    parser.add_argument("--min_reason_chars", type=int, default=20)
    parser.add_argument("--max_reason_chars", type=int, default=400)
    args = parser.parse_args()

    rows = read_jsonl(args.input_path)
    kept = []
    dropped_rows = []
    label_mismatch_rows = []
    drop_reasons = Counter()

    for row in rows:
        is_kept, drop_reason, normalized = evaluate_row(row, args.min_reason_chars, args.max_reason_chars)
        if not is_kept:
            drop_reasons[drop_reason] += 1
            dropped_rows.append(build_dropped_row(row, drop_reason, normalized, args.min_reason_chars, args.max_reason_chars))
            if drop_reason == "label_mismatch":
                label_mismatch_rows.append(
                    {
                        **row,
                        "teacher_label": normalized["teacher_label"],
                        "teacher_reason": normalized["teacher_reason"],
                        "teacher_evidence": normalized["teacher_evidence"],
                    }
                )
            continue

        kept.append(
            {
                **row,
                "reason": normalized["teacher_reason"],
                "evidence": normalized["teacher_evidence"],
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.label_mismatch_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.label_mismatch_output_path.open("w", encoding="utf-8") as f:
        for row in label_mismatch_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.dropped_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.dropped_output_path.open("w", encoding="utf-8") as f:
        for row in dropped_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": len(rows),
                "kept": len(kept),
                "dropped": len(rows) - len(kept),
                "drop_reasons": dict(drop_reasons),
                "kept_label_counts": dict(Counter(row["label"] for row in kept)),
                "label_mismatch_saved": len(label_mismatch_rows),
                "output_path": str(args.output_path),
                "label_mismatch_output_path": str(args.label_mismatch_output_path),
                "dropped_output_path": str(args.dropped_output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
