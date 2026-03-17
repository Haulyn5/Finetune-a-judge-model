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
    - Refusal-style explanations are low-value supervision for this task.
"""

import argparse
import json
from pathlib import Path


REFUSAL_PATTERNS = [
    "cannot help",
    "can't help",
    "cannot assist",
    "can't assist",
    "sorry, but",
    "i won't provide",
    "i cannot provide",
]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter pseudo-labeled rationale data for the main Qwen SFT path.")
    parser.add_argument("--input_path", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "interim" / "pseudo_raw.jsonl")
    parser.add_argument("--output_path", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "pseudo_filtered.jsonl")
    parser.add_argument("--min_reason_chars", type=int, default=20)
    parser.add_argument("--max_reason_chars", type=int, default=400)
    args = parser.parse_args()

    rows = read_jsonl(args.input_path)
    kept = []

    for row in rows:
        teacher = row.get("teacher_output") or {}
        label = str(teacher.get("label", "")).strip().lower()
        reason = str(teacher.get("reason", "")).strip()
        evidence = teacher.get("evidence") or []
        confidence = teacher.get("confidence")

        # Matching the gold binary label keeps the structured supervision aligned
        # with the earlier dataset normalization step.
        if label != row["label"]:
            continue
        if not isinstance(evidence, list) or len(evidence) < 1:
            continue

        source_text = f"{row['question']}\n{row['response']}"
        if not evidence_matches(source_text, [str(x) for x in evidence]):
            continue
        if not (args.min_reason_chars <= len(reason) <= args.max_reason_chars):
            continue

        lowered_reason = reason.lower()
        if any(pattern in lowered_reason for pattern in REFUSAL_PATTERNS):
            continue
        if confidence is None:
            confidence = 0.5

        kept.append(
            {
                **row,
                "reason": reason,
                "evidence": [str(x).strip() for x in evidence],
                "confidence": float(confidence),
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"input": len(rows), "kept": len(kept), "output_path": str(args.output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
