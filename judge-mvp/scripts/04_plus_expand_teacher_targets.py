#!/usr/bin/env python3
"""Expand teacher pseudo labels to per-label targets while reusing existing rows.

Temporary helper for step 04.

Goal:
    Reuse already teacher-processed rows from ``data/interim/pseudo_raw.jsonl``
    and only generate new teacher outputs for the missing safe/unsafe deficit.

Default targets:
    - safe: 7500
    - unsafe: 7500

Behavior:
    - Backs up the current pseudo_raw file before any generation/merge.
    - Skips rows whose ``id`` already exists in the existing teacher file.
    - Generates only the missing rows.
    - Merges new rows back by ``id`` using the existing step04 merge logic.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from _common import choose_idle_gpu
from _prompts import load_named_prompt_bundle, render_user_prompt
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[0]
DEFAULT_INPUT_PATH = PROJECT_DIR / "data" / "processed" / "train.jsonl"
DEFAULT_EXISTING_PSEUDO_PATH = PROJECT_DIR / "data" / "interim" / "pseudo_raw.jsonl"
DEFAULT_NEW_OUTPUT_PATH = PROJECT_DIR / "data" / "interim" / "pseudo_raw_plus_new.jsonl"
DEFAULT_MERGED_OUTPUT_PATH = PROJECT_DIR / "data" / "interim" / "pseudo_raw_expanded.jsonl"
DEFAULT_BACKUP_DIR = PROJECT_DIR / "backups" / "teacher_data"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_TEACHER_MODEL_DIR = Path(__file__).resolve().parents[2] / "PretrainedModels" / "Qwen3.5-27B"
STEP05_METADATA_FIELDS = {
    "drop_reason",
    "dropped_at_step",
    "filter_config",
    "teacher_label",
    "teacher_reason",
    "teacher_evidence",
}


def sanitize_source_row(row: dict) -> dict:
    """Remove retry-helper metadata before writing a fresh teacher result."""
    return {key: value for key, value in row.items() if key not in STEP05_METADATA_FIELDS}


def read_jsonl(path: Path) -> list[dict]:
    """Read normalized JSONL rows used for teacher prompting."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_json(text: str) -> dict | None:
    """Extract structured teacher output from JSON or fallback text."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            allowed_keys = {"label", "reason", "evidence"}
            return {key: value for key, value in parsed.items() if key in allowed_keys}

    reason_match = re.search(
        r"^(The response is (?:safe|unsafe) because[\s\S]*?)(?:\n\s*Evidence\s*:\s*|$)",
        cleaned,
        re.IGNORECASE,
    )
    if not reason_match:
        return None

    reason = reason_match.group(1).strip()
    evidence_matches = re.findall(r"(?:^|\n)\s*\d+\.\s*[\"']?(.+?)[\"']?\s*(?=\n\s*\d+\.|\Z)", cleaned, re.DOTALL)
    evidence = [item.strip().strip('"').strip("'") for item in evidence_matches if item.strip()]
    return {
        "reason": reason,
        "evidence": evidence,
    }


def fill_missing_label(parsed: dict | None, gold_label: str) -> dict | None:
    """Backfill missing teacher label when JSON omitted it."""
    if parsed is None:
        return None
    if parsed.get("label") in {"safe", "unsafe"}:
        return parsed

    reason = str(parsed.get("reason", "")).strip().lower()
    label_match = re.search(r"\bresponse is (safe|unsafe)\b", reason)
    if label_match:
        parsed["label"] = label_match.group(1)
    else:
        parsed["label"] = gold_label
    return parsed


def resolve_teacher_model_path(model_name: str, model_dir: Path) -> Path:
    """Ensure the teacher model is available under the requested local directory."""
    if model_dir.exists() and any(model_dir.iterdir()):
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_name, local_dir=str(model_dir))
    return model_dir


def build_run_log_path(log_dir: Path) -> Path:
    """Create a dated log path so repeated runs do not overwrite prior logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"step04_plus_teacher_run_{timestamp}.jsonl"


def emit_event(event: str, **payload) -> None:
    """Print a compact JSON progress event for long-running step04_plus runs."""
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def merge_rerun_outputs(base_output_path: Path, rerun_output_path: Path, merged_output_path: Path, rerun_tag: str | None) -> dict:
    """Merge rerun rows into a prior pseudo_raw file using id as the primary key."""
    if not base_output_path.exists():
        raise FileNotFoundError(f"Base output path does not exist: {base_output_path}")

    with base_output_path.open("r", encoding="utf-8") as f:
        base_rows = [json.loads(line) for line in f if line.strip()]
    with rerun_output_path.open("r", encoding="utf-8") as f:
        rerun_rows = [json.loads(line) for line in f if line.strip()]

    rerun_by_id = {row["id"]: row for row in rerun_rows}
    merged_ids = set()
    replaced = 0

    with merged_output_path.open("w", encoding="utf-8") as f:
        for row in base_rows:
            row_id = row.get("id")
            if row_id in rerun_by_id:
                f.write(json.dumps(rerun_by_id[row_id], ensure_ascii=False) + "\n")
                merged_ids.add(row_id)
                replaced += 1
            else:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        appended = 0
        for row in rerun_rows:
            row_id = row.get("id")
            if row_id in merged_ids:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            appended += 1

    return {
        "base_output_path": str(base_output_path),
        "rerun_output_path": str(rerun_output_path),
        "merged_output_path": str(merged_output_path),
        "rerun_tag": rerun_tag,
        "base_count": len(base_rows),
        "rerun_count": len(rerun_rows),
        "replaced": replaced,
        "appended": appended,
        "merged_total": len(base_rows) - replaced + len(rerun_rows),
    }


def write_run_log(log_path: Path, events: list[dict]) -> None:
    """Write JSON log events in one append to reduce per-sample I/O overhead."""
    if not events:
        return
    with log_path.open("a", encoding="utf-8") as f:
        for payload in events:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_messages(row: dict, prompt_bundle) -> list[dict]:
    """Format one training row into a chat messages list."""
    return [
        {"role": "system", "content": prompt_bundle.system},
        {
            "role": "user",
            "content": render_user_prompt(
                prompt_bundle.user,
                question=row["question"],
                response=row["response"],
                label=row.get("label", ""),
            ),
        },
    ]


def format_prompts_no_think(messages_list: list[list[dict]], tokenizer: AutoTokenizer) -> list[str]:
    """Apply the Qwen3.5 chat template with thinking disabled."""
    return [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in messages_list
    ]



def backup_existing_pseudo(existing_pseudo_path: Path, backup_dir: Path) -> Path | None:
    """Copy the current teacher file to a timestamped backup path."""
    if not existing_pseudo_path.exists():
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{existing_pseudo_path.stem}_before_step04_plus_{timestamp}{existing_pseudo_path.suffix}"
    shutil.copy2(existing_pseudo_path, backup_path)
    return backup_path



def index_rows_by_id(rows: list[dict]) -> dict[str, dict]:
    """Index rows by id using the last occurrence if duplicates exist."""
    indexed: dict[str, dict] = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if row_id:
            indexed[row_id] = row
    return indexed



def count_labels(rows: list[dict]) -> dict[str, int]:
    """Count safe/unsafe labels in a row collection."""
    counts = Counter()
    for row in rows:
        label = row.get("label")
        if label in {"safe", "unsafe"}:
            counts[label] += 1
    return {"safe": counts.get("safe", 0), "unsafe": counts.get("unsafe", 0)}



def compute_deficits(existing_counts: dict[str, int], target_safe: int, target_unsafe: int) -> dict[str, int]:
    """Compute how many additional rows are needed per label."""
    return {
        "safe": max(0, target_safe - existing_counts.get("safe", 0)),
        "unsafe": max(0, target_unsafe - existing_counts.get("unsafe", 0)),
    }



def select_missing_rows(
    train_rows: list[dict],
    existing_by_id: dict[str, dict],
    deficits: dict[str, int],
    seed: int,
) -> tuple[list[dict], dict]:
    """Select new rows to generate while skipping already processed ids."""
    rng = random.Random(seed)
    grouped_candidates: dict[str, list[dict]] = {"safe": [], "unsafe": []}
    skipped_existing = 0

    for row in train_rows:
        row_id = str(row.get("id", "")).strip()
        label = row.get("label")
        if label not in grouped_candidates:
            continue
        if deficits.get(label, 0) <= 0:
            continue
        if row_id in existing_by_id:
            skipped_existing += 1
            continue
        grouped_candidates[label].append(row)

    selected: list[dict] = []
    selected_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    shortfalls: dict[str, int] = {}

    for label in ["safe", "unsafe"]:
        label_candidates = list(grouped_candidates[label])
        rng.shuffle(label_candidates)
        need = deficits.get(label, 0)
        chosen = label_candidates[:need]
        selected.extend(chosen)
        candidate_counts[label] = len(label_candidates)
        selected_counts[label] = len(chosen)
        shortfalls[label] = max(0, need - len(chosen))

    rng.shuffle(selected)
    summary = {
        "sampling_seed": seed,
        "candidate_counts": candidate_counts,
        "requested_counts": deficits,
        "selected_counts": selected_counts,
        "selected_total": len(selected),
        "shortfalls": shortfalls,
        "skipped_existing": skipped_existing,
    }
    return selected, summary



def summarize_targets(existing_counts: dict[str, int], generated_counts: dict[str, int], targets: dict[str, int]) -> dict:
    """Build a compact target progress summary."""
    final_counts = {
        "safe": existing_counts.get("safe", 0) + generated_counts.get("safe", 0),
        "unsafe": existing_counts.get("unsafe", 0) + generated_counts.get("unsafe", 0),
    }
    return {
        "existing_counts": existing_counts,
        "generated_counts": generated_counts,
        "final_counts": final_counts,
        "targets": targets,
        "remaining_shortfalls": {
            "safe": max(0, targets["safe"] - final_counts["safe"]),
            "unsafe": max(0, targets["unsafe"] - final_counts["unsafe"]),
        },
    }



def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)



def generate_rows(args, rows: list[dict], run_log_events: list[dict]) -> tuple[dict[str, int], Path]:
    """Run teacher generation for the selected new rows only."""
    from vllm import LLM, SamplingParams

    selected_gpu = args.gpu_id if args.gpu_id is not None else choose_idle_gpu()
    if selected_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        emit_event("device_selected", selected_gpu=selected_gpu, device_mode="single_gpu")
    else:
        emit_event("device_selected", selected_gpu=None, device_mode="cpu")

    ensure_parent(args.new_output_path)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = build_run_log_path(args.log_dir)

    local_teacher_path = resolve_teacher_model_path(args.teacher_model, args.teacher_model_dir)
    emit_event("teacher_model_ready", teacher_model=args.teacher_model, teacher_model_dir=str(local_teacher_path))

    tokenizer = AutoTokenizer.from_pretrained(local_teacher_path)
    prompt_bundle = load_named_prompt_bundle("teacher")
    messages_list = [build_messages(row, prompt_bundle) for row in rows]
    prompts = format_prompts_no_think(messages_list, tokenizer)
    emit_event("prompts_built", prompt_count=len(prompts), prompt_bundle="teacher", enable_thinking=False)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=0.8,
        top_k=20,
        presence_penalty=1.5,
    )

    emit_event(
        "generation_config",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        rerun_tag=args.rerun_tag,
        output_path=str(args.new_output_path),
    )

    emit_event(
        "llm_initializing",
        model=str(local_teacher_path),
        tokenizer=str(local_teacher_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
    )
    llm = LLM(
        model=str(local_teacher_path),
        tokenizer=str(local_teacher_path),
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
    )
    emit_event("llm_ready", model=str(local_teacher_path), row_count=len(rows))

    generated_counts = Counter()
    with args.new_output_path.open("w", encoding="utf-8") as f:
        completed = 0
        for batch_start in range(0, len(rows), args.batch_size):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            batch_prompts = prompts[batch_start : batch_start + args.batch_size]
            batch_index = batch_start // args.batch_size + 1
            emit_event(
                "batch_started",
                batch_index=batch_index,
                batch_size=len(batch_rows),
                completed=completed,
                total=len(rows),
            )
            batch_started_at = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            batch_duration_sec = max(time.time() - batch_started_at, 1e-6)

            for row, prompt, output in zip(batch_rows, batch_prompts, outputs):
                completed += 1
                result = output.outputs[0]
                decoded = result.text.strip()
                parsed = extract_json(decoded)
                parsed = fill_missing_label(parsed, row["label"])
                prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
                generated_token_ids = result.token_ids or []
                generated_token_count = len(generated_token_ids)
                sample_duration_sec = round(batch_duration_sec / len(batch_rows), 3)
                record = {
                    **sanitize_source_row(row),
                    "teacher_model": args.teacher_model,
                    "teacher_model_dir": str(local_teacher_path),
                    "teacher_raw_text": decoded,
                    "teacher_output": parsed,
                    "rerun_tag": args.rerun_tag,
                }
                generated_counts[row["label"]] += 1
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                run_log_events.append(
                    {
                        "event": "sample_completed",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "index": completed,
                        "total": len(rows),
                        "batch_index": batch_index,
                        "batch_size": len(batch_rows),
                        "id": row.get("id"),
                        "gold_label": row.get("label"),
                        "parsed_ok": parsed is not None,
                        "sample_duration_sec": sample_duration_sec,
                        "prompt_token_count": prompt_token_count,
                        "generated_token_count": generated_token_count,
                        "output_chars": len(decoded),
                    }
                )
                emit_event(
                    "sample_completed",
                    sample_index=completed,
                    total=len(rows),
                    id=row.get("id"),
                    parsed_ok=parsed is not None,
                    sample_duration_sec=sample_duration_sec,
                    generated_token_count=generated_token_count,
                )

            emit_event(
                "batch_completed",
                batch_index=batch_index,
                batch_size=len(batch_rows),
                completed=completed,
                total=len(rows),
                batch_duration_sec=round(batch_duration_sec, 3),
                avg_sample_duration_sec=round(batch_duration_sec / len(batch_rows), 3),
            )

    emit_event("generation_saved", saved=len(rows), output_path=str(args.new_output_path))
    return {"safe": generated_counts.get("safe", 0), "unsafe": generated_counts.get("unsafe", 0)}, run_log_path



def main() -> None:
    parser = argparse.ArgumentParser(description="Expand teacher pseudo labels to target safe/unsafe counts while skipping already processed ids.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--existing_pseudo_path", type=Path, default=DEFAULT_EXISTING_PSEUDO_PATH)
    parser.add_argument("--new_output_path", type=Path, default=DEFAULT_NEW_OUTPUT_PATH)
    parser.add_argument("--merged_output_path", type=Path, default=DEFAULT_MERGED_OUTPUT_PATH)
    parser.add_argument("--backup_dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--target_safe", type=int, default=7500)
    parser.add_argument("--target_unsafe", type=int, default=7500)
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument("--teacher_model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher_model_dir", type=Path, default=DEFAULT_TEACHER_MODEL_DIR)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rerun_tag", default="expand_to_target_counts")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--log_dir", type=Path, default=PROJECT_DIR / "logs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enforce_eager", action="store_true", default=True)
    parser.add_argument("--fail_if_target_unreachable", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.target_safe < 0 or args.target_unsafe < 0:
        parser.error("target counts must be non-negative.")
    if args.new_output_path == args.existing_pseudo_path:
        parser.error("--new_output_path must differ from --existing_pseudo_path.")
    if args.merged_output_path == args.new_output_path:
        parser.error("--merged_output_path must differ from --new_output_path.")

    train_rows = read_jsonl(args.input_path)
    existing_rows = read_jsonl(args.existing_pseudo_path) if args.existing_pseudo_path.exists() else []
    existing_by_id = index_rows_by_id(existing_rows)
    existing_counts = count_labels(existing_rows)
    targets = {"safe": args.target_safe, "unsafe": args.target_unsafe}
    deficits = compute_deficits(existing_counts, args.target_safe, args.target_unsafe)

    selected_rows, selection_summary = select_missing_rows(
        train_rows=train_rows,
        existing_by_id=existing_by_id,
        deficits=deficits,
        seed=args.sampling_seed,
    )
    shortfalls = selection_summary["shortfalls"]
    unreachable = any(shortfalls[label] > 0 for label in ["safe", "unsafe"])

    summary = {
        "input_path": str(args.input_path),
        "existing_pseudo_path": str(args.existing_pseudo_path),
        "existing_rows": len(existing_rows),
        "existing_counts": existing_counts,
        "targets": targets,
        "deficits": deficits,
        **selection_summary,
        "dry_run": args.dry_run,
    }

    if args.fail_if_target_unreachable and unreachable:
        raise ValueError(
            "Target unreachable after skipping already processed rows. "
            f"Shortfalls={shortfalls}, candidate_counts={selection_summary['candidate_counts']}"
        )

    if deficits["safe"] == 0 and deficits["unsafe"] == 0:
        summary["status"] = "already_satisfied"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.dry_run:
        summary["status"] = "dry_run_ready"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    backup_path = backup_existing_pseudo(args.existing_pseudo_path, args.backup_dir)
    summary["backup_path"] = str(backup_path) if backup_path is not None else None

    run_started_at = time.time()
    run_log_events = [
        {
            "event": "run_started",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(args.input_path),
            "existing_pseudo_path": str(args.existing_pseudo_path),
            "new_output_path": str(args.new_output_path),
            "merged_output_path": str(args.merged_output_path),
            "backup_path": str(backup_path) if backup_path is not None else None,
            "teacher_model": args.teacher_model,
            "teacher_model_dir": str(args.teacher_model_dir),
            "target_safe": args.target_safe,
            "target_unsafe": args.target_unsafe,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "rerun_tag": args.rerun_tag,
            **summary,
        }
    ]

    emit_event("input_loaded", input_path=str(args.input_path), total_input_rows=len(train_rows), selected_rows=len(selected_rows))
    emit_event("existing_teacher_loaded", existing_pseudo_path=str(args.existing_pseudo_path), existing_rows=len(existing_rows), existing_counts=existing_counts)
    emit_event("selection_summary", **selection_summary)

    generated_counts = {"safe": 0, "unsafe": 0}
    run_log_path = args.log_dir / f"step04_plus_teacher_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    if selected_rows:
        generated_counts, step04_log_path = generate_rows(args, selected_rows, run_log_events)
        run_log_path = step04_log_path.with_name(step04_log_path.name.replace("step04_teacher_run_", "step04_plus_teacher_run_"))
    else:
        ensure_parent(args.new_output_path)
        with args.new_output_path.open("w", encoding="utf-8"):
            pass
        emit_event("generation_skipped", reason="no_new_rows_selected", output_path=str(args.new_output_path))

    if args.existing_pseudo_path.exists():
        ensure_parent(args.merged_output_path)
        merge_summary = merge_rerun_outputs(
            base_output_path=args.existing_pseudo_path,
            rerun_output_path=args.new_output_path,
            merged_output_path=args.merged_output_path,
            rerun_tag=args.rerun_tag,
        )
    else:
        ensure_parent(args.merged_output_path)
        shutil.copy2(args.new_output_path, args.merged_output_path)
        merge_summary = {
            "base_output_path": None,
            "rerun_output_path": str(args.new_output_path),
            "merged_output_path": str(args.merged_output_path),
            "rerun_tag": args.rerun_tag,
            "base_count": 0,
            "rerun_count": len(selected_rows),
            "replaced": 0,
            "appended": len(selected_rows),
            "merged_total": len(selected_rows),
        }

    target_summary = summarize_targets(existing_counts, generated_counts, targets)
    final_summary = {
        **summary,
        **target_summary,
        "generated_output_path": str(args.new_output_path),
        "merged_output_path": str(args.merged_output_path),
        "merge_summary": merge_summary,
        "run_duration_sec": round(time.time() - run_started_at, 3),
        "status": "completed",
    }
    run_log_events.append(
        {
            "event": "run_completed",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            **final_summary,
        }
    )
    write_run_log(run_log_path, run_log_events)
    emit_event("run_log_saved", run_log_path=str(run_log_path))
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
