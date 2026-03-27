#!/usr/bin/env python3
"""Generate rationale-style pseudo labels with a teacher LLM.

Pipeline step:
    04 / 08

Goal:
    Use a stronger instruction-following teacher model to produce structured
    supervision for the main Qwen judge path. This is the first step that moves
    beyond plain binary classification toward ``label/reason/evidence``.

Inputs:
    - ``data/processed/train.jsonl`` or another normalized binary JSONL file.

Outputs:
    - ``data/interim/pseudo_raw.jsonl`` with raw teacher text and parsed JSON.

Key assumptions:
    - The teacher is prompted to return JSON only.
    - In practice, LLM output is not perfectly reliable, so we keep both the raw
      decoded text and the best-effort parsed JSON for later filtering.
    - vLLM is used as the inference backend to improve throughput on large Qwen
      teacher models.
    - Thinking mode is disabled by passing ``enable_thinking=False`` to
      ``tokenizer.apply_chat_template()``, per the Qwen3.5 README.
    - Teacher generation should cover both binary labels evenly, so this script
      can optionally sample a balanced subset before generation.
    - This step currently assumes a bf16-capable GPU environment because the
      vLLM teacher path is initialized with ``dtype=\"bfloat16\"``.
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from _common import choose_idle_gpu
from _prompts import load_named_prompt_bundle, render_user_prompt
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


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
        return [json.loads(line) for line in f]


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
    return log_dir / f"step04_teacher_run_{timestamp}.jsonl"


def emit_event(event: str, **payload) -> None:
    """Print a compact JSON progress event for long-running step 04 runs."""
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


def format_prompts_no_think(
    messages_list: list[list[dict]], tokenizer: AutoTokenizer
) -> list[str]:
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


def select_rows(rows: list[dict], max_samples: int, seed: int, strategy: str) -> tuple[list[dict], dict]:
    """Select rows for teacher generation.

    ``balanced`` draws an equal number of ``safe`` and ``unsafe`` examples.
    ``first_n`` preserves the earlier deterministic smoke-test behavior.
    """
    if strategy == "first_n":
        selected = rows[:max_samples]
        return selected, {
            "sampling_strategy": strategy,
            "sampling_seed": seed,
            "requested_samples": max_samples,
            "selected_samples": len(selected),
            "selected_label_counts": dict(Counter(row["label"] for row in selected)),
        }

    grouped: dict[str, list[dict]] = {"safe": [], "unsafe": []}
    for row in rows:
        label = row.get("label")
        if label in grouped:
            grouped[label].append(row)

    if max_samples % 2 != 0:
        raise ValueError("Balanced sampling requires an even --max_samples value.")

    target_per_label = max_samples // 2
    availability = {label: len(label_rows) for label, label_rows in grouped.items()}
    shortages = {label: target_per_label - count for label, count in availability.items() if count < target_per_label}
    if shortages:
        raise ValueError(
            "Balanced sampling could not satisfy the requested sample count. "
            f"Requested per label={target_per_label}, availability={availability}."
        )

    rng = random.Random(seed)
    selected = []
    for label in ["safe", "unsafe"]:
        label_rows = list(grouped[label])
        rng.shuffle(label_rows)
        selected.extend(label_rows[:target_per_label])

    rng.shuffle(selected)
    return selected, {
        "sampling_strategy": strategy,
        "sampling_seed": seed,
        "requested_samples": max_samples,
        "target_per_label": target_per_label,
        "available_label_counts": availability,
        "selected_samples": len(selected),
        "selected_label_counts": dict(Counter(row["label"] for row in selected)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rationale pseudo labels with a teacher model for the main structured judge path.")
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "train.jsonl",
    )
    parser.add_argument("--output_path", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "interim" / "pseudo_raw.jsonl")
    parser.add_argument(
        "--merge_base_output_path",
        type=Path,
        default=None,
        help="Optional prior pseudo_raw path to merge rerun rows into after generation. Requires --merged_output_path.",
    )
    parser.add_argument(
        "--merged_output_path",
        type=Path,
        default=None,
        help="Optional final merged pseudo_raw path written after combining --merge_base_output_path with this run's output.",
    )
    parser.add_argument("--teacher_model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument(
        "--teacher_model_dir",
        type=Path,
        default=DEFAULT_TEACHER_MODEL_DIR,
        help="Local directory used to store and load the teacher model. Defaults to <repo>/../PretrainedModels/<model-name>/.",
    )
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--sampling_strategy", choices=["balanced", "first_n"], default="balanced")
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--rerun_tag",
        default=None,
        help="Optional tag recorded in each output row for drop-subset reruns or other targeted teacher passes.",
    )
    parser.add_argument("--gpu_id", type=int, default=None, help="Use a specific single GPU. If omitted, the script picks the least-used GPU by current memory usage.")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs",
        help="Directory for dated JSONL runtime logs so each run can be audited later.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of prompts to submit to vLLM together. Larger values improve throughput but use more VRAM.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Cap the serving context length for vLLM. Keeping this far below the model maximum improves memory efficiency for this pipeline.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Target GPU memory utilization for vLLM KV cache allocation.",
    )
    args = parser.parse_args()

    if bool(args.merge_base_output_path) != bool(args.merged_output_path):
        parser.error("--merge_base_output_path and --merged_output_path must be provided together.")
    if args.merge_base_output_path is not None and args.merge_base_output_path == args.output_path:
        parser.error("--output_path must differ from --merge_base_output_path so rerun outputs do not overwrite the base file before merge.")

    selected_gpu = args.gpu_id if args.gpu_id is not None else choose_idle_gpu()
    if selected_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        emit_event("device_selected", selected_gpu=selected_gpu, device_mode="single_gpu")
    else:
        emit_event("device_selected", selected_gpu=None, device_mode="cpu")

    all_rows = read_jsonl(args.input_path)
    rows, sampling_summary = select_rows(all_rows, args.max_samples, args.sampling_seed, args.sampling_strategy)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = build_run_log_path(args.log_dir)
    run_started_at = time.time()
    emit_event("input_loaded", input_path=str(args.input_path), total_input_rows=len(all_rows), selected_rows=len(rows))
    local_teacher_path = resolve_teacher_model_path(args.teacher_model, args.teacher_model_dir)
    emit_event("teacher_model_ready", teacher_model=args.teacher_model, teacher_model_dir=str(local_teacher_path))
    emit_event("sampling_summary", **sampling_summary)

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

    run_log_events = [
        {
            "event": "run_started",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(args.input_path),
            "output_path": str(args.output_path),
            "teacher_model": args.teacher_model,
            "teacher_model_dir": str(local_teacher_path),
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "rerun_tag": args.rerun_tag,
            "selected_gpu": selected_gpu,
            "row_count": len(rows),
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "backend": "vllm",
            "enable_thinking": False,
            **sampling_summary,
        }
    ]

    emit_event(
        "generation_config",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        rerun_tag=args.rerun_tag,
        output_path=str(args.output_path),
    )

    emit_event(
        "llm_initializing",
        model=str(local_teacher_path),
        tokenizer=str(local_teacher_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
    )

    llm = LLM(
        model=str(local_teacher_path),
        tokenizer=str(local_teacher_path),
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        dtype="bfloat16",
    )
    emit_event("llm_ready", model=str(local_teacher_path), row_count=len(rows))

    with args.output_path.open("w", encoding="utf-8") as f:
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

    emit_event("generation_saved", saved=len(rows), output_path=str(args.output_path))

    merge_summary = None
    if args.merge_base_output_path is not None and args.merged_output_path is not None:
        emit_event(
            "merge_started",
            merge_base_output_path=str(args.merge_base_output_path),
            rerun_output_path=str(args.output_path),
            merged_output_path=str(args.merged_output_path),
        )
        args.merged_output_path.parent.mkdir(parents=True, exist_ok=True)
        merge_summary = merge_rerun_outputs(
            base_output_path=args.merge_base_output_path,
            rerun_output_path=args.output_path,
            merged_output_path=args.merged_output_path,
            rerun_tag=args.rerun_tag,
        )
        emit_event("merge_completed", **merge_summary)

    run_log_events.append(
        {
            "event": "run_completed",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "saved": len(rows),
            "output_path": str(args.output_path),
            "run_duration_sec": round(time.time() - run_started_at, 3),
            "merge_summary": merge_summary,
        }
    )
    write_run_log(run_log_path, run_log_events)
    emit_event("run_log_saved", run_log_path=str(run_log_path))


if __name__ == "__main__":
    main()
