#!/usr/bin/env python3
"""Unified step-08 entry point for generation and single-file evaluation.

Pipeline step:
    08 / 08

Goal:
    Keep step 08 as the only entry point for:
    - generating Qwen base predictions on the canonical test set
    - generating Qwen LoRA predictions on the canonical test set
    - evaluating one prediction file
    - running generation + evaluation sequentially for base / lora / both

Important note:
    Step 08 intentionally uses the Transformers backend only. During project
    development, vLLM support for LoRA-adapted Qwen3.5 inference proved unstable
    in this workflow because of compatibility and version-conflict issues, so the
    project standardizes on Transformers here. Step 04 still uses vLLM for
    teacher-data generation.

Step 09 owns multi-model comparison and markdown analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from _common import choose_idle_gpu
from _eval_common import evaluate_prediction_file, read_jsonl, validate_reference_rows, write_json
from _prompts import load_named_prompt_bundle, load_prompt_bundle, render_user_prompt

DEFAULT_MODEL_PATH = Path("/root/project/PretrainedModels/Qwen/Qwen3.5-4B")
DEFAULT_MODEL_NAME = str(DEFAULT_MODEL_PATH)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_FILE = ROOT / "data" / "processed" / "test.jsonl"
DEFAULT_REFERENCE_FILE = ROOT / "data" / "processed" / "test.jsonl"
DEFAULT_BASE_OUTPUT_FILE = ROOT / "outputs" / "qwen_base_test_predictions.jsonl"
DEFAULT_LORA_OUTPUT_FILE = ROOT / "outputs" / "qwen_lora_test_predictions.jsonl"
DEFAULT_BASE_METRICS_FILE = ROOT / "outputs" / "qwen_base_metrics.json"
DEFAULT_LORA_METRICS_FILE = ROOT / "outputs" / "qwen_lora_metrics.json"
VALID_LABELS = {"safe", "unsafe"}


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_label(value) -> str:
    return str(value).strip().lower()


def safe_json_loads(text: str) -> dict | None:
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


def validate_test_rows(rows: list[dict], source_path: Path) -> None:
    if not rows:
        raise ValueError(f"Test set is empty: {source_path}")
    required_fields = ("id", "question", "response", "label")
    for line_number, row in enumerate(rows, start=1):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"Invalid row in {source_path} line {line_number}: missing fields {missing}.")
        label = normalize_label(row["label"])
        if label not in VALID_LABELS:
            raise ValueError(
                f"Invalid label in {source_path} line {line_number}: got {row['label']!r}, expected one of {sorted(VALID_LABELS)}."
            )


def ensure_output_path(output_file: Path, overwrite: bool) -> None:
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. Use a new path or pass --overwrite to replace it."
        )


def resolve_main_prompt_bundle(args) -> object:
    if args.main_system_prompt is not None or args.main_user_prompt is not None:
        if args.main_system_prompt is None or args.main_user_prompt is None:
            raise ValueError("--main_system_prompt and --main_user_prompt must be provided together.")
        return load_prompt_bundle(args.main_system_prompt, args.main_user_prompt)

    if getattr(args, "adapter_path", None) is not None:
        adapter_system = args.adapter_path / "main_system.txt"
        adapter_user = args.adapter_path / "main_user.txt"
        if adapter_system.exists() and adapter_user.exists():
            return load_prompt_bundle(adapter_system, adapter_user)

    return load_named_prompt_bundle("main")


def build_messages(row: dict, prompt_bundle) -> list[dict]:
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


def parse_prediction(text: str) -> tuple[str, dict | None, str, list[str]]:
    prediction_json = extract_json_block(text)
    if prediction_json is None:
        return "safe", None, "", []

    pred_label = normalize_label(prediction_json.get("label", ""))
    if pred_label not in VALID_LABELS:
        pred_label = "safe"

    reason = str(prediction_json.get("reason", "") or "")
    evidence = prediction_json.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if str(evidence).strip() else []
    evidence = [str(item) for item in evidence if str(item).strip()]
    return pred_label, prediction_json, reason, evidence


def build_record(row: dict, generated_text: str) -> dict:
    pred_label, prediction_json, reason, evidence = parse_prediction(generated_text)
    return {
        **row,
        "pred_label": pred_label,
        "prediction": generated_text,
        "prediction_json": prediction_json,
        "reason": reason,
        "evidence": evidence,
    }


def load_completed_ids(temp_output_file: Path) -> set[str]:
    if not temp_output_file.exists():
        return set()

    completed_ids: set[str] = set()
    with temp_output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" in row:
                completed_ids.add(str(row["id"]))
    return completed_ids


def log_progress(event: str, done: int, total: int, start_time: float, output_file: Path, extra: dict | None = None) -> None:
    elapsed = time.time() - start_time
    avg_seconds = elapsed / done if done else 0.0
    remaining = total - done
    eta_seconds = avg_seconds * remaining if done else None
    payload = {
        "event": event,
        "progress": {
            "done": done,
            "total": total,
            "percent": round((done / total) * 100, 2) if total else 0.0,
            "elapsed_seconds": round(elapsed, 2),
            "avg_seconds_per_sample": round(avg_seconds, 4) if done else None,
            "eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
            "output_file": str(output_file),
        },
    }
    if extra:
        payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def stream_with_transformers(job: dict, rows: list[dict], prompt_bundle, writer, start_time: float, already_done: int = 0) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(job["model_name_or_path"], use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        job["model_name_or_path"],
        dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    if job["adapter_path"] is not None:
        model = PeftModel.from_pretrained(model, job["adapter_path"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    done = already_done
    total = job["total_rows"]
    generation_kwargs = {
        "max_new_tokens": job["max_new_tokens"],
        "do_sample": job["temperature"] > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if job["temperature"] > 0:
        generation_kwargs["temperature"] = job["temperature"]

    for batch_index, batch_rows in enumerate(batched(rows, job["transformers_batch_size"]), start=1):
        rendered_prompts = [
            tokenizer.apply_chat_template(
                build_messages(row, prompt_bundle),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch_rows
        ]
        inputs = tokenizer(rendered_prompts, return_tensors="pt", padding=True).to(device)
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            generated = model.generate(**inputs, **generation_kwargs)

        for row, prompt_length, generated_ids in zip(batch_rows, prompt_lengths, generated):
            generated_text = tokenizer.decode(generated_ids[int(prompt_length) :], skip_special_tokens=True).strip()
            writer.write(json.dumps(build_record(row, generated_text), ensure_ascii=False) + "\n")
            done += 1

        writer.flush()
        if done == 1 or done % job["log_every"] == 0 or done == total:
            log_progress(
                event="generation_progress",
                done=done,
                total=total,
                start_time=start_time,
                output_file=job["output_file"],
                extra={"variant": job["variant_name"], "batch_index": batch_index},
            )
    return done


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("base", "lora", "both"), required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test_file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--main_system_prompt", type=Path, default=None)
    parser.add_argument("--main_user_prompt", type=Path, default=None)
    parser.add_argument("--adapter_path", type=Path, default=None)
    parser.add_argument("--output_file", type=Path, default=None, help="Prediction output path for single-variant mode.")
    parser.add_argument("--base_output_file", type=Path, default=DEFAULT_BASE_OUTPUT_FILE)
    parser.add_argument("--lora_output_file", type=Path, default=DEFAULT_LORA_OUTPUT_FILE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--transformers_batch_size", type=int, default=8)


def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference_file", type=Path, default=DEFAULT_REFERENCE_FILE)
    parser.add_argument("--output_file", type=Path, default=ROOT / "outputs" / "metrics.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified step-08 generation and single-file evaluation entry point.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate prediction files only.")
    add_generation_args(generate_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate one prediction file only.")
    evaluate_parser.add_argument("--prediction_file", type=Path, required=True)
    add_evaluation_args(evaluate_parser)

    run_parser = subparsers.add_parser("run", help="Generate then evaluate.")
    add_generation_args(run_parser)
    run_parser.add_argument("--reference_file", type=Path, default=DEFAULT_REFERENCE_FILE)
    run_parser.add_argument("--metrics_output_file", type=Path, default=None, help="Metrics output path for single-variant mode.")
    run_parser.add_argument("--base_metrics_output_file", type=Path, default=DEFAULT_BASE_METRICS_FILE)
    run_parser.add_argument("--lora_metrics_output_file", type=Path, default=DEFAULT_LORA_METRICS_FILE)

    return parser


def validate_common_generation_args(args) -> None:
    if args.log_every <= 0:
        raise ValueError("--log_every must be a positive integer.")
    if args.transformers_batch_size <= 0:
        raise ValueError("--transformers_batch_size must be a positive integer.")
    if args.mode in {"lora", "both"} and args.adapter_path is None:
        raise ValueError(f"--mode {args.mode} requires --adapter_path.")
    if args.mode == "both" and args.output_file is not None:
        raise ValueError("--output_file is only valid for --mode base or --mode lora.")
    if args.mode != "both" and args.output_file is None:
        raise ValueError("Single-variant generation requires --output_file.")
    if args.mode == "both" and args.base_output_file == args.lora_output_file:
        raise ValueError("In --mode both, --base_output_file and --lora_output_file must be different.")


def build_variant_jobs(args) -> list[dict]:
    validate_common_generation_args(args)
    if args.mode == "base":
        return [
            {
                "variant_name": "qwen_base",
                "adapter_path": None,
                "output_file": args.output_file,
                "metrics_output_file": getattr(args, "metrics_output_file", None),
            }
        ]
    if args.mode == "lora":
        return [
            {
                "variant_name": "qwen_lora",
                "adapter_path": args.adapter_path,
                "output_file": args.output_file,
                "metrics_output_file": getattr(args, "metrics_output_file", None),
            }
        ]
    return [
        {
            "variant_name": "qwen_base",
            "adapter_path": None,
            "output_file": args.base_output_file,
            "metrics_output_file": getattr(args, "base_metrics_output_file", None),
        },
        {
            "variant_name": "qwen_lora",
            "adapter_path": args.adapter_path,
            "output_file": args.lora_output_file,
            "metrics_output_file": getattr(args, "lora_metrics_output_file", None),
        },
    ]


def with_generation_defaults(args, job: dict) -> dict:
    return {
        **job,
        "model_name_or_path": args.model_name_or_path,
        "test_file": args.test_file,
        "main_system_prompt": args.main_system_prompt,
        "main_user_prompt": args.main_user_prompt,
        "overwrite": args.overwrite,
        "resume": args.resume,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "gpu_id": args.gpu_id,
        "log_every": args.log_every,
        "transformers_batch_size": args.transformers_batch_size,
    }


def select_gpu(gpu_id: int | None) -> int | None:
    selected_gpu = gpu_id if gpu_id is not None else choose_idle_gpu()
    if selected_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        print(json.dumps({"selected_gpu": selected_gpu, "device_mode": "single_gpu"}, ensure_ascii=False), flush=True)
    else:
        print(json.dumps({"selected_gpu": None, "device_mode": "cpu"}, ensure_ascii=False), flush=True)
    return selected_gpu


def run_generation_job(job: dict) -> dict:
    test_rows = read_jsonl(job["test_file"])
    validate_test_rows(test_rows, job["test_file"])
    prompt_bundle = resolve_main_prompt_bundle(argparse.Namespace(**job))

    output_file = job["output_file"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_output_file = output_file.with_suffix(output_file.suffix + ".tmp")
    completed_ids = load_completed_ids(temp_output_file) if job["resume"] else set()
    if job["resume"]:
        pending_rows = [row for row in test_rows if str(row["id"]) not in completed_ids]
    else:
        ensure_output_path(output_file, job["overwrite"])
        pending_rows = test_rows

    job = {**job, "total_rows": len(test_rows)}
    start_time = time.time()
    if job["resume"]:
        print(
            json.dumps(
                {
                    "event": "generation_resume",
                    "variant": job["variant_name"],
                    "completed_rows": len(completed_ids),
                    "pending_rows": len(pending_rows),
                    "temp_output_file": str(temp_output_file),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    writer_mode = "a" if job["resume"] and temp_output_file.exists() else "w"
    with temp_output_file.open(writer_mode, encoding="utf-8") as writer:
        done = stream_with_transformers(job, pending_rows, prompt_bundle, writer, start_time, already_done=len(completed_ids))

    temp_output_file.replace(output_file)
    elapsed_seconds = round(time.time() - start_time, 2)
    payload = {
        "variant": job["variant_name"],
        "output_file": str(output_file),
        "test_file": str(job["test_file"]),
        "rows": len(test_rows),
        "completed_rows": done,
        "backend": "transformers",
        "model_name_or_path": job["model_name_or_path"],
        "adapter_path": str(job["adapter_path"]) if job["adapter_path"] else None,
        "main_system_prompt": str(job["main_system_prompt"]) if job["main_system_prompt"] else None,
        "main_user_prompt": str(job["main_user_prompt"]) if job["main_user_prompt"] else None,
        "overwrite": job["overwrite"],
        "resume": job["resume"],
        "elapsed_seconds": elapsed_seconds,
    }
    print(json.dumps({"event": "generation_completed", **payload}, ensure_ascii=False, indent=2), flush=True)
    return payload


def evaluate_one_prediction_file(prediction_file: Path, reference_file: Path, output_file: Path, model_name: str) -> dict:
    reference_rows = read_jsonl(reference_file)
    validate_reference_rows(reference_rows, reference_file)
    payload = evaluate_prediction_file(
        prediction_path=prediction_file,
        reference_rows=reference_rows,
        reference_path=reference_file,
        model_name=model_name,
    )
    write_json(output_file, payload)
    return payload


def run_step08_variant(job: dict, reference_file: Path) -> dict:
    variant_start = time.time()
    generation_result = run_generation_job(job)
    evaluation_start = time.time()
    metrics_output_file = job["metrics_output_file"]
    if metrics_output_file is None:
        raise ValueError(f"Missing metrics output path for variant {job['variant_name']}.")
    evaluation_payload = evaluate_one_prediction_file(
        prediction_file=job["output_file"],
        reference_file=reference_file,
        output_file=metrics_output_file,
        model_name=job["variant_name"],
    )
    evaluation_elapsed = round(time.time() - evaluation_start, 2)
    variant_elapsed = round(time.time() - variant_start, 2)
    result = {
        "variant": job["variant_name"],
        "generation": generation_result,
        "evaluation": {
            "output_file": str(metrics_output_file),
            "elapsed_seconds": evaluation_elapsed,
            "payload": evaluation_payload,
        },
        "elapsed_seconds": variant_elapsed,
    }
    print(
        json.dumps(
            {
                "event": "variant_completed",
                "variant": job["variant_name"],
                "generation_elapsed_seconds": generation_result["elapsed_seconds"],
                "evaluation_elapsed_seconds": evaluation_elapsed,
                "variant_elapsed_seconds": variant_elapsed,
                "prediction_file": str(job["output_file"]),
                "metrics_file": str(metrics_output_file),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return result


def run_step08_pipeline(args) -> dict:
    reference_rows = read_jsonl(args.reference_file)
    validate_reference_rows(reference_rows, args.reference_file)
    jobs = [with_generation_defaults(args, job) for job in build_variant_jobs(args)]
    if args.mode == "both":
        metric_paths = [job["metrics_output_file"] for job in jobs]
        if len(set(metric_paths)) != len(metric_paths):
            raise ValueError("In --mode both, base and lora metrics output files must be different.")

    pipeline_start = time.time()
    results = []
    try:
        for job in jobs:
            results.append(run_step08_variant(job, args.reference_file))
    except Exception as exc:
        failure_payload = {
            "event": "pipeline_failed",
            "mode": args.mode,
            "reference_file": str(args.reference_file),
            "completed_variants": [result["variant"] for result in results],
            "artifacts": {
                result["variant"]: {
                    "prediction_file": result["generation"]["output_file"],
                    "metrics_file": result["evaluation"]["output_file"],
                }
                for result in results
            },
            "error": str(exc),
            "elapsed_seconds": round(time.time() - pipeline_start, 2),
        }
        print(json.dumps(failure_payload, ensure_ascii=False, indent=2), flush=True)
        raise

    payload = {
        "event": "pipeline_completed",
        "mode": args.mode,
        "reference_file": str(args.reference_file),
        "variants": results,
        "elapsed_seconds": round(time.time() - pipeline_start, 2),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        payload = evaluate_one_prediction_file(
            prediction_file=args.prediction_file,
            reference_file=args.reference_file,
            output_file=args.output_file,
            model_name="single_run",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "generate":
        jobs = [with_generation_defaults(args, job) for job in build_variant_jobs(args)]
        select_gpu(args.gpu_id)
        if args.mode == "both":
            if len({job["output_file"] for job in jobs}) != len(jobs):
                raise ValueError("In --mode both, base and lora output files must be different.")
        results = [run_generation_job(job) for job in jobs]
        print(json.dumps({"event": "generation_pipeline_completed", "mode": args.mode, "variants": results}, ensure_ascii=False, indent=2))
        return

    if args.mode != "both" and args.metrics_output_file is None:
        raise ValueError("Single-variant run requires --metrics_output_file.")
    if args.mode == "both" and args.metrics_output_file is not None:
        raise ValueError("--metrics_output_file is only valid for --mode base or --mode lora.")
    select_gpu(args.gpu_id)
    run_step08_pipeline(args)


if __name__ == "__main__":
    main()
