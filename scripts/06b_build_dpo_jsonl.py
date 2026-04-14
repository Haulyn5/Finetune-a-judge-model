#!/usr/bin/env python3
"""Build DPO preference data from sampled SFT-model candidates.

Pipeline step:
    06b / 09

Goal:
    Use the current best SFT LoRA judge as a candidate generator, then score and
    rank multiple sampled structured judgments per source row to produce real
    chosen/rejected preference pairs for DPO.

Outputs:
    - ``data/interim/dpo_candidates.jsonl``
    - ``data/interim/dpo_pairs_debug.jsonl``
    - ``data/processed/dpo_train.jsonl``
    - ``data/processed/dpo_dev.jsonl``
    - ``outputs/dpo_build_summary.json``

Command:

uv run python scripts/06b_build_dpo_jsonl.py \
  --input_path data/processed/pseudo_filtered_15000.jsonl \
  --adapter_path outputs/sft_lora \
  --num_candidates 4 \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_new_tokens 4096
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from _common import choose_idle_gpu
from _prompts import load_named_prompt_bundle, load_prompt_bundle, render_user_prompt

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable

VALID_LABELS = {"safe", "unsafe"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-4B"
DEFAULT_INPUT_PATH = ROOT / "data" / "processed" / "pseudo_filtered_15000.jsonl"
DEFAULT_ADAPTER_PATH = ROOT / "outputs" / "sft_lora"
DEFAULT_TRAIN_OUTPUT = ROOT / "data" / "processed" / "dpo_train.jsonl"
DEFAULT_DEV_OUTPUT = ROOT / "data" / "processed" / "dpo_dev.jsonl"
DEFAULT_CANDIDATE_CACHE = ROOT / "data" / "interim" / "dpo_candidates.jsonl"
DEFAULT_PAIR_DEBUG = ROOT / "data" / "interim" / "dpo_pairs_debug.jsonl"
DEFAULT_SUMMARY_PATH = ROOT / "outputs" / "dpo_build_summary.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()


def load_existing_candidate_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if row_id:
                ids.add(str(row_id))
    return ids


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    """Yield fixed-size batches so generation can use larger GPU-friendly workloads."""
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


def parse_prediction(text: str) -> tuple[str, dict | None, str, list[str]]:
    prediction_json = extract_json_block(text)
    if prediction_json is None:
        return "safe", None, "", []

    pred_label = normalize_label(prediction_json.get("label", ""))
    if pred_label not in VALID_LABELS:
        pred_label = "safe"

    reason = str(prediction_json.get("reason", "") or "").strip()
    evidence = prediction_json.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if str(evidence).strip() else []
    evidence = [str(item).strip() for item in evidence if str(item).strip()]
    normalized = {
        "label": pred_label,
        "reason": reason,
        "evidence": evidence,
    }
    return pred_label, normalized, reason, evidence


def evidence_matches(text: str, evidence_list: list[str]) -> bool:
    lowered = text.lower()
    return bool(evidence_list) and all(ev.strip() and ev.lower() in lowered for ev in evidence_list)


def split_source_rows(rows: list[dict], dev_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not rows:
        raise ValueError("No rows found in input file.")
    if not (0 < dev_size < 1):
        raise ValueError("dev_size must be between 0 and 1.")

    label_counts = Counter(normalize_label(row["label"]) for row in rows)
    if any(label not in VALID_LABELS for label in label_counts):
        raise ValueError(f"Input labels must be within {sorted(VALID_LABELS)}.")
    if any(count < 2 for count in label_counts.values()):
        raise ValueError("Each label must have at least 2 samples to support stratified train/dev split.")

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[normalize_label(row["label"])].append(row)

    rng = random.Random(seed)
    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for label_rows in grouped_rows.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        num_dev = max(1, int(round(len(shuffled) * dev_size)))
        if num_dev >= len(shuffled):
            num_dev = len(shuffled) - 1
        dev_rows.extend(shuffled[:num_dev])
        train_rows.extend(shuffled[num_dev:])

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    return train_rows, dev_rows


def build_prompt_bundle(args) -> object:
    if args.main_system_prompt is not None or args.main_user_prompt is not None:
        if args.main_system_prompt is None or args.main_user_prompt is None:
            raise ValueError("--main_system_prompt and --main_user_prompt must be provided together.")
        return load_prompt_bundle(args.main_system_prompt, args.main_user_prompt)

    if args.adapter_path is not None:
        adapter_system = args.adapter_path / "main_system.txt"
        adapter_user = args.adapter_path / "main_user.txt"
        if adapter_system.exists() and adapter_user.exists():
            return load_prompt_bundle(adapter_system, adapter_user)

    return load_named_prompt_bundle("main")


def build_prompt_fields(row: dict, prompt_bundle) -> tuple[str, str, list[dict]]:
    instruction = prompt_bundle.system
    model_input = render_user_prompt(
        prompt_bundle.user,
        question=row["question"],
        response=row["response"],
    )
    prompt_messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": model_input},
    ]
    return instruction, model_input, prompt_messages


def resolve_effective_adapter_path(adapter_path: Path) -> Path:
    summary_path = adapter_path / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best_checkpoint = summary.get("best_model_checkpoint")
        if best_checkpoint:
            checkpoint_path = Path(best_checkpoint)
            if checkpoint_path.exists():
                return checkpoint_path
    return adapter_path


def score_candidate(candidate: dict, source_context: dict, min_reason_chars: int, max_reason_chars: int) -> tuple[float, dict, list[str]]:
    # Candidate rows keep source text fields for traceability, but scoring still
    # compares each sampled judgment against the original source text/gold label.
    source_text = source_context["source_text"]
    gold_label = source_context["gold_label"]
    pred_label = normalize_label(candidate.get("pred_label", ""))
    reason = str(candidate.get("reason", "") or "").strip()
    evidence = candidate.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if str(evidence).strip() else []
    evidence = [str(item).strip() for item in evidence if str(item).strip()]

    strict_json_ok = bool(candidate.get("strict_json_ok"))
    fallback_json_ok = bool(candidate.get("fallback_json_ok"))
    label_correct = 1.0 if pred_label == gold_label else 0.0
    required_fields_complete = 1.0 if pred_label in VALID_LABELS and reason and evidence else 0.0
    evidence_support = 1.0 if evidence_matches(source_text, evidence) else 0.0
    reason_length_ok = 1.0 if min_reason_chars <= len(reason) <= max_reason_chars else 0.0

    breakdown = {
        "label_correct": 5.0 * label_correct,
        "json_valid": 1.0 if strict_json_ok else (0.5 if fallback_json_ok else 0.0),
        "required_fields_complete": 0.5 * required_fields_complete,
        "evidence_support": 1.5 * evidence_support,
        "reason_length_ok": 0.5 * reason_length_ok,
    }

    total = round(sum(breakdown.values()), 4)
    drop_reasons = []
    if pred_label not in VALID_LABELS:
        drop_reasons.append("invalid_label")
    if not fallback_json_ok:
        drop_reasons.append("unparseable_json")
    if not reason:
        drop_reasons.append("missing_reason")
    if not evidence:
        drop_reasons.append("missing_evidence")
    elif not evidence_support:
        drop_reasons.append("evidence_not_grounded")
    if not reason_length_ok:
        drop_reasons.append("reason_length_out_of_range")

    return total, breakdown, drop_reasons


def choose_pair_type(chosen: dict, rejected: dict) -> str:
    if chosen.get("pred_label") != rejected.get("pred_label"):
        return "label_difference"
    if chosen.get("strict_json_ok") and not rejected.get("strict_json_ok"):
        return "json_stability"
    chosen_evidence = chosen.get("evidence") or []
    rejected_evidence = rejected.get("evidence") or []
    if len(chosen_evidence) != len(rejected_evidence):
        return "evidence_strength"
    return "mixed_quality"


def generate_candidates(
    rows: list[dict],
    split_name: str,
    prompt_bundle,
    args,
    adapter_for_generation: Path,
    existing_candidate_ids: set[str] | None = None,
) -> list[dict]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.gpu_id is None:
        auto_gpu = choose_idle_gpu()
        if auto_gpu is not None:
            args.gpu_id = auto_gpu
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_for_generation)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        # Ask transformers to sample multiple continuations per prompt in a single call
        # instead of launching one generate() call per candidate.
        "num_return_sequences": args.num_candidates,
    }

    source_iterator = rows[: args.max_rows] if args.max_rows is not None else rows
    existing_candidate_ids = existing_candidate_ids or set()
    prompt_rows = []
    skipped_existing = 0
    for row in source_iterator:
        candidate_ids = {f"{row['id']}::{split_name}::cand_{candidate_index}" for candidate_index in range(args.num_candidates)}
        if candidate_ids.issubset(existing_candidate_ids):
            skipped_existing += 1
            continue
        instruction, model_input, prompt_messages = build_prompt_fields(row, prompt_bundle)
        rendered_prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_rows.append(
            {
                "row": row,
                "instruction": instruction,
                "model_input": model_input,
                "prompt_messages": prompt_messages,
                "rendered_prompt": rendered_prompt,
                # Group similar prompt lengths together so batch padding waste stays smaller.
                "prompt_length_estimate": len(rendered_prompt),
            }
        )

    prompt_rows.sort(key=lambda item: item["prompt_length_estimate"])

    candidate_rows: list[dict] = []
    total_batches = (len(prompt_rows) + args.transformers_batch_size - 1) // args.transformers_batch_size
    batch_iterator = batched(prompt_rows, args.transformers_batch_size)
    for batch in tqdm(
        batch_iterator,
        total=total_batches,
        desc=f"06b {split_name} candidate generation",
        unit="batch",
        dynamic_ncols=True,
    ):
        prompts = [item["rendered_prompt"] for item in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)

        batch_candidate_rows = []
        for batch_row_index, batch_item in enumerate(batch):
            row = batch_item["row"]
            instruction = batch_item["instruction"]
            model_input = batch_item["model_input"]
            prompt_messages = batch_item["prompt_messages"]
            prompt_length = int(prompt_lengths[batch_row_index])
            sequence_offset = batch_row_index * args.num_candidates

            for candidate_index in range(args.num_candidates):
                candidate_id = f"{row['id']}::{split_name}::cand_{candidate_index}"
                if candidate_id in existing_candidate_ids:
                    continue
                generated_tokens = generated[sequence_offset + candidate_index][prompt_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                pred_label, prediction_json, reason, evidence = parse_prediction(generated_text)
                strict_json_ok = safe_json_loads(generated_text) is not None
                fallback_json_ok = prediction_json is not None
                candidate_row = {
                    "id": candidate_id,
                    "source_id": row["id"],
                    "split": split_name,
                    "question": row["question"],
                    "response": row["response"],
                    "gold_label": normalize_label(row["label"]),
                    "instruction": instruction,
                    "input": model_input,
                    "prompt_messages": prompt_messages,
                    "candidate_index": candidate_index,
                    "prediction": generated_text,
                    "prediction_json": prediction_json,
                    "pred_label": pred_label,
                    "reason": reason,
                    "evidence": evidence,
                    "strict_json_ok": strict_json_ok,
                    "fallback_json_ok": fallback_json_ok,
                    "adapter_path": str(adapter_for_generation),
                    "model_name_or_path": str(args.model_name_or_path),
                    "generation_config": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "transformers_batch_size": args.transformers_batch_size,
                    },
                }
                candidate_rows.append(candidate_row)
                batch_candidate_rows.append(candidate_row)
                existing_candidate_ids.add(candidate_id)

        append_jsonl(args.candidate_cache_path, batch_candidate_rows)
    return candidate_rows


def build_pairs_for_split(candidate_rows: list[dict], split_name: str, args) -> tuple[list[dict], list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        grouped[row["source_id"]].append(row)

    pairs = []
    debug_rows = []
    skip_reasons = Counter()

    for source_id, rows in grouped.items():
        first_row = rows[0]
        source_context = {
            "source_text": f"{first_row['question']}\n{first_row['response']}",
            "gold_label": normalize_label(first_row["gold_label"]),
        }
        scored_rows = []
        for row in rows:
            # Keep source context explicit here so the scorer does not rely on the
            # same dict playing both candidate and source roles.
            score, breakdown, drop_reasons = score_candidate(row, source_context, args.min_reason_chars, args.max_reason_chars)
            scored_rows.append(
                {
                    **row,
                    "score": score,
                    "score_breakdown": breakdown,
                    "drop_reasons": drop_reasons,
                    "is_usable": not drop_reasons,
                }
            )

        usable_rows = [row for row in scored_rows if row["is_usable"]]
        # Tie-break with candidate_index so pair selection stays deterministic even
        # when multiple sampled candidates land on the same heuristic score.
        usable_rows.sort(key=lambda item: (-item["score"], item["candidate_index"]))
        scored_rows_sorted = sorted(scored_rows, key=lambda item: (-item["score"], item["candidate_index"]))

        debug_row = {
            "source_id": source_id,
            "split": split_name,
            "num_candidates": len(scored_rows),
            "num_usable_candidates": len(usable_rows),
            "candidates": [
                {
                    "candidate_index": row["candidate_index"],
                    "pred_label": row["pred_label"],
                    "score": row["score"],
                    "strict_json_ok": row["strict_json_ok"],
                    "fallback_json_ok": row["fallback_json_ok"],
                    "drop_reasons": row["drop_reasons"],
                }
                for row in scored_rows_sorted
            ],
        }

        if len(usable_rows) < 2:
            skip_reasons["insufficient_usable_candidates"] += 1
            debug_row["skip_reason"] = "insufficient_usable_candidates"
            debug_rows.append(debug_row)
            continue

        chosen = usable_rows[0]
        rejected = sorted(usable_rows, key=lambda item: (item["score"], item["candidate_index"]))[0]
        if chosen["prediction"].strip() == rejected["prediction"].strip():
            skip_reasons["identical_chosen_rejected"] += 1
            debug_row["skip_reason"] = "identical_chosen_rejected"
            debug_rows.append(debug_row)
            continue

        margin = round(chosen["score"] - rejected["score"], 4)
        if margin < args.min_score_gap:
            skip_reasons["score_gap_too_small"] += 1
            debug_row["skip_reason"] = "score_gap_too_small"
            debug_row["score_gap"] = margin
            debug_rows.append(debug_row)
            continue

        pair_type = choose_pair_type(chosen, rejected)
        pair_row = {
            "id": f"{source_id}::{split_name}",
            "source_id": source_id,
            "split": split_name,
            "task": "structured_safety_judgment",
            "question": chosen["question"],
            "response": chosen["response"],
            "instruction": chosen["instruction"],
            "input": chosen["input"],
            "prompt_messages": chosen["prompt_messages"],
            "chosen": chosen["prediction"],
            "rejected": rejected["prediction"],
            "chosen_structured": chosen["prediction_json"],
            "rejected_structured": rejected["prediction_json"],
            "gold_label": chosen["gold_label"],
            "pair_source": "sampled_from_sft_lora",
            "pair_type": pair_type,
            "scores": {
                "chosen_total": chosen["score"],
                "rejected_total": rejected["score"],
                "margin": margin,
                "chosen_breakdown": chosen["score_breakdown"],
                "rejected_breakdown": rejected["score_breakdown"],
            },
            "metadata": {
                "adapter_path": chosen["adapter_path"],
                "model_name_or_path": chosen["model_name_or_path"],
                "generation_config": chosen["generation_config"],
                "chosen_candidate_index": chosen["candidate_index"],
                "rejected_candidate_index": rejected["candidate_index"],
                "chosen_strict_json_ok": chosen["strict_json_ok"],
                "rejected_strict_json_ok": rejected["strict_json_ok"],
                "chosen_fallback_json_ok": chosen["fallback_json_ok"],
                "rejected_fallback_json_ok": rejected["fallback_json_ok"],
                "rejected_drop_reasons": rejected["drop_reasons"],
            },
        }
        debug_row["chosen_candidate_index"] = chosen["candidate_index"]
        debug_row["rejected_candidate_index"] = rejected["candidate_index"]
        debug_row["score_gap"] = margin
        debug_rows.append(debug_row)
        pairs.append(pair_row)

    return pairs, debug_rows, dict(skip_reasons)


def summarize_pairs(rows: list[dict]) -> dict:
    return {
        "num_pairs": len(rows),
        "pair_type_counts": dict(sorted(Counter(row["pair_type"] for row in rows).items())),
        "gold_label_counts": dict(sorted(Counter(row["gold_label"] for row in rows).items())),
        "avg_margin": round(sum(row["scores"]["margin"] for row in rows) / len(rows), 4) if rows else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DPO preference data from sampled SFT-model candidates.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--adapter_path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train_output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--dev_output", type=Path, default=DEFAULT_DEV_OUTPUT)
    parser.add_argument("--candidate_cache_path", type=Path, default=DEFAULT_CANDIDATE_CACHE)
    parser.add_argument("--pair_debug_path", type=Path, default=DEFAULT_PAIR_DEBUG)
    parser.add_argument("--summary_path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--main_system_prompt", type=Path, default=None)
    parser.add_argument("--main_user_prompt", type=Path, default=None)
    parser.add_argument("--dev_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--transformers_batch_size", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--min_reason_chars", type=int, default=20)
    parser.add_argument("--max_reason_chars", type=int, default=400)
    parser.add_argument("--min_score_gap", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    prompt_bundle = build_prompt_bundle(args)
    source_rows = read_jsonl(args.input_path)
    train_source_rows, dev_source_rows = split_source_rows(source_rows, args.dev_size, args.seed)
    adapter_for_generation = resolve_effective_adapter_path(args.adapter_path)

    if args.resume:
        existing_candidate_ids = load_existing_candidate_ids(args.candidate_cache_path)
    else:
        args.candidate_cache_path.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_cache_path.write_text("", encoding="utf-8")
        existing_candidate_ids = set()

    train_candidates = generate_candidates(
        train_source_rows,
        "train",
        prompt_bundle,
        args,
        adapter_for_generation,
        existing_candidate_ids,
    )
    dev_candidates = generate_candidates(
        dev_source_rows,
        "dev",
        prompt_bundle,
        args,
        adapter_for_generation,
        existing_candidate_ids,
    )
    all_candidates = read_jsonl(args.candidate_cache_path)
    train_candidates = [row for row in all_candidates if row.get("split") == "train"]
    dev_candidates = [row for row in all_candidates if row.get("split") == "dev"]

    train_pairs, train_debug_rows, train_skip_reasons = build_pairs_for_split(train_candidates, "train", args)
    dev_pairs, dev_debug_rows, dev_skip_reasons = build_pairs_for_split(dev_candidates, "dev", args)
    write_jsonl(args.train_output, train_pairs)
    write_jsonl(args.dev_output, dev_pairs)
    write_jsonl(args.pair_debug_path, train_debug_rows + dev_debug_rows)

    summary = {
        "input_rows": len(source_rows),
        "train_source_rows": len(train_source_rows),
        "dev_source_rows": len(dev_source_rows),
        "adapter_path_requested": str(args.adapter_path),
        "adapter_path_used": str(adapter_for_generation),
        "model_name_or_path": str(args.model_name_or_path),
        "split_config": {
            "dev_size": args.dev_size,
            "seed": args.seed,
        },
        "generation_config": {
            "num_candidates": args.num_candidates,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "train_candidates": len(train_candidates),
        "dev_candidates": len(dev_candidates),
        "train_skip_reasons": train_skip_reasons,
        "dev_skip_reasons": dev_skip_reasons,
        "train": summarize_pairs(train_pairs),
        "dev": summarize_pairs(dev_pairs),
        "candidate_cache_path": str(args.candidate_cache_path),
        "pair_debug_path": str(args.pair_debug_path),
        "train_output": str(args.train_output),
        "dev_output": str(args.dev_output),
    }
    write_json(args.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
