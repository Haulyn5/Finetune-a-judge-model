#!/usr/bin/env python3
"""Train the main Qwen 4B-class structured judge with LoRA or QLoRA.

Pipeline step:
    07 / 08

Goal:
    Fine-tune the project's main model path so it can output structured safety
    judgments with ``label``, ``reason``, and ``evidence``.
    Unlike step 03, this is not a quick baseline; it is the primary training
    path for the MVP.

Inputs:
    - ``data/processed/sft_train.jsonl``
    - ``data/processed/sft_dev.jsonl``

Outputs:
    - LoRA adapter files under ``outputs/sft_lora/``
    - ``outputs/sft_lora/main_system.txt`` and ``outputs/sft_lora/main_user.txt``
      for later inference/eval reuse

Key assumptions:
    - Default model target is a Qwen 4B-class checkpoint.
    - The intended hardware envelope is roughly an A100 80GB.
    - LoRA keeps the script simple for learning; optional 4-bit loading provides
      a QLoRA-style memory reduction path when bitsandbytes is available.

Usage example:
    - CUDA_VISIBLE_DEVICES=2 uv run python scripts/07_train_sft_lora.py --train_file data/processed/sft_train_15000.jsonl --eval_file data/processed/sft_dev_15000.jsonl --output_dir outputs/sft_lora_15000_lr_5e-5_completion_only --tensorboard_run_name step07_sft_lora_15000 --num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 5e-5 --max_seq_length 2048

Tensorboard monitoring:
    - uv run tensorboard --logdir outputs/sft_lora_15000_lr_5e-5_completion_only/runs --port 43355 --host 0.0.0.0
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from _prompts import load_named_prompt_bundle
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-4B"


def emit_event(event: str, **payload) -> None:
    """Print a compact JSON event so long SFT runs are easy to monitor."""
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def read_jsonl(path: Path) -> list[dict]:
    """Read SFT JSONL rows from disk."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def format_example(row: dict, tokenizer: AutoTokenizer) -> dict:
    """Render one chat-style SFT example into the model's native chat template."""
    text = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return {"text": text}


def summarize_best_checkpoint(trainer) -> dict:
    """Return a stable summary of best-checkpoint selection state."""
    best_metric = trainer.state.best_metric
    best_checkpoint = trainer.state.best_model_checkpoint
    finite_eval_losses = [
        float(item["eval_loss"])
        for item in trainer.state.log_history
        if "eval_loss" in item and isinstance(item["eval_loss"], (int, float)) and math.isfinite(item["eval_loss"])
    ]
    has_finite_eval_loss = bool(finite_eval_losses)
    best_metric_is_finite = isinstance(best_metric, (int, float)) and math.isfinite(best_metric)
    if best_checkpoint and best_metric_is_finite:
        return {
            "best_model_checkpoint": best_checkpoint,
            "best_metric_name": "eval_loss",
            "best_metric_value": float(best_metric),
            "best_checkpoint_status": "selected",
            "has_finite_eval_loss": has_finite_eval_loss,
        }

    return {
        "best_model_checkpoint": None,
        "best_metric_name": "eval_loss",
        "best_metric_value": float(best_metric) if best_metric_is_finite else None,
        "best_checkpoint_status": "unavailable_non_finite_eval_loss",
        "has_finite_eval_loss": has_finite_eval_loss,
    }


def write_training_summary(path: Path, payload: dict) -> None:
    """Persist a compact training summary for later eval/debug usage."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def positive_int(value: str) -> int:
    """Argparse helper that only accepts positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the main LoRA/QLoRA SFT judge model for structured safety judgment. This is the project's primary Qwen path, not the step-03 baseline.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME, help=f"Base model ID or local path. Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--train_file", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "sft_train.jsonl")
    parser.add_argument("--eval_file", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "sft_dev.jsonl")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "sft_lora")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--tensorboard_run_name", default=None, help="Optional TensorBoard run name. Defaults to a timestamped name.")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=positive_int, default=32, help="LoRA rank r. lora_alpha is automatically set to 2 * r.")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.tensorboard_run_name or f"step07_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    lora_alpha = args.lora_rank * 2

    train_rows = read_jsonl(args.train_file)
    eval_rows = read_jsonl(args.eval_file)
    if not train_rows or not eval_rows:
        raise ValueError("Train and eval files must both be non-empty. Run step 06 first.")

    emit_event(
        "data_loaded",
        train_file=str(args.train_file),
        eval_file=str(args.eval_file),
        train_examples=len(train_rows),
        eval_examples=len(eval_rows),
        output_dir=str(args.output_dir),
        run_name=run_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    emit_event(
        "tokenizer_ready",
        model_name_or_path=args.model_name_or_path,
        pad_token=tokenizer.pad_token,
        eos_token=tokenizer.eos_token,
        padding_side=tokenizer.padding_side,
        compute_dtype=str(compute_dtype),
        load_in_4bit=args.load_in_4bit,
    )
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    emit_event("model_loaded", model_name_or_path=args.model_name_or_path, load_in_4bit=args.load_in_4bit)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
        emit_event("kbit_training_prepared", enabled=True)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, peft_config)
    emit_event(
        "lora_ready",
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        target_modules=sorted(peft_config.target_modules),
    )

    train_ds = Dataset.from_list([format_example(row, tokenizer) for row in train_rows])
    eval_ds = Dataset.from_list([format_example(row, tokenizer) for row in eval_rows])
    emit_event("datasets_ready", train_dataset_size=len(train_ds), eval_dataset_size=len(eval_ds))

    warmup_steps = max(1, len(train_rows) // (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    training_config_snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(args.output_dir),
        "run_name": run_name,
        "model_name_or_path": args.model_name_or_path,
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "tokenizer": {
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "padding_side": tokenizer.padding_side,
        },
        "precision": {
            "compute_dtype": str(compute_dtype),
            "load_in_4bit": args.load_in_4bit,
            "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        },
        "lora": {
            "r": args.lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        },
        "training": {
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_seq_length": args.max_seq_length,
            "warmup_steps": warmup_steps,
            "lr_scheduler_type": "cosine",
            "logging_steps": args.logging_steps,
            "save_total_limit": args.save_total_limit,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": ["tensorboard"],
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "remove_unused_columns": False,
            "dataset_text_field": "text",
            "packing": False,
            "completion_only_loss": True,
        },
    }
    training_config_path = args.output_dir / "training_config.json"
    write_training_summary(training_config_path, training_config_snapshot)
    emit_event("training_config_saved", training_config_path=str(training_config_path))

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        run_name=run_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        completion_only_loss=True,
    )

    emit_event(
        "training_args_ready",
        output_dir=str(args.output_dir),
        run_name=run_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    emit_event("training_started", output_dir=str(args.output_dir), run_name=run_name)
    trainer.train()
    best_checkpoint_summary = summarize_best_checkpoint(trainer)
    emit_event(
        "best_checkpoint_selected",
        output_dir=str(args.output_dir),
        run_name=run_name,
        **best_checkpoint_summary,
    )
    emit_event("training_finished", output_dir=str(args.output_dir), run_name=run_name)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    training_summary = {
        "output_dir": str(args.output_dir),
        "model_name_or_path": args.model_name_or_path,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "run_name": run_name,
        "global_step": trainer.state.global_step,
        **best_checkpoint_summary,
    }
    write_training_summary(args.output_dir / "training_summary.json", training_summary)
    emit_event("artifacts_saved", output_dir=str(args.output_dir), training_summary_path=str(args.output_dir / "training_summary.json"))

    prompt_bundle = load_named_prompt_bundle("main")
    with (args.output_dir / "main_system.txt").open("w", encoding="utf-8") as f:
        f.write(prompt_bundle.system + "\n")
    with (args.output_dir / "main_user.txt").open("w", encoding="utf-8") as f:
        f.write(prompt_bundle.user + "\n")
    emit_event("prompt_templates_saved", output_dir=str(args.output_dir))

    print(
        json.dumps(
            training_summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
