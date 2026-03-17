#!/usr/bin/env python3
"""Train the main Qwen 4B-class structured judge with LoRA or QLoRA.

Pipeline step:
    07 / 08

Goal:
    Fine-tune the project's main model path so it can output structured safety
    judgments with ``label``, ``reason``, ``evidence``, and ``confidence``.
    Unlike step 03, this is not a quick baseline; it is the primary training
    path for the MVP.

Inputs:
    - ``data/processed/sft_train.jsonl``
    - ``data/processed/sft_dev.jsonl``

Outputs:
    - LoRA adapter files under ``outputs/sft_lora/``
    - ``outputs/sft_lora/prompt_template.txt`` for later inference/eval reuse

Key assumptions:
    - Default model target is a Qwen 4B-class checkpoint.
    - The intended hardware envelope is roughly an A100 80GB.
    - LoRA keeps the script simple for learning; optional 4-bit loading provides
      a QLoRA-style memory reduction path when bitsandbytes is available.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


DEFAULT_PROMPT_TEMPLATE = "System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n{assistant}"


def read_jsonl(path: Path) -> list[dict]:
    """Read SFT JSONL rows from disk."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def format_example(row: dict, eos_token: str) -> dict:
    """Render one chat-style SFT example into a single training string.

    Args:
        row: SFT dataset row containing a 3-message conversation.
        eos_token: Token appended so the model learns a clean response boundary.

    Returns:
        A dictionary with the ``text`` field expected by ``SFTTrainer``.
    """
    messages = row["messages"]
    text = DEFAULT_PROMPT_TEMPLATE.format(
        system=messages[0]["content"],
        user=messages[1]["content"],
        assistant=messages[2]["content"],
    ) + eos_token
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the main LoRA/QLoRA SFT judge model for structured safety judgment. This is the project's primary Qwen path, not the step-03 baseline.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--train_file", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "sft_train.jsonl")
    parser.add_argument("--eval_file", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "sft_dev.jsonl")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "sft_lora")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.train_file)
    eval_rows = read_jsonl(args.eval_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if (torch.cuda.is_available() or args.load_in_4bit) else None,
    )
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    # These target modules correspond to the main projection layers inside the
    # transformer blocks, which is a common LoRA choice for adapting Qwen-class
    # models without updating every parameter.
    model = get_peft_model(model, peft_config)

    train_ds = Dataset.from_list([format_example(row, tokenizer.eos_token) for row in train_rows])
    eval_ds = Dataset.from_list([format_example(row, tokenizer.eos_token) for row in eval_rows])

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
        # On A100 80GB, bf16 is typically the preferred mixed-precision mode.
        # fp16 remains as a fallback for environments that do not expose bf16.
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    # The defaults here favor a readable, conservative starting point for a
    # Qwen 4B-class model: small per-device batch size, gradient accumulation,
    # and a 1024-token context that is usually enough for question/response +
    # structured answer training on an A100 80GB.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    prompt_template = (
        "You are a content safety judge. Return JSON with keys label, reason, evidence, confidence.\n\n"
        "Question:\n{question}\n\nResponse:\n{response}\n"
    )
    with (args.output_dir / "prompt_template.txt").open("w", encoding="utf-8") as f:
        f.write(prompt_template)

    print(json.dumps({"output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
