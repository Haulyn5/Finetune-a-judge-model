#!/usr/bin/env python3
"""Train a lightweight reference baseline for binary safety classification.

Pipeline step:
    03 / 08

Goal:
    Provide a fast, easy-to-understand comparison point for the dataset and
    metric pipeline. This script is intentionally *not* the main model path of
    the project. The real structured judge training path is step 04 -> 08 with
    a Qwen 4B-class model.

Inputs:
    - ``data/processed/train.jsonl``
    - ``data/processed/dev.jsonl``
    - ``data/processed/test.jsonl``

Outputs:
    - ``outputs/baseline_cls/metrics.json``
    - ``outputs/baseline_cls/test_predictions.jsonl``
    - Hugging Face trainer checkpoints under ``outputs/baseline_cls/``
    - TensorBoard event files under ``outputs/baseline_cls/runs/`` by default

Input schema:
    Each JSONL row must contain ``id``, ``question``, ``response``, ``label``.
    ``label`` must be one of ``safe`` or ``unsafe``.

Key assumptions:
    - Concatenating question + response is a simple baseline featureization.
    - ``macro_f1`` is a better model-selection metric than raw accuracy when
      class balance may shift.
    - We report extra unsafe metrics because missing unsafe content is usually
      more costly than an average-case error in safety settings.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
from _common import choose_idle_gpu
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABEL2ID = {"safe": 0, "unsafe": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
REQUIRED_FIELDS = ("id", "question", "response", "label")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into memory.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of parsed dictionaries.
    """
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def validate_rows(rows: list[dict], split_name: str, source_path: Path) -> None:
    """Validate baseline input rows and raise learner-friendly errors.

    Args:
        rows: Parsed JSONL records for one split.
        split_name: Human-readable split name such as ``train``.
        source_path: File path used in the error message.

    Raises:
        ValueError: If a row misses required fields or uses an invalid label.
    """
    if not rows:
        raise ValueError(f"{split_name} split is empty: {source_path}")

    for line_number, row in enumerate(rows, start=1):
        missing_fields = [field for field in REQUIRED_FIELDS if field not in row]
        if missing_fields:
            raise ValueError(
                f"Invalid row in {source_path} line {line_number}: missing fields {missing_fields}. "
                f"Expected keys: {list(REQUIRED_FIELDS)}."
            )

        invalid_empty_fields = [field for field in ("id", "question", "response") if not str(row[field]).strip()]
        if invalid_empty_fields:
            raise ValueError(
                f"Invalid row in {source_path} line {line_number}: fields {invalid_empty_fields} must be non-empty strings."
            )

        label = str(row["label"]).strip().lower()
        if label not in LABEL2ID:
            raise ValueError(
                f"Invalid label in {source_path} line {line_number}: got {row['label']!r}, "
                f"expected one of {sorted(LABEL2ID.keys())}."
            )


def summarize_split(rows: list[dict], split_name: str) -> dict:
    """Return split size and label distribution for transparent training logs."""
    label_counter = Counter(str(row["label"]).strip().lower() for row in rows)
    return {
        "split": split_name,
        "rows": len(rows),
        "label_distribution": dict(sorted(label_counter.items())),
    }


def build_text(example: dict) -> str:
    """Build the classifier input text from one normalized example.

    We concatenate question and response because the safety label depends on the
    interaction between the user request and the assistant answer, not on either
    field in isolation.
    """
    return f"Question:\n{example['question']}\n\nResponse:\n{example['response']}"


def to_hf_dataset(rows: list[dict]) -> Dataset:
    """Convert normalized JSONL rows into a Hugging Face Dataset.

    Args:
        rows: Validated project-format records.

    Returns:
        A dataset containing model text, numeric labels, and raw fields for
        later prediction export.
    """
    records = []
    for row in rows:
        normalized_label = str(row["label"]).strip().lower()
        records.append(
            {
                "id": str(row["id"]),
                "text": build_text(row),
                "label": LABEL2ID[normalized_label],
                "gold_label": normalized_label,
                "question": str(row["question"]),
                "response": str(row["response"]),
            }
        )
    return Dataset.from_list(records)


def compute_metrics(eval_pred):
    """Compute classification metrics used for model selection and analysis."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)

    # Macro F1 gives equal weight to safe and unsafe classes, so it is usually a
    # better summary metric than accuracy for safety data with possible imbalance.
    macro_f1 = f1_score(labels, preds, average="macro")

    # We separately track the unsafe class because missing unsafe examples is a
    # high-cost failure mode for a safety judge.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[LABEL2ID["unsafe"]],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "unsafe_precision": float(precision[0]),
        "unsafe_recall": float(recall[0]),
        "unsafe_f1": float(f1[0]),
    }


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Apply a numerically stable softmax to one logit vector."""
    shifted = logits - np.max(logits)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def configure_random_seed(seed: int) -> None:
    """Set Python, NumPy, and Transformers random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a lightweight reference baseline on question + response. This is a comparison baseline, not the project's main Qwen judge path."
    )
    parser.add_argument("--train_file", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "train.jsonl")
    parser.add_argument("--dev_file", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "dev.jsonl")
    parser.add_argument("--test_file", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "test.jsonl")
    parser.add_argument("--model_name_or_path", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "baseline_cls")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--logging_dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "baseline_cls" / "runs")
    parser.add_argument("--report_to", default="tensorboard", choices=["tensorboard", "none"])
    parser.add_argument("--gpu_id", type=int, default=None, help="Use a specific single GPU. If omitted, the script picks the least-used GPU by current memory usage.")
    args = parser.parse_args()

    selected_gpu = args.gpu_id if args.gpu_id is not None else choose_idle_gpu()
    if selected_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        print(json.dumps({"selected_gpu": selected_gpu, "device_mode": "single_gpu"}, ensure_ascii=False))
    else:
        print(json.dumps({"selected_gpu": None, "device_mode": "cpu"}, ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_random_seed(args.seed)

    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file)
    test_rows = read_jsonl(args.test_file)

    validate_rows(train_rows, "train", args.train_file)
    validate_rows(dev_rows, "dev", args.dev_file)
    validate_rows(test_rows, "test", args.test_file)

    data_summary = {
        "seed": args.seed,
        "train": summarize_split(train_rows, "train"),
        "dev": summarize_split(dev_rows, "dev"),
        "test": summarize_split(test_rows, "test"),
    }
    print(json.dumps(data_summary, ensure_ascii=False, indent=2))

    train_ds = to_hf_dataset(train_rows)
    dev_ds = to_hf_dataset(dev_rows)
    test_ds = to_hf_dataset(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize(batch: dict) -> dict:
        """Tokenize baseline text examples with truncation."""
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    dev_ds = dev_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        logging_dir=str(args.logging_dir),
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        report_to=[args.report_to] if args.report_to != "none" else [],
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(test_ds)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

    predictions = trainer.predict(test_ds)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    with (args.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, pred_id, logits in zip(test_rows, pred_ids, predictions.predictions):
            probs = stable_softmax(np.asarray(logits, dtype=np.float64))
            record = {
                **row,
                "pred_label": ID2LABEL[int(pred_id)],
                "scores": {"safe": float(probs[0]), "unsafe": float(probs[1])},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(eval_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
