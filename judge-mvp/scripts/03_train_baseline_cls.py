#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABEL2ID = {"safe": 0, "unsafe": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_text(example: dict) -> str:
    return f"Question:\n{example['question']}\n\nResponse:\n{example['response']}"


def to_hf_dataset(rows: list[dict]) -> Dataset:
    records = []
    for row in rows:
        records.append(
            {
                "id": row["id"],
                "text": build_text(row),
                "label": LABEL2ID[row["label"]],
                "gold_label": row["label"],
                "question": row["question"],
                "response": row["response"],
            }
        )
    return Dataset.from_list(records)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline text classifier on question+response.")
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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file)
    test_rows = read_jsonl(args.test_file)

    train_ds = to_hf_dataset(train_rows)
    dev_ds = to_hf_dataset(dev_rows)
    test_ds = to_hf_dataset(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize(batch: dict) -> dict:
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
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
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
            probs = np.exp(logits) / np.exp(logits).sum()
            record = {
                **row,
                "pred_label": ID2LABEL[int(pred_id)],
                "scores": {"safe": float(probs[0]), "unsafe": float(probs[1])},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(eval_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
