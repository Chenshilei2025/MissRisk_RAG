from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from scripts.lib.missrisk_common import classification_metrics, pos_weight_from_labels, set_seed, sigmoid_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model A answer-bearing cross-encoder.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--encoder", default="BAAI/bge-reranker-base")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def input_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[QUESTION] {row.get('question') or ''}",
            f"[OBLIGATION] {row.get('obligation_text') or ''}",
            f"[MODALITY] {row.get('modality') or ''}",
            f"[UNIT] {row.get('unit_content') or row.get('unit_aux_text') or ''}",
        ]
    )


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_len: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            input_text(row),
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": float(row.get("label_answer_bearing", 0) or 0),
            "row_index": idx,
        }


def make_collate(tokenizer: Any):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for idx, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[idx, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[idx, :length] = torch.tensor(item["attention_mask"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float),
            "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        }

    return collate


def forward_logits(model: Any, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    logits = out.logits
    if logits.shape[-1] == 1:
        return logits.squeeze(-1)
    return logits[:, 1] - logits[:, 0]


def ranking_metrics(rows: list[dict[str, Any]], probs: np.ndarray) -> dict[str, float]:
    by_q: dict[str, list[tuple[int, float]]] = {}
    for row, prob in zip(rows, probs, strict=True):
        by_q.setdefault(str(row.get("question_id")), []).append(
            (int(row.get("label_answer_bearing", 0) or 0), float(prob))
        )
    recalls = {1: [], 3: [], 5: [], 10: []}
    reciprocal_ranks = []
    pool_sizes = []
    for items in by_q.values():
        if not any(label for label, _ in items):
            continue
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        pool_sizes.append(len(ranked))
        first_pos = None
        for idx, (label, _) in enumerate(ranked, start=1):
            if label:
                first_pos = idx
                break
        if first_pos is not None:
            reciprocal_ranks.append(1.0 / first_pos)
        for k in recalls:
            recalls[k].append(float(any(label for label, _ in ranked[:k])))
    return {
        "question_count": len(reciprocal_ranks),
        "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        **{f"recall_at_{k}": float(np.mean(values)) if values else 0.0 for k, values in recalls.items()},
    }


def evaluate(
    model: Any,
    loader: DataLoader,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    logits_all = []
    labels_all = []
    order = []
    with torch.no_grad():
        for batch in loader:
            logits = forward_logits(model, batch, device)
            logits_all.extend(logits.detach().float().cpu().tolist())
            labels_all.extend(batch["labels"].detach().float().cpu().tolist())
            order.extend(batch["row_index"].detach().cpu().tolist())
    logits = np.asarray(logits_all, dtype=float)
    probs = sigmoid_np(logits)
    ordered_rows = [rows[int(idx)] for idx in order]
    metrics = classification_metrics(labels_all, probs)
    metrics.update(ranking_metrics(ordered_rows, probs))
    predictions = [
        {
            "example_id": row.get("example_id"),
            "question_id": row.get("question_id"),
            "unit_id": row.get("unit_id"),
            "source_id": row.get("source_id"),
            "modality": row.get("modality"),
            "label_answer_bearing": int(label),
            "logit_answer_bearing": float(logit),
            "p_answer_bearing": float(prob),
        }
        for row, label, logit, prob in zip(ordered_rows, labels_all, logits, probs, strict=True)
    ]
    return metrics, predictions


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    train_rows = read_jsonl(args.train, args.max_train_samples)
    dev_rows = read_jsonl(args.dev, args.max_dev_samples)
    train_ds = PairDataset(train_rows, tokenizer, args.max_len)
    dev_ds = PairDataset(dev_rows, tokenizer, args.max_len)
    collate = make_collate(tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    dev_dl = DataLoader(dev_ds, batch_size=args.bs, shuffle=False, collate_fn=collate)
    model = AutoModelForSequenceClassification.from_pretrained(args.encoder, num_labels=1).to(device)
    labels = [float(row.get("label_answer_bearing", 0) or 0) for row in train_rows]
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_from_labels(labels)], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_dl) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * args.warmup_ratio),
        total_steps,
    )
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl, start=1):
            logits = forward_logits(model, batch, device)
            loss = loss_fn(logits, batch["labels"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach().cpu())
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": running / step}), flush=True)
        dev_metrics, dev_predictions = evaluate(model, dev_dl, dev_rows, device)
        report = {"epoch": epoch, "train_loss": running / max(1, len(train_dl)), "dev": dev_metrics}
        history.append(report)
        print("[dev] " + json.dumps(report, sort_keys=True), flush=True)
        score = dev_metrics.get("auroc", 0.0)
        if not np.isnan(score) and score > best:
            best = float(score)
            torch.save(model.state_dict(), args.out / "model_A.pt")
            save_jsonl(args.out / "dev_predictions.best.jsonl", dev_predictions)
    tokenizer.save_pretrained(args.out)
    metrics_payload = {"history": history, "best_dev_auroc": best, "args": {k: str(v) for k, v in vars(args).items()}}
    if args.test:
        model.load_state_dict(torch.load(args.out / "model_A.pt", map_location=device))
        test_rows = read_jsonl(args.test)
        test_ds = PairDataset(test_rows, tokenizer, args.max_len)
        test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, collate_fn=collate)
        test_metrics, test_predictions = evaluate(model, test_dl, test_rows, device)
        metrics_payload["test"] = test_metrics
        save_jsonl(args.out / "test_predictions.jsonl", test_predictions)
    save_json(args.out / "metrics.json", metrics_payload)


if __name__ == "__main__":
    main()
