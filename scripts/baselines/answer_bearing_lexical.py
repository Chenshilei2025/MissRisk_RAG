from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.lib.common import ensure_dir, iter_jsonl, normalize_space, write_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight lexical Model A baseline for answer-bearing unit prediction."
    )
    parser.add_argument("--train-path", type=Path, default=Path("data_missrisk/processed/mmdocrag_model_a/train_pairs.jsonl"))
    parser.add_argument("--dev-path", type=Path, default=Path("data_missrisk/processed/mmdocrag_model_a/dev_pairs.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/answer_bearing_lexical"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--max-features", type=int, default=20000)
    return parser.parse_args()


def tokens(row: dict) -> list[str]:
    text = "\n".join(
        normalize_space(row.get(key))
        for key in ("question", "obligation_text", "unit_content", "unit_aux_text", "modality")
    )
    return TOKEN_RE.findall(text.lower())


def build_vocab(rows: list[dict], max_features: int) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts.update(set(tokens(row)))
    return {token: idx for idx, (token, _) in enumerate(counts.most_common(max_features))}


def vector(row: dict, vocab: dict[str, int]) -> dict[int, float]:
    counts: dict[int, float] = defaultdict(float)
    for token in tokens(row):
        idx = vocab.get(token)
        if idx is not None:
            counts[idx] += 1.0
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return {idx: value / norm for idx, value in counts.items()}


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def train(rows: list[dict], vocab: dict[str, int], epochs: int, lr: float, l2: float) -> tuple[list[float], float]:
    weights = [0.0] * len(vocab)
    bias = 0.0
    examples = [(vector(row, vocab), int(row.get("label_answer_bearing") or 0)) for row in rows]
    for _ in range(epochs):
        for feats, label in examples:
            score = bias + sum(weights[idx] * value for idx, value in feats.items())
            pred = sigmoid(score)
            grad = pred - label
            bias -= lr * grad
            for idx, value in feats.items():
                weights[idx] -= lr * (grad * value + l2 * weights[idx])
    return weights, bias


def predict(row: dict, vocab: dict[str, int], weights: list[float], bias: float) -> float:
    feats = vector(row, vocab)
    return sigmoid(bias + sum(weights[idx] * value for idx, value in feats.items()))


def metrics(rows: list[dict], scores: list[float]) -> dict[str, float]:
    labels = [int(row.get("label_answer_bearing") or 0) for row in rows]
    if not labels:
        return {"count": 0}
    preds = [int(score >= 0.5) for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    acc = sum(1 for y, p in zip(labels, preds) if y == p) / len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    brier = sum((score - y) ** 2 for score, y in zip(scores, labels)) / len(labels)
    return {
        "count": len(labels),
        "positive_count": sum(labels),
        "accuracy_at_0_5": acc,
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "brier_score": brier,
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    train_rows = list(iter_jsonl(args.train_path))
    dev_rows = list(iter_jsonl(args.dev_path)) if args.dev_path.exists() else []
    if not train_rows:
        raise SystemExit(f"No training rows found at {args.train_path}")
    vocab = build_vocab(train_rows, args.max_features)
    weights, bias = train(train_rows, vocab, args.epochs, args.lr, args.l2)
    train_scores = [predict(row, vocab, weights, bias) for row in train_rows]
    dev_scores = [predict(row, vocab, weights, bias) for row in dev_rows]
    write_json(
        args.output_dir / "metrics.json",
        {
            "model": "lexical_logistic_regression_baseline",
            "target": "label_answer_bearing",
            "train": metrics(train_rows, train_scores),
            "dev": metrics(dev_rows, dev_scores),
            "warning": "This is a fast audit baseline, not the final VLM/LoRA Model A proposed in MissRisk_RAG.md.",
        },
    )
    write_json(
        args.output_dir / "model.json",
        {"bias": bias, "vocab": vocab, "weights": weights},
    )
    if dev_rows:
        predictions = []
        for row, score in zip(dev_rows, dev_scores):
            predictions.append(
                {
                    "example_id": row.get("example_id"),
                    "question_id": row.get("question_id"),
                    "unit_id": row.get("unit_id"),
                    "label_answer_bearing": row.get("label_answer_bearing"),
                    "p_answer_bearing": score,
                }
            )
        write_jsonl(args.output_dir / "dev_predictions.jsonl", predictions)
    print(f"Wrote lexical Model A baseline to {args.output_dir}")


if __name__ == "__main__":
    main()
