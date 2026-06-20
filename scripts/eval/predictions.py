from __future__ import annotations

import argparse
import math
from pathlib import Path

from scripts.lib.common import iter_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MissRisk predictions.")
    parser.add_argument("--predictions-path", type=Path, default=Path("outputs/models/missrisk_lexical/dev_predictions.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/eval/missrisk_metrics.json"))
    parser.add_argument("--score-key", default="p_joint_miss")
    return parser.parse_args()


def auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = [(score, label) for label, score in zip(labels, scores) if label == 1]
    negatives = [(score, label) for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for ps, _ in positives:
        for ns, _ in negatives:
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return wins / (len(positives) * len(negatives))


def ece(labels: list[int], scores: list[float], bins: int = 10) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    error = 0.0
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        bucket = [(label, score) for label, score in zip(labels, scores) if low <= score < high or (idx == bins - 1 and score == 1.0)]
        if not bucket:
            continue
        acc = sum(label for label, _ in bucket) / len(bucket)
        conf = sum(score for _, score in bucket) / len(bucket)
        error += len(bucket) / total * abs(acc - conf)
    return error


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.predictions_path))
    pairs = [
        (int(row["label_joint_miss"]), float(row[args.score_key]))
        for row in rows
        if row.get("label_joint_miss") is not None and row.get(args.score_key) is not None
    ]
    if not pairs:
        raise SystemExit(f"No evaluable rows found in {args.predictions_path}")
    labels = [label for label, _ in pairs]
    scores = [score for _, score in pairs]
    preds = [int(score >= 0.5) for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    metrics = {
        "count": len(labels),
        "positive_count": sum(labels),
        "score_key": args.score_key,
        "accuracy_at_0_5": sum(1 for y, p in zip(labels, preds) if y == p) / len(labels),
        "precision_at_0_5": tp / (tp + fp) if tp + fp else 0.0,
        "recall_at_0_5": tp / (tp + fn) if tp + fn else 0.0,
        "brier_score": sum((score - label) ** 2 for label, score in pairs) / len(pairs),
        "rmse": math.sqrt(sum((score - label) ** 2 for label, score in pairs) / len(pairs)),
        "auroc": auroc(labels, scores),
        "expected_calibration_error": ece(labels, scores),
    }
    write_json(args.output_path, metrics)
    print(f"Wrote metrics to {args.output_path}")


if __name__ == "__main__":
    main()
