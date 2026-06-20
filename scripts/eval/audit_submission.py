from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.lib.missrisk_common import classification_metrics, safe_auroc


JSON = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-submission audit for MissRisk-RAG leakage and shortcut baselines."
    )
    parser.add_argument(
        "--model-a-dir",
        type=Path,
        default=Path("data_missrisk/processed/model_a_training_mix_normalized"),
        help="Directory with train/dev/test *_pairs.jsonl for Model A.",
    )
    parser.add_argument(
        "--bc-dir",
        type=Path,
        default=Path("data_missrisk/processed/model_bc_training_mix"),
        help="Directory with missrisk_* and detectability_* jsonl files.",
    )
    parser.add_argument("--model-b-dir", type=Path, default=None)
    parser.add_argument("--model-c-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/missrisk_submission_audit.json"))
    return parser.parse_args()


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_key(row: JSON) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(
        row.get("source_id")
        or row.get("doc_name")
        or row.get("source_file")
        or meta.get("source_id")
        or meta.get("doc_name")
        or meta.get("source_file")
        or row.get("question_id")
        or ""
    )


def read_split_sources(patterns: dict[str, Path]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for split, path in patterns.items():
        out[split] = {source_key(row) for row in iter_jsonl(path) if source_key(row)}
    return out


def leakage_report(name: str, split_sources: dict[str, set[str]]) -> JSON:
    pairs = {}
    splits = sorted(split_sources)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = sorted(split_sources[left] & split_sources[right])
            pairs[f"{left}_vs_{right}"] = {
                "overlap_count": len(overlap),
                "sample": overlap[:20],
            }
    return {
        "name": name,
        "source_counts": {split: len(values) for split, values in split_sources.items()},
        "pairwise_overlap": pairs,
        "source_clean": all(item["overlap_count"] == 0 for item in pairs.values()),
    }


def rows(path: Path) -> list[JSON]:
    return list(iter_jsonl(path) or [])


def label_entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def group_label_report(data: list[JSON], label_key: str, group_keys: tuple[str, ...]) -> JSON:
    grouped: dict[tuple[str, ...], Counter[int]] = defaultdict(Counter)
    for row in data:
        label = row.get(label_key)
        if label is None:
            continue
        key = tuple(str(row.get(item)) for item in group_keys)
        grouped[key][int(label)] += 1
    pure = 0
    items = []
    for key, counter in sorted(grouped.items()):
        total = sum(counter.values())
        pos_rate = counter[1] / total if total else 0.0
        entropy = label_entropy(counter)
        if entropy == 0.0:
            pure += 1
        items.append(
            {
                "group": dict(zip(group_keys, key, strict=True)),
                "count": total,
                "positive_rate": pos_rate,
                "entropy": entropy,
                "labels": {str(k): v for k, v in sorted(counter.items())},
            }
        )
    return {
        "group_keys": list(group_keys),
        "group_count": len(items),
        "pure_group_count": pure,
        "pure_group_fraction": pure / len(items) if items else 0.0,
        "groups": items,
    }


def train_group_rates(train: list[JSON], label_key: str, group_keys: tuple[str, ...]) -> tuple[dict[tuple[str, ...], float], float]:
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    labels = []
    for row in train:
        label = row.get(label_key)
        if label is None:
            continue
        y = int(label)
        labels.append(y)
        key = tuple(str(row.get(item)) for item in group_keys)
        grouped[key].append(y)
    fallback = sum(labels) / len(labels) if labels else 0.0
    rates = {key: sum(values) / len(values) for key, values in grouped.items()}
    return rates, fallback


def grouped_baseline(
    train: list[JSON],
    eval_rows: list[JSON],
    *,
    label_key: str,
    group_keys: tuple[str, ...],
) -> JSON:
    rates, fallback = train_group_rates(train, label_key, group_keys)
    labels = []
    probs = []
    missing = 0
    for row in eval_rows:
        label = row.get(label_key)
        if label is None:
            continue
        key = tuple(str(row.get(item)) for item in group_keys)
        labels.append(int(label))
        if key not in rates:
            missing += 1
        probs.append(rates.get(key, fallback))
    metrics = classification_metrics(labels, probs) if labels else {"count": 0}
    metrics["auroc"] = safe_auroc(labels, probs) if labels else float("nan")
    return {
        "label_key": label_key,
        "group_keys": list(group_keys),
        "fallback_positive_rate": fallback,
        "missing_group_count": missing,
        "metrics": metrics,
    }


def load_metrics(path: Path | None) -> JSON | None:
    if path is None or not path.exists():
        return None
    metrics_path = path / "metrics.json"
    test_path = path / "test_metrics.json"
    payload: JSON = {}
    if metrics_path.exists():
        payload["metrics"] = json.loads(metrics_path.read_text())
    if test_path.exists():
        payload["test_metrics"] = json.loads(test_path.read_text())
    return payload or None


def main() -> None:
    args = parse_args()
    model_a_sources = read_split_sources(
        {split: args.model_a_dir / f"{split}_pairs.jsonl" for split in ("train", "dev", "test")}
    )
    bc_miss_sources = read_split_sources(
        {split: args.bc_dir / f"missrisk_{split}.jsonl" for split in ("train", "dev", "test")}
    )
    bc_detect_sources = read_split_sources(
        {split: args.bc_dir / f"detectability_{split}.jsonl" for split in ("train", "dev", "test")}
    )

    detect_train = rows(args.bc_dir / "detectability_train.jsonl")
    detect_dev = rows(args.bc_dir / "detectability_dev.jsonl")
    miss_train = rows(args.bc_dir / "missrisk_train.jsonl")
    miss_dev = rows(args.bc_dir / "missrisk_dev.jsonl")
    miss_test = rows(args.bc_dir / "missrisk_test.jsonl")

    report: JSON = {
        "source_leakage": {
            "model_a": leakage_report("model_a", model_a_sources),
            "bc_missrisk": leakage_report("bc_missrisk", bc_miss_sources),
            "bc_detectability": leakage_report("bc_detectability", bc_detect_sources),
        },
        "detectability_shortcuts": {
            "dev_by_state": group_label_report(detect_dev, "label_detectable", ("state_id",)),
            "dev_by_modality_state": group_label_report(detect_dev, "label_detectable", ("modality", "state_id")),
            "state_rule_baseline": grouped_baseline(
                detect_train,
                detect_dev,
                label_key="label_detectable",
                group_keys=("state_id",),
            ),
            "modality_state_rule_baseline": grouped_baseline(
                detect_train,
                detect_dev,
                label_key="label_detectable",
                group_keys=("modality", "state_id"),
            ),
        },
        "missrisk_trivial_baselines": {
            "dev_state": grouped_baseline(
                miss_train,
                miss_dev,
                label_key="label_joint_miss",
                group_keys=("state_id",),
            ),
            "dev_modality_state": grouped_baseline(
                miss_train,
                miss_dev,
                label_key="label_joint_miss",
                group_keys=("modality", "state_id"),
            ),
            "test_state": grouped_baseline(
                miss_train,
                miss_test,
                label_key="label_joint_miss",
                group_keys=("state_id",),
            ),
            "test_modality_state": grouped_baseline(
                miss_train,
                miss_test,
                label_key="label_joint_miss",
                group_keys=("modality", "state_id"),
            ),
        },
        "trained_metrics": {
            "model_b": load_metrics(args.model_b_dir),
            "model_c": load_metrics(args.model_c_dir),
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
