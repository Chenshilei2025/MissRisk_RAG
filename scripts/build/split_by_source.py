from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.lib.common import split_for_key, write_json, write_jsonl


JSON = dict[str, Any]
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite Model A and Model B/C jsonl files with one shared source-level split."
    )
    parser.add_argument("--model-a-in", type=Path, required=True)
    parser.add_argument("--bc-in", type=Path, required=True)
    parser.add_argument("--model-a-out", type=Path, required=True)
    parser.add_argument("--bc-out", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--split-map", type=Path, default=None)
    return parser.parse_args()


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def source_key(row: JSON) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    dataset = row.get("dataset") or meta.get("dataset") or "unknown"
    source = (
        row.get("source_id")
        or row.get("doc_name")
        or meta.get("source_id")
        or meta.get("doc_name")
        or row.get("source_file")
        or meta.get("source_file")
        or row.get("question_id")
    )
    return f"{dataset}::{source}"


def load_existing_split_map(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "source_to_split" in payload:
        payload = payload["source_to_split"]
    return {str(key): str(value) for key, value in payload.items()}


def collect_sources(paths: list[Path]) -> set[str]:
    sources = set()
    for path in paths:
        for row in iter_jsonl(path) or []:
            sources.add(source_key(row))
    return sources


def assign_splits(sources: set[str], existing: dict[str, str], train_ratio: float, dev_ratio: float) -> dict[str, str]:
    out = dict(existing)
    for key in sorted(sources):
        if out.get(key) not in SPLITS:
            out[key] = split_for_key(key, train_ratio=train_ratio, dev_ratio=dev_ratio)
    return out


def rewrite_rows(input_paths: list[Path], output_dir: Path, source_to_split: dict[str, str]) -> dict[str, Any]:
    buckets: dict[str, list[JSON]] = defaultdict(list)
    source_by_split: dict[str, set[str]] = defaultdict(set)
    rows_in = 0
    for path in input_paths:
        for row in iter_jsonl(path) or []:
            rows_in += 1
            row = dict(row)
            key = source_key(row)
            split = source_to_split[key]
            row["split"] = split
            if isinstance(row.get("metadata"), dict):
                row["metadata"] = dict(row["metadata"])
                row["metadata"]["split"] = split
            buckets[split].append(row)
            source_by_split[split].add(key)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(output_dir / f"{split}_pairs.jsonl", buckets.get(split, []))
    return {
        "rows_in": rows_in,
        "rows_out": sum(len(rows) for rows in buckets.values()),
        "row_counts": {split: len(buckets.get(split, [])) for split in SPLITS},
        "source_counts": {split: len(source_by_split.get(split, set())) for split in SPLITS},
        "label_answer_bearing": {
            split: dict(Counter(str(row.get("label_answer_bearing")) for row in buckets.get(split, [])))
            for split in SPLITS
        },
    }


def rewrite_bc(input_dir: Path, output_dir: Path, source_to_split: dict[str, str]) -> dict[str, Any]:
    all_miss = []
    all_detect = []
    for split in SPLITS:
        all_miss.extend(iter_jsonl(input_dir / f"missrisk_{split}.jsonl") or [])
        all_detect.extend(iter_jsonl(input_dir / f"detectability_{split}.jsonl") or [])
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, data in (("missrisk", all_miss), ("detectability", all_detect)):
        buckets: dict[str, list[JSON]] = defaultdict(list)
        source_by_split: dict[str, set[str]] = defaultdict(set)
        for row in data:
            row = dict(row)
            key = source_key(row)
            split = source_to_split[key]
            row["split"] = split
            buckets[split].append(row)
            source_by_split[split].add(key)
        for split in SPLITS:
            write_jsonl(output_dir / f"{name}_{split}.jsonl", buckets.get(split, []))
        report[name] = {
            "rows_in": len(data),
            "rows_out": sum(len(rows) for rows in buckets.values()),
            "row_counts": {split: len(buckets.get(split, [])) for split in SPLITS},
            "source_counts": {split: len(source_by_split.get(split, set())) for split in SPLITS},
            "label_joint_miss": {
                split: dict(Counter(str(row.get("label_joint_miss")) for row in buckets.get(split, [])))
                for split in SPLITS
            },
            "label_detectable": {
                split: dict(Counter(str(row.get("label_detectable")) for row in buckets.get(split, [])))
                for split in SPLITS
            },
        }
    readme = (
        "# Source-Clean Model B/C Split\n\n"
        "Generated by `scripts/build/split_by_source.py` using a shared hash split over "
        "`dataset::source_id/doc_name`. These files are intended to be used by Model A, B, and C "
        "for leakage-free pre-submission runs.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    model_a_inputs = [args.model_a_in / f"{split}_pairs.jsonl" for split in SPLITS]
    bc_inputs = [
        *(args.bc_in / f"missrisk_{split}.jsonl" for split in SPLITS),
        *(args.bc_in / f"detectability_{split}.jsonl" for split in SPLITS),
    ]
    existing = load_existing_split_map(args.split_map)
    source_to_split = assign_splits(
        collect_sources([*model_a_inputs, *bc_inputs]),
        existing,
        args.train_ratio,
        args.dev_ratio,
    )
    model_a_report = rewrite_rows(model_a_inputs, args.model_a_out, source_to_split)
    bc_report = rewrite_bc(args.bc_in, args.bc_out, source_to_split)
    payload = {
        "source_to_split": source_to_split,
        "split_counts": dict(Counter(source_to_split.values())),
        "model_a": model_a_report,
        "bc": bc_report,
        "train_ratio": args.train_ratio,
        "dev_ratio": args.dev_ratio,
    }
    write_json(args.model_a_out / "source_split_manifest.json", payload)
    write_json(args.bc_out / "source_split_manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
