from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.lib.common import JSON, ensure_dir, first_nonempty, iter_jsonl, normalize_space, write_json, write_jsonl


TEXT_STATES = [
    ("metadata_only", ["metadata"]),
    ("text_only", ["text"]),
    ("full_observation", ["text", "metadata"]),
]
IMAGE_STATES = [
    ("metadata_only", ["metadata"]),
    ("caption_only", ["caption_or_ocr"]),
    ("source_image_vlm", ["source_image", "caption_or_ocr", "metadata"]),
    ("full_observation", ["source_image", "caption_or_ocr", "metadata"]),
]
TABLE_STATES = [
    ("metadata_only", ["metadata"]),
    ("caption_only", ["caption_or_ocr"]),
    ("table_flattened", ["table_text", "caption_or_ocr", "metadata"]),
    ("source_image_vlm", ["source_image", "caption_or_ocr", "metadata"]),
    ("full_observation", ["table_text", "source_image", "caption_or_ocr", "metadata"]),
]
MULTIMODAL_STATES = [
    ("metadata_only", ["metadata"]),
    ("ocr_only", ["text"]),
    ("source_image_vlm", ["source_image", "caption_or_ocr", "metadata"]),
    ("full_observation", ["text", "source_image", "caption_or_ocr", "metadata"]),
]


ALL_CHANNELS = {
    "metadata",
    "text",
    "caption_or_ocr",
    "table_text",
    "source_image",
    "layout_context",
    "neighbor_units",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled observation-state rows for Model B/C."
    )
    parser.add_argument("--model-a-dir", type=Path, default=Path("data_missrisk/processed/mmdocrag_model_a"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_missrisk/processed/mmdocrag_model_bc_pilot"))
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument(
        "--include-b0",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include non-answer-bearing hard negatives as Model C negatives with D undefined.",
    )
    return parser.parse_args()


def content_for_unit(unit: JSON) -> str:
    raw = unit.get("raw_content") or {}
    return normalize_space(first_nonempty(raw.get("text"), raw.get("table_text"), raw.get("ocr"), raw.get("caption"), ""))


def modality(unit: JSON) -> str:
    return str(unit.get("modality") or "").lower()


def states_for_unit(unit: JSON) -> list[tuple[str, list[str]]]:
    mod = modality(unit)
    if mod == "text":
        return TEXT_STATES
    if mod == "table":
        return TABLE_STATES
    if mod in {"image", "caption"}:
        return IMAGE_STATES
    return MULTIMODAL_STATES


def visible_content(unit: JSON, qa: JSON | None, state_id: str, channels: list[str]) -> str:
    raw = unit.get("raw_content") or {}
    metadata = unit.get("metadata") or {}
    locator = unit.get("locator") or {}
    pieces: list[str] = []
    if "metadata" in channels:
        pieces.append(
            "\n".join(
                f"{key}: {value}"
                for key, value in {
                    "source_id": unit.get("source_id"),
                    "domain": metadata.get("domain"),
                    "page_id": locator.get("page_id"),
                    "quote_id": locator.get("quote_id"),
                    "modality": unit.get("modality"),
                }.items()
                if value is not None
            )
        )
    if "text" in channels:
        pieces.append(str(raw.get("text") or raw.get("ocr") or ""))
    if "caption_or_ocr" in channels:
        pieces.append(str(raw.get("caption") or raw.get("ocr") or ""))
    if "table_text" in channels:
        pieces.append(str(raw.get("table_text") or ""))
    if "source_image" in channels:
        path = raw.get("image_path")
        pieces.append(f"[source_image_available] {path}" if path else "[source_image_unavailable]")
    visible = "\n".join(normalize_space(piece) for piece in pieces if normalize_space(piece))
    if state_id == "metadata_only" and not visible:
        visible = f"unit_id: {unit.get('unit_id')}\nmodality: {unit.get('modality')}"
    return visible


def rule_detectable(unit: JSON, state_id: str, channels: list[str]) -> tuple[int | None, str, str]:
    label_b = unit.get("label_answer_bearing")
    if label_b != 1:
        return None, "not_applicable_b0", "Detectability is undefined for non-answer-bearing units."

    mod = modality(unit)
    if state_id == "metadata_only":
        return 0, "rule_controlled_hidden", "Only metadata is visible, so the oracle answer claim is intentionally hidden."
    if state_id == "full_observation":
        return 1, "oracle_full_observation", "The oracle answer-bearing unit is fully observed."
    if mod == "text" and "text" in channels:
        return 1, "oracle_text_unit", "The answer-bearing text payload is visible."
    if mod == "table" and "table_text" in channels:
        return 1, "oracle_table_text", "The table content is visible as structured/flattened text."
    if mod in {"image", "multimodal"} and "source_image" in channels:
        return 1, "oracle_visual_source", "The source visual channel is available for VLM inspection."
    return 0, "rule_partial_hidden", "The state omits the channel most likely needed to discover this oracle evidence."


def make_row(unit: JSON, qa: JSON | None, state_id: str, channels: list[str]) -> JSON:
    label_d, confidence, reason = rule_detectable(unit, state_id, channels)
    label_b = int(unit.get("label_answer_bearing") or 0)
    label_miss = int(label_b == 1 and label_d == 0)
    hidden = sorted(ALL_CHANNELS - set(channels))
    question_id = (unit.get("metadata") or {}).get("question_id") or (qa or {}).get("question_id")
    obligation = ((qa or {}).get("obligations") or [{}])[0]
    visible = visible_content(unit, qa, state_id, channels)
    return {
        "example_id": f"{unit['unit_id']}::{state_id}",
        "dataset": (unit.get("metadata") or {}).get("dataset"),
        "question_id": question_id,
        "question": (qa or {}).get("question"),
        "obligation_id": obligation.get("obligation_id") or obligation.get("id"),
        "obligation_text": obligation.get("text"),
        "unit_id": unit["unit_id"],
        "source_id": unit.get("source_id"),
        "state_id": state_id,
        "modality": unit.get("modality"),
        "visible_channels": channels,
        "hidden_channels": hidden,
        "visible_content": visible,
        "unit_content": content_for_unit(unit),
        "image_path": (unit.get("raw_content") or {}).get("image_path"),
        "label_answer_bearing": label_b,
        "label_detectable": label_d,
        "label_joint_miss": label_miss,
        "loss_mask_bear": 1,
        "loss_mask_detect": int(label_d is not None),
        "loss_mask_miss": 1,
        "label_confidence": confidence,
        "label_reason": reason,
        "split": (unit.get("metadata") or {}).get("split") or (qa or {}).get("split") or "train",
        "answer_short_for_audit_only": (qa or {}).get("answer_short_for_audit_only"),
        "source_file": (unit.get("metadata") or {}).get("source_file") or (qa or {}).get("source_file"),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    units_path = args.model_a_dir / "units.jsonl"
    qa_path = args.model_a_dir / "qa.jsonl"
    if not units_path.exists():
        raise SystemExit(f"Missing {units_path}. Run a build_*_units.py script first.")

    qa_by_id = {row["question_id"]: row for row in iter_jsonl(qa_path)} if qa_path.exists() else {}
    rows: list[JSON] = []
    for idx, unit in enumerate(iter_jsonl(units_path, limit=args.max_units)):
        if not args.include_b0 and unit.get("label_answer_bearing") != 1:
            continue
        qid = (unit.get("metadata") or {}).get("question_id")
        qa = qa_by_id.get(qid)
        for state_id, channels in states_for_unit(unit):
            rows.append(make_row(unit, qa, state_id, channels))

    by_split: dict[str, list[JSON]] = defaultdict(list)
    detect_by_split: dict[str, list[JSON]] = defaultdict(list)
    for row in rows:
        split = str(row.get("split") or "train")
        by_split[split].append(row)
        if row["loss_mask_detect"]:
            detect_by_split[split].append(row)

    for split in ("train", "calib", "dev", "test"):
        write_jsonl(args.output_dir / f"missrisk_{split}.jsonl", by_split.get(split, []))
        write_jsonl(args.output_dir / f"detectability_{split}.jsonl", detect_by_split.get(split, []))
    write_json(
        args.output_dir / "quality_report.json",
        {
            "row_count": len(rows),
            "detectability_row_count": sum(row["loss_mask_detect"] for row in rows),
            "joint_miss_positive_count": sum(row["label_joint_miss"] for row in rows),
            "answer_bearing_positive_count": sum(row["label_answer_bearing"] for row in rows),
            "splits": {split: len(items) for split, items in by_split.items()},
            "state_counts": {
                state: sum(1 for row in rows if row["state_id"] == state)
                for state in sorted({row["state_id"] for row in rows})
            },
            "label_rule": "label_joint_miss = 1 iff B=1 and D=0; D is null for B=0 rows.",
        },
    )
    readme = (
        "# Model B/C Observation-State Rows\n\n"
        "Rows are controlled observation interventions generated from Model A units.\n"
        "`label_detectable` is only defined for answer-bearing units. Non-answer-bearing\n"
        "hard negatives keep `label_detectable: null`, `loss_mask_detect: 0`, and\n"
        "`label_joint_miss: 0`.\n"
    )
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
