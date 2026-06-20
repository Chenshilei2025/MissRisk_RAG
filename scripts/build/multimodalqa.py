from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.lib.common import (
    JSON,
    compact_answer,
    ensure_dir,
    first_nonempty,
    iter_jsonl,
    make_obligation,
    normalize_space,
    split_for_key,
    truncate,
    write_json,
    write_jsonl,
)

from agentic_mm_rag.observation.units import (
    Modality,
    ObservationUnit,
    ObservationViews,
    SourceType,
    TableView,
    TextView,
    VisualView,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MultiModalQA into MissRiskBench Model A files."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data_missrisk/raw/multimodalqa"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_missrisk/processed/multimodalqa_model_a"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-negatives-per-question", type=int, default=6)
    return parser.parse_args()


def load_corpus(input_dir: Path, max_docs_per_kind: int | None = None) -> dict[str, JSON]:
    corpus: dict[str, JSON] = {}
    specs = [
        ("MMQA_texts.jsonl.gz", "text"),
        ("MMQA_tables.jsonl.gz", "table"),
        ("MMQA_images.jsonl.gz", "image"),
    ]
    for filename, kind in specs:
        path = input_dir / filename
        if not path.exists():
            continue
        for row in iter_jsonl(path, limit=max_docs_per_kind):
            doc_id = str(first_nonempty(row.get("id"), row.get("doc_id"), row.get("table_id"), row.get("image_id")))
            if not doc_id:
                continue
            row["_missrisk_kind"] = kind
            corpus[doc_id] = row
    return corpus


def doc_text(row: JSON, kind: str) -> str:
    if kind == "text":
        return truncate(str(first_nonempty(row.get("text"), row.get("content"), row.get("passage"), row.get("title"), "")))
    if kind == "table":
        title = normalize_space(first_nonempty(row.get("title"), row.get("page_title"), ""))
        header = row.get("header") or row.get("headers") or []
        rows = row.get("rows") or row.get("table") or row.get("data") or []
        pieces = [title, json.dumps(header, ensure_ascii=False), json.dumps(rows[:20] if isinstance(rows, list) else rows, ensure_ascii=False)]
        return truncate(" ".join(piece for piece in pieces if piece and piece != "[]"))
    caption = first_nonempty(row.get("caption"), row.get("title"), row.get("alt"), row.get("description"), "")
    return truncate(str(caption))


def make_unit(dataset_split: str, question_id: str, row: JSON, label: int) -> ObservationUnit:
    doc_id = str(first_nonempty(row.get("id"), row.get("doc_id"), row.get("table_id"), row.get("image_id")))
    kind = str(row.get("_missrisk_kind") or "text")
    modality = {"text": Modality.TEXT, "table": Modality.TABLE, "image": Modality.IMAGE}.get(kind, Modality.TEXT)
    content = doc_text(row, kind)
    image_path = first_nonempty(row.get("image_path"), row.get("path"), row.get("file_name"), row.get("filename"))
    raw_content: dict[str, Any] = {}
    views = ObservationViews()
    if modality == Modality.TEXT:
        raw_content["text"] = content
        views.text_view = TextView(content=content, source_key="text")
        source_type = SourceType.DOCUMENT
    elif modality == Modality.TABLE:
        raw_content["table_text"] = content
        views.text_view = TextView(content=content, source_key="table_text")
        views.table_view = TableView(table_text=content, source_key="table_text")
        source_type = SourceType.TABLE
    else:
        raw_content["caption"] = content
        if image_path:
            raw_content["image_path"] = str(image_path)
        views.text_view = TextView(content=content, source_key="caption")
        if image_path:
            views.visual_view = VisualView(image_path=str(image_path), asset_available=True)
        source_type = SourceType.IMAGE
    return ObservationUnit(
        unit_id=f"multimodalqa:{question_id}:{doc_id}:{kind}",
        source_id=doc_id,
        source_type=source_type,
        modality=modality,
        locator={"doc_id": doc_id, "doc_part": kind},
        raw_content=raw_content,
        views=views,
        metadata={"dataset": "multimodalqa", "question_id": question_id, "split": dataset_split},
        label_answer_bearing=label,
    )


def candidate_ids(record: JSON) -> list[str]:
    metadata = record.get("metadata") or {}
    ids: list[str] = []
    for key in ("text_doc_ids", "image_doc_ids"):
        ids.extend(str(item) for item in metadata.get(key) or [])
    table_id = metadata.get("table_id")
    if table_id:
        ids.append(str(table_id))
    return list(dict.fromkeys(ids))


def supporting_ids(record: JSON) -> set[str]:
    ids = set()
    for item in record.get("supporting_context") or []:
        if isinstance(item, dict) and item.get("doc_id"):
            ids.add(str(item["doc_id"]))
    for answer in record.get("answers") or []:
        if not isinstance(answer, dict):
            continue
        for text_instance in answer.get("text_instances") or []:
            if isinstance(text_instance, dict) and text_instance.get("doc_id"):
                ids.add(str(text_instance["doc_id"]))
        for image_instance in answer.get("image_instances") or []:
            if isinstance(image_instance, dict) and image_instance.get("doc_id"):
                ids.add(str(image_instance["doc_id"]))
        if answer.get("table_indices"):
            table_id = (record.get("metadata") or {}).get("table_id")
            if table_id:
                ids.add(str(table_id))
    return ids


def make_pair(qa: JSON, unit: ObservationUnit) -> JSON:
    text = unit.raw_content.get("text") or unit.raw_content.get("table_text") or unit.raw_content.get("caption") or ""
    label = int(unit.label_answer_bearing or 0)
    return {
        "example_id": unit.unit_id,
        "dataset": "multimodalqa",
        "model_a_task": "vl_answer_bearing",
        "question_id": qa["question_id"],
        "question": qa["question"],
        "obligation_id": qa["obligations"][0]["obligation_id"],
        "obligation_text": qa["obligations"][0]["text"],
        "unit_id": unit.unit_id,
        "source_id": unit.source_id,
        "modality": unit.modality.value,
        "unit_content": text,
        "unit_aux_text": text,
        "split": unit.metadata.get("split"),
        "label_answer_bearing": label,
        "label_text": "Yes" if label else "No",
        "target_text": "Yes" if label else "No",
        "requires_visual_input": unit.modality in {Modality.IMAGE, Modality.TABLE},
        "image_path": unit.raw_content.get("image_path"),
        "text_view": unit.views.text_view,
        "visual_view": unit.views.visual_view,
        "table_view": unit.views.table_view,
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    corpus = load_corpus(args.input_dir)
    if not corpus:
        raise SystemExit(f"No MultiModalQA corpus files found under {args.input_dir}")

    split_files = [("train", "MMQA_train.jsonl.gz"), ("dev", "MMQA_dev.jsonl.gz"), ("test", "MMQA_test.jsonl.gz")]
    qa_rows: list[JSON] = []
    unit_rows: list[ObservationUnit] = []
    pair_rows: list[JSON] = []
    dropped: list[JSON] = []
    seen = 0
    for dataset_split, filename in split_files:
        path = args.input_dir / filename
        if not path.exists():
            continue
        for record in iter_jsonl(path):
            if args.max_examples is not None and seen >= args.max_examples:
                break
            raw_qid = str(record.get("qid") or seen)
            question_id = f"multimodalqa:{raw_qid}"
            question = normalize_space(record.get("question"))
            if not question:
                continue
            gold = supporting_ids(record)
            candidates = candidate_ids(record)
            selected_ids = list(gold)
            selected_ids.extend(item for item in candidates if item not in gold)
            selected_ids = selected_ids[: len(gold) + args.max_negatives_per_question]
            units: list[ObservationUnit] = []
            missing: list[str] = []
            split = split_for_key(first_nonempty(next(iter(gold), None), raw_qid))
            for doc_id in selected_ids:
                row = corpus.get(doc_id)
                if row is None:
                    missing.append(doc_id)
                    continue
                units.append(make_unit(split, question_id, row, int(doc_id in gold)))
            if not units:
                dropped.append({"question_id": question_id, "reason": "no_candidate_units_found", "missing_doc_ids": missing})
                continue
            modalities = [str(item).lower() for item in (record.get("metadata") or {}).get("modalities") or []]
            obligation = make_obligation(question_id, question, modalities)
            qa = {
                "dataset": "multimodalqa",
                "question_id": question_id,
                "q_id": raw_qid,
                "question": question,
                "answer_short_for_audit_only": compact_answer(record),
                "question_type": (record.get("metadata") or {}).get("type"),
                "evidence_modality_type": modalities,
                "obligations": [obligation],
                "gold_unit_ids": [unit.unit_id for unit in units if unit.label_answer_bearing == 1],
                "unit_ids": [unit.unit_id for unit in units],
                "source_file": filename,
                "split": split,
                "missing_doc_ids": missing,
            }
            qa_rows.append(qa)
            unit_rows.extend(units)
            pair_rows.extend(make_pair(qa, unit) for unit in units)
            seen += 1
        if args.max_examples is not None and seen >= args.max_examples:
            break

    by_split: dict[str, list[JSON]] = defaultdict(list)
    for pair in pair_rows:
        by_split[str(pair.get("split") or "train")].append(pair)

    write_jsonl(args.output_dir / "qa.jsonl", qa_rows)
    write_jsonl(args.output_dir / "units.jsonl", unit_rows)
    for split in ("train", "dev", "test"):
        write_jsonl(args.output_dir / f"{split}_pairs.jsonl", by_split.get(split, []))
    if dropped:
        write_jsonl(args.output_dir / "dropped_examples.jsonl", dropped)
    write_json(
        args.output_dir / "manifest.json",
        {
            "dataset": "multimodalqa",
            "question_count": len(qa_rows),
            "unit_count": len(unit_rows),
            "pair_count": len(pair_rows),
            "positive_pair_count": sum(int(row["label_answer_bearing"]) for row in pair_rows),
            "dropped_count": len(dropped),
            "splits": {split: len(rows) for split, rows in by_split.items()},
        },
    )


if __name__ == "__main__":
    main()
