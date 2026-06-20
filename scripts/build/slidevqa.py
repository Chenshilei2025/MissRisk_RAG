from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from scripts.lib.common import (
    JSON,
    ensure_dir,
    find_json_inputs,
    first_nonempty,
    iter_jsonl,
    lexical_overlap,
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
    TextView,
    VisualView,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local SlideVQA-style JSONL annotations into MissRiskBench Model A files."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data_missrisk/raw/slidevqa"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_missrisk/processed/slidevqa_model_a"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-negatives-per-question", type=int, default=6)
    return parser.parse_args()


def slide_candidates(record: JSON) -> list[JSON]:
    for key in ("slides", "slide_pages", "pages", "candidates"):
        value = record.get(key)
        if isinstance(value, list):
            return [item if isinstance(item, dict) else {"text": str(item)} for item in value]
    return []


def evidence_slide_ids(record: JSON) -> set[str]:
    ids = set()
    for key in ("evidence_slide_ids", "evidence_pages", "gold_slide_ids", "answer_slide_ids"):
        for value in record.get(key) or []:
            ids.add(str(value))
    answer = record.get("answer")
    if isinstance(answer, dict):
        for key in ("evidence_slide_ids", "evidence_pages", "gold_slide_ids"):
            for value in answer.get(key) or []:
                ids.add(str(value))
    return ids


def make_unit(question_id: str, split: str, record: JSON, slide: JSON, label: int) -> ObservationUnit:
    deck_id = str(first_nonempty(record.get("deck_id"), record.get("slide_deck"), record.get("presentation_id"), question_id))
    slide_id = str(first_nonempty(slide.get("slide_id"), slide.get("page_id"), slide.get("id"), slide.get("index"), "slide"))
    text = truncate(str(first_nonempty(slide.get("ocr"), slide.get("text"), slide.get("title"), slide.get("caption"), "")))
    image_path = first_nonempty(slide.get("image_path"), slide.get("slide_image"), slide.get("path"))
    raw_content = {"ocr": text} if text else {}
    if image_path:
        raw_content["image_path"] = str(image_path)
    views = ObservationViews(
        text_view=TextView(content=text, source_key="ocr") if text else None,
        visual_view=VisualView(image_path=str(image_path), asset_available=True) if image_path else None,
    )
    return ObservationUnit(
        unit_id=f"slidevqa:{question_id}:{deck_id}:{slide_id}",
        source_id=deck_id,
        source_type=SourceType.SLIDE,
        modality=Modality.MULTIMODAL if image_path and text else (Modality.IMAGE if image_path else Modality.OCR),
        locator={"deck_id": deck_id, "slide_id": slide_id},
        raw_content=raw_content,
        views=views,
        metadata={"dataset": "slidevqa", "question_id": question_id, "split": split},
        label_answer_bearing=label,
    )


def make_pair(qa: JSON, unit: ObservationUnit) -> JSON:
    text = unit.raw_content.get("ocr") or unit.raw_content.get("caption") or ""
    label = int(unit.label_answer_bearing or 0)
    return {
        "example_id": unit.unit_id,
        "dataset": "slidevqa",
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
        "image_path": unit.raw_content.get("image_path"),
        "requires_visual_input": bool(unit.raw_content.get("image_path")),
        "label_answer_bearing": label,
        "label_text": "Yes" if label else "No",
        "target_text": "Yes" if label else "No",
        "text_view": unit.views.text_view,
        "visual_view": unit.views.visual_view,
    }


def select_units(question: str, units: list[ObservationUnit], max_negatives: int) -> list[ObservationUnit]:
    positives = [unit for unit in units if unit.label_answer_bearing == 1]
    negatives = [unit for unit in units if unit.label_answer_bearing == 0]
    scored = []
    for unit in negatives:
        text = str(unit.raw_content.get("ocr") or unit.raw_content.get("caption") or "")
        scored.append((lexical_overlap(question, text), unit.unit_id, unit))
    scored.sort(reverse=True)
    return positives + [unit for _, _, unit in scored[:max_negatives]]


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    inputs = find_json_inputs(args.input_dir)
    inputs = [path for path in inputs if ".cache" not in str(path) and "/repo/" not in str(path)]
    if not inputs:
        raise SystemExit(
            f"No SlideVQA JSONL files found under {args.input_dir}. "
            "Download/prepare a local JSONL with question, slides, and evidence slide ids first."
        )

    qa_rows: list[JSON] = []
    unit_rows: list[ObservationUnit] = []
    pair_rows: list[JSON] = []
    dropped: list[JSON] = []
    seen = 0
    for path in inputs:
        for record in iter_jsonl(path):
            if args.max_examples is not None and seen >= args.max_examples:
                break
            question = normalize_space(first_nonempty(record.get("question"), record.get("query")))
            slides = slide_candidates(record)
            if not question or not slides:
                dropped.append({"source_file": str(path), "reason": "missing_question_or_slides"})
                continue
            raw_qid = str(first_nonempty(record.get("qid"), record.get("question_id"), seen))
            question_id = f"slidevqa:{raw_qid}"
            gold = evidence_slide_ids(record)
            deck_id = str(first_nonempty(record.get("deck_id"), record.get("slide_deck"), record.get("presentation_id"), raw_qid))
            split = split_for_key(deck_id)
            units = []
            for idx, slide in enumerate(slides):
                slide_id = str(first_nonempty(slide.get("slide_id"), slide.get("page_id"), slide.get("id"), slide.get("index"), idx))
                units.append(make_unit(question_id, split, record, slide, int(slide_id in gold)))
            selected = select_units(question, units, args.max_negatives_per_question)
            obligation = make_obligation(question_id, question, ["slide", "ocr", "image"])
            qa = {
                "dataset": "slidevqa",
                "question_id": question_id,
                "q_id": raw_qid,
                "question": question,
                "answer_short_for_audit_only": first_nonempty(record.get("answer"), record.get("answers")),
                "obligations": [obligation],
                "gold_unit_ids": [unit.unit_id for unit in selected if unit.label_answer_bearing == 1],
                "unit_ids": [unit.unit_id for unit in selected],
                "source_file": path.name,
                "deck_id": deck_id,
                "split": split,
            }
            qa_rows.append(qa)
            unit_rows.extend(selected)
            pair_rows.extend(make_pair(qa, unit) for unit in selected)
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
            "dataset": "slidevqa",
            "input_paths": [str(path) for path in inputs],
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
