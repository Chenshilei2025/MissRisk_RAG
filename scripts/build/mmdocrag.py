from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.lib.common import (
    JSON,
    compact_answer,
    ensure_dir,
    find_json_inputs,
    first_nonempty,
    iter_jsonl,
    lexical_overlap,
    make_obligation,
    normalize_space,
    split_for_key,
    stable_id,
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


TEXT_SECTION_RE = re.compile(r"Text Quotes are:\s*(.*?)(?:\nImage Quotes are:|\Z)", re.S)
IMAGE_SECTION_RE = re.compile(r"Image Quotes are:\s*(.*?)(?:\n\nThe user question is:|\Z)", re.S)
QUESTION_RE = re.compile(r"The user question is:\s*(.*)", re.S)
TEXT_QUOTE_RE = re.compile(r"\n?\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", re.S)
IMAGE_QUOTE_RE = re.compile(r"\n?image(\d+)\s+is described as:\s*(.*?)(?=\nimage\d+\s+is described as:|\Z)", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MMDocRAG raw annotations into MissRiskBench Model A files."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data_missrisk/raw/mmdocrag"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_missrisk/processed/mmdocrag_model_a"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-negatives-per-question", type=int, default=6)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--calib-ratio", type=float, default=0.1)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    return parser.parse_args()


def four_way_split_for_key(key: str, train_ratio: float, calib_ratio: float, dev_ratio: float) -> str:
    bucket = int(stable_id(key, length=8), 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + calib_ratio:
        return "calib"
    if bucket < train_ratio + calib_ratio + dev_ratio:
        return "dev"
    return "test"


def extract_user_and_assistant(record: JSON) -> tuple[str, str]:
    messages = record.get("messages") or []
    user = ""
    assistant = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user":
            user = content
        elif role == "assistant":
            assistant = content
    return user, assistant


def parse_message_record(record: JSON) -> tuple[str, list[JSON], str]:
    user, assistant = extract_user_and_assistant(record)
    question_match = QUESTION_RE.search(user)
    question = normalize_space(question_match.group(1) if question_match else record.get("question"))

    quotes: list[JSON] = []
    text_section = TEXT_SECTION_RE.search(user)
    if text_section:
        for idx, content in TEXT_QUOTE_RE.findall(text_section.group(1)):
            quotes.append({"quote_id": f"text{idx}", "quote_type": "text", "content": normalize_space(content)})

    image_section = IMAGE_SECTION_RE.search(user)
    if image_section:
        for idx, content in IMAGE_QUOTE_RE.findall(image_section.group(1)):
            quotes.append({"quote_id": f"image{idx}", "quote_type": "image", "content": normalize_space(content)})

    gold_ids = set(str(q) for q in record.get("gold_quotes") or [])
    for idx in re.findall(r"\[(\d+)\]", assistant):
        gold_ids.add(f"text{idx}")
    for idx in re.findall(r"\(image\s*(\d+)\)", assistant, flags=re.I):
        gold_ids.add(f"image{idx}")
    for quote in quotes:
        quote["label_answer_bearing"] = int(quote["quote_id"] in gold_ids)
    return question, quotes, assistant


def parse_structured_record(record: JSON) -> tuple[str, list[JSON], str]:
    question = normalize_space(record.get("question"))
    quotes: list[JSON] = []
    gold_ids = {str(item) for item in (record.get("gold_quotes") or [])}
    for idx, item in enumerate(record.get("text_quotes") or [], start=1):
        quote_id = str(item.get("quote_id") or item.get("id") or f"text{idx}") if isinstance(item, dict) else f"text{idx}"
        content = item.get("text") if isinstance(item, dict) else item
        quotes.append(
            {
                "quote_id": quote_id,
                "quote_type": "text",
                "content": normalize_space(content),
                "locator": item if isinstance(item, dict) else {},
                "label_answer_bearing": int(quote_id in gold_ids),
            }
        )
    for idx, item in enumerate(record.get("img_quotes") or record.get("image_quotes") or [], start=1):
        quote_id = str(item.get("quote_id") or item.get("id") or f"image{idx}") if isinstance(item, dict) else f"image{idx}"
        content = first_nonempty(
            item.get("caption") if isinstance(item, dict) else None,
            item.get("description") if isinstance(item, dict) else None,
            item.get("text") if isinstance(item, dict) else None,
            item if not isinstance(item, dict) else "",
        )
        quotes.append(
            {
                "quote_id": quote_id,
                "quote_type": "image",
                "content": normalize_space(content),
                "locator": item if isinstance(item, dict) else {},
                "label_answer_bearing": int(quote_id in gold_ids),
            }
        )
    return question, quotes, normalize_space(record.get("answer_interleaved"))


def modality_for_quote(quote: JSON, evidence_modalities: list[str]) -> Modality:
    quote_type = str(quote.get("quote_type") or "").lower()
    content = str(quote.get("content") or "").lower()
    all_modalities = " ".join(evidence_modalities).lower()
    if quote_type.startswith("text"):
        return Modality.TEXT
    if "table" in all_modalities or "table" in content:
        return Modality.TABLE
    return Modality.IMAGE


def build_unit(
    *,
    dataset: str,
    source_file: str,
    record: JSON,
    question_id: str,
    question: str,
    quote: JSON,
    split: str,
) -> ObservationUnit:
    evidence_modalities = [str(item).lower() for item in record.get("evidence_modality_type") or []]
    modality = modality_for_quote(quote, evidence_modalities)
    doc_name = str(first_nonempty(record.get("doc_name"), record.get("document_name"), f"{source_file}:{question_id}"))
    quote_id = str(quote["quote_id"])
    locator = dict(quote.get("locator") or {})
    locator.update(
        {
            "doc_name": doc_name,
            "quote_id": quote_id,
            "page_id": first_nonempty(locator.get("page_id"), quote.get("page_id")),
            "layout_id": first_nonempty(locator.get("layout_id"), quote.get("layout_id")),
        }
    )
    content = truncate(str(quote.get("content") or ""))
    image_path = first_nonempty(
        quote.get("image_path"),
        locator.get("image_path"),
        locator.get("img_path"),
    )
    raw_content: dict[str, Any] = {}
    views = ObservationViews()
    if modality == Modality.TEXT:
        raw_content["text"] = content
        views.text_view = TextView(content=content, source_key="text")
    elif modality == Modality.TABLE:
        raw_content["caption"] = content
        raw_content["table_text"] = content
        if image_path:
            raw_content["image_path"] = str(image_path)
        views.text_view = TextView(content=content, source_key="caption")
        views.table_view = TableView(table_text=content, image_path=str(image_path) if image_path else None, source_key="table_text")
        if image_path:
            views.visual_view = VisualView(image_path=str(image_path), asset_available=True)
    else:
        raw_content["caption"] = content
        if image_path:
            raw_content["image_path"] = str(image_path)
        views.text_view = TextView(content=content, source_key="caption")
        if image_path:
            views.visual_view = VisualView(image_path=str(image_path), asset_available=True)

    return ObservationUnit(
        unit_id=f"mmdocrag:{source_file}:{question_id}:{quote_id}",
        source_id=doc_name,
        source_type=SourceType.DOCUMENT,
        modality=modality,
        locator={k: v for k, v in locator.items() if v is not None},
        raw_content=raw_content,
        views=views,
        metadata={
            "dataset": dataset,
            "source_file": source_file,
            "question_id": f"mmdocrag:{source_file}:{question_id}",
            "domain": record.get("domain"),
            "question_type": record.get("question_type"),
            "split": split,
            "quote_type": quote.get("quote_type"),
        },
        label_answer_bearing=int(quote.get("label_answer_bearing") or 0),
    )


def make_pair(qa: JSON, unit: ObservationUnit) -> JSON:
    unit_text = unit.raw_content.get("text") or unit.raw_content.get("table_text") or unit.raw_content.get("caption") or ""
    requires_visual = unit.modality.value in {"image", "table", "multimodal"}
    label = int(unit.label_answer_bearing or 0)
    prompt = (
        "Answer with exactly one token: Yes or No.\n"
        "Yes means the candidate unit itself contains answer-bearing evidence.\n"
        "No means it is irrelevant, only topically related, or insufficient.\n\n"
        f"[QUESTION]\n{qa['question']}\n\n"
        f"[OBLIGATION]\n{qa['obligations'][0]['text']}\n\n"
        f"[UNIT_MODALITY]\n{unit.modality.value}\n"
        f"[VISUAL_INPUT]\n{'A source image/table crop may be attached; use pixels as primary evidence.' if requires_visual else 'No image is attached for this unit; judge from text.'}\n\n"
        f"[AUXILIARY_TEXT]\n{unit_text}"
    )
    image_path = unit.raw_content.get("image_path")
    return {
        "example_id": unit.unit_id,
        "dataset": "mmdocrag",
        "model_a_task": "vl_answer_bearing",
        "question_id": qa["question_id"],
        "question": qa["question"],
        "obligation_id": qa["obligations"][0]["obligation_id"],
        "obligation_text": qa["obligations"][0]["text"],
        "unit_id": unit.unit_id,
        "unit_content": unit_text,
        "unit_aux_text": unit_text,
        "modality": unit.modality.value,
        "source_id": unit.source_id,
        "quote_id": unit.locator.get("quote_id"),
        "split": unit.metadata.get("split"),
        "requires_visual_input": requires_visual,
        "image_path": image_path,
        "label_answer_bearing": label,
        "label_text": "Yes" if label else "No",
        "target_text": "Yes" if label else "No",
        "query_text": f"[QUESTION] {qa['question']}\n[OBLIGATION] {qa['obligations'][0]['text']}",
        "text_view": unit.views.text_view,
        "visual_view": unit.views.visual_view,
        "table_view": unit.views.table_view,
        "vl_prompt": prompt,
    }


def select_hard_negatives(question: str, positives: list[ObservationUnit], negatives: list[ObservationUnit], max_negatives: int) -> list[ObservationUnit]:
    positive_text = " ".join(
        str(unit.raw_content.get("text") or unit.raw_content.get("table_text") or unit.raw_content.get("caption") or "")
        for unit in positives
    )
    query = f"{question} {positive_text}"
    scored = []
    for unit in negatives:
        unit_text = str(unit.raw_content.get("text") or unit.raw_content.get("table_text") or unit.raw_content.get("caption") or "")
        scored.append((lexical_overlap(query, unit_text), unit.unit_id, unit))
    scored.sort(reverse=True)
    return [unit for _, _, unit in scored[:max_negatives]]


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    input_paths = find_json_inputs(args.input_dir, names=["train.jsonl", "dev_15.jsonl", "dev_20.jsonl", "evaluation_15.jsonl", "evaluation_20.jsonl"])
    if not input_paths:
        raise SystemExit(f"No MMDocRAG jsonl files found under {args.input_dir}")

    qa_rows: list[JSON] = []
    all_units: list[ObservationUnit] = []
    all_pairs: list[JSON] = []
    seen = 0
    for path in input_paths:
        for line_idx, record in enumerate(iter_jsonl(path)):
            if args.max_examples is not None and seen >= args.max_examples:
                break
            question, quotes, answer_text = (
                parse_structured_record(record) if record.get("question") else parse_message_record(record)
            )
            if not question or not quotes:
                continue
            raw_qid = str(first_nonempty(record.get("q_id"), record.get("id"), record.get("old_id"), line_idx))
            question_id = f"mmdocrag:{path.stem}:{raw_qid}"
            source_id = str(first_nonempty(record.get("doc_name"), question_id))
            split = four_way_split_for_key(
                source_id,
                train_ratio=args.train_ratio,
                calib_ratio=args.calib_ratio,
                dev_ratio=args.dev_ratio,
            )
            modalities = [str(m).lower() for m in record.get("evidence_modality_type") or []]
            obligation = make_obligation(question_id, question, modalities)

            units = [
                build_unit(
                    dataset="mmdocrag",
                    source_file=path.stem,
                    record=record,
                    question_id=raw_qid,
                    question=question,
                    quote=quote,
                    split=split,
                )
                for quote in quotes
            ]
            positives = [unit for unit in units if unit.label_answer_bearing == 1]
            negatives = [unit for unit in units if unit.label_answer_bearing == 0]
            selected_units = positives + select_hard_negatives(question, positives, negatives, args.max_negatives_per_question)
            qa = {
                "dataset": "mmdocrag",
                "question_id": question_id,
                "q_id": raw_qid,
                "old_id": record.get("old_id"),
                "source_file": path.name,
                "doc_name": source_id,
                "domain": record.get("domain"),
                "question_type": record.get("question_type"),
                "question": question,
                "answer_short_for_audit_only": first_nonempty(record.get("answer_short"), compact_answer(record), answer_text),
                "evidence_modality_type": modalities,
                "obligations": [obligation],
                "gold_quote_ids": [unit.locator.get("quote_id") for unit in positives],
                "gold_unit_ids": [unit.unit_id for unit in positives],
                "unit_ids": [unit.unit_id for unit in selected_units],
                "candidate_count_after_cleaning": len(units),
                "candidate_count_after_balancing": len(selected_units),
                "split": split,
            }
            qa_rows.append(qa)
            all_units.extend(selected_units)
            all_pairs.extend(make_pair(qa, unit) for unit in selected_units)
            seen += 1
        if args.max_examples is not None and seen >= args.max_examples:
            break

    by_split: dict[str, list[JSON]] = defaultdict(list)
    for pair in all_pairs:
        by_split[str(pair.get("split") or "train")].append(pair)

    write_jsonl(args.output_dir / "qa.jsonl", qa_rows)
    write_jsonl(args.output_dir / "units.jsonl", all_units)
    for split in ("train", "calib", "dev", "test"):
        write_jsonl(args.output_dir / f"{split}_pairs.jsonl", by_split.get(split, []))
    write_json(
        args.output_dir / "manifest.json",
        {
            "dataset": "mmdocrag",
            "input_paths": [str(path) for path in input_paths],
            "question_count": len(qa_rows),
            "unit_count": len(all_units),
            "pair_count": len(all_pairs),
            "positive_pair_count": sum(int(pair["label_answer_bearing"]) for pair in all_pairs),
            "splits": {split: len(rows) for split, rows in by_split.items()},
            "split_ratios": {
                "train": args.train_ratio,
                "calib": args.calib_ratio,
                "dev": args.dev_ratio,
                "test": max(0.0, 1.0 - args.train_ratio - args.calib_ratio - args.dev_ratio),
            },
            "notes": [
                "Gold labels use explicit gold_quotes when present; otherwise they are inferred from citations in the assistant answer.",
                "Hard negatives are selected by lexical overlap with the question and positive evidence text.",
            ],
        },
    )


if __name__ == "__main__":
    main()
