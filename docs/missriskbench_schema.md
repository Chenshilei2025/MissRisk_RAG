# MissRiskBench Schema

This document records the shared JSONL contracts used by the project.

## ObservationUnit

Each line in `units.jsonl` should map to `ObservationUnit`:

```json
{
  "unit_id": "docA_page12_image3",
  "source_id": "docA",
  "source_type": "document",
  "modality": "image",
  "locator": {"page_id": 12, "layout_id": 3},
  "raw_content": {"ocr": "...", "caption": "...", "image_path": "..."},
  "metadata": {"dataset": "mmdocrag"},
  "label_answer_bearing": 1
}
```

## Detectability Example

```json
{
  "question_id": "mmdoc_0001",
  "obligation_id": "o1",
  "unit_id": "image3",
  "state_id": "image_description_only",
  "visible_content": {"caption": "..."},
  "hidden_channels": ["source_image", "vlm_inspection"],
  "oracle_answer": "...",
  "label_answer_bearing": 1,
  "label_detectable": 0,
  "label_joint_miss": 1
}
```

## Split Rule

Split by source, not by random question:

- MMDocRAG: `doc_name`
- SlideVQA: deck id
- MultiModalQA: source item id
- Video datasets: video id
