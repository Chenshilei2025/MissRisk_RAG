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

## External Retrieval Candidate

Adapters should preserve retrieval provenance when converting external RAG
outputs:

```json
{
  "unit": {
    "unit_id": "doc1_page2_chart",
    "source_id": "doc1",
    "source_type": "document",
    "modality": "image",
    "locator": {"page_id": 2, "block_id": "chart"},
    "raw_content": {"caption": "A chart showing 42."}
  },
  "initial_state": {
    "unit_id": "doc1_page2_chart",
    "state_id": "retrieved_content",
    "observed_channels": {"caption": true},
    "visible_content": {"caption": "A chart showing 42."}
  },
  "rank": 1,
  "score": 0.91,
  "retriever_name": "colpali",
  "query_variant": null,
  "retrieval_metadata": {"source_type": "document", "modality": "image"}
}
```

The adapter layer should not decide final answerability. It only records what
the upstream RAG system retrieved and what it already observed. MissRisk models
then estimate residual answer-miss risk over these units and states.
