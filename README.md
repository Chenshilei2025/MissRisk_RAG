# MissRisk-RAG

Learning when answer-bearing evidence remains undetected in multimodal RAG.

This repository is organized around the final proposal in `MissRisk_RAG.md`.
The first implementation milestone is a lightweight, shared project skeleton for
building MissRiskBench, training answer-bearing/detectability/miss-risk models,
and evaluating observation policies.

## Project Layout

```text
data_missrisk/
  raw/
    mmdocrag/
    slidevqa/
    multimodalqa/
    tvqa/
    activitynet/
    qvhighlights/
  processed/
scripts/
  build_mmdocrag_units.py
  build_slidevqa_units.py
  build_multimodalqa_units.py
  generate_observation_states.py
  label_detectability.py
  train_answer_bearing.py
  train_missrisk.py
  eval_missrisk.py
agentic_mm_rag/
  adapters/
    base.py
    generic.py
    schemas.py
  controller/
    planner.py
    observer.py
    answer_gate.py
  memory/
    cache.py
    observation_trace.py
  observation/
    units.py
    states.py
    obligations.py
    risk.py
    policy.py
    report.py
  retrieval/
    candidates.py
  runtime/
    events.py
    session.py
    runner.py
    trace.py
  sources/
    base.py
  tools/
    base.py
    registry.py
configs/
docs/
tests/
outputs/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data Milestones

1. Convert MMDocRAG quotes into `ObservationUnit` records.
2. Generate controlled `ObservationState` variants for oracle units.
3. Build answer-bearing pairs and joint miss-risk training examples.
4. Evaluate calibration, hidden-answer recall, false-unanswerable reduction, and
   observation cost.

## Connecting External Multimodal RAG Repositories

MissRisk-RAG treats external RAG systems as producers of retrieved candidates.
Adapters convert repository-specific retrieval outputs into three neutral
contracts:

- `ObservationUnit`: what source unit may contain answer-bearing evidence.
- `ObservationState`: which channels the upstream system has already observed.
- `RetrievedUnit`: rank, score, retriever name, and retrieval metadata.

For simple dict-like outputs, use `GenericDictAdapter`:

```python
from agentic_mm_rag.adapters.generic import GenericDictAdapter
from agentic_mm_rag.retrieval import RetrievalQuery

adapter = GenericDictAdapter()
batch = adapter.convert(
    query=RetrievalQuery(question_id="q1", question="What value is shown?"),
    raw_items=[
        {
            "item_id": "doc1_page2_chart",
            "source_id": "doc1",
            "rank": 1,
            "score": 0.91,
            "retriever_name": "colpali",
            "content": {"caption": "A chart showing 42."},
            "locator": {"page_id": 2, "block_id": "chart"},
            "metadata": {"source_type": "document", "modality": "image"},
        }
    ],
).retrieval_batch
```

For a full repository integration, implement `ExternalRAGAdapter` and
`UnitMapper` in `agentic_mm_rag/adapters/base.py`. If the adapter can re-open
original files, implement `SourceStore` from `agentic_mm_rag/sources/base.py`.

## Agentic Execution Shell

The agentic layer is intentionally an execution shell, not the paper's central
claim. It coordinates observation actions around the miss-risk model:

```text
question
  -> obligations
  -> retrieved units
  -> observation states
  -> miss-risk estimates
  -> risk-reducing observation actions
  -> answer / abstain / under-observed
```

Key modules:

- `runtime/`: session budget, events, traces, and the small runner.
- `controller/`: obligation planning, action selection, and answer gating.
- `tools/`: typed observation tools and registry.
- `memory/`: observation result cache and trace store.

This keeps the engineering benefits of an agent runtime while preserving the
research story: calibrated residual answer-miss risk under partial observation.

## Branches

Suggested collaboration branches are created from the initial project skeleton:

- `data/mmdocrag-pipeline`
- `data/slidevqa-multimodalqa`
- `model/missrisk-estimator`
- `eval/policy-and-qa`
- `paper/docs`
