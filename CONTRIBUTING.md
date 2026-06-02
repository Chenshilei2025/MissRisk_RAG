# Contributing to MissRisk-RAG

MissRisk-RAG is a research engineering project. The central claim is not an
agent workflow; it is calibrated residual answer-miss risk under partial
multimodal observation.

When adding features, keep the question in view:

```text
Under the current observation state, could answer-bearing evidence still remain undetected?
```

## Branch Boundaries

Use the existing collaboration branches unless a maintainer creates a narrower
feature branch.

| Branch | Scope |
| --- | --- |
| `data/mmdocrag-pipeline` | MMDocRAG parsing, oracle units, gold quote mapping, hard negatives, source split |
| `data/slidevqa-multimodalqa` | SlideVQA and MultiModalQA conversion, supporting context mapping, dataset-specific schemas |
| `model/missrisk-estimator` | Model A/B/C training, inference APIs, calibration, model configs |
| `eval/policy-and-qa` | ObservationAction generation, greedy policy evaluation, QA/abstention evaluation |
| `paper/docs` | Roadmap, interfaces, experiment protocol, data card, paper figures and writing |

Keep changes close to the branch scope. Cross-cutting schema changes should be
small, documented, and covered by tests.

## Data Policy

Do not commit raw datasets, model checkpoints, generated embeddings, downloaded
PDFs, videos, or large output artifacts.

Use these directories locally:

```text
data_missrisk/raw/
data_missrisk/processed/
outputs/
```

Only `.gitkeep`, lightweight configs, schema docs, and small synthetic test
fixtures should be committed.

## Interface Contracts

Prefer extending the existing contracts instead of inventing parallel schemas:

- `ObservationUnit`: source unit that may contain evidence.
- `ObservationState`: channels already observed for that unit.
- `SearchObligation`: fact-facing requirement derived from the question.
- `RetrievedUnit`: ranked candidate from an external RAG pipeline.
- `ObservationAction`: proposed transition between observation states.
- `MissRiskEstimate`: answer-bearing, detectability, and joint miss-risk scores.

If a dataset or RAG repository needs extra fields, put them in `metadata`,
`locator`, `raw_content`, `visible_content`, or `retrieval_metadata` first.
Promote fields to the top level only when multiple modules need them.

## PR Checklist

Before opening or sharing a PR:

```bash
python -m pytest -q
```

Also check:

- The change follows the branch scope.
- New public interfaces have tests.
- Dataset outputs are not committed.
- The agentic layer remains an execution shell, not the research claim.
- Documentation is updated when schemas or workflows change.

## Implementation Priorities

The next high-value tasks are:

1. Build MissRiskBench data conversion, starting with MMDocRAG.
2. Generate ObservationAction candidates from missing observation channels.
3. Add typed observation tools for OCR, VLM inspection, table parsing, PDF page
   image reading, video frame sampling, and graph/source expansion.
4. Add Model A/B/C training and inference interfaces.
5. Add calibration and false-unanswerable evaluation.

## Style

- Keep modules small and explicit.
- Use Pydantic models for shared data contracts.
- Use `rg` for code search.
- Avoid unrelated refactors.
- Keep comments short and useful.
