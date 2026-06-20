# Roadmap

This roadmap follows the proposal in `MissRisk_RAG.md`. The project should move
from a small pilot to a stronger KDD-ready system without letting the agentic
runtime become the main research story.

## Phase 0: Shared Contracts

Status: mostly complete.

- Define `ObservationUnit`, `ObservationState`, and `SearchObligation`.
- Define retrieval adapter contracts for external multimodal RAG repositories.
- Define `ObservationAction`, greedy policy, runtime trace, and answer gate.
- Keep branch boundaries and interface docs up to date.

## Phase 1: MMDocRAG Pilot

Branch: `data/mmdocrag-pipeline`

Goal: produce a 50-QA pilot split that can train and evaluate the first models.

Tasks:

- Parse MMDocRAG annotation files.
- Convert `text_quotes` and `img_quotes` into `ObservationUnit` records.
- Map `gold_quotes` to `label_answer_bearing = 1`.
- Generate hard negatives from non-gold candidate quotes.
- Split by `doc_name`, not random question.
- Write `units.jsonl`, `qa.jsonl`, and VL `answer_bearing_pairs.jsonl` with
  image/table/page inputs preserved instead of collapsed into caption text.

Deliverable:

```text
data_missrisk/processed/
  units.jsonl
  qa.jsonl
  answer_bearing_pairs.jsonl
```

## Phase 2: ObservationAction Generation

Branch: `eval/policy-and-qa`

Goal: automatically propose actions from partial observation states.

Tasks:

- Map missing channels to state transitions:
  - `caption_only -> source_image_vlm`
  - `ocr_only -> table_structure`
  - `sparse_frames -> dense_frames`
  - `graph_only -> graph_plus_source_chunks`
- Attach action cost and expected next state.
- Produce deterministic action candidates before model training.
- Add tests for each modality family.

Deliverable:

```text
agentic_mm_rag/controller/action_generator.py
```

## Phase 3: Detectability States

Branches: `data/mmdocrag-pipeline`, `data/slidevqa-multimodalqa`

Goal: construct controlled observation interventions for oracle units.

Tasks:

- Generate states such as `quote_text_only`, `image_description_only`,
  `source_image_vlm`, `table_flattened`, and `full_observation`.
- Create detectability examples with `label_detectable`.
- Keep labels independent of a single baseline retriever.
- Prepare a small human-check list.

Deliverable:

```text
data_missrisk/processed/detectability_states.jsonl
```

## Phase 4: Model A/B/C

Branch: `model/missrisk-estimator`

Goal: train and evaluate the three model heads.

Tasks:

- Model A: answer-bearing unit predictor.
- Train Model A as a general VL answer-bearing scorer that can inspect source
  pixels directly for image/table units.
- Model B: conditional detectability model.
- Model C: joint miss-risk estimator.
- Add inference API that returns `MissRiskEstimate`.
- Add calibration split and Brier/ECE reporting.

Deliverables:

```text
scripts/baselines/answer_bearing_lexical.py
scripts/train/model_c.py
agentic_mm_rag/models/
```

## Phase 5: Policy and QA Evaluation

Branch: `eval/policy-and-qa`

Goal: show that joint miss-risk greedy observation improves hidden-answer and
false-unanswerable behavior under fixed cost.

Baselines:

- Vanilla top-k RAG
- Multi-query RAG
- Adaptive top-k expansion
- Evidence sufficiency verifier
- Answer-bearing only policy
- Detectability only policy
- Joint miss-risk greedy
- Oracle policy

Metrics:

- Oracle answer unit recall
- Risk reduction per action
- False-unanswerable reduction
- True-unanswerable accuracy
- Unsupported answer rate
- Observation cost
- Brier score and ECE

## Phase 6: Dataset Expansion

Branch: `data/slidevqa-multimodalqa`

Goal: expand beyond MMDocRAG while preserving source-level split rules.

Tasks:

- Convert SlideVQA evidence slides into units.
- Convert MultiModalQA supporting contexts into units.
- Add 100-200 video evaluation cases only after document/slide results work.

## Phase 7: Paper Package

Branch: `paper/docs`

Goal: turn experiments into a clear MissRisk-RAG story.

Tasks:

- Write data card and experiment protocol.
- Add reliability diagrams and risk bucket tables.
- Add case studies from `ObservationTrace`.
- Keep wording centered on discoverability and residual miss risk.
