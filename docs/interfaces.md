# Interface Boundaries

This document defines the shared contracts collaborators should use. The goal is
to make dataset conversion, external RAG adapters, observation tools, and models
interoperate without coupling everything to one framework.

## ObservationUnit

File: `agentic_mm_rag/observation/units.py`

Meaning: the smallest source unit that can be observed.

Examples:

- text quote
- PDF page block
- slide
- table block
- chart/image block
- OCR region
- video segment
- frame group
- graph relation

Use:

- `unit_id` for stable unit identity.
- `source_id` for the parent document/deck/video.
- `source_type` and `modality` for coarse type.
- `locator` for page id, block id, timestamp, row/column, or graph path.
- `raw_content` for text, OCR, caption, image path, table text, etc.
- `metadata` for dataset-specific fields.

## ObservationState

File: `agentic_mm_rag/observation/states.py`

Meaning: what the system has already observed for one unit.

Use:

- `state_id` for named states such as `caption_only` or `source_image_vlm`.
- `observed_channels` for booleans over text/OCR/caption/source image/VLM/table/video/graph.
- `quality` for OCR confidence, caption confidence, visual coverage, etc.
- `visible_content` for content available under the current state.
- `hidden_channels` for channels that could still be observed.

## SearchObligation

File: `agentic_mm_rag/observation/obligations.py`

Meaning: a fact-facing requirement that must be checked to answer the question.

The first planner may emit one broad obligation. Better planners should split
questions into multiple obligations only when the split changes retrieval or
observation behavior.

## RetrievedUnit

File: `agentic_mm_rag/retrieval/candidates.py`

Meaning: one ranked candidate returned by an external RAG pipeline.

Use it to preserve:

- `rank`
- `score`
- `retriever_name`
- `query_variant`
- `retrieval_metadata`

Do not lose retrieval provenance; it is needed for baselines and ablations.

## ExternalRAGAdapter

File: `agentic_mm_rag/adapters/base.py`

Meaning: the bridge from another multimodal RAG repository into MissRisk-RAG.

An adapter should:

1. Normalize repository-specific retrieval results into `ExternalRetrievedItem`.
2. Map each item into `ObservationUnit`.
3. Build the initial `ObservationState`.
4. Return a `RetrievalBatch`.

For simple dict outputs, use `GenericDictAdapter`.

## ObservationAction

File: `agentic_mm_rag/observation/policy.py`

Meaning: a possible transition from one observation state to another.

The policy score is:

```text
expected_risk_reduction / cost
```

The action generator should propose candidates; the greedy policy should choose
among them. Do not hide action choice inside free-form LLM reasoning.

## ObservationTool

File: `agentic_mm_rag/tools/base.py`

Meaning: a typed tool that performs an observation action.

Future concrete tools:

- `OCRTool`
- `VLMInspectionTool`
- `TableParserTool`
- `PDFPageImageTool`
- `VideoFrameSamplerTool`
- `GraphSourceExpanderTool`

Tools should return `ToolResult` with the next `ObservationState`.

## MissRiskEstimate

File: `agentic_mm_rag/observation/risk.py`

Meaning: model outputs for one question/obligation/unit/state bundle.

Fields:

- `p_answer_bearing`
- `p_detectable_given_bearing`
- `p_joint_miss`

The core model target is:

```text
P(B = 1, D = 0 | q, o, u, s)
```

`p_joint_miss` should be trained directly, not treated as an independent product
unless an ablation explicitly does so.

## Runtime Trace

Files:

- `agentic_mm_rag/runtime/events.py`
- `agentic_mm_rag/runtime/trace.py`

Meaning: observable execution record for debugging, calibration analysis, and
paper case studies.

Each run should record:

- risk estimates
- selected actions
- budget usage
- answer gate decision

## AnswerGate

File: `agentic_mm_rag/controller/answer_gate.py`

Meaning: final decision boundary after evidence support and residual miss risk.

Allowed decisions:

```text
answer
under_observed
abstain_true_unanswerable
```

The gate should remain simple and auditable. More complex learned gating should
be an ablation, not the default.

## Extension Rule

When adding a feature:

1. Reuse these interfaces.
2. Add fields to `metadata` first when the field is dataset-specific.
3. Add tests before depending on the interface from another module.
4. Update this document if the public contract changes.
