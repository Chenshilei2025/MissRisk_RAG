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
  observation/
    units.py
    states.py
    obligations.py
    risk.py
    policy.py
    report.py
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

## Branches

Suggested collaboration branches are created from the initial project skeleton:

- `data/mmdocrag-pipeline`
- `data/slidevqa-multimodalqa`
- `model/missrisk-estimator`
- `eval/policy-and-qa`
- `paper/docs`
