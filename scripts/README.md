# MissRisk-RAG Scripts

The scripts directory is organized as a small workflow toolbox rather than a flat pile of entrypoints.

```text
scripts/
  build/
    mmdocrag.py              # MMDocRAG -> Model A units/pairs
    multimodalqa.py          # MultiModalQA -> Model A units/pairs
    slidevqa.py              # SlideVQA -> Model A units/pairs
    observation_states.py    # Model A units -> Model B/C observation-state rows
    split_by_source.py       # shared source-level split rewrites

  train/
    model_a.py               # answer-bearing cross-encoder
    model_b.py               # conditional detectability model
    model_c.py               # joint miss-risk model

  eval/
    predictions.py           # generic prediction metrics
    audit_raw_mmdocrag.py    # raw MMDocRAG parse audit
    audit_submission.py      # source leakage + shortcut baseline audit

  run/
    week1_4_mmdocrag.py      # strict proposal Week 1-4 command generator
    ablation_grid.py         # shortcut / auxiliary-loss ablation command generator

  baselines/
    answer_bearing_lexical.py

  lib/
    common.py
    missrisk_common.py
    missrisk_models.py
```

Prefer invoking scripts from the repository root, for example:

```bash
python scripts/run/week1_4_mmdocrag.py --encoder /path/to/encoder
python scripts/train/model_c.py --help
```
