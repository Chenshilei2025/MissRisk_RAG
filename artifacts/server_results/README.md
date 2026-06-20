# Server Result Archive

This directory contains the small, handoff-friendly files copied from the AutoDL server on 2026-06-20.

It intentionally does not include large checkpoints. The server checkpoints remain under:

```text
/root/autodl-tmp/MissRisk_RAG/outputs/models/
```

Specific server weight paths:

```text
Model A:
/root/autodl-tmp/MissRisk_RAG/outputs/models/answer_bearing_vl_mix_v2/adapter_model.safetensors

Model B hard-v3:
/root/autodl-tmp/MissRisk_RAG/outputs/models/model_b_detectability_hard_v3_seed13/model_B.pt

Model C full + auxiliary heads:
/root/autodl-tmp/MissRisk_RAG/outputs/models/source_clean_ablation/missrisk_full_all_seed13/model_C.pt

Model C no auxiliary heads:
/root/autodl-tmp/MissRisk_RAG/outputs/models/source_clean_ablation/missrisk_full_no_aux_seed13/model_C.pt

Old shortcut-prone Model B, audit only:
/root/autodl-tmp/MissRisk_RAG/outputs/models/source_clean_ablation/detectability_full_seed13/model_B.pt
```

## Contents

- `summary.json`: compact summary of Model A/B/C results and audit findings.
- `model_a/`: Model A answer-bearing retrieval metrics and training audit/config.
- `model_b/`: Model B hard-v3 data audit and best dev predictions.
- `model_c/`: Model C source-clean metrics and test metrics.
- `audits/`: shortcut/source-clean audit reports.
- `logs/`: Model B hard-v3 training log.
- `data_reports/`: hard-v3 detectability data quality report.

## Important Caveats

- Old Model B source-clean detectability results are shortcut-prone and should not be used as a main claim.
- Model B hard-v3 has strong dev metrics, but the run did not write final `metrics.json`, `test_metrics.json`, or `temperature.json`; these should be regenerated from the saved server checkpoint.
- Model C should be retrained/evaluated after rebuilding C data with the hard-v3 detectability formulation.
