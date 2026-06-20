# MissRisk-RAG

MissRisk-RAG studies a failure mode in multimodal RAG: when the system cannot answer, it often cannot tell whether the corpus truly lacks evidence or whether answer-bearing evidence is still hidden in partially observed source units.

The central prediction target is:

```text
R_t(q, o, u, s_t) = P(B_u = 1, D_u(s_t) = 0 | q, o, u, s_t)
```

- `B_u = 1`: source unit `u` contains answer-bearing evidence.
- `D_u(s_t) = 1`: under the current observation state `s_t`, that evidence is discoverable.
- `R_t`: residual answer-miss risk.

This repository now contains both the engineering shell and server-side pilot results for Model A/B/C. Key server results have been copied into `artifacts/server_results/` for handoff.

## Current Handoff Status

The project has moved past the initial contract-only phase. The server has already produced pilot training/evaluation results on MissRiskBench-style processed data derived from MMDocRAG and MultiModalQA.

Local result archive:

```text
artifacts/server_results/
  summary.json
  model_a/
  model_b/
  model_c/
  audits/
  logs/
  data_reports/
```

Large model weights were not copied locally by default. The important metrics, logs, audits, and prediction files were copied. The large checkpoints remain on the AutoDL server under:

```text
/root/autodl-tmp/MissRisk_RAG/outputs/models/
```

Specific trained-weight locations on the server:

```text
# Model A: Qwen2.5-VL LoRA answer-bearing scorer
/root/autodl-tmp/MissRisk_RAG/outputs/models/answer_bearing_vl_mix_v2/
/root/autodl-tmp/MissRisk_RAG/outputs/models/answer_bearing_vl_mix_v2/adapter_model.safetensors

# Model B: repaired hard-v3 detectability model
/root/autodl-tmp/MissRisk_RAG/outputs/models/model_b_detectability_hard_v3_seed13/
/root/autodl-tmp/MissRisk_RAG/outputs/models/model_b_detectability_hard_v3_seed13/model_B.pt

# Model C: source-clean joint miss-risk models
/root/autodl-tmp/MissRisk_RAG/outputs/models/source_clean_ablation/missrisk_full_all_seed13/
/root/autodl-tmp/MissRisk_RAG/outputs/models/source_clean_ablation/missrisk_full_all_seed13/model_C.pt
/root/autodl-tmp/MissRisk_RAG/outputs/models/source_clean_ablation/missrisk_full_no_aux_seed13/
/root/autodl-tmp/MissRisk_RAG/outputs/models/source_clean_ablation/missrisk_full_no_aux_seed13/model_C.pt

# Old shortcut-prone Model B, kept only for audit/comparison
/root/autodl-tmp/MissRisk_RAG/outputs/models/source_clean_ablation/detectability_full_seed13/model_B.pt
```

## Data Used

The experiments were not trained directly on raw MMDocRAG/MultiModalQA files. They used MissRiskBench-style processed splits with:

- `ObservationUnit`
- `ObservationState`
- `label_answer_bearing`
- `label_detectable`
- `label_joint_miss`
- source-clean train/dev/test splits for the more reliable B/C runs

Important processed datasets on the server:

```text
data_missrisk/processed/model_a_training_mix_normalized/
data_missrisk/processed/model_bc_training_mix_source_clean/
data_missrisk/processed/model_b_detectability_hard_v3/
```

The local archive includes data audits, not the full training datasets.

## Model A: Answer-Bearing Unit Predictor

Purpose: predict whether a source unit contains answer-bearing evidence, not merely lexical relevance.

Best pilot model:

```text
outputs/models/answer_bearing_vl_mix_v2/
```

Base model:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

LoRA configuration:

```text
r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = q_proj, k_proj, v_proj, o_proj
train_batch_size = 1
num_train_epochs = 1
global_step = 2926
```

Data size:

```text
train examples = 23,402
dev examples   = 5,044
train labels   = 11,702 positive / 11,700 negative
dev labels     = 2,523 positive / 2,521 negative
```

E1 result file:

```text
artifacts/server_results/model_a/model_a_experiment1_answer_bearing_retrieval.json
```

E1 summary:

| System | Accuracy | F1 | Hard-neg acc | MRR | Any R@1 | Unit R@1 | Unit R@5 | Unit R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| token overlap | 0.6350 | 0.6364 | 0.6315 | 0.8627 | 0.7532 | 0.4645 | 0.9453 | 0.9972 |
| TF-IDF cosine | 0.6907 | 0.6908 | 0.6906 | 0.8945 | 0.8091 | 0.4990 | 0.9580 | 0.9972 |
| BM25 lexical | 0.6943 | 0.6944 | 0.6942 | 0.8984 | 0.8201 | 0.5057 | 0.9532 | 0.9976 |
| Model A mix v2 | 0.8265 | 0.8258 | 0.8310 | 0.9702 | 0.9434 | 0.5818 | 0.9734 | 0.9988 |

Conclusion: Model A is a good pilot result. It clearly improves classification and ranking over lexical baselines. The biggest ranking gains are at top-1 / MRR; Recall@10 is saturated for almost all systems.

## Model B: Conditional Detectability

Purpose: estimate whether an oracle answer-bearing unit is discoverable under a given observation state.

Base model used:

```text
BAAI/bge-reranker-base
local path on server:
/root/autodl-tmp/local_hf_models/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70
```

Training configuration:

```text
batch_size = 16
epochs = 3
learning_rate = 2e-5
max_len = 384
dropout = 0.1
weight_decay = 0.01
warmup_ratio = 0.06
input_profile = full
state features = enabled, dim 37
```

Important caveat: the first source-clean Model B dataset had a shortcut. A `(modality, state_id)` rule could predict detectability with AUROC 1.0. Those old Model B results should not be used as substantive evidence.

To fix this, a harder `hard_v3` detectability dataset was generated:

```text
data_missrisk/processed/model_b_detectability_hard_v3/
```

Hard-v3 idea:

- discovery states expose observed content/inspection text;
- same state/modality contains both positive and counterfactual negative examples;
- no model-visible counterfactual marker is inserted;
- the model must compare question/obligation against visible observation content.

Hard-v3 data size:

```text
train rows = 79,532
dev rows   = 12,786
train positives = 27,200
dev positives   = 4,316
```

Result files:

```text
artifacts/server_results/model_b/model_b_hard_v3_data_audit.json
artifacts/server_results/model_b/model_b_hard_v3_dev_predictions.best.jsonl
artifacts/server_results/logs/model_b_hard_v3_seed13_20260619_231658.log
```

Hard-v3 dev results from log:

| Epoch | Train loss | Accuracy | AUROC | F1 | Brier | ECE mass | Precision | Recall | Specificity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2040 | 0.9574 | 0.9948 | 0.9389 | 0.0282 | 0.0061 | 0.9102 | 0.9694 | 0.9512 |
| 1 | 0.0903 | 0.9593 | 0.9949 | 0.9418 | 0.0272 | 0.0044 | 0.9087 | 0.9775 | 0.9499 |

Training saved the best checkpoint and best dev predictions on the server:

```text
outputs/models/model_b_detectability_hard_v3_seed13/model_B.pt
outputs/models/model_b_detectability_hard_v3_seed13/dev_predictions.best.jsonl
```

However, `metrics.json` and `temperature.json` were not written for this run because the process stopped after the epoch-2 training tail before final eval/calibration. This should be fixed by rerunning final evaluation/calibration from the saved `model_B.pt`.

Conclusion: the hard-v3 Model B result is promising and much healthier than the old shortcut-prone result. It still needs test evaluation and calibration artifacts before being treated as final.

## Model C: Joint Miss-Risk Estimator

Purpose: estimate:

```text
P(B_u = 1, D_u(s_t) = 0 | q, o, u, s_t)
```

Base model:

```text
BAAI/bge-reranker-base
```

Main source-clean configuration:

```text
batch_size = 16
epochs = 3
learning_rate = 2e-5
max_len = 384
dropout = 0.1
weight_decay = 0.01
input_profile = full
state features = enabled
lambda_b = 0.3
lambda_d = 0.5
```

Main test result:

```text
artifacts/server_results/model_c/missrisk_full_all_seed13_test_metrics.json
```

| Model C variant | Miss AUROC | Accuracy | F1 | Brier | ECE mass | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full + auxiliary heads | 0.9585 | 0.8900 | 0.8076 | 0.0802 | 0.0619 | 0.6883 | 0.9768 | 0.8632 |
| no auxiliary heads | 0.9625 | 0.8947 | 0.8135 | 0.0750 | 0.0546 | 0.6992 | 0.9724 | 0.8707 |

Caveat: the no-aux variant has slightly better miss-risk metrics, but its auxiliary bear/detect heads are intentionally disabled and not meaningful. The full model is more interpretable, but the no-aux result shows that the direct joint miss head is strong.

Important audit result:

```text
artifacts/server_results/audits/source_clean_submission_audit.json
```

The old detectability labels were shortcut-prone:

```text
detectability modality+state rule AUROC = 1.0
detectability state rule AUROC          = 0.9866
```

For miss-risk itself, state-only baselines were weaker:

```text
missrisk test state rule AUROC          = 0.8318
missrisk test modality+state AUROC      = 0.8415
```

Conclusion: Model C has a strong pilot signal, but the next owner should retrain/evaluate C against the repaired hard-v3 detectability design or an equivalent shortcut-resistant MissRiskBench version.

## What Is Implemented In Code

Core contracts:

```text
agentic_mm_rag/observation/
agentic_mm_rag/adapters/
agentic_mm_rag/retrieval/
agentic_mm_rag/sources/
agentic_mm_rag/tools/
agentic_mm_rag/controller/
agentic_mm_rag/runtime/
```

Scripts:

```text
scripts/build/
scripts/train/model_a.py
scripts/train/model_b.py
scripts/train/model_c.py
scripts/eval/
scripts/run/
scripts/baselines/
```

The engineering shell includes observation units/states, adapter contracts, source-store/tool contracts, action generation, controller logic, and runtime traces. The core contribution should remain the risk/discoverability modeling rather than agentic planning.

## Recommended Next Steps

1. Re-run Model B hard-v3 final eval/calibration from the saved server checkpoint and write:

```text
metrics.json
test_metrics.json
temperature.json
```

2. Rebuild Model C training data using the shortcut-resistant hard-v3 detectability formulation.

3. Retrain Model C and compare:

```text
direct joint miss
full auxiliary heads
product p_bear * (1 - p_detect)
state-only / modality-state baselines
```

4. Add calibration plots:

```text
reliability diagram
ECE / Brier
state-wise calibration
modality-wise calibration
hidden-answer vs true-unanswerable calibration
```

5. Implement cost-aware evidence acquisition evaluation:

```text
risk reduction per action
oracle answer unit recall vs cost
false-unanswerable reduction vs cost
```

6. Do not use the old Model B full/source-clean result as a main claim because of the state shortcut.

## How To Run Locally

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

The local machine may not contain the raw datasets or server checkpoints. Use `artifacts/server_results/summary.json` for a compact handoff summary.

## Related Documents

- [MissRisk_RAG.md](MissRisk_RAG.md): research plan.
- [docs/missriskbench_schema.md](docs/missriskbench_schema.md): MissRiskBench JSONL schema draft.
- [docs/roadmap.md](docs/roadmap.md): development roadmap.
- [artifacts/server_results/summary.json](artifacts/server_results/summary.json): copied server-result summary.
