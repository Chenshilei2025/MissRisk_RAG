# MissRisk-RAG

> MissRisk-RAG 要估计多模态 RAG 在当前 observation state 下，答案证据是否仍然藏在未被有效观察的 source units 中。

也就是说，项目主线不是再做一个新的 Agentic RAG workflow，而是做：

```text
P(B_u = 1, D_u(s_t) = 0 | q, o, u, s_t)
```

其中：

- `B_u = 1`：source unit `u` 中存在 answer-bearing evidence；
- `D_u(s_t) = 1`：当前 observation state `s_t` 下该证据可被发现；
- `D_u(s_t) = 0`：当前 observation state 下该证据仍不可发现；
- 目标输出是 joint residual miss-risk，而不是 baseline retriever failure。

## 当前仓库一句话状态

现在仓库已经有：

```text
shared contracts + external RAG adapter interface + observation action generation
+ deterministic controller + runtime trace shell
```

现在仓库还没有：

```text
真实数据集转换实现、真实 MissRisk 模型、真实 OCR/VLM/table/video 工具、
真实外部多模态 RAG 仓库接入。
```

## 已经实现的部分

### 1. Observation Contracts

位置：`agentic_mm_rag/observation/`

已经定义：

- `ObservationUnit`
- `ObservationState`
- `SearchObligation`
- `MissRiskEstimate`
- `RiskFeatures`
- `ObservationAction`
- `MissRiskReport`

这些是后面数据转换、模型训练、runtime、policy evaluation 的共同接口。

### 2. Retrieval / External RAG Adapter Contracts

位置：`agentic_mm_rag/adapters/` 和 `agentic_mm_rag/retrieval/`

已经实现：

- `ExternalRetrievedItem`
- `MissRiskAdapterOutput`
- `RetrievalQuery`
- `RetrievedUnit`
- `RetrievalBatch`
- `ExternalRAGAdapter`
- `UnitMapper`
- `GenericDictAdapter`

`GenericDictAdapter` 可以把简单 dict 风格的 retrieved items 转成 MissRisk-RAG 的 neutral contracts。

示例：

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

注意：当前没有具体数据集 adapter 实现。等数据集下载后，再根据真实 schema 写转换逻辑。

### 3. Source Store Contract

位置：`agentic_mm_rag/sources/base.py`

已经定义：

- `ObservationPayload`
- `SourceStore`

这是给后续 PDF、slide、image、video、graph source reopen 用的接口。现在还没有具体 store 实现。

### 4. Observation Tool Contracts

位置：`agentic_mm_rag/tools/`

已经定义：

- `ObservationTool`
- `ToolResult`
- `ToolRegistry`

现在只有工具接口和 registry，没有具体 OCR、VLM、table parser、video frame sampler 工具。

### 5. ObservationAction 自动生成

位置：

- `agentic_mm_rag/observation/policy.py`
- `agentic_mm_rag/controller/action_generator.py`

已经实现规则版 action generator：

- image / multimodal unit 可生成 `run_ocr`、`inspect_source_image`
- table unit 可生成 `parse_table_structure`
- video segment / frame group 可生成 `sample_sparse_frames`、`sample_dense_frames`
- graph relation 可生成 `expand_graph_source_chunks`

这些 action 由 missing observation channels 推出，并带有：

- `cost`
- `from_state_id`
- `to_state_id`
- `current_risk`
- `expected_next_risk`

当前 `expected_next_risk` 是规则启发式。等 Model C 训练出来后，应改为用 miss-risk estimator 对 action 后的新 state 重新估计风险。

### 6. Planner / Controller / Answer Gate

位置：`agentic_mm_rag/controller/`

当前实现：

- `ObligationPlanner`：规则版，只生成一个 broad obligation；
- `ObservationController`：选择 `expected_risk_reduction / cost` 最大的 action；
- `AnswerGate`：根据 answer support 和 residual miss-risk 阈值输出：
  - `answer`
  - `under_observed`
  - `abstain_true_unanswerable`

重要：planner 和 controller 当前不需要配置具体 LLM/VLM。它们应保持简单、可审计，不作为论文创新点。

### 7. Runtime Shell

位置：`agentic_mm_rag/runtime/`

已经实现：

- `MissRiskSession`
- `Budget`
- `RuntimeEvent`
- `InMemoryEventSink`
- `ObservationTrace`
- `MissRiskRunner`

`MissRiskRunner` 接受两个外部 callable：

```python
RiskScorer = Callable[[str], MissRiskEstimate]
ActionProposer = Callable[[MissRiskEstimate], list[ObservationAction]]
```

所以 runtime 本身不绑定具体模型。后续训练出的 MissRisk estimator 可以作为 `risk_scorer` 注入。

## 仍然只是 Scaffold 的部分

### 1. 数据集转换脚本

位置：`scripts/`

当前只是占位：

- `build_mmdocrag_units.py`
- `build_slidevqa_units.py`
- `build_multimodalqa_units.py`

这些脚本还没有真实转换逻辑。不要假设它们已经支持 MMDocRAG、SlideVQA 或 MultiModalQA。

等真实数据下载后，需要按实际字段实现：

- `ObservationUnit` 转换；
- `label_answer_bearing` 标注；
- hard negatives；
- split key；
- QA metadata；
- source locator；
- initial `ObservationState`。

### 2. Observation State / Detectability 构造

当前只是占位：

- `generate_observation_states.py`
- `label_detectability.py`

还没有实现 controlled observation interventions，也没有 detectability label 生成。

### 3. 模型训练

当前只是占位：

- `train_answer_bearing.py`
- `train_missrisk.py`

还没有具体模型，也没有训练代码。

后续应实现：

- Model A：answer-bearing unit predictor；
- Model B：conditional detectability model；
- Model C：joint miss-risk estimator。

核心模型是 Model C：

```text
question + obligation + unit representation + observation state features
  -> p_joint_miss
```

### 4. Evaluation

当前只是占位：

- `eval_missrisk.py`

后续需要实现：

- oracle answer unit recall；
- AUROC；
- Brier score；
- expected calibration error；
- false-unanswerable rate；
- observation cost；
- policy risk reduction。

## 数据集计划

1. MMDocRAG：主数据，优先做 50 QA pilot；
2. SlideVQA：补充 slide/page observation；
3. MultiModalQA：扩大 text/table/image 规模；
4. TVQA / ActivityNet Captions / QVHighlights：后续小规模 video eval，不作为第一阶段主训练集。

当前仓库保留了 raw data 目录：

```text
data_missrisk/raw/
  mmdocrag/
  slidevqa/
  multimodalqa/
  tvqa/
  activitynet/
  qvhighlights/
```

但没有下载数据，也没有真实 dataset adapter。

## 模型配置原则

现在不需要给 planner 或 controller 配具体模型。

需要模型的地方是后续的：

- answer-bearing predictor；
- detectability model；
- joint miss-risk estimator；
- OCR / VLM / table parser / frame sampler 等 observation tools。

第一版 observation policy 不训练，保持确定性：

```text
a* = argmax_a ExpectedRiskReduction(a) / Cost(a)
```

这和 proposal 保持一致：主贡献应集中在 risk / discoverability modeling，而不是 agentic planning。

## 如何运行当前仓库

安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

测试：

```bash
pytest -q
```

当前测试覆盖：

- observation contracts；
- generic adapter contracts；
- runtime shell；
- tool registry；
- observation action generation。

## 下一步建议

优先级建议：

1. 下载 MMDocRAG，确认真实 annotation schema；
2. 实现 MMDocRAG pilot conversion；
3. 生成 `units.jsonl`、`qa.jsonl`、`answer_bearing_pairs.jsonl`；
4. 构造 controlled observation states；
5. 做 detectability label pipeline；
6. 再开始 Model A/B/C。

不要过早复杂化 planner、controller 或 agent runtime。当前工程壳已经足够支撑第一阶段数据和模型工作。

## 相关文档

- [MissRisk_RAG.md](MissRisk_RAG.md)：最终研究方案；
- [docs/interfaces.md](docs/interfaces.md)：接口边界；
- [docs/missriskbench_schema.md](docs/missriskbench_schema.md)：MissRiskBench JSONL schema 草案；
- [docs/roadmap.md](docs/roadmap.md)：阶段路线图。
