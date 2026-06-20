# MissRisk-RAG: KDD 研究计划

**Title (primary):** MissRisk-RAG: Calibrated Estimation of Undetected Answer Evidence in Partially-Observed Multimodal Corpora

**Title (alt):** Know What You Haven't Observed: Calibrated Answer-Miss Risk for Multimodal Retrieval

**目标会议:** ACM SIGKDD（KDD 2027）
**主投 Track:** Research
**版本:** KDD Proposal v1

---

## 0. 一句话定位（KDD 母题）

在只被**部分观察**的大规模多模态语料中，估计"答案知识存在但尚未被发现"的概率，并对该风险做**校准**，进而用  risk-aware 的方式决定继续观察、回答还是拒答。

> We estimate, and calibrate, the probability that answer-bearing evidence exists in a multimodal corpus but remains **undetected** under the system's current partial observation state.

这不是 RAG 的 accuracy 提升，而是一个 **partially-observed multimodal data 上的校准风险估计 + 知识发现** 问题——正对 KDD 的 Foundations（概率/校准）、Modern AI & Big Data（多模态/LLM）、Trustworthy & Responsible（reliability / abstention）三个 scope。

---

## 1. Motivation：为什么这是一个 KDD 问题

普通多模态 RAG 的逻辑是 `retrieve top-k → reason → answer`。它只能回答"我找到了什么"，不能回答"我还有哪些地方没有有效观察、答案是否可能被漏掉"。

在多模态语料里，"找不到"高度 ambiguous：
- 视频答案藏在未抽帧或相邻时段；
- 文档答案藏在表格、图表、脚注、OCR 区域；
- 图片 caption 不含答案，但源图里有；
- graph 抽取漏了一条 relation，不代表关系不存在；
- 文本检索失败不代表视觉证据不存在。

因此系统在 abstain 时，**分不清"corpus 里真没有"与"有但当前观察状态下不可发现"**。这正是一个知识发现问题：在部分观察的数据上估计**潜在但未被发现的知识**，并量化不确定性。KDD 关心的"知道自己不知道、可校准、可问责、cost-aware"全部落在这里。

**与 accuracy-driven RAG 的区别（一句话给 reviewer）：** 我们把"答案是否仍藏在未观察的 source units 中"当作一个可学习、可校准的预测目标，而不是把它隐含在 end-to-end 的回答正确率里。

---

## 2. Problem Formulation

给定 question `q`、search obligations `o`、source units `u`、当前观察状态 `s_t`，估计 answer-bearing evidence 是否仍未被发现：

```
R_t(q, o, u) = P(B_u = 1, D_u(s_t) = 0 | q, o, u, s_t)
```

- `B_u = 1`：unit `u` 含 answer-bearing evidence；
- `D_u(s_t) = 1`：当前观察状态 `s_t` 下该证据**可被发现**；`D_u(s_t)=0` 即仍不可发现；
- `R_t`：residual miss risk（条件联合概率，**非**独立假设）。

链式法则解读：`R_t = P(B=1 | q,o,u,s) · P(D=0 | B=1,q,o,u,s)`；实现上直接训练 joint head。

**三个核心对象（data model）：**

- **Observation Unit**：corpus 中可被观察的最小 source 单元（text quote / paragraph / page / slide / table block / chart-image block / OCR region / video segment / frame group / graph edge-path）。带 `unit_id`、`source_type`、`modality`、`locator`、`raw_content`。
- **Observation State**：当前系统对某 unit 看到了什么（如 `{text:false, ocr:true, caption:true, source_image:false, vlm_inspection:false, table_structure:false}` + per-channel quality 分数）。离散状态包括 `text_only / ocr_only / caption_only / table_flattened / source_image_vlm / sparse_frames / dense_frames / graph_only / full_observation`。
- **Search Obligation**：为回答问题必须核查的事实面向，带 `required_modalities`（如 `["transcript","video_segment","graph_relation"]`）。

---

## 3. Method

三个组件，**Model C（校准的 joint miss-risk）是主贡献**，A/B 作为辅助头与监督来源。

### 3.1 Model A — Answer-Bearing Unit Predictor

目标 `P(B_u=1 | q,o,u)`：unit 是否含能回答/支持/反驳/改变 obligation 的证据（注意：relevant ≠ answer-bearing）。

- 输入：`[QUESTION] q [OBLIGATION] o [UNIT] text/OCR/caption/image-desc/table-text/graph-relation/metadata`
- 标签：`1 = oracle answer-bearing unit`，`0 = hard negative`
- 正样本：MMDocRAG `gold_quotes`、SlideVQA evidence slide、MultiModalQA supporting context、TVQA localized moment
- 负样本（hard）：candidate quotes 中非 gold、同文档相邻页、同 deck 非 evidence slide、dense top-k 但不支持、same-entity wrong relation、same-table wrong row/column
- 模型：DeBERTa-v3 / BGE / Jina reranker / Qwen-Llama LoRA reranker（cross-encoder）
- Loss：`BCE(y_bear, p_bear)` 或 pairwise ranking

### 3.2 Model B — Conditional Detectability Model

目标 `P(D_u(s)=1 | B_u=1, q, o, u, s)`：若 unit 确含答案，在观察状态 `s` 下系统能否发现它。

- **关键设计——controlled observation interventions（可规模化）**：对每个 oracle answer unit 程序化构造多种 observation state，再判断该 state 是否足以恢复 oracle answer / claim。
- 标签生成：exact/numeric match → entailment model → LLM judge → VLM judge → 少量人工抽查校验。
- 为什么更稳：不依赖某 baseline 的历史失败、不过拟合某检索策略、学的是 observation state 本身是否足够、可测 held-out state/modality 泛化。

### 3.3 Model C — Joint Miss-Risk Estimator（主模型）

```
R_t(q,o,u,s_t) = P(B_u=1, D_u(s_t)=0 | q,o,u,s_t)
```

- 输入：question + obligation + unit representation + observation-state features
- 训练标签：`y_miss = 1 iff B=1 and D=0`
- 多任务 loss：`L = L_miss + λ_b · L_bear + λ_d · L_detect`
  - `L_miss` 主任务；`L_bear` 辅助学 answer-bearing；`L_detect` 辅助学 state detectability
- 好处：不需独立假设；不把某 baseline 失败当真值标签。
- **KDD 重点：output 必须校准。** 配合 calibration split + temperature/isotonic 校准，主图为 reliability diagram。

### 3.4 Evidence Acquisition Policy（确定性）

第一版用确定性贪心：

```
a* = argmax_a ExpectedRiskReduction(a) / Cost(a)
ExpectedRiskReduction(a) = R_t - E[R_{t+1} | action a]
```

动作改变观察状态：`ocr_only → source_image_vlm`、`sparse_frames → dense_frames`、`caption_only → full_observation`、`graph_only → graph_plus_source_chunks`。用 Model C 重新估计 action 后风险，选单位成本下风险下降最大的动作；据此决定继续观察 / 回答 / 拒答。

---

## 4. MissRiskBench（一等贡献）

> MissRiskBench: a benchmark for answer-bearing evidence missed under partial multimodal observation.

不从零标注，而是把已有带 evidence 的数据集转成 MissRisk 格式，**强调可规模化的标注流水线**。

### 4.1 为什么不能直接用普通 QA 数据集
普通 QA 只给 `question, answer`（最多 supporting context）。MissRisk 需要：oracle answer units、hard negative units、observation states、detectability labels、hidden-answer cases、false-unanswerable cases、true-unanswerable cases。没有这些就无法证明模型真在估计"答案是否仍未被发现"。

### 4.2 数据来源
- **Tier 1 主数据 — MMDocRAG**（multimodal document RAG，4,055 expert QA pairs；`gold_quotes` 直接当 oracle units，non-gold candidate 当 hard negatives；含 long PDF 与 image quote 文件）。
- **Tier 1 补强 — SlideVQA**（slide deck 天然分 page/slide units，含 evidence slide selection，图/表/文本/布局齐全，极适合构造 observation state）。
- **Tier 2 — MultiModalQA**（text/table/image，~29,918 examples，扩 answer-bearing predictor 训练规模）。
- **Tier 3 视频小规模扩展（仅 eval，100–200 例）**：TVQA / ActivityNet Captions / QVHighlights，提升多模态完整性，不拖慢主线。

### 4.3 标准样本格式（要点）
每条含 `question_id / question / answer / source / obligations[] / units[]`，每个 unit 标 `label_answer_bearing`；外加 detectability 样本（`obligation_id / unit_id / state_id / visible_content / hidden_channels / oracle_answer / label_answer_bearing / label_detectable / label_joint_miss`）。

### 4.4 四类 answerability split（核心区分）
- **answerable_easy (25%)**：普通 RAG 可解。
- **answerable_hidden (35%)**：答案存在，但 relevance top-k / text-only RAG 难找到（gold 不在 dense top-k；gold 是 image/table/chart 而 query 更易召回 text；需 source image/VLM；gold 在相邻 page/segment；answer-bearing 与 high-overlap negative 极像）。
- **false_unanswerable (20%)**：baseline 输出 not answerable，但 oracle answer unit 存在——**MissRisk 最该打赢的类别**。
- **true_unanswerable (20%)**：corpus 中确实没有，系统应拒答（entity/relation/numeric swap、ask absent visual detail、cross-doc impossible；**必须人工抽查，不能只靠 LLM 生成**）。

### 4.5 Split 规则
按 **source split**，绝不按 question 随机：train 70% / calibration 10% / dev 10% / test 10%；MMDocRAG 按 `doc_name`、SlideVQA 按 deck、视频按 video。否则泄漏 source-specific layout/entity。

### 4.6 规模建议（KDD 用 Strong 版交稿）
- MMDocRAG 2k QA + SlideVQA 1k QA + MultiModalQA 1k QA + Video eval 200 QA（Total 4k+）。
- 派生：Answer-bearing pairs 80k–200k；Detectability states 30k–100k；Human-checked evaluation 500–800 cases。
- （Minimum 版 700 QA 仅用于内部 pilot，不建议作为 KDD 投稿规模。）

---

## 5. Experiments（按 KDD 优先级重排）

KDD reviewer 看的顺序：calibration / reliability 优先，accuracy 最后。

### E1 — Answer-Bearing Retrieval
*Model A 是否比 relevance retriever 更会找 oracle answer units?*
对比 BM25 / dense / existing reranker / answer-bearing predictor。
指标：Oracle Answer Unit Recall@k、MRR、Recall under fixed budget、hard-negative accuracy。

### E2 — Detectability Prediction（主图之一）
*Conditional detectability 能否预测不同 observation state 下答案是否可发现?*
训练 oracle units + controlled states；测 held-out documents / states / modalities。
指标：AUROC、Brier、ECE、state-wise accuracy、cross-state generalization。

### E3 — Joint Miss-Risk Calibration（一号实验）
*joint miss-risk score 是否校准?*
指标：reliability diagram、ECE、bucketed miss frequency、Brier。
关键图：predicted risk bucket `0.0–0.1 → true miss ≈ 0.1`、`0.8–0.9 → true miss ≈ 0.8`。

### E4 — Acquisition Policy
*用 joint miss-risk 做贪心观察，是否比 relevance/adaptive 更有效?*
对比 top-k relevance / adaptive top-k / uncertainty-based / answer-bearing only / detectability only / joint-miss greedy / oracle greedy。
指标：risk reduction per action、oracle answer unit recall vs cost、false-unanswerable reduction vs cost。

### E5 — Downstream Abstention QA（最后）
指标：Answer Accuracy、False-Unanswerable Rate、True-Unanswerable Accuracy、Abstention F1、Unsupported Answer Rate、Observation Cost、Latency / tool calls。
**对比 selective-prediction / learn-to-abstain baseline**，呼应"知道自己不知道"。

---

## 6. Baselines

1. Vanilla top-k RAG
2. Multi-query RAG
3. Adaptive top-k expansion
4. Evidence sufficiency verifier（如 SURE 类）
5. Selective prediction / learn-to-abstain
6. Answer-bearing only policy
7. Detectability only policy
8. Joint miss-risk **without** calibration
9. Joint miss-risk **with** calibration
10. Oracle observation policy

目的：证明这不是单纯 reranker、不是多检索、不是 sufficiency verifier、不是工程 pipeline，而是 **calibration + observation-state modeling 真的有贡献**。

---

## 7. Related Work（按 KDD 邻居重写）

不再以 RAG 文献为主轴，先连 KDD 母语，再谈 RAG：

- **Selective prediction / learning to abstain / learn-to-defer**：我们估计 miss-risk 以决定 answer/abstain，但区分"真无答案 vs 未观察到"。
- **Uncertainty quantification & calibration**：主贡献是多模态观察状态下 answer-miss risk 的 practical calibration，不声称理论 coverage（区别于 conformal / certified RAG）。
- **Active learning / cost-aware acquisition**：observation policy 本质是 cost-aware active evidence acquisition。
- **Cost-sensitive learning / novelty & coverage estimation**：估计"潜在未发现知识"与 coverage/novelty 检测相邻。
- **RAG 线（精简）**：SURE-RAG（retrieved evidence 是否足够）vs 本文（未被观察的 source units 是否仍藏答案）；MARA（动态扩 evidence）vs 本文（建模 observation state 下证据是否仍不可发现）；Stateful evidence-driven RAG（evidence 累积）vs 本文（unobserved-space risk estimation）；Counter-evidence RAG（找反证）vs 本文（估计当前观察下证据是否仍未发现）。

---

## 9. Timeline

| 阶段 | 任务 |
|---|---|---|
| M1 | 数据流水线 | 解析 MMDocRAG/SlideVQA → ObservationUnit；构造 hard negatives；50 QA pilot |
| M2 | Answer-Bearing | 训/调 cross-encoder；对比 BM25/dense；报 Recall@k；选出 hidden-answer 样本 |
| M3 | Detectability | 对 oracle units 构造 controlled states；exact/NLI/LLM/VLM 生成标签 + 人工抽查 100–200；训 Model B |
| M4 | Joint Miss-Risk | 训 joint head；calibration split；reliability diagram；risk-bucket 分析 |
| M5 | Policy + 扩数据 | 实现 cost-aware greedy；跑 downstream；加 MultiModalQA + 视频 eval；整理 500–800 人工精评 |
| M6 | 论文初稿 | intro/method/experiments；主图；KDD 版 related work |
| M7 | 补实验 + 提交 | ablation、case study、limitation、artifacts/data card、终稿 |
---

## 10. 成功标准

**Pilot 阶段：** Oracle Answer Unit Recall@10 比 dense 高 ≥10%；false-unanswerable rate 降 ≥15%；joint miss-risk 对 missed oracle units 有明显排序能力；calibration curve 单调；hidden-answer split 上明显优于 adaptive top-k。

**正式实验：** hidden-answer recall 提升；false-unanswerable 显著降低；true-unanswerable 不明显变差；unsupported answer rate 不上升；同等 cost 下 oracle answer unit recall 更高；risk calibration（ECE/Brier）优于 uncertainty baseline。

---

## 11. KDD 版 Abstract 草稿

Multimodal retrieval-augmented generation systems often justify answers with retrieved evidence, but rarely estimate whether answer-bearing evidence remains **undetected** in source units that were only partially observed. A failure to answer may reflect the absence of evidence, or merely insufficient observation of relevant tables, figures, OCR regions, video segments, or graph relations. We formulate **residual answer-miss risk** as the calibrated probability that an answer-bearing source unit remains undetected under the current multimodal observation state. We construct controlled observation interventions over oracle answer units and train a joint miss-risk estimator with auxiliary answer-bearing and detectability heads, then derive a calibrated risk score. A deterministic, cost-aware acquisition policy selects observation actions by expected miss-risk reduction per cost, letting the system answer, abstain, or declare under-observation. We build **MissRiskBench** from multimodal document and video QA with oracle units, hard negatives, controlled observation states, and hidden-answer / unanswerable splits. Experiments evaluate answer-unit recall, detectability and miss-risk calibration, false-unanswerable reduction, and downstream abstention—showing that observation-state modeling and calibration, not retrieval scale, drive the gains.

---

## 12. 参考链接

- MMDocRAG: https://huggingface.co/datasets/MMDocIR/MMDocRAG ・ paper https://arxiv.org/abs/2505.16470
- SlideVQA: https://github.com/nttmdlab-nlp/SlideVQA ・ paper https://arxiv.org/abs/2301.04883
- MultiModalQA: https://github.com/allenai/multimodalqa ・ paper https://arxiv.org/abs/2104.06039
- TVQA: https://arxiv.org/abs/1809.01696 ・ QVHighlights: https://arxiv.org/abs/2107.09609 ・ ActivityNet Captions: https://cs.stanford.edu/people/ranjaykrishna/densevid/
- KDD 2026 CFP（节奏参考）: https://kdd2026.kdd.org/research-track-call-for-papers/