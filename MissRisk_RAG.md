# MissRisk-RAG 最终优化方案

**副标题： Learning When Answer-Bearing Evidence Remains Undetected in Multimodal RAG**

日期： 2026-06-01  
目标： AAAI-27 / 多模态 RAG 方向研究计划  
版本： Final Proposal v3

---

## 0. 一句话结论

本文不再主打 `Agentic RAG`。Agent 只是工程实现方式，不是创新主线。

最终建议主线是：

> **MissRisk-RAG**：在多模态 RAG 中，系统不仅要判断已检索证据是否支持答案，还要估计答案证据是否仍然藏在未被有效观察的 source units 中。

更准确地说：

> We estimate the probability that answer-bearing evidence remains undetected under the current multimodal observation state.

中文解释：

> 不是预测某个检索器会不会失败，也不是简单判断证据够不够，而是学习“答案证据在当前观察状态下是否仍不可发现”。

---

## 1. 为什么要推翻原来的 Agentic MM-RAG 主线

原仓库中的 `Agentic-MM-RAG` 有较好的工程基础：typed tools、多专家检索、evidence board、reflection、fusion、guardrail、document/video store adapter。这些东西可以作为实现基础，但不建议作为 AAAI 论文的核心创新。

原因如下。

### 1.1 Agentic RAG 已经太拥挤

Planner、multi-agent、tool use、reflection、self-correction、evidence board 等机制已经非常常见。审稿人会自然把这类工作归入：

- agentic RAG workflow;
- multi-agent retrieval;
- adaptive retrieval;
- evidence sufficiency checking;
- retrieval with reflection。

如果论文标题是 `Agentic Multimodal RAG`，审稿人会先入为主地问：

> 你的 agent 和已有 agentic RAG 本质上有什么不同？

这很难回答。

### 1.2 Evidence Board 也不是足够新的主线

Evidence board、evidence pool、gap/conflict、audit context 已经接近已有 work 的表述。把这些作为主创新，容易被认为是系统工程组合。

### 1.3 需要一个更底层的问题定义

好的 AAAI story 不应该是：

> 我们加了 agent、工具和反思，所以效果更好。

而应该是：

MissRisk-RAG Final Proposal · Page 1

---

> 现有 RAG 忽略了一个本质失败模式，我们提出新的建模对象、训练目标和评估协议来解决它。

MissRisk-RAG 的本质失败模式是：

> RAG 找不到答案时，并不知道答案是真的不存在，还是藏在未被有效观察的多模态区域中。

这比“证据是否足够”更深一层。

---

## 2. 核心研究问题

普通 RAG 的逻辑：

`retrieve top-k evidence -> reason over retrieved evidence -> answer`

它只能回答：

> 我找到了什么。

它不能回答：

> 我还有哪些地方没有有效观察？答案是否可能被漏掉？

在多模态场景里，这个问题尤其严重：

・视频中答案可能藏在未抽到的帧或相邻时间段；  
・文档答案可能藏在表格、图表、脚注、OCR 区域；  
・图片 caption 不包含答案，但源图中有答案；  
・graph 没有 relation 可能是抽取漏了，不代表关系不存在；  
・文本检索失败不代表视觉证据不存在。

因此本文的问题定义是：

`Given a question q, search obligations o, source units u,`

`and current observation state s_t,`

`estimate whether answer-bearing evidence remains undetected.`

形式化：

`R_t(q, o, u) = P(B_u = 1, D_u(s_t) = 0 | q, o, u, s_t)`

其中：

・`B_u = 1` 表示 unit `u` 中存在 answer-bearing evidence;  
・`D_u(s_t) = 1` 表示在当前观察状态 `s_t` 下，该证据可以被发现;  
・`D_u(s_t) = 0` 表示当前观察状态下该证据仍不可发现;  
・`R_t` 是 residual miss risk。

注意：

> 这不是 `P_bear * P_fail` 的独立假设，而是条件联合概率。

可以使用链式法则解释：

`P(B=1, D=0 | q,o,u,s)`

`= P(B=1 | q,o,u,s) * P(D=0 | B=1,q,o,u,s)`

但实现上推荐直接训练 joint miss-risk head。

---

## 3. 最终模式： Discoverability-Centered RAG

建议把工作内部称为 `Discoverability-RAG`，论文标题保留 `MissRisk-RAG`。

MissRisk-RAG Final Proposal · Page 2

---

核心思想:

> RAG 不应只优化 relevance，而应优化 answer evidence 的 discoverability。

## 3.1 三个核心对象

### Observation Unit

`ObservationUnit` 是 corpus 中可被观察的最小 source 单元。

示例：

{
  "unit_id": "docA_page12_image3",
  "source_type": "document",
  "modality": "image",
  "locator": {
    "doc_id": "docA",
    "page_id": 12,
    "layout_id": 3
  },
  "raw_content": {
    "ocr": "...",
    "caption": "...",
    "image_path": "..."
  }
}

可以是:

• text quote;  
• paragraph;  
• page;  
• slide;  
• table block;  
• chart/image block;  
• OCR region;  
• video segment;  
• frame group;  
• graph edge/path.

### Observation State

`ObservationState` 表示当前系统对某个 unit 看到了什么。

{
  "unit_id": "docA_page12_image3",
  "state": {
    "text": false,
    "ocr": true,
    "caption": true,
    "source_image": false,

MissRisk-RAG Final Proposal · Page 3

---

"vlm_inspection": false,

    "table_structure": false

},

"quality": {

    "ocr": 0.52,

    "caption": 0.71,

    "visual": 0.0

}

}

同一个 unit 可以有多种 state:

• `text_only`
• `ocr_only`
• `caption_only`
• `table_flattened`
• `source_image_vlm`
• `sparse_frames`
• `dense_frames`
• `graph_only`
• `full_observation`

**Search Obligation**

`SearchObligation` 是为了回答问题必须检查的事实面向。

示例问题:

> Do chimpanzees have a designated leader who dictates the strategy, or is it a more fluid process?

Obligations:

[

{

    "id": "o1",

    "text": "Check whether the source explicitly states that a single leader dictates the strategy.",

    "required_modalities": ["transcript", "graph_relation"]

},

{

    "id": "o2",

    "text": "Check whether multiple chimpanzees take different roles such as driver, blocker, and ambusher.",

    "required_modalities": ["transcript", "video_segment", "graph_relation"]

},

{

    "id": "o3",

    "text": "Check whether visual scenes show distributed coordination or role switching.",

    "required_modalities": ["video_segment", "frame_group"]

}

]

MissRisk-RAG Final Proposal · Page 4

---

# 4. 模型设计

最终不要训练一个模糊的“大风险模型”。建议训练三部分，其中第三部分是主模型。

## 4.1 Model A: Answer-Bearing Unit Predictor

目标：

`P(B_u = 1 | q, o, u)`

含义：

> unit `u` 是否包含能回答、支持、反驳或改变 obligation `o` 的证据?

这不是普通 relevance。一个 unit 可以 relevant 但不是 answer-bearing。

输入：

`[QUESTION] q`

`[OBLIGATION] o`

`[UNIT] text / OCR / caption / image description / table text / graph relation / metadata`

标签：

`1 = oracle answer-bearing unit`

`0 = hard negative unit`

正样本来源：

• MMDocRAG `gold_quotes`  
• SlideVQA evidence slide/page  
• MultiModalQA supporting context  
• TVQA localized moment

负样本来源：

• candidate quotes 中非 gold 的;  
• 同文档相邻页;  
• 同 slide deck 非 evidence slides;  
• dense top-k 但不支持答案;  
• same entity wrong relation;  
• same table wrong row/column。

建议模型：

• DeBERTa-v3-base/large cross-encoder;  
• BGE reranker family;  
• Jina reranker;  
• Qwen/Llama LoRA reranker。

Loss:

`L_bear = BCE(y_bear, p_bear)`

或 pairwise ranking:

`score(q,o,u_positive) > score(q,o,u_negative)`

## 4.2 Model B: Conditional Detectability Model

这是 v3 的关键修正。

不要训练:

MissRisk-RAG Final Proposal · Page 5

---

`□□ baseline □□□□`

而要训练：

`□□□ observation state □□oracle evidence □□□□□□`

目标：

`P(D_u(s) = 1 | B_u = 1, q, o, u, s)`

含义：

> 如果 unit `u` 确实含答案，在 observation state `s` 下，系统能不能发现它?

训练方式： controlled observation interventions。

对每个 oracle answer unit，人工构造不同 observation states：

`text_only`

`ocr_only`

`caption_only`

`table_flattened`

`source_image_vlm`

`sparse_frames`

`dense_frames`

`graph_only`

`full_observation`

然后判断当前 state 是否足以恢复 oracle answer / oracle claim。

标签：

`D = 1: □□ state □□□□□`

`D = 0: □□ state □□□□□□`

如何生成标签：

1. exact match / numeric match;
2. entailment model;
3. LLM judge;
4. VLM judge;
5. 少量人工抽查校验。

为什么这样更稳：

• 不依赖某个 baseline 的历史失败；  
• 不会过拟合某个检索策略；  
• 学的是 observation state 本身是否足够；  
• 可以测试 held-out state/action 泛化。

## 4.3 Model C: Joint Miss-Risk Estimator

这是主模型。

目标：

`R_t(q,o,u,s_t) = P(B_u = 1, D_u(s_t) = 0 | q,o,u,s_t)`

输入：

`question + obligation + unit representation + observation state features`

输出：

---

`joint miss-risk score`

训练标签:

`y_miss = 1 iff B = 1 and D = 0`

多任务 loss:

`L = L_miss + lambda_b * L_bear + lambda_d * L_detect`

其中:

• `L_miss` 是主任务;

• `L_bear` 帮助模型学习 answer-bearing;

• `L_detect` 帮助模型学习 state detectability。

这样避免了两个问题:

1. 不需要假设 `P_bear` 和 `P_fail` 独立;
2. 不需要把某个 baseline 的失败作为真实风险标签。

MissRisk-RAG Final Proposal · Page 6

---

## 5. Observation Policy: 不要作为主创新

OPD / observation policy 第一版不需要训练。

使用确定性贪心:

`a* = argmax_a ExpectedRiskReduction(a) / Cost(a)`

其中:

`ExpectedRiskReduction(a)`

`= R_t - E[R_{t+1} | action a]`

动作会改变 observation state:

`ocr_only -> source_image_vlm`

`sparse_frames -> dense_frames`

`caption_only -> full_observation`

`graph_only -> graph_plus_source_chunks`

重新用 joint miss-risk model 估计 action 后风险:

`R_{t+1}(q,o,u,s_t after a)`

选择单位成本下降最大的 action。

为什么不用训练 policy:

• 训练 policy 需要大量交互日志;

• 容易过拟合数据集;

• 审稿人会质疑为何不用简单贪心;

• 主贡献应集中在 risk/discoverability model。

可以在扩展实验中比较:

• relevance greedy;

• answer-bearing greedy;

• detectability greedy;

• joint miss-risk greedy;

• oracle policy.

MissRisk-RAG Final Proposal · Page 7

---

# 6. 数据集策略：MissRiskBench

你的方向不能只拿现成 QA benchmark 跑 answer accuracy。必须构造一个派生数据集：

> **MissRiskBench: a benchmark for answer-bearing evidence missed under partial multimodal observation.**

它不是完全从零标注，而是把已有带 evidence 的数据集转成 MissRisk 格式。

## 6.1 为什么不能直接用普通 QA 数据集

普通 QA 数据集通常只给：

`question, answer`

最多给 supporting context。

但 MissRisk 需要：

• oracle answer units;  
• hard negative units;  
• observation states;  
• detectability labels;  
• hidden-answer cases;  
• false-unanswerable cases;  
• true-unanswerable cases。

没有这些，无法证明模型真的在估计“答案是否仍未被发现”。

## 6.2 Tier 1: 主数据 MMDocRAG

来源：

• Hugging Face: https://huggingface.co/datasets/MMDocIR/MMDocRAG  
• Project: https://mmdocrag.github.io/MMDocRAG/  
• Paper: https://arxiv.org/abs/2505.16470

适合原因：

• 本身就是 multimodal document RAG benchmark;  
• 有 4,055 expert-annotated QA pairs;  
• 每条包含 text quotes、image quotes、gold quotes;  
• gold quotes 可直接当 oracle answer units;  
• non-gold candidate quotes 可直接当 hard negatives;  
• 有 long document PDFs 和 image quote files。

根据数据说明，MMDocRAG 的 annotation format 包含：

• `q_id`  
• `doc_name`  
• `domain`  
• `question`  
• `evidence_modality_type`  
• `question_type`  
• `text_quotes`  
• `img_quotes`  
• `gold_quotes`  
• `answer_short`  
• `answer_interleaved`

转化方式：

MissRisk-RAG Final Proposal · Page 8

---

ObservationUnit = text_quote or img_quote
```

```
B=1 = quote_id in gold_quotes
```

```
B=0 = quote_id not in gold_quotes
```

```
Hard negatives = non-gold quotes ranked high by lexical/dense retrieval
```

可构造 states:

• `quote_text_only`  
• `image_description_only`  
• `page_context_only`  
• `source_image_vlm`  
• `full_quote`  
• `masked_gold`

推荐第一步:

```
□□ 50 QA □ pilot□□□□ 500□□□□ 2k+□
```

## 6.3 Tier 1 补强：SlideVQA

来源:

• GitHub: https://github.com/nttmdlab-nlp/SlideVQA  
• Paper: https://arxiv.org/abs/2301.04883

适合原因:

• slide deck 天然可分成 page/slide units;  
• 任务包含 evidence slide selection;  
• slide 中常有图、表、文本、布局;  
• 非常适合 observation state 构造。

转化方式:

```
ObservationUnit = slide page / layout block / OCR block
```

```
B=1 = evidence slide/page
```

```
B=0 = same deck non-evidence slide
```

可构造 states:

• `title_only`  
• `ocr_only`  
• `layout_text_only`  
• `source_slide_image`  
• `vlm_slide_summary`  
• `full_slide`

## 6.4 Tier 2: MultiModalQA

来源:

• GitHub: https://github.com/allenai/multimodalqa  
• Project: https://allenai.github.io/multimodalqa/  
• Paper: https://arxiv.org/abs/2104.06039

适合原因:

• 包含 text、tables、images;  
• 规模较大，约 29,918 examples;  
• supporting context 可转为 answer-bearing units;

MissRisk-RAG Final Proposal · Page 9

---

• 适合扩大 answer-bearing predictor 训练规模。

转化方式：

> ObservationUnit = text passage / table block / image item
>
> B=1 = supporting context item
>
> B=0 = retrieved non-supporting item

可构造 states:

• `text_only`  
• `table_flattened`  
• `table_schema_plus_rows`  
• `image_caption_only`  
• `full_multimodal_context`

## 6.5 Tier 3: 视频小规模扩展

视频不是第一阶段主训练集，原因：

• 下载和权限复杂；  
• 视频帧处理成本高；  
• oracle moment 对齐费时；  
• 容易拖慢实验。

但可以做 100-200 个 case/eval，提升多模态完整性。

推荐：

### TVQA

• Website: https://tvqa.cs.unc.edu/  
• Paper: https://arxiv.org/abs/1809.01696

优点：

• 有 video QA；  
• 有 temporal localization；  
• 字幕和时间片段适合做 video observation units。

### ActivityNet Captions

• Original: https://cs.stanford.edu/people/ranjaykrishna/densevid/  
• Hugging Face example: https://huggingface.co/datasets/Leyo/ActivityNet_Captions

优点：

• 有 temporal segment captions；  
• 可自动生成 QA；  
• oracle segment 明确。

### QVHighlights

• Paper: https://arxiv.org/abs/2107.09609  
• Project: https://github.com/jayleicn/moment_detr

优点：

• query-focused moment annotations；  
• 适合测试 observation policy / moment recall。

---

## 7. MissRiskBench 构造细节

MissRisk-RAG Final Proposal · Page 10

---

## 7.1 标准样本格式

{
    "question_id": "mmdoc_0001",
    "question": "...",
    "answer": "...",
    "source_id": "...",
    "source_type": "document",
    "obligations": [
        {
            "id": "o1",
            "text": "Check the numeric value in the chart.",
            "required_modalities": ["image", "chart", "ocr"]
        }
    ],
    "units": [
        {
            "unit_id": "text1",
            "modality": "text",
            "content": "...",
            "page_id": 3,
            "label_answer_bearing": 0
        },
        {
            "unit_id": "image3",
            "modality": "image",
            "img_path": "...",
            "img_description": "...",
            "page_id": 4,
            "label_answer_bearing": 1
        }
    ],
    "answerability_type": "answerable_hidden"
}

## 7.2 Detectability 样本格式

{
    "question_id": "mmdoc_0001",
    "obligation_id": "o1",
    "unit_id": "image3",
    "state_id": "image_description_only",
    "visible_content": "image description only...",
MissRisk-RAG Final Proposal · Page 11

---

"hidden_channels": ["source_image", "vlm_inspection"],

    "oracle_answer": "...",

    "label_answer_bearing": 1,

    "label_detectable": 0,

    "label_joint_miss": 1

}

## 7.3 Hidden-answer 构造

Hidden-answer 的定义：

> 答案存在，但普通 relevance top-k 或 text-only RAG 难以找到。

构造规则：

1. gold unit 不在 vanilla dense top-k;
2. gold unit 是 image/table/chart，而 query 更容易召回 text quote;
3. gold unit 需要 source image/VLM，image description 不够;
4. gold unit 在相邻 page/slide/segment，不在初始窗口;
5. answer-bearing unit 与 high-overlap negative 很相似。

## 7.4 False-unanswerable 构造

定义：

> baseline 输出 not answerable，但 oracle answer unit 存在。

步骤：

1. 跑 vanilla RAG / adaptive RAG / sufficiency RAG;
2. 找到输出 not answerable 的样本;
3. 检查数据集中 gold unit 是否存在;
4. 用人工或 verifier 确认答案可由 gold unit 支持。

这是 MissRisk-RAG 最应该打赢的类别。

## 7.5 True-unanswerable 构造

定义：

> corpus 中确实没有答案，系统应拒答。

构造方式：

• entity swap;  
• relation swap;  
• numeric/year swap;  
• ask absent visual detail;  
• cross-document impossible question。

注意：

> true-unanswerable 必须人工抽查，不能只靠 LLM 生成。

建议比例：

    answerable_easy: 25%

    answerable_hidden: 35%

    false_unanswerable: 20%

    true_unanswerable: 20%

MissRisk-RAG Final Proposal · Page 12

---

8. 训练集规模建议

## 8.1 Minimum Version

3-4 周内可做：

```text
MMDocRAG: 300 QA
SlideVQA: 200 QA
MultiModalQA: 200 QA
Total: 700 QA
```

派生：

```text
Answer-bearing pairs: 10k-30k
Detectability states: 5k-15k
Human-checked test: 300 cases
```

## 8.2 Strong Version

AAAI 正式投稿建议：

```text
MMDocRAG: 2k QA
SlideVQA: 1k QA
MultiModalQA: 1k QA
Video eval: 200 QA
Total: 4k+ QA
```

派生：

```text
Answer-bearing pairs: 80k-200k
Detectability states: 30k-100k
Human-checked evaluation: 500-800 cases
```

## 8.3 Split 规则

必须按 source split, 不要按 question 随机 split。

```text
train: 70%
calibration: 10%
dev: 10%
test: 10%
```

MMDocRAG 要按 `doc_name` split;
SlideVQA 要按 deck split;
视频要按 video split。

否则会泄漏 source-specific layout/entity。

---

# 9. 实验设计

## 9.1 Experiment 1: Answer-Bearing Retrieval

MissRisk-RAG Final Proposal · Page 13

---

问题：

> Model A 是否比 relevance retriever 更会找 oracle answer units?

对比：

• BM25 / lexical;  
• dense retriever;  
• existing reranker;  
• answer-bearing predictor.

指标：

• Oracle Answer Unit Recall@k;  
• MRR;  
• Recall under fixed budget;  
• hard negative accuracy。

## 9.2 Experiment 2: Detectability Prediction

问题：

> Conditional detectability 是否能预测不同 observation states 下答案是否可发现?

训练：

• oracle units;  
• controlled observation states。

测试：

• held-out documents;  
• held-out states;  
• held-out modalities。

指标：

• AUROC;  
• Brier score;  
• ECE;  
• state-wise accuracy;  
• cross-state generalization。

## 9.3 Experiment 3: Joint Miss-Risk Calibration

问题：

> joint miss-risk score 是否校准?

指标：

• reliability diagram;  
• expected calibration error;  
• bucketed miss frequency;  
• Brier score。

关键图：

`Predicted risk bucket 0.0-0.1 -> true miss frequency around 0.1`

`Predicted risk bucket 0.8-0.9 -> true miss frequency around 0.8`

## 9.4 Experiment 4: Observation Policy

问题：

MissRisk-RAG Final Proposal · Page 14

---

> 用 joint miss-risk 做贪心观察，是否比 relevance/adaptive 更有效?

对比：

- top-k relevance;
- adaptive top-k;
- uncertainty-based retrieval;
- answer-bearing only;
- detectability only;
- joint miss-risk greedy;
- oracle greedy.

指标：

- risk reduction per action;
- oracle answer unit recall vs cost;
- false-unanswerable reduction vs cost.

### 9.5 Experiment 5: Downstream QA

指标：

- Answer Accuracy;
- False-Unanswerable Rate;
- True-Unanswerable Accuracy;
- Abstention F1;
- Unsupported Answer Rate;
- Observation Cost;
- Latency / tool calls。

### 10. Baseline 设置

必须包含：

1. Vanilla top-k RAG
2. Multi-query RAG
3. Adaptive top-k expansion
4. Evidence sufficiency verifier
5. Current Agentic-MM-RAG
6. Answer-bearing only policy
7. Detectability only policy
8. Joint miss-risk without calibration
9. Joint miss-risk with calibration
10. Oracle observation policy

这样可以证明：

- 不是单纯 reranker;
- 不是多检索;
- 不是 sufficiency verifier;
- 不是 agent workflow;
- calibration 和 observation-state modeling 真的有贡献。

### 11. Related Work 边界

MissRisk-RAG Final Proposal · Page 15

---

## 11.1 SURE-RAG

SURE-RAG 关注 evidence sufficiency、support/refute/insufficient 和 selective answering。  
区别：

> SURE-RAG 判断 retrieved evidence 是否足够；MissRisk-RAG 判断未被有效观察的 source units 是否仍可能藏有答案。

## 11.2 MARA

MARA 做 multimodal adaptive retrieval 和 self-reflective evidence control。  
区别：

> MARA 动态扩展 evidence；MissRisk-RAG 建模 observation state 下答案证据是否仍不可发现。

## 11.3 Stateful Evidence-Driven RAG

该方向做 evidence pool、gap/conflict、iterative retrieval。  
区别：

> 它是 evidence-state accumulation；MissRisk-RAG 是 unobserved-space risk estimation。

## 11.4 EVA-RAG / Counter-Evidence RAG

这些工作把 answer 当 hypothesis，找 anti-context 或反证。  
区别：

> MissRisk-RAG 不要求存在反证，它估计答案证据是否在当前观察状态下仍未被发现。

## 11.5 Conformal / Certified RAG

这些工作关注 statistical coverage 或 generation risk。  
区别：

> MissRisk-RAG 不声称理论 coverage guarantee，而是提出 practical calibration of answer-miss risk over multimodal observation units。

---

## 12. 预期效果与成功标准

Pilot 阶段成功标准：

• Oracle Answer Unit Recall@10 比 dense retrieval 高 10% 以上；  
• false-unanswerable rate 降低 15% 以上；  
• joint miss-risk 对 missed oracle units 有明显排序能力；  
• calibration curve 有单调趋势；  
• hidden-answer split 上明显优于 adaptive top-K。

正式实验目标：

• hidden-answer recall 提升；  
• false-unanswerable 显著降低；  
• true-unanswerable 不明显变差；  
• unsupported answer rate 不上升；  
• 同等 cost 下 oracle answer unit recall 更高；  
• risk calibration 优于 uncertainty baseline。

MissRisk-RAG Final Proposal · Page 16

---

## 13. 实施路线

### Week 1: Pilot 数据

• 下载/解析 MMDocRAG dev/evaluation jsonl;  
• 提取 question、text_quotes、img_quotes、gold_quotes;  
• 转 ObservationUnit;  
• 构造 hard negatives;  
• 做 50 QA pilot。

### Week 2: Answer-Bearing Predictor

• 训练/微调 cross-encoder;  
• 对比 BM25/dense;  
• 报 Recall@k;  
• 选出 hidden-answer 样本。

### Week 3: Detectability States

• 对 oracle units 构造 states;  
• 用 exact/NLI/LLM/VLM 生成 detectable 标签;  
• 人工抽查 100-200 条;  
• 训练 conditional detectability model。

### Week 4: Joint Miss-Risk

• 训练 joint head;  
• 做 calibration split;  
• 画 reliability diagram;  
• 做 risk bucket 分析。

### Week 5: Policy + QA

• 实现 greedy risk-reduction policy;  
• 跑 downstream RAG;  
• 对比 top-k/adaptive/sufficiency baselines。

### Week 6: 扩数据

• 加 SlideVQA;  
• 加 MultiModalQA;  
• 整理 300-600 条人工精评测试集。

### Week 7: 论文初稿

• 写 introduction/method/experiments;  
• 画主图;  
• 写 related work 差异。

### Week 8: 补实验和提交

• ablation;  
• case study;  
• limitation;

MissRisk-RAG Final Proposal · Page 17

---

• code/data card;
• final writing.

---

## 14. 文件结构建议

`data_missrisk/`

`  raw/`

`    mmdocrag/`

`    slidevqa/`

`    multimodalqa/`

`  processed/`

`    units.jsonl`

`    qa.jsonl`

`    answer_bearing_pairs.jsonl`

`    detectability_states.jsonl`

`    missrisk_train.jsonl`

`    missrisk_dev.jsonl`

`    missrisk_test.jsonl`

` `

`scripts/`

`  build_mmdocrag_units.py`

`  build_slidevqa_units.py`

`  build_multimodalqa_units.py`

`  generate_observation_states.py`

`  label_detectability.py`

`  train_answer_bearing.py`

`  train_missrisk.py`

`  eval_missrisk.py`

` `

`agentic_mm_rag/`

`  observation/`

`    units.py`

`    states.py`

`    obligations.py`

`    risk.py`

`    policy.py`

`    report.py`

---

## 15. 最终 Abstract 草稿

MissRisk-RAG Final Proposal · Page 18

---

Multimodal retrieval-augmented generation systems often justify answers with retrieved evidence, but they rarely estimate whether
answer-bearing evidence remains undetected in source units that were only partially observed. This distinction matters: a failure
to answer may reflect the absence of evidence, or simply insufficient observation of relevant tables, figures, OCR regions, video
segments, or graph relations. We introduce MissRisk-RAG, a framework that estimates residual answer-miss risk as the probability
that an answer-bearing source unit remains undetected under the current multimodal observation state. Instead of learning failures
from logs of a particular retriever, MissRisk-RAG constructs controlled observation interventions over oracle answer units and
trains a joint miss-risk estimator with auxiliary answer-bearing and detectability heads. A deterministic observation policy then
selects actions by expected miss-risk reduction per cost, enabling the system to answer, abstain, or declare under-observation. We
construct MissRiskBench from multimodal document and video QA datasets with oracle answer units, hard negatives, controlled
observation states, and hidden-answer/unanswerable splits. Experiments evaluate answer-unit recall, detectability calibration,
false-unanswerable reduction, and downstream abstention accuracy.

---

## 16. 最终写作原则

1. 不说 agent 是创新，只说 observation actions。
2. 不说 `P_bear * P_fail`，说 conditional joint miss risk。
3. 不说 baseline failure predictor，说 controlled detectability supervision。
4. 不说 coverage guarantee，说 calibrated residual miss-risk estimate。
5. 不只报 accuracy，要报 answer-unit recall、false-unanswerable、calibration、cost。
6. 不把视频作为第一阶段主数据，先用 MMDocRAG/SlideVQA 做硬结果。

---

## 17. 最终推荐标题

首选：

> **MissRisk-RAG: Learning When Answer-Bearing Evidence Remains Undetected in Multimodal Retrieval-Augmented Generation**
> *

备选：

> **MissRisk-RAG: Estimating Undetected Answer Evidence in Multimodal Retrieval-Augmented Generation**

备选：

> **Know Where You Looked: Discoverability-Aware Multimodal Retrieval-Augmented Generation**

---

## 18. 参考链接

• MMDocRAG dataset: https://huggingface.co/datasets/MMDocIR/MMDocRAG  
• MMDocRAG project: https://mmdocrag.github.io/MMDocRAG/  
• MMDocRAG paper: https://arxiv.org/abs/2505.16470  
• SlideVQA GitHub: https://github.com/nttmdlab-nlp/SlideVQA  
• SlideVQA paper: https://arxiv.org/abs/2301.04883  
• MultiModalQA GitHub: https://github.com/allenai/multimodalqa  
• MultiModalQA paper: https://arxiv.org/abs/2104.06039  
• TVQA paper: https://arxiv.org/abs/1809.01696  
• QVHighlights paper: https://arxiv.org/abs/2107.09609  
• ActivityNet Captions: https://cs.stanford.edu/people/ranjaykrishna/densevid/

MissRisk-RAG Final Proposal · Page 19
