# DocRAG Evaluation

评估目录只负责衡量系统效果，不承担业务代码和普通接口回归测试。

## Stage B Closure（正式基线）

Stage B 于 2026-09-26 收口。冻结代码 commit：
`358ee89b262e42616e8fd9097db86ff51ebd4b17`。正式评估基线为
`generation-single_turn-20260925T050759588111Z`（20/20 Generation）与
`judge-20260925T055551281478Z` 加 `judge-20260925T065944162831Z`
（补跑 RAG-03 后 Judge 20/20）。人工参与复核记录：
`eval/manual_review_v2.json`。详细逐题结果及 manifest 保留在
`eval/runs/stage_b_closure/stage-b-closure-20260925T071022Z/`。

Generation 原始请求/答案结果不因后续指标审计而重写。最终拒答指标使用
`correct_refusal_without_sources_v1` 口径；其离线拒答检测器在评估后有小幅
修正，不改变 Generation、Retrieval 或 Prompt。归档源码快照的 HEAD 是
`453e57296a89c178d5b887e17a71a0a5dcdbb407` 且当时工作区有改动。冻结 commit
保留归档源码，并包含评估后拒答指标检测器/测试差异及阶段文档更新；该离线
改动不改变 Generation 与 Retrieval 运行链路。完整对应关系记录在 closure manifest。

另有最新 run 18/20 出现偶发本地后端 HTTP 500，根因尚未定位。此为
Reliability/工程稳定性阶段的非阻塞 Known Issue，不覆盖上述正式评估基线。

## 目录

```text
eval/
├─ core.py                 # 公共路径、JSON、认证和文档索引客户端
├─ dataset/                # 固定数据集与 PDF
│  ├─ documents.json
│  ├─ single_turn.json
│  ├─ follow_up.json
│  └─ pdfs/
├─ runners/
│  ├─ retrieval.py         # 检索对照，不调用 LLM
│  ├─ summary_retrieval.py # 模板多查询总结检索，不调用 LLM
│  ├─ generation.py        # 单轮/追问端到端生成评估
│  ├─ judge.py             # 对已有答案执行 LLM-as-a-Judge
│  └─ hiv_acceptance.py    # HIV 知识库线上业务验收
├─ judges/
│  └─ llm_judge.py         # Judge Prompt、解析与评分校验
├─ metrics/
│  ├─ retrieval.py
│  └─ generation.py
├─ tools/                  # 校验、标注、诊断、阈值校准
└─ runs/                   # 运行产物，默认不提交 Git
```

## 数据契约

`scenario` 描述交互形式：`single_turn` 或 `follow_up`。

`category` 描述题目类别：`fact`、`term`、`comparison`、`summary`、
`source_check` 或 `unanswerable`。两者不能混用。

每个案例至少包含：

```text
case_id
document_key
scenario
category
query
answerable
expected_task
reference_answer
answer_points
gold_pages
gold_evidence
```

同一个答案要点在摘要、正文或附录有多个等价出处时，使用证据组：

```json
{
  "evidence_id": "LORA-01-E2",
  "alternatives": [
    {"page": 1, "text": "freezes the pre-trained model weights"},
    {"page": 4, "text": "W0 is frozen and does not receive gradient updates"}
  ]
}
```

一个证据组命中任意 `alternative` 即视为命中一次，Recall 分母仍是证据组数量。

## 1. 校验数据

```cmd
cd /d D:\MedRag
D:\Anaconda\envs\medrag\python.exe -m eval.tools.validate_dataset --require-gold-evidence
```

这里只证明数据结构、PDF 哈希、页码和原文证据合法。

查看题型、追问和跨文档覆盖缺口：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.tools.audit_dataset_coverage
```

单文档案例使用 `document_key`；跨文档案例使用 `document_keys`。跨文档生成评估会通过公开 API 创建评估知识库，并使用 `selected_document_ids` 限定案例范围。检索评估会对同一文档集合执行一次 Dense 查询；跨文档案例不计算容易产生歧义的页码指标，只计算带 `document_key` 的 evidence 指标。

## 2. 检索评估

先启动本地后端，再执行：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.retrieval --modes dense dense_rerank hybrid_rerank --candidate-k 15 --top-k 3 --metric-k 3 5 15
```

低成本运行单题：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.retrieval --modes dense dense_rerank hybrid_rerank --candidate-k 15 --top-k 3 --metric-k 3 5 15 --case-id RAG-01
```

重复传入 `--case-id` 可以运行一组指定案例。

输出按 `candidate`、`final`、`filtered` 三阶段记录 HitRate、Recall 和
MRR，并单独报告无答案题的证据误返回率。该流程不调用 Query Rewrite
或 LLM，不能代表最终答案质量。

总结类题另用确定性模板子查询评估真实 Summary 检索编排：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.summary_retrieval --case-id COT-03 --context-k 10
```

不传 `--case-id` 时运行全部 `summary` 题。结果保存在
`eval/runs/summary_retrieval/`，记录子查询、最终 sources 和证据
Recall/MRR。评估进程禁用可选的 LLM 子查询生成；Embedding API
和本地 Reranker 仍按项目配置运行。`context-k=10` 与普通检索的
`top-k=3` 不可直接比较，应先在相同 K 下做对照。该流程不评估
最终答案质量。

## 3. 生成评估

单轮：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.generation --suite single_turn
```

追问：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.generation --suite follow_up
```

全部：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.generation --suite all
```

低成本验证单题：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.generation --suite single_turn --case-id RAG-01 --request-interval 0
```

生成评估会调用真实 LLM，并保存数据集指纹、服务版本、模型/检索配置、
Prompt 指纹、逐题来源、失败阶段、P50/P95 延迟和供应商 Token 用量。
无答案误答率使用拒答短语与语义模式启发式统计，并在逐题结果中保留
拒答判定轨迹；该指标仍需人工抽检。正确性、完整性、忠实性和引用正确性
由后续 Judge 评分并保留理由。正确拒答且没有检索来源时，Faithfulness
和 Citation Correctness 标记为 N/A；原始 Judge 分数保留在 `raw_scores`。

## 4. LLM Judge

先完成生成评估，再把对应的 `results.json` 交给 Judge：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.judge --input eval\runs\generation\<run_id>\results.json
```

先低成本验证单题：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.judge --input eval\runs\generation\<run_id>\results.json --case-id RAG-01 --request-interval 0
```

Judge 只读取已有答案，不会重新上传文档或执行 RAG，也不会修改原始结果。
它自动评价答案正确性、完整性、忠实性和引用正确性，产物写入
`eval/runs/judge/`。Judge 可能存在同模型偏差，因此结果必须保留理由，并进行
人工抽检；Recall、MRR 等检索指标仍由确定性检索评估计算。

## 5. 单题诊断

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.tools.inspect_retrieval_case --run eval\runs\retrieval\<run_id>\results.json --case-id COT-02
```

该工具只读取已有结果，不重新索引、不调用模型，也不会把 gold evidence
泄露给正常检索流程。

## 6. 线上验收

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.hiv_acceptance --base-url https://your-space.hf.space
```

线上验收只检查真实部署的业务闭环，不与离线 Recall/MRR 混合统计。
