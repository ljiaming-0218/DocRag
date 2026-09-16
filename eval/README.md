# DocRAG Evaluation

评估目录只负责衡量系统效果，不承担业务代码和普通接口回归测试。

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

## 2. 检索评估

先启动本地后端，再执行：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.retrieval --modes dense dense_rerank hybrid_rerank --candidate-k 15 --top-k 3
```

低成本运行单题：

```cmd
D:\Anaconda\envs\medrag\python.exe -m eval.runners.retrieval --modes dense dense_rerank hybrid_rerank --candidate-k 15 --top-k 3 --case-id RAG-01
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

生成评估会调用真实 LLM。执行成功率、路由、拒答和引用存在性可以自动
统计；正确性、完整性、忠实性和引用正确性仍需人工复核。

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
