# DocRAG 小型 RAG 评估集

该目录用于评估 DocRAG 的检索效果、Query Rewrite、多轮追问和回答证据覆盖。它是项目回归评估集，不是模型训练集，也不是 OCR 测试集。

## 数据规模

- 3 篇公开论文。
- 每篇 5 道题。
- 共 15 道题。
- 题型包括事实、术语、对比、总结、无答案和多轮追问。

论文来源：

1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
2. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
3. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

## 文件说明

```text
eval/
├─ rag_dataset/
│  ├─ pdfs/                       # 评估 PDF
│  ├─ documents.json              # 来源、页数和 SHA-256
│  ├─ questions.json              # 问题、答案要点和 gold evidence
│  ├─ eval_template.csv           # 人工评分模板
│  ├─ validate_dataset.py         # 数据集结构校验
│  └─ results/                    # 原始响应和指标结果
├─ run_eval.py                    # 非追问题执行器
├─ run_follow_up_eval.py          # 多轮追问题执行器
├─ calculate_metrics.py           # 检索指标统计
└─ debug_retrieval.py             # 单题候选召回诊断
```

`gold_answer`、`expected_rewritten_query` 和 `source_pages` 只能用于评估，不能发送给被测系统，否则会造成 gold 信息泄漏。

## 固定测试条件

对比不同版本时应保持：

- 相同 PDF 和问题集。
- 相同 `user_type=general`。
- 相同 `n_results=3`。
- 相同 `history_limit=6`。
- 相同 Embedding、Reranker 和 LLM。
- 相同 chunk_size 和 chunk_overlap。
- 非追问题使用独立 conversation。
- 追问题先发送设定的历史问题，再发送当前问题。

只能改变一个实验变量，否则无法判断效果变化来自哪个优化。

## 运行步骤

### 1. 启动依赖和后端

确保 MongoDB、FastAPI 和模型可用：

```cmd
cd /d D:\MedRag\medrag\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. 校验数据集

另开终端：

```cmd
cd /d D:\MedRag
D:\Anaconda\envs\medrag\python.exe eval\rag_dataset\validate_dataset.py
```

预期输出为 3 篇 PDF、15 道问题且题型统计正确。

### 3. 执行普通问题

```cmd
D:\Anaconda\envs\medrag\python.exe eval\run_eval.py
```

结果写入：

```text
eval/rag_dataset/results/normal_results.json
```

### 4. 执行追问题

```cmd
D:\Anaconda\envs\medrag\python.exe eval\run_follow_up_eval.py
```

结果写入：

```text
eval/rag_dataset/results/follow_up_results.json
```

### 5. 计算检索指标

```cmd
D:\Anaconda\envs\medrag\python.exe eval\calculate_metrics.py
```

结果写入：

```text
eval/rag_dataset/results/metrics_summary.json
```

注意：只运行 `calculate_metrics.py` 不会调用最新系统，它只会重新统计已有 JSON。修改检索代码后，必须先重新执行普通题和追问题。

## 指标口径

### Execution Success Rate

成功完成接口请求的问题数除以总问题数。它只表示调用链完成，不代表答案正确。

### HitRate@K

一道题返回的前 K 个结果中是否至少包含一个 gold evidence。适合判断“是否命中过证据”。

### Recall@K

前 K 个结果覆盖全部 gold evidence 的比例。总结题和多证据题比 HitRate 更依赖 Recall。

### MRR@K

第一个正确证据排名的倒数。MRR 高表示正确证据通常更靠前。

无答案题没有 gold evidence，因此 HitRate、Recall 和 MRR 记为 `null`，不能记为 0。无答案题应单独评估拒答准确性、错误引用和幻觉。

## 人工评分

每项建议使用 0-2 分：

- 检索相关性
- Query Rewrite 正确性
- 答案完整性
- 答案忠实度
- 引用准确性

另记录：

- 是否幻觉
- 是否正确拒答
- 错误阶段
- 备注

页码命中不等于 chunk 内容一定相关。对于同一页包含多个语义段落的情况，应继续人工检查具体 chunk。

## 当前保存的旧基线

| 指标 | 数值 |
| --- | ---: |
| Execution Success Rate | 1.0000 |
| HitRate@3 | 0.6667 |
| Recall@3 | 0.5972 |
| MRR@3 | 0.5417 |

分题型结果显示：

- 事实题表现最好。
- 术语题基本能命中，但首条排序仍可优化。
- 对比题和追问题存在漏召回。
- 旧结果中的两道总结题均未命中 gold 页码。
- LoRA 文档整体弱于另外两篇文档。

这些原始响应早于最新 Summary Multi-Query 修改。因此，这组数值只能作为旧版本基线，不能用于证明当前优化已经生效。

## 失败诊断顺序

1. 查看 `rewritten_query` 是否忠实、完整且无 gold 泄漏。
2. 检查正确证据是否进入向量候选池。
3. 检查正确证据在 Rerank 前后的排名变化。
4. 检查最终 sources 是否含正确 chunk，而不只看页码。
5. sources 正确但回答失败时，再检查 Prompt 和 LLM。

Reranker 只能重新排列已有候选。正确证据未进入候选池时，应优先检查 Query Rewrite、语言适配、Embedding、chunk 和候选召回数。

## 评估边界

- 15 道题适合验证流程和做版本回归，不足以代表生产质量。
- 当前自动指标以页码 gold 为主，尚未完整覆盖 chunk relevance、答案忠实度和引用准确性。
- 只对成功样本计算质量均值会高估系统，因此执行成功率必须单独报告。
- 对比优化前后结果时，应保留原始响应、运行配置和代码版本。
