# DocRAG 评估说明

评估目录只负责固定数据集、离线指标、Judge 和线上业务验收，不承担业务代码和普通接口回归测试。

## 当前状态

- 数据集包含 3 篇公开论文、20 道单轮题和 5 道追问题，题型覆盖事实、术语、总结、对比、无答案、来源核查和跨文档。
- Stage B 的正式 Generation/Judge 归档仍保留在 `eval/runs/stage_b_closure/stage-b-closure-20260925T071022Z/`，它是冻结快照结果，不代表当前提交全部稳定。
- 当前提交的固定 20 题本地运行报告为 `eval/runs/generation/generation-single_turn-20260929T085541756035Z/summary.json`：20/20 执行成功、无 HTTP 500；证据召回为 0.632，P95 延迟约 199.9 秒，1 个可回答问题被误拒答。此前 5/20 结果保留为 MongoDB 依赖故障历史。
- 当前线上业务验收报告为 `eval/runs/release_acceptance/release-acceptance-20260929T075433283381Z.json`：`passed=true`。
- 当前 Judge 报告为 `eval/runs/judge/judge-20260929T094048797812Z/summary.json`：20/20 评分完成；正确性 0.855、完整性 0.786、忠实性 0.989、引用正确性 0.933。

## 数据契约

每个案例至少包含：

```text
case_id
document_key 或 document_keys
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

单文档案例使用 `document_key`；跨文档案例使用 `document_keys`。修改 PDF、问题或证据后，dataset fingerprint 会变化，不同 fingerprint 的结果不能直接比较。

## 校验数据集

```powershell
Set-Location D:\\MedRag
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.tools.validate_dataset --require-gold-evidence
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.tools.audit_dataset_coverage
```

这些命令只校验 PDF hash、页码、gold evidence 和题型覆盖，不调用 LLM。

## 检索评估

先启动本地后端：

```powershell
Set-Location D:\\MedRag\\medrag\\backend
D:\\Anaconda\\envs\\medrag\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

再运行：

```powershell
Set-Location D:\\MedRag
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.runners.retrieval --modes dense dense_rerank hybrid_rerank --candidate-k 15 --top-k 3 --metric-k 3 5 15
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.runners.summary_retrieval
```

检索评估记录候选召回、Rerank 后结果、阈值过滤、HitRate、Recall、MRR 和无答案误返回率，不代表最终回答质量。

## 生成评估

单轮和追问入口：

```powershell
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.runners.generation --suite single_turn
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.runners.generation --suite follow_up
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.runners.generation --suite all
```

生成评估保存数据集 fingerprint、版本、逐题来源、失败阶段、延迟和 token 使用。执行成功率只表示 HTTP/流程完成，不等于答案正确率。

Judge 只读取已有结果：

```powershell
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.runners.judge --input eval\\runs\\generation\\<run_id>\\results.json
```

Judge 分数需要结合逐题理由和人工抽检，不能替代检索指标或人工复核。本次 Judge 使用与回答相同的 ModelScope 模型，属于辅助证据。

## 单题诊断

```powershell
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.tools.inspect_retrieval_case --run eval\\runs\\retrieval\\<run_id>\\results.json --case-id COT-02
```

该工具只读取已有结果，不重新索引，也不会把 gold evidence 注入正常检索流程。

## 线上验收

线上验收会在真实部署完成注册、登录、PDF 上传、索引任务、会话提问、来源和历史消息检查：

```powershell
$releaseCommit = '8770e28ec88e5d7797e622117cd749eace20f168'
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.runners.release_acceptance `
  --base-url https://maoxiao-1205-medrag.hf.space `
  --expected-build-commit $releaseCommit `
  --preflight-only

D:\\Anaconda\\envs\\medrag\\python.exe -m eval.runners.release_acceptance `
  --base-url https://maoxiao-1205-medrag.hf.space `
  --expected-build-commit $releaseCommit
```

报告写入 `eval/runs/release_acceptance/`。线上验收通过只证明部署业务闭环，不代表固定效果集稳定或答案全部正确。
