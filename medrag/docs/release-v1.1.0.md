# DocRAG v1.1.0

状态：**演示候选 / 条件结项**

## 版本范围

- PDF 按页解析、OCR 回退、Fixed/Recursive 切片和切片元数据。
- document hash 去重、index fingerprint、index generation 和 active generation 切换。
- MongoDB 持久化 ingestion job，支持 202、状态查询、失败重试、幂等和 force reindex。
- Dense、BM25、RRF、CrossEncoder Rerank、证据阈值、Query Rewrite 和 Summary Multi-Query。
- JWT + Argon2 登录、知识库、多文档检索、会话历史和 sources 持久化。
- 前端知识库、文档、索引状态、会话、错误状态和引用来源展示。

## 发布证据

| 项目 | 值 |
| --- | --- |
| GitHub 源码提交 | `fe2ac23f138397fae8a0b4a257fec00ab7fd1be3` |
| GitHub CI | Run `36536543292`，通过 |
| HF Space 部署提交 | `8770e28ec88e5d7797e622117cd749eace20f168` |
| `/health` | `build_commit` 与 HF 部署提交一致，版本 `0.2.0` |
| `/ready` | 200；四项依赖检查均为 `ok` |
| 线上验收 | `passed=true` |
| 线上验收报告 | `eval/runs/release_acceptance/release-acceptance-20260929T075433283381Z.json` |
| 本地工程测试 | 用户运行：374 passed、9 skipped、6 warnings |

线上验收真实完成了注册、登录、PDF 上传、索引任务完成、active generation、会话提问、sources 展示和历史消息读取。索引耗时约 25.7 秒，回答耗时约 55.6 秒，返回 3 个来源，来源页为 1、2、9。

## 稳定性与效果证据

固定单轮数据集共有 20 题。最新运行报告为：

`eval/runs/generation/generation-single_turn-20260929T085541756035Z/summary.json`

结果为 20/20 执行成功、无 HTTP 500。效果指标为：证据命中率 0.824、证据召回 0.632、无答案误报率 0.333、任务路由准确率 0.800，P50/P95 延迟约 109.8/199.9 秒。Judge 20/20 完成：正确性 0.855、完整性 0.786、忠实性 0.989、引用正确性 0.933；尚未进行人工评分。低分案例为 COT-01、COT-03、COT-TERM-01、MULTI-SUMMARY-01。

此前一次 5/20 运行记录到 MongoDB replica set 连接故障，作为历史 Reliability 风险保留。当前版本可以用于展示和面试演示，不能写成“生产稳定”或“答案全部正确”。

## 已知限制

- 固定效果集规模较小，未覆盖足够多的复杂表格、图片和 OCR 样本。
- Summary、跨文档证据覆盖和长回答延迟仍需继续观察。
- 尚无 10K/100K/500K chunks 容量、并发和长期持久化基准。
- Ingestion worker 仍在应用进程内运行。
- 免费 Space 的本地文件和 Chroma 目录不保证长期持久化。
- JWT 尚未提供 HttpOnly Cookie、Refresh Token、吊销和 RBAC。

## 复现入口

```powershell
Set-Location D:\\MedRag
D:\\Anaconda\\envs\\medrag\\python.exe -m compileall -q medrag\\backend eval tests
git diff --check
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.runners.release_acceptance `
  --base-url https://maoxiao-1205-medrag.hf.space `
  --expected-build-commit 8770e28ec88e5d7797e622117cd749eace20f168 `
  --preflight-only
```
