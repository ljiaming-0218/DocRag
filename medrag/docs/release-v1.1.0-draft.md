# DocRAG v1.1.0 发布说明草案

**状态：候选，尚未发布；不得将本文件当作线上验收结果。**

## 本版范围

- 多文档知识库、JWT 登录与用户资源归属校验。
- 固定/递归切片、索引指纹、独立索引代次与 active generation 切换。
- Dense + BM25 + RRF + CrossEncoder Rerank、证据阈值、Query Rewrite、Summary Multi-Query。
- MongoDB 持久化索引任务，`POST /pdf/index` 返回 `202 + job_id`，提供状态查询、失败重试与前端进度显示；有效索引可返回 200 复用。

## 已有评估证据

- Stage B 在冻结源码快照上，以 3 篇公开 PDF、20 道单轮题形成正式 Generation/Judge 归档：20/20 执行完成并得到 Judge 结果。该数字不表示 20 道答案全部正确。
- 正式归档的 Citation Evidence Recall 约 0.662，问答 P95 约 185 秒；Judge 的平均分和逐题问题以 `eval/runs/stage_b_closure/stage-b-closure-20260925T071022Z/` 为准。
- 评估集另有 5 道追问题。2026-09-25 后续 20 题 run 有两次本地 HTTP 500，成功 18/20；它没有覆盖正式基线，根因尚未确定。
- 2026-09-29 本地重跑原先失败的 `RAG-TERM-01`、`RAG-SOURCE-01`，两题均成功；耗时约 70.5 秒、95.9 秒。此结果只说明该次未复现偶发故障。
- 用户于 2026-09-29 在以 `75d1f3f` 为基底、含未提交改动的本地工作树运行全量 pytest：374 passed、9 skipped、6 warnings。此前两个失败用例定向重跑为 2 passed、1 warning。测试结果来自用户终端输出，尚不对应已冻结的发布提交或线上部署。
- 2026-09-29 再查线上 `/health.build_commit` 为 `18175efd5eb9d88e39e13afb69cf361ef1202926`，与本地候选源码不一致；`/ready` 一次返回 502，立即重试返回 200。该波动原因尚未确定。

## 发布前必须填入的证据

| 项目 | 待填值 |
| --- | --- |
| GitHub 发布源码 SHA | 待 CI 通过后确定 |
| GitHub CI URL 与结果 | 待新提交触发并通过 |
| 候选工作树本地 pytest | 2026-09-29 用户运行：374 passed、9 skipped、6 warnings；发布提交形成后仍需核对 CI |
| 固定 20 题回归报告 | 待执行 |
| Hugging Face 部署提交 SHA | 待同步后确定 |
| `/health.build_commit` | 必须等于 GitHub 发布源码 SHA |
| `/ready` 和线上业务验收报告 | 待执行 `eval.runners.release_acceptance` |
| 本地与线上数据环境隔离核对 | 待确认配置，不记录 Secret 值 |

## 已知限制

- 当前以 PDF 文本提取和 OCR 回退为主，尚未验证复杂表格、图像及版面语义理解质量。
- 固定效果集规模较小；Hybrid 在现有固定集上未证明最终 Evidence Recall@3 优于 Dense + Rerank。
- Summary 和跨文档问题仍有证据覆盖不全的 bad case。
- 线上容量、并发、持久磁盘可靠性与长期可用性尚无基准。
- 索引 worker 在 FastAPI 应用进程内启动；MongoDB Job 状态可恢复，不等同于已部署独立 worker 服务。

结项门槛与执行命令见 `medrag/docs/release-closeout-plan.md`。以上待填项完成后，才能将本草案改为正式发布说明并创建 tag。
