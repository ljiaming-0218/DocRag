# DocRAG v1.1.0 结项记录

## Harness 本轮收尾补充（2026-10-18）

**本地演示版完成收尾**：研究运行循环、独立 worker、持久化与恢复、连续研究和历史删除已经落地；H4 三题九组真实模型对照已保存，八组有文本或报告（含一组部分完成），一组供应商 429 失败。

交付入口：[README](../../README.md)、[项目说明](项目说明.md)、[演示脚本](demo-script.md)。本轮未提交新的发布版本或部署到 Hugging Face；未重跑全量 pytest。按本轮约定，后续生产质量和容量工作不作为演示版结项门槛。

---

以下保留 **2026-09-29 基础 RAG v1.1.0 历史结项记录**，其中提交号、CI、线上验收与测试数量不代表本轮 Harness 状态。

历史记录更新时间：2026-09-29

## 结论

当前版本可以作为 **演示候选/条件结项版本** 归档，不能作为严格生产发布版本归档。

原因是主业务闭环已在真实 Hugging Face Space 完成验收，最新固定 20 题也已 20/20 执行成功并完成 Judge；但证据召回为 0.632、P95 约 199.9 秒，Judge 正确性 0.855、完整性 0.786。拒答启发式将 RAG-01 误标为拒答（Judge 四项均为 1.0），LORA-04 是实际无答案误答，因此只能作为演示候选归档。此前一次 5/20 的 HTTP 500 运行与 MongoDB 连接故障相关，作为历史 Reliability 风险保留。

## 证据矩阵

| 门槛 | 结果 | 证据 |
| --- | --- | --- |
| 代码与 CI | 通过 | GitHub `fe2ac23f138397fae8a0b4a257fec00ab7fd1be3`；CI run `36536543292` |
| 本地工程测试 | 通过当前用户运行结果 | `374 passed, 9 skipped, 6 warnings` |
| 版本一致性 | 通过 | HF `8770e28ec88e5d7797e622117cd749eace20f168`；`/health.build_commit` 一致 |
| 线上就绪 | 通过 | `/ready` 200；MongoDB、Chroma、上传目录、LLM 配置均 ok |
| 线上业务闭环 | 通过 | 注册、登录、PDF 上传、job completed、active generation、提问、sources、历史消息 |
| 固定 20 题稳定性 | 执行层通过，质量仍需观察 | 20/20 成功、无 HTTP 500；报告见 `eval/runs/generation/generation-single_turn-20260929T085541756035Z/` |
| LLM Judge | 完成，辅助通过 | 正确性 0.855、完整性 0.786、忠实性 0.989、引用正确性 0.933；报告见 `eval/runs/judge/judge-20260929T094048797812Z/` |

线上验收报告：`eval/runs/release_acceptance/release-acceptance-20260929T075433283381Z.json`。

## 可以对外声明

- 已实现可部署的多文档 PDF RAG 主链路。
- 已实现 JWT 登录、资源归属校验、持久化索引任务和索引代次切换。
- 已实现 Dense/Hybrid 检索、RRF、CrossEncoder Rerank、Query Rewrite、Summary Multi-Query 和 sources 持久化。
- GitHub CI 通过，线上真实注册到问答的业务链路通过。
- 当前版本定位为演示候选，稳定性回归仍有已记录的 MongoDB 依赖故障。

## 不可以声明

- 20/20 稳定通过。
- 生产级高可用、企业级并发或高精度 RAG。
- 固定 20 题的回答全部正确。
- 已完成容量、长期持久化和复杂版面理解验证。

## 后续唯一必要动作

如果要升级为严格发布：

1. 人工抽检 Judge 低分案例 COT-01、COT-03、COT-TERM-01、MULTI-SUMMARY-01，以及无答案误答 LORA-04。
2. 修正拒答检测口径，避免把“文档未明确说明某个子点”当成整题拒答；再审查 CoT 术语 Prompt、Summary 证据覆盖和跨文档上下文组织。
3. 补充延迟、容量和复杂 PDF 基准；若后续出现 HTTP 500，再按错误阶段和依赖状态复现。

本次已完成固定 20 题补跑，当前文档按最新 20/20 报告归档；此前 5/20 运行仍作为历史依赖故障记录。

## 复核命令

```powershell
Set-Location D:\\MedRag
git status --short --branch
git rev-parse HEAD
D:\\Anaconda\\envs\\medrag\\python.exe -m compileall -q medrag\\backend eval tests
git diff --check
```

线上验收复核：

```powershell
$releaseCommit = '8770e28ec88e5d7797e622117cd749eace20f168'
D:\\Anaconda\\envs\\medrag\\python.exe -m eval.runners.release_acceptance `
  --base-url https://maoxiao-1205-medrag.hf.space `
  --expected-build-commit $releaseCommit `
  --preflight-only
```
