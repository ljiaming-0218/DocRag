# DocRAG 当前版本结项执行单（2026-09-29）

本执行单以 `75d1f3ff219d44e3d895de530b5359390d2c5807` 为起点。只有下列门槛全部通过，才把后续确定的提交标为新发布版本；历史 Stage B 结果不当作新提交的测试结果。

## 现状与结项门槛

| 门槛 | 当前证据 | 通过标准 |
| --- | --- | --- |
| G1 代码与 CI | 用户于 2026-09-29 在以 `75d1f3f` 为基底、含未提交改动的工作树运行全量 pytest：374 passed、9 skipped、6 warnings；此前 2 个失败用例的定向重跑均通过。结果由用户终端输出提供，尚无对应发布提交和 CI 结果。 | 当前发布提交的 CI 通过；本地同提交 pytest 通过；`compileall` 与 `git diff --check` 通过。 |
| G2 稳定性 | Stage B 后续 run 18/20；`RAG-TERM-01` 的 `POST /conversations` 和 `RAG-SOURCE-01` 的 `POST /ask` 返回 500。现存本地后端日志早于该 run，无法从日志确认根因。2026-09-29 本地同配置重跑两题均成功，分别约 70.5 秒和 95.9 秒，未复现偶发 500。 | 保留服务端 traceback 和错误阶段；再跑固定 20 题，无未解释的 HTTP 500；任何失败均进入逐题 bad case。 |
| G3 版本一致性 | GitHub HEAD 为 `75d1f3f`，Space 仓库曾推送 `7716e56`；2026-09-29 再查线上 `/health.build_commit` 仍为 `18175efd5eb9d88e39e13afb69cf361ef1202926`。同次 `/ready` 首次 502、立即重试 200（MongoDB、Chroma、上传目录、LLM 配置均 ok），需排查偶发网关错误。 | GitHub 发布提交、Space 部署来源、`/health.build_commit` 三者的映射可复核；`/ready` 稳定返回 200。 |
| G4 线上业务闭环 | 尚无当前发布提交的真实线上上传和问答报告。 | 独立验收用户完成注册、登录、PDF 上传、job completed、active generation、会话提问、来源与历史消息；报告记录提交与耗时。 |
| G5 发布说明 | `release-v1.0.0.md` 是历史版本。 | 新发布说明准确写出当前提交、评估样本与指标、已知限制、验收报告和部署环境边界。 |

## 执行顺序

1. **固定候选提交并核对 CI。** 保持对 Retrieval、Generation 主流程的改动最小。用户已在当前工作树运行全量 pytest，结果 374 passed、9 skipped；这尚不等于未来发布提交的测试结果。提交后核对 CI 对应的 commit SHA 与结果，必要时在同一提交复跑。
2. **复现并定位本地 500。** 启动同配置后端并保存 stderr/stdout；先分别运行 `RAG-TERM-01`、`RAG-SOURCE-01`，记录请求时间、HTTP 响应、服务端 traceback、MongoDB 和模型供应商错误。`POST /conversations` 不调用 LLM，因此不能用“模型限流”单独解释两处故障。只有找到具体异常分支才改业务代码；改后重跑两题与 20 题固定集。
3. **同步并核验线上版本。** 发布提交先进入 GitHub，再在基于 `space/main` 的隔离工作树中只同步部署所需源码；不执行 force push。核对 Space 构建/运行环境中的 `BUILD_COMMIT`，使 `/health` 返回实际发布源码提交。确认线上与本地使用不同的 MongoDB 数据库及 Chroma/上传目录。
4. **执行真实线上验收。** 先运行只读预检；版本一致后运行 `eval.runners.release_acceptance` 完整流程。一次成功只证明这条工程业务链路可用，不等于 RAG 效果达标；效果仍引用固定数据集和逐题分析。
5. **冻结发布说明。** 从当前提交、CI、固定集和线上报告抄录实际值，列出 Summary/跨文档证据覆盖、复杂 PDF 结构解析、延迟和容量未验证等限制。最后才创建发布 tag。

## 命令（PowerShell）

```powershell
Set-Location D:\MedRag
git status --short --branch
git rev-parse HEAD
D:\Anaconda\envs\medrag\python.exe -m compileall -q medrag\backend eval tests
git diff --check
```

用户已在当前工作树运行过 pytest；如发布提交形成后需要复跑，命令为：

```powershell
D:\Anaconda\envs\medrag\python.exe -m pytest -q
```

固定题重现前先启动本地后端并保存日志；随后：

```powershell
D:\Anaconda\envs\medrag\python.exe -m eval.runners.generation --suite single_turn --case-id RAG-TERM-01 --case-id RAG-SOURCE-01 --request-interval 0
```

线上预检不写入服务端。完整验收会创建独立测试用户、文档与会话，报告保存在被 Git 忽略的 `eval/runs/release_acceptance/`：

```powershell
$releaseCommit = '<最终发布源码的完整 SHA>'
D:\Anaconda\envs\medrag\python.exe -m eval.runners.release_acceptance --base-url https://maoxiao-1205-medrag.hf.space --expected-build-commit $releaseCommit --preflight-only
D:\Anaconda\envs\medrag\python.exe -m eval.runners.release_acceptance --base-url https://maoxiao-1205-medrag.hf.space --expected-build-commit $releaseCommit
```

## 新版本说明草案边界

可声明：多文档知识库、版本化索引、Hybrid 检索、JWT 登录、持久化索引任务及前端状态展示已在源码实现；Stage B 在冻结快照上完成 3 篇 PDF、20 道单轮题的正式 Generation/Judge 归档，并另有 5 道追问样本。

必须同时声明：20/20 是正式 run 的执行与 Judge 完成情况，不是回答全对；归档证据召回约 0.662，P95 约 185 秒；另一次 run 18/20 的 500 尚待定位；当前发布提交的效果、线上业务闭环和容量表现要以新验收结果为准。不得把本执行单写成已发布说明。
