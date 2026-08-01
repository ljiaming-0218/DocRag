# DocRAG Agent v1.0.0

## 版本定位

`v1.0.0` 是 DocRAG Agent 的首个稳定演示版本，用于 PDF 文献辅助阅读、RAG 流程学习和大模型应用开发项目展示。该版本优先保证主链路稳定、回答可追溯和结果可评估，不以堆叠功能为目标。

## 核心能力

- 使用 PyMuPDF 按页解析 PDF，并对疑似扫描页提供可选 OCR 回退。
- 按 `chunk_size` 和 `chunk_overlap` 切分文本，保留页码、chunk 序号等 metadata。
- 使用 Embedding、Chroma 向量检索和 CrossEncoder Rerank 构建两阶段检索链路。
- 使用 `user_id` 隔离文档和会话，通过 `document_hash` 识别重复上传。
- 使用 MongoDB 持久化用户、文档、会话、消息及回答对应的 sources。
- 使用最近对话历史和 Query Rewrite 支持多轮追问；改写失败时回退原始问题。
- 使用规则 Agent Router 路由普通问答、总结、阅读报告、术语解释和证据核查任务。
- 对 LLM 超时、连接失败、限流和异常状态进行分类处理。
- 提供自动化测试、前端冒烟测试和小型 RAG 评估集。

## 测试基线

| 项目 | 结果 |
| --- | ---: |
| 自动化测试 | 75 passed |
| 评估文档 | 3 篇公开论文 |
| 评估问题 | 19 道 |
| 执行成功率 | 0.8421 |
| HitRate@3 | 0.7692 |
| Recall@3 | 0.6026 |
| MRR@3 | 0.6410 |

本轮 16 道非追问题执行成功；3 道追问题在连续评估后被免费模型供应商限流并返回 503，因此执行成功率下降不能直接归因于 Query Rewrite。指标来自 `eval/rag_dataset/results/metrics_summary.json`。

## 工程问题与处理

- **CORS**：显式配置允许的前端来源，并在部署环境使用同源访问，避免浏览器拦截请求。
- **Chroma HNSW 索引损坏**：将运行目录移出源码目录；损坏后重建向量索引，不把本地持久化数据提交到 Git。
- **Query Rewrite 失败**：保存原始问题和改写问题，限制改写只补全指代；调用失败时回退原始 query。
- **总结任务证据覆盖不足**：使用 Summary Multi-Query 扩大候选覆盖，再统一去重和 Rerank。
- **外部模型不稳定**：区分连接、超时、限流和 API 状态异常，避免把供应商故障误判为 RAG 质量问题。

## 已知限制

- 首次索引大型 PDF 可能耗时较长，当前没有后台任务进度展示。
- 免费 LLM 服务可能出现限流、休眠或冷启动。
- 当前轻量用户模式依赖前端保存并传递 `user_id`，不等同于正式登录鉴权。
- 总结题和对比题仍存在多证据召回不完整、正确证据排序靠后的情况。
- OCR 主要用于文本回退，尚未实现表格结构和图片语义理解。

## 版本信息

- GitHub Tag：`v1.0.0`
- GitHub Commit：`99e3281`
- Hugging Face Commit：`5ef2192`
- 在线演示：<https://maoxiao-1205-medrag.hf.space/>
