---
title: DocRAG Agent
emoji: 📚
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
short_description: 面向 PDF 文献阅读的 RAG 智能助手
app_port: 7860
---

# DocRAG Agent：面向 PDF 文献阅读的 RAG 智能助手

DocRAG Agent 是一个面向学术论文、技术文档、课程资料和项目报告的 PDF 文献阅读助手。系统以检索增强生成（RAG）为主链路，支持文档解析、文本切分、向量检索、重排、多轮问答、引用来源展示和文档级历史会话。

项目重点不是单纯调用大模型，而是实现一条可调试、可评估、可持久化的文档问答流程。系统回答仅用于辅助阅读，重要结论应回到原始文档和引用片段核验。

## 在线演示

- Hugging Face Space：<https://maoxiao-1205-medrag.hf.space/>
- 免费实例可能休眠，首次访问需要等待模型和服务启动。
- 演示环境受免费实例资源和本地磁盘持久性限制，请勿上传敏感文件。

## 当前能力

### 文档处理

- 上传 PDF，并使用 PyMuPDF 按页提取文本和页码。
- 通过 `chunk_size`、`chunk_overlap` 切分文本，并保留页码、块序号、文档标识等 metadata。
- 计算 PDF 内容的 SHA-256 `document_hash`。
- 通过 `user_id + document_hash` 识别同一用户重复上传的相同文件，避免重复解析和索引。
- 对疑似扫描页支持可选的 Tesseract 中英文 OCR 回退。
- 记录文档主要语言，用于跨语言 Query Rewrite。

### 检索与生成

- 使用 `BAAI/bge-small-zh-v1.5` 生成文本向量。
- 使用 Chroma 持久化向量，并通过 `user_id + document_id` 过滤检索范围。
- 先扩大向量候选池，再使用 `BAAI/bge-reranker-base` 对 `query + chunk` 重排。
- 保存并展示回答对应的 sources，包括页码、chunk 和检索分数。
- 检索无证据时直接拒答，不调用 LLM 基于参数知识补充答案。
- 对 OpenRouter/OpenAI-compatible API 的超时、限流、连接失败和异常状态进行分类处理。

### 用户与多轮会话

- 支持轻量用户创建或选择，并保存默认 `user_type`。
- 文档和会话按 `user_id` 进行逻辑隔离。
- 同一文档可创建多个 conversation，也可继续历史会话。
- user 和 assistant 消息持久化到 MongoDB。
- assistant 消息保存 `sources`、`rewritten_query`、`retrieval_queries` 和 `task_type`。
- 每轮问答读取最近若干条消息作为短期上下文，不无限拼接全部历史。
- Query Rewrite 将含指代的追问改写为可独立检索的问题；失败时回退原始 query。
- `user_type` 支持普通用户、本科生、研究生、科研人员、开发者和教师等回答侧重点。

### 任务路由

当前使用轻量规则 Router：

- 包含“总结、摘要、概括、归纳”时进入 `summary`。
- 包含“阅读报告、分析这篇文献”时进入 `report`。
- 其他问题进入普通 `qa`。

总结任务使用 Summary Multi-Query：根据初始证据生成多个检索子查询，分别召回、重排、去重并控制同页片段数量，再组装总结上下文。增强流程失败时回退原查询，避免破坏基础问答链路。

## 系统流程

```text
创建或选择用户
  → 上传 PDF
  → 计算 document_hash，检查重复文档
  → PyMuPDF 按页解析，必要时 OCR
  → 文本切分与 metadata 构造
  → Embedding
  → Chroma 向量持久化
  → 创建或恢复文档会话
  → 读取最近历史
  → Query Rewrite
  → Agent Router 识别 qa / summary / report
  → 向量召回 + CrossEncoder Rerank
  → 组装 Prompt
  → 调用 LLM
  → 保存 answer、sources 和检索调试信息
  → 前端展示回答与可折叠引用
```

更完整的模块说明见：

- [项目说明](medrag/docs/项目说明.md)
- [系统架构图](medrag/docs/系统架构图.md)
- [评估集说明](eval/rag_dataset/README.md)

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 后端接口 | Python 3.11、FastAPI、Uvicorn |
| PDF 解析 | PyMuPDF |
| 扫描页 OCR | Tesseract OCR |
| Embedding | Sentence Transformers、`BAAI/bge-small-zh-v1.5` |
| Reranker | CrossEncoder、`BAAI/bge-reranker-base` |
| 向量数据库 | Chroma |
| 业务数据 | MongoDB、PyMongo Async API |
| 大模型 | OpenRouter / OpenAI-compatible API |
| 前端 | HTML、CSS、JavaScript |
| 测试与评估 | pytest、pytest-asyncio、自建 RAG 小型评估集 |
| 部署 | Docker、Hugging Face Space |

## 目录结构

```text
MedRag/
├─ medrag/
│  ├─ backend/
│  │  ├─ routers/        # HTTP 参数、响应和状态码
│  │  ├─ services/       # 业务编排、检索、Prompt 和模型调用
│  │  ├─ stores/         # MongoDB 集合读写
│  │  ├─ prompts/        # QA、Summary、Report、Query Rewrite 模板
│  │  ├─ config.py
│  │  ├─ main.py
│  │  ├─ requirements.txt
│  │  └─ .env.example
│  ├─ frontend/
│  │  ├─ index.html
│  │  ├─ app.js
│  │  └─ styles.css
│  └─ docs/
├─ eval/
│  ├─ rag_dataset/       # 3 篇 PDF、15 道问题和 gold evidence
│  ├─ run_eval.py
│  ├─ run_follow_up_eval.py
│  ├─ calculate_metrics.py
│  └─ debug_retrieval.py
├─ tests/                # 业务层自动化回归测试
├─ runtime/              # 本地运行数据，不提交 Git
├─ Dockerfile
├─ pytest.ini
├─ requirements-dev.txt
└─ README.md
```

## 核心数据模型

### users

- `user_id`
- `username`
- `default_user_type`
- `created_at`
- `updated_at`

### documents

- `document_id`
- `user_id`
- `document_hash`
- `filename`
- `language`
- `created_at`
- `updated_at`

### conversations

- `conversation_id`
- `user_id`
- `document_id`
- `title`
- `user_type`
- `created_at`
- `updated_at`

### messages

- `message_id`
- `conversation_id`
- `role`
- `content`
- `task_type`
- `rewritten_query`
- `retrieval_queries`
- `sources`
- `created_at`

## 本地运行

### 1. 环境要求

- Python 3.11
- MongoDB
- 可选：Tesseract OCR 及 `eng`、`chi_sim` 语言包

### 2. 安装依赖

```cmd
conda activate medrag
cd /d D:\MedRag\medrag\backend
python -m pip install -r requirements.txt
```

### 3. 配置环境变量

复制配置模板：

```cmd
copy .env.example .env
```

配置示例：

```env
OPENROUTER_API_KEY=your_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=your_model
CHROMA_DIR=D:/MedRag/runtime/chroma_db
UPLOAD_DIR=D:/MedRag/runtime/uploads
MONGODB_URI=mongodb://127.0.0.1:27017
MONGODB_DB_NAME=docrag
OCR_ENABLED=false
OCR_LANGUAGES=eng+chi_sim
OCR_DPI=300
OCR_MIN_TEXT_CHARS=20
TESSDATA_PREFIX=D:/Anaconda/envs/medrag/share/tessdata
```

API Key 只能放在后端环境变量中。`.env`、上传文件、Chroma 数据和日志均不得提交到 Git。

### 4. 启动

```cmd
cd /d D:\MedRag\medrag\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

访问：

- 前端：<http://127.0.0.1:8000/>
- Swagger：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

模型已缓存且 Hugging Face 网络不可用时，可在启动前设置：

```cmd
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```

## 主要接口

| 方法与路径 | 功能 |
| --- | --- |
| `GET /health` | 服务存活检查 |
| `POST /users` | 创建或获取轻量用户 |
| `POST /pdf/parse` | 解析 PDF 并返回分页文本 |
| `POST /pdf/chunks` | 查看切分结果 |
| `POST /pdf/index` | 去重、解析、切分、向量化并建立索引 |
| `POST /pdf/search` | 调试指定用户和文档的检索结果 |
| `POST /pdf/prompt-preview` | 预览单轮 RAG Prompt |
| `POST /pdf/answer` | 单轮 RAG 问答兼容接口 |
| `POST /conversations` | 基于用户和文档创建会话 |
| `GET /conversations` | 按用户或文档获取会话列表 |
| `GET /conversations/{id}/messages` | 校验用户归属并获取历史消息 |
| `POST /conversations/{id}/ask` | 执行多轮 Query Rewrite、检索和回答 |

请求与响应的完整字段以 Swagger 为准。

## 自动化测试

安装开发依赖：

```cmd
cd /d D:\MedRag
D:\Anaconda\envs\medrag\python.exe -m pip install -r requirements-dev.txt
```

运行：

```cmd
D:\Anaconda\envs\medrag\python.exe -m pytest -q
```

当前业务层回归测试覆盖：

- 不指定文档时查询用户全部会话。
- 拒绝空 `document_id`。
- 同一用户和相同 `document_hash` 复用已有文档。
- Query Rewrite 调用失败时回退原问题。
- assistant 消息保存 sources。

这些是 Mock 单元测试，不替代 MongoDB、Chroma、真实模型和完整接口的集成测试。

## RAG 评估

评估集包含 3 篇公开论文、15 道问题，覆盖事实、术语、对比、总结、无答案和多轮追问。

```cmd
cd /d D:\MedRag
python eval\rag_dataset\validate_dataset.py
python eval\run_eval.py
python eval\run_follow_up_eval.py
python eval\calculate_metrics.py
```

当前仓库保存的基线结果：

| 指标 | 数值 |
| --- | ---: |
| Execution Success Rate | 1.0000 |
| HitRate@3 | 0.6667 |
| Recall@3 | 0.5972 |
| MRR@3 | 0.5417 |

该结果说明接口链路能够完成，但总结问题、多证据覆盖和部分 LoRA/追问题仍有改进空间。结果文件早于最近的 Summary Multi-Query 修改，冻结新版本前必须重新执行评估，不能把旧指标描述为最新优化效果。

## 工程边界

- 当前是轻量用户模式，`user_id` 由前端保存并传入，不等同于密码登录、Session 或 JWT 鉴权。
- `user_id + document_id` 用于资源归属和逻辑隔离，但客户端仍可能伪造 `user_id`。
- OCR 只解决扫描页文字提取；复杂表格、公式、图表语义和普通图片尚未接入视觉模型。
- Chroma 使用本地文件持久化，免费云实例重启后可能丢失索引。
- `/health` 当前属于存活检查，不验证 MongoDB、Chroma、模型和 LLM 供应商的可用性。
- 当前评估集规模较小，适合回归和基线比较，不代表生产环境效果。
- 现阶段缺少系统化接口集成测试、真实鉴权、监控指标、并发测试和容量测试。

## 后续计划

1. 重跑最新评估并分析 Summary、LORA-03、LORA-05 的候选召回和重排过程。
2. 增加接口契约、MongoDB/Chroma 集成测试和线上冒烟测试。
3. 补充结构化日志、请求级追踪和依赖服务 readiness。
4. 在需要正式多用户服务时引入真实登录态与 JWT/Session 鉴权。
5. 通过受控对照实验评估 Hybrid Search、Reranker 参数和多语言 Embedding，而不是直接替换生产链路。

## 项目声明

本项目用于学习和演示 PDF 文献 RAG 工程流程。系统生成内容不构成医学、法律、投资或其他专业建议。
