# AGENTS.md

## 1. 项目身份

本项目名称为 **DocRAG / MedRAG V2**。

项目当前已经从最初的医学文献问答 Demo 演进为通用文档 RAG Agent。

当前目标不是继续堆叠表面功能，而是逐步升级为：

> 一个具备文档解析、结构化切片、索引版本管理、多文档知识库、混合检索、Rerank、会话记忆、Query Rewrite、Agent Router、离线评估与工程化监控能力的可部署 RAG 系统。

当前项目根目录：

```text
D:\MedRag
```

主要源码目录：

```text
D:\MedRag\medrag
```

评估目录：

```text
D:\MedRag\eval
```

Codex 在本项目中工作时，始终将：

```text
D:\MedRag
```

视为项目根目录。

---

# 2. 当前项目阶段

当前项目已经达到：

> 完整可演示的单文档 RAG 应用

当前不是入门 Demo，但暂时不要描述为“企业级多文档知识库”。

现有能力包括：

* PDF 上传
* PDF 文本解析
* OCR fallback
* 文档切片
* Embedding
* Chroma 向量检索
* CrossEncoder Rerank
* Query Rewrite
* 多轮会话短期记忆
* conversation_id
* 用户级逻辑隔离
* 历史会话
* sources 持久化
* Agent Router
* QA
* Summary
* Term Explanation
* Reading Report
* Source Check
* 用户类型 Prompt
* MongoDB 数据持久化
* 异常处理
* 日志
* 离线 RAG 评估集
* Recall / MRR / 引用准确性等指标
* 自动化测试

当前稳定测试基线：

```text
76 passed
```

后续任何修改都不能无理由破坏现有测试基线。

---

# 3. 当前核心问题

当前 DocRAG 最大短板不是 LLM，而是：

```text
文档处理
↓
Chunking
↓
索引生命周期
↓
多文档知识库
↓
Retrieval
↓
Evaluation
```

当前开发优先级：

```text
P0
切片 V2
索引版本管理
多文档知识库
Chunk 级评估

P1
结构感知 PDF 解析
Hybrid Retrieval
文档处理状态
认证系统

P2
VectorStore 抽象
容量测试
Batch Embedding
模型延迟加载
DTO 化
工程质量优化
```

禁止跳过 P0，直接为了“技术看起来高级”去做 P2。

---

# 4. Codex 总体工作原则

在任何代码修改之前：

1. 读取当前目标文件。
2. 搜索对应调用链。
3. 查看相关测试。
4. 检查配置。
5. 查看数据结构。
6. 判断修改影响范围。
7. 给出简短实施方案。
8. 再修改代码。

默认工作流程：

```text
Inspect
↓
Plan
↓
Implement
↓
Test
↓
Stabilize
↓
Continue Phase
↓
Phase Evaluation
```

禁止：

* 没有读取代码就直接重写。
* 为了解决局部问题进行大面积重构。
* 无理由替换已有技术栈。
* 同时修改多个核心变量。
* 删除稳定功能以换取实现方便。
* 为了“企业级”盲目引入复杂组件。
* 为了部署免费平台删除 Rerank 等核心能力。
* 为了赶进度跳过测试。

---

# 5. 当前开发策略

当前采用：

> Implementation First → Stabilization → Evaluation

即：

```text
先完成一个完整 Phase
↓
保证代码和测试稳定
↓
再统一进行实验验证
```

不要每修改一个小功能就运行完整 RAG 实验。

开发过程中主要进行：

```text
Code
+
Unit Tests
+
Integration Tests
+
Regression Tests
```

阶段完成后再进行：

```text
Offline RAG Evaluation
```

---

# 6. 测试与实验必须区分

下面属于工程测试：

```text
pytest
```

下面属于效果实验：

```text
Fixed vs Recursive
Dense vs Hybrid
Recall@K
MRR
nDCG
Citation Accuracy
Faithfulness
```

开发过程中可以频繁运行测试。

不要频繁运行完整 RAG 实验。

当前基线：

```text
76 passed
```

已有测试出现 regression 时：

```text
禁止宣布任务完成
```

必须定位并修复。

---

# 7. 当前 V2 Roadmap

严格按以下阶段推进。

## Phase 1

```text
Chunking V2
+
Index Versioning
```

包含：

```text
fixed strategy
recursive strategy
chunk metadata
index fingerprint
force_reindex
相关测试
```

阶段稳定之后：

```text
统一运行 Fixed vs Recursive 实验
```

---

## Phase 2

```text
Knowledge Base
+
Multi-document Retrieval
```

包含：

```text
knowledge_bases
kb_id
多文档上传
知识库级检索
指定文档范围检索
用户隔离
前端知识库管理
```

阶段稳定后：

```text
统一验证多文档检索
```

---

## Phase 3

```text
Hybrid Retrieval
```

包含：

```text
Dense Retrieval
BM25 / Sparse Retrieval
RRF
Candidate Dedup
CrossEncoder Rerank
Evidence Threshold
```

阶段稳定后统一对比：

```text
Dense

Dense + Rerank

Hybrid + RRF + Rerank
```

---

## Phase 4

```text
Document Processing Engineering
```

包含：

```text
processing status
failure state
batch embedding
batch vector write
retry
layout metadata
logging
observability
```

---

## Phase 5

```text
Capacity Benchmark
+
VectorStore Decision
```

规模：

```text
10K chunks
100K chunks
500K chunks
```

记录：

```text
index time
write throughput
P50
P95
P99
memory
disk
concurrency error
```

达到真实瓶颈后再考虑：

```text
Qdrant
Milvus
```

---

# 8. 当前立即执行任务

现在只执行：

```text
Phase 1
```

当前第一主线：

```text
Chunking V2
+
Index Versioning
```

重点文件：

```text
medrag/backend/services/text_splitter_service.py
medrag/backend/services/document_ingestion_service.py
medrag/backend/services/vector_store_service.py
tests/
eval/
```

当前阶段不要主动修改：

```text
Knowledge Base
Hybrid Retrieval
Vector Database
Frontend 大改
GraphRAG
Multi-Agent
```

---

# 9. 当前切片问题

重点检查：

```text
medrag/backend/services/text_splitter_service.py
```

当前 fixed splitter 本质类似：

```python
text[i:i + chunk_size]
```

存在问题：

* 截断句子。
* 截断段落。
* 截断标题。
* 截断表格。
* 无法识别章节。
* 跨页内容处理较弱。
* chunk 缺少结构化 metadata。

当前 fixed splitter 必须保留。

原因：

> Fixed Chunk 是后续效果对比的 baseline。

禁止直接删除 fixed。

---

# 10. Chunk Strategy 设计

当前需要设计统一切片接口。

推荐策略：

```text
fixed
recursive
```

未来再扩展：

```text
structure
parent_child
semantic
```

但当前 Phase 1 只实现：

```text
fixed
recursive
```

---

# 11. Chunk 数据结构

推荐逐步统一 Chunk 数据结构。

建议字段：

```text
chunk_id
document_id
user_id
text
page_start
page_end
start_char
end_char
section_title
strategy
chunk_size
chunk_overlap
chunk_version
parent_chunk_id
metadata
```

第一阶段至少稳定提供：

```text
text
page_start
page_end
start_char
end_char
strategy
chunk_version
```

不是所有未来字段都必须在第一轮实现。

不要为了追求一次性完整而过度设计。

---

# 12. Fixed Strategy

保留现有：

```text
strategy = fixed
```

作用：

```text
Regression Baseline
Retrieval Baseline
Chunking Baseline
```

Fixed 的现有语义应尽量保持不变。

新增 Recursive 不能导致 Fixed 出现非预期行为变化。

---

# 13. Recursive Strategy

新增：

```text
strategy = recursive
```

推荐优先级：

```text
\n\n
↓
\n
↓
。！？；
↓
.!?;
↓
空格
↓
字符
```

目标是尽量保持：

```text
章节
段落
句子
```

完整。

只有上一级 separator 无法将文本控制到目标长度时，才继续向下递归。

必须支持：

* 中文。
* 英文。
* 中英文混排。
* 无标点文本。
* 超长句子。
* 极短文本。
* 多段落。
* 多章节。

---

# 14. Chunk Overlap

Fixed 和 Recursive 都要正确处理：

```text
chunk_overlap
```

Overlap 必须：

* 不大于 chunk_size。
* 不产生无限循环。
* 不产生大量重复 chunk。
* 不导致空 chunk。
* 不破坏 metadata。

必须增加边界校验。

例如：

```text
chunk_overlap >= chunk_size
```

需要明确处理。

---

# 15. Chunking Tests

新增或完善：

```text
tests/test_text_splitter_service.py
```

至少覆盖：

## Case 1

中文普通段落。

验证：

* chunk 非空。
* 长度合理。
* Recursive 尽量不截断句子。

## Case 2

英文论文摘要。

验证英文标点边界。

## Case 3

中英文混排。

例如：

```text
RAG 的核心是 Retrieval-Augmented Generation。
The retriever retrieves relevant chunks.
```

## Case 4

超长单句。

验证最终字符 fallback。

## Case 5

多个章节。

例如：

```text
1 Introduction

...

2 Related Work

...
```

## Case 6

跨页内容。

允许：

```text
page_start != page_end
```

## Case 7

表格文本。

至少避免明显破坏整行。

## Case 8

空字符串。

## Case 9

文本长度小于 chunk_size。

## Case 10

非法 overlap。

---

# 16. 索引版本管理

重点检查：

```text
medrag/backend/services/document_ingestion_service.py
```

当前禁止继续使用：

```text
发现已有 chunks
=
永远跳过索引
```

作为最终逻辑。

原因：

未来修改：

```text
Chunk Strategy
Embedding Model
Chunk Size
Chunk Overlap
Chunk Version
```

都可能使已有索引过期。

---

# 17. Index Fingerprint

需要设计：

```text
index_fingerprint
```

推荐包含：

```text
chunk_strategy
chunk_size
chunk_overlap
chunk_version
embedding_model
embedding_version
index_version
```

逻辑类似：

```python
fingerprint = hash(
    chunk_strategy
    + chunk_size
    + chunk_overlap
    + chunk_version
    + embedding_model
    + embedding_version
    + index_version
)
```

要求：

```text
old_fingerprint == new_fingerprint
```

才可以安全复用索引。

如果：

```text
old_fingerprint != new_fingerprint
```

需要重新索引。

---

# 18. Index Version

需要维护显式：

```text
index_version
```

Index Version 代表整体索引 schema 或索引流程版本。

例如：

```text
index_version = v1
```

未来修改核心索引逻辑后：

```text
index_version = v2
```

不要只依赖代码变化判断。

---

# 19. Chunk Version

Chunk Strategy 需要：

```text
chunk_version
```

例如：

```text
fixed:v1
recursive:v1
```

未来 Recursive 算法改变：

```text
recursive:v2
```

即可触发重新索引。

---

# 20. Embedding Version

Embedding 模型属于索引的一部分。

修改：

```text
embedding model
```

必须认为旧索引可能失效。

原因：

```text
Chunk
↓
Embedding Model
↓
Vector Space
```

不同模型产生的向量空间不能直接视为同一索引。

Index Fingerprint 必须包含：

```text
embedding_model
```

必要时还包含：

```text
embedding_revision
embedding_dimension
```

---

# 21. Force Reindex

需要支持：

```text
force_reindex
```

语义：

```text
force_reindex = false
```

执行正常 fingerprint 判断。

```text
force_reindex = true
```

忽略旧索引状态，强制重建。

不要依赖：

```text
delete database manually
```

来完成重新索引。

---

# 22. Phase 1 开发顺序

严格按以下顺序：

```text
Step 1
读取当前 text_splitter_service.py

Step 2
搜索 splitter 所有调用点

Step 3
确认当前 chunk 数据结构

Step 4
设计统一 Chunk Strategy 接口

Step 5
保留 Fixed

Step 6
实现 Recursive

Step 7
增加 Chunk Metadata

Step 8
补充 Chunk Tests

Step 9
运行相关测试

Step 10
实现 Index Fingerprint

Step 11
实现 force_reindex

Step 12
补充 ingestion/index tests

Step 13
运行全量 pytest

Step 14
修复 Regression

Step 15
确认 Phase 1 工程功能完整

Step 16
再统一进行 Phase 1 Evaluation
```

不要做到 Recursive 后就停下来跑完整实验。

---

# 23. Phase 1 完成定义

Phase 1 必须完成：

```text
Fixed 可用
+
Recursive 可用
+
统一接口
+
Metadata
+
Index Fingerprint
+
force_reindex
+
单元测试
+
全量测试通过
```

然后才进入：

```text
Phase 1 Evaluation
```

---

# 24. Phase 1 实验

Phase 1 开发完成后统一验证：

```text
Fixed
vs
Recursive
```

实验时保持：

```text
相同 Evaluation Dataset
相同 Embedding
相同 Reranker
相同 Query Rewrite
相同 LLM
相同 Prompt
相同 Retrieval Top-K
```

唯一核心变量：

```text
chunk_strategy
```

评估：

```text
Recall@3
Recall@5
MRR
Citation Accuracy
Evidence Recall
Answer Accuracy
```

开发阶段不要频繁执行这些实验。

---

# 25. Parent-Child Chunk

Parent-Child 属于后续阶段。

当前不要实现。

未来设计原则：

```text
Child
=
Retrieval Unit

Parent
=
Context Unit
```

例如：

```text
Parent 1200 tokens
Child 300 tokens
```

流程：

```text
Query
↓
Retrieve Child
↓
parent_chunk_id
↓
Return Parent
```

目的：

```text
Retrieval Precision
+
Context Completeness
```

---

# 26. Semantic Chunk

Semantic Chunk 暂时只作为后续实验候选。

不要作为默认方案。

原因：

* 成本更高。
* 需要 Embedding。
* Threshold 敏感。
* Index 时间增加。
* 行为更难解释。
* 未必比 Recursive 更好。

只有后续实验明确证明收益后再考虑。

---

# 27. Phase 2：Knowledge Base

Phase 1 完成后进入：

```text
Knowledge Base
+
Multi-document Retrieval
```

资源层级：

```text
User
└── KnowledgeBase
    ├── Document
    │   └── Chunk
    └── Conversation
```

新增：

```text
knowledge_bases
```

建议字段：

```text
kb_id
user_id
name
description
created_at
updated_at
```

---

# 28. Document 与 KB 关系

Document 增加：

```text
kb_id
```

Chunk 增加：

```text
kb_id
```

Conversation 可增加：

```text
kb_id
```

最终实现：

```text
一个用户
↓
多个知识库
↓
每个知识库多个文档
```

---

# 29. Retrieval Filter 升级

当前：

```text
user_id + document_id
```

未来：

```text
user_id
+
kb_id
+
optional document_ids
```

语义：

```text
document_ids = None
```

代表：

> 检索整个知识库。

指定：

```text
document_ids = [...]
```

代表：

> 只检索选择的文档。

任何情况下：

```text
user_id
```

都不能移除。

---

# 30. 多文档知识库功能

Phase 2 目标至少支持：

* 创建知识库。
* 删除知识库。
* 重命名知识库。
* 查看知识库列表。
* 一个知识库上传多个 PDF。
* 查看知识库文档。
* 删除文档。
* 对整个知识库提问。
* 选择部分文档提问。
* Conversation 与知识库关联。
* 用户数据隔离。

---

# 31. Phase 3：Retrieval V2

Phase 2 稳定后再进入。

目标流程：

```text
Query
↓
Query Rewrite
↓
Dense Retrieval
+
Sparse Retrieval
↓
RRF
↓
Dedup
↓
CrossEncoder Rerank
↓
Threshold
↓
Context Builder
↓
LLM
```

---

# 32. Dense Retrieval

现有 Dense Retrieval 必须保留。

它仍然是：

```text
Baseline
```

不要因为增加 BM25 就删除 Dense。

---

# 33. Sparse Retrieval

Sparse/BM25 主要解决：

* 模型名称。
* 缩写。
* 公式编号。
* 专业术语。
* 专有名词。
* 文献编号。
* 精确实体。
* 数字匹配。

Dense 更擅长：

```text
Semantic Similarity
```

Sparse 更擅长：

```text
Exact Matching
```

---

# 34. RRF

Dense 与 Sparse 优先使用：

```text
Reciprocal Rank Fusion
```

不要直接：

```text
dense_score + bm25_score
```

除非已经实现可靠 score normalization。

---

# 35. CrossEncoder Rerank

当前 CrossEncoder 是项目核心能力之一。

不得因为 Hybrid Retrieval 删除。

正确职责：

```text
Dense + Sparse
=
提高 Recall

CrossEncoder
=
提高 Precision
```

流程：

```text
Retrieve More
↓
Fuse
↓
Rerank
↓
Return Few
```

---

# 36. Evidence Threshold

后续加入：

```text
relevance_threshold
```

证据不足时：

```text
拒答
```

不要默认让 LLM 使用自身知识补齐。

DocRAG 默认应该：

```text
Evidence Grounded
```

---

# 37. Summary 特殊处理

当前 Summary 类问题存在：

```text
Recall@3 较低
```

Summary 不能简单使用普通 Top-K。

未来考虑：

```text
Multi Query
+
Section Coverage
+
Document Coverage
```

例如：

```text
Introduction
Method
Experiment
Conclusion
```

分开召回。

不要简单：

```text
top_k = 50
```

解决问题。

---

# 38. PDF Parsing V2

当前：

```text
Page
→
Plain Text
```

未来逐步升级：

```text
Page
→
Layout Elements
```

可能包括：

```text
Heading
Paragraph
Table
Figure Caption
Equation
List
Header
Footer
```

metadata：

```text
block_type
bbox
page
section
section_level
```

但不要在 Phase 1 提前实现。

---

# 39. OCR

OCR 是 fallback。

正确流程：

```text
Native PDF
↓
Text Extractable?
↓
Yes → Native
No → OCR
```

不要所有 PDF 默认 OCR。

保留：

```text
ocr_used
```

等 metadata。

---

# 40. Document Processing Status

未来 Document 增加状态：

```text
pending
parsing
chunking
embedding
indexing
completed
failed
```

并记录：

```text
processing_stage
error_code
error_message
processed_at
```

避免：

```text
MongoDB 有 Document
但 VectorStore 索引不完整
```

---

# 41. Batch Embedding

未来大文档不能：

```text
all chunks
→
all embeddings
→
all vector writes
```

一次完成。

需要配置：

```text
embedding_batch_size
vector_write_batch_size
```

降低：

* OOM。
* 内存峰值。
* 单次失败损失。
* 长请求。

---

# 42. Model Lazy Loading

如果当前：

```text
Embedding
Reranker
```

在 import 时立即加载，未来逐步修改为：

```text
Application Start
↓
Service Initialization
↓
Lazy Load
```

模型不可用不应该导致整个项目无法 import。

---

# 43. VectorStore 抽象

当前不要急着迁移数据库。

先逐步抽象：

```python
class VectorStore:
    add(...)
    search(...)
    delete(...)
    delete_document(...)
    delete_knowledge_base(...)
```

当前实现：

```text
ChromaVectorStore
```

未来：

```text
QdrantVectorStore
MilvusVectorStore
```

业务逻辑尽量不要直接耦合某个数据库 SDK。

---

# 44. Chroma 迁移原则

禁止因为：

```text
Qdrant 更企业级
Milvus 更高级
```

直接迁移。

必须先 benchmark。

规模：

```text
10K
100K
500K chunks
```

记录：

```text
Index Time
Write Throughput
P50
P95
P99
RAM
Disk
Concurrent Errors
```

达到瓶颈后再决定。

---

# 45. Evaluation 原则

Evaluation 很重要，但当前采用：

```text
Phase-level Evaluation
```

而不是：

```text
Every-change Evaluation
```

即：

```text
一个 Phase 开发完成
+
测试稳定
↓
统一实验
```

---

# 46. 固定 Evaluation Dataset

长期保留一套 frozen dataset。

包括：

```text
Fact
Summary
No Answer
Terminology
Comparison
Multi-hop
Multi-document
```

后续扩展：

```text
Exact Entity
Acronym
Table QA
Cross-document Comparison
```

---

# 47. Chunk-Level Evaluation

未来不能只使用：

```text
gold_page
```

还要逐步增加：

```text
gold_chunk_id
gold_evidence
gold_span
```

因为：

```text
Hit Page
!=
Hit Evidence
```

未来指标：

```text
Recall@K
MRR
nDCG
Evidence Recall
Evidence Precision
Citation Accuracy
Faithfulness
Answer Correctness
```

---

# 48. Query Rewrite

当前 Query Rewrite 保留。

Rewrite 必须尽量保持：

* 专有名词。
* 模型名。
* 数字。
* 文件名。
* 论文名称。
* 表格编号。
* 技术缩写。

例如：

```text
BGE-M3 在表 4 的 Recall@10 是多少？
```

不能重写成：

```text
该模型的效果怎么样？
```

---

# 49. Conversation Memory

Memory 的作用：

```text
理解追问
```

不是：

```text
保存所有历史然后全部塞入 Prompt
```

当前优先：

```text
最近 N 轮
```

未来可以考虑：

```text
Summary Memory
```

当前不要提前实现长期记忆系统。

---

# 50. Agent Router

当前 Router：

```text
qa
summary
term
report
source_check
```

继续保留。

原则：

```text
Rule First
LLM Second
Fallback QA
```

不要为了 Agent 概念引入复杂编排框架。

项目核心：

```text
High-quality RAG
```

而不是：

```text
Complex Agent Orchestration
```

---

# 51. 用户类型 Prompt

当前用户类型：

```text
undergraduate
graduate
researcher
developer
teacher
general
```

回答策略：

## undergraduate

* 少术语。
* 多解释。
* 强调理解。

## graduate

* 方法。
* 实验。
* 创新。
* 对比。

## researcher

* Related Work。
* Method。
* Limitations。
* Reproduction。

## developer

* Architecture。
* API。
* Implementation。
* Deployment。

## teacher

* 内容结构。
* 教学重点。
* 可提问点。

## general

* 简洁。
* 清晰。

用户类型只改变：

```text
Answer Style
```

不能改变：

```text
Evidence
```

---

# 52. 用户隔离与认证边界

当前：

```text
user_id
```

实现的是：

```text
Logical User Isolation
```

不是：

```text
Real Authentication
```

README 和简历可以写：

```text
用户级数据隔离
```

暂时不要写：

```text
完善权限系统
RBAC
企业级认证
```

除非后续真实实现：

```text
JWT
OAuth
Session Authentication
RBAC
```

---

# 53. MongoDB 职责

MongoDB 负责：

```text
users
documents
conversations
messages
sources
```

未来：

```text
knowledge_bases
```

MongoDB 定位：

```text
Application Metadata
+
Persistence
```

Vector DB 定位：

```text
Embedding Retrieval
```

不要混淆职责。

---

# 54. 项目命名统一

正式定位逐步统一为：

```text
DocRAG
```

旧医学命名逐步清理。

例如：

```text
medical_chunks
```

未来改成：

```text
document_chunks
```

但修改前必须考虑：

* 已有数据。
* collection migration。
* tests。
* backward compatibility。

不要为了改名字破坏现有索引。

---

# 55. DTO 与内部数据结构

逐步减少：

```python
chunk["文本块"]
chunk["页码"]
```

这种字符串字典操作。

未来优先：

```python
@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
```

或者：

```text
Pydantic Model
```

但：

```text
禁止一次性重构整个项目
```

只在当前修改模块自然迁移。

---

# 56. API 修改原则

新增或修改接口必须考虑：

```text
Request
Response
Status Code
Error Code
Frontend Compatibility
Backward Compatibility
Tests
```

不能只保证：

```text
Postman 可以调用
```

还需要考虑前端和已有数据。

---

# 57. Error Handling

错误类型逐步统一：

```text
PDF_PARSE_FAILED
OCR_FAILED
CHUNK_FAILED
EMBEDDING_FAILED
VECTOR_WRITE_FAILED
RETRIEVAL_FAILED
RERANK_FAILED
LLM_TIMEOUT
INDEX_VERSION_MISMATCH
```

错误需要：

```text
可定位
可记录
可恢复
可重试
```

---

# 58. Logging

关键日志字段：

```text
request_id
user_id
document_id
kb_id
file_name

parse_time
chunk_time
embedding_time
index_time
retrieval_time
rerank_time
llm_time

chunk_count
candidate_count
reranked_count

index_version
chunk_strategy

error_stage
error_code
```

禁止记录：

* API Key。
* Password。
* Token。
* 完整隐私文档内容。

---

# 59. Security

禁止：

```text
API Key 放前端
数据库密码硬编码
Secret commit 到 Git
```

使用：

```text
.env
Environment Variables
Deployment Secrets
```

配置变化时：

```text
.env.example
```

同步维护。

---

# 60. Deployment 原则

项目首先保证：

```text
Local Runnable
```

部署其次。

如果免费平台因为：

```text
Memory
CPU
Outbound Restriction
Model Size
```

无法支持：

```text
Embedding
Reranker
```

不要为了部署而删除核心能力。

尤其：

```text
CrossEncoder Rerank
```

属于当前项目重要技术能力。

---

# 61. Frontend 原则

前端目标：

```text
Professional AI Knowledge Base UI
```

重点：

* Conversation List。
* Knowledge Base List。
* Document List。
* Upload Status。
* Sources。
* Loading。
* Error State。
* Empty State。
* Responsive Layout。

Phase 2 之前不要优先进行大规模 UI 重构。

---

# 62. 性能优化原则

优先级：

```text
Correctness
↓
Maintainability
↓
Tests
↓
Evaluation
↓
Performance
```

没有 profiling 结果时，不要凭感觉做性能优化。

---

# 63. 修改前必须检查

每个任务开始时：

```text
1. git status
2. 查看目标文件
3. 查看相关测试
4. 搜索调用点
5. 查看配置
6. 判断是否存在用户未提交修改
```

如果存在未提交修改：

```text
禁止覆盖
禁止 reset
禁止 clean
```

---

# 64. Git 安全规则

禁止自动执行：

```bash
git reset --hard
git clean -fd
git checkout -- .
git push --force
```

除非用户明确授权。

推荐使用：

```bash
git status
git diff
```

确认变化。

---

# 65. 每轮任务的执行方式

默认：

```text
Inspect
↓
Plan
↓
Implement
↓
Test
↓
Fix Regression
↓
Summarize
```

如果当前 Phase 尚未完成：

```text
继续 Phase
```

不要自动跑完整效果实验。

---

# 66. Codex 完成任务后的汇报格式

任务结束后输出以下内容。

## 修改内容

列出：

```text
file
change
reason
```

## 实现结果

说明当前功能是否完成。

## 测试结果

例如：

```text
pytest tests/test_text_splitter_service.py -v
```

结果：

```text
10 passed
```

再运行：

```text
pytest
```

结果：

```text
82 passed
```

## Regression

明确说明是否存在回归。

## 风险

说明：

```text
compatibility
migration
data
performance
```

## 当前 Phase 进度

例如：

```text
Phase 1

[x] Fixed
[x] Recursive
[x] Chunk Metadata
[ ] Index Fingerprint
[ ] Force Reindex
[ ] Phase Evaluation
```

## 下一步

只给最相关的：

```text
1–3 个任务
```

---

# 67. Definition of Done

普通代码任务完成必须满足：

```text
功能可运行
+
相关测试通过
+
没有明显 Regression
+
必要配置同步
+
必要 README 更新
```

---

# 68. Phase Definition of Done

一个完整 Phase 完成必须满足：

```text
功能闭环
+
Tests Pass
+
Regression Fixed
+
接口基本稳定
+
数据结构基本稳定
```

之后才进入：

```text
Phase Evaluation
```

---

# 69. RAG Evaluation Definition of Done

阶段实验需要：

```text
固定 Dataset
+
固定环境
+
明确 Baseline
+
Before / After Metrics
```

如果优化下降：

必须明确：

```text
Regression
```

不要隐藏。

---

# 70. 当前禁止主动引入的技术

当前不要主动添加：

```text
GraphRAG
Knowledge Graph
Multi-Agent
Long-term Agent Memory
Fine-tuning
LoRA
Full Microservices
Kubernetes
Kafka
Redis Cluster
Milvus Cluster
复杂 RBAC
```

除非：

1. 当前功能确实需要。
2. 用户明确要求。
3. 有合理工程依据。

---

# 71. 技术选择原则

任何新技术都需要回答：

```text
解决什么问题？
```

以及：

```text
为什么当前方案解决不了？
```

优先：

```text
Problem
→
Design
→
Implementation
→
Validation
```

禁止：

```text
Technology
→
Technology
→
Technology
```

---

# 72. 面试价值

项目升级需要能够回答：

```text
为什么这样切片？
为什么需要 Reranker？
为什么需要 Query Rewrite？
为什么增加 BM25？
为什么使用 RRF？
为什么需要 Index Fingerprint？
为什么 Summary 不能普通 Top-K？
什么时候应该迁移 Qdrant？
为什么保留 Fixed Baseline？
```

每个重要能力最终都形成：

```text
Problem
Design
Trade-off
Implementation
Result
```

---

# 73. 简历项目定位

当前可以描述：

> 设计并实现 DocRAG 文档智能问答系统，完成 PDF 文档解析、向量检索、CrossEncoder Rerank、Query Rewrite、多轮会话记忆、用户级数据隔离、Agent Router 与离线 RAG 评估体系，并围绕结构化切片、多文档知识库、索引版本管理和混合检索持续进行工程升级。

当前不要写：

```text
企业级知识库
亿级向量
高并发平台
完善 RBAC
完整微服务架构
```

除非真实实现并验证。

---

# 74. Windows 开发环境

主要运行环境：

```text
Windows
```

命令优先：

```text
CMD
PowerShell
```

路径优先使用：

```text
D:\MedRag
```

不要默认：

```text
/home/user/project
```

如果必须提供 Linux 命令，要同时说明 Windows 对应方式。

---

# 75. Python 代码要求

遵循：

* 类型注解。
* 合理 docstring。
* 清晰函数职责。
* 避免超长函数。
* 避免重复逻辑。
* 避免隐藏副作用。
* 保持现有代码风格。
* 优先小步修改。
* 不过度设计。

不要为了所谓 Clean Architecture 大规模重建项目。

---

# 76. 测试命令

修改 splitter 时优先：

```bash
pytest tests/test_text_splitter_service.py -v
```

然后全量：

```bash
pytest
```

当前历史稳定基线：

```text
76 passed
```

如果测试数量增加是正常现象。

如果已有测试失败：

```text
必须处理
```

---

# 77. 遇到未知情况

优先：

```text
Read Code
Read Tests
Read Config
Read Logs
Search Call Sites
```

不要猜。

如果项目文件本身可以回答问题：

```text
不要询问用户
```

只有以下情况才需要询问：

* 产品方向选择。
* 不可恢复操作。
* 业务规则无法从代码判断。
* 需要用户 API Key。
* 需要真实外部账户信息。

---

# 78. 不允许破坏用户现有工作

如果发现用户已有修改：

```text
不得覆盖
不得丢弃
不得 reset
```

必须基于当前 working tree 继续工作。

如果存在冲突：

先说明。

---

# 79. 当前 Phase 1 状态模板

每次 Phase 1 结束任务都更新认知：

```text
Phase 1: Chunking V2 + Index Version

[ ] Unified Chunk Interface
[ ] Fixed Strategy
[ ] Recursive Strategy
[ ] Chunk Metadata
[ ] Splitter Tests
[ ] Index Fingerprint
[ ] Force Reindex
[ ] Ingestion Tests
[ ] Full Regression Tests
[ ] Phase 1 Evaluation
```

只有全部开发项完成后：

```text
再运行 Phase 1 Evaluation
```

---

# 80. 当前最重要规则

Codex 必须记住：

> 当前阶段先开发，阶段完成后再统一做实验。

不要：

```text
改一个函数
→
跑一次完整 RAG Evaluation
```

应该：

```text
完成一个完整 Phase
→
Tests Stable
→
统一 Evaluation
```

---

# 81. 实验触发条件

默认情况下不要主动运行完整 RAG 实验。

只有以下情况可以提前实验：

1. 用户明确要求。
2. 当前设计必须依赖实验结果才能继续。
3. 出现明显效果 Regression，需要定位。
4. 两种技术实现无法通过工程判断选择。
5. 当前 Phase 已经开发完成。

除此之外：

```text
继续开发当前 Phase
```

---

# 82. 当前最优先任务

现在从：

```text
medrag/backend/services/text_splitter_service.py
```

开始。

第一目标：

```text
Fixed
+
Recursive
+
统一 Chunk Strategy
```

然后继续：

```text
Chunk Metadata
↓
Index Fingerprint
↓
force_reindex
↓
Tests
```

等 Phase 1 开发完成，再进行一次完整实验。

---

# 83. 项目最终目标链路

DocRAG V2 最终希望形成：

```text
Upload Documents
↓
Document Parsing
↓
Versioned Chunking
↓
Embedding
↓
Knowledge Base
↓
Dense + Sparse Retrieval
↓
RRF
↓
CrossEncoder Rerank
↓
Evidence Filter
↓
Context Builder
↓
LLM
↓
Citation
↓
Conversation Persistence
↓
Offline Evaluation
```

---

# 84. 最终原则

始终遵守：

> 不堆技术，以真实问题驱动设计。

> 先把功能做完整，再统一进行阶段实验。

> 开发过程必须测试，但不要求频繁跑效果实验。

> 不凭感觉判断 RAG 优化是否有效，阶段结束后统一验证。

> 不破坏稳定基线，小步迭代。

> 不为了部署牺牲项目核心能力。

> 不一次修改多个核心变量。

> 不为了“企业级”三个字过度设计。

> 每个重要技术升级都应该最终能够从代码、工程设计、实验结果和面试表达四个角度解释清楚。

---

# 85. Codex 当前行动指令

读取本文件后：

1. 将 `D:\MedRag` 视为项目根目录。
2. 首先查看 `git status`。
3. 读取当前目标模块和测试。
4. 不覆盖用户已有修改。
5. 当前专注 Phase 1。
6. 先实现完整功能。
7. 修改后运行相关测试。
8. 阶段未完成时继续开发，不主动运行完整 RAG 效果实验。
9. Phase 1 完成后，再统一进行 Fixed vs Recursive Evaluation。
10. 每轮结束汇报修改文件、测试结果、Phase 进度和下一步。
