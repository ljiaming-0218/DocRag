# Dataset

本目录是评估的唯一数据源。

- `documents.json`：文档清单、来源、页数和 SHA-256。
- `single_turn.json`：单轮问题。
- `follow_up.json`：包含历史上下文的追问。
- `pdfs/`：固定评估文档。

修改题目、证据或 PDF 后，所有新结果都会产生不同的 dataset fingerprint。
对比两次实验时，必须保证 fingerprint 相同，否则不能把差异归因于检索算法。
