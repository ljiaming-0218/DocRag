# Dataset

本目录是评估的唯一数据源。

- `documents.json`：文档清单、来源、页数和 SHA-256。
- `single_turn.json`：单轮问题。
- `follow_up.json`：包含历史上下文的追问。
- `pdfs/`：固定评估文档。

修改题目、证据或 PDF 后，所有新结果都会产生不同的 dataset fingerprint。
对比两次实验时，必须保证 fingerprint 相同，否则不能把差异归因于检索算法。

单文档案例使用 `document_key`。跨文档案例使用 `document_keys`，两者不能同时存在：

```json
{
  "case_id": "MULTI-COMP-01",
  "document_keys": ["lora", "cot"],
  "n_results": 4
}
```

跨文档案例的每条 `gold_evidence` 必须包含 `document_key`，避免不同 PDF 的相同页码或相似文本被误判为命中。`n_results` 是可选的案例级上下文数量覆盖，主要用于需要多篇文档证据的总结或比较题。
