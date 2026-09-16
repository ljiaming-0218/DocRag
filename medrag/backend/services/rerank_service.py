import logging
from time import perf_counter

from sentence_transformers import CrossEncoder


logger = logging.getLogger(__name__)
reranker_model = CrossEncoder(
    "BAAI/bge-reranker-base",
    local_files_only=True,
)

def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    started_at = perf_counter()
    if not query.strip():
        raise ValueError("查询内容不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if not chunks:
        raise ValueError("chunks 不能为空")
    

    pairs = []
    for chunk in chunks:
        pairs.append([query,chunk["文本块"]])
        
    
    scores = reranker_model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked_chunks = sorted(
        chunks,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )[:top_k]
    logger.info(
        "rerank_completed candidates=%s kept=%s rerank_ms=%.2f",
        len(chunks),
        len(ranked_chunks),
        (perf_counter() - started_at) * 1000,
    )
    return ranked_chunks

