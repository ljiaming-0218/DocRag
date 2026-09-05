import logging

from config import EVIDENCE_RERANK_MIN_SCORE


logger = logging.getLogger(__name__)


def filter_relevant_evidence(
    chunks: list[dict],
    min_score: float | None = None,
) -> list[dict]:
    """Keep only evidence that passes the configured reranker threshold."""
    threshold = (
        EVIDENCE_RERANK_MIN_SCORE
        if min_score is None
        else min_score
    )
    if threshold is None:
        return chunks

    filtered = [
        chunk
        for chunk in chunks
        if chunk.get("rerank_score") is not None
        and chunk["rerank_score"] >= threshold
    ]
    logger.info(
        "evidence_filter_applied threshold=%s candidates=%s kept=%s",
        threshold,
        len(chunks),
        len(filtered),
    )
    return filtered
