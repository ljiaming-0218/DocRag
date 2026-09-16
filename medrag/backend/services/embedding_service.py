import logging
from threading import Lock
from time import perf_counter

from openai import OpenAI

from config import (
    EMBEDDING_API_BASE,
    EMBEDDING_API_KEY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_TIMEOUT_SECONDS,
)


logger = logging.getLogger(__name__)
MODEL_NAME = EMBEDDING_MODEL
PROVIDER_NAME = EMBEDDING_PROVIDER
_client = None
_client_lock = Lock()


def get_embedding_client() -> OpenAI:
    """Create one OpenAI-compatible embedding client on first use."""
    global _client

    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        if not EMBEDDING_API_KEY:
            raise RuntimeError("EMBEDDING_API_KEY 未配置")
        if not EMBEDDING_API_BASE:
            raise RuntimeError("EMBEDDING_API_BASE 未配置")

        _client = OpenAI(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_API_BASE,
            timeout=EMBEDDING_TIMEOUT_SECONDS,
            max_retries=EMBEDDING_MAX_RETRIES,
        )
        logger.info(
            "embedding_client_initialized provider=%s model=%s",
            EMBEDDING_PROVIDER,
            EMBEDDING_MODEL,
        )
        return _client


def _validate_embeddings(
    vectors: list[list[float]],
    expected_count: int,
) -> None:
    if len(vectors) != expected_count:
        raise RuntimeError(
            "Embedding 返回数量与输入数量不一致: "
            f"expected={expected_count}, actual={len(vectors)}"
        )

    if EMBEDDING_DIMENSION is None:
        return
    for index, vector in enumerate(vectors):
        if len(vector) != EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Embedding 维度与配置不一致: "
                f"index={index}, configured={EMBEDDING_DIMENSION}, "
                f"actual={len(vector)}"
            )


def get_embedding(text: str) -> list[float]:
    started_at = perf_counter()
    vectors = get_embeddings([text])
    logger.info(
        "embedding_query_completed provider=%s model=%s text_length=%s "
        "elapsed_seconds=%.3f",
        EMBEDDING_PROVIDER,
        EMBEDDING_MODEL,
        len(text),
        perf_counter() - started_at,
    )
    return vectors[0]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding 输入文本不能为空")

    started_at = perf_counter()
    try:
        response = get_embedding_client().embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
    except Exception as exc:
        logger.warning(
            "embedding_api_failed provider=%s model=%s batch_size=%s "
            "error_type=%s elapsed_seconds=%.3f",
            EMBEDDING_PROVIDER,
            EMBEDDING_MODEL,
            len(texts),
            type(exc).__name__,
            perf_counter() - started_at,
        )
        raise

    ordered_items = sorted(response.data, key=lambda item: item.index)
    actual_indices = [item.index for item in ordered_items]
    expected_indices = list(range(len(texts)))
    if actual_indices != expected_indices:
        raise RuntimeError(
            "Embedding 返回索引与输入顺序不一致: "
            f"expected={expected_indices}, actual={actual_indices}"
        )
    vectors = [list(item.embedding) for item in ordered_items]
    _validate_embeddings(vectors, len(texts))
    logger.info(
        "embedding_api_completed provider=%s model=%s batch_size=%s "
        "elapsed_seconds=%.3f",
        EMBEDDING_PROVIDER,
        EMBEDDING_MODEL,
        len(texts),
        perf_counter() - started_at,
    )
    return vectors


def embed_chunks(
    chunks: list[dict],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[dict]:
    if batch_size <= 0:
        raise ValueError("embedding batch_size 必须大于 0")

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(
        range(0, len(chunks), batch_size),
        start=1,
    ):
        batch_started_at = perf_counter()
        batch = chunks[start:start + batch_size]
        texts = [chunk["文本块"] for chunk in batch]
        sentence_vectors = get_embeddings(texts)
        if len(sentence_vectors) != len(batch):
            raise RuntimeError(
                "Embedding 返回数量与文本块数量不一致: "
                f"expected={len(batch)}, actual={len(sentence_vectors)}"
            )

        for chunk, vector in zip(batch, sentence_vectors):
            chunk["embedding"] = vector

        logger.info(
            "embedding_batch_completed batch_index=%s total_batches=%s "
            "batch_size=%s elapsed_seconds=%.3f",
            batch_index,
            total_batches,
            len(batch),
            perf_counter() - batch_started_at,
        )

    return chunks
