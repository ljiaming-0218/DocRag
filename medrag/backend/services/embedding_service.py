import logging
from time import perf_counter

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_BATCH_SIZE


logger = logging.getLogger(__name__)
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
model = SentenceTransformer(MODEL_NAME, local_files_only=True)


def get_embedding(text: str) -> list[float]:
    sentence_vector = model.encode([text])[0]
    return sentence_vector.tolist()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    sentence_vectors = model.encode(texts)
    return sentence_vectors.tolist()


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
