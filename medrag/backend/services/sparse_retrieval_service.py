import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import jieba
from rank_bm25 import BM25Okapi

from services.vector_store_service import get_chunks_by_documents


TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:[.@_+\-/][a-z0-9]+)*|[\u4e00-\u9fff]+",
    re.IGNORECASE,
)
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
jieba.setLogLevel(logging.WARNING)
SPARSE_INDEX_CACHE_MAX_ENTRIES = 32


@dataclass
class SparseIndex:
    chunks: list[dict]
    model: BM25Okapi


SparseDocumentScope = tuple[tuple[str, str], ...]
SparseCacheKey = tuple[str, SparseDocumentScope]
_sparse_index_cache: OrderedDict[SparseCacheKey, SparseIndex] = OrderedDict()
_sparse_index_cache_lock = RLock()


def _build_cache_key(
    user_id: str,
    document_ids: list[str],
    index_generations: dict[str, str | None] | None = None,
) -> SparseCacheKey:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")
    if not document_ids:
        raise ValueError("document_ids cannot be empty")

    normalized_document_ids = []
    for document_id in document_ids:
        document_id = document_id.strip()
        if not document_id:
            raise ValueError("document_id cannot be empty")
        if document_id not in normalized_document_ids:
            normalized_document_ids.append(document_id)

    document_scope = tuple(
        sorted(
            (
                document_id,
                (index_generations or {}).get(document_id) or "",
            )
            for document_id in normalized_document_ids
        )
    )
    return user_id, document_scope


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize Chinese text while preserving exact English terms."""
    if not text or not text.strip():
        return []

    tokens = []
    for segment in TOKEN_PATTERN.findall(text.lower()):
        if CHINESE_PATTERN.search(segment):
            tokens.extend(
                token.strip()
                for token in jieba.cut_for_search(segment)
                if token.strip()
            )
            tokens.extend(
                segment[index:index + 2]
                for index in range(len(segment) - 1)
            )
        else:
            tokens.append(segment)
    return tokens


def build_bm25_index(chunks: list[dict]) -> SparseIndex:
    """Build an in-memory BM25 index for non-empty chunks."""
    valid_chunks = []
    tokenized_corpus = []

    for chunk in chunks:
        text = chunk.get("文本块", "").strip()
        tokens = tokenize_for_bm25(text)
        if not tokens:
            continue
        valid_chunks.append(chunk)
        tokenized_corpus.append(tokens)

    if not valid_chunks:
        raise ValueError("chunks must contain searchable text")

    return SparseIndex(
        chunks=valid_chunks,
        model=BM25Okapi(tokenized_corpus),
    )


def get_or_build_sparse_index(
    user_id: str,
    document_ids: list[str],
    index_generations: dict[str, str | None] | None = None,
) -> SparseIndex | None:
    """Return a cached scope index or build it from Chroma on first use."""
    cache_key = _build_cache_key(
        user_id,
        document_ids,
        index_generations,
    )
    with _sparse_index_cache_lock:
        cached_index = _sparse_index_cache.get(cache_key)
        if cached_index is not None:
            _sparse_index_cache.move_to_end(cache_key)
            return cached_index

    scoped_document_ids = [
        document_id
        for document_id, _ in cache_key[1]
    ]
    scoped_generations = {
        document_id: generation_id
        for document_id, generation_id in cache_key[1]
        if generation_id
    }
    if scoped_generations:
        chunks = get_chunks_by_documents(
            cache_key[0],
            scoped_document_ids,
            scoped_generations,
        )
    else:
        chunks = get_chunks_by_documents(
            cache_key[0],
            scoped_document_ids,
        )
    if not chunks:
        return None
    try:
        sparse_index = build_bm25_index(chunks)
    except ValueError:
        return None

    with _sparse_index_cache_lock:
        existing_index = _sparse_index_cache.get(cache_key)
        if existing_index is not None:
            _sparse_index_cache.move_to_end(cache_key)
            return existing_index

        _sparse_index_cache[cache_key] = sparse_index
        while len(_sparse_index_cache) > SPARSE_INDEX_CACHE_MAX_ENTRIES:
            _sparse_index_cache.popitem(last=False)
    return sparse_index


def invalidate_sparse_indexes(user_id: str, document_id: str) -> int:
    """Invalidate every cached scope containing the reindexed document."""
    user_id = user_id.strip()
    document_id = document_id.strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")
    if not document_id:
        raise ValueError("document_id cannot be empty")

    with _sparse_index_cache_lock:
        invalid_keys = [
            cache_key
            for cache_key in _sparse_index_cache
            if cache_key[0] == user_id
            and any(
                scoped_document_id == document_id
                for scoped_document_id, _ in cache_key[1]
            )
        ]
        for cache_key in invalid_keys:
            del _sparse_index_cache[cache_key]
    return len(invalid_keys)


def clear_sparse_index_cache() -> None:
    """Clear process-local sparse indexes, mainly for tests and shutdown."""
    with _sparse_index_cache_lock:
        _sparse_index_cache.clear()


def search_sparse_index(
    query: str,
    sparse_index: SparseIndex,
    candidate_k: int,
) -> list[dict]:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0")

    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []

    scores = sparse_index.model.get_scores(query_tokens)
    ranked_items = sorted(
        enumerate(scores),
        key=lambda item: (-float(item[1]), item[0]),
    )

    results = []
    for chunk_index, score in ranked_items:
        score = float(score)
        if score <= 0:
            continue

        chunk = sparse_index.chunks[chunk_index]
        results.append(
            {
                **chunk,
                "距离": None,
                "元数据": dict(chunk.get("元数据") or {}),
                "sparse_score": score,
                "sparse_rank": len(results) + 1,
                "retrieval_sources": ["sparse"],
            }
        )
        if len(results) >= candidate_k:
            break
    return results


def search_sparse_chunks(
    query: str,
    chunks: list[dict],
    candidate_k: int,
) -> list[dict]:
    """Rank chunks with BM25 and return only positively matched candidates."""
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0")
    if not chunks:
        return []

    sparse_index = build_bm25_index(chunks)
    return search_sparse_index(query, sparse_index, candidate_k)


def retrieve_sparse_candidates(
    user_id: str,
    document_ids: list[str],
    query: str,
    candidate_k: int,
    index_generations: dict[str, str | None] | None = None,
) -> list[dict]:
    """Load a user-scoped corpus from Chroma and run BM25 retrieval."""
    user_id = user_id.strip()
    query = query.strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")
    if not document_ids:
        raise ValueError("document_ids cannot be empty")
    if not query:
        raise ValueError("query cannot be empty")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0")

    sparse_index = get_or_build_sparse_index(
        user_id,
        document_ids,
        index_generations,
    )
    if sparse_index is None:
        return []
    return search_sparse_index(query, sparse_index, candidate_k)
