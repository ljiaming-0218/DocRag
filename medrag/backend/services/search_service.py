import logging

from config import RETRIEVAL_MODE
from services.embedding_service import get_embedding
from services.evidence_filter_service import filter_relevant_evidence
from services.rerank_service import rerank_chunks
from services.retrieval_fusion_service import reciprocal_rank_fusion
from services.sparse_retrieval_service import retrieve_sparse_candidates
from services.summary_query_service import build_summary_subqueries
from services.vector_store_service import (
    query_chunks,
    query_chunks_by_documents,
)


logger = logging.getLogger(__name__)
SUPPORTED_RETRIEVAL_MODES = {"dense", "hybrid"}


def resolve_retrieval_mode() -> str:
    retrieval_mode = RETRIEVAL_MODE.strip().lower()
    if retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
        raise ValueError(
            "RETRIEVAL_MODE must be one of: dense, hybrid"
        )
    return retrieval_mode


def format_dense_candidates(search_results: dict) -> list[dict]:
    documents = search_results.get("documents") or [[]]
    distances = search_results.get("distances") or [[]]
    metadatas = search_results.get("metadatas") or [[]]
    ids = search_results.get("ids") or [[]]

    documents = documents[0] if documents else []
    distances = distances[0] if distances else []
    metadatas = metadatas[0] if metadatas else []
    ids = ids[0] if ids else []

    results = []
    for index, document in enumerate(documents):
        if not is_useful_chunk(document):
            continue

        candidate = {
            "文本块": document,
            "距离": distances[index],
            "元数据": metadatas[index],
            "dense_rank": len(results) + 1,
            "retrieval_sources": ["dense"],
        }
        if index < len(ids):
            candidate["chunk_id"] = ids[index]
        results.append(candidate)
    return results


def fuse_dense_and_sparse_candidates(
    user_id: str,
    document_ids: list[str],
    query: str,
    dense_candidates: list[dict],
    candidate_k: int,
    index_generations: dict[str, str | None] | None = None,
) -> list[dict]:
    try:
        if index_generations and any(index_generations.values()):
            sparse_candidates = retrieve_sparse_candidates(
                user_id,
                document_ids,
                query,
                candidate_k,
                index_generations,
            )
        else:
            sparse_candidates = retrieve_sparse_candidates(
                user_id,
                document_ids,
                query,
                candidate_k,
            )
        sparse_candidates = [
            candidate
            for candidate in sparse_candidates
            if is_useful_chunk(candidate.get("文本块", ""))
        ]
        if not sparse_candidates:
            return dense_candidates

        fused_candidates = reciprocal_rank_fusion(
            {
                "dense": dense_candidates,
                "sparse": sparse_candidates,
            },
            limit=candidate_k,
        )
        return fused_candidates or dense_candidates
    except Exception as error:
        logger.warning(
            "hybrid_retrieval_fallback mode=dense error_type=%s",
            type(error).__name__,
        )
        return dense_candidates


def retrieve_dense_candidate_chunks(
    user_id: str,
    document_id: str,
    query: str,
    candidate_k: int,
    index_generation_id: str | None = None,
) -> list[dict]:
    query_embedding = get_embedding(query)
    query_arguments = [
        user_id,
        document_id,
        query_embedding,
        candidate_k,
    ]
    if index_generation_id:
        query_arguments.append(index_generation_id)
    search_results = query_chunks(*query_arguments)
    return format_dense_candidates(search_results)


def retrieve_candidate_chunks(
    user_id: str,
    document_id: str,
    query: str,
    candidate_k: int,
    index_generation_id: str | None = None,
) -> list[dict]:
    user_id = user_id.strip()
    document_id = document_id.strip()
    query = query.strip()

    if not user_id:
        raise ValueError("user_id 不能为空")
    if not document_id:
        raise ValueError("document_id 不能为空")
    if not query:
        raise ValueError("查询内容不能为空")
    if candidate_k <= 0:
        raise ValueError("candidate_k 必须大于 0")

    retrieval_mode = resolve_retrieval_mode()
    dense_arguments = [user_id, document_id, query, candidate_k]
    if index_generation_id:
        dense_arguments.append(index_generation_id)
    dense_candidates = retrieve_dense_candidate_chunks(*dense_arguments)
    if retrieval_mode == "dense":
        return dense_candidates
    fusion_arguments = [
        user_id,
        [document_id],
        query,
        dense_candidates,
        candidate_k,
    ]
    if index_generation_id:
        fusion_arguments.append({document_id: index_generation_id})
    return fuse_dense_and_sparse_candidates(*fusion_arguments)


def search_relevant_chunks(
    user_id: str,
    document_id: str,
    query: str,
    n_results: int = 3,
    index_generation_id: str | None = None,
) -> list[dict]:
    if n_results <= 0:
        raise ValueError("n_results 必须大于 0")

    candidates = retrieve_candidate_chunks(
        user_id,
        document_id,
        query,
        n_results * 5,
        index_generation_id,
    )
    if not candidates:
        return []

    ranked_chunks = rerank_chunks(query, candidates, n_results)
    return filter_relevant_evidence(ranked_chunks)


def retrieve_candidate_chunks_for_documents(
    user_id: str,
    document_ids: list[str],
    query: str,
    candidate_k: int,
    index_generations: dict[str, str | None] | None = None,
) -> list[dict]:
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

    retrieval_mode = resolve_retrieval_mode()
    query_arguments = [
        user_id,
        document_ids,
        get_embedding(query),
        candidate_k,
    ]
    if index_generations and any(index_generations.values()):
        query_arguments.append(index_generations)
    search_results = query_chunks_by_documents(*query_arguments)

    dense_candidates = format_dense_candidates(search_results)
    if retrieval_mode == "dense":
        return dense_candidates
    fusion_arguments = [
        user_id,
        document_ids,
        query,
        dense_candidates,
        candidate_k,
    ]
    if index_generations and any(index_generations.values()):
        fusion_arguments.append(index_generations)
    return fuse_dense_and_sparse_candidates(*fusion_arguments)


def search_relevant_chunks_for_documents(
    user_id: str,
    document_ids: list[str],
    query: str,
    n_results: int = 3,
    index_generations: dict[str, str | None] | None = None,
) -> list[dict]:
    if n_results <= 0:
        raise ValueError("n_results must be greater than 0")

    candidates = retrieve_candidate_chunks_for_documents(
        user_id,
        document_ids,
        query,
        n_results * 5,
        index_generations,
    )
    if not candidates:
        return []
    ranked_chunks = rerank_chunks(query, candidates, n_results)
    return filter_relevant_evidence(ranked_chunks)


def search_summary_chunks(
    user_id: str,
    document_id: str,
    query: str,
    seed_k: int = 5,
    per_query_k: int = 10,
    per_query_keep: int = 4,
    context_k: int = 12,
    index_generation_id: str | None = None,
) -> dict:
    if seed_k <= 0:
        raise ValueError("seed_k 必须大于 0")
    if per_query_k <= 0:
        raise ValueError("per_query_k 必须大于 0")
    if context_k <= 0:
        raise ValueError("context_k 必须大于 0")


    if per_query_keep <= 0:
        raise ValueError("per_query_keep 必须大于 0")

    if per_query_keep > per_query_k:
        raise ValueError("per_query_keep 不能大于 per_query_k")
    
    seed_candidates = retrieve_candidate_chunks(
        user_id,
        document_id,
        query,
        max(seed_k * 3, per_query_k),
        index_generation_id,
    )
    if not seed_candidates:
        return {
            "sources": [],
            "retrieval_queries": [query],
        }

    seed_sources = rerank_chunks(
        query,
        seed_candidates,
        min(seed_k, len(seed_candidates)),
    )
    subqueries = build_summary_subqueries(query, seed_sources)
    retrieval_queries = list(
        dict.fromkeys([query, *subqueries])
    )

    merged_chunks = {}
    merge_candidate_chunks(
        merged_chunks,
        seed_sources,
        query,
    )

    for retrieval_query in retrieval_queries[1:]:
        candidates = retrieve_candidate_chunks(
            user_id,
            document_id,
            retrieval_query,
            per_query_k,
            index_generation_id,
        )
        if not candidates:
            continue

        ranked_candidates = rerank_chunks(
            retrieval_query,
            candidates,
            min(per_query_keep, len(candidates)),
        )

        merge_candidate_chunks(
            merged_chunks,
            ranked_candidates,
            retrieval_query,
        )

    if not merged_chunks:
        return {
            "sources": [],
            "retrieval_queries": retrieval_queries,
        }

    ranked_chunks = rerank_chunks(
        query,
        list(merged_chunks.values()),
        len(merged_chunks),
    )
    ranked_chunks = filter_relevant_evidence(ranked_chunks)
    sources = select_diverse_chunks(
        ranked_chunks,
        context_k,
    )

    return {
        "sources": sources,
        "retrieval_queries": retrieval_queries,
    }


def search_summary_chunks_for_documents(
    user_id: str,
    document_ids: list[str],
    query: str,
    seed_k: int = 5,
    per_query_k: int = 10,
    per_query_keep: int = 4,
    context_k: int = 12,
    index_generations: dict[str, str | None] | None = None,
) -> dict:
    if seed_k <= 0:
        raise ValueError("seed_k must be greater than 0")
    if per_query_k <= 0:
        raise ValueError("per_query_k must be greater than 0")
    if per_query_keep <= 0 or per_query_keep > per_query_k:
        raise ValueError("invalid per_query_keep")
    if context_k <= 0:
        raise ValueError("context_k must be greater than 0")

    seed_candidates = retrieve_candidate_chunks_for_documents(
        user_id,
        document_ids,
        query,
        max(seed_k * 3, per_query_k),
        index_generations,
    )
    if not seed_candidates:
        return {
            "sources": [],
            "retrieval_queries": [query],
        }

    seed_sources = rerank_chunks(
        query,
        seed_candidates,
        min(seed_k, len(seed_candidates)),
    )
    subqueries = build_summary_subqueries(query, seed_sources)
    retrieval_queries = list(dict.fromkeys([query, *subqueries]))

    merged_chunks = {}
    merge_candidate_chunks(merged_chunks, seed_sources, query)

    for retrieval_query in retrieval_queries[1:]:
        candidates = retrieve_candidate_chunks_for_documents(
            user_id,
            document_ids,
            retrieval_query,
            per_query_k,
            index_generations,
        )
        if not candidates:
            continue

        ranked_candidates = rerank_chunks(
            retrieval_query,
            candidates,
            min(per_query_keep, len(candidates)),
        )
        merge_candidate_chunks(
            merged_chunks,
            ranked_candidates,
            retrieval_query,
        )

    ranked_chunks = rerank_chunks(
        query,
        list(merged_chunks.values()),
        len(merged_chunks),
    )
    ranked_chunks = filter_relevant_evidence(ranked_chunks)
    return {
        "sources": select_diverse_chunks(ranked_chunks, context_k),
        "retrieval_queries": retrieval_queries,
    }


def merge_candidate_chunks(
    merged_chunks: dict,
    candidates: list[dict],
    matched_query: str,
) -> None:
    for candidate in candidates:
        metadata = candidate["元数据"]
        chunk_index = metadata.get("chunk_index")
        if chunk_index is None:
            key = (
                metadata.get("document_id"),
                metadata.get("page_number"),
                candidate["文本块"],
            )
        else:
            key = (
                metadata.get("document_id"),
                chunk_index,
            )
        existing = merged_chunks.get(key)

        if existing is None:
            merged_candidate = {
                **candidate,
                "matched_queries": [matched_query],
            }
            merged_chunks[key] = merged_candidate
            continue

        if matched_query not in existing["matched_queries"]:
            existing["matched_queries"].append(matched_query)

        candidate_distance = candidate.get("距离")
        existing_distance = existing.get("距离")
        if (
            candidate_distance is not None
            and (
                existing_distance is None
                or candidate_distance < existing_distance
            )
        ):
            existing["距离"] = candidate_distance


def select_diverse_chunks(
    chunks: list[dict],
    limit: int,
    max_chunks_per_page: int = 2,
) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit 必须大于 0")
    if max_chunks_per_page <= 0:
        raise ValueError("max_chunks_per_page 必须大于 0")

    selected = []
    page_counts = {}

    for chunk in chunks:
        metadata = chunk["元数据"]
        page_key = (
            metadata.get("document_id"),
            metadata.get("page_number"),
        )
        page_count = page_counts.get(page_key, 0)

        if page_count >= max_chunks_per_page:
            continue

        selected.append(chunk)
        page_counts[page_key] = page_count + 1

        if len(selected) >= limit:
            break

    return selected


def is_useful_chunk(text: str) -> bool:
    useless_keywords = [
        "creative commons",
        "reprints and permissions",
        "publisher’s note",
        "copyright",
        "license",
    ]
    lower_text = text.lower()
    return not any(
        keyword in lower_text
        for keyword in useless_keywords
    )
