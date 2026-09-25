import logging
import re
from time import perf_counter

from config import (
    RETRIEVAL_MODE,
    SUMMARY_CONTEXT_K,
    SUMMARY_SEED_K,
    SUMMARY_SUBQUERY_KEEP_K,
    SUMMARY_SUBQUERY_RETRIEVE_K,
)
from services.embedding_service import get_embedding
from services.evidence_filter_service import filter_relevant_evidence
from services.rerank_service import rerank_chunks
from services.retrieval_fusion_service import reciprocal_rank_fusion
from services.sparse_retrieval_service import retrieve_sparse_candidates
from services.summary_query_service import build_summary_subqueries
from services.vector_store_service import (
    get_chunks_by_documents,
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
    sparse_started_at = perf_counter()
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
        logger.info(
            "bm25_retrieval_completed query_length=%s candidates=%s "
            "bm25_retrieval_ms=%.2f",
            len(query),
            len(sparse_candidates),
            (perf_counter() - sparse_started_at) * 1000,
        )
        if not sparse_candidates:
            return dense_candidates

        rrf_started_at = perf_counter()
        fused_candidates = reciprocal_rank_fusion(
            {
                "dense": dense_candidates,
                "sparse": sparse_candidates,
            },
            limit=candidate_k,
        )
        logger.info(
            "rrf_completed dense_candidates=%s sparse_candidates=%s "
            "fused_candidates=%s rrf_ms=%.2f",
            len(dense_candidates),
            len(sparse_candidates),
            len(fused_candidates),
            (perf_counter() - rrf_started_at) * 1000,
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
    started_at = perf_counter()
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
    candidates = format_dense_candidates(search_results)
    logger.info(
        "dense_retrieval_completed document_count=1 query_length=%s "
        "candidates=%s dense_retrieval_ms=%.2f",
        len(query),
        len(candidates),
        (perf_counter() - started_at) * 1000,
    )
    return candidates


def retrieve_dense_candidate_chunks_for_documents(
    user_id: str,
    document_ids: list[str],
    query: str,
    candidate_k: int,
    index_generations: dict[str, str | None] | None = None,
) -> list[dict]:
    """Run one dense query over an explicitly scoped document set."""
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

    started_at = perf_counter()
    query_arguments = [
        user_id,
        document_ids,
        get_embedding(query),
        candidate_k,
    ]
    if index_generations and any(index_generations.values()):
        query_arguments.append(index_generations)
    candidates = format_dense_candidates(
        query_chunks_by_documents(*query_arguments)
    )
    logger.info(
        "dense_retrieval_completed document_count=%s query_length=%s "
        "candidates=%s dense_retrieval_ms=%.2f",
        len(document_ids),
        len(query),
        len(candidates),
        (perf_counter() - started_at) * 1000,
    )
    return candidates


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


def _normalize_retrieval_queries(queries: list[str]) -> list[str]:
    normalized_queries = []
    seen = set()
    for query in queries:
        query = query.strip()
        normalized = query.casefold()
        if query and normalized not in seen:
            normalized_queries.append(query)
            seen.add(normalized)
    if not normalized_queries:
        raise ValueError("queries cannot be empty")
    return normalized_queries


def search_relevant_chunks_for_queries(
    user_id: str,
    document_id: str,
    queries: list[str],
    n_results: int = 3,
    index_generation_id: str | None = None,
) -> list[dict]:
    """Retrieve with original and rewritten queries before one rerank pass."""
    if n_results <= 0:
        raise ValueError("n_results 必须大于 0")
    queries = _normalize_retrieval_queries(queries)
    if len(queries) == 1:
        return search_relevant_chunks(
            user_id,
            document_id,
            queries[0],
            n_results,
            index_generation_id,
        )

    candidate_k = n_results * 5
    candidates_by_query = {
        f"query_{index}": retrieve_candidate_chunks(
            user_id,
            document_id,
            query,
            candidate_k,
            index_generation_id,
        )
        for index, query in enumerate(queries)
    }
    candidates = reciprocal_rank_fusion(
        candidates_by_query,
        limit=candidate_k,
    )
    if not candidates:
        return []
    rerank_query = "\n".join(queries)
    return filter_relevant_evidence(
        rerank_chunks(rerank_query, candidates, n_results)
    )


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
    dense_started_at = perf_counter()
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
    logger.info(
        "dense_retrieval_completed document_count=%s query_length=%s "
        "candidates=%s dense_retrieval_ms=%.2f",
        len(document_ids),
        len(query),
        len(dense_candidates),
        (perf_counter() - dense_started_at) * 1000,
    )
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


def search_relevant_chunks_for_document_queries(
    user_id: str,
    document_ids: list[str],
    queries: list[str],
    n_results: int = 3,
    index_generations: dict[str, str | None] | None = None,
) -> list[dict]:
    """Run multi-query retrieval over a knowledge-base document scope."""
    if n_results <= 0:
        raise ValueError("n_results must be greater than 0")
    queries = _normalize_retrieval_queries(queries)
    if len(queries) == 1:
        return search_relevant_chunks_for_documents(
            user_id,
            document_ids,
            queries[0],
            n_results,
            index_generations,
        )

    candidate_k = n_results * 5
    candidates_by_query = {
        f"query_{index}": retrieve_candidate_chunks_for_documents(
            user_id,
            document_ids,
            query,
            candidate_k,
            index_generations,
        )
        for index, query in enumerate(queries)
    }
    candidates = reciprocal_rank_fusion(
        candidates_by_query,
        limit=candidate_k,
    )
    if not candidates:
        return []
    rerank_query = "\n".join(queries)
    return filter_relevant_evidence(
        rerank_chunks(rerank_query, candidates, n_results)
    )


def search_comparison_chunks_for_documents(
    user_id: str,
    document_ids: list[str],
    query: str,
    per_document_candidate_k: int = 8,
    per_document_keep: int = 2,
    context_k: int = 12,
    index_generations: dict[str, str | None] | None = None,
    original_query: str | None = None,
) -> dict:
    """Retrieve comparison evidence without allowing one document to dominate."""
    if not document_ids:
        raise ValueError("document_ids cannot be empty")
    if per_document_candidate_k <= 0:
        raise ValueError("per_document_candidate_k must be greater than 0")
    if per_document_keep <= 0:
        raise ValueError("per_document_keep must be greater than 0")
    if per_document_keep > per_document_candidate_k:
        raise ValueError(
            "per_document_keep cannot exceed per_document_candidate_k"
        )
    if context_k <= 0:
        raise ValueError("context_k must be greater than 0")

    comparison_queries = build_comparison_retrieval_queries(
        query,
        original_query,
    )
    ranked_by_document = {}
    for document_id in document_ids:
        generation_id = (
            index_generations.get(document_id)
            if index_generations
            else None
        )
        candidates = reciprocal_rank_fusion(
            {
                f"query_{index}": retrieve_candidate_chunks(
                    user_id,
                    document_id,
                    retrieval_query,
                    per_document_candidate_k,
                    generation_id,
                )
                for index, retrieval_query in enumerate(comparison_queries)
            },
            limit=per_document_candidate_k * len(comparison_queries),
        )
        if not candidates:
            continue

        ranked = rerank_chunks(
            query,
            candidates,
            min(per_document_keep, len(candidates)),
        )
        ranked = filter_relevant_evidence(ranked)
        if ranked:
            ranked_by_document[document_id] = ranked

    return {
        "sources": select_document_balanced_chunks(
            ranked_by_document,
            document_ids,
            context_k,
        ),
        "retrieval_queries": comparison_queries,
    }


def build_comparison_retrieval_queries(
    query: str,
    original_query: str | None = None,
) -> list[str]:
    """Add subject queries so one side of a comparison cannot hide the other."""
    queries = _normalize_retrieval_queries([original_query or "", query])
    patterns = (
        r"\bcompare\s+(.+?)\s+(?:and|with|versus|vs\.?)\s+(.+?)"
        r"(?=[:\uFF1A,\uFF0C?\uFF1F]|$)",
        r"(?:\u6bd4\u8f83|\u5bf9\u6bd4)\s*(.+?)\s*"
        r"(?:\u548c|\u4e0e|\u53ca)\s*(.+?)"
        r"(?=[:\uFF1A,\uFF0C?\uFF1F]|$)",
    )
    for value in list(queries):
        for pattern in patterns:
            match = re.search(pattern, value, flags=re.IGNORECASE)
            if match:
                queries.extend(part.strip() for part in match.groups())
                break
    return _normalize_retrieval_queries(queries)


def select_document_balanced_chunks(
    chunks_by_document: dict[str, list[dict]],
    document_ids: list[str],
    limit: int,
) -> list[dict]:
    """Round-robin ranked chunks so every represented document gets a turn."""
    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    selected = []
    rank = 0
    while len(selected) < limit:
        added = False
        for document_id in document_ids:
            document_chunks = chunks_by_document.get(document_id, [])
            if rank >= len(document_chunks):
                continue
            selected.append(document_chunks[rank])
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
        rank += 1
    return selected


def search_summary_chunks(
    user_id: str,
    document_id: str,
    query: str,
    seed_k: int = SUMMARY_SEED_K,
    per_query_k: int = SUMMARY_SUBQUERY_RETRIEVE_K,
    per_query_keep: int = SUMMARY_SUBQUERY_KEEP_K,
    context_k: int = SUMMARY_CONTEXT_K,
    index_generation_id: str | None = None,
    original_query: str | None = None,
) -> dict:
    started_at = perf_counter()
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
    
    base_queries = _normalize_retrieval_queries([
        original_query or query,
        query,
    ])
    merged_chunks = {}
    for base_query in base_queries:
        seed_candidates = retrieve_candidate_chunks(
            user_id,
            document_id,
            base_query,
            max(seed_k * 3, per_query_k),
            index_generation_id,
        )
        if not seed_candidates:
            continue
        seed_sources = filter_relevant_evidence(
            rerank_chunks(
                base_query,
                seed_candidates,
                min(seed_k, len(seed_candidates)),
            )
        )
        merge_candidate_chunks(
            merged_chunks,
            seed_sources,
            base_query,
        )

    seed_sources = (
        select_query_covered_chunks(
            list(merged_chunks.values()),
            base_queries,
            seed_k,
        )
        if merged_chunks
        else []
    )
    subqueries = build_summary_subqueries(query, seed_sources)
    retrieval_queries = _normalize_retrieval_queries([
        *base_queries,
        *subqueries,
    ])

    for retrieval_query in retrieval_queries[len(base_queries):]:
        candidates = retrieve_candidate_chunks(
            user_id,
            document_id,
            retrieval_query,
            per_query_k,
            index_generation_id,
        )
        if not candidates:
            continue

        ranked_candidates = filter_relevant_evidence(
            rerank_chunks(
                retrieval_query,
                candidates,
                len(candidates),
            )
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

    sources = select_query_covered_chunks(
        list(merged_chunks.values()),
        retrieval_queries,
        context_k,
    )
    logger.info(
        "summary_retrieval_completed document_count=1 subquery_count=%s "
        "summary_retrieve_count=%s summary_final_context_count=%s "
        "total_ms=%.2f",
        len(retrieval_queries),
        len(merged_chunks),
        len(sources),
        (perf_counter() - started_at) * 1000,
    )

    return {
        "sources": sources,
        "retrieval_queries": retrieval_queries,
    }


def search_summary_chunks_for_documents(
    user_id: str,
    document_ids: list[str],
    query: str,
    seed_k: int = SUMMARY_SEED_K,
    per_query_k: int = SUMMARY_SUBQUERY_RETRIEVE_K,
    per_query_keep: int = SUMMARY_SUBQUERY_KEEP_K,
    context_k: int = SUMMARY_CONTEXT_K,
    index_generations: dict[str, str | None] | None = None,
    original_query: str | None = None,
    overview: bool = False,
) -> dict:
    started_at = perf_counter()
    if seed_k <= 0:
        raise ValueError("seed_k must be greater than 0")
    if per_query_k <= 0:
        raise ValueError("per_query_k must be greater than 0")
    if per_query_keep <= 0 or per_query_keep > per_query_k:
        raise ValueError("invalid per_query_keep")
    if context_k <= 0:
        raise ValueError("context_k must be greater than 0")

    if overview:
        return search_knowledge_base_overview_chunks(
            user_id,
            document_ids,
            query,
            context_k=context_k,
            index_generations=index_generations,
        )

    base_queries = _normalize_retrieval_queries([
        original_query or query,
        query,
    ])
    merged_chunks = {}
    for base_query in base_queries:
        seed_candidates = retrieve_candidate_chunks_for_documents(
            user_id,
            document_ids,
            base_query,
            max(seed_k * 3, per_query_k),
            index_generations,
        )
        if not seed_candidates:
            continue
        seed_sources = filter_relevant_evidence(
            rerank_chunks(
                base_query,
                seed_candidates,
                min(seed_k, len(seed_candidates)),
            )
        )
        merge_candidate_chunks(
            merged_chunks,
            seed_sources,
            base_query,
        )

    seed_sources = (
        select_query_covered_chunks(
            list(merged_chunks.values()),
            base_queries,
            seed_k,
        )
        if merged_chunks
        else []
    )
    subqueries = build_summary_subqueries(query, seed_sources)
    retrieval_queries = _normalize_retrieval_queries([
        *base_queries,
        *subqueries,
    ])

    for retrieval_query in retrieval_queries[len(base_queries):]:
        candidates = retrieve_candidate_chunks_for_documents(
            user_id,
            document_ids,
            retrieval_query,
            per_query_k,
            index_generations,
        )
        if not candidates:
            continue

        ranked_candidates = filter_relevant_evidence(
            rerank_chunks(
                retrieval_query,
                candidates,
                len(candidates),
            )
        )
        merge_candidate_chunks(
            merged_chunks,
            ranked_candidates,
            retrieval_query,
        )

    sources = select_query_covered_chunks(
        list(merged_chunks.values()),
        retrieval_queries,
        context_k,
    )
    logger.info(
        "summary_retrieval_completed document_count=%s subquery_count=%s "
        "summary_retrieve_count=%s summary_final_context_count=%s "
        "total_ms=%.2f",
        len(document_ids),
        len(retrieval_queries),
        len(merged_chunks),
        len(sources),
        (perf_counter() - started_at) * 1000,
    )
    return {
        "sources": sources,
        "retrieval_queries": retrieval_queries,
    }


def search_knowledge_base_overview_chunks(
    user_id: str,
    document_ids: list[str],
    query: str,
    context_k: int,
    index_generations: dict[str, str | None] | None = None,
) -> dict:
    """Select opening or abstract chunks from every active document."""
    chunks = get_chunks_by_documents(
        user_id,
        document_ids,
        index_generations,
    )
    if not chunks:
        return {
            "sources": [],
            "retrieval_queries": [query],
        }

    overview_sources = select_document_overview_chunks(
        chunks,
        document_ids,
        query,
        context_k,
    )
    summary_queries = build_summary_subqueries(query, overview_sources)
    core_query = next(
        (
            item for item in summary_queries
            if "core method" in item.lower()
        ),
        summary_queries[0] if summary_queries else query,
    )
    per_document_keep = max(
        2,
        (context_k + len(document_ids) - 1) // len(document_ids),
    )
    per_document_candidate_k = max(15, per_document_keep * 5)
    ranked_by_document = {}
    document_queries = _normalize_retrieval_queries([query, core_query])
    retrieval_queries = list(document_queries)

    for document_id in document_ids:
        generation_id = (
            index_generations.get(document_id)
            if index_generations
            else None
        )
        candidates_by_query = {
            f"query_{index}": retrieve_candidate_chunks(
                user_id,
                document_id,
                document_query,
                per_document_candidate_k,
                generation_id,
            )
            for index, document_query in enumerate(document_queries)
        }
        candidates = reciprocal_rank_fusion(
            candidates_by_query,
            limit=per_document_candidate_k,
        )
        if not candidates:
            continue
        ranked = rerank_chunks(
            core_query,
            candidates,
            min(per_document_keep, len(candidates)),
        )
        filtered_ranked = filter_relevant_evidence(ranked)
        if filtered_ranked:
            ranked_by_document[document_id] = filtered_ranked

    selected = select_document_balanced_chunks(
        ranked_by_document,
        document_ids,
        context_k,
    )
    represented_documents = {
        chunk.get("元数据", {}).get("document_id")
        for chunk in selected
    }
    for source in overview_sources:
        document_id = source.get("元数据", {}).get("document_id")
        if document_id in represented_documents:
            continue
        selected.append(source)
        represented_documents.add(document_id)
        if len(selected) >= context_k:
            break

    return {
        "sources": selected[:context_k],
        "retrieval_queries": retrieval_queries,
    }


def select_document_overview_chunks(
    chunks: list[dict],
    document_ids: list[str],
    query: str,
    context_k: int,
) -> list[dict]:
    """Prefer each document's abstract and nearby opening content."""
    chunks_by_document = {
        document_id: []
        for document_id in document_ids
    }
    for chunk in chunks:
        metadata = chunk.get("元数据", {})
        document_id = metadata.get("document_id")
        if (
            document_id in chunks_by_document
            and is_useful_chunk(chunk.get("文本块", ""))
        ):
            chunks_by_document[document_id].append(chunk)

    available_document_count = sum(
        bool(document_chunks)
        for document_chunks in chunks_by_document.values()
    )
    if available_document_count == 0:
        return []

    per_document_keep = max(
        1,
        min(2, context_k // available_document_count),
    )
    sources = []

    for document_id in document_ids:
        document_chunks = chunks_by_document[document_id]
        document_chunks.sort(
            key=lambda chunk: (
                chunk["元数据"].get("page_number", float("inf")),
                chunk["元数据"].get("chunk_index", float("inf")),
            )
        )
        if not document_chunks:
            continue

        abstract_index = next(
            (
                index
                for index, chunk in enumerate(document_chunks[:10])
                if "abstract" in chunk["文本块"].lower()
                or "摘要" in chunk["文本块"]
            ),
            None,
        )
        if abstract_index is None:
            start_index = 1 if len(document_chunks) > 1 else 0
        else:
            start_index = abstract_index

        selected_chunks = document_chunks[
            start_index:start_index + per_document_keep
        ]
        for chunk in selected_chunks:
            source = {
                **chunk,
                "距离": chunk.get("距离"),
                "retrieval_sources": ["document_overview"],
                "matched_queries": [query],
            }
            sources.append(source)
            if len(sources) >= context_k:
                return sources

    return sources


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
            candidate_score = candidate.get("rerank_score")
            merged_candidate = {
                **candidate,
                "matched_queries": [matched_query],
                "matched_query_scores": {
                    matched_query: candidate_score,
                },
            }
            merged_chunks[key] = merged_candidate
            continue

        if matched_query not in existing["matched_queries"]:
            existing["matched_queries"].append(matched_query)

        candidate_score = candidate.get("rerank_score")
        existing.setdefault("matched_query_scores", {})[
            matched_query
        ] = candidate_score
        existing_score = existing.get("rerank_score")
        if (
            candidate_score is not None
            and (
                existing_score is None
                or candidate_score > existing_score
            )
        ):
            existing["rerank_score"] = candidate_score

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


def select_query_covered_chunks(
    chunks: list[dict],
    retrieval_queries: list[str],
    limit: int,
    max_chunks_per_page: int = 2,
) -> list[dict]:
    """Select evidence by query coverage before filling by local score."""
    if limit <= 0:
        raise ValueError("limit 必须大于 0")

    ranked_chunks = sorted(
        chunks,
        key=lambda chunk: chunk.get("rerank_score", float("-inf")),
        reverse=True,
    )
    selected = []
    selected_keys = set()
    page_counts = {}

    def add_chunk(
        chunk: dict,
        *,
        enforce_page_limit: bool = True,
    ) -> bool:
        metadata = chunk.get("元数据", {})
        chunk_key = (
            metadata.get("document_id"),
            metadata.get("chunk_index"),
            chunk.get("文本块") if metadata.get("chunk_index") is None else None,
        )
        page_key = (
            metadata.get("document_id"),
            metadata.get("page_number"),
        )
        if chunk_key in selected_keys:
            return False
        if (
            enforce_page_limit
            and page_counts.get(page_key, 0) >= max_chunks_per_page
        ):
            return False
        selected.append(chunk)
        selected_keys.add(chunk_key)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        return True

    query_buckets = {
        query: sorted(
            [
                chunk
                for chunk in chunks
                if query in chunk.get("matched_queries", [])
            ],
            key=lambda chunk: chunk.get(
                "matched_query_scores",
                {},
            ).get(
                query,
                chunk.get("rerank_score", float("-inf")),
            ),
            reverse=True,
        )
        for query in retrieval_queries
    }
    for query in retrieval_queries:
        if len(selected) >= limit:
            break
        for chunk in query_buckets.get(query, []):
            if add_chunk(chunk, enforce_page_limit=False):
                break

    for chunk in ranked_chunks:
        if len(selected) >= limit:
            break
        add_chunk(chunk)
    return selected


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
