from services.embedding_service import get_embedding
from services.rerank_service import rerank_chunks
from services.summary_query_service import build_summary_subqueries
from services.vector_store_service import (
    query_chunks,
    query_chunks_by_documents,
)


def retrieve_candidate_chunks(
    user_id: str,
    document_id: str,
    query: str,
    candidate_k: int,
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

    query_embedding = get_embedding(query)
    search_results = query_chunks(
        user_id,
        document_id,
        query_embedding,
        candidate_k,
    )

    results = []
    for document, distance, metadata in zip(
        search_results["documents"][0],
        search_results["distances"][0],
        search_results["metadatas"][0],
    ):
        if not is_useful_chunk(document):
            continue

        results.append(
            {
                "文本块": document,
                "距离": distance,
                "元数据": metadata,
            }
        )

    return results


def search_relevant_chunks(
    user_id: str,
    document_id: str,
    query: str,
    n_results: int = 3,
) -> list[dict]:
    if n_results <= 0:
        raise ValueError("n_results 必须大于 0")

    candidates = retrieve_candidate_chunks(
        user_id,
        document_id,
        query,
        n_results * 5,
    )
    if not candidates:
        return []

    return rerank_chunks(query, candidates, n_results)


def retrieve_candidate_chunks_for_documents(
    user_id: str,
    document_ids: list[str],
    query: str,
    candidate_k: int,
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

    search_results = query_chunks_by_documents(
        user_id,
        document_ids,
        get_embedding(query),
        candidate_k,
    )

    results = []
    for document, distance, metadata in zip(
        search_results["documents"][0],
        search_results["distances"][0],
        search_results["metadatas"][0],
    ):
        if not is_useful_chunk(document):
            continue
        results.append(
            {
                "文本块": document,
                "距离": distance,
                "元数据": metadata,
            }
        )
    return results


def search_relevant_chunks_for_documents(
    user_id: str,
    document_ids: list[str],
    query: str,
    n_results: int = 3,
) -> list[dict]:
    if n_results <= 0:
        raise ValueError("n_results must be greater than 0")

    candidates = retrieve_candidate_chunks_for_documents(
        user_id,
        document_ids,
        query,
        n_results * 5,
    )
    if not candidates:
        return []
    return rerank_chunks(query, candidates, n_results)


def search_summary_chunks(
    user_id: str,
    document_id: str,
    query: str,
    seed_k: int = 5,
    per_query_k: int = 10,
    per_query_keep: int = 4,
    context_k: int = 12,
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

        if candidate["距离"] < existing["距离"]:
            existing["距离"] = candidate["距离"]


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
