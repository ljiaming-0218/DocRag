from unittest.mock import Mock

import pytest

from services import search_service as service


def make_candidate(
    text: str,
    *,
    document_id: str = "document-1",
    chunk_index: int = 0,
    distance: float | None = 0.2,
) -> dict:
    return {
        "chunk_id": f"{document_id}-{chunk_index}",
        "文本块": text,
        "距离": distance,
        "元数据": {
            "document_id": document_id,
            "page_number": 1,
            "chunk_index": chunk_index,
        },
    }


def test_dense_mode_does_not_call_sparse_retrieval(monkeypatch):
    monkeypatch.setattr(service, "RETRIEVAL_MODE", "dense")
    monkeypatch.setattr(
        service,
        "get_embedding",
        Mock(return_value=[0.1]),
    )
    query_chunks = Mock(return_value={
        "ids": [["chunk-1"]],
        "documents": [["LoRA evidence"]],
        "distances": [[0.2]],
        "metadatas": [[{
            "document_id": "document-1",
            "page_number": 1,
            "chunk_index": 0,
        }]],
    })
    sparse_retrieval = Mock()
    monkeypatch.setattr(service, "query_chunks", query_chunks)
    monkeypatch.setattr(
        service,
        "retrieve_sparse_candidates",
        sparse_retrieval,
    )

    result = service.retrieve_candidate_chunks(
        "user-1",
        "document-1",
        "LoRA",
        10,
    )

    assert result[0]["chunk_id"] == "chunk-1"
    assert result[0]["dense_rank"] == 1
    assert result[0]["retrieval_sources"] == ["dense"]
    sparse_retrieval.assert_not_called()


def test_hybrid_search_fuses_candidates_then_reranks_once(monkeypatch):
    monkeypatch.setattr(service, "RETRIEVAL_MODE", "hybrid")
    dense = [make_candidate("dense evidence")]
    sparse = [
        make_candidate(
            "sparse evidence",
            document_id="document-2",
            distance=None,
        )
    ]
    fused = [*dense, *sparse]

    dense_retrieval = Mock(return_value=dense)
    sparse_retrieval = Mock(return_value=sparse)
    fusion = Mock(return_value=fused)
    rerank = Mock(side_effect=lambda _query, chunks, top_k: chunks[:top_k])
    monkeypatch.setattr(
        service,
        "retrieve_dense_candidate_chunks",
        dense_retrieval,
    )
    monkeypatch.setattr(
        service,
        "retrieve_sparse_candidates",
        sparse_retrieval,
    )
    monkeypatch.setattr(service, "reciprocal_rank_fusion", fusion)
    monkeypatch.setattr(service, "rerank_chunks", rerank)
    evidence_filter = Mock(side_effect=lambda chunks: chunks)
    monkeypatch.setattr(
        service,
        "filter_relevant_evidence",
        evidence_filter,
    )

    result = service.search_relevant_chunks(
        "user-1",
        "document-1",
        "LoRA",
        n_results=2,
    )

    dense_retrieval.assert_called_once_with(
        "user-1",
        "document-1",
        "LoRA",
        10,
    )
    sparse_retrieval.assert_called_once_with(
        "user-1",
        ["document-1"],
        "LoRA",
        10,
    )
    fusion.assert_called_once_with(
        {"dense": dense, "sparse": sparse},
        limit=10,
    )
    rerank.assert_called_once_with("LoRA", fused, 2)
    evidence_filter.assert_called_once_with(fused)
    assert result == fused


def test_hybrid_retrieval_contract_merges_same_chunk(monkeypatch):
    monkeypatch.setattr(service, "RETRIEVAL_MODE", "hybrid")
    dense = [make_candidate("shared evidence", distance=0.2)]
    sparse = [
        {
            **make_candidate("shared evidence", distance=None),
            "sparse_score": 4.5,
            "sparse_rank": 1,
            "retrieval_sources": ["sparse"],
        }
    ]
    monkeypatch.setattr(
        service,
        "retrieve_dense_candidate_chunks",
        Mock(return_value=dense),
    )
    monkeypatch.setattr(
        service,
        "retrieve_sparse_candidates",
        Mock(return_value=sparse),
    )

    result = service.retrieve_candidate_chunks(
        "user-1",
        "document-1",
        "LoRA",
        10,
    )

    assert len(result) == 1
    assert result[0]["距离"] == 0.2
    assert result[0]["sparse_score"] == 4.5
    assert result[0]["retrieval_sources"] == ["dense", "sparse"]
    assert result[0]["rrf_score"] > 0


@pytest.mark.parametrize("failed_component", ["sparse", "fusion"])
def test_hybrid_failure_falls_back_to_dense(
    monkeypatch,
    failed_component,
):
    monkeypatch.setattr(service, "RETRIEVAL_MODE", "hybrid")
    dense = [make_candidate("dense evidence")]
    monkeypatch.setattr(
        service,
        "retrieve_dense_candidate_chunks",
        Mock(return_value=dense),
    )
    sparse = Mock(return_value=[make_candidate("sparse evidence")])
    fusion = Mock(return_value=dense)
    if failed_component == "sparse":
        sparse.side_effect = RuntimeError("sparse unavailable")
    else:
        fusion.side_effect = RuntimeError("fusion unavailable")
    monkeypatch.setattr(service, "retrieve_sparse_candidates", sparse)
    monkeypatch.setattr(service, "reciprocal_rank_fusion", fusion)
    monkeypatch.setattr(
        service,
        "rerank_chunks",
        Mock(side_effect=lambda _query, chunks, _top_k: chunks),
    )
    monkeypatch.setattr(
        service,
        "filter_relevant_evidence",
        Mock(side_effect=lambda chunks: chunks),
    )

    result = service.search_relevant_chunks(
        "user-1",
        "document-1",
        "LoRA",
        3,
    )

    assert result == dense


def test_hybrid_multi_document_search_preserves_scope(monkeypatch):
    monkeypatch.setattr(service, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(service, "get_embedding", Mock(return_value=[0.1]))
    monkeypatch.setattr(
        service,
        "query_chunks_by_documents",
        Mock(return_value={
            "ids": [["chunk-1"]],
            "documents": [["dense evidence"]],
            "distances": [[0.2]],
            "metadatas": [[{
                "document_id": "document-1",
                "page_number": 1,
                "chunk_index": 0,
            }]],
        }),
    )
    sparse_retrieval = Mock(return_value=[])
    monkeypatch.setattr(
        service,
        "retrieve_sparse_candidates",
        sparse_retrieval,
    )

    result = service.retrieve_candidate_chunks_for_documents(
        "user-1",
        ["document-1", "document-2"],
        "compare methods",
        10,
    )

    sparse_retrieval.assert_called_once_with(
        "user-1",
        ["document-1", "document-2"],
        "compare methods",
        10,
    )
    assert result[0]["retrieval_sources"] == ["dense"]


def test_invalid_retrieval_mode_stops_before_retrieval(monkeypatch):
    monkeypatch.setattr(service, "RETRIEVAL_MODE", "unsupported")
    dense_retrieval = Mock()
    monkeypatch.setattr(
        service,
        "retrieve_dense_candidate_chunks",
        dense_retrieval,
    )

    with pytest.raises(ValueError, match="dense, hybrid"):
        service.retrieve_candidate_chunks(
            "user-1",
            "document-1",
            "LoRA",
            10,
        )

    dense_retrieval.assert_not_called()


def test_summary_merge_accepts_sparse_only_distance():
    merged = {}
    service.merge_candidate_chunks(
        merged,
        [make_candidate("sparse evidence", distance=None)],
        "LoRA",
    )
    service.merge_candidate_chunks(
        merged,
        [make_candidate("dense evidence", distance=0.3)],
        "low-rank adaptation",
    )

    candidate = next(iter(merged.values()))
    assert candidate["距离"] == 0.3
    assert candidate["matched_queries"] == [
        "LoRA",
        "low-rank adaptation",
    ]


def test_summary_filters_seed_evidence_before_coverage_merge(monkeypatch):
    candidate = make_candidate("summary evidence")
    candidate["rerank_score"] = -1.0
    rerank = Mock(side_effect=lambda _query, chunks, _top_k: chunks)
    evidence_filter = Mock(return_value=[])
    monkeypatch.setattr(
        service,
        "retrieve_candidate_chunks",
        Mock(return_value=[candidate]),
    )
    monkeypatch.setattr(
        service,
        "build_summary_subqueries",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(service, "rerank_chunks", rerank)
    monkeypatch.setattr(
        service,
        "filter_relevant_evidence",
        evidence_filter,
    )

    result = service.search_summary_chunks(
        "user-1",
        "document-1",
        "summarize the paper",
    )

    assert rerank.call_count == 1
    evidence_filter.assert_called_once()
    assert result["sources"] == []


def test_summary_uses_original_query_when_rewrite_has_no_candidates(
    monkeypatch,
):
    original_candidate = make_candidate("original-query evidence")

    def retrieve(_user_id, _document_id, query, *_args):
        if query == "原始总结问题":
            return [original_candidate]
        return []

    monkeypatch.setattr(
        service,
        "retrieve_candidate_chunks",
        Mock(side_effect=retrieve),
    )
    monkeypatch.setattr(
        service,
        "rerank_chunks",
        Mock(side_effect=lambda _query, chunks, _top_k: chunks),
    )
    monkeypatch.setattr(
        service,
        "filter_relevant_evidence",
        Mock(side_effect=lambda chunks: chunks),
    )
    monkeypatch.setattr(
        service,
        "build_summary_subqueries",
        Mock(return_value=[]),
    )

    result = service.search_summary_chunks(
        "user-1",
        "document-1",
        "rewritten summary query",
        original_query="原始总结问题",
    )

    assert result["sources"] == [original_candidate]
    assert result["retrieval_queries"] == [
        "原始总结问题",
        "rewritten summary query",
    ]


def test_summary_selection_preserves_subquery_coverage():
    method = make_candidate("method", chunk_index=1)
    method["rerank_score"] = 0.9
    method["matched_queries"] = ["method query"]
    result = make_candidate("result", chunk_index=2)
    result["rerank_score"] = 0.1
    result["matched_queries"] = ["result query"]

    selected = service.select_query_covered_chunks(
        [method, result],
        ["method query", "result query"],
        limit=2,
    )

    assert selected == [method, result]


def test_knowledge_base_overview_preserves_document_coverage(monkeypatch):
    document_ids = ["document-1", "document-2", "document-3"]
    chunks = []
    for document_id in document_ids:
        chunks.extend([
            make_candidate(
                f"{document_id} title and authors",
                document_id=document_id,
                chunk_index=0,
            ),
            make_candidate(
                f"Abstract: {document_id} research objective",
                document_id=document_id,
                chunk_index=1,
            ),
            make_candidate(
                f"{document_id} method and conclusion",
                document_id=document_id,
                chunk_index=2,
            ),
        ])
    read_chunks = Mock(return_value=chunks)
    monkeypatch.setattr(
        service,
        "get_chunks_by_documents",
        read_chunks,
    )
    summary_queries = Mock()
    monkeypatch.setattr(
        service,
        "build_summary_subqueries",
        summary_queries,
    )

    result = service.search_summary_chunks_for_documents(
        "user-1",
        document_ids,
        "What do the documents in the current literature cover?",
        per_query_k=2,
        per_query_keep=2,
        context_k=6,
        index_generations={
            document_id: f"generation-{document_id[-1]}"
            for document_id in document_ids
        },
        overview=True,
    )

    returned_document_ids = [
        source["元数据"]["document_id"]
        for source in result["sources"]
    ]
    assert returned_document_ids == [
        "document-1",
        "document-1",
        "document-2",
        "document-2",
        "document-3",
        "document-3",
    ]
    assert result["retrieval_queries"] == [
        "What do the documents in the current literature cover?"
    ]
    assert result["sources"][0]["文本块"].startswith("Abstract:")
    assert result["sources"][1]["文本块"].endswith(
        "method and conclusion"
    )
    read_chunks.assert_called_once_with(
        "user-1",
        document_ids,
        {
            document_id: f"generation-{document_id[-1]}"
            for document_id in document_ids
        },
    )
    summary_queries.assert_not_called()


def test_knowledge_base_overview_uses_original_query_after_rewrite(
    monkeypatch,
):
    chunks = [
        make_candidate(
            "Abstract: document-1 research objective",
            document_id="document-1",
            chunk_index=1,
        ),
        make_candidate(
            "Abstract: document-2 research objective",
            document_id="document-2",
            chunk_index=1,
        ),
    ]
    read_chunks = Mock(return_value=chunks)
    summary_queries = Mock()
    monkeypatch.setattr(service, "get_chunks_by_documents", read_chunks)
    monkeypatch.setattr(
        service,
        "build_summary_subqueries",
        summary_queries,
    )

    result = service.search_summary_chunks_for_documents(
        "user-1",
        ["document-1", "document-2"],
        "Describe the research scope of the available material.",
        context_k=2,
        original_query="这个知识库中的文档分别研究了什么问题？",
        overview=True,
    )

    assert len(result["sources"]) == 2
    assert {
        source["元数据"]["document_id"]
        for source in result["sources"]
    } == {"document-1", "document-2"}
    summary_queries.assert_not_called()
