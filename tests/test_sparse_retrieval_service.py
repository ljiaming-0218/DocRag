from unittest.mock import Mock

import pytest

from services import sparse_retrieval_service as service


@pytest.fixture(autouse=True)
def clear_sparse_cache():
    service.clear_sparse_index_cache()
    yield
    service.clear_sparse_index_cache()


def make_chunk(
    text: str,
    *,
    document_id: str,
    chunk_index: int,
) -> dict:
    return {
        "chunk_id": f"{document_id}-{chunk_index}",
        "文本块": text,
        "元数据": {
            "user_id": "user-1",
            "document_id": document_id,
            "page_number": 1,
            "chunk_index": chunk_index,
        },
    }


def make_corpus() -> list[dict]:
    return [
        make_chunk(
            "BGE-M3 supports multilingual dense and sparse retrieval.",
            document_id="document-1",
            chunk_index=0,
        ),
        make_chunk(
            "LoRA performs low-rank adaptation for large language models.",
            document_id="document-2",
            chunk_index=0,
        ),
        make_chunk(
            "Recall@10 is reported in Table 4 of the experiment.",
            document_id="document-2",
            chunk_index=1,
        ),
        make_chunk(
            "今天的天气适合户外活动。",
            document_id="document-3",
            chunk_index=0,
        ),
    ]


def test_tokenize_preserves_exact_terms_and_chinese_tokens():
    tokens = service.tokenize_for_bm25(
        "LoRA 与 BGE-M3 的 Recall@10 使用低秩适配。"
    )

    assert "lora" in tokens
    assert "bge-m3" in tokens
    assert "recall@10" in tokens
    assert any("低秩" in token for token in tokens)


@pytest.mark.parametrize(
    ("query", "expected_document_id"),
    [
        ("BGE-M3", "document-1"),
        ("LoRA", "document-2"),
        ("Recall@10", "document-2"),
    ],
)
def test_search_sparse_chunks_ranks_exact_term_first(
    query,
    expected_document_id,
):
    result = service.search_sparse_chunks(query, make_corpus(), 3)

    assert result
    assert result[0]["元数据"]["document_id"] == expected_document_id
    assert result[0]["sparse_score"] > 0
    assert result[0]["sparse_rank"] == 1
    assert result[0]["距离"] is None
    assert result[0]["retrieval_sources"] == ["sparse"]


def test_search_sparse_chunks_returns_global_top_k_across_documents():
    chunks = make_corpus()
    chunks.append(
        make_chunk(
            "LoRA LoRA LoRA is discussed in another document.",
            document_id="document-4",
            chunk_index=0,
        )
    )

    result = service.search_sparse_chunks("LoRA", chunks, 2)

    assert len(result) == 2
    assert {
        item["元数据"]["document_id"]
        for item in result
    } == {"document-2", "document-4"}
    assert [item["sparse_rank"] for item in result] == [1, 2]


def test_search_sparse_chunks_returns_empty_for_no_corpus_or_match():
    assert service.search_sparse_chunks("LoRA", [], 3) == []
    assert service.search_sparse_chunks(
        "unmatched-special-term",
        make_corpus(),
        3,
    ) == []


@pytest.mark.parametrize(
    ("query", "candidate_k", "message"),
    [
        ("", 3, "query cannot be empty"),
        ("LoRA", 0, "candidate_k must be greater than 0"),
    ],
)
def test_search_sparse_chunks_rejects_invalid_input(
    query,
    candidate_k,
    message,
):
    with pytest.raises(ValueError, match=message):
        service.search_sparse_chunks(query, make_corpus(), candidate_k)


def test_build_bm25_index_rejects_chunks_without_searchable_text():
    with pytest.raises(ValueError, match="searchable text"):
        service.build_bm25_index([
            {"文本块": "   ", "元数据": {}},
            {"文本块": "***", "元数据": {}},
        ])


def test_retrieve_sparse_candidates_uses_user_scoped_corpus(monkeypatch):
    load_chunks = Mock(return_value=make_corpus())
    monkeypatch.setattr(service, "get_chunks_by_documents", load_chunks)

    result = service.retrieve_sparse_candidates(
        " user-1 ",
        ["document-1", "document-2"],
        "LoRA",
        2,
    )

    load_chunks.assert_called_once_with(
        "user-1",
        ["document-1", "document-2"],
    )
    assert result[0]["元数据"]["document_id"] == "document-2"


def test_sparse_cache_and_corpus_are_generation_scoped(monkeypatch):
    load_chunks = Mock(return_value=make_corpus())
    monkeypatch.setattr(service, "get_chunks_by_documents", load_chunks)
    generations = {
        "document-1": "generation-a",
        "document-2": "generation-b",
    }

    result = service.retrieve_sparse_candidates(
        "user-1",
        ["document-1", "document-2"],
        "LoRA",
        2,
        generations,
    )

    assert result
    load_chunks.assert_called_once_with(
        "user-1",
        ["document-1", "document-2"],
        generations,
    )


def test_retrieve_sparse_candidates_reuses_normalized_scope_cache(
    monkeypatch,
):
    load_chunks = Mock(return_value=make_corpus())
    monkeypatch.setattr(service, "get_chunks_by_documents", load_chunks)

    first = service.retrieve_sparse_candidates(
        "user-1",
        ["document-2", "document-1"],
        "LoRA",
        2,
    )
    second = service.retrieve_sparse_candidates(
        "user-1",
        ["document-1", "document-2", "document-1"],
        "BGE-M3",
        2,
    )

    assert first
    assert second
    load_chunks.assert_called_once_with(
        "user-1",
        ["document-1", "document-2"],
    )


def test_invalidate_sparse_indexes_rebuilds_affected_scope(monkeypatch):
    load_chunks = Mock(return_value=make_corpus())
    monkeypatch.setattr(service, "get_chunks_by_documents", load_chunks)

    service.retrieve_sparse_candidates(
        "user-1",
        ["document-1", "document-2"],
        "LoRA",
        2,
    )
    invalidated = service.invalidate_sparse_indexes(
        "user-1",
        "document-2",
    )
    service.retrieve_sparse_candidates(
        "user-1",
        ["document-1", "document-2"],
        "LoRA",
        2,
    )

    assert invalidated == 1
    assert load_chunks.call_count == 2


def test_invalidate_sparse_indexes_keeps_unrelated_scope(monkeypatch):
    load_chunks = Mock(return_value=make_corpus())
    monkeypatch.setattr(service, "get_chunks_by_documents", load_chunks)

    service.get_or_build_sparse_index("user-1", ["document-1"])
    service.get_or_build_sparse_index("user-1", ["document-2"])

    assert service.invalidate_sparse_indexes(
        "user-1",
        "document-1",
    ) == 1
    service.get_or_build_sparse_index("user-1", ["document-2"])

    assert load_chunks.call_count == 2


def test_sparse_index_cache_evicts_least_recent_scope(monkeypatch):
    load_chunks = Mock(return_value=make_corpus())
    monkeypatch.setattr(service, "get_chunks_by_documents", load_chunks)
    monkeypatch.setattr(service, "SPARSE_INDEX_CACHE_MAX_ENTRIES", 2)

    service.get_or_build_sparse_index("user-1", ["document-1"])
    service.get_or_build_sparse_index("user-1", ["document-2"])
    service.get_or_build_sparse_index("user-1", ["document-3"])
    service.get_or_build_sparse_index("user-1", ["document-1"])

    assert load_chunks.call_count == 4


def test_retrieve_sparse_candidates_handles_unsearchable_corpus(
    monkeypatch,
):
    monkeypatch.setattr(
        service,
        "get_chunks_by_documents",
        Mock(return_value=[{"文本块": "***", "元数据": {}}]),
    )

    assert service.retrieve_sparse_candidates(
        "user-1",
        ["document-1"],
        "LoRA",
        3,
    ) == []


@pytest.mark.parametrize(
    ("user_id", "document_ids", "query", "candidate_k", "message"),
    [
        ("", ["document-1"], "LoRA", 3, "user_id cannot be empty"),
        ("user-1", [], "LoRA", 3, "document_ids cannot be empty"),
        ("user-1", ["document-1"], "", 3, "query cannot be empty"),
        (
            "user-1",
            ["document-1"],
            "LoRA",
            0,
            "candidate_k must be greater than 0",
        ),
    ],
)
def test_retrieve_sparse_candidates_rejects_invalid_input(
    monkeypatch,
    user_id,
    document_ids,
    query,
    candidate_k,
    message,
):
    load_chunks = Mock()
    monkeypatch.setattr(service, "get_chunks_by_documents", load_chunks)

    with pytest.raises(ValueError, match=message):
        service.retrieve_sparse_candidates(
            user_id,
            document_ids,
            query,
            candidate_k,
        )

    load_chunks.assert_not_called()
