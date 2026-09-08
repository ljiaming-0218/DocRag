from unittest.mock import Mock

import pytest

from eval.run_retrieval_eval import (
    calculate_page_metrics,
    run_retrieval_mode,
    validate_run_config,
)


def make_candidate(page: int, chunk_index: int) -> dict:
    return {
        "chunk_id": f"chunk-{chunk_index}",
        "文本块": f"evidence-{chunk_index}",
        "距离": 0.1 * (chunk_index + 1),
        "元数据": {
            "document_id": "document-1",
            "page_number": page,
            "chunk_index": chunk_index,
        },
        "dense_rank": chunk_index + 1,
        "retrieval_sources": ["dense"],
    }


def make_dependencies(candidates: list[dict]) -> dict:
    return {
        "retrieve_dense": Mock(return_value=candidates),
        "fuse_hybrid": Mock(return_value=candidates),
        "rerank": Mock(
            side_effect=lambda _query, chunks, top_k: list(
                reversed(chunks)
            )[:top_k]
        ),
        "filter_evidence": Mock(side_effect=lambda chunks: chunks),
    }


def test_validate_run_config_rejects_candidate_pool_smaller_than_top_k():
    with pytest.raises(ValueError, match="candidate_k"):
        validate_run_config(["dense"], candidate_k=2, top_k=3)


def test_page_metrics_are_not_applicable_without_gold_pages():
    metrics = calculate_page_metrics([], [], top_k=3)

    assert metrics["retrieval_evaluable"] is False
    assert metrics["recall_at_3"] is None


def test_page_metrics_calculate_hit_recall_and_first_relevant_rank():
    candidates = [
        {"page_number": 9},
        {"page_number": 2},
        {"page_number": 4},
    ]

    metrics = calculate_page_metrics([2, 4, 7], candidates, top_k=3)

    assert metrics["hit_rate_at_3"] == 1.0
    assert metrics["recall_at_3"] == pytest.approx(2 / 3)
    assert metrics["mrr_at_3"] == 0.5


def test_dense_mode_does_not_call_fusion_or_rerank():
    dependencies = make_dependencies(
        [make_candidate(1, 0), make_candidate(2, 1)]
    )

    result = run_retrieval_mode(
        "dense",
        dependencies,
        user_id="user-1",
        document_id="document-1",
        index_generation_id="generation-1",
        query="LoRA",
        candidate_k=10,
        top_k=1,
    )

    dependencies["fuse_hybrid"].assert_not_called()
    dependencies["rerank"].assert_not_called()
    assert len(result["final_results"]) == 1


def test_dense_rerank_reranks_dense_candidates_once():
    dependencies = make_dependencies(
        [make_candidate(1, 0), make_candidate(2, 1)]
    )

    result = run_retrieval_mode(
        "dense_rerank",
        dependencies,
        user_id="user-1",
        document_id="document-1",
        index_generation_id="generation-1",
        query="LoRA",
        candidate_k=10,
        top_k=1,
    )

    dependencies["fuse_hybrid"].assert_not_called()
    dependencies["rerank"].assert_called_once()
    assert result["final_results"][0]["chunk_id"] == "chunk-1"


def test_hybrid_rerank_fuses_before_reranking():
    dependencies = make_dependencies(
        [make_candidate(1, 0), make_candidate(2, 1)]
    )

    run_retrieval_mode(
        "hybrid_rerank",
        dependencies,
        user_id="user-1",
        document_id="document-1",
        index_generation_id="generation-1",
        query="LoRA",
        candidate_k=10,
        top_k=1,
    )

    dependencies["fuse_hybrid"].assert_called_once()
    dependencies["rerank"].assert_called_once()
