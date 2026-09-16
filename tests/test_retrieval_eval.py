from unittest.mock import Mock

import pytest

from eval.runners.retrieval import (
    calculate_page_metrics,
    run_retrieval_mode,
    select_cases,
    select_documents,
    validate_run_config,
)
from eval.metrics.retrieval import (
    calculate_evidence_metrics,
    normalize_evidence_text,
    summarize_retrieval_results,
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


def test_evidence_normalization_handles_pdf_line_break_hyphenation():
    assert normalize_evidence_text("non-\nparametric  memory") == (
        "nonparametric memory"
    )
    assert normalize_evidence_text("pre-trained model") == (
        "pretrained model"
    )


def test_evidence_metrics_handle_pdf_merged_latin_words():
    candidates = [{
        "page_number": 1,
        "text": (
            "the parametricmemory is a pre-trained seq2seq model "
            "and the non-parametric memory is a dense vector index"
        ),
    }]
    evidence = [{
        "evidence_id": "RAG-01-E1",
        "page": 1,
        "text": "the parametric memory is a pre-trained seq2seq model",
    }]

    metrics = calculate_evidence_metrics(evidence, candidates, top_k=1)

    assert metrics["evidence_recall_at_1"] == 1.0
    assert metrics["matched_evidence_ids"] == ["RAG-01-E1"]


def test_compact_evidence_fallback_rejects_short_accidental_match():
    evidence = [{"evidence_id": "E1", "page": 1, "text": "a b c"}]
    candidates = [{"page_number": 1, "text": "unrelated abc text"}]

    metrics = calculate_evidence_metrics(evidence, candidates, top_k=1)

    assert metrics["evidence_recall_at_1"] == 0.0


def test_select_retrieval_cases_and_documents_for_single_case():
    cases = [
        {"case_id": "RAG-01", "document_key": "rag"},
        {"case_id": "LORA-01", "document_key": "lora"},
    ]
    documents = [
        {"document_key": "rag"},
        {"document_key": "lora"},
    ]

    selected_cases = select_cases(cases, ["RAG-01"])
    selected_documents = select_documents(documents, selected_cases)

    assert selected_cases == [cases[0]]
    assert selected_documents == [documents[0]]


def test_select_retrieval_cases_rejects_unknown_case():
    with pytest.raises(ValueError, match="UNKNOWN"):
        select_cases(
            [{"case_id": "RAG-01", "document_key": "rag"}],
            ["UNKNOWN"],
        )


def test_evidence_metrics_require_matching_page_and_text():
    candidates = [
        {"page_number": 2, "text": "The pretrained weights are frozen."},
        {"page_number": 4, "text": "The pretrained weights are frozen."},
    ]
    evidence = [{
        "evidence_id": "LORA-01-E1",
        "page": 4,
        "text": "pretrained weights are frozen",
    }]

    metrics = calculate_evidence_metrics(evidence, candidates, top_k=2)

    assert metrics["evidence_hit_rate_at_2"] == 1.0
    assert metrics["evidence_recall_at_2"] == 1.0
    assert metrics["evidence_mrr_at_2"] == 0.5
    assert metrics["matched_evidence_ids"] == ["LORA-01-E1"]


def test_evidence_group_matches_any_alternative_without_growing_denominator():
    candidates = [{
        "page_number": 1,
        "text": "LoRA freezes the pre-trained model weights.",
    }]
    evidence = [{
        "evidence_id": "LORA-01-G1",
        "alternatives": [
            {"page": 1, "text": "freezes the pre-trained model weights"},
            {"page": 4, "text": "W0 is frozen"},
        ],
    }]

    metrics = calculate_evidence_metrics(evidence, candidates, top_k=1)

    assert metrics["metric_basis"] == "gold_evidence_group"
    assert metrics["evidence_recall_at_1"] == 1.0
    assert metrics["matched_evidence_ids"] == ["LORA-01-G1"]


def test_retrieval_summary_keeps_modes_separate():
    evidence_metrics = {
        "evidence_evaluable": True,
        "evidence_hit_rate_at_3": 1.0,
        "evidence_recall_at_3": 0.5,
        "evidence_mrr_at_3": 1.0,
    }
    run = {
        "run_id": "run-1",
        "config": {"modes": ["dense"], "top_k": 3},
        "results": [{
            "category": "fact",
            "answerable": True,
            "modes": {
                "dense": {
                    "elapsed_seconds": 0.25,
                    "metrics": {"final": {"evidence": evidence_metrics}},
                }
            },
        }],
    }

    summary = summarize_retrieval_results(run)

    dense = summary["modes"]["dense"]
    assert dense["execution_success_rate"] == 1.0
    assert dense["stages"]["final"]["evidence_recall_at_3"] == 0.5
    assert dense["latency_seconds"]["p95"] == 0.25


def test_retrieval_summary_counts_failed_cases_by_category():
    run = {
        "run_id": "run-2",
        "config": {"modes": ["dense"], "top_k": 3},
        "results": [
            {
                "category": "summary",
                "answerable": True,
                "modes": {
                    "dense": {
                        "status": "error",
                        "error_type": "RuntimeError",
                    },
                },
            },
            {
                "category": "summary",
                "answerable": True,
                "modes": {
                    "dense": {
                        "elapsed_seconds": 0.5,
                        "metrics": {
                            "final": {
                                "evidence": {
                                    "evidence_evaluable": True,
                                    "evidence_hit_rate_at_3": 1.0,
                                    "evidence_recall_at_3": 1.0,
                                    "evidence_mrr_at_3": 1.0,
                                },
                            },
                        },
                    },
                },
            },
        ],
    }

    summary = summarize_retrieval_results(run)

    summary_cases = summary["modes"]["dense"]["by_category"]["summary"]
    assert summary_cases["total_cases"] == 2
    assert summary_cases["successful_cases"] == 1
    assert summary_cases["execution_success_rate"] == 0.5


def test_retrieval_summary_reports_unanswerable_false_positives():
    run = {
        "config": {"modes": ["dense"], "top_k": 3},
        "results": [{
            "category": "unanswerable",
            "answerable": False,
            "modes": {
                "dense": {
                    "elapsed_seconds": 0.1,
                    "filtered_results": [{"chunk_id": "noise"}],
                }
            },
        }],
    }

    summary = summarize_retrieval_results(run)

    no_answer = summary["modes"]["dense"]["unanswerable"]
    assert no_answer["cases"] == 1
    assert no_answer["false_positive_rate"] == 1.0
