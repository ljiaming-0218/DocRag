from unittest.mock import Mock

import pytest

from eval.runners.summary_retrieval import (
    build_evidence_loss_trace,
    calculate_coverage_metrics,
    run_case,
    summarize,
)
from medrag.backend.services.summary_query_service import (
    build_default_summary_queries,
)


def make_scope(document_id: str) -> dict:
    return {
        "document_id": document_id,
        "active_index_generation_id": f"generation-{document_id}",
    }


def make_source(document_id: str, text: str = "gold evidence") -> dict:
    return {
        "chunk_id": f"chunk-{document_id}",
        "文本块": text,
        "元数据": {
            "document_id": document_id,
            "page_number": 1,
            "chunk_index": 1,
        },
        "rerank_score": 0.8,
    }


class NoopCapture:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.fixture
def disable_trace_capture(monkeypatch):
    monkeypatch.setattr(
        "eval.runners.summary_retrieval.SummaryTraceRecorder.capture",
        lambda _self: NoopCapture(),
    )


def test_run_case_routes_single_document_focused_summary(disable_trace_capture):
    single = Mock(return_value={
        "sources": [make_source("doc-rag")],
        "retrieval_queries": ["summary"],
    })
    dependencies = {
        "module": Mock(),
        "single": single,
        "multi": Mock(),
        "overview": Mock(),
    }
    case = {
        "case_id": "RAG-03",
        "category": "summary",
        "summary_scope": "focused",
        "document_key": "rag",
        "query": "summary",
        "gold_evidence": [{
            "evidence_id": "E1",
            "page": 1,
            "text": "gold evidence",
        }],
    }

    result = run_case(
        case,
        {"rag": make_scope("doc-rag")},
        "user-1",
        dependencies,
        10,
    )

    assert result["status"] == "success"
    single.assert_called_once()
    dependencies["multi"].assert_not_called()
    dependencies["overview"].assert_not_called()


def test_run_case_routes_multi_document_overview(disable_trace_capture):
    overview = Mock(return_value={
        "sources": [make_source("doc-rag"), make_source("doc-lora")],
        "retrieval_queries": ["overview"],
    })
    dependencies = {
        "module": Mock(),
        "single": Mock(),
        "multi": Mock(),
        "overview": overview,
    }
    case = {
        "case_id": "MULTI-SUMMARY-01",
        "category": "summary",
        "summary_scope": "knowledge_base_overview",
        "document_keys": ["rag", "lora"],
        "query": "overview",
        "gold_evidence": [],
    }

    result = run_case(
        case,
        {"rag": make_scope("doc-rag"), "lora": make_scope("doc-lora")},
        "user-1",
        dependencies,
        10,
    )

    assert result["status"] == "success"
    overview.assert_called_once()
    dependencies["single"].assert_not_called()
    dependencies["multi"].assert_not_called()
    assert result["metrics"]["document_coverage_rate"] == 1.0


def test_coverage_metrics_separate_document_and_evidence_coverage():
    case = {
        "document_keys": ["rag", "lora"],
        "gold_evidence": [{"evidence_id": "E1"}, {"evidence_id": "E2"}],
    }
    metrics = calculate_coverage_metrics(
        case,
        [{"document_key": "rag"}],
        {"matched_evidence_ids": ["E1"]},
    )

    assert metrics["document_coverage_rate"] == 0.5
    assert metrics["evidence_coverage_rate"] == 0.5


def test_evidence_loss_trace_identifies_filter_loss():
    events = [{
        "stage": "rerank",
        "query": "summary method",
        "candidates": [{
            "rank": 2,
            "chunk_id": "chunk-1",
            "document_key": "rag",
            "page_number": 1,
            "chunk_index": 1,
            "rerank_score": 0.01,
            "matched_evidence_ids": ["E1"],
        }],
    }]

    trace = build_evidence_loss_trace(
        [{"evidence_id": "E1"}],
        events,
        overview=False,
    )

    assert trace[0]["lost_at"] == "threshold_filter"
    assert trace[0]["stage_matches"]["rerank"][0]["chunk_id"] == "chunk-1"


def test_summary_reports_required_metrics():
    metrics = {
        "evidence_evaluable": True,
        "evidence_recall_at_10": 0.5,
        "evidence_mrr_at_10": 0.25,
        "evidence_ndcg_at_10": 0.4,
        "document_coverage_rate": 1.0,
        "evidence_coverage_rate": 0.5,
    }

    result = summarize([{"status": "success", "metrics": metrics}], 10)

    assert result["evidence_recall_at_10"] == 0.5
    assert result["evidence_mrr_at_10"] == 0.25
    assert result["evidence_ndcg_at_10"] == 0.4
    assert result["document_coverage_rate"] == 1.0
    assert result["evidence_coverage_rate"] == 0.5


def test_chinese_summary_templates_include_english_retrieval_dimensions():
    queries = build_default_summary_queries(
        "总结论文的算术、常识、符号推理任务和推理效率。"
    )

    assert any("datasets" in query for query in queries)
    assert any("inference latency" in query for query in queries)
    assert any("symbolic reasoning" in query for query in queries)
