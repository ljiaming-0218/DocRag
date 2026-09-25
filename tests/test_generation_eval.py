from argparse import Namespace

from eval.metrics.generation import (
    build_manual_review_template,
    is_refusal,
    summarize_generation_results,
)
from eval.runners.generation import (
    build_error_result,
    build_public_config,
    calculate_citation_evidence_metrics,
    select_cases,
    select_documents,
    validate_generation_config,
)


def make_success_case(
    *,
    category: str = "fact",
    answerable: bool = True,
    answer: str = "文献说明 RAG 使用外部知识。",
    sources: list[dict] | None = None,
) -> dict:
    return {
        "category": category,
        "expected_task": "qa",
        "answerable": answerable,
        "status": "success",
        "elapsed_seconds": 0.5,
        "actual": {
            "task": "qa",
            "answer": answer,
            "sources": sources if sources is not None else [{}],
        },
        "citation_evidence": {
            "evidence_evaluable": answerable,
            "evidence_hit_rate": 1.0 if answerable else None,
            "evidence_recall": 0.5 if answerable else None,
        },
        "manual_review": build_manual_review_template(),
    }


def test_public_generation_config_does_not_store_password():
    config = build_public_config(Namespace(
        username="eval-user",
        password="secret-password",
        n_results=3,
    ))

    assert config == {"username": "eval-user", "n_results": 3}


def test_select_generation_cases_rejects_unknown_case_id():
    cases = [{"case_id": "RAG-01"}]

    try:
        select_cases(cases, ["UNKNOWN"])
    except ValueError as error:
        assert "UNKNOWN" in str(error)
    else:
        raise AssertionError("未知 case_id 应被拒绝")


def test_select_documents_only_keeps_documents_required_by_cases():
    documents = [
        {"document_key": "rag"},
        {"document_key": "lora"},
        {"document_key": "cot"},
    ]

    selected = select_documents(
        documents,
        [{"document_key": "rag"}],
    )

    assert selected == [{"document_key": "rag"}]


def test_select_documents_supports_multi_document_cases():
    documents = [
        {"document_key": "rag"},
        {"document_key": "lora"},
        {"document_key": "cot"},
    ]

    selected = select_documents(
        documents,
        [{"document_keys": ["rag", "lora"]}],
    )

    assert selected == documents[:2]


def test_generation_config_rejects_overlap_equal_to_chunk_size():
    config = Namespace(
        chunk_size=500,
        chunk_overlap=500,
        history_limit=6,
        n_results=3,
        timeout=180,
        request_interval=0,
    )

    try:
        validate_generation_config(config)
    except ValueError as error:
        assert "chunk_overlap" in str(error)
    else:
        raise AssertionError("非法 overlap 应被拒绝")


def test_refusal_detection_matches_project_fallback_text():
    assert is_refusal("\u5f53\u524d\u6587\u6863\u672a\u63d0\u4f9b\u76f8\u5173\u4fe1\u606f") is True
    assert is_refusal("当前文献未提供相关信息。") is True
    assert is_refusal(
        "\u53c2\u8003\u7247\u6bb5\u5df2\u8986\u76d6\u95ee\u9898\u3002"
        + "x" * 250
        + "\u7247\u6bb5\u672a\u63d0\u4f9b\u6b64\u5185\u5bb9"
    ) is False
    assert is_refusal("论文提出了低秩矩阵分解方法。") is False


def test_generation_summary_keeps_failed_cases_in_denominator():
    result_data = {
        "run_id": "generation-1",
        "results": [
            make_success_case(category="summary"),
            {
                "category": "summary",
                "status": "error",
                "error_type": "HTTPError",
            },
        ],
    }

    summary = summarize_generation_results(result_data)
    summary_group = summary["by_category"]["summary"]

    assert summary_group["total_cases"] == 2
    assert summary_group["successful_cases"] == 1
    assert summary_group["execution_success_rate"] == 0.5


def test_generation_summary_does_not_invent_manual_scores():
    summary = summarize_generation_results({
        "results": [make_success_case()],
    })

    correctness = summary["overall"]["manual_review"][
        "answer_correctness"
    ]
    assert correctness["reviewed_cases"] == 0
    assert correctness["average_score"] is None


def test_generation_summary_reports_no_answer_and_token_usage():
    unanswerable = make_success_case(
        answerable=False,
        answer="论文声称存在该实验结果。",
        sources=[],
    )
    unanswerable["actual"]["token_usage"] = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }

    summary = summarize_generation_results({"results": [unanswerable]})
    overall = summary["overall"]

    assert overall["no_answer_false_positive_rate"] == 1.0
    assert overall["token_usage"]["total_tokens"] == 120


def test_generation_error_classifies_llm_failure_stage():
    response = type("Response", (), {
        "status_code": 503,
        "json": lambda self: {"error": "LLM_RATE_LIMITED"},
    })()
    error = RuntimeError("service unavailable")
    error.response = response

    result = build_error_result(
        {
            "case_id": "RAG-01",
            "document_key": "rag",
            "scenario": "single_turn",
            "category": "fact",
            "query": "question",
            "expected_task": "qa",
            "answerable": True,
        },
        error,
        1.25,
    )

    assert result["failure_stage"] == "generation"
    assert result["error_code"] == "LLM_RATE_LIMITED"


def test_citation_evidence_metrics_require_page_and_text_match():
    metrics = calculate_citation_evidence_metrics(
        [{
            "evidence_id": "RAG-01-E1",
            "page": 1,
            "text": "dense vector index of Wikipedia",
        }],
        [{
            "文本块": "The non-parametric memory is a dense vector index of Wikipedia.",
            "元数据": {"page_number": 1},
        }],
    )

    assert metrics["evidence_evaluable"] is True
    assert metrics["evidence_hit_rate"] == 1.0
    assert metrics["evidence_recall"] == 1.0
