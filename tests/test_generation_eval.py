from argparse import Namespace

from eval.metrics.generation import (
    build_manual_review_template,
    is_refusal,
    summarize_generation_results,
)
from eval.runners.generation import (
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
    assert is_refusal("当前文献未提供相关信息。") is True
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
