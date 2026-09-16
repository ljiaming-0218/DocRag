from unittest.mock import Mock

import pytest

from eval.judges.llm_judge import (
    build_judge_prompt,
    judge_case,
    parse_judge_response,
)
from eval.metrics.generation import summarize_generation_results
from eval.runners.judge import select_cases


def make_case() -> dict:
    return {
        "case_id": "RAG-01",
        "category": "fact",
        "query": "RAG 的非参数化记忆是什么？",
        "answerable": True,
        "reference_answer": "Wikipedia 稠密向量索引。",
        "answer_points": ["稠密向量索引"],
        "status": "success",
        "actual": {
            "answer": "它是 Wikipedia 的稠密向量索引。",
            "sources": [{
                "文本块": "The non-parametric memory is a dense vector index.",
                "元数据": {"page_number": 1},
            }],
        },
        "manual_review": {},
        "elapsed_seconds": 1.0,
    }


def judge_json(score: float = 1.0) -> str:
    fields = (
        "answer_correctness",
        "answer_completeness",
        "faithfulness",
        "citation_correctness",
    )
    items = ",".join(
        f'"{field}":{{"score":{score},"reason":"证据支持"}}'
        for field in fields
    )
    return "{" + items + "}"


def test_build_judge_prompt_contains_answer_reference_and_sources():
    prompt = build_judge_prompt(make_case())

    assert "Wikipedia 稠密向量索引" in prompt
    assert "dense vector index" in prompt
    assert "只输出一个 JSON 对象" in prompt


def test_parse_judge_response_accepts_fenced_json():
    parsed = parse_judge_response("```json\n" + judge_json(0.75) + "\n```")

    assert parsed["answer_correctness"]["score"] == 0.75


def test_parse_judge_response_rejects_score_outside_range():
    with pytest.raises(ValueError, match="0 到 1"):
        parse_judge_response(judge_json(1.5))


def test_judge_case_calls_llm_once_and_keeps_reasons():
    generate = Mock(return_value=judge_json())

    result = judge_case(make_case(), generate)

    generate.assert_called_once()
    assert generate.call_args.kwargs["operation"] == "eval_judge"
    assert result["status"] == "success"
    assert result["scores"]["faithfulness"]["reason"] == "证据支持"


def test_judge_case_skips_failed_generation_case():
    generate = Mock()

    result = judge_case({"status": "error"}, generate)

    generate.assert_not_called()
    assert result["status"] == "skipped"


def test_generation_summary_aggregates_llm_judge_scores():
    case = make_case()
    case["llm_judge"] = {
        "status": "success",
        "scores": parse_judge_response(judge_json(0.5)),
    }

    summary = summarize_generation_results({"results": [case]})

    correctness = summary["overall"]["llm_judge"]["scores"][
        "answer_correctness"
    ]
    assert correctness == {"judged_cases": 1, "average_score": 0.5}
    assert summary["overall"]["llm_judge"][
        "execution_success_rate"
    ] == 1.0


def test_generation_summary_keeps_failed_judges_in_denominator():
    success = make_case()
    success["llm_judge"] = {
        "status": "success",
        "scores": parse_judge_response(judge_json()),
    }
    failed = make_case()
    failed["case_id"] = "RAG-02"
    failed["llm_judge"] = {
        "status": "error",
        "error_type": "LLMServiceError",
    }

    summary = summarize_generation_results({"results": [success, failed]})
    judge = summary["overall"]["llm_judge"]

    assert judge["attempted_cases"] == 2
    assert judge["successful_cases"] == 1
    assert judge["execution_success_rate"] == 0.5
    assert judge["error_types"] == {"LLMServiceError": 1}


def test_select_cases_rejects_unknown_case_id():
    with pytest.raises(ValueError, match="UNKNOWN"):
        select_cases([make_case()], ["UNKNOWN"])
