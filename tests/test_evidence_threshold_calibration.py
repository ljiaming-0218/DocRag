import pytest

from eval.calibrate_evidence_threshold import (
    calibrate,
    evaluate_threshold,
    get_successful_cases,
)


def make_source(page: int, score: float) -> dict:
    return {
        "rerank_score": score,
        "元数据": {"page_number": page},
    }


def make_case(
    question_id: str,
    *,
    answerable: bool,
    gold_pages: list[int],
    sources: list[dict],
) -> tuple[dict, dict]:
    question = {
        "question_id": question_id,
        "question_type": "fact" if answerable else "unanswerable",
        "expected_answerable": answerable,
        "source_pages": gold_pages,
    }
    result = {
        "question_id": question_id,
        "status": "success",
        "actual": {"sources": sources},
    }
    return question, result


def test_threshold_balances_gold_hit_and_unanswerable_rejection():
    answerable_question, answerable_result = make_case(
        "Q-1",
        answerable=True,
        gold_pages=[1],
        sources=[make_source(1, 0.9)],
    )
    unanswerable_question, unanswerable_result = make_case(
        "Q-2",
        answerable=False,
        gold_pages=[],
        sources=[make_source(8, 0.2)],
    )

    report = calibrate(
        {"results": [answerable_result, unanswerable_result]},
        [answerable_question, unanswerable_question],
    )

    assert report["recommended_threshold"] == 0.9
    assert report["recommended_metrics"]["true_positive"] == 1
    assert report["recommended_metrics"]["true_negative"] == 1
    assert report["baseline_metrics"]["false_positive"] == 1
    assert report["recommended_case_decisions"][1][
        "system_would_answer"
    ] is False


def test_evaluate_threshold_tracks_partial_gold_page_recall():
    cases = [{
        "question_id": "Q-1",
        "question_type": "summary",
        "expected_answerable": True,
        "gold_source_pages": [1, 2],
        "sources": [make_source(1, 0.9), make_source(2, 0.4)],
    }]

    metrics = evaluate_threshold(cases, threshold=0.5, top_k=3)

    assert metrics["true_positive"] == 1
    assert metrics["mean_gold_page_recall"] == 0.5


def test_missing_rerank_score_fails_fast():
    question, result = make_case(
        "Q-1",
        answerable=True,
        gold_pages=[1],
        sources=[{"元数据": {"page_number": 1}}],
    )

    with pytest.raises(ValueError, match="rerank_score"):
        get_successful_cases(
            {"results": [result]},
            {"Q-1": question},
        )


def test_error_cases_do_not_enter_calibration():
    question = {
        "question_id": "Q-1",
        "expected_answerable": True,
        "source_pages": [1],
    }

    cases = get_successful_cases(
        {"results": [{"question_id": "Q-1", "status": "error"}]},
        {"Q-1": question},
    )

    assert cases == []


def test_single_class_dataset_does_not_recommend_threshold():
    question, result = make_case(
        "Q-1",
        answerable=True,
        gold_pages=[1],
        sources=[make_source(1, 0.9)],
    )

    report = calibrate({"results": [result]}, [question])

    assert report["calibration_ready"] is False
    assert report["recommended_threshold"] is None
