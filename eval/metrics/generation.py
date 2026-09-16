"""Deterministic summaries for end-to-end generation evaluation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median


REFUSAL_MARKERS = (
    "当前文献未提供",
    "文献中未提供",
    "根据当前文献内容无法确定",
    "无法从当前文献",
    "没有提供相关信息",
    "insufficient information",
    "not provided in the document",
)
MANUAL_REVIEW_FIELDS = (
    "answer_correctness",
    "answer_completeness",
    "faithfulness",
    "citation_correctness",
)
LLM_JUDGE_FIELDS = MANUAL_REVIEW_FIELDS


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile_value * len(ordered)))
    return ordered[rank - 1]


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def is_refusal(answer: str | None) -> bool:
    normalized = (answer or "").strip().casefold()
    return any(marker.casefold() in normalized for marker in REFUSAL_MARKERS)


def build_manual_review_template() -> dict:
    return {
        "answer_correctness": None,
        "answer_completeness": None,
        "faithfulness": None,
        "citation_correctness": None,
        "notes": None,
    }


def _summarize_cases(cases: list[dict]) -> dict:
    successful = [case for case in cases if case.get("status") == "success"]
    latencies = [
        float(case["elapsed_seconds"])
        for case in successful
        if case.get("elapsed_seconds") is not None
    ]
    route_checks = [
        case.get("actual", {}).get("task")
        == case.get("expected_task")
        for case in successful
        if case.get("expected_task")
    ]
    refusal_checks = [
        is_refusal(case.get("actual", {}).get("answer"))
        != bool(case.get("answerable"))
        for case in successful
        if case.get("answerable") is not None
    ]
    source_behavior_checks = [
        bool(case.get("actual", {}).get("sources"))
        == bool(case.get("answerable"))
        for case in successful
        if case.get("answerable") is not None
    ]
    citation_metrics = [
        case.get("citation_evidence", {})
        for case in successful
        if case.get("citation_evidence", {}).get(
            "evidence_evaluable"
        ) is True
    ]
    review_summary = {}
    for field in MANUAL_REVIEW_FIELDS:
        values = [
            float(case["manual_review"][field])
            for case in successful
            if case.get("manual_review", {}).get(field) is not None
        ]
        review_summary[field] = {
            "reviewed_cases": len(values),
            "average_score": average(values),
        }

    judge_summary = {}
    judgements = [
        case.get("llm_judge", {})
        for case in successful
        if case.get("llm_judge")
    ]
    successful_judgements = [
        judgement
        for judgement in judgements
        if judgement.get("status") == "success"
    ]
    judge_errors = Counter(
        judgement.get("error_type", "unknown")
        for judgement in judgements
        if judgement.get("status") == "error"
    )
    for field in LLM_JUDGE_FIELDS:
        values = [
            float(judgement["scores"][field]["score"])
            for judgement in successful_judgements
            if judgement.get("scores", {}).get(field, {}).get("score")
            is not None
        ]
        judge_summary[field] = {
            "judged_cases": len(values),
            "average_score": average(values),
        }

    error_types = Counter(
        case.get("error_type", "unknown")
        for case in cases
        if case.get("status") != "success"
    )
    return {
        "total_cases": len(cases),
        "successful_cases": len(successful),
        "execution_success_rate": (
            len(successful) / len(cases) if cases else 0.0
        ),
        "task_route_accuracy": average(
            [float(value) for value in route_checks]
        ),
        "refusal_accuracy_heuristic": average(
            [float(value) for value in refusal_checks]
        ),
        "source_presence_accuracy": average(
            [float(value) for value in source_behavior_checks]
        ),
        "citation_evidence_evaluable_cases": len(citation_metrics),
        "citation_evidence_hit_rate": average([
            metric["evidence_hit_rate"] for metric in citation_metrics
        ]),
        "citation_evidence_recall": average([
            metric["evidence_recall"] for metric in citation_metrics
        ]),
        "latency_seconds": {
            "p50": median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "manual_review": review_summary,
        "llm_judge": {
            "eligible_cases": len(successful),
            "attempted_cases": len(judgements),
            "successful_cases": len(successful_judgements),
            "execution_success_rate": (
                len(successful_judgements) / len(judgements)
                if judgements
                else None
            ),
            "scores": judge_summary,
            "error_types": dict(judge_errors),
        },
        "error_types": dict(error_types),
    }


def summarize_generation_results(result_data: dict) -> dict:
    results = result_data.get("results", [])
    grouped = defaultdict(list)
    for case in results:
        grouped[case.get("category", "unknown")].append(case)

    return {
        "run_id": result_data.get("run_id"),
        "dataset_fingerprint": result_data.get("dataset_fingerprint"),
        "config": result_data.get("config", {}),
        "overall": _summarize_cases(results),
        "by_category": {
            category: _summarize_cases(cases)
            for category, cases in sorted(grouped.items())
        },
        "metric_notes": {
            "refusal_accuracy_heuristic": (
                "基于固定拒答短语，仅用于回归提示，不替代人工判断"
            ),
            "source_presence_accuracy": (
                "仅判断引用是否存在，不代表引用与答案真正一致"
            ),
            "manual_review_scale": "0 到 1；None 表示尚未人工复核",
            "llm_judge_scale": (
                "0 到 1；LLM Judge 是辅助指标，不能替代人工抽检"
            ),
        },
    }
