"""Evidence-aware metrics and summaries for retrieval-only evaluation."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from statistics import median


def normalize_evidence_text(text: str) -> str:
    """Normalize PDF and chunk text without changing its semantic content."""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.translate(str.maketrans({
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
    }))
    normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _compact_latin_text(text: str) -> str:
    """Remove layout whitespace for long PDF-derived Latin evidence spans."""
    return re.sub(r"\s+", "", text)


def _candidate_matches_span(candidate: dict, evidence: dict) -> bool:
    expected_page = evidence.get("page")
    if (
        expected_page is not None
        and candidate.get("page_number") != expected_page
    ):
        return False

    evidence_text = normalize_evidence_text(evidence.get("text", ""))
    candidate_text = normalize_evidence_text(candidate.get("text", ""))
    if not evidence_text:
        return False
    if evidence_text in candidate_text:
        return True

    # PDF extraction may merge adjacent Latin words. Use the compact
    # fallback only for long spans to limit accidental matches.
    compact_evidence = _compact_latin_text(evidence_text)
    compact_candidate = _compact_latin_text(candidate_text)
    return bool(
        len(compact_evidence) >= 24
        and compact_evidence.isascii()
        and compact_evidence in compact_candidate
    )


def evidence_alternatives(evidence: dict) -> list[dict]:
    """Return equivalent source spans for one evidence claim."""
    alternatives = evidence.get("alternatives")
    if isinstance(alternatives, list) and alternatives:
        return alternatives
    return [evidence]


def candidate_matches_evidence(candidate: dict, evidence: dict) -> bool:
    """Return whether a chunk supports an evidence claim at any valid span."""
    return any(
        _candidate_matches_span(candidate, alternative)
        for alternative in evidence_alternatives(evidence)
    )


def calculate_evidence_metrics(
    gold_evidence: list[dict],
    candidates: list[dict],
    top_k: int,
) -> dict:
    """Calculate evidence-level HitRate, Recall, and MRR."""
    hit_key = f"evidence_hit_rate_at_{top_k}"
    recall_key = f"evidence_recall_at_{top_k}"
    mrr_key = f"evidence_mrr_at_{top_k}"
    if not gold_evidence:
        return {
            "evidence_evaluable": False,
            "metric_basis": None,
            hit_key: None,
            recall_key: None,
            mrr_key: None,
            "matched_evidence_ids": [],
        }

    retrieved = candidates[:top_k]
    matched_ids = []
    first_relevant_rank = None
    for evidence_index, evidence in enumerate(gold_evidence, start=1):
        evidence_id = evidence.get("evidence_id") or f"evidence-{evidence_index}"
        matching_ranks = [
            rank
            for rank, candidate in enumerate(retrieved, start=1)
            if candidate_matches_evidence(candidate, evidence)
        ]
        if matching_ranks:
            matched_ids.append(evidence_id)
            rank = min(matching_ranks)
            if first_relevant_rank is None or rank < first_relevant_rank:
                first_relevant_rank = rank

    return {
        "evidence_evaluable": True,
        "metric_basis": (
            "gold_evidence_group"
            if any(item.get("alternatives") for item in gold_evidence)
            else "gold_evidence_text"
        ),
        hit_key: 1.0 if matched_ids else 0.0,
        recall_key: len(matched_ids) / len(gold_evidence),
        mrr_key: 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "matched_evidence_ids": matched_ids,
    }


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile_value * len(ordered)))
    return ordered[rank - 1]


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize_stage(
    modes: list[dict],
    *,
    stage: str,
    top_k: int,
) -> dict:
    evidence_metrics = [
        mode.get("metrics", {}).get(stage, {}).get("evidence", {})
        for mode in modes
    ]
    evidence_metrics = [
        metric
        for metric in evidence_metrics
        if metric.get("evidence_evaluable") is True
    ]
    hit_key = f"evidence_hit_rate_at_{top_k}"
    recall_key = f"evidence_recall_at_{top_k}"
    mrr_key = f"evidence_mrr_at_{top_k}"
    return {
        "evidence_evaluable_cases": len(evidence_metrics),
        hit_key: _average([metric[hit_key] for metric in evidence_metrics]),
        recall_key: _average([metric[recall_key] for metric in evidence_metrics]),
        mrr_key: _average([metric[mrr_key] for metric in evidence_metrics]),
    }


def _summarize_mode_cases(cases: list[dict], mode_name: str, config: dict) -> dict:
    modes = [
        case.get("modes", {}).get(mode_name, {})
        for case in cases
    ]
    successful = [
        mode
        for mode in modes
        if mode and mode.get("status") != "error"
    ]
    latencies = [mode["elapsed_seconds"] for mode in successful]
    candidate_k = config.get("candidate_k", 15)
    top_k = config.get("top_k", 3)
    unanswerable = [
        case
        for case in cases
        if case.get("answerable") is False
        and case.get("modes", {}).get(mode_name, {}).get("status") != "error"
    ]
    unanswerable_with_evidence = sum(
        bool(case["modes"][mode_name].get("filtered_results"))
        for case in unanswerable
    )
    return {
        "total_cases": len(cases),
        "successful_cases": len(successful),
        "execution_success_rate": (
            len(successful) / len(cases) if cases else 0.0
        ),
        "stages": {
            "candidate": _summarize_stage(
                successful,
                stage="candidate",
                top_k=candidate_k,
            ),
            "final": _summarize_stage(
                successful,
                stage="final",
                top_k=top_k,
            ),
            "filtered": _summarize_stage(
                successful,
                stage="filtered",
                top_k=top_k,
            ),
        },
        "unanswerable": {
            "cases": len(unanswerable),
            "returned_evidence_cases": unanswerable_with_evidence,
            "false_positive_rate": (
                unanswerable_with_evidence / len(unanswerable)
                if unanswerable
                else None
            ),
        },
        "latency_seconds": {
            "p50": median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
    }


def summarize_retrieval_results(result_data: dict) -> dict:
    """Aggregate per-mode retrieval quality and latency from one frozen run."""
    mode_names = result_data.get("config", {}).get("modes", [])
    config = result_data.get("config", {})
    results = result_data.get("results", [])
    summary = {
        "run_id": result_data.get("run_id"),
        "config": result_data.get("config", {}),
        "modes": {},
    }

    for mode_name in mode_names:
        by_category = defaultdict(list)
        for case in results:
            category = case.get("category", "unknown")
            by_category[category].append(case)

        mode_summary = _summarize_mode_cases(results, mode_name, config)
        mode_summary["by_category"] = {
            category: _summarize_mode_cases(
                category_cases,
                mode_name,
                config,
            )
            for category, category_cases in sorted(by_category.items())
        }
        summary["modes"][mode_name] = mode_summary
    return summary
