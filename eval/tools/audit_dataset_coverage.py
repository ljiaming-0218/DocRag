"""Report coverage gaps in the frozen DocRAG evaluation dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.core import DATASET_DIR, case_document_keys, load_json


REQUIRED_CATEGORIES = {
    "fact",
    "term",
    "summary",
    "comparison",
    "source_check",
    "unanswerable",
}
RECOMMENDED_MIN_CASES_PER_CATEGORY = 3
RECOMMENDED_MIN_FOLLOW_UP_CASES = 5


def load_cases(dataset_dir: Path = DATASET_DIR) -> list[dict]:
    cases = []
    for filename in ("single_turn.json", "follow_up.json"):
        payload = load_json(dataset_dir / filename)
        cases.extend(payload.get("cases") or [])
    return cases


def build_coverage_report(cases: list[dict]) -> dict:
    category_counts = Counter(case.get("category", "unknown") for case in cases)
    scenario_counts = Counter(case.get("scenario", "unknown") for case in cases)
    answerable_cases = [case for case in cases if case.get("answerable") is True]
    unanswerable_cases = [case for case in cases if case.get("answerable") is False]
    multi_document_cases = [
        case
        for case in cases
        if len(case_document_keys(case)) > 1
    ]
    document_counts = Counter(
        key
        for case in cases
        for key in case_document_keys(case)
    )

    category_gaps = {
        category: RECOMMENDED_MIN_CASES_PER_CATEGORY - category_counts[category]
        for category in sorted(REQUIRED_CATEGORIES)
        if category_counts[category] < RECOMMENDED_MIN_CASES_PER_CATEGORY
    }
    missing_gold_evidence = [
        case.get("case_id")
        for case in answerable_cases
        if not case.get("gold_evidence")
    ]
    duplicate_case_ids = sorted(
        case_id
        for case_id, count in Counter(
            case.get("case_id") for case in cases
        ).items()
        if count > 1
    )

    gaps = []
    if category_gaps:
        gaps.append("category_sample_count")
    if scenario_counts["follow_up"] < RECOMMENDED_MIN_FOLLOW_UP_CASES:
        gaps.append("follow_up_sample_count")
    if missing_gold_evidence:
        gaps.append("gold_evidence_coverage")
    if duplicate_case_ids:
        gaps.append("duplicate_case_ids")
    if not multi_document_cases:
        gaps.append("multi_document_contract")

    return {
        "total_cases": len(cases),
        "answerable_cases": len(answerable_cases),
        "unanswerable_cases": len(unanswerable_cases),
        "multi_document_cases": len(multi_document_cases),
        "category_counts": dict(sorted(category_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "document_counts": dict(sorted(document_counts.items())),
        "category_gaps": category_gaps,
        "missing_gold_evidence": missing_gold_evidence,
        "duplicate_case_ids": duplicate_case_ids,
        "gaps": gaps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DocRAG evaluation dataset coverage without running RAG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON coverage report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_coverage_report(load_cases())
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
