"""Inspect one case from a saved retrieval run without rerunning models."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.core import load_json


def metric_value(metrics: dict, prefix: str) -> float | None:
    return next(
        (value for key, value in metrics.items() if key.startswith(prefix)),
        None,
    )


def find_case(result_data: dict, case_id: str) -> dict:
    for case in result_data.get("results", []):
        if case.get("case_id") == case_id:
            return case
    raise KeyError(f"结果中不存在 case_id={case_id}")


def print_case(case: dict, candidate_limit: int) -> None:
    print(f"case_id: {case['case_id']}")
    print(f"category: {case['category']}")
    print(f"query: {case['query']}")
    print(f"gold_pages: {case.get('gold_pages') or []}")
    print(f"gold_evidence: {len(case.get('gold_evidence') or [])}")

    for mode_name, mode in case.get("modes", {}).items():
        print(f"\n[{mode_name}] status={mode.get('status', 'success')}")
        if mode.get("status") == "error":
            print(mode.get("error"))
            continue
        for stage in ("candidate", "final", "filtered"):
            evidence = mode.get("metrics", {}).get(stage, {}).get(
                "evidence",
                {},
            )
            recall = metric_value(evidence, "evidence_recall_at_")
            mrr = metric_value(evidence, "evidence_mrr_at_")
            matched = evidence.get("matched_evidence_ids") or []
            print(
                f"{stage}: recall={recall}, mrr={mrr}, "
                f"matched={matched}"
            )

        print("top candidates:")
        for candidate in mode.get("candidates", [])[:candidate_limit]:
            print(
                f"  #{candidate['rank']} page={candidate.get('page_number')} "
                f"chunk={candidate.get('chunk_index')} "
                f"dense={candidate.get('dense_rank')} "
                f"sparse={candidate.get('sparse_rank')} "
                f"rerank={candidate.get('rerank_score')}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查单题检索结果")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--candidate-limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_limit <= 0:
        raise ValueError("candidate-limit 必须大于 0")
    result_data = load_json(args.run)
    print_case(find_case(result_data, args.case_id), args.candidate_limit)


if __name__ == "__main__":
    main()
