"""Evaluate the production summary retrieval path with deterministic queries."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from time import perf_counter

from eval.core import (
    BACKEND_ROOT,
    DATASET_DIR,
    RUNS_DIR,
    EvalApiClient,
    build_run_id,
    dataset_fingerprint,
    load_json,
    save_json,
    utc_now_iso,
)
from eval.metrics.retrieval import calculate_evidence_metrics
from eval.runners.retrieval import (
    select_cases,
    select_documents,
    serialize_candidates,
)


logger = logging.getLogger(__name__)


def load_summary_dependencies():
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))

    from services import search_service
    from services.summary_query_service import build_default_summary_queries

    # Keep the real retrieval orchestration, but make query generation
    # deterministic so this runner never calls the answer LLM.
    search_service.build_summary_subqueries = (
        lambda query, _seed_sources: build_default_summary_queries(query)
    )
    return search_service.search_summary_chunks


def run_case(case: dict, scope: dict, user_id: str, search, context_k: int) -> dict:
    started_at = perf_counter()
    try:
        response = search(
            user_id,
            scope["document_id"],
            case["query"],
            context_k=context_k,
            index_generation_id=scope.get("active_index_generation_id"),
        )
        sources = serialize_candidates(response["sources"])
        metrics = calculate_evidence_metrics(
            case["gold_evidence"], sources, context_k,
        )
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "query": case["query"],
            "status": "success",
            "elapsed_seconds": round(perf_counter() - started_at, 6),
            "retrieval_queries": response["retrieval_queries"],
            "sources": sources,
            "metrics": metrics,
        }
    except Exception as error:
        logger.exception("summary_retrieval_case_failed case_id=%s", case["case_id"])
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "query": case["query"],
            "status": "error",
            "elapsed_seconds": round(perf_counter() - started_at, 6),
            "error_type": type(error).__name__,
            "error": str(error),
        }


def summarize(results: list[dict], context_k: int) -> dict:
    successful = [item for item in results if item["status"] == "success"]
    evaluable = [
        item for item in successful
        if item["metrics"]["evidence_evaluable"]
    ]
    def average(key: str) -> float | None:
        return (
            sum(item["metrics"][key] for item in evaluable) / len(evaluable)
            if evaluable else None
        )

    return {
        "total_cases": len(results),
        "successful_cases": len(successful),
        "execution_success_rate": len(successful) / len(results) if results else None,
        "evidence_evaluable_cases": len(evaluable),
        f"evidence_hit_rate_at_{context_k}": average(
            f"evidence_hit_rate_at_{context_k}"
        ),
        f"evidence_recall_at_{context_k}": average(
            f"evidence_recall_at_{context_k}"
        ),
        f"evidence_mrr_at_{context_k}": average(
            f"evidence_mrr_at_{context_k}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评估确定性多查询 Summary 检索")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--context-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--chunk-strategy", default="recursive")
    parser.add_argument(
        "--username", default=os.environ.get("DOCRAG_EVAL_USERNAME", "docrag-retrieval-eval"),
    )
    parser.add_argument(
        "--password", default=os.environ.get("DOCRAG_EVAL_PASSWORD", "docrag-eval-password"),
    )
    args = parser.parse_args()
    if args.context_k <= 0:
        parser.error("context-k 必须大于 0")
    logging.basicConfig(level=logging.INFO)

    documents_path = DATASET_DIR / "documents.json"
    cases_path = DATASET_DIR / "single_turn.json"
    summary_cases = [
        case for case in load_json(cases_path)["cases"]
        if case["category"] == "summary"
    ]
    cases = select_cases(summary_cases, args.case_ids)
    if any(case["category"] != "summary" for case in cases):
        parser.error("只支持 category=summary 的案例")
    documents = select_documents(load_json(documents_path), cases)

    with EvalApiClient(args.base_url) as client:
        client.check_ready()
        user_id = client.authenticate(args.username, args.password)
        scopes = client.index_documents(
            user_id, documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunk_strategy=args.chunk_strategy,
        )

    search = load_summary_dependencies()
    run_id = build_run_id("summary-retrieval")
    run_dir = RUNS_DIR / "summary_retrieval" / run_id
    data = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "dataset_fingerprint": dataset_fingerprint(documents_path, cases_path),
        "config": {
            "context_k": args.context_k,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "chunk_strategy": args.chunk_strategy,
            "query_variant": "original",
            "summary_queries": "deterministic_templates",
            "llm_called": False,
            "case_ids": args.case_ids,
        },
        "document_scopes": scopes,
        "results": [],
    }
    for case in cases:
        print(f"执行 {case['case_id']}")
        data["results"].append(
            run_case(case, scopes[case["document_key"]], user_id, search, args.context_k)
        )
        save_json(data, run_dir / "results.json")
    save_json(data["config"], run_dir / "config.json")
    save_json(summarize(data["results"], args.context_k), run_dir / "summary.json")
    print(f"Summary 检索评估完成: {run_dir}")


if __name__ == "__main__":
    main()
