"""Evaluate the production Summary retrieval paths deterministically."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import contextmanager
from time import perf_counter

from eval.core import (
    BACKEND_ROOT,
    DATASET_DIR,
    RUNS_DIR,
    EvalApiClient,
    build_run_id,
    case_document_keys,
    dataset_fingerprint,
    load_json,
    save_json,
    utc_now_iso,
)
from eval.metrics.retrieval import (
    calculate_evidence_metrics,
    candidate_matches_evidence,
)
from eval.runners.retrieval import (
    select_cases,
    select_documents,
    serialize_candidates,
)


logger = logging.getLogger(__name__)
SUPPORTED_SUMMARY_SCOPES = {"focused", "knowledge_base_overview"}


def load_summary_dependencies() -> dict:
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))

    from services import search_service
    from services.summary_query_service import build_default_summary_queries

    # Keep production retrieval orchestration while avoiding an LLM dependency.
    search_service.build_summary_subqueries = (
        lambda query, _seed_sources: build_default_summary_queries(query)
    )
    return {
        "module": search_service,
        "single": search_service.search_summary_chunks,
        "multi": search_service.search_summary_chunks_for_documents,
        "overview": search_service.search_knowledge_base_overview_chunks,
    }


def _candidate_reference(candidate: dict) -> dict:
    return {
        "rank": candidate["rank"],
        "chunk_id": candidate.get("chunk_id"),
        "document_key": candidate.get("document_key"),
        "page_number": candidate.get("page_number"),
        "chunk_index": candidate.get("chunk_index"),
        "rerank_score": candidate.get("rerank_score"),
    }


class SummaryTraceRecorder:
    """Capture compact stage snapshots without changing production services."""

    def __init__(
        self,
        search_service,
        document_keys_by_id: dict[str, str],
        gold_evidence: list[dict],
    ) -> None:
        self.search_service = search_service
        self.document_keys_by_id = document_keys_by_id
        self.gold_evidence = gold_evidence
        self.events: list[dict] = []
        self._active_query: str | None = None

    def record(self, stage: str, query: str | None, chunks: list[dict]) -> None:
        serialized = serialize_candidates(chunks, self.document_keys_by_id)
        candidates = []
        for candidate in serialized:
            reference = _candidate_reference(candidate)
            reference["matched_evidence_ids"] = [
                evidence.get("evidence_id")
                for evidence in self.gold_evidence
                if candidate_matches_evidence(candidate, evidence)
            ]
            candidates.append(reference)
        self.events.append({
            "stage": stage,
            "query": query,
            "candidates": candidates,
        })

    @contextmanager
    def capture(self):
        service = self.search_service
        original_single = service.retrieve_candidate_chunks
        original_multi = service.retrieve_candidate_chunks_for_documents
        original_rerank = service.rerank_chunks
        original_filter = service.filter_relevant_evidence

        def retrieve_single(*args, **kwargs):
            result = original_single(*args, **kwargs)
            self._active_query = args[2]
            self.record("candidate", self._active_query, result)
            return result

        def retrieve_multi(*args, **kwargs):
            result = original_multi(*args, **kwargs)
            self._active_query = args[2]
            self.record("candidate", self._active_query, result)
            return result

        def rerank(query, chunks, top_k):
            result = original_rerank(query, chunks, top_k)
            self._active_query = query
            self.record("rerank", query, result)
            return result

        def filter_evidence(chunks, min_score=None):
            result = original_filter(chunks, min_score)
            self.record("filter", self._active_query, result)
            return result

        service.retrieve_candidate_chunks = retrieve_single
        service.retrieve_candidate_chunks_for_documents = retrieve_multi
        service.rerank_chunks = rerank
        service.filter_relevant_evidence = filter_evidence
        try:
            yield
        finally:
            service.retrieve_candidate_chunks = original_single
            service.retrieve_candidate_chunks_for_documents = original_multi
            service.rerank_chunks = original_rerank
            service.filter_relevant_evidence = original_filter


def _matching_references(
    events: list[dict],
    stage: str,
    evidence_id: str,
) -> list[dict]:
    matches = []
    seen = set()
    for event in events:
        if event["stage"] != stage:
            continue
        for candidate in event["candidates"]:
            if evidence_id not in candidate["matched_evidence_ids"]:
                continue
            key = (event.get("query"), candidate.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            matches.append({
                "query": event.get("query"),
                **_candidate_reference(candidate),
            })
    return matches


def build_evidence_loss_trace(
    gold_evidence: list[dict],
    events: list[dict],
    *,
    overview: bool,
) -> list[dict]:
    traces = []
    for evidence in gold_evidence:
        evidence_id = evidence.get("evidence_id")
        stage_matches = {
            stage: _matching_references(events, stage, evidence_id)
            for stage in ("candidate", "rerank", "filter", "final")
        }
        if stage_matches["final"]:
            lost_at = None
        elif stage_matches["filter"]:
            lost_at = "final_selection"
        elif stage_matches["rerank"]:
            lost_at = "threshold_filter"
        elif stage_matches["candidate"]:
            lost_at = "rerank_truncation"
        else:
            lost_at = "candidate_missing"
        traces.append({
            "evidence_id": evidence_id,
            "lost_at": lost_at,
            "stage_matches": stage_matches,
        })
    return traces


def calculate_coverage_metrics(
    case: dict,
    sources: list[dict],
    evidence_metrics: dict,
) -> dict:
    expected_documents = set(case_document_keys(case))
    retrieved_documents = {
        source.get("document_key")
        for source in sources
        if source.get("document_key") in expected_documents
    }
    matched_evidence = evidence_metrics.get("matched_evidence_ids", [])
    return {
        "expected_document_keys": sorted(expected_documents),
        "retrieved_document_keys": sorted(retrieved_documents),
        "document_coverage_rate": (
            len(retrieved_documents) / len(expected_documents)
            if expected_documents else None
        ),
        "evidence_coverage_rate": (
            len(matched_evidence) / len(case.get("gold_evidence", []))
            if case.get("gold_evidence") else None
        ),
    }


def run_case(
    case: dict,
    scopes: dict,
    user_id: str,
    dependencies: dict,
    context_k: int,
) -> dict:
    started_at = perf_counter()
    document_keys = case_document_keys(case)
    document_scopes = [scopes[key] for key in document_keys]
    document_ids = [scope["document_id"] for scope in document_scopes]
    generations = {
        scope["document_id"]: scope.get("active_index_generation_id")
        for scope in document_scopes
    }
    document_keys_by_id = dict(zip(document_ids, document_keys))
    summary_scope = case.get("summary_scope")
    if summary_scope not in SUPPORTED_SUMMARY_SCOPES:
        raise ValueError(f"unsupported summary_scope: {summary_scope}")

    recorder = SummaryTraceRecorder(
        dependencies["module"],
        document_keys_by_id,
        case["gold_evidence"],
    )
    try:
        with recorder.capture():
            if summary_scope == "knowledge_base_overview":
                response = dependencies["overview"](
                    user_id,
                    document_ids,
                    case["query"],
                    context_k=context_k,
                    index_generations=generations,
                )
            elif len(document_ids) == 1:
                response = dependencies["single"](
                    user_id,
                    document_ids[0],
                    case["query"],
                    context_k=context_k,
                    index_generation_id=generations[document_ids[0]],
                    original_query=case["query"],
                )
            else:
                response = dependencies["multi"](
                    user_id,
                    document_ids,
                    case["query"],
                    context_k=context_k,
                    index_generations=generations,
                    original_query=case["query"],
                )

        sources = serialize_candidates(
            response["sources"],
            document_keys_by_id,
        )
        recorder.record("final", None, response["sources"])
        metrics = calculate_evidence_metrics(
            case["gold_evidence"], sources, context_k,
        )
        metrics.update(calculate_coverage_metrics(case, sources, metrics))
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "summary_scope": summary_scope,
            "document_keys": document_keys,
            "query": case["query"],
            "status": "success",
            "elapsed_seconds": round(perf_counter() - started_at, 6),
            "retrieval_queries": response["retrieval_queries"],
            "sources": sources,
            "trace_events": recorder.events,
            "evidence_loss_trace": build_evidence_loss_trace(
                case["gold_evidence"],
                recorder.events,
                overview=(summary_scope == "knowledge_base_overview"),
            ),
            "metrics": metrics,
        }
    except Exception as error:
        logger.exception(
            "summary_retrieval_case_failed case_id=%s",
            case["case_id"],
        )
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "summary_scope": summary_scope,
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
        values = [
            item["metrics"][key]
            for item in evaluable
            if item["metrics"].get(key) is not None
        ]
        return sum(values) / len(values) if values else None

    return {
        "total_cases": len(results),
        "successful_cases": len(successful),
        "execution_success_rate": len(successful) / len(results) if results else None,
        "evidence_evaluable_cases": len(evaluable),
        f"evidence_recall_at_{context_k}": average(
            f"evidence_recall_at_{context_k}"
        ),
        f"evidence_mrr_at_{context_k}": average(
            f"evidence_mrr_at_{context_k}"
        ),
        f"evidence_ndcg_at_{context_k}": average(
            f"evidence_ndcg_at_{context_k}"
        ),
        "document_coverage_rate": average("document_coverage_rate"),
        "evidence_coverage_rate": average("evidence_coverage_rate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate production-aligned Summary retrieval",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--context-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--chunk-strategy", default="recursive")
    parser.add_argument(
        "--username",
        default=os.environ.get("DOCRAG_EVAL_USERNAME", "docrag-retrieval-eval"),
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("DOCRAG_EVAL_PASSWORD", "docrag-eval-password"),
    )
    args = parser.parse_args()
    if args.context_k <= 0:
        parser.error("context-k must be greater than 0")
    logging.basicConfig(level=logging.INFO)

    documents_path = DATASET_DIR / "documents.json"
    cases_path = DATASET_DIR / "single_turn.json"
    summary_cases = [
        case for case in load_json(cases_path)["cases"]
        if case["category"] == "summary"
    ]
    cases = select_cases(summary_cases, args.case_ids)
    if any(
        case.get("summary_scope") not in SUPPORTED_SUMMARY_SCOPES
        for case in cases
    ):
        parser.error("every Summary case must define a supported summary_scope")
    documents = select_documents(load_json(documents_path), cases)

    with EvalApiClient(args.base_url) as client:
        client.check_ready()
        user_id = client.authenticate(args.username, args.password)
        scopes = client.index_documents(
            user_id,
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunk_strategy=args.chunk_strategy,
        )

    dependencies = load_summary_dependencies()
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
            "production_summary_paths": True,
            "llm_called": False,
            "case_ids": args.case_ids,
        },
        "document_scopes": scopes,
        "results": [],
    }
    for case in cases:
        print(f"执行 {case['case_id']}")
        data["results"].append(
            run_case(case, scopes, user_id, dependencies, args.context_k)
        )
        save_json(data, run_dir / "results.json")
    save_json(data["config"], run_dir / "config.json")
    save_json(summarize(data["results"], args.context_k), run_dir / "summary.json")
    print(f"Summary 检索评估完成: {run_dir}")


if __name__ == "__main__":
    main()
