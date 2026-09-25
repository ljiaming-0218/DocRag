"""Run retrieval-only comparisons without calling Query Rewrite or an LLM."""

import argparse
import copy
import logging
import os
import sys
from time import perf_counter
from typing import Callable

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
    summarize_retrieval_results,
)


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODES = ("dense", "dense_rerank", "hybrid_rerank")
DEFAULT_METRIC_K_VALUES = (3, 5, 15)
SUPPORTED_MODES = set(DEFAULT_MODES)

logger = logging.getLogger(__name__)


def validate_run_config(
    modes: list[str],
    candidate_k: int,
    top_k: int,
    metric_k_values: list[int] | None = None,
) -> None:
    unsupported = set(modes) - SUPPORTED_MODES
    if unsupported:
        raise ValueError(
            "不支持的检索模式: " + ", ".join(sorted(unsupported))
        )
    if candidate_k <= 0:
        raise ValueError("candidate_k 必须大于 0")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if candidate_k < top_k:
        raise ValueError("candidate_k 不能小于 top_k")
    metric_k_values = metric_k_values or list(DEFAULT_METRIC_K_VALUES)
    if any(value <= 0 for value in metric_k_values):
        raise ValueError("metric_k 必须大于 0")
    if max(metric_k_values) > candidate_k:
        raise ValueError("metric_k 不能大于 candidate_k")


def select_cases(cases: list[dict], case_ids: list[str] | None) -> list[dict]:
    if not case_ids:
        return cases
    requested = set(case_ids)
    selected = [case for case in cases if case.get("case_id") in requested]
    found = {case.get("case_id") for case in selected}
    missing = requested - found
    if missing:
        raise ValueError("找不到评估案例: " + ", ".join(sorted(missing)))
    return selected


def select_documents(documents: list[dict], cases: list[dict]) -> list[dict]:
    required_keys = {
        key
        for case in cases
        for key in case_document_keys(case)
    }
    selected = [
        document
        for document in documents
        if document.get("document_key") in required_keys
    ]
    found = {document.get("document_key") for document in selected}
    missing = required_keys - found
    if missing:
        raise ValueError("找不到评估文档: " + ", ".join(sorted(missing)))
    return selected


def load_retrieval_dependencies() -> dict[str, Callable]:
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))

    from services.evidence_filter_service import filter_relevant_evidence
    from services.rerank_service import rerank_chunks
    from services.search_service import (
        fuse_dense_and_sparse_candidates,
        retrieve_dense_candidate_chunks_for_documents,
    )

    return {
        "retrieve_dense": retrieve_dense_candidate_chunks_for_documents,
        "fuse_hybrid": fuse_dense_and_sparse_candidates,
        "rerank": rerank_chunks,
        "filter_evidence": filter_relevant_evidence,
    }


def serialize_candidate(
    candidate: dict,
    rank: int,
    document_keys_by_id: dict[str, str] | None = None,
) -> dict:
    metadata = candidate.get("元数据") or {}
    document_keys_by_id = document_keys_by_id or {}
    return {
        "rank": rank,
        "chunk_id": candidate.get("chunk_id"),
        "document_id": metadata.get("document_id"),
        "document_key": document_keys_by_id.get(metadata.get("document_id")),
        "page_number": metadata.get("page_number"),
        "chunk_index": metadata.get("chunk_index"),
        "text": candidate.get("文本块", ""),
        "distance": candidate.get("距离"),
        "dense_rank": candidate.get("dense_rank"),
        "sparse_rank": candidate.get("sparse_rank"),
        "rrf_score": candidate.get("rrf_score"),
        "rerank_score": candidate.get("rerank_score"),
        "retrieval_sources": candidate.get("retrieval_sources", []),
    }


def serialize_candidates(
    candidates: list[dict],
    document_keys_by_id: dict[str, str] | None = None,
) -> list[dict]:
    return [
        serialize_candidate(candidate, rank, document_keys_by_id)
        for rank, candidate in enumerate(candidates, start=1)
    ]


def calculate_page_metrics(
    gold_pages: list[int],
    candidates: list[dict],
    top_k: int,
    *,
    document_count: int = 1,
) -> dict:
    if not gold_pages or document_count > 1:
        return {
            "retrieval_evaluable": False,
            "reason": (
                "page metrics are ambiguous across multiple documents"
                if document_count > 1
                else "case has no gold pages"
            ),
            f"hit_rate_at_{top_k}": None,
            f"recall_at_{top_k}": None,
            f"mrr_at_{top_k}": None,
        }

    retrieved_pages = [
        candidate.get("page_number")
        for candidate in candidates[:top_k]
    ]
    gold_set = set(gold_pages)
    matched = gold_set & set(retrieved_pages)
    reciprocal_rank = 0.0
    for rank, page_number in enumerate(retrieved_pages, start=1):
        if page_number in gold_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "retrieval_evaluable": True,
        f"hit_rate_at_{top_k}": 1.0 if matched else 0.0,
        f"recall_at_{top_k}": len(matched) / len(gold_set),
        f"mrr_at_{top_k}": reciprocal_rank,
    }


def run_retrieval_mode(
    mode: str,
    dependencies: dict[str, Callable],
    *,
    user_id: str,
    document_ids: list[str],
    index_generations: dict[str, str | None],
    document_keys_by_id: dict[str, str],
    query: str,
    candidate_k: int,
    top_k: int,
) -> dict:
    started_at = perf_counter()
    dense_candidates = dependencies["retrieve_dense"](
        user_id,
        document_ids,
        query,
        candidate_k,
        index_generations,
    )

    if mode == "dense":
        candidates = dense_candidates
        candidate_snapshot = serialize_candidates(
            copy.deepcopy(candidates),
            document_keys_by_id,
        )
        ranked_results = candidates
        filtered_ranked_results = ranked_results
        final_results = ranked_results[:top_k]
        filtered_results = final_results
    elif mode == "dense_rerank":
        candidates = dense_candidates
        candidate_snapshot = serialize_candidates(
            copy.deepcopy(candidates),
            document_keys_by_id,
        )
        ranked_results = (
            dependencies["rerank"](
                query,
                candidates,
                len(candidates),
            )
            if candidates
            else []
        )
        filtered_ranked_results = dependencies["filter_evidence"](
            ranked_results
        )
        final_results = ranked_results[:top_k]
        filtered_results = dependencies["filter_evidence"](
            final_results
        )
    elif mode == "hybrid_rerank":
        candidates = dependencies["fuse_hybrid"](
            user_id,
            document_ids,
            query,
            dense_candidates,
            candidate_k,
            index_generations,
        )
        candidate_snapshot = serialize_candidates(
            copy.deepcopy(candidates),
            document_keys_by_id,
        )
        ranked_results = (
            dependencies["rerank"](
                query,
                candidates,
                len(candidates),
            )
            if candidates
            else []
        )
        filtered_ranked_results = dependencies["filter_evidence"](
            ranked_results
        )
        final_results = ranked_results[:top_k]
        filtered_results = dependencies["filter_evidence"](
            final_results
        )
    else:
        raise ValueError(f"不支持的检索模式: {mode}")

    return {
        "mode": mode,
        "elapsed_seconds": round(perf_counter() - started_at, 6),
        "candidate_count": len(candidates),
        "candidates": candidate_snapshot,
        "ranked_results": serialize_candidates(
            ranked_results,
            document_keys_by_id,
        ),
        "filtered_ranked_results": serialize_candidates(
            filtered_ranked_results,
            document_keys_by_id,
        ),
        "final_results": serialize_candidates(
            final_results,
            document_keys_by_id,
        ),
        "filtered_results": serialize_candidates(
            filtered_results,
            document_keys_by_id,
        ),
    }


def run_question(
    question: dict,
    modes: list[str],
    dependencies: dict[str, Callable],
    *,
    user_id: str,
    document_scopes: dict[str, dict],
    candidate_k: int,
    top_k: int,
    metric_k_values: list[int],
) -> dict:
    result = {
        "case_id": question["case_id"],
        "document_keys": case_document_keys(question),
        "scenario": question["scenario"],
        "category": question["category"],
        "query": question["query"],
        "answerable": question["answerable"],
        "query_variant": "original",
        "gold_pages": question.get("gold_pages") or [],
        "gold_evidence": question.get("gold_evidence") or [],
        "status": "success",
        "error": None,
        "modes": {},
    }
    scoped_keys = case_document_keys(question)
    document_ids = [
        document_scopes[key]["document_id"] for key in scoped_keys
    ]
    index_generations = {
        document_scopes[key]["document_id"]: document_scopes[key].get(
            "active_index_generation_id"
        )
        for key in scoped_keys
    }
    document_keys_by_id = {
        document_scopes[key]["document_id"]: key for key in scoped_keys
    }

    for mode in modes:
        try:
            mode_result = run_retrieval_mode(
                mode,
                dependencies,
                user_id=user_id,
                document_ids=document_ids,
                index_generations=index_generations,
                document_keys_by_id=document_keys_by_id,
                query=question["query"],
                candidate_k=candidate_k,
                top_k=top_k,
            )
            mode_result["page_metrics"] = calculate_page_metrics(
                result["gold_pages"],
                mode_result["final_results"],
                top_k,
                document_count=len(scoped_keys),
            )
            mode_result["metrics"] = {}
            stage_specs = (
                ("candidate", "candidates", candidate_k),
                ("final", "final_results", top_k),
                ("filtered", "filtered_results", top_k),
            )
            for stage_name, result_key, stage_top_k in stage_specs:
                stage_candidates = mode_result[result_key]
                mode_result["metrics"][stage_name] = {
                    "evidence": calculate_evidence_metrics(
                        result["gold_evidence"],
                        stage_candidates,
                        stage_top_k,
                    ),
                    "page": calculate_page_metrics(
                        result["gold_pages"],
                        stage_candidates,
                        stage_top_k,
                        document_count=len(scoped_keys),
                    ),
                }
            for stage_name, result_key in (
                ("candidate_by_k", "candidates"),
                ("ranked_by_k", "ranked_results"),
                ("filtered_ranked_by_k", "filtered_ranked_results"),
            ):
                stage_candidates = mode_result[result_key]
                mode_result["metrics"][stage_name] = {
                    str(metric_k): {
                        "evidence": calculate_evidence_metrics(
                            result["gold_evidence"],
                            stage_candidates,
                            metric_k,
                        ),
                        "page": calculate_page_metrics(
                            result["gold_pages"],
                            stage_candidates,
                            metric_k,
                            document_count=len(scoped_keys),
                        ),
                    }
                    for metric_k in metric_k_values
                }
            result["modes"][mode] = mode_result
        except Exception as error:
            result["status"] = "partial_error"
            result["modes"][mode] = {
                "mode": mode,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比 Dense、Dense+Rerank 和 Hybrid+Rerank 检索。"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        choices=sorted(SUPPORTED_MODES),
    )
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--metric-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_METRIC_K_VALUES),
        dest="metric_k_values",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--chunk-strategy", default="recursive")
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="只运行指定案例；可重复传入。",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get(
            "DOCRAG_EVAL_USERNAME",
            "docrag-retrieval-eval",
        ),
    )
    parser.add_argument(
        "--password",
        default=os.environ.get(
            "DOCRAG_EVAL_PASSWORD",
            "docrag-eval-password",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_run_config(
        args.modes,
        args.candidate_k,
        args.top_k,
        args.metric_k_values,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    documents_path = DATASET_DIR / "documents.json"
    cases_path = DATASET_DIR / "single_turn.json"
    run_id = build_run_id("retrieval")
    run_dir = RUNS_DIR / "retrieval" / run_id
    documents = load_json(documents_path)
    questions = load_json(cases_path)["cases"]
    questions = select_cases(questions, args.case_ids)
    documents = select_documents(documents, questions)

    with EvalApiClient(args.base_url) as client:
        print("1. 检查本地后端")
        client.check_ready()
        print("2. 登录或创建评估用户")
        user_id = client.authenticate(
            args.username,
            args.password,
        )
        print("3. 复用或建立评估文档索引")
        document_scopes = client.index_documents(
            user_id,
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunk_strategy=args.chunk_strategy,
        )

    print("4. 加载本地检索模型")
    dependencies = load_retrieval_dependencies()
    result_data = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "dataset_fingerprint": dataset_fingerprint(
            documents_path,
            cases_path,
        ),
        "config": {
            "base_url": args.base_url,
            "modes": args.modes,
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "metric_k_values": args.metric_k_values,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "chunk_strategy": args.chunk_strategy,
            "case_ids": args.case_ids,
            "query_variant": "original",
            "llm_called": False,
        },
        "user_id": user_id,
        "document_scopes": document_scopes,
        "results": [],
    }

    print(f"5. 执行 {len(questions)} 道检索题")
    for index, question in enumerate(questions, start=1):
        case_id = question["case_id"]
        print(f"[{index}/{len(questions)}] {case_id}")
        case_result = run_question(
            question,
            args.modes,
            dependencies,
            user_id=user_id,
            document_scopes=document_scopes,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            metric_k_values=args.metric_k_values,
        )
        result_data["results"].append(case_result)
        save_json(result_data, run_dir / "results.json")

    save_json(result_data["config"], run_dir / "config.json")
    save_json(
        summarize_retrieval_results(result_data),
        run_dir / "summary.json",
    )
    print(f"检索评估完成: {run_dir}")


if __name__ == "__main__":
    main()
