"""Run end-to-end generation evaluation through the public DocRAG API."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from eval.core import (
    DATASET_DIR,
    RUNS_DIR,
    EvalApiClient,
    build_run_id,
    dataset_fingerprint,
    load_json,
    save_json,
    utc_now_iso,
)
from eval.metrics.generation import (
    build_manual_review_template,
    summarize_generation_results,
)
from eval.metrics.retrieval import calculate_evidence_metrics


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_USERNAME = "docrag-generation-eval"
DEFAULT_PASSWORD = "docrag-eval-password"
DEFAULT_REQUEST_INTERVAL_SECONDS = 30.0
SUPPORTED_SUITES = ("single_turn", "follow_up", "all")

logger = logging.getLogger(__name__)


def build_public_config(args: argparse.Namespace) -> dict:
    return {
        key: value
        for key, value in vars(args).items()
        if key != "password"
    }


def validate_generation_config(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
    if not 1 <= args.history_limit <= 20:
        raise ValueError("history_limit 必须在 1 到 20 之间")
    if not 1 <= args.n_results <= 10:
        raise ValueError("n_results 必须在 1 到 10 之间")
    if args.timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    if args.request_interval < 0:
        raise ValueError("request_interval 不能小于 0")


def load_cases(suite: str) -> tuple[list[dict], list[Path]]:
    paths = []
    cases = []
    for scenario, filename in (
        ("single_turn", "single_turn.json"),
        ("follow_up", "follow_up.json"),
    ):
        if suite not in (scenario, "all"):
            continue
        path = DATASET_DIR / filename
        paths.append(path)
        cases.extend(load_json(path)["cases"])
    return cases, paths


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
    required_keys = {case["document_key"] for case in cases}
    selected = [
        document
        for document in documents
        if document.get("document_key") in required_keys
    ]
    found_keys = {document.get("document_key") for document in selected}
    missing = required_keys - found_keys
    if missing:
        raise ValueError(
            "评估案例引用了未定义文档: " + ", ".join(sorted(missing))
        )
    return selected


def extract_source_pages(sources: list[dict]) -> list[int]:
    pages = []
    for source in sources:
        page = (source.get("元数据") or {}).get("page_number")
        if page is not None and page not in pages:
            pages.append(page)
    return pages


def calculate_citation_evidence_metrics(
    gold_evidence: list[dict],
    sources: list[dict],
) -> dict:
    candidates = [
        {
            "page_number": (source.get("元数据") or {}).get(
                "page_number"
            ),
            "text": source.get("文本块", ""),
        }
        for source in sources
    ]
    top_k = max(1, len(candidates))
    metrics = calculate_evidence_metrics(
        gold_evidence,
        candidates,
        top_k,
    )
    return {
        "evidence_evaluable": metrics["evidence_evaluable"],
        "evidence_hit_rate": metrics[f"evidence_hit_rate_at_{top_k}"],
        "evidence_recall": metrics[f"evidence_recall_at_{top_k}"],
        "matched_evidence_ids": metrics["matched_evidence_ids"],
    }


def build_success_result(
    case: dict,
    *,
    document_id: str,
    conversation_id: str,
    response: dict,
    elapsed_seconds: float,
    generated_history: list[dict] | None = None,
) -> dict:
    sources = response.get("sources") or []
    result = {
        "case_id": case["case_id"],
        "document_key": case["document_key"],
        "document_id": document_id,
        "conversation_id": conversation_id,
        "scenario": case["scenario"],
        "category": case["category"],
        "query": case["query"],
        "expected_rewritten_query": case.get("expected_rewritten_query"),
        "expected_task": case["expected_task"],
        "answerable": case["answerable"],
        "reference_answer": case.get("reference_answer"),
        "answer_points": case.get("answer_points") or [],
        "gold_pages": case.get("gold_pages") or [],
        "gold_evidence": case.get("gold_evidence") or [],
        "status": "success",
        "error": None,
        "elapsed_seconds": elapsed_seconds,
        "actual": {
            "rewritten_query": response.get("rewritten_query"),
            "retrieval_queries": response.get("retrieval_queries") or [],
            "answer": response.get("answer"),
            "sources": sources,
            "source_pages": extract_source_pages(sources),
            "task": response.get("task_type"),
        },
        "citation_evidence": calculate_citation_evidence_metrics(
            case.get("gold_evidence") or [],
            sources,
        ),
        "manual_review": build_manual_review_template(),
    }
    if generated_history is not None:
        result["expected_history"] = case.get("history") or []
        result["generated_history"] = generated_history
    return result


def build_error_result(case: dict, error: Exception) -> dict:
    response = getattr(error, "response", None)
    return {
        "case_id": case["case_id"],
        "document_key": case["document_key"],
        "scenario": case["scenario"],
        "category": case["category"],
        "query": case["query"],
        "expected_task": case.get("expected_task"),
        "answerable": case.get("answerable"),
        "status": "error",
        "error_type": type(error).__name__,
        "status_code": getattr(response, "status_code", None),
        "error": str(error),
        "actual": None,
        "manual_review": build_manual_review_template(),
    }


def run_case(
    client: EvalApiClient,
    user_id: str,
    document_scopes: dict[str, dict],
    case: dict,
    *,
    history_limit: int,
    n_results: int,
    user_type: str,
    request_interval: float,
) -> dict:
    document_id = document_scopes[case["document_key"]]["document_id"]
    conversation_id = client.create_conversation(
        user_id,
        document_id,
        f"Eval {case['case_id']}",
    )
    generated_history = None
    if case["scenario"] == "follow_up":
        generated_history = []
        for message in case.get("history") or []:
            if message.get("role") != "user":
                continue
            response, elapsed = client.ask(
                user_id,
                conversation_id,
                message["content"],
                history_limit=history_limit,
                n_results=n_results,
                user_type=user_type,
            )
            generated_history.append({
                "query": message["content"],
                "answer": response.get("answer"),
                "rewritten_query": response.get("rewritten_query"),
                "sources": response.get("sources") or [],
                "elapsed_seconds": elapsed,
            })
            if request_interval > 0:
                time.sleep(request_interval)

    response, elapsed = client.ask(
        user_id,
        conversation_id,
        case["query"],
        history_limit=history_limit,
        n_results=n_results,
        user_type=user_type,
    )
    return build_success_result(
        case,
        document_id=document_id,
        conversation_id=conversation_id,
        response=response,
        elapsed_seconds=elapsed,
        generated_history=generated_history,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 DocRAG 生成评估")
    parser.add_argument(
        "--suite",
        choices=SUPPORTED_SUITES,
        default="single_turn",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="只运行指定案例，可重复传入",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--chunk-strategy", default="recursive")
    parser.add_argument("--history-limit", type=int, default=6)
    parser.add_argument("--n-results", type=int, default=3)
    parser.add_argument("--user-type", default="general")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL_SECONDS,
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    validate_generation_config(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    documents_path = DATASET_DIR / "documents.json"
    documents = load_json(documents_path)
    cases, case_paths = load_cases(args.suite)
    cases = select_cases(cases, args.case_ids)
    documents = select_documents(documents, cases)
    run_id = build_run_id(f"generation-{args.suite}")
    run_dir = RUNS_DIR / "generation" / run_id
    result_data = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "dataset_fingerprint": dataset_fingerprint(
            documents_path,
            *case_paths,
        ),
        "config": build_public_config(args),
        "user_id": None,
        "document_scopes": {},
        "results": [],
    }

    with EvalApiClient(args.base_url, timeout=args.timeout) as client:
        print("1. 检查后端服务")
        client.check_ready()
        print("2. 登录或创建评估用户")
        user_id = client.authenticate(args.username, args.password)
        result_data["user_id"] = user_id
        print("3. 复用或建立评估文档索引")
        scopes = client.index_documents(
            user_id,
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunk_strategy=args.chunk_strategy,
        )
        result_data["document_scopes"] = scopes

        print(f"4. 执行 {len(cases)} 道 {args.suite} 生成题")
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['case_id']}")
            try:
                case_result = run_case(
                    client,
                    user_id,
                    scopes,
                    case,
                    history_limit=args.history_limit,
                    n_results=args.n_results,
                    user_type=args.user_type,
                    request_interval=args.request_interval,
                )
            except Exception as error:
                logger.exception(
                    "generation_case_failed case_id=%s",
                    case["case_id"],
                )
                case_result = build_error_result(case, error)
            result_data["results"].append(case_result)
            save_json(result_data, run_dir / "results.json")
            if index < len(cases) and args.request_interval > 0:
                time.sleep(args.request_interval)

    save_json(
        summarize_generation_results(result_data),
        run_dir / "summary.json",
    )
    save_json(result_data["config"], run_dir / "config.json")
    print(f"生成评估完成: {run_dir}")


if __name__ == "__main__":
    main()
