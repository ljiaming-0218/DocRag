"""Run retrieval-only comparisons without calling Query Rewrite or an LLM."""

import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "medrag" / "backend"
DATASET_DIR = Path(__file__).resolve().parent / "rag_dataset"
PDF_DIR = DATASET_DIR / "pdfs"
RESULTS_DIR = DATASET_DIR / "results" / "retrieval_runs"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODES = ("dense", "dense_rerank", "hybrid_rerank")
SUPPORTED_MODES = set(DEFAULT_MODES)

logger = logging.getLogger(__name__)


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validate_run_config(
    modes: list[str],
    candidate_k: int,
    top_k: int,
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


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    **kwargs,
) -> dict:
    kwargs.setdefault("timeout", 180)
    response = session.request(method, url, **kwargs)
    response.raise_for_status()
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as error:
        raise ValueError(f"接口没有返回 JSON: {response.text[:300]}") from error


def check_api(session: requests.Session, base_url: str) -> None:
    response = session.get(f"{base_url}/health", timeout=10)
    response.raise_for_status()


def authenticate_eval_user(
    session: requests.Session,
    base_url: str,
    username: str,
    password: str,
) -> str:
    credentials = {"username": username, "password": password}
    response = session.post(
        f"{base_url}/auth/login",
        json=credentials,
        timeout=30,
    )

    if response.status_code == 401:
        response = session.post(
            f"{base_url}/auth/register",
            json={**credentials, "default_user_type": "general"},
            timeout=30,
        )

    response.raise_for_status()
    payload = response.json()
    token = payload["access_token"]
    user_id = payload["user"]["user_id"]
    session.headers.update({"Authorization": f"Bearer {token}"})
    return user_id


def index_documents(
    session: requests.Session,
    base_url: str,
    user_id: str,
    documents: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str,
) -> dict[str, dict]:
    indexed_documents = {}
    for document in documents:
        pdf_path = PDF_DIR / document["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"评估 PDF 不存在: {pdf_path}")

        params = {
            "user_id": user_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "strategy": chunk_strategy,
        }
        with pdf_path.open("rb") as pdf_file:
            payload = request_json(
                session,
                "POST",
                f"{base_url}/pdf/index",
                params=params,
                files={
                    "file": (
                        document["filename"],
                        pdf_file,
                        "application/pdf",
                    )
                },
            )

        indexed_documents[document["document_key"]] = {
            "document_id": payload["document_id"],
            "active_index_generation_id": payload.get(
                "active_index_generation_id"
            ),
            "index_fingerprint": payload.get("index_fingerprint"),
        }
    return indexed_documents


def load_retrieval_dependencies() -> dict[str, Callable]:
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    from services.evidence_filter_service import filter_relevant_evidence
    from services.rerank_service import rerank_chunks
    from services.search_service import (
        fuse_dense_and_sparse_candidates,
        retrieve_dense_candidate_chunks,
    )

    return {
        "retrieve_dense": retrieve_dense_candidate_chunks,
        "fuse_hybrid": fuse_dense_and_sparse_candidates,
        "rerank": rerank_chunks,
        "filter_evidence": filter_relevant_evidence,
    }


def serialize_candidate(candidate: dict, rank: int) -> dict:
    metadata = candidate.get("元数据") or {}
    return {
        "rank": rank,
        "chunk_id": candidate.get("chunk_id"),
        "document_id": metadata.get("document_id"),
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


def serialize_candidates(candidates: list[dict]) -> list[dict]:
    return [
        serialize_candidate(candidate, rank)
        for rank, candidate in enumerate(candidates, start=1)
    ]


def calculate_page_metrics(
    gold_pages: list[int],
    candidates: list[dict],
    top_k: int,
) -> dict:
    if not gold_pages:
        return {
            "retrieval_evaluable": False,
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
    document_id: str,
    index_generation_id: str | None,
    query: str,
    candidate_k: int,
    top_k: int,
) -> dict:
    started_at = perf_counter()
    dense_candidates = dependencies["retrieve_dense"](
        user_id,
        document_id,
        query,
        candidate_k,
        index_generation_id,
    )

    if mode == "dense":
        candidates = dense_candidates
        candidate_snapshot = serialize_candidates(
            copy.deepcopy(candidates)
        )
        final_results = candidates[:top_k]
        filtered_results = final_results
    elif mode == "dense_rerank":
        candidates = dense_candidates
        candidate_snapshot = serialize_candidates(
            copy.deepcopy(candidates)
        )
        final_results = (
            dependencies["rerank"](
                query,
                candidates,
                min(top_k, len(candidates)),
            )
            if candidates
            else []
        )
        filtered_results = dependencies["filter_evidence"](
            final_results
        )
    elif mode == "hybrid_rerank":
        candidates = dependencies["fuse_hybrid"](
            user_id,
            [document_id],
            query,
            dense_candidates,
            candidate_k,
            {document_id: index_generation_id},
        )
        candidate_snapshot = serialize_candidates(
            copy.deepcopy(candidates)
        )
        final_results = (
            dependencies["rerank"](
                query,
                candidates,
                min(top_k, len(candidates)),
            )
            if candidates
            else []
        )
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
        "final_results": serialize_candidates(final_results),
        "filtered_results": serialize_candidates(filtered_results),
    }


def run_question(
    question: dict,
    modes: list[str],
    dependencies: dict[str, Callable],
    *,
    user_id: str,
    document_scope: dict,
    candidate_k: int,
    top_k: int,
) -> dict:
    result = {
        "question_id": question["question_id"],
        "document_key": question["document_key"],
        "question_type": question["question_type"],
        "query": question["question"],
        "query_variant": "original",
        "gold_source_pages": question.get("source_pages") or [],
        "gold_evidence": question.get("gold_evidence") or [],
        "status": "success",
        "error": None,
        "modes": {},
    }

    for mode in modes:
        try:
            mode_result = run_retrieval_mode(
                mode,
                dependencies,
                user_id=user_id,
                document_id=document_scope["document_id"],
                index_generation_id=document_scope.get(
                    "active_index_generation_id"
                ),
                query=question["question"],
                candidate_k=candidate_k,
                top_k=top_k,
            )
            mode_result["page_metrics"] = calculate_page_metrics(
                result["gold_source_pages"],
                mode_result["final_results"],
                top_k,
            )
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
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--chunk-strategy", default="recursive")
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
    validate_run_config(args.modes, args.candidate_k, args.top_k)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_id = build_run_id()
    run_dir = RESULTS_DIR / run_id
    documents = load_json(DATASET_DIR / "documents.json")
    questions = [
        question
        for question in load_json(DATASET_DIR / "questions.json")
        if question.get("question_type") != "follow_up"
    ]

    with requests.Session() as session:
        print("1. 检查本地后端")
        check_api(session, args.base_url)
        print("2. 登录或创建评估用户")
        user_id = authenticate_eval_user(
            session,
            args.base_url,
            args.username,
            args.password,
        )
        print("3. 复用或建立评估文档索引")
        document_scopes = index_documents(
            session,
            args.base_url,
            user_id,
            documents,
            args.chunk_size,
            args.chunk_overlap,
            args.chunk_strategy,
        )

    print("4. 加载本地检索模型")
    dependencies = load_retrieval_dependencies()
    result_data = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "modes": args.modes,
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "chunk_strategy": args.chunk_strategy,
            "query_variant": "original",
            "llm_called": False,
        },
        "user_id": user_id,
        "document_scopes": document_scopes,
        "results": [],
    }

    print(f"5. 执行 {len(questions)} 道检索题")
    for index, question in enumerate(questions, start=1):
        question_id = question["question_id"]
        print(f"[{index}/{len(questions)}] {question_id}")
        document_scope = document_scopes[question["document_key"]]
        case_result = run_question(
            question,
            args.modes,
            dependencies,
            user_id=user_id,
            document_scope=document_scope,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
        )
        result_data["results"].append(case_result)
        save_json(result_data, run_dir / "retrieval_results.json")

    save_json(result_data["config"], run_dir / "run_config.json")
    print(f"检索评估完成: {run_dir}")


if __name__ == "__main__":
    main()
