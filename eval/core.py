"""Shared paths, JSON helpers, and HTTP setup for evaluation runners."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


logger = logging.getLogger(__name__)

EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parent
BACKEND_ROOT = PROJECT_ROOT / "medrag" / "backend"
DATASET_DIR = EVAL_ROOT / "dataset"
PDF_DIR = DATASET_DIR / "pdfs"
RUNS_DIR = EVAL_ROOT / "runs"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_run_id(kind: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{kind}-{timestamp}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_fingerprint(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class EvalApiClient:
    """Small API client shared by retrieval and generation evaluations."""

    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "EvalApiClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def request_json(self, method: str, path: str, **kwargs: Any) -> dict:
        timeout = kwargs.pop("timeout", self.timeout)
        url = f"{self.base_url}{path}"
        started_at = time.perf_counter()
        try:
            response = self.session.request(
                method,
                url,
                timeout=timeout,
                **kwargs,
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception(
                "eval_http_request_failed method=%s path=%s elapsed=%.2f",
                method,
                path,
                time.perf_counter() - started_at,
            )
            raise
        logger.info(
            "eval_http_request_succeeded method=%s path=%s status=%s "
            "elapsed=%.2f",
            method,
            path,
            response.status_code,
            time.perf_counter() - started_at,
        )
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as error:
            raise ValueError(
                f"接口没有返回 JSON: {response.text[:500]}"
            ) from error

    def check_ready(self) -> None:
        for path in ("/health", "/ready"):
            self.request_json("GET", path, timeout=10.0)

    def authenticate(self, username: str, password: str) -> str:
        credentials = {"username": username, "password": password}
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json=credentials,
            timeout=30.0,
        )
        if response.status_code == 401:
            response = self.session.post(
                f"{self.base_url}/auth/register",
                json={**credentials, "default_user_type": "general"},
                timeout=30.0,
            )
        response.raise_for_status()
        payload = response.json()
        self.session.headers.update({
            "Authorization": f"Bearer {payload['access_token']}"
        })
        return payload["user"]["user_id"]

    def index_documents(
        self,
        user_id: str,
        documents: list[dict],
        *,
        chunk_size: int,
        chunk_overlap: int,
        chunk_strategy: str,
    ) -> dict[str, dict]:
        scopes = {}
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
            started_at = time.perf_counter()
            with pdf_path.open("rb") as pdf_file:
                payload = self.request_json(
                    "POST",
                    "/pdf/index",
                    params=params,
                    files={
                        "file": (
                            document["filename"],
                            pdf_file,
                            "application/pdf",
                        )
                    },
                )
            scopes[document["document_key"]] = {
                "document_id": payload["document_id"],
                "active_index_generation_id": payload.get(
                    "active_index_generation_id"
                ),
                "index_fingerprint": payload.get("index_fingerprint"),
                "index_elapsed_seconds": round(
                    time.perf_counter() - started_at,
                    6,
                ),
            }
        return scopes

    def create_conversation(
        self,
        user_id: str,
        document_id: str,
        title: str,
    ) -> str:
        payload = self.request_json(
            "POST",
            "/conversations",
            json={
                "user_id": user_id,
                "document_id": document_id,
                "title": title,
            },
        )
        return payload["conversation_id"]

    def ask(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
        *,
        history_limit: int,
        n_results: int,
        user_type: str,
    ) -> tuple[dict, float]:
        started_at = time.perf_counter()
        payload = self.request_json(
            "POST",
            f"/conversations/{conversation_id}/ask",
            json={
                "user_id": user_id,
                "query": query,
                "history_limit": history_limit,
                "n_results": n_results,
                "user_type": user_type,
            },
        )
        return payload, round(time.perf_counter() - started_at, 6)
