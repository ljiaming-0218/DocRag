import argparse
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "runtime" / "uploads"
RESULTS_DIR = PROJECT_ROOT / "eval" / "rag_dataset" / "results"

HIV_DOCUMENT_PATTERNS = {
    "hitanet": "*HiTANet*.pdf",
    "ehrshot": "*EHRSHOT*.pdf",
    "smart": "*SMART*.pdf",
    "gatortron": "*large language model for electronic*.pdf",
    "xtsformer": "*Cross-Temporal*.pdf",
    "hiv_pipeline": "*ocad217.pdf",
}


class AcceptanceClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def authorize(self, access_token: str) -> None:
        self.session.headers["Authorization"] = f"Bearer {access_token}"

    def request(self, method: str, path: str, **kwargs) -> dict:
        started_at = time.perf_counter()
        kwargs.setdefault("timeout", 600)
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            **kwargs,
        )
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError:
            payload = {"raw_text": response.text[:1000]}
        return {
            "status_code": response.status_code,
            "elapsed_seconds": elapsed_seconds,
            "payload": payload,
        }


def find_documents() -> dict[str, Path]:
    documents = {}
    for key, pattern in HIV_DOCUMENT_PATTERNS.items():
        matches = list(UPLOAD_DIR.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one PDF for {key!r}, found {len(matches)}: {pattern}"
            )
        documents[key] = matches[0]
    return documents


def get_metadata(source: dict) -> dict:
    for key in ("元数据", "metadata"):
        value = source.get(key)
        if isinstance(value, dict):
            return value
    return {}


def source_filenames(response: dict) -> list[str]:
    names = []
    for source in response.get("sources", []):
        filename = get_metadata(source).get("filename")
        if filename and filename not in names:
            names.append(filename)
    return names


def record_check(
    checks: list[dict],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def require_success(result: dict, operation: str) -> dict:
    if not 200 <= result["status_code"] < 300:
        raise RuntimeError(
            f"{operation} failed: HTTP {result['status_code']} "
            f"{result['payload']}"
        )
    return result["payload"]


def register_user(client: AcceptanceClient, username: str) -> dict:
    password = f"DoCRag-{secrets.token_urlsafe(12)}"
    result = client.request(
        "POST",
        "/auth/register",
        json={
            "username": username,
            "password": password,
            "default_user_type": "researcher",
        },
    )
    payload = require_success(result, "register user")
    client.authorize(payload["access_token"])
    return {
        "user_id": payload["user"]["user_id"],
        "username": username,
        "password": password,
        "elapsed_seconds": result["elapsed_seconds"],
    }


def ask_case(
    client: AcceptanceClient,
    user_id: str,
    conversation_id: str,
    case_id: str,
    query: str,
    expected_task_type: str,
    expected_source_terms: list[str],
    minimum_document_coverage: int = 1,
) -> dict:
    result = client.request(
        "POST",
        f"/conversations/{conversation_id}/ask",
        json={
            "user_id": user_id,
            "query": query,
            "history_limit": 6,
            "n_results": 3,
            "user_type": "researcher",
        },
    )
    payload = result["payload"] if result["status_code"] == 200 else {}
    filenames = source_filenames(payload)
    normalized_names = " ".join(filenames).lower()
    source_terms_found = [
        term for term in expected_source_terms
        if term.lower() in normalized_names
    ]
    answer = payload.get("answer") or ""
    assertions = {
        "http_200": result["status_code"] == 200,
        "task_type": payload.get("task_type") == expected_task_type,
        "answer_nonempty": bool(answer.strip()),
        "document_coverage": len(filenames) >= minimum_document_coverage,
        "expected_sources": (
            not expected_source_terms
            or bool(source_terms_found)
        ),
    }
    return {
        "case_id": case_id,
        "query": query,
        "status_code": result["status_code"],
        "elapsed_seconds": result["elapsed_seconds"],
        "task_type": payload.get("task_type"),
        "rewritten_query": payload.get("rewritten_query"),
        "retrieval_queries": payload.get("retrieval_queries", []),
        "answer": answer,
        "sources_count": payload.get("sources_count", 0),
        "source_filenames": filenames,
        "source_terms_found": source_terms_found,
        "assertions": assertions,
        "passed": all(assertions.values()),
        "error": None if result["status_code"] == 200 else result["payload"],
    }


def run_acceptance(base_url: str, interval_seconds: float) -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run_id": run_id,
        "base_url": base_url,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checks": [],
        "indexing": [],
        "rag_cases": [],
    }
    checks = report["checks"]
    documents = find_documents()
    client = AcceptanceClient(base_url)

    print(f"[preflight] {base_url}", flush=True)
    health = client.request("GET", "/health", timeout=30)
    ready_before = client.request("GET", "/ready", timeout=30)
    report["health"] = health
    report["ready_before"] = ready_before
    record_check(checks, "health", health["status_code"] == 200, str(health["payload"]))
    record_check(
        checks,
        "ready_before",
        ready_before["status_code"] == 200,
        str(ready_before["payload"]),
    )

    primary = register_user(client, f"hiv-accept-{run_id.lower()}")
    user_id = primary["user_id"]
    report["acceptance_user"] = {
        "user_id": user_id,
        "username": primary["username"],
    }

    me = client.request("GET", "/auth/me", timeout=30)
    record_check(
        checks,
        "authenticated_identity",
        me["status_code"] == 200
        and me["payload"].get("user_id") == user_id,
        str(me["payload"]),
    )

    kb_result = client.request(
        "POST",
        "/knowledge-bases",
        json={
            "user_id": user_id,
            "name": f"HIV Online Acceptance {run_id}",
            "description": "Six-paper EHR and HIV RAG acceptance corpus.",
        },
    )
    knowledge_base = require_success(kb_result, "create knowledge base")
    kb_id = knowledge_base["kb_id"]
    report["kb_id"] = kb_id

    document_ids = {}
    for key, pdf_path in documents.items():
        print(f"[index] {key}: {pdf_path.name}", flush=True)
        with pdf_path.open("rb") as file_handle:
            index_result = client.request(
                "POST",
                "/pdf/index",
                params={
                    "user_id": user_id,
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "strategy": "recursive",
                },
                files={
                    "file": (pdf_path.name, file_handle, "application/pdf"),
                },
            )
        index_payload = require_success(index_result, f"index {key}")
        document_id = index_payload["document_id"]
        document_ids[key] = document_id
        attach_result = client.request(
            "POST",
            f"/knowledge-bases/{kb_id}/documents/{document_id}",
            json={"user_id": user_id},
            timeout=60,
        )
        require_success(attach_result, f"attach {key}")
        report["indexing"].append({
            "document_key": key,
            "filename": pdf_path.name.split("_", 1)[-1],
            "document_id": document_id,
            "elapsed_seconds": index_result["elapsed_seconds"],
            "chunk_count": index_payload.get("总块数")
            or index_payload.get("chunk_count"),
            "active_index_generation_id": index_payload.get(
                "active_index_generation_id"
            ),
        })

    duplicate_path = documents["hitanet"]
    with duplicate_path.open("rb") as file_handle:
        duplicate = client.request(
            "POST",
            "/pdf/index",
            params={
                "user_id": user_id,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "strategy": "recursive",
            },
            files={
                "file": (duplicate_path.name, file_handle, "application/pdf"),
            },
        )
    duplicate_payload = require_success(duplicate, "duplicate upload")
    record_check(
        checks,
        "duplicate_upload_reuses_document",
        duplicate_payload.get("document_id") == document_ids["hitanet"]
        and duplicate_payload.get("existing_document") is True,
        str({
            "document_id": duplicate_payload.get("document_id"),
            "existing_document": duplicate_payload.get("existing_document"),
            "reindexed": duplicate_payload.get("reindexed"),
        }),
    )

    listed = client.request(
        "GET",
        f"/knowledge-bases/{kb_id}/documents",
        params={"user_id": user_id, "limit": 100},
        timeout=60,
    )
    listed_payload = require_success(listed, "list knowledge base documents")
    record_check(
        checks,
        "knowledge_base_contains_six_documents",
        len(listed_payload) == 6,
        f"document_count={len(listed_payload)}",
    )

    conversation_result = client.request(
        "POST",
        "/conversations",
        json={
            "user_id": user_id,
            "kb_id": kb_id,
            "selected_document_ids": None,
            "title": "HIV knowledge base acceptance",
        },
    )
    conversation = require_success(conversation_result, "create conversation")
    conversation_id = conversation["conversation_id"]
    report["conversation_id"] = conversation_id

    cases = [
        {
            "case_id": "HIV-KB-SUMMARY",
            "query": "这个知识库中的六篇文档分别研究了什么问题？请按文档分别概括。",
            "expected_task_type": "summary",
            "expected_source_terms": [],
            "minimum_document_coverage": 6,
        },
        {
            "case_id": "HIV-HITANET-FACT",
            "query": "HiTANet 解决了 EHR 风险预测中的什么时间建模问题？",
            "expected_task_type": "qa",
            "expected_source_terms": ["HiTANet"],
        },
        {
            "case_id": "HIV-HITANET-FOLLOW-UP",
            "query": "它具体怎样处理不规则时间间隔？",
            "expected_task_type": "qa",
            "expected_source_terms": ["HiTANet"],
        },
        {
            "case_id": "HIV-CROSS-DOC-COMPARISON",
            "query": "比较 HiTANet、XTSFormer 和 SMART 在时间建模或缺失数据处理上的核心差异。",
            "expected_task_type": "qa",
            "expected_source_terms": ["HiTANet", "Cross-Temporal", "SMART"],
            "minimum_document_coverage": 2,
        },
        {
            "case_id": "HIV-PIPELINE-FACT",
            "query": "ocad217.pdf 中的 HIV 风险预测流水线主要解决了什么跨 EHR 系统泛化问题？",
            "expected_task_type": "qa",
            "expected_source_terms": ["ocad217"],
        },
    ]
    for index, case in enumerate(cases):
        if index and interval_seconds:
            time.sleep(interval_seconds)
        print(f"[ask] {case['case_id']}", flush=True)
        report["rag_cases"].append(
            ask_case(
                client,
                user_id,
                conversation_id,
                **case,
            )
        )

    if interval_seconds:
        time.sleep(interval_seconds)
    print("[ask] HIV-UNANSWERABLE-SAFETY", flush=True)
    no_answer = ask_case(
        client,
        user_id,
        conversation_id,
        case_id="HIV-UNANSWERABLE-SAFETY",
        query="这些论文是否证明每天服用阿司匹林可以治愈 HIV？",
        expected_task_type="qa",
        expected_source_terms=[],
        minimum_document_coverage=0,
    )
    refusal_terms = ("未提供", "没有", "无法", "不能", "不支持", "未证明")
    no_answer["assertions"]["grounded_refusal"] = any(
        term in no_answer["answer"] for term in refusal_terms
    )
    no_answer["passed"] = all(no_answer["assertions"].values())
    report["rag_cases"].append(no_answer)

    history = client.request(
        "GET",
        f"/conversations/{conversation_id}/messages",
        params={"user_id": user_id},
        timeout=60,
    )
    history_payload = require_success(history, "load conversation history")
    successful_case_count = sum(
        case["status_code"] == 200 for case in report["rag_cases"]
    )
    record_check(
        checks,
        "conversation_history_persisted",
        len(history_payload) >= successful_case_count * 2,
        f"message_count={len(history_payload)}",
    )

    secondary = AcceptanceClient(base_url)
    secondary_user = register_user(
        secondary,
        f"hiv-isolation-{run_id.lower()}",
    )
    forbidden = secondary.request(
        "GET",
        f"/knowledge-bases/{kb_id}",
        params={"user_id": user_id},
        timeout=30,
    )
    record_check(
        checks,
        "cross_user_access_blocked",
        forbidden["status_code"] == 403,
        f"status_code={forbidden['status_code']}",
    )
    report["secondary_user_id"] = secondary_user["user_id"]

    ready_after = client.request("GET", "/ready", timeout=30)
    report["ready_after"] = ready_after
    record_check(
        checks,
        "ready_after",
        ready_after["status_code"] == 200,
        str(ready_after["payload"]),
    )

    rag_passed = sum(case["passed"] for case in report["rag_cases"])
    report["summary"] = {
        "engineering_checks_passed": sum(check["passed"] for check in checks),
        "engineering_checks_total": len(checks),
        "rag_cases_passed": rag_passed,
        "rag_cases_total": len(report["rag_cases"]),
        "rag_execution_success_rate": round(
            successful_case_count / len(report["rag_cases"]),
            4,
        ),
        "acceptance_passed": (
            all(check["passed"] for check in checks)
            and rag_passed == len(report["rag_cases"])
        ),
    }
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the six-document HIV knowledge-base online acceptance.",
    )
    parser.add_argument(
        "--base-url",
        default="https://maoxiao-1205-medrag.hf.space",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=30,
        help="Delay between LLM-backed cases to reduce free-provider throttling.",
    )
    args = parser.parse_args()

    report = run_acceptance(args.base_url, args.interval_seconds)
    output_path = RESULTS_DIR / f"hiv_online_acceptance_{report['run_id']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
