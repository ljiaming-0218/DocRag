"""Check the deployed release with one public PDF and an isolated test user."""

from __future__ import annotations

import argparse
import secrets

from eval.core import (
    DATASET_DIR,
    RUNS_DIR,
    EvalApiClient,
    build_run_id,
    load_json,
    save_json,
    utc_now_iso,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_acceptance(
    base_url: str,
    expected_build_commit: str,
    *,
    preflight_only: bool = False,
) -> dict:
    report = {
        "run_id": build_run_id("release-acceptance"),
        "started_at": utc_now_iso(),
        "base_url": base_url,
        "expected_build_commit": expected_build_commit,
        "preflight_only": preflight_only,
        "checks": {},
        "passed": False,
    }
    stage = "preflight"
    try:
        with EvalApiClient(base_url, timeout=300.0) as client:
            health = client.request_json("GET", "/health", timeout=30.0)
            require(health.get("status") == "ok", "health 状态异常")
            report["checks"]["health"] = health
            require(
                health.get("build_commit") == expected_build_commit,
                "线上 build_commit 与预期提交不一致: "
                f"actual={health.get('build_commit')} "
                f"expected={expected_build_commit}",
            )
            ready = client.request_json("GET", "/ready", timeout=30.0)
            require(ready.get("status") == "ready", "ready 状态异常")
            report["checks"]["ready"] = ready

            if preflight_only:
                report["passed"] = True
                return report

            stage = "authentication"
            username = f"release-{secrets.token_hex(6)}"
            password = secrets.token_urlsafe(24)
            user_id = client.authenticate(username, password)
            login = client.request_json(
                "POST", "/auth/login",
                json={"username": username, "password": password},
                timeout=30.0,
            )
            require(login.get("user", {}).get("user_id") == user_id,
                    "登录用户与注册用户不一致")
            me = client.request_json("GET", "/auth/me", timeout=30.0)
            require(me.get("user_id") == user_id, "认证身份校验失败")
            report["checks"]["authentication"] = True

            stage = "ingestion"
            document = next(
                item for item in load_json(DATASET_DIR / "documents.json")
                if item["document_key"] == "rag"
            )
            scopes = client.index_documents(
                user_id, [document], chunk_size=500,
                chunk_overlap=50, chunk_strategy="recursive",
            )
            scope = scopes["rag"]
            report["checks"]["ingestion"] = {
                "document_id": scope["document_id"],
                "job_id": scope["job_id"],
                "active_index_generation_id": scope[
                    "active_index_generation_id"
                ],
                "elapsed_seconds": scope["index_elapsed_seconds"],
            }

            stage = "conversation"
            conversation_id = client.create_conversation(
                user_id, f"Release acceptance {report['run_id']}",
                document_id=scope["document_id"],
            )
            report["checks"]["conversation_id"] = conversation_id

            stage = "answer"
            answer, elapsed = client.ask(
                user_id, conversation_id,
                "RAG 模型中的参数化记忆和非参数化记忆分别是什么？",
                history_limit=6, n_results=3, user_type="general",
            )
            sources = answer.get("sources") or []
            require(bool(str(answer.get("answer") or "").strip()), "回答为空")
            require(bool(sources), "回答没有引用来源")
            require(
                all(
                    (source.get("元数据") or {}).get("document_id")
                    == scope["document_id"]
                    for source in sources
                ),
                "引用来源不属于已上传文档",
            )
            report["checks"]["answer"] = {
                "elapsed_seconds": elapsed,
                "sources_count": len(sources),
                "source_pages": sorted({
                    (source.get("元数据") or {}).get("page_number")
                    for source in sources
                    if (source.get("元数据") or {}).get("page_number")
                    is not None
                }),
            }

            stage = "history"
            messages = client.request_json(
                "GET", f"/conversations/{conversation_id}/messages",
                params={"user_id": user_id}, timeout=30.0,
            )
            require(isinstance(messages, list) and len(messages) >= 2,
                    "会话消息没有持久化")
            report["checks"]["history_message_count"] = len(messages)
            report["passed"] = True
    except Exception as error:
        report["failure"] = {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error)[:1000],
        }
    finally:
        report["finished_at"] = utc_now_iso()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="DocRAG release acceptance")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-build-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    report = run_acceptance(
        args.base_url, args.expected_build_commit,
        preflight_only=args.preflight_only,
    )
    destination = (
        RUNS_DIR / "release_acceptance" / f"{report['run_id']}.json"
    )
    save_json(report, destination)
    print(f"passed={report['passed']} report={destination}")
    if not report["passed"]:
        print(report["failure"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
