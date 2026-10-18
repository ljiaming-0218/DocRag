"""Nine-run real-services A/B/C pilot. No production code changes or pytest."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from time import perf_counter, sleep
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "medrag" / "backend"))
import config
from stores import database, conversation_store, message_store
from services import chat_service, conversation_service, llm_service, search_service
from harness.research_task import create_research_task
from harness.research_tools import ensure_task_scope, execute_action
from harness.research_progress import build_evidence_context
from harness.research_worker import run_once
from research_live_replay import ExperimentStore

CASES = [
    {"id": "fact", "query": "LoRA 的核心方法是什么？它如何减少可训练参数？请根据原文回答。",
     "topic": "method trainable parameters", "expected": "冻结预训练权重、低秩适配与减少可训练参数；不将 LoRA 等同串联 adapter。",
     "order": ["A", "B", "C"]},
    {"id": "comparison", "query": "比较这三篇论文分别解决的问题、核心方法，以及实验结果为什么不能直接比较。逐项引用原文，证据不足的部分明确说明。",
     "topic": "research problem proposed method evaluation tasks limitations",
     "expected": "逐篇说明问题和方法，解释任务或评估条件差异；不凭空选出统一性能冠军。",
     "order": ["B", "C", "A"]},
    {"id": "no_answer", "query": "请给出这三篇论文在同一个模型、同一个数据集和同一个评价指标下的统一对比数值。若参考文档没有这组对比，请明确说明证据不足，不要编造。",
     "topic": "evaluation benchmark dataset metric comparison results",
     "expected": "没有直接证据时说明不足，不编造统一比较数值，也不把未检索到等同于论文中绝不存在。",
     "order": ["C", "A", "B"]},
]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


async def pilot(source_id, output, resume=False, interval=30):
    output.mkdir(parents=True, exist_ok=True)
    await database.connect_database()
    temporary_db = "docrag_h4_" + uuid4().hex[:16]
    store = ExperimentStore(temporary_db)
    temp = database.get_database().client[temporary_db]
    original_request = llm_service._request_completion
    original_single = search_service.retrieve_candidate_chunks
    original_multi = search_service.retrieve_candidate_chunks_for_documents
    current = None
    next_request_at = perf_counter() + (60 if resume else 0)

    def request(prompt, operation, **kwargs):
        nonlocal next_request_at
        if len(current["requests"]) >= 8:
            raise RuntimeError("PILOT_MODEL_BUDGET_EXHAUSTED")
        wait = max(0, next_request_at - perf_counter())
        if wait:
            logging.info("PILOT_THROTTLE wait_seconds=%.1f", wait)
            sleep(wait)
            current["throttle_wait_seconds"] += wait
        next_request_at = perf_counter() + interval
        remaining = 5400 - (perf_counter() - current["started"])
        if remaining <= 0:
            raise RuntimeError("PILOT_WALL_BUDGET_EXHAUSTED")
        kwargs["timeout_seconds"] = min(kwargs.get("timeout_seconds") or 600, 600, remaining)
        kwargs["max_retries"] = 0
        role = llm_service.resolve_llm_role(operation)
        record = {"operation": operation, "role": role, "model": llm_service.get_llm_config(role).model,
                  "timeout_seconds": kwargs["timeout_seconds"], "prompt": prompt}
        current["requests"].append(record)
        started = perf_counter()
        try:
            response = original_request(prompt, operation, **kwargs)
            record["usage"] = response.usage.model_dump() if response.usage else None
            message = response.choices[0].message
            record["content"] = message.content
            record["tool_calls"] = [call.model_dump() for call in message.tool_calls or []]
            return response
        except Exception as error:
            record["error"] = getattr(error, "code", type(error).__name__)
            if record["error"] == "LLM_RATE_LIMITED":
                next_request_at = perf_counter() + 60
            raise
        finally:
            record["elapsed_seconds"] = round(perf_counter() - started, 3)

    def retrieve(function, *args, **kwargs):
        row = {"query": args[2], "documents": args[1], "candidate_k": args[3]}
        current["retrievals"].append(row)
        started = perf_counter()
        result = function(*args, **kwargs)
        row["elapsed_seconds"] = round(perf_counter() - started, 3)
        row["candidate_count"] = len(result)
        return result

    rows = json.loads((output / "summary.json").read_text(encoding="utf-8")) if resume else []
    corrected_attempts = []
    if resume:
        for row in list(rows):
            path = output / (row["case"] + "_" + row["arm"] + ".json")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("error_detail") == "PILOT_RETRIEVAL_BUDGET_EXHAUSTED":
                archive = path.with_name(path.stem + "_capped_attempt.json")
                path.rename(archive)
                corrected_attempts.append(str(archive))
                rows.remove(row)
    try:
        source = await database.get_database()["research_tasks"].find_one({"task_id": source_id})
        if source is None:
            raise ValueError("source task not found")
        manifest = {"source_task_id": source_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "sample_size": "3 tasks x 3 arms x 1 repeat", "cases": CASES,
            "arms": {"A": "Existing ask_conversation including router/rewrite/native retrieval",
                     "B": "Fixed: original query retrieval, one topic query per document, one generation",
                     "C": "Current fresh Harness task, no inherited evidence"},
            "budgets": {"model_calls": 8, "atomic_retrieval_calls": "observed; native fixed fanout preserved", "model_call_seconds": 600,
                        "wall_seconds": 5400, "harness_tool_calls": 12},
            "configuration": {"answer_provider": config.ANSWER_LLM_PROVIDER, "answer_model": config.ANSWER_LLM_MODEL,
                "rewrite_provider": config.REWRITE_LLM_PROVIDER, "rewrite_model": config.REWRITE_LLM_MODEL,
                "retrieval_mode": config.RETRIEVAL_MODE, "threshold": config.EVIDENCE_RERANK_MIN_SCORE},
            "document_ids": source["document_ids"], "index_generations": source["index_generations"],
            "limitations": "Exploratory bundled-pipeline comparison, not a causal ablation or accuracy benchmark. Native A retrieval/context policies are preserved. Backend execution, not UI authentication."}
        if resume:
            previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            if previous["index_generations"] != manifest["index_generations"] or previous["cases"] != CASES:
                raise ValueError("resume configuration changed")
            manifest = previous
            if corrected_attempts:
                initial = output / "manifest_initial_capped.json"
                if not initial.exists():
                    write_json(initial, previous)
                manifest["evaluation_limit_correction"] = {"preserved_attempts": corrected_attempts,
                    "reason": "The harness tool cap was incorrectly applied to native A atomic retrieval fanout; remove the pilot-only cap and rerun those interrupted baselines."}
            manifest["budgets"]["atomic_retrieval_calls"] = "observed; native fixed fanout preserved"
            manifest["rate_limit_continuation"] = {"request_interval_seconds": interval,
                "initial_cooldown_seconds": 60, "note": "Paused after a 429; completed and failed rows preserved. Unfinished trial restarted; wall time includes throttling."}
        manifest.setdefault("temporary_databases", []).append(temporary_db)
        write_json(output / "manifest.json", manifest)
        await store.create_indexes()
        with patch.object(conversation_store, "get_database", return_value=temp), patch.object(
                message_store, "get_database", return_value=temp), patch.object(llm_service, "_request_completion", request), patch.object(
                search_service, "retrieve_candidate_chunks", lambda *a, **k: retrieve(original_single, *a, **k)), patch.object(
                search_service, "retrieve_candidate_chunks_for_documents", lambda *a, **k: retrieve(original_multi, *a, **k)):
            for case in CASES:
                for arm in case["order"]:
                    if any(row["case"] == case["id"] and row["arm"] == arm for row in rows):
                        continue
                    task = await create_research_task(source["owner_id"], source["kb_id"], case["query"], source["document_ids"])
                    if task.index_generations != source["index_generations"]:
                        raise ValueError("index snapshot changed")
                    await ensure_task_scope(task)
                    current = {"case": case["id"], "arm": arm, "query": case["query"],
                               "started": perf_counter(), "requests": [], "retrievals": [], "status": "running",
                               "throttle_wait_seconds": 0}
                    logging.info("PILOT_START case=%s arm=%s", case["id"], arm)
                    try:
                        async with asyncio.timeout(5400):
                            if arm == "A":
                                conv = await conversation_service.create_conversation(source["owner_id"],
                                    kb_id=source["kb_id"], selected_document_ids=source["document_ids"], title="H4 pilot")
                                current["result"] = await chat_service.ask_conversation(source["owner_id"],
                                    conv["conversation_id"], case["query"], n_results=3, user_type="general")
                                current["answer"] = current["result"]["answer"]
                            elif arm == "B":
                                for docs, query in [(list(task.document_ids), case["query"])] + [
                                        ([doc], task.document_filenames[doc].replace("_", " ") + " " + case["topic"])
                                        for doc in task.document_ids]:
                                    await execute_action(task, {"name": "search_documents", "args": {
                                        "query": query, "document_ids": docs, "n_results": 3}}, model_proposed=False)
                                prompt = ("你是文档研究助手。只根据以下证据回答问题，逐项引用 evidence_id。"
                                    "证据不足就明确说明，不编造数值，不使用自身知识补齐。\n问题：" + case["query"]
                                    + "\n证据：" + json.dumps(build_evidence_context(task), ensure_ascii=False))
                                current["answer"] = await asyncio.to_thread(llm_service.generate_answer, prompt, operation="answer")
                                current["task"] = {k:v for k,v in asdict(task).items() if k != "owner_id"}
                            else:
                                task.status = "ready"
                                await store.add(task)
                                await store.enqueue_for_user(task.task_id, task.owner_id)
                                await run_once("h4-pilot", store=store)
                                done = await store.get_for_user(task.task_id, task.owner_id)
                                current["task"] = {k:v for k,v in asdict(done).items() if k != "owner_id"}
                                current["status"] = done.status
                                current["stop_reason"] = done.stop_reason
                                current["answer"] = done.report
                            await ensure_task_scope(task)
                            if current["status"] == "running":
                                current["status"] = "completed"
                    except Exception as error:
                        current["status"] = "failed"
                        current["error"] = getattr(error, "code", type(error).__name__)
                        current["error_detail"] = str(error)[:400]
                    current["elapsed_seconds"] = round(perf_counter() - current.pop("started"), 3)
                    current["model_calls"] = len(current["requests"])
                    current["retrieval_calls"] = len(current["retrievals"])
                    usages = [r["usage"] for r in current["requests"] if r.get("usage")]
                    current["tokens"] = sum(u.get("total_tokens", 0) for u in usages) if usages else None
                    # Remove application owner IDs from persisted response metadata.
                    if "result" in current:
                        current["result"].pop("assistant_message", None)
                    write_json(output / (case["id"] + "_" + arm + ".json"), current)
                    rows.append({k:current.get(k) for k in ("case", "arm", "status", "elapsed_seconds",
                        "model_calls", "retrieval_calls", "tokens", "stop_reason", "error", "throttle_wait_seconds")})
                    write_json(output / "summary.json", rows)
                    logging.info("PILOT_DONE %s", json.dumps(rows[-1], ensure_ascii=True))
    finally:
        try:
            for name in ("conversations", "messages", "research_tasks", "research_events", "research_evidence"):
                await temp[name].drop()
        finally:
            await database.close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-interval", type=float, default=30)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(pilot(args.source_task, args.output, args.resume, args.request_interval))
