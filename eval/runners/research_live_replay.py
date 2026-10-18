"""Replay one authorized research task against real services, without pytest.

Keeps application metadata/index reads on the configured database; writes the
experiment's task/event/evidence records to a unique temporary database.
"""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "medrag" / "backend"))

from bson.codec_options import CodecOptions
import config
from stores import database
from harness import research_runner
from harness.research_mongo_store import MongoResearchTaskStore
from harness.research_task import create_research_task, inherit_research_context
from harness.research_tools import ensure_task_scope, MAX_MODEL_ACTIONS, MAX_TOOL_CALLS, MAX_WALL_SECONDS
from harness.research_worker import run_once


class ExperimentStore(MongoResearchTaskStore):
    def __init__(self, name):
        self.name = name

    def _collection(self, name):
        return database.get_database().client[self.name][name].with_options(
            codec_options=CodecOptions(tz_aware=True, tzinfo=timezone.utc))


async def replay(source_id, output, same_start_context=False, model_call_seconds=None, wall_seconds=None):
    await database.connect_database()
    temporary_db = "docrag_h2_live_" + uuid4().hex[:16]
    store = ExperimentStore(temporary_db)
    model_outputs = []
    model_inputs = []
    propose = research_runner._propose_action

    def record_proposal(*args, **kwargs):
        model_inputs.append({"prompt": args[0], "timeout_seconds": args[1],
                             "finalize_only": args[2] if len(args) > 2 else False})
        result = propose(*args, **kwargs)
        model_outputs.append(result)
        logging.info("research_replay_proposal number=%s tools=%s", len(model_outputs),
                     [item.get("name") or item.get("function", {}).get("name") for item in result])
        return result

    started = perf_counter()
    try:
        source = await MongoResearchTaskStore().tasks.find_one({"task_id": source_id})
        if source is None:
            raise ValueError("source task not found")
        # Use the supplied task's exact owner/scope through the existing service.
        # This is a backend experiment, not an authenticated API/UI acceptance.
        task = await create_research_task(source["owner_id"], source["kb_id"],
                                          source["query"], source["document_ids"])
        if task.index_generations != source["index_generations"]:
            raise ValueError("source index generations changed; replay aborted")
        if same_start_context:
            task.wall_time_seconds = source.get("wall_time_seconds", 180)
            task.model_call_seconds = source.get("model_call_seconds", 45)
            if source.get("parent_task_id"):
                parent = await MongoResearchTaskStore().get_for_user(source["parent_task_id"], source["owner_id"])
                if parent is None:
                    raise ValueError("original parent is unavailable; same-start replay aborted")
                inherit_research_context(task, parent)
        if model_call_seconds is not None:
            task.model_call_seconds = model_call_seconds
        if wall_seconds is not None:
            task.wall_time_seconds = wall_seconds
        await ensure_task_scope(task)
        task.status = "ready"
        await store.create_indexes()
        await store.add(task)
        await store.enqueue_for_user(task.task_id, task.owner_id)
        research_runner._propose_action = record_proposal
        await run_once("live-replay-" + uuid4().hex, store=store)
        result = await store.get_for_user(task.task_id, task.owner_id)
        snapshot = asdict(result)
        snapshot.pop("owner_id")
        artifact = {
            "source_task_id": source_id, "acceptance": "real backend replay; no UI authentication test",
            "same_start_context": same_start_context,
            "initial_inherited_evidence_count": len(task.inherited_evidence_ids),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(perf_counter() - started, 2),
            "configuration": {
                "provider": config.ANSWER_LLM_PROVIDER, "model": config.ANSWER_LLM_MODEL,
                "retrieval_mode": config.RETRIEVAL_MODE,
                "evidence_rerank_min_score": config.EVIDENCE_RERANK_MIN_SCORE,
                "model_limit": MAX_MODEL_ACTIONS, "tool_limit": MAX_TOOL_CALLS,
                "wall_seconds": task.wall_time_seconds, "model_call_seconds": task.model_call_seconds,
            },
            "task": snapshot, "model_inputs": model_inputs, "model_outputs": model_outputs,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2,
                                    default=lambda value: value.isoformat()), encoding="utf-8")
        print(json.dumps({"task_id": result.task_id, "status": result.status,
                          "stop_reason": result.stop_reason, "model_turns": result.model_turns,
                          "tool_calls": result.tool_calls, "evidence_count": len(result.evidence),
                          "report_generated": result.report is not None,
                          "errors": [{"action": e["name"], "code": e["error_code"],
                                      "detail": e.get("error_detail")} for e in result.events if e.get("error_code")],
                          "artifact": str(output)}, ensure_ascii=True, indent=2), flush=True)
    finally:
        research_runner._propose_action = propose
        try:
            for collection in ("research_tasks", "research_events", "research_evidence"):
                await database.get_database().client[temporary_db][collection].drop()
        finally:
            await database.close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--same-start-context", action="store_true", help="Reuse original parent evidence and time limits")
    parser.add_argument("--model-call-seconds", type=float, help="Explicit per-model request limit for this run")
    parser.add_argument("--wall-seconds", type=float, help="Explicit total time limit for this run")
    args = parser.parse_args()
    if any(value is not None and value <= 0 for value in (args.model_call_seconds, args.wall_seconds)):
        parser.error("time limits must be positive")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(replay(args.source_task, args.output, args.same_start_context,
                       args.model_call_seconds, args.wall_seconds))
