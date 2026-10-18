"""Controlled H3 probe: original scoped queries -> gap-driven query rewrites.

Real model/retrieval/scope checks; task state stays in this experiment process.
No pytest, indexing changes, authentication bypass, or application task writes.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import asyncio
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "medrag" / "backend"))
import config
from stores import database
from harness import research_runner
from harness.research_task import create_research_task
from harness.research_tools import ensure_task_scope, MAX_MODEL_ACTIONS, MAX_TOOL_CALLS


async def probe(source_id, output):
    await database.connect_database()
    original_proposal = research_runner._propose_action
    responses = []

    def record(*args, **kwargs):
        response = original_proposal(*args, **kwargs)
        responses.append({"focus_document_id": args[3] if len(args) > 3 else None, "response": response})
        return response

    started = perf_counter()
    try:
        source = await database.get_database().research_tasks.find_one({"task_id": source_id})
        if source is None:
            raise ValueError("source task not found")
        task = await create_research_task(source["owner_id"], source["kb_id"], source["query"], source["document_ids"])
        if task.index_generations != source["index_generations"]:
            raise ValueError("source index generations changed")
        if len(task.document_ids) > 3:
            raise ValueError("this controlled probe supports up to three documents")
        await ensure_task_scope(task)
        task.started_at = datetime.now(timezone.utc)
        task.coverage_attempted = True
        # Reproduce the old scoped-query policy with the same question, documents,
        # index, retrieval configuration and threshold; all calls use the real tool.
        for document_id in task.document_ids:
            event = research_runner._new_event(task, "harness")
            event["phase"] = "original_scoped_query"
            action = {"name": "search_documents", "args": {
                "query": task.query[:1000], "n_results": 3, "document_ids": [document_id]}}
            event["name"], event["args_summary"] = action["name"], research_runner._args_summary(action)
            task.pending_event, task.pending_action = event, {"action": action, "attempts": 0}
            await research_runner._execute_pending(task, None)
            if task.status != "running":
                break
        baseline_ids = set(task.evidence)
        baseline_by_document = {doc: [eid for eid, item in task.evidence.items() if item["document_id"] == doc]
                                for doc in task.document_ids}
        task.coverage_queue = [doc for doc in task.document_ids if not baseline_by_document[doc]][:3]
        gaps = list(task.coverage_queue)
        research_runner._propose_action = record
        if task.status == "running":
            await research_runner.run_research_task(task)
        assert task.model_turns <= MAX_MODEL_ACTIONS and task.tool_calls <= MAX_TOOL_CALLS
        followups = [{"document_id": event["feedback"]["document_id"],
                      "filename": event["feedback"]["filename"],
                      "previous_queries": event["feedback"]["previous_queries"],
                      "rewritten_query": event["args_summary"].get("query"),
                      "evidence_ids": (event.get("result") or {}).get("evidence_ids", []),
                      "error_code": event.get("error_code")} for event in task.events if event.get("feedback")]
        snapshot = asdict(task)
        snapshot.pop("owner_id")
        artifact = {"source_task_id": source_id, "acceptance": "Controlled real-services gap-feedback probe, not a UI or persistence acceptance",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": round(perf_counter() - started, 2),
                    "configuration": {"provider": config.ANSWER_LLM_PROVIDER, "model": config.ANSWER_LLM_MODEL,
                                      "retrieval_mode": config.RETRIEVAL_MODE,
                                      "threshold": config.EVIDENCE_RERANK_MIN_SCORE,
                                      "model_limit": MAX_MODEL_ACTIONS, "tool_limit": MAX_TOOL_CALLS},
                    "baseline_evidence_by_document": baseline_by_document, "initial_gaps": gaps,
                    "followups": followups, "new_evidence_ids": sorted(set(task.evidence) - baseline_ids),
                    "task": snapshot, "model_responses": responses}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=lambda value: value.isoformat()), encoding="utf-8")
        print(json.dumps({"status": task.status, "stop_reason": task.stop_reason,
                          "baseline_evidence_count": len(baseline_ids), "final_evidence_count": len(task.evidence),
                          "model_turns": task.model_turns, "tool_calls": task.tool_calls,
                          "followups": followups, "artifact": str(output)}, ensure_ascii=True, indent=2), flush=True)
    finally:
        research_runner._propose_action = original_proposal
        await database.close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(probe(args.source_task, args.output))
