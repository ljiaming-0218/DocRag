"""Real Mongo/process recovery smoke with fixture model/retrieval; no pytest.

Run from repo root: python tests/research_recovery_smoke.py
Uses a unique docrag_h2_smoke_* database and removes its three collections.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "medrag" / "backend"))
from stores import database
from harness import research_runner, research_tools, research_worker
from harness.research_mongo_store import MongoResearchTaskStore, ResearchLeaseLostError
from harness.research_task import ResearchTask


async def fixture_scope(task):
    pass


def fixture_search(task, query, n_results, document_ids):
    return [{"chunk_id": "chunk-1", "\u6587\u672c\u5757": "Fixture evidence.",
             "\u5143\u6570\u636e": {"user_id": task.owner_id, "document_id": "doc-1",
                      "index_generation_id": "gen-1", "page_number": 1}}]


def fixture_propose(prompt, timeout_seconds, finalize_only=False):
    state = json.loads(prompt.split("\n", 1)[1])
    if not state["evidence"]:
        name, args = "search_documents", {"query": "fixture"}
    elif not state["notes"]:
        name, args = "save_research_note", {
            "question": "Method?", "conclusion": "Fixture evidence.",
            "evidence_ids": [state["evidence"][0]["evidence_id"]], "open_questions": []}
    else:
        name, args = "finish_report", {"title": "Fixture report", "claims": [{
            "text": "Fixture evidence.", "status": "supported",
            "evidence_ids": [state["evidence"][0]["evidence_id"]]}]}
    return [{"type": "function", "name": name, "arguments": json.dumps(args)}]


class CrashStore(MongoResearchTaskStore):
    def __init__(self, phase=""):
        self.phase = phase

    async def checkpoint(self, task, worker_id, token):
        await super().checkpoint(task, worker_id, token)
        pending = task.pending_action
        action = pending["action"]["name"] if pending else ""
        phases = {
            "model_pending": task.pending_event is not None and pending is None,
            "search_pending": action == "search_documents",
            "search_completed": bool(task.evidence) and not task.notes and pending is None,
            "note_pending": action == "save_research_note",
            "note_completed": bool(task.notes) and task.report is None and pending is None,
            "report_pending": action == "finish_report",
            "report_completed": task.report is not None and task.status == "completed",
        }
        if phases.get(self.phase):
            os._exit(23)  # Deliberate process death after a real transaction commit.


async def connect(name):
    assert name.startswith("docrag_h2_smoke_") and len(name) > 20
    await database.connect_database()
    database.database = database.client[name]


async def child(name, phase):
    await connect(name)
    research_tools.ensure_task_scope = fixture_scope
    research_worker.ensure_task_scope = fixture_scope
    research_tools._search_documents = fixture_search
    research_runner._propose_action = fixture_propose
    try:
        await research_worker.run_once("child-" + uuid4().hex, store=CrashStore(phase), lease_seconds=2)
    finally:
        await database.close_database()


async def new_task(store):
    task = ResearchTask(task_id=uuid4().hex, owner_id="fixture-owner", kb_id="fixture-kb",
                        document_ids=("doc-1",), index_generations={"doc-1": "gen-1"},
                        created_at=datetime.now(timezone.utc), query="Explain the method", status="ready")
    await store.add(task)
    return await store.enqueue_for_user(task.task_id, task.owner_id)


async def run_child(name, phase=""):
    result = await asyncio.to_thread(subprocess.run, [sys.executable, __file__, "--database", name,
                                    "--crash-at", phase], capture_output=True, text=True, timeout=40)
    assert result.returncode == (23 if phase else 0), result.stderr[-3000:]


async def smoke():
    name = "docrag_h2_smoke_" + uuid4().hex[:16]
    await connect(name)
    store = MongoResearchTaskStore()
    results = []
    try:
        await store.create_indexes()
        for phase in ("model_pending", "search_pending", "search_completed", "note_pending",
                      "note_completed", "report_pending", "report_completed"):
            task = await new_task(store)
            await run_child(name, phase)
            before = await store.get_for_user(task.task_id, task.owner_id)
            action_id = before.pending_event["action_id"] if before.pending_event else None
            await asyncio.sleep(2.1)
            await run_child(name)
            after = await store.get_for_user(task.task_id, task.owner_id)
            assert after.status == "completed", (phase, after.stop_reason)
            assert len(after.notes) == 1 and len(after.evidence) == 1 and after.report
            assert after.started_at == before.started_at
            assert after.index_generations == {"doc-1": "gen-1"}
            assert before.model_turns <= after.model_turns <= 8
            assert before.tool_calls <= after.tool_calls <= 12
            ids = [event["action_id"] for event in after.events]
            assert len(ids) == len(set(ids))
            if action_id:
                assert action_id in ids
            if phase in {"search_pending", "note_pending", "report_pending"}:
                assert after.tool_calls == 4 and after.model_turns == 3
            if phase == "model_pending":
                assert after.model_turns == 4
                assert after.events[0]["error_code"] == "model_response_unconfirmed"
            assert await store.events.count_documents({"task_id": task.task_id}) == len(ids)
            assert await store.evidence.count_documents({"task_id": task.task_id}) == 1
            assert await store.get_for_user(task.task_id, "another-owner") is None
            results.append({"phase": phase, "status": after.status, "turns": after.model_turns,
                            "tools": after.tool_calls, "notes": len(after.notes)})

        # An expired worker cannot overwrite a new owner's state or a cancellation.
        task = await new_task(store)
        old, old_token = await store.claim("old-worker", 1)
        await asyncio.sleep(1.1)
        current, token = await store.claim("new-worker", 10)
        assert current.task_id == old.task_id
        try:
            await store.checkpoint(old, "old-worker", old_token)
            raise AssertionError("stale worker committed")
        except ResearchLeaseLostError:
            pass
        cancelled = await store.cancel_for_user(task.task_id, task.owner_id)
        assert cancelled.status == "cancelled"
        try:
            await store.checkpoint(current, "new-worker", token)
            raise AssertionError("cancelled worker committed")
        except ResearchLeaseLostError:
            pass
        assert await store.claim("third-worker") is None

        task = await new_task(store)
        await store.tasks.update_one({"task_id": task.task_id}, {"$set": {
            "started_at": datetime.now(timezone.utc) - timedelta(seconds=181), "model_turns": 2}})
        await run_child(name)
        expired = await store.get_for_user(task.task_id, task.owner_id)
        assert expired.stop_reason == "wall_time_exhausted" and expired.model_turns == 2
        print(json.dumps({"recovery": results, "stale_lease": "passed", "cancel": "passed",
                          "deadline": "passed", "model_retrieval": "fixtures"}, indent=2))
    finally:
        try:
            for collection in ("research_tasks", "research_events", "research_evidence"):
                await database.client[name][collection].drop()
        finally:
            await database.close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database")
    parser.add_argument("--crash-at", default="")
    args = parser.parse_args()
    asyncio.run(child(args.database, args.crash_at) if args.database else smoke())
