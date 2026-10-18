"""Focused H3 scope/budget/checkpoint smoke, with model and retrieval fixtures.

Run directly: python tests/research_followup_smoke.py (does not run pytest).
"""
import asyncio
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "medrag" / "backend"))
from harness import research_runner, research_tools
from harness.research_task import ResearchTask


def task():
    return ResearchTask(task_id="fixture", owner_id="owner", kb_id="kb",
                        document_ids=("doc-1", "doc-2"), query="Summarize both papers",
                        document_filenames={"doc-1": "First.pdf", "doc-2": "Second.pdf"},
                        index_generations={"doc-1": "gen-1", "doc-2": "gen-2"},
                        created_at=datetime.now(timezone.utc))


def call(name, args):
    return [{"type": "function", "name": name, "arguments": json.dumps(args)}]


async def smoke():
    searches, proposals = [], []
    research_tools.ensure_task_scope = AsyncMock()

    def search(current, query, count, documents):
        assert count == 3
        searches.append((query, documents))
        number = "2" if documents == ["doc-2"] else "1"
        return [{"chunk_id": "chunk-" + number, "\u6587\u672c\u5757": "Original method evidence",
                 "\u5143\u6570\u636e": {"user_id": "owner", "document_id": "doc-" + number,
                          "index_generation_id": "gen-" + number, "page_number": 1}}]

    def propose(prompt, timeout, finalize=False, focus=None):
        progress = json.loads(prompt.split("\n", 1)[1])
        proposals.append(focus)
        if focus:
            assert progress["investigation_gap"]["previous_queries"] == ["broad topic"]
            return call("search_documents", {"query": "Second paper proposed method research problem", "n_results": 10})
        if not progress["evidence"]:
            return call("search_documents", {"query": "broad topic"})
        return call("finish_report", {"title": "Report", "findings": [
            {"text": "Method evidence", "evidence_ids": [item["evidence_id"]]}
            for item in progress["evidence"]], "unresolved": []})

    research_runner._propose_action = propose
    research_tools._search_documents = search
    snapshots = []

    async def checkpoint(current):
        snapshots.append(deepcopy(asdict(current)))
        if current.pending_event and current.pending_event.get("feedback") and current.pending_action:
            raise asyncio.CancelledError()

    initial = task()
    try:
        await research_runner.run_research_task(initial, checkpoint)
        raise AssertionError("interruption was not injected")
    except asyncio.CancelledError:
        pass
    restored = ResearchTask(**snapshots[-1])
    assert restored.coverage_queue == [] and restored.pending_action["attempts"] == 1
    await research_runner.run_research_task(restored)
    assert restored.status == "completed"
    assert proposals == [None, "doc-2", None]
    assert searches[-1] == ("Second paper proposed method research problem", ["doc-2"])
    assert restored.model_turns == restored.model_actions == 3 and restored.tool_calls == 4
    assert restored.events[1]["attempts"] == 2 and restored.events[1]["feedback"]
    assert len({event["action_id"] for event in restored.events}) == 3

    unsafe = task()
    unsafe.coverage_attempted, unsafe.coverage_queue = True, ["doc-2"]
    research_runner._propose_action = Mock(return_value=call("search_documents", {
        "query": "another document", "document_ids": ["outside"]}))
    count = len(searches)
    await research_runner.run_research_task(unsafe)
    assert unsafe.stop_reason == "invalid_model_action" and len(searches) == count
    assert unsafe.tool_calls == 0

    final = task()
    final.coverage_queue, final.model_turns = ["doc-2"], 7
    final.events = [{"name": "search_documents", "result": {"evidence_ids": []}}]
    research_runner._propose_action = Mock(return_value=call("finish_report", {
        "title": "Unknown", "findings": [], "unresolved": [{"text": "Not established"}]}))
    await research_runner.run_research_task(final)
    assert final.model_turns == 8 and len(searches) == count
    assert research_runner._propose_action.call_args.args[2] is True

    legacy = task()
    legacy.config_version = "h2-v1"
    research_runner._propose_action = Mock(side_effect=[
        call("search_documents", {"query": "broad topic"}),
        call("finish_report", {"title": "Report", "findings": [
            {"text": "First method", "evidence_ids": ["fixture:chunk-1"]},
            {"text": "Second method", "evidence_ids": ["fixture:chunk-2"]}], "unresolved": []}),
    ])
    await research_runner.run_research_task(legacy)
    assert legacy.status == "completed" and legacy.model_turns == 2 and legacy.tool_calls == 3
    assert searches[-1] == (legacy.query, ["doc-2"])
    print(json.dumps({"result": "passed", "services": "fixtures", "rewritten_scoped_search": True,
                      "pending_action_recovery": True, "out_of_focus_scope_blocked": True,
                      "report_budget_reserved": True, "h2_task_compatibility": True}))


if __name__ == "__main__":
    asyncio.run(smoke())
