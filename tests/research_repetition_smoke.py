"""Focused runner acceptance with fixture model/retrieval; run directly, no pytest."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "medrag" / "backend"))
from harness import research_runner, research_tools
from harness.research_task import ResearchTask


def make_task():
    return ResearchTask(task_id="repeat", owner_id="owner", kb_id="kb", document_ids=("doc",),
        index_generations={"doc": "gen"}, created_at=datetime.now(timezone.utc), query="Explain the method")


def call(name, args):
    return [{"type": "function", "name": name, "arguments": json.dumps(args)}]


def report():
    return call("finish_report", {"title": "Report", "findings": [
        {"text": "Original method evidence", "evidence_ids": ["repeat:chunk"]}], "unresolved": []})


async def smoke():
    searches = []
    def search(task, query, count, documents):
        searches.append((query, count, documents))
        return [{"chunk_id": "chunk", "\u6587\u672c\u5757": "Original method evidence",
                 "\u5143\u6570\u636e": {"user_id": "owner", "document_id": "doc",
                    "index_generation_id": "gen", "page_number": 1}}]

    saved = []
    async def checkpoint(task):
        if task.events and task.events[-1].get("error_code") == "repeated_search":
            saved.append(deepcopy(asdict(task)))
            raise asyncio.CancelledError()

    initial = make_task()
    proposal = call("search_documents", {"query": "method details"})
    with patch.object(research_tools, "ensure_task_scope", AsyncMock()), patch.object(research_tools, "_search_documents", side_effect=search):
        with patch.object(research_runner, "_propose_action", side_effect=[proposal, proposal]):
            try:
                await research_runner.run_research_task(initial, checkpoint)
                raise AssertionError("expected checkpoint interruption")
            except asyncio.CancelledError:
                pass
        restored = ResearchTask(**saved[-1])
        assert restored.duplicate_searches == 1 and restored.model_turns == 2
        assert restored.tool_calls == 1 and len(searches) == 1
        previous_id = restored.events[0]["action_id"]
        feedback = restored.last_result
        assert feedback["previous_action_id"] == previous_id
        assert feedback["evidence_ids"] == ["repeat:chunk"] and not feedback["tool_executed"]
        finals = []
        def resume(prompt, timeout, finalize=False):
            progress = json.loads(prompt.split("\n", 1)[1])
            finals.append(finalize)
            if len(finals) == 1:
                assert progress["last_result"]["required_action"] == "change_query_or_read_evidence"
                return call("search_documents", {"query": " method   details ", "document_ids": ["doc"]})
            assert finalize and progress["last_result"]["required_action"] == "finish_report"
            return report()
        with patch.object(research_runner, "_propose_action", side_effect=resume):
            await research_runner.run_research_task(restored)
        assert restored.status == "completed" and restored.report
        assert len(searches) == 1 and restored.model_turns == 4
        assert restored.tool_calls == restored.model_actions == 2
        assert sum(e.get("error_code") == "repeated_search" for e in restored.events) == 2
        assert finals == [False, True]

        # Higher Top-K and genuinely different query are allowed, even if results overlap.
        changed = make_task()
        with patch.object(research_runner, "_propose_action", side_effect=[proposal,
                call("search_documents", {"query": "method details", "n_results": 5}),
                call("search_documents", {"query": "specific experiment"}), report()]):
            before = len(searches)
            await research_runner.run_research_task(changed)
        assert changed.report and len(searches) - before == 3
        assert changed.duplicate_searches == 0

        # Pre-existing H3 tasks retain their execution policy.
        legacy = make_task()
        legacy.config_version = "h3-v1"
        with patch.object(research_runner, "_propose_action", side_effect=[proposal, proposal, report()]):
            before = len(searches)
            await research_runner.run_research_task(legacy)
        assert legacy.report and len(searches) - before == 2 and legacy.tool_calls == 3

        # Report-only enforcement cannot be evaded by proposing a different query.
        refusal = ResearchTask(**saved[-1])
        refusal.duplicate_searches = 2
        with patch.object(research_runner, "_propose_action", return_value=call(
                "search_documents", {"query": "another query"})):
            before = len(searches)
            await research_runner.run_research_task(refusal)
        assert refusal.stop_reason == "no_new_evidence" and len(searches) == before

        # Document subsets differ; invalid scope never gets treated as a cached success.
        scoped = make_task()
        scoped.document_ids = ("doc", "other-doc")
        assert research_runner._search_key(scoped, {"query": "method"}) != research_runner._search_key(
            scoped, {"query": "method", "document_ids": ["doc"]})
        assert research_runner._search_key(scoped, {"query": "method", "document_ids": ["outside"]}) is None
        assert research_runner._search_key(scoped, {"query": "method", "user_id": "other"}) is None
    print(json.dumps({"result": "passed", "services": "fixtures", "duplicate_retrieval_skipped": True,
        "feedback_survives_restart": True, "budget_preserved": True, "report_only_after_two_blocks": True,
        "different_query_scope_top_k_allowed": True, "h3_v1_compatibility": True}))


if __name__ == "__main__":
    asyncio.run(smoke())
