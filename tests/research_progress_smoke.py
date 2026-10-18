"""Direct context/runner acceptance with fixtures; does not invoke pytest."""
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
from harness.research_progress import build_evidence_context, build_research_progress
from harness.research_task import ResearchTask


def make_task():
    return ResearchTask(task_id="progress", owner_id="owner", kb_id="kb", document_ids=("doc",),
        index_generations={"doc": "gen"}, created_at=datetime.now(timezone.utc), query="What is the method?")


def call(name, args):
    return [{"type": "function", "name": name, "arguments": json.dumps(args)}]


async def smoke():
    snapshots, prompts = [], []
    source = "Background. " * 90 + "METHOD_DETAIL: a controlled retrieval method."
    evidence_id = "progress:chunk"
    def propose(prompt, timeout, finalize=False):
        state = json.loads(prompt.split("\n", 1)[1])
        progress = state["research_progress"]
        prompts.append(progress)
        if not state["evidence"]:
            return call("search_documents", {"query": "method"})
        if not progress["subquestions"]:
            return call("save_research_note", {"question": "What is the method?",
                "conclusion": "The preview is insufficient", "evidence_ids": [evidence_id],
                "open_questions": ["What is the exact method?"]})
        if progress["suggested_next_step"]["action"] == "read_evidence":
            assert progress["current_question"] == "What is the exact method?"
            assert "METHOD_DETAIL" not in state["evidence"][0]["preview"]
            assert progress["subquestions"][0]["state"] == "open"
            return call("read_evidence", {"evidence_id": progress["suggested_next_step"]["evidence_id"]})
        if state["last_result"].get("text"):
            assert "METHOD_DETAIL" in state["last_result"]["text"]
            return call("save_research_note", {"question": "What is the method?",
                "conclusion": "A controlled retrieval method", "evidence_ids": [evidence_id], "open_questions": []})
        assert len(progress["subquestions"]) == 1 and progress["subquestions"][0]["state"] == "recorded_unverified"
        return call("finish_report", {"title": "Method", "findings": [{"text": "A controlled retrieval method",
            "evidence_ids": [evidence_id]}], "unresolved": []})

    async def checkpoint(task):
        if task.notes and task.pending_action is None:
            snapshots.append(deepcopy(asdict(task)))
            raise asyncio.CancelledError()

    with patch.object(research_tools, "ensure_task_scope", AsyncMock()), patch.object(
            research_tools, "_search_documents", return_value=[{"chunk_id": "chunk", "\u6587\u672c\u5757": source,
                "\u5143\u6570\u636e": {"user_id": "owner", "document_id": "doc", "index_generation_id": "gen", "page_number": 2}}]), patch.object(
            research_runner, "_propose_action", side_effect=propose):
        original = make_task()
        try:
            await research_runner.run_research_task(original, checkpoint)
            raise AssertionError("expected checkpoint interruption")
        except asyncio.CancelledError:
            pass
        restored = ResearchTask(**snapshots[-1])
        assert build_research_progress(restored) == build_research_progress(original)
        await research_runner.run_research_task(restored)
    assert restored.status == "completed" and restored.model_turns == restored.tool_calls == 5
    assert len(restored.notes) == 2 and restored.notes[0]["open_questions"]
    assert len([e for e in restored.events if e["name"] == "read_evidence"]) == 1
    assert build_research_progress(restored)["verification_status"] == "unverified_model_notes"
    # The real failure pattern: nine short chunks must not trigger a read-all loop.
    short = make_task()
    short.document_ids = ("a", "b", "c")
    short.evidence = {str(i): {"evidence_id": str(i), "document_id": short.document_ids[i // 3],
        "page_number": i + 1, "text": "Complete short source. " * 20} for i in range(9)}
    context = build_evidence_context(short)
    assert len(context) == 9 and all(not item["is_truncated"] for item in context)
    assert {item["evidence_id"] for item in context} == set(short.evidence)
    assert build_research_progress(short)["suggested_next_step"]["action"] == "save_research_note"
    short.notes = [{"question": "Main method?", "conclusion": "Not established by this source",
                    "evidence_ids": ["0"], "open_questions": ["What is the main method?"]}]
    assert build_research_progress(short)["suggested_next_step"]["action"] == "search_documents"
    # Earlier reads stay available after last_result has advanced to another action.
    short.evidence["0"]["text"] = "X" * 1200 + "EARLIER_READ"
    short.events = [{"name": "read_evidence", "args_summary": {"evidence_id": "0"}, "result": {"evidence_id": "0"}}]
    context = build_evidence_context(short)
    assert "EARLIER_READ" in next(item["preview"] for item in context if item["evidence_id"] == "0")

    # Long notes are bounded; the raw originals and unverified status remain intact.
    large = make_task()
    large.notes = [{"question": str(i) + "Q" * 1000, "conclusion": "C" * 4000,
                    "evidence_ids": [], "open_questions": ["O" * 1000] * 10} for i in range(20)]
    summary = build_research_progress(large)
    assert len(summary["subquestions"]) == 6 and summary["omitted_note_count"] == 14
    assert len(json.dumps(summary)) < 14000 and len(large.notes) == 20
    bounded = deepcopy(short)
    bounded.evidence = {str(i): {"evidence_id": str(i), "document_id": bounded.document_ids[i % 3],
        "page_number": 1, "text": "T" * 4000} for i in range(50)}
    bounded.events = [{"name": "read_evidence", "args_summary": {"evidence_id": str(i)},
                      "result": {"evidence_id": str(i)}} for i in range(40, 50)]
    context = build_evidence_context(bounded)
    assert len(context) <= 12 and sum(len(item["preview"]) for item in context) <= 12000
    assert {item["document_id"] for item in context} == set(bounded.document_ids)
    bounded.events = []
    context = build_evidence_context(bounded)
    assert "49" in {item["evidence_id"] for item in context}
    assert all(item["state"] == "open" and item["omitted_open_question_count"] == 7 for item in summary["subquestions"])
    # Finalization and document focus override optional suggestions.
    for finalize, focus, expected in [(True, None, "finish_report"), (False, "doc", "search_documents")]:
        prompt = research_runner._build_prompt(restored, None, finalize, focus)
        state = json.loads(prompt.split("\n", 1)[1])
        assert state["research_progress"]["suggested_next_step"]["action"] == expected
    for version in ("h2-v1", "h3-v1", "h3-v2"):
        old = make_task()
        old.config_version = version
        state = json.loads(research_runner._build_prompt(old, None).split("\n", 1)[1])
        assert "research_progress" not in state
    print(json.dumps({"result": "passed", "services": "fixtures", "subquestion_progress": True,
        "read_original_after_preview": True, "note_revision_preserves_originals": True,
        "restored_progress_matches": True, "context_bounded": True, "legacy_prompt_unchanged": True,
        "existing_budgets_preserved": True}))


if __name__ == "__main__":
    asyncio.run(smoke())
