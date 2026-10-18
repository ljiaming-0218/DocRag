import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from harness import research_runner, research_tools
from harness.research_task import ResearchTask


def make_task() -> ResearchTask:
    return ResearchTask(
        task_id="task-1",
        owner_id="user-1",
        kb_id="kb-1",
        document_ids=("doc-1",),
        index_generations={"doc-1": "generation-1"},
        created_at=datetime.now(timezone.utc),
        query="Compare the methods",
    )


def tool_call(name: str, args: dict) -> list[dict]:
    return [{"type": "function", "name": name, "arguments": json.dumps(args)}]


def test_model_must_request_exactly_one_function_tool():
    assert research_runner._parse_action(tool_call("search_documents", {"query": "RAG"})) == {
        "name": "search_documents", "args": {"query": "RAG"},
    }
    with pytest.raises(ValueError, match="exactly one"):
        research_runner._parse_action([])
    with pytest.raises(ValueError, match="exactly one"):
        research_runner._parse_action(tool_call("search_documents", {}) * 2)


def test_report_status_is_assigned_from_separate_tool_fields():
    action = research_runner._parse_action(tool_call("finish_report", {
        "title": "Report", "findings": [{"text": "Cited finding", "evidence_ids": ["e1"]}],
        "unresolved": [{"text": "Unanswered question", "role": "comparison"}],
    }))
    assert [claim["status"] for claim in action["args"]["claims"]] == ["supported", "unknown"]
    assert action["args"]["claims"][1]["evidence_ids"] == []
    with pytest.raises(ValueError, match="nonempty evidence_ids"):
        research_runner._parse_action(tool_call("finish_report", {
            "title": "Report", "findings": [{"text": "Uncited", "evidence_ids": []}], "unresolved": [],
        }))


@pytest.mark.asyncio
async def test_runner_records_search_and_cited_report(monkeypatch):
    task = make_task()
    evidence_id = "task-1:chunk-1"
    actions = [
        {"name": "search_documents", "args": {"query": "method", "n_results": 2}},
        {"name": "finish_report", "args": {
            "title": "Comparison",
            "claims": [{
                "text": "The method is described in the paper",
                "status": "supported",
                "evidence_ids": [evidence_id],
            }],
        }},
    ]
    propose = Mock(side_effect=[tool_call(item["name"], item["args"]) for item in actions])
    monkeypatch.setattr(research_runner, "_propose_action", propose)
    monkeypatch.setattr(research_tools, "ensure_task_scope", AsyncMock())
    monkeypatch.setattr(research_tools, "_search_documents", Mock(return_value=[{
        "chunk_id": "chunk-1",
        "文本块": "Original method evidence",
        "元数据": {
            "user_id": "user-1",
            "document_id": "doc-1",
            "index_generation_id": "generation-1",
            "page_number": 2,
        },
        "retrieval_sources": ["dense"],
    }]))

    await research_runner.run_research_task(task)

    assert task.status == "completed"
    assert task.model_turns == 2
    assert task.model_actions == 2
    assert len(task.events) == 2
    assert task.events[0]["result"] == {"evidence_ids": [evidence_id]}
    assert task.events[0]["args_summary"]["query"] == "method"
    assert task.events[0]["token_usage"] == "unavailable"
    assert task.events[1]["result"] == {"claim_count": 1}
    assert task.report["claims"][0]["evidence_ids"] == [evidence_id]
    assert task.report["document_coverage"]["uncited_document_ids"] == []
    assert task.report["verification_status"] == "needs_semantic_review"
    assert task.report["cross_document_claim_count"] == 0
    assert "Original method evidence" not in str(task.events)
    assert "Original method evidence" in propose.call_args_list[1].args[0]


@pytest.mark.asyncio
async def test_harness_searches_missing_document_within_budget(monkeypatch):
    task = make_task()
    task.document_ids = ("doc-1", "doc-2")
    task.index_generations["doc-2"] = "generation-2"
    task.query = "Summarize both papers"
    proposals = Mock(side_effect=[
        tool_call("search_documents", {"query": "methods"}),
        tool_call("search_documents", {"query": "Second paper research problem and proposed method", "document_ids": ["doc-2"]}),
        tool_call("finish_report", {"title": "Report", "claims": [
            {"text": "First method", "status": "supported", "evidence_ids": ["task-1:chunk-1"]},
            {"text": "Second method", "status": "supported", "evidence_ids": ["task-1:chunk-2"]},
        ]}),
    ])
    monkeypatch.setattr(research_runner, "_propose_action", proposals)
    monkeypatch.setattr(research_tools, "ensure_task_scope", AsyncMock())

    def search(_task, _query, _n_results, document_ids):
        document_id = "doc-2" if document_ids == ["doc-2"] else "doc-1"
        number = "2" if document_id == "doc-2" else "1"
        return [{"chunk_id": f"chunk-{number}", "文本块": f"method {number}", "元数据": {
            "user_id": "user-1", "document_id": document_id,
            "index_generation_id": f"generation-{number}", "page_number": 1,
        }}]

    search_mock = Mock(side_effect=search)
    monkeypatch.setattr(research_tools, "_search_documents", search_mock)
    await research_runner.run_research_task(task)
    assert task.status == "completed"
    assert task.model_actions == 3
    assert task.model_turns == 3
    assert task.tool_calls == 3
    assert [event["actor"] for event in task.events] == ["model", "harness", "model"]
    assert task.events[1]["args_summary"]["document_ids"] == ["doc-2"]
    assert task.events[1]["result"]["evidence_ids"] == ["task-1:chunk-2"]
    assert task.events[1]["feedback"]["previous_queries"] == ["methods"]
    assert task.events[1]["feedback"]["document_id"] == "doc-2"
    assert "investigation_gap" in proposals.call_args_list[1].args[0]
    assert [call.args[3] for call in search_mock.call_args_list] == [["doc-1", "doc-2"], ["doc-2"]]


@pytest.mark.asyncio
async def test_invalid_model_action_stops_with_step_record(monkeypatch):
    task = make_task()
    monkeypatch.setattr(research_runner, "_propose_action", Mock(return_value=[]))

    await research_runner.run_research_task(task)

    assert task.status == "stopped"
    assert task.stop_reason == "invalid_model_action"
    assert task.model_turns == 1
    assert task.model_actions == 0
    assert len(task.events) == 1
    assert task.events[0]["error_code"] == "invalid_model_action"
    assert "exactly one" in task.events[0]["error_detail"]
    assert task.finished_at is not None


@pytest.mark.asyncio
async def test_invalid_report_preserves_validation_reason(monkeypatch):
    task = make_task()
    task.events = [{"name": "search_documents", "result": {"evidence_ids": []}}]
    monkeypatch.setattr(research_tools, "ensure_task_scope", AsyncMock())
    monkeypatch.setattr(research_runner, "_propose_action", Mock(return_value=tool_call(
        "finish_report", {"title": "Invalid report", "claims": [{
            "text": "Unsupported claim", "status": "supported", "evidence_ids": ["not-in-ledger"],
        }]},
    )))
    await research_runner.run_research_task(task)
    assert task.status == "stopped" and task.report is None
    assert task.events[-1]["error_code"] == "invalid_action"
    assert task.events[-1]["error_detail"] == "evidence_id is outside this task's evidence ledger"
    assert task.model_turns == 2  # One correction, with both requests charged.


@pytest.mark.asyncio
async def test_report_gets_one_correction_within_existing_budget(monkeypatch):
    task = make_task()
    task.events = [{"name": "search_documents", "result": {"evidence_ids": []}}]
    monkeypatch.setattr(research_tools, "ensure_task_scope", AsyncMock())
    proposal = Mock(side_effect=[
        tool_call("finish_report", {"title": "Draft", "claims": [{
            "text": "Unconfirmed finding", "status": "supported", "evidence_ids": [],
        }]}),
        tool_call("finish_report", {"title": "Report", "claims": [{
            "text": "The finding remains unknown", "status": "unknown", "evidence_ids": [],
        }]}),
    ])
    monkeypatch.setattr(research_runner, "_propose_action", proposal)
    await research_runner.run_research_task(task)
    assert task.status == "completed"
    assert task.model_turns == task.model_actions == task.tool_calls == 2
    assert task.events[-2]["error_code"] == "invalid_action"
    assert task.report["claims"][0]["status"] == "unknown"
    assert task.report["run_limitations"] == [{"code": "retrieval_gap", "document_ids": ["doc-1"]}]
    assert proposal.call_args_list[-1].args[2] is True
    assert "supported claim requires evidence" in proposal.call_args_list[-1].args[0]


@pytest.mark.asyncio
async def test_slow_tool_stops_at_wall_budget(monkeypatch):
    task = make_task()
    task.wall_time_seconds = 0.2
    monkeypatch.setattr(
        research_runner, "_propose_action",
        Mock(return_value=tool_call("search_documents", {"query": "method"})),
    )

    async def slow_action(_task, _action, **kwargs):
        await asyncio.sleep(1)
        return {"evidence": []}

    monkeypatch.setattr(research_runner, "execute_action", slow_action)

    await research_runner.run_research_task(task)

    assert task.status == "stopped"
    assert task.stop_reason == "wall_time_exhausted"
    assert task.events[0]["error_code"] == "wall_time_exhausted"
    assert task.model_turns == 1


@pytest.mark.asyncio
async def test_repeated_empty_searches_get_one_final_report_turn(monkeypatch):
    task = make_task()
    proposals = Mock(side_effect=[
        tool_call("search_documents", {"query": "method"}),
        tool_call("search_documents", {"query": "method again"}),
        tool_call("finish_report", {"title": "Partial", "claims": [
            {"text": "No supported comparison", "status": "unknown", "evidence_ids": []},
        ]}),
    ])
    monkeypatch.setattr(research_runner, "_propose_action", proposals)
    monkeypatch.setattr(research_tools, "ensure_task_scope", AsyncMock())
    monkeypatch.setattr(research_tools, "_search_documents", Mock(return_value=[]))

    await research_runner.run_research_task(task)

    assert task.status == "completed"
    assert task.report["claims"][0]["status"] == "unknown"
    assert [item["name"] for item in task.events] == [
        "search_documents", "search_documents", "finish_report",
    ]
    assert proposals.call_args_list[2].args[2] is True


@pytest.mark.asyncio
async def test_last_model_turn_is_reserved_for_report(monkeypatch):
    task = make_task()
    task.model_turns = research_runner.MAX_MODEL_ACTIONS - 1
    task.model_actions = research_runner.MAX_MODEL_ACTIONS - 1
    task.tool_calls = research_runner.MAX_MODEL_ACTIONS - 1
    task.events.append({"name": "search_documents", "result": {"evidence_ids": []}})
    propose = Mock(return_value=tool_call("finish_report", {
        "title": "Partial", "claims": [{
            "text": "No answer", "status": "unknown", "evidence_ids": [],
        }],
    }))
    monkeypatch.setattr(research_runner, "_propose_action", propose)
    monkeypatch.setattr(research_tools, "ensure_task_scope", AsyncMock())

    await research_runner.run_research_task(task)

    assert task.status == "completed"
    assert task.model_turns == research_runner.MAX_MODEL_ACTIONS
    assert propose.call_args.args[2] is True
