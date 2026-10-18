from unittest.mock import AsyncMock, Mock

import pytest

from harness import research_task, research_tools


def scope(generation: str = "generation-1") -> dict:
    return {
        "kb_id": "kb-1",
        "document_ids": ["doc-1"],
        "index_generations": {"doc-1": generation},
        "document_filenames": {"doc-1": "paper.pdf"},
    }


@pytest.mark.asyncio
async def test_task_freezes_authorized_scope_and_rejects_legacy_index(monkeypatch):
    resolver = AsyncMock(return_value=scope())
    monkeypatch.setattr(research_task, "resolve_knowledge_base_scope", resolver)
    task = await research_task.create_research_task("user-1", "kb-1", "Compare", ["doc-1"])
    assert task.owner_id == "user-1"
    assert task.document_ids == ("doc-1",)
    assert task.index_generations == {"doc-1": "generation-1"}
    assert task.document_filenames == {"doc-1": "paper.pdf"}
    resolver.assert_awaited_once_with("user-1", "kb-1", ["doc-1"])

    resolver.return_value = scope("")
    with pytest.raises(research_task.ResearchIndexUnavailableError, match="explicit index generation"):
        await research_task.create_research_task("user-1", "kb-1", "Compare")


@pytest.mark.asyncio
async def test_search_uses_frozen_scope_and_ledger_ids(monkeypatch):
    monkeypatch.setattr(research_task, "resolve_knowledge_base_scope", AsyncMock(return_value=scope()))
    monkeypatch.setattr(research_tools, "resolve_knowledge_base_scope", AsyncMock(return_value=scope()))
    monkeypatch.setattr(research_tools, "has_chunks", Mock(return_value=True))
    search = Mock(return_value=[{
        "chunk_id": "chunk-1",
        "文本块": "原文证据",
        "元数据": {
            "user_id": "user-1",
            "document_id": "doc-1",
            "index_generation_id": "generation-1",
            "page_number": 2,
        },
        "retrieval_sources": ["dense"],
        "rerank_score": 0.8,
    }])
    monkeypatch.setattr(research_tools, "_search_documents", search)
    task = await research_task.create_research_task("user-1", "kb-1", "Compare")

    result = await research_tools.execute_action(task, {
        "name": "search_documents", "args": {"query": "研究问题"},
    })
    search.assert_called_once_with(task, "研究问题", 3, ["doc-1"])
    evidence_id = result["evidence"][0]["evidence_id"]
    assert evidence_id == f"{task.task_id}:chunk-1"
    assert (await research_tools.execute_action(task, {
        "name": "read_evidence", "args": {"evidence_id": evidence_id},
    }))["text"] == "原文证据"
    with pytest.raises(ValueError, match="unsupported research action arguments"):
        await research_tools.execute_action(task, {
            "name": "search_documents",
            "args": {"query": "x", "user_id": "other-user"},
        })


@pytest.mark.asyncio
async def test_search_document_subset_cannot_expand_task_scope(monkeypatch):
    monkeypatch.setattr(research_task, "resolve_knowledge_base_scope", AsyncMock(return_value=scope()))
    monkeypatch.setattr(research_tools, "resolve_knowledge_base_scope", AsyncMock(return_value=scope()))
    monkeypatch.setattr(research_tools, "has_chunks", Mock(return_value=True))
    search = Mock(return_value=[])
    monkeypatch.setattr(research_tools, "_search_documents", search)
    task = await research_task.create_research_task("user-1", "kb-1", "Compare")

    with pytest.raises(ValueError, match="subset"):
        await research_tools.execute_action(task, {
            "name": "search_documents",
            "args": {"query": "CoT", "document_ids": ["other-doc"]},
        })
    search.assert_not_called()


@pytest.mark.asyncio
async def test_generation_change_stops_task_before_search(monkeypatch):
    monkeypatch.setattr(research_task, "resolve_knowledge_base_scope", AsyncMock(return_value=scope()))
    monkeypatch.setattr(research_tools, "resolve_knowledge_base_scope", AsyncMock(return_value=scope("generation-2")))
    search = Mock()
    monkeypatch.setattr(research_tools, "_search_documents", search)
    task = await research_task.create_research_task("user-1", "kb-1", "Compare")

    with pytest.raises(research_tools.ResearchStoppedError, match="index_generation_changed"):
        await research_tools.execute_action(task, {
            "name": "search_documents", "args": {"query": "x"},
        })
    assert task.status == "stopped"
    search.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result_user_id", "result_document_id"),
    [("other-user", "doc-1"), ("user-1", "other-doc")],
)
async def test_out_of_scope_search_result_never_enters_ledger(
    monkeypatch, result_user_id, result_document_id,
):
    monkeypatch.setattr(research_task, "resolve_knowledge_base_scope", AsyncMock(return_value=scope()))
    monkeypatch.setattr(research_tools, "resolve_knowledge_base_scope", AsyncMock(return_value=scope()))
    monkeypatch.setattr(research_tools, "has_chunks", Mock(return_value=True))
    monkeypatch.setattr(research_tools, "_search_documents", Mock(return_value=[{
        "chunk_id": "other-chunk",
        "文本块": "另一用户的内容",
        "元数据": {
            "user_id": result_user_id,
            "document_id": result_document_id,
            "index_generation_id": "generation-1",
        },
    }]))
    task = await research_task.create_research_task("user-1", "kb-1", "Compare")

    with pytest.raises(research_tools.ResearchStoppedError, match="retrieval_scope_mismatch"):
        await research_tools.execute_action(task, {
            "name": "search_documents", "args": {"query": "x"},
        })
    assert task.evidence == {}


@pytest.mark.asyncio
async def test_report_requires_evidence_for_supported_claim(monkeypatch):
    monkeypatch.setattr(research_task, "resolve_knowledge_base_scope", AsyncMock(return_value=scope()))
    monkeypatch.setattr(research_tools, "resolve_knowledge_base_scope", AsyncMock(return_value=scope()))
    monkeypatch.setattr(research_tools, "has_chunks", Mock(return_value=True))
    task = await research_task.create_research_task("user-1", "kb-1", "Compare")

    with pytest.raises(ValueError, match="completed search"):
        await research_tools.execute_action(task, {
            "name": "finish_report", "args": {
                "title": "报告",
                "claims": [{"text": "尚无证据", "status": "unknown", "evidence_ids": []}],
            },
        })
    task.events.append({"name": "search_documents", "result": {"evidence_ids": []}})

    with pytest.raises(ValueError, match="requires evidence"):
        await research_tools.execute_action(task, {
            "name": "finish_report", "args": {
                "title": "报告",
                "claims": [{"text": "确定事实", "status": "supported", "evidence_ids": []}],
            },
        })
    assert task.report is None
    report = await research_tools.execute_action(task, {
        "name": "finish_report", "args": {
            "title": "报告",
            "claims": [{"text": "尚无证据", "status": "unknown", "evidence_ids": []}],
        },
    })
    assert report["claims"][0]["status"] == "unknown"
    assert report["document_coverage"]["uncited_document_ids"] == ["doc-1"]
    assert report["run_limitations"] == [{"code": "retrieval_gap", "document_ids": ["doc-1"]}]
    assert task.status == "completed"


@pytest.mark.asyncio
async def test_report_audit_counts_cross_document_claims(monkeypatch):
    two_doc_scope = {
        "kb_id": "kb-1",
        "document_ids": ["doc-1", "doc-2"],
        "index_generations": {"doc-1": "generation-1", "doc-2": "generation-2"},
    }
    monkeypatch.setattr(research_task, "resolve_knowledge_base_scope", AsyncMock(return_value=two_doc_scope))
    monkeypatch.setattr(research_tools, "resolve_knowledge_base_scope", AsyncMock(return_value=two_doc_scope))
    monkeypatch.setattr(research_tools, "has_chunks", Mock(return_value=True))
    task = await research_task.create_research_task("user-1", "kb-1", "Compare")
    task.events.append({"name": "search_documents", "result": {"evidence_ids": ["e1", "e2"]}})
    task.evidence = {
        "e1": {"document_id": "doc-1", "text": "first"},
        "e2": {"document_id": "doc-2", "text": "second"},
    }

    report = await research_tools.execute_action(task, {
        "name": "finish_report", "args": {
            "title": "Comparison",
            "claims": [
                {"text": "The methods differ", "status": "supported", "evidence_ids": ["e1", "e2"], "role": "comparison"},
                {"text": "The measurements are not directly comparable", "status": "supported", "evidence_ids": ["e1", "e2"], "role": "limitation"},
            ],
        },
    })

    assert report["cross_document_claim_count"] == 1
    assert report["run_limitations"] == []
    assert report["document_coverage"]["uncited_document_ids"] == []
    assert report["question_coverage"]["status"] == "structurally_complete"


@pytest.mark.asyncio
async def test_comparison_report_with_only_summaries_is_partial(monkeypatch):
    two_doc_scope = {
        "kb_id": "kb-1", "document_ids": ["doc-1", "doc-2"],
        "index_generations": {"doc-1": "generation-1", "doc-2": "generation-2"},
    }
    monkeypatch.setattr(research_task, "resolve_knowledge_base_scope", AsyncMock(return_value=two_doc_scope))
    monkeypatch.setattr(research_tools, "resolve_knowledge_base_scope", AsyncMock(return_value=two_doc_scope))
    monkeypatch.setattr(research_tools, "has_chunks", Mock(return_value=True))
    task = await research_task.create_research_task("user-1", "kb-1", "判断两篇论文是否存在关联并说明依据")
    task.events.append({"name": "search_documents", "result": {"evidence_ids": ["e1", "e2"]}})
    task.evidence = {
        "e1": {"document_id": "doc-1", "text": "first"},
        "e2": {"document_id": "doc-2", "text": "second"},
    }
    report = await research_tools.execute_action(task, {
        "name": "finish_report", "args": {"title": "Summaries", "claims": [
            {"text": "First method", "status": "supported", "evidence_ids": ["e1"]},
            {"text": "Second method", "status": "supported", "evidence_ids": ["e2"]},
        ]},
    })
    assert task.status == "partial"
    assert task.stop_reason == "report_incomplete"
    assert {gap["code"] for gap in report["question_coverage"]["gaps"]} == {
        "missing_cross_document_comparison", "missing_comparison_limitations",
    }


@pytest.mark.asyncio
async def test_budget_stops_before_running_another_tool(monkeypatch):
    monkeypatch.setattr(research_task, "resolve_knowledge_base_scope", AsyncMock(return_value=scope()))
    search = Mock()
    monkeypatch.setattr(research_tools, "_search_documents", search)
    task = await research_task.create_research_task("user-1", "kb-1", "Compare")
    task.model_actions = research_tools.MAX_MODEL_ACTIONS

    with pytest.raises(research_tools.ResearchStoppedError, match="budget_exhausted"):
        await research_tools.execute_action(task, {
            "name": "search_documents", "args": {"query": "x"},
        })
    assert task.status == "stopped"
    search.assert_not_called()
