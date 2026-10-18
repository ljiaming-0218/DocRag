"""Run directly; fixture scope/model, real isolated Mongo persistence, no pytest."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "medrag" / "backend"))
from api_errors import APIError
from harness.research_mongo_store import MongoResearchTaskStore
from harness.research_task import ResearchTask
from harness import research_runner, research_tools
from routers import research_task_router as api
from stores import database
from stores.conversation_store import delete_conversation_for_user


async def smoke():
    await database.connect_database()
    name = "docrag_cont_smoke_" + uuid4().hex[:16]
    database.database = database.client[name]
    try:
        store = MongoResearchTaskStore()
        await store.create_indexes()
        parent = ResearchTask(
            task_id="parent", owner_id="owner", kb_id="kb", document_ids=("doc",),
            index_generations={"doc": "gen"}, created_at=datetime.now(timezone.utc),
            query="Original question", status="completed", model_turns=7, tool_calls=10,
            evidence={"parent:chunk": {"evidence_id": "parent:chunk", "chunk_id": "chunk",
                "document_id": "doc", "generation_id": "gen", "page_number": 1, "text": "Raw source"}},
            report={"title": "Previous report", "claims": [{"text": "Unverified finding",
                "status": "supported", "evidence_ids": ["parent:chunk"]}]},
        )
        await store.add(parent)
        before = deepcopy(await store.get_for_user("parent", "owner"))

        async def create(owner, kb, query, docs):
            return ResearchTask(task_id=uuid4().hex, owner_id=owner, kb_id=kb,
                document_ids=tuple(docs or ["doc"]), index_generations={"doc": "gen"},
                created_at=datetime.now(timezone.utc), query=query)

        with patch.object(api, "create_research_task", side_effect=create), patch.object(api, "ensure_task_scope", AsyncMock()):
            payload = api.CreateResearchTaskRequest(kb_id="kb", query="Explain that finding", parent_task_id="parent")
            response = await api.create_research_task_endpoint(payload, "owner", store)
            child = await store.get_for_user(response["task_id"], "owner")
            assert child.parent_task_id == "parent" and child.model_turns == child.tool_calls == 0
            assert child.evidence and list(child.evidence) == [child.task_id + ":chunk"]
            assert child.followup_context["previous_question"] == parent.query
            assert (await store.get_for_user("parent", "owner")) == before
            assert await store.get_for_user(child.task_id, "other") is None
            for bad, owner, code in [
                (payload, "other", 404),
                (api.CreateResearchTaskRequest(kb_id="other-kb", query="Q", parent_task_id="parent"), "owner", 409),
            ]:
                try:
                    await api.create_research_task_endpoint(bad, owner, store)
                    raise AssertionError("invalid parent/scope accepted")
                except APIError as error:
                    assert error.status_code == code
            independent = await api.create_research_task_endpoint(
                api.CreateResearchTaskRequest(kb_id="kb", query="Independent"), "owner", store)
            assert independent["parent_task_id"] is None and not independent["evidence"]
            with patch.object(api, "create_research_task", side_effect=create) as factory:
                async def changed(*args):
                    task = await create(*args)
                    task.index_generations = {"doc": "new-gen"}
                    return task
                factory.side_effect = changed
                try:
                    await api.create_research_task_endpoint(payload, "owner", store)
                    raise AssertionError("changed generation accepted")
                except APIError as error:
                    assert error.status_code == 409

        child.status = "running"
        def propose(prompt, *args):
            progress = json.loads(prompt.split("\n", 1)[1])
            assert progress["followup_context_unverified"]["previous_question"] == parent.query
            return [{"type": "function", "name": "finish_report", "arguments": json.dumps({
                "title": "Follow-up", "findings": [{"text": "Raw source", "evidence_ids": list(child.evidence)}],
                "unresolved": []})}]
        with patch.object(research_runner, "_propose_action", side_effect=propose), patch.object(research_tools, "ensure_task_scope", AsyncMock()):
            await research_runner.run_research_task(child)
        assert child.report and child.status == "completed" and child.model_turns == 1
        # Owner-scoped deletion also removes initial evidence and recorded events.
        assert not await store.delete_for_user(child.task_id, "other")
        await store.events.insert_one({"task_id": child.task_id, "action_id": "delete-smoke", "value": {}})
        assert (await api.delete_research_task_endpoint(child.task_id, "owner", store))["deleted"]
        assert await store.get_for_user(child.task_id, "owner") is None
        assert await store.evidence.count_documents({"task_id": child.task_id}) == 0
        assert await store.events.count_documents({"task_id": child.task_id}) == 0
        assert await store.get_for_user("parent", "owner") == before
        # A queued task must be cancelled before deletion.
        await store.enqueue_for_user(independent["task_id"], "owner")
        try:
            await api.delete_research_task_endpoint(independent["task_id"], "owner", store)
            raise AssertionError("active task deleted")
        except APIError as error:
            assert error.status_code == 409
        await store.cancel_for_user(independent["task_id"], "owner")
        await api.delete_research_task_endpoint(independent["task_id"], "owner", store)
        await database.database["conversations"].insert_one({"_id": "chat", "user_id": "owner"})
        await database.database["messages"].insert_one({"_id": "message", "conversation_id": "chat", "sources": [{"text": "Source"}]})
        assert not await delete_conversation_for_user("chat", "other")
        assert await database.database["messages"].count_documents({}) == 1
        assert await delete_conversation_for_user("chat", "owner")
        assert await database.database["messages"].count_documents({}) == 0
        print(json.dumps({"result": "passed", "storage": "real isolated MongoDB",
            "model_scope": "fixtures", "checks": ["parent ownership", "fixed scope and generation",
            "atomic initial evidence persistence", "independent task", "bounded inherited context",
            "child evidence IDs", "parent unchanged", "runner uses previous context",
            "owner-scoped research and chat deletion", "associated data cleanup", "active task deletion blocked"]}))
    finally:
        try:
            for collection in ("research_tasks", "research_events", "research_evidence", "conversations", "messages"):
                await database.database.drop_collection(collection)
        finally:
            await database.close_database()


if __name__ == "__main__":
    asyncio.run(smoke())
