from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api_errors import APIError, api_error_handler
from dependencies.auth import get_current_user_id
from harness.research_store import ResearchTaskStore
from harness.research_task import ResearchIndexUnavailableError, ResearchTask
from harness.research_tools import ResearchStoppedError
from routers import research_task_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(research_task_router.router)
    app.add_exception_handler(APIError, api_error_handler)
    app.state.research_task_store = ResearchTaskStore()
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    with TestClient(app) as test_client:
        yield test_client


def make_task() -> ResearchTask:
    return ResearchTask(
        task_id="task-1",
        owner_id="user-1",
        kb_id="kb-1",
        document_ids=("doc-1",),
        index_generations={"doc-1": "generation-1"},
        created_at=datetime(2026, 10, 14, tzinfo=timezone.utc),
        query="Compare three papers",
    )


def test_create_and_get_use_authenticated_owner(client, monkeypatch):
    create = AsyncMock(return_value=make_task())
    check_scope = AsyncMock()
    monkeypatch.setattr(research_task_router, "create_research_task", create)
    monkeypatch.setattr(research_task_router, "ensure_task_scope", check_scope)

    response = client.post("/research-tasks", json={
        "kb_id": "kb-1", "query": "Compare three papers",
        "selected_document_ids": ["doc-1"],
    })
    assert response.status_code == 201
    assert response.json()["status"] == "ready"
    assert response.json()["model_turns"] == 0
    assert response.json()["index_generations"] == {"doc-1": "generation-1"}
    create.assert_awaited_once_with(
        "user-1", "kb-1", "Compare three papers", ["doc-1"]
    )
    check_scope.assert_awaited_once()

    response = client.get("/research-tasks/task-1")
    assert response.status_code == 200
    assert response.json()["task_id"] == "task-1"
    assert response.json()["evidence"] == []


def test_other_user_cannot_read_task(client):
    task = make_task()
    client.app.state.research_task_store._tasks[task.task_id] = task
    client.app.dependency_overrides[get_current_user_id] = lambda: "user-2"

    response = client.get("/research-tasks/task-1")
    assert response.status_code == 404
    assert response.json()["error"] == "RESEARCH_TASK_NOT_FOUND"


def test_create_rejects_unavailable_index(client, monkeypatch):
    monkeypatch.setattr(
        research_task_router, "create_research_task",
        AsyncMock(return_value=make_task()),
    )
    monkeypatch.setattr(
        research_task_router, "ensure_task_scope",
        AsyncMock(side_effect=ResearchStoppedError("index_generation_unavailable")),
    )

    response = client.post(
        "/research-tasks", json={"kb_id": "kb-1", "query": "Compare"}
    )
    assert response.status_code == 409
    assert response.json()["error"] == "RESEARCH_SCOPE_UNAVAILABLE"
    assert client.get("/research-tasks/task-1").status_code == 404


def test_create_maps_missing_generation_to_conflict(client, monkeypatch):
    monkeypatch.setattr(
        research_task_router, "create_research_task",
        AsyncMock(side_effect=ResearchIndexUnavailableError("generation missing")),
    )
    response = client.post(
        "/research-tasks", json={"kb_id": "kb-1", "query": "Compare"}
    )
    assert response.status_code == 409
    assert response.json()["error"] == "RESEARCH_SCOPE_UNAVAILABLE"


def test_create_requires_bearer_token():
    app = FastAPI()
    app.include_router(research_task_router.router)
    app.add_exception_handler(APIError, api_error_handler)
    app.state.research_task_store = ResearchTaskStore()

    with TestClient(app) as client:
        response = client.post(
            "/research-tasks", json={"kb_id": "kb-1", "query": "Compare"}
        )
    assert response.status_code == 401
    assert response.json()["error"] == "AUTHENTICATION_REQUIRED"


def test_create_rejects_blank_research_question(client):
    response = client.post(
        "/research-tasks", json={"kb_id": "kb-1", "query": "   "}
    )
    assert response.status_code == 422


def test_run_enqueues_task_once(client):
    task = make_task()
    task.status = "ready"
    client.app.state.research_task_store._tasks[task.task_id] = task

    response = client.post("/research-tasks/task-1/run")
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert client.post("/research-tasks/task-1/run").status_code == 409
    assert task.started_at is not None


@pytest.mark.asyncio
async def test_task_store_has_capacity_and_owner_boundary():
    store = ResearchTaskStore(max_tasks=1)
    task = make_task()

    await store.add(task)
    assert await store.get_for_user("task-1", "user-2") is None
    with pytest.raises(RuntimeError, match="capacity"):
        await store.add(ResearchTask(
            task_id="task-2",
            owner_id="user-1",
            kb_id="kb-1",
            document_ids=("doc-1",),
            index_generations={"doc-1": "generation-1"},
            created_at=datetime.now(timezone.utc),
        ))


def test_cancel_is_owner_scoped_and_terminal(client):
    task = make_task()
    client.app.state.research_task_store._tasks[task.task_id] = task
    client.app.dependency_overrides[get_current_user_id] = lambda: "user-2"
    assert client.post("/research-tasks/task-1/cancel").status_code == 404
    assert task.status == "running"
    client.app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    response = client.post("/research-tasks/task-1/cancel")
    assert response.json()["status"] == "cancelled"
    assert response.json()["stop_reason"] == "user_cancelled"
    assert client.post("/research-tasks/task-1/run").status_code == 409
