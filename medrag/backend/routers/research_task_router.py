from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator

from api_errors import APIError
from dependencies.auth import get_current_user_id
from harness.research_mongo_store import MongoResearchTaskStore
from harness.research_task import (
    ResearchIndexUnavailableError,
    ResearchTask,
    create_research_task,
    inherit_research_context,
)
from harness.research_tools import ResearchStoppedError, ensure_task_scope
from services.document_service import DocumentIndexUnavailableError, DocumentNotReadyError


router = APIRouter(prefix="/research-tasks", tags=["research-tasks"])


class CreateResearchTaskRequest(BaseModel):
    kb_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=2000)
    selected_document_ids: list[str] | None = Field(default=None, max_length=100)
    parent_task_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("research query cannot be blank")
        return value


def get_research_task_store(request: Request) -> MongoResearchTaskStore:
    store = getattr(request.app.state, "research_task_store", None)
    if store is None:
        raise APIError(503, "RESEARCH_STORE_UNAVAILABLE", "Research task store unavailable")
    return store


def public_task(task: ResearchTask) -> dict:
    return {
        "task_id": task.task_id,
        "parent_task_id": task.parent_task_id,
        "parent_query": (task.followup_context or {}).get("previous_question"),
        "kb_id": task.kb_id,
        "document_ids": list(task.document_ids),
        "document_filenames": dict(task.document_filenames),
        "index_generations": dict(task.index_generations),
        "created_at": task.created_at,
        "query": task.query,
        "status": task.status,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        "stop_reason": task.stop_reason,
        "model_turns": task.model_turns,
        "model_actions": task.model_actions,
        "tool_calls": task.tool_calls,
        "wall_time_seconds": task.wall_time_seconds,
        "model_call_seconds": task.model_call_seconds,
        "evidence": list(task.evidence.values()),
        "notes": list(task.notes),
        "report": task.report,
        "events": list(task.events) + ([task.pending_event] if task.pending_event else []),
    }


@router.post("", status_code=201)
async def create_research_task_endpoint(
    payload: CreateResearchTaskRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    store: Annotated[MongoResearchTaskStore, Depends(get_research_task_store)],
) -> dict:
    try:
        parent = None
        documents = payload.selected_document_ids
        if payload.parent_task_id:
            parent = await store.get_for_user(payload.parent_task_id, user_id)
            if parent is None:
                raise APIError(404, "RESEARCH_TASK_NOT_FOUND", "Parent research task not found")
            if parent.status in {"ready", "queued", "running"}:
                raise APIError(409, "RESEARCH_PARENT_NOT_FINISHED", "Wait for the previous task to finish")
            if payload.kb_id != parent.kb_id or (documents is not None and set(documents) != set(parent.document_ids)):
                raise APIError(409, "RESEARCH_SCOPE_MISMATCH", "Follow-up must retain the previous document scope")
            documents = list(parent.document_ids)
        task = await create_research_task(
            user_id, payload.kb_id, payload.query, documents
        )
        if parent is not None:
            inherit_research_context(task, parent)
        await ensure_task_scope(task)
        task.status = "ready"
        await store.add(task)
    except PermissionError as error:
        raise APIError(403, "RESEARCH_SCOPE_FORBIDDEN", str(error)) from error
    except LookupError as error:
        raise APIError(404, "RESEARCH_SCOPE_NOT_FOUND", str(error)) from error
    except ValueError as error:
        raise APIError(400, "INVALID_RESEARCH_REQUEST", str(error)) from error
    except (
        ResearchIndexUnavailableError,
        ResearchStoppedError,
        DocumentIndexUnavailableError,
        DocumentNotReadyError,
    ) as error:
        raise APIError(409, "RESEARCH_SCOPE_UNAVAILABLE", str(error)) from error
    except RuntimeError as error:
        raise APIError(503, "RESEARCH_STORE_UNAVAILABLE", str(error)) from error
    return public_task(task)


@router.get("/{task_id}")
async def get_research_task_endpoint(
    task_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    store: Annotated[MongoResearchTaskStore, Depends(get_research_task_store)],
) -> dict:
    task = await store.get_for_user(task_id, user_id)
    if task is None:
        raise APIError(404, "RESEARCH_TASK_NOT_FOUND", "Research task not found")
    return public_task(task)


@router.delete("/{task_id}")
async def delete_research_task_endpoint(
    task_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    store: Annotated[MongoResearchTaskStore, Depends(get_research_task_store)],
) -> dict:
    try:
        deleted = await store.delete_for_user(task_id, user_id)
    except ValueError as error:
        raise APIError(409, "RESEARCH_TASK_ACTIVE", str(error)) from error
    if not deleted:
        raise APIError(404, "RESEARCH_TASK_NOT_FOUND", "Research task not found")
    return {"deleted": True}


@router.post("/{task_id}/run", status_code=202)
async def run_research_task_endpoint(
    task_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    store: Annotated[MongoResearchTaskStore, Depends(get_research_task_store)],
) -> dict:
    try:
        task = await store.enqueue_for_user(task_id, user_id)
    except ValueError as error:
        raise APIError(409, "RESEARCH_TASK_NOT_READY", str(error)) from error
    if task is None:
        raise APIError(404, "RESEARCH_TASK_NOT_FOUND", "Research task not found")
    return public_task(task)


@router.post("/{task_id}/cancel")
async def cancel_research_task_endpoint(
    task_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    store: Annotated[MongoResearchTaskStore, Depends(get_research_task_store)],
) -> dict:
    task = await store.cancel_for_user(task_id, user_id)
    if task is None:
        raise APIError(404, "RESEARCH_TASK_NOT_FOUND", "Research task not found")
    return public_task(task)
