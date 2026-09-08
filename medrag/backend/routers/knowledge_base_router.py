from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api_errors import APIError
from dependencies.auth import get_current_user_id, require_matching_user
from services.knowledge_base_service import (
    add_document_to_knowledge_base,
    create_knowledge_base,
    edit_knowledge_base,
    get_knowledge_base,
    list_knowledge_base_documents,
    list_knowledge_bases,
    remove_document_from_knowledge_base,
    remove_knowledge_base,
)

router = APIRouter(
    prefix="/knowledge-bases",
    tags=["knowledge-bases"],
)


class CreateKnowledgeBaseRequest(BaseModel):
    user_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


class UpdateKnowledgeBaseRequest(BaseModel):
    user_id: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)


class KnowledgeBaseDocumentRequest(BaseModel):
    user_id: str = Field(min_length=1)


def _raise_api_error(error: Exception) -> None:
    if isinstance(error, PermissionError):
        raise APIError(403, "KNOWLEDGE_BASE_FORBIDDEN", str(error)) from error
    if isinstance(error, LookupError):
        raise APIError(404, "KNOWLEDGE_BASE_NOT_FOUND", str(error)) from error
    raise APIError(400, "INVALID_KNOWLEDGE_BASE_REQUEST", str(error)) from error


@router.post("", status_code=201)
async def create_knowledge_base_endpoint(
    request: CreateKnowledgeBaseRequest,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    try:
        user_id = require_matching_user(
            authenticated_user_id,
            request.user_id,
        )
        return await create_knowledge_base(
            user_id,
            request.name,
            request.description,
        )
    except (ValueError, LookupError, PermissionError) as error:
        _raise_api_error(error)


@router.get("")
async def list_knowledge_bases_endpoint(
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    user_id: str = Query(...),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[dict]:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        return await list_knowledge_bases(user_id, limit)
    except (ValueError, LookupError, PermissionError) as error:
        _raise_api_error(error)


@router.get("/{kb_id}")
async def get_knowledge_base_endpoint(
    kb_id: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    user_id: str = Query(...),
) -> dict:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        return await get_knowledge_base(user_id, kb_id)
    except (ValueError, LookupError, PermissionError) as error:
        _raise_api_error(error)


@router.patch("/{kb_id}")
async def update_knowledge_base_endpoint(
    kb_id: str,
    request: UpdateKnowledgeBaseRequest,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    try:
        user_id = require_matching_user(
            authenticated_user_id,
            request.user_id,
        )
        return await edit_knowledge_base(
            user_id,
            kb_id,
            request.name,
            request.description,
        )
    except (ValueError, LookupError, PermissionError) as error:
        _raise_api_error(error)


@router.delete("/{kb_id}")
async def delete_knowledge_base_endpoint(
    kb_id: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    user_id: str = Query(...),
) -> dict:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        return await remove_knowledge_base(user_id, kb_id)
    except (ValueError, LookupError, PermissionError) as error:
        _raise_api_error(error)


@router.post("/{kb_id}/documents/{document_id}")
async def add_document_to_knowledge_base_endpoint(
    kb_id: str,
    document_id: str,
    request: KnowledgeBaseDocumentRequest,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    try:
        user_id = require_matching_user(
            authenticated_user_id,
            request.user_id,
        )
        return await add_document_to_knowledge_base(
            user_id,
            kb_id,
            document_id,
        )
    except (ValueError, LookupError, PermissionError) as error:
        _raise_api_error(error)


@router.get("/{kb_id}/documents")
async def list_knowledge_base_documents_endpoint(
    kb_id: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    user_id: str = Query(...),
    limit: int = Query(default=100, ge=1, le=100),
) -> list[dict]:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        return await list_knowledge_base_documents(user_id, kb_id, limit)
    except (ValueError, LookupError, PermissionError) as error:
        _raise_api_error(error)


@router.delete("/{kb_id}/documents/{document_id}")
async def remove_document_from_knowledge_base_endpoint(
    kb_id: str,
    document_id: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    user_id: str = Query(...),
) -> dict:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        return await remove_document_from_knowledge_base(
            user_id,
            kb_id,
            document_id,
        )
    except (ValueError, LookupError, PermissionError) as error:
        _raise_api_error(error)
