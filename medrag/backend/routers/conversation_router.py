from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from api_errors import APIError
from services.llm_service import LLMServiceError
from services.document_service import DocumentNotReadyError
from services.message_service import list_messages
from services.conversation_service import create_conversation,list_conversations
from services.chat_service import ask_conversation
from dependencies.auth import get_current_user_id, require_matching_user
router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)

class CreateConversationRequest(BaseModel):
    user_id: str
    document_id: str | None = None
    kb_id: str | None = None
    selected_document_ids: list[str] | None = None
    title: str = "新会话"

class AskConversationRequest(BaseModel):
    user_id: str
    query: str = Field(min_length=1)
    history_limit: int = Field(default=6, ge=1, le=20)
    n_results: int = Field(default=3, ge=1, le=10)
    user_type: str | None = None

@router.post("", status_code=201)
async def create_conversation_endpoint(
    request: CreateConversationRequest,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    try:
        user_id = require_matching_user(
            authenticated_user_id,
            request.user_id,
        )
        conversation = await create_conversation(
            user_id=user_id,
            document_id=request.document_id,
            title=request.title,
            kb_id=request.kb_id,
            selected_document_ids=request.selected_document_ids,
        )
    except ValueError as error:
        raise APIError(
            status_code=400,
            code="INVALID_CONVERSATION_REQUEST",
            message=str(error),
        ) from error
    
    return conversation

@router.get("")
async def list_conversations_endpoint(
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    user_id: str = Query(...),
    document_id: str | None = Query(default=None),
    kb_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[dict]:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        return await list_conversations(
            user_id=user_id,
            document_id=document_id,
            limit=limit,
            kb_id=kb_id,
        )
    except ValueError as error:
        raise APIError(
            status_code=400,
            code="INVALID_CONVERSATION_REQUEST",
            message=str(error),
        ) from error

@router.get("/{conversation_id}/messages")
async def list_messages_endpoint(
    conversation_id: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    user_id: str = Query(...),
) -> list[dict]:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        return await list_messages(user_id, conversation_id)
    except APIError:
        raise
    except PermissionError as error:
        raise APIError(
            status_code=403,
            code="CONVERSATION_FORBIDDEN",
            message=str(error),
        ) from error
    except LookupError as error:
        raise APIError(
            status_code=404,
            code="CONVERSATION_NOT_FOUND",
            message=str(error),
        ) from error
    except ValueError as error:
        raise APIError(
            status_code=400,
            code="INVALID_CONVERSATION_REQUEST",
            message=str(error),
        ) from error


@router.post("/{conversation_id}/ask")
async def ask_conversation_endpoint(
    conversation_id: str,
    request: AskConversationRequest,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    try:
        user_id = require_matching_user(
            authenticated_user_id,
            request.user_id,
        )
        result = await ask_conversation(
            user_id,
            conversation_id,
            request.query,
            request.history_limit,
            request.n_results,
            request.user_type,
        )
    except APIError:
        raise
    except DocumentNotReadyError as error:
        raise APIError(
            status_code=409,
            code="DOCUMENT_NOT_READY",
            message=str(error),
        ) from error
    except LLMServiceError as error:
        raise APIError(
            status_code=error.status_code,
            code=error.code,
            message=str(error),
        ) from error
    except PermissionError as error:
        raise APIError(
            status_code=403,
            code="CONVERSATION_FORBIDDEN",
            message=str(error),
        ) from error
    except LookupError as error:
        raise APIError(
            status_code=404,
            code="CONVERSATION_NOT_FOUND",
            message=str(error),
        ) from error
    except ValueError as error:
        raise APIError(
            status_code=400,
            code="INVALID_CONVERSATION_REQUEST",
            message=str(error),
        ) from error
    
    return result
