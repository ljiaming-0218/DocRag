from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.llm_service import LLMServiceError
from services.message_service import list_messages
from services.conversation_service import create_conversation,list_conversations
from services.chat_service import ask_conversation
router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)

class CreateConversationRequest(BaseModel):
    user_id: str
    document_id: str
    title: str = "新会话"

class AskConversationRequest(BaseModel):
    user_id: str
    query: str = Field(min_length=1)
    history_limit: int = Field(default=6, ge=1, le=20)
    n_results: int = Field(default=3, ge=1, le=10)
    user_type: str | None = None

@router.post("", status_code=201)
async def create_conversation_endpoint(request: CreateConversationRequest,) -> dict:
    try:
        conversation = await create_conversation(request.user_id, request.document_id, request.title)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return conversation

@router.get("")
async def list_conversations_endpoint(user_id: str = Query(...), document_id: str | None = Query(default=None), limit: int = Query(default=50, ge=1, le=100),) -> list[dict]:
    try:
        return await list_conversations(
            user_id=user_id,
            document_id=document_id,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{conversation_id}/messages")
async def list_messages_endpoint(conversation_id: str, user_id: str = Query(...)) -> list[dict]:
    try:
        return await list_messages(user_id, conversation_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{conversation_id}/ask")
async def ask_conversation_endpoint(conversation_id: str, request: AskConversationRequest,) -> dict:
    try:
        result = await ask_conversation(request.user_id, conversation_id, request.query, request.history_limit, request.n_results, request.user_type)
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "error": exc.code,
                "message": str(exc),
            },
        ) from exc
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return result