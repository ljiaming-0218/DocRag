from datetime import datetime, timezone
from uuid import uuid4

from services.document_service import get_existing_document_for_user
from services.knowledge_base_service import (
    get_knowledge_base_for_user,
    resolve_knowledge_base_scope,
)
from stores.conversation_store import find_conversations,insert_conversation

from services.user_service import get_existing_user

async def create_conversation(
    user_id: str,
    document_id: str | None = None,
    title: str = "新会话",
    kb_id: str | None = None,
    selected_document_ids: list[str] | None = None,
) -> dict:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id不能为空")

    user = await get_existing_user(user_id)

    document_id = document_id.strip() if document_id else None
    kb_id = kb_id.strip() if kb_id else None
    if bool(document_id) == bool(kb_id):
        raise ValueError("provide exactly one of document_id or kb_id")

    if document_id:
        if selected_document_ids is not None:
            raise ValueError(
                "selected_document_ids requires a knowledge base"
            )
        await get_existing_document_for_user(user_id, document_id)
    else:
        scope = await resolve_knowledge_base_scope(
            user_id,
            kb_id,
            selected_document_ids,
        )
        if selected_document_ids is not None:
            selected_document_ids = scope["document_ids"]
    # 清理标题，空标题改为“新会话”
    title = title.strip() or "新会话"
    # 生成 now 和 conversation_id
    now = datetime.now(timezone.utc)
    conversation_id = str(uuid4())
    conversation = {
        "_id": conversation_id, 
        "title": title,
        "document_id": document_id,
        "kb_id": kb_id,
        "selected_document_ids": selected_document_ids,
        "user_id": user_id,  
        "user_type": user["default_user_type"],
        "created_at": now,
        "updated_at": now,
    }

    # await 调用 insert_conversation()
    await insert_conversation(conversation)
    return {
        "conversation_id": conversation["_id"],
        "title": conversation["title"],
        "document_id": conversation["document_id"],
        "kb_id": conversation["kb_id"],
        "selected_document_ids": conversation["selected_document_ids"],
        "user_type": conversation["user_type"],
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
    }


async def list_conversations(
    user_id: str,
    document_id: str | None = None,
    limit: int = 50,
    kb_id: str | None = None,
) -> list[dict]:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id 不能为空")
    
    await get_existing_user(user_id) 

    if limit <= 0 or limit > 100:
        raise ValueError("limit 必须在 1 到 100 之间")

    if document_id is not None:
        document_id = document_id.strip()

        if not document_id:
            raise ValueError("document_id 不能为空")

        await get_existing_document_for_user(user_id, document_id)

    if kb_id is not None:
        kb_id = kb_id.strip()
        if not kb_id:
            raise ValueError("kb_id cannot be empty")
        if document_id is not None:
            raise ValueError("document_id and kb_id cannot be used together")
        await get_knowledge_base_for_user(user_id, kb_id)

    if kb_id is None:
        conversations = await find_conversations(
            user_id=user_id,
            document_id=document_id,
            limit=limit,
        )
    else:
        conversations = await find_conversations(
            user_id=user_id,
            document_id=None,
            limit=limit,
            kb_id=kb_id,
        )
    results = []

    for conversation in conversations:
        results.append({
            "conversation_id": conversation["_id"],
            "user_id": conversation["user_id"],
            "title": conversation["title"],
            "document_id": conversation.get("document_id"),
            "kb_id": conversation.get("kb_id"),
            "selected_document_ids": conversation.get(
                "selected_document_ids"
            ),
            "user_type": conversation.get("user_type", "general"),
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
        })

    return results
