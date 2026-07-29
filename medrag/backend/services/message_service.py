from datetime import datetime, timezone
from uuid import uuid4

from stores.conversation_store import find_conversation_by_id, update_conversation_updated_at
from stores.message_store import find_recent_messages_by_conversation, insert_message, find_messages_by_conversation


async def create_message(
    conversation_id: str,
    role: str,
    content: str,
    task_type: str | None = None,
    sources: list[dict] | None = None,
    rewritten_query: str | None = None,
    retrieval_queries: list[str] | None = None,
) -> dict:
    conversation_id = conversation_id.strip()
    if not conversation_id:
        raise ValueError("conversation_id 不能为空")

    conversation = await find_conversation_by_id(conversation_id)
    if conversation is None:
        raise ValueError("会话不存在")
    role = role.strip().lower()
    if role not in {"user", "assistant"}:
        raise ValueError("role 只能是 user 或 assistant")

    content = content.strip()
    if not content:
        raise ValueError("content 不能为空")

    task_type = task_type.strip().lower() if task_type else None

    if sources is None:
        sources = []

    if not isinstance(sources, list):
        raise ValueError("sources 必须是列表")

    if retrieval_queries is None:
        retrieval_queries = []
    if not isinstance(retrieval_queries, list):
        raise ValueError("retrieval_queries 必须是列表")
    if any(
        not isinstance(query, str)
        for query in retrieval_queries
    ):
        raise ValueError("retrieval_queries 只能包含字符串")

    retrieval_queries = [
        query.strip()
        for query in retrieval_queries
        if query.strip()
    ]

    now = datetime.now(timezone.utc)
    message_id = str(uuid4())

    message = {
        "_id": message_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "sources": sources,
        "rewritten_query": rewritten_query,
        "retrieval_queries": retrieval_queries,
        "created_at": now,
        "task_type": task_type,
    }

    await insert_message(message)
    await update_conversation_updated_at(conversation_id, now)
    return {
        "message_id": message["_id"],
        "conversation_id": message["conversation_id"],
        "role": message["role"],
        "content": message["content"],
        "sources": message["sources"],
        "rewritten_query": message["rewritten_query"],
        "retrieval_queries": message["retrieval_queries"],
        "created_at": message["created_at"],
        "task_type": message.get("task_type"),
    }


async def list_messages(user_id: str, conversation_id: str) -> list[dict]:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id 不能为空")

    conversation_id = conversation_id.strip()
    if not conversation_id:
        raise ValueError("conversation_id 不能为空")

    conversation = await find_conversation_by_id(conversation_id)

    if conversation is None:
        raise ValueError("会话不存在")

    if conversation["user_id"] != user_id:
        raise PermissionError("当前用户无权访问该会话")

    messages = await find_messages_by_conversation(conversation_id)

    results = []
    for message in messages:
        results.append({
            "message_id": message["_id"],
            "conversation_id": message["conversation_id"],
            "role": message["role"],
            "content": message["content"],
            "sources": message.get("sources", []),
            "rewritten_query": message.get("rewritten_query"),
            "retrieval_queries": message.get(
                "retrieval_queries",
                [],
            ),
            "created_at": message["created_at"],
            "task_type": message.get("task_type"),
        })

    return results


async def get_recent_messages(
    conversation_id: str,
    limit: int = 6,
) -> list[dict]:
    if limit < 1 or limit > 20:
        raise ValueError("limit 必须在 1 到 20 之间")
    conversation_id = conversation_id.strip()
    if not conversation_id:
        raise ValueError("conversation_id 不能为空")

    conversation = await find_conversation_by_id(conversation_id)

    if conversation is None:
        raise ValueError("会话不存在")
    messages = await find_recent_messages_by_conversation(conversation_id, limit)

    results = []

    for message in messages:
        results.append({
            "message_id": message["_id"],
            "conversation_id": message["conversation_id"],
            "role": message["role"],
            "content": message["content"],
            "sources": message.get("sources", []),
            "rewritten_query": message.get("rewritten_query"),
            "retrieval_queries": message.get(
                "retrieval_queries",
                [],
            ),
            "created_at": message["created_at"],
            "task_type": message.get("task_type"),
        })

    return results
