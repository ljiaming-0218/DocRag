from asyncio import to_thread

from services.agent_router_service import route_task
from services.document_service import get_existing_document_for_user
from services.llm_service import generate_answer
from services.message_service import create_message, get_recent_messages
from services.prompt_service import (
    build_rag_prompt,
    build_report_prompt,
    build_summary_prompt,
)
from services.query_rewrite_service import rewrite_query
from services.search_service import (
    search_relevant_chunks,
    search_summary_chunks,
)
from services.user_service import ALLOWED_USER_TYPES, get_existing_user
from stores.conversation_store import find_conversation_by_id


NO_SOURCE_ANSWER = "当前文献未提供相关信息。"


async def prepare_ask_context(
    user_id: str,
    conversation_id: str,
    query: str,
    history_limit: int = 6,
    n_results: int = 3,
    user_type: str | None = None,
    task_type: str = "qa",
) -> dict:
    user_id = user_id.strip()
    conversation_id = conversation_id.strip()
    query = query.strip()

    if not user_id:
        raise ValueError("user_id 不能为空")
    if not conversation_id:
        raise ValueError("conversation_id 不能为空")
    if not query:
        raise ValueError("query 不能为空")
    if history_limit < 1 or history_limit > 20:
        raise ValueError("history_limit 必须在 1 到 20 之间")
    if n_results < 1 or n_results > 10:
        raise ValueError("n_results 必须在 1 到 10 之间")

    conversation = await find_conversation_by_id(conversation_id)
    if conversation is None:
        raise ValueError("会话不存在")
    if conversation["user_id"] != user_id:
        raise ValueError("当前用户无权访问该会话")

    document = await get_existing_document_for_user(
        user_id,
        conversation["document_id"],
    )
    user = await get_existing_user(user_id)

    request_user_type = (
        user_type.strip().lower()
        if user_type and user_type.strip()
        else None
    )
    if (
        request_user_type
        and request_user_type not in ALLOWED_USER_TYPES
    ):
        raise ValueError("非法的 user_type")

    resolved_user_type = (
        request_user_type
        or conversation.get("user_type")
        or user.get("default_user_type")
        or "general"
    )

    history = await get_recent_messages(
        conversation_id,
        history_limit,
    )
    user_message = await create_message(
        conversation_id,
        "user",
        query,
    )
    rewritten_query = await to_thread(
        rewrite_query,
        history,
        query,
        document.get("language", "unknown"),
    )

    if task_type == "summary":
        summary_result = await to_thread(
            search_summary_chunks,
            user_id,
            conversation["document_id"],
            rewritten_query,
        )
        sources = summary_result["sources"]
        retrieval_queries = summary_result["retrieval_queries"]
    else:
        sources = await to_thread(
            search_relevant_chunks,
            user_id,
            conversation["document_id"],
            rewritten_query,
            n_results,
        )
        retrieval_queries = [rewritten_query]

    prompt = build_rag_prompt(
        query,
        sources,
        resolved_user_type,
        history,
    )

    return {
        "conversation_id": conversation["_id"],
        "document_id": conversation["document_id"],
        "user_id": conversation["user_id"],
        "history": history,
        "query": query,
        "user_type": resolved_user_type,
        "rewritten_query": rewritten_query,
        "retrieval_queries": retrieval_queries,
        "user_message": user_message,
        "sources": sources,
        "sources_count": len(sources),
        "prompt": prompt,
    }


async def ask_conversation(
    user_id: str,
    conversation_id: str,
    query: str,
    history_limit: int = 6,
    n_results: int = 3,
    user_type: str | None = None,
) -> dict:
    task_type = route_task(query)
    context = await prepare_ask_context(
        user_id=user_id,
        conversation_id=conversation_id,
        query=query,
        history_limit=history_limit,
        n_results=n_results,
        user_type=user_type,
        task_type=task_type,
    )

    if not context["sources_count"]:
        answer = NO_SOURCE_ANSWER
    elif task_type == "report":
        prompt = build_report_prompt(
            context["rewritten_query"],
            context["sources"],
            context["user_type"],
            context["history"],
        )
        answer = await to_thread(
            generate_answer,
            prompt,
            operation="report",
        )
    elif task_type == "summary":
        prompt = build_summary_prompt(
            context["rewritten_query"],
            context["sources"],
            context["user_type"],
            context["history"],
        )
        answer = await to_thread(
            generate_answer,
            prompt,
            operation="summary",
        )
    else:
        answer = await to_thread(
            generate_answer,
            context["prompt"],
            operation="answer",
        )

    assistant_message = await create_message(
        conversation_id,
        "assistant",
        answer,
        task_type=task_type,
        sources=context["sources"],
        rewritten_query=context["rewritten_query"],
        retrieval_queries=context["retrieval_queries"],
    )

    return {
        "conversation_id": context["conversation_id"],
        "user_type": context["user_type"],
        "answer": answer,
        "sources": context["sources"],
        "sources_count": context["sources_count"],
        "rewritten_query": context["rewritten_query"],
        "retrieval_queries": context["retrieval_queries"],
        "assistant_message": assistant_message,
        "task_type": task_type,
    }
