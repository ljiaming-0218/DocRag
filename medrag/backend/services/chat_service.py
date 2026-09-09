from asyncio import to_thread

from services.agent_router_service import route_task
from services.document_service import (
    ensure_document_ready,
    get_existing_document_for_user,
)
from services.knowledge_base_service import resolve_knowledge_base_scope
from services.llm_service import generate_answer
from services.message_service import (
    create_message,
    get_recent_messages,
    set_message_status,
)
from services.prompt_service import (
    build_rag_prompt,
    build_report_prompt,
    build_source_check_prompt,
    build_summary_prompt,
    build_term_prompt,
)
from services.query_rewrite_service import rewrite_query
from services.search_service import (
    search_relevant_chunks,
    search_relevant_chunks_for_documents,
    search_summary_chunks,
    search_summary_chunks_for_documents,
)
from services.user_service import ALLOWED_USER_TYPES, get_existing_user
from stores.conversation_store import find_conversation_by_id


NO_SOURCE_ANSWER = "当前文献未提供相关信息。"
SAFETY_CLASSIFIER_MARKERS = (
    "user safety:",
    "response safety:",
)


def normalize_generated_answer(answer: str) -> str:
    normalized_answer = answer.strip()
    lowered_answer = normalized_answer.lower()
    if all(
        marker in lowered_answer
        for marker in SAFETY_CLASSIFIER_MARKERS
    ):
        return NO_SOURCE_ANSWER
    return normalized_answer


def _enrich_source_filenames(
    sources: list[dict],
    document_filenames: dict[str, str],
) -> None:
    for source in sources:
        metadata = next(
            (
                value
                for value in source.values()
                if isinstance(value, dict) and "document_id" in value
            ),
            None,
        )
        if metadata is None:
            continue
        filename = document_filenames.get(metadata["document_id"])
        if filename:
            metadata.setdefault("filename", filename)


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
        raise LookupError("会话不存在")
    if conversation["user_id"] != user_id:
        raise PermissionError("当前用户无权访问该会话")

    kb_id = conversation.get("kb_id")
    if kb_id:
        scope = await resolve_knowledge_base_scope(
            user_id,
            kb_id,
            conversation.get("selected_document_ids"),
        )
        document_ids = scope["document_ids"]
        document_filenames = scope["document_filenames"]
        document_language = scope["language"]
        knowledge_base_name = scope["knowledge_base_name"]
        index_generations = scope.get(
            "index_generations",
            {
                document_id: None
                for document_id in document_ids
            },
        )
    else:
        document = await get_existing_document_for_user(
            user_id,
            conversation["document_id"],
        )
        ensure_document_ready(document)
        document_ids = [conversation["document_id"]]
        document_filenames = {
            document["_id"]: document["filename"],
        }
        document_language = document.get("language", "unknown")
        knowledge_base_name = None
        index_generations = {
            document["_id"]: document.get(
                "active_index_generation_id"
            )
        }
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
        status="pending",
    )

    try:
        rewritten_query = await to_thread(
            rewrite_query,
            history,
            query,
            document_language,
        )
        if task_type == "summary" and kb_id:
            summary_result = await to_thread(
                search_summary_chunks_for_documents,
                user_id,
                document_ids,
                rewritten_query,
                index_generations=index_generations,
            )
            sources = summary_result["sources"]
            retrieval_queries = summary_result["retrieval_queries"]
        elif task_type == "summary":
            summary_result = await to_thread(
                search_summary_chunks,
                user_id,
                conversation["document_id"],
                rewritten_query,
                index_generation_id=index_generations.get(
                    conversation["document_id"]
                ),
            )
            sources = summary_result["sources"]
            retrieval_queries = summary_result["retrieval_queries"]
        elif kb_id:
            sources = await to_thread(
                search_relevant_chunks_for_documents,
                user_id,
                document_ids,
                rewritten_query,
                n_results,
                index_generations,
            )
            retrieval_queries = [rewritten_query]
        else:
            sources = await to_thread(
                search_relevant_chunks,
                user_id,
                conversation["document_id"],
                rewritten_query,
                n_results,
                index_generations.get(conversation["document_id"]),
            )
            retrieval_queries = [rewritten_query]

        _enrich_source_filenames(sources, document_filenames)

        prompt = build_rag_prompt(
            query,
            sources,
            resolved_user_type,
            history,
        )
    except Exception as exc:
        await set_message_status(
            user_message["message_id"],
            "failed",
            error_code=getattr(exc, "code", type(exc).__name__),
            error_message=str(exc),
        )
        raise




    return {
        "conversation_id": conversation["_id"],
        "document_id": conversation["document_id"],
        "kb_id": kb_id,
        "knowledge_base_name": knowledge_base_name,
        "document_ids": document_ids,
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

    try:
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
        elif task_type == "term":
            prompt = build_term_prompt(
                context["rewritten_query"],
                context["sources"],
                context["user_type"],
                context["history"],
            )
            answer = await to_thread(
                generate_answer,
                prompt,
                operation="term",
            )
        elif task_type == "source_check":
            prompt = build_source_check_prompt(
                context["rewritten_query"],
                context["sources"],
                context["user_type"],
                context["history"],
            )
            answer = await to_thread(
                generate_answer,
                prompt,
                operation="source_check",
            )
        else:
            answer = await to_thread(
                generate_answer,
                context["prompt"],
                operation="answer",
            )

        answer = normalize_generated_answer(answer)

        assistant_message = await create_message(
            conversation_id,
            "assistant",
            answer,
            task_type=task_type,
            sources=context["sources"],
            rewritten_query=context["rewritten_query"],
            retrieval_queries=context["retrieval_queries"],
        )
    except Exception as exc:
        await set_message_status(
            context["user_message"]["message_id"],
            "failed",
            error_code=getattr(exc, "code", type(exc).__name__),
            error_message=str(exc),
        )
        raise

    await set_message_status(
        context["user_message"]["message_id"],
        "completed",
    )

    return {
        "conversation_id": context["conversation_id"],
        "kb_id": context.get("kb_id"),
        "knowledge_base_name": context.get("knowledge_base_name"),
        "document_ids": context.get("document_ids", []),
        "user_type": context["user_type"],
        "answer": answer,
        "sources": context["sources"],
        "sources_count": context["sources_count"],
        "rewritten_query": context["rewritten_query"],
        "retrieval_queries": context["retrieval_queries"],
        "assistant_message": assistant_message,
        "task_type": task_type,
    }
