from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from services import chat_service
from services import message_service


NOW = datetime(2026, 7, 30, tzinfo=timezone.utc)


def make_message(
    message_id: str,
    *,
    role: str = "user",
    status: str | None = "completed",
) -> dict:
    message = {
        "_id": message_id,
        "conversation_id": "conversation-1",
        "role": role,
        "content": f"content-{message_id}",
        "sources": [],
        "rewritten_query": None,
        "retrieval_queries": [],
        "created_at": NOW,
        "task_type": None,
    }
    if status is not None:
        message["status"] = status
        message["error_code"] = (
            "LLM_TIMEOUT" if status == "failed" else None
        )
        message["error_message"] = (
            "request timeout" if status == "failed" else None
        )
    return message


def make_ask_context() -> dict:
    return {
        "conversation_id": "conversation-1",
        "document_id": "document-1",
        "user_id": "user-1",
        "history": [],
        "query": "What is RAG?",
        "user_type": "general",
        "rewritten_query": "What is retrieval-augmented generation?",
        "retrieval_queries": [
            "What is retrieval-augmented generation?",
        ],
        "user_message": {
            "message_id": "user-message-1",
        },
        "sources": [{"文本块": "RAG combines retrieval and generation."}],
        "sources_count": 1,
        "prompt": "answer with the provided source",
    }


@pytest.mark.asyncio
async def test_set_message_status_awaits_store_update(monkeypatch):
    update_status = AsyncMock(return_value=True)
    monkeypatch.setattr(
        message_service,
        "update_message_status",
        update_status,
    )

    await message_service.set_message_status(
        "message-1",
        "failed",
        "LLM_TIMEOUT",
        "request timeout",
    )

    update_status.assert_awaited_once_with(
        "message-1",
        "failed",
        "LLM_TIMEOUT",
        "request timeout",
    )


@pytest.mark.asyncio
async def test_set_message_status_rejects_missing_message(monkeypatch):
    monkeypatch.setattr(
        message_service,
        "update_message_status",
        AsyncMock(return_value=False),
    )

    with pytest.raises(ValueError, match="消息不存在"):
        await message_service.set_message_status(
            "missing-message",
            "completed",
        )


@pytest.mark.asyncio
async def test_recent_history_excludes_failed_and_pending_messages(
    monkeypatch,
):
    messages = [
        make_message("legacy", status=None),
        make_message("completed"),
        make_message("pending", status="pending"),
        make_message("failed", status="failed"),
    ]
    monkeypatch.setattr(
        message_service,
        "find_conversation_by_id",
        AsyncMock(return_value={"_id": "conversation-1"}),
    )
    monkeypatch.setattr(
        message_service,
        "find_recent_messages_by_conversation",
        AsyncMock(return_value=messages),
    )

    result = await message_service.get_recent_messages(
        "conversation-1",
        limit=6,
    )

    assert [
        message["message_id"]
        for message in result
    ] == ["legacy", "completed"]


@pytest.mark.asyncio
async def test_list_messages_returns_failure_status(monkeypatch):
    monkeypatch.setattr(
        message_service,
        "find_conversation_by_id",
        AsyncMock(return_value={
            "_id": "conversation-1",
            "user_id": "user-1",
        }),
    )
    monkeypatch.setattr(
        message_service,
        "find_messages_by_conversation",
        AsyncMock(return_value=[
            make_message("failed", status="failed"),
        ]),
    )

    result = await message_service.list_messages(
        "user-1",
        "conversation-1",
    )

    assert result[0]["status"] == "failed"
    assert result[0]["error_code"] == "LLM_TIMEOUT"
    assert result[0]["error_message"] == "request timeout"


@pytest.mark.asyncio
async def test_successful_answer_marks_user_message_completed(
    monkeypatch,
):
    context = make_ask_context()
    set_status = AsyncMock()
    create_message = AsyncMock(return_value={
        "message_id": "assistant-message-1",
    })

    monkeypatch.setattr(
        chat_service,
        "route_task",
        Mock(return_value="qa"),
    )
    monkeypatch.setattr(
        chat_service,
        "prepare_ask_context",
        AsyncMock(return_value=context),
    )
    monkeypatch.setattr(
        chat_service,
        "generate_answer",
        Mock(return_value="RAG uses retrieved evidence."),
    )
    monkeypatch.setattr(
        chat_service,
        "create_message",
        create_message,
    )
    monkeypatch.setattr(
        chat_service,
        "set_message_status",
        set_status,
    )

    result = await chat_service.ask_conversation(
        "user-1",
        "conversation-1",
        "What is RAG?",
    )

    assert result["answer"] == "RAG uses retrieved evidence."
    set_status.assert_awaited_once_with(
        "user-message-1",
        "completed",
    )
    create_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_evidence_rejects_without_calling_llm(monkeypatch):
    context = make_ask_context()
    context["sources"] = []
    context["sources_count"] = 0
    generate_answer = Mock()
    create_message = AsyncMock(return_value={
        "message_id": "assistant-message-1",
    })

    monkeypatch.setattr(
        chat_service,
        "route_task",
        Mock(return_value="qa"),
    )
    monkeypatch.setattr(
        chat_service,
        "prepare_ask_context",
        AsyncMock(return_value=context),
    )
    monkeypatch.setattr(chat_service, "generate_answer", generate_answer)
    monkeypatch.setattr(chat_service, "create_message", create_message)
    monkeypatch.setattr(
        chat_service,
        "set_message_status",
        AsyncMock(),
    )

    result = await chat_service.ask_conversation(
        "user-1",
        "conversation-1",
        "What is RAG?",
    )

    assert result["answer"] == chat_service.NO_SOURCE_ANSWER
    assert result["sources"] == []
    generate_answer.assert_not_called()
    assert create_message.await_args.kwargs["sources"] == []


@pytest.mark.asyncio
async def test_safety_classifier_output_becomes_grounded_refusal(monkeypatch):
    context = make_ask_context()
    create_message = AsyncMock(return_value={
        "message_id": "assistant-message-1",
    })

    monkeypatch.setattr(
        chat_service,
        "route_task",
        Mock(return_value="qa"),
    )
    monkeypatch.setattr(
        chat_service,
        "prepare_ask_context",
        AsyncMock(return_value=context),
    )
    monkeypatch.setattr(
        chat_service,
        "generate_answer",
        Mock(return_value=(
            "User Safety: unsafe\n"
            "Response Safety: safe\n"
            "Safety Categories: medical advice"
        )),
    )
    monkeypatch.setattr(chat_service, "create_message", create_message)
    monkeypatch.setattr(
        chat_service,
        "set_message_status",
        AsyncMock(),
    )

    result = await chat_service.ask_conversation(
        "user-1",
        "conversation-1",
        "Can aspirin cure HIV?",
    )

    assert result["answer"] == chat_service.NO_SOURCE_ANSWER
    assert (
        create_message.await_args.args[2]
        == chat_service.NO_SOURCE_ANSWER
    )


@pytest.mark.asyncio
async def test_llm_failure_marks_user_message_failed(monkeypatch):
    context = make_ask_context()
    set_status = AsyncMock()

    def raise_timeout(*args, **kwargs):
        raise RuntimeError("simulated timeout")

    monkeypatch.setattr(
        chat_service,
        "route_task",
        Mock(return_value="qa"),
    )
    monkeypatch.setattr(
        chat_service,
        "prepare_ask_context",
        AsyncMock(return_value=context),
    )
    monkeypatch.setattr(
        chat_service,
        "generate_answer",
        raise_timeout,
    )
    monkeypatch.setattr(
        chat_service,
        "set_message_status",
        set_status,
    )

    with pytest.raises(RuntimeError, match="simulated timeout"):
        await chat_service.ask_conversation(
            "user-1",
            "conversation-1",
            "What is RAG?",
        )

    set_status.assert_awaited_once_with(
        "user-message-1",
        "failed",
        error_code="RuntimeError",
        error_message="simulated timeout",
    )
