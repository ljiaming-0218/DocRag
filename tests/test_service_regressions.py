from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from services import conversation_service
from services import document_service
from services import message_service
from services import query_rewrite_service


NOW = datetime(2026, 7, 29, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_list_conversations_without_document_id(monkeypatch):
    mock_user = AsyncMock(return_value={"user_id": "user-1"})
    mock_document = AsyncMock()
    mock_find = AsyncMock(return_value=[{
        "_id": "conversation-1",
        "user_id": "user-1",
        "document_id": "document-1",
        "title": "测试会话",
        "user_type": "general",
        "created_at": NOW,
        "updated_at": NOW,
    }])

    monkeypatch.setattr(
        conversation_service, "get_existing_user", mock_user
    )
    monkeypatch.setattr(
        conversation_service,
        "get_existing_document_for_user",
        mock_document,
    )
    monkeypatch.setattr(
        conversation_service, "find_conversations", mock_find
    )

    result = await conversation_service.list_conversations("user-1")

    assert result[0]["conversation_id"] == "conversation-1"
    mock_document.assert_not_awaited()
    mock_find.assert_awaited_once_with(
        user_id="user-1",
        document_id=None,
        limit=50,
    )


@pytest.mark.asyncio
async def test_list_conversations_rejects_blank_document_id(monkeypatch):
    monkeypatch.setattr(
        conversation_service,
        "get_existing_user",
        AsyncMock(return_value={"user_id": "user-1"}),
    )
    mock_find = AsyncMock()
    monkeypatch.setattr(
        conversation_service, "find_conversations", mock_find
    )

    with pytest.raises(ValueError):
        await conversation_service.list_conversations(
            "user-1",
            document_id="   ",
        )

    mock_find.assert_not_awaited()


@pytest.mark.asyncio
async def test_document_reuses_same_user_and_hash(monkeypatch):
    existing_document = {
        "_id": "document-1",
        "user_id": "user-1",
        "filename": "paper.pdf",
        "document_hash": "hash-1",
        "language": "en",
        "created_at": NOW,
        "updated_at": NOW,
    }

    monkeypatch.setattr(
        document_service,
        "find_document_by_user_and_hash",
        AsyncMock(return_value=existing_document),
    )
    mock_insert = AsyncMock()
    monkeypatch.setattr(
        document_service, "insert_document", mock_insert
    )

    result = await document_service.get_or_create_document(
        "user-1",
        "paper.pdf",
        "hash-1",
    )

    assert result["document_id"] == "document-1"
    assert result["existing_document"] is True
    mock_insert.assert_not_awaited()


def test_query_rewrite_falls_back_on_llm_error(monkeypatch):
    monkeypatch.setattr(
        query_rewrite_service,
        "detect_text_language",
        lambda query: "zh",
    )

    def raise_error(*args, **kwargs):
        raise RuntimeError("模拟大模型服务失败")

    monkeypatch.setattr(
        query_rewrite_service,
        "generate_answer",
        raise_error,
    )

    result = query_rewrite_service.rewrite_query(
        history=[{"role": "user", "content": "介绍 LoRA"}],
        query="它有什么优势？",
        target_language="zh",
    )

    assert result == "它有什么优势？"


@pytest.mark.asyncio
async def test_assistant_message_keeps_sources(monkeypatch):
    sources = [{
        "page_number": 4,
        "chunk_index": 29,
        "text": "LoRA freezes pretrained weights.",
    }]

    monkeypatch.setattr(
        message_service,
        "find_conversation_by_id",
        AsyncMock(return_value={"_id": "conversation-1"}),
    )
    mock_insert = AsyncMock()
    mock_update = AsyncMock()
    monkeypatch.setattr(
        message_service, "insert_message", mock_insert
    )
    monkeypatch.setattr(
        message_service,
        "update_conversation_updated_at",
        mock_update,
    )

    result = await message_service.create_message(
        conversation_id="conversation-1",
        role="assistant",
        content="LoRA 会冻结预训练权重。",
        sources=sources,
        rewritten_query="What parameters does LoRA freeze?",
    )

    inserted_message = mock_insert.await_args.args[0]

    assert result["sources"] == sources
    assert inserted_message["sources"] == sources
    assert inserted_message["role"] == "assistant"
    mock_update.assert_awaited_once()