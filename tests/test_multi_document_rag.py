from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import conversation_router
from services import (
    chat_service,
    conversation_service,
    knowledge_base_service,
    search_service,
    vector_store_service,
)
from services.prompt_service import build_context


NOW = datetime(2026, 8, 28, tzinfo=timezone.utc)


def make_knowledge_base() -> dict:
    return {
        "_id": "kb-1",
        "user_id": "user-1",
        "name": "RAG Papers",
        "description": "",
        "created_at": NOW,
        "updated_at": NOW,
    }


@pytest.mark.asyncio
async def test_resolve_knowledge_base_scope_returns_selected_documents(
    monkeypatch,
):
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_by_id",
        AsyncMock(return_value=make_knowledge_base()),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_document_records",
        AsyncMock(return_value=[
            {
                "document": {
                    "_id": "document-1",
                    "user_id": "user-1",
                    "filename": "rag.pdf",
                    "language": "en",
                }
            },
            {
                "document": {
                    "_id": "document-2",
                    "user_id": "user-1",
                    "filename": "lora.pdf",
                    "language": "en",
                }
            },
        ]),
    )

    result = await knowledge_base_service.resolve_knowledge_base_scope(
        "user-1",
        "kb-1",
        [" document-2 ", "document-2"],
    )

    assert result["document_ids"] == ["document-2"]
    assert result["document_filenames"] == {
        "document-2": "lora.pdf",
    }
    assert result["language"] == "en"


@pytest.mark.asyncio
async def test_resolve_knowledge_base_scope_rejects_unlinked_document(
    monkeypatch,
):
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_by_id",
        AsyncMock(return_value=make_knowledge_base()),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_document_records",
        AsyncMock(return_value=[{
            "document": {
                "_id": "document-1",
                "user_id": "user-1",
                "filename": "rag.pdf",
            }
        }]),
    )

    with pytest.raises(PermissionError, match="not available"):
        await knowledge_base_service.resolve_knowledge_base_scope(
            "user-1",
            "kb-1",
            ["document-from-another-user"],
        )


@pytest.mark.asyncio
async def test_create_knowledge_base_conversation_saves_scope(monkeypatch):
    monkeypatch.setattr(
        conversation_service,
        "get_existing_user",
        AsyncMock(return_value={
            "user_id": "user-1",
            "default_user_type": "general",
        }),
    )
    monkeypatch.setattr(
        conversation_service,
        "resolve_knowledge_base_scope",
        AsyncMock(return_value={
            "document_ids": ["document-2"],
        }),
    )
    insert = AsyncMock()
    monkeypatch.setattr(
        conversation_service,
        "insert_conversation",
        insert,
    )

    result = await conversation_service.create_conversation(
        user_id="user-1",
        kb_id="kb-1",
        selected_document_ids=[" document-2 "],
        title="Comparison",
    )

    assert result["document_id"] is None
    assert result["kb_id"] == "kb-1"
    assert result["selected_document_ids"] == ["document-2"]
    inserted = insert.await_args.args[0]
    assert inserted["kb_id"] == "kb-1"


@pytest.mark.asyncio
async def test_create_conversation_rejects_ambiguous_scope(monkeypatch):
    monkeypatch.setattr(
        conversation_service,
        "get_existing_user",
        AsyncMock(return_value={
            "user_id": "user-1",
            "default_user_type": "general",
        }),
    )

    with pytest.raises(ValueError, match="exactly one"):
        await conversation_service.create_conversation(
            user_id="user-1",
            document_id="document-1",
            kb_id="kb-1",
        )


@pytest.mark.asyncio
async def test_prepare_ask_context_uses_multi_document_search(monkeypatch):
    source = {
        "文本块": "RAG retrieves evidence.",
        "距离": 0.2,
        "元数据": {
            "document_id": "document-2",
            "page_number": 2,
            "chunk_index": 3,
        },
    }
    monkeypatch.setattr(
        chat_service,
        "find_conversation_by_id",
        AsyncMock(return_value={
            "_id": "conversation-1",
            "user_id": "user-1",
            "document_id": None,
            "kb_id": "kb-1",
            "selected_document_ids": None,
            "user_type": "general",
        }),
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_knowledge_base_scope",
        AsyncMock(return_value={
            "document_ids": ["document-1", "document-2"],
            "knowledge_base_name": "RAG Papers",
            "document_filenames": {
                "document-1": "rag.pdf",
                "document-2": "lora.pdf",
            },
            "language": "en",
        }),
    )
    legacy_document_lookup = AsyncMock()
    monkeypatch.setattr(
        chat_service,
        "get_existing_document_for_user",
        legacy_document_lookup,
    )
    monkeypatch.setattr(
        chat_service,
        "get_existing_user",
        AsyncMock(return_value={"default_user_type": "general"}),
    )
    monkeypatch.setattr(
        chat_service,
        "get_recent_messages",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        chat_service,
        "create_message",
        AsyncMock(return_value={"message_id": "message-1"}),
    )
    monkeypatch.setattr(
        chat_service,
        "rewrite_query",
        Mock(return_value="What is RAG?"),
    )
    multi_search = Mock(return_value=[source])
    monkeypatch.setattr(
        chat_service,
        "search_relevant_chunks_for_documents",
        multi_search,
    )
    monkeypatch.setattr(
        chat_service,
        "build_rag_prompt",
        Mock(return_value="prompt"),
    )

    result = await chat_service.prepare_ask_context(
        "user-1",
        "conversation-1",
        "What is RAG?",
    )

    assert result["kb_id"] == "kb-1"
    assert result["knowledge_base_name"] == "RAG Papers"
    assert result["document_ids"] == ["document-1", "document-2"]
    assert result["sources"][0]["元数据"]["filename"] == "lora.pdf"
    multi_search.assert_called_once_with(
        "user-1",
        ["document-1", "document-2"],
        "What is RAG?",
        3,
    )
    legacy_document_lookup.assert_not_awaited()


def test_multi_document_vector_query_uses_user_and_document_filter(
    monkeypatch,
):
    class FakeCollection:
        def __init__(self):
            self.query_kwargs = None

        def query(self, **kwargs):
            self.query_kwargs = kwargs
            return {
                "ids": [["chunk-1"]],
                "documents": [["text"]],
                "metadatas": [[{"document_id": "document-1"}]],
                "distances": [[0.1]],
            }

    collection = FakeCollection()
    client = Mock()
    client.get_collection.return_value = collection
    monkeypatch.setattr(
        vector_store_service.chromadb,
        "PersistentClient",
        Mock(return_value=client),
    )

    vector_store_service.query_chunks_by_documents(
        "user-1",
        ["document-1", "document-2", "document-1"],
        [0.1, 0.2],
        15,
    )

    assert collection.query_kwargs["where"] == {
        "$and": [
            {"user_id": "user-1"},
            {
                "document_id": {
                    "$in": ["document-1", "document-2"],
                }
            },
        ]
    }


def test_multi_document_search_reranks_global_candidates(monkeypatch):
    monkeypatch.setattr(
        search_service,
        "get_embedding",
        Mock(return_value=[0.1]),
    )
    query = Mock(return_value={
        "documents": [["from rag", "from lora"]],
        "distances": [[0.3, 0.2]],
        "metadatas": [[
            {
                "document_id": "document-1",
                "page_number": 1,
                "chunk_index": 1,
            },
            {
                "document_id": "document-2",
                "page_number": 1,
                "chunk_index": 1,
            },
        ]],
    })
    rerank = Mock(side_effect=lambda _query, chunks, top_k: chunks[:top_k])
    monkeypatch.setattr(
        search_service,
        "query_chunks_by_documents",
        query,
    )
    monkeypatch.setattr(search_service, "rerank_chunks", rerank)

    result = search_service.search_relevant_chunks_for_documents(
        "user-1",
        ["document-1", "document-2"],
        "compare methods",
        2,
    )

    assert len(result) == 2
    query.assert_called_once_with(
        "user-1",
        ["document-1", "document-2"],
        [0.1],
        10,
    )
    assert rerank.call_count == 1


def test_page_diversity_is_scoped_by_document():
    chunks = [
        {
            "文本块": f"chunk-{index}",
            "元数据": {
                "document_id": document_id,
                "page_number": 1,
            },
        }
        for index, document_id in enumerate(
            ["document-1", "document-1", "document-2"]
        )
    ]

    result = search_service.select_diverse_chunks(
        chunks,
        limit=3,
        max_chunks_per_page=2,
    )

    assert len(result) == 3


def test_prompt_context_identifies_source_document():
    context = build_context([{
        "文本块": "LoRA updates low-rank matrices.",
        "距离": 0.2,
        "元数据": {
            "filename": "lora.pdf",
            "document_id": "document-2",
            "page_number": 4,
            "chunk_index": 8,
        },
    }])

    assert "lora.pdf" in context
    assert "LoRA updates low-rank matrices" in context


def test_create_conversation_api_accepts_knowledge_base_scope(monkeypatch):
    app = FastAPI()
    app.include_router(conversation_router.router)
    create = AsyncMock(return_value={
        "conversation_id": "conversation-1",
        "document_id": None,
        "kb_id": "kb-1",
        "selected_document_ids": ["document-1"],
    })
    monkeypatch.setattr(conversation_router, "create_conversation", create)

    with TestClient(app) as client:
        response = client.post(
            "/conversations",
            json={
                "user_id": "user-1",
                "kb_id": "kb-1",
                "selected_document_ids": ["document-1"],
                "title": "RAG comparison",
            },
        )

    assert response.status_code == 201
    create.assert_awaited_once_with(
        user_id="user-1",
        document_id=None,
        title="RAG comparison",
        kb_id="kb-1",
        selected_document_ids=["document-1"],
    )
