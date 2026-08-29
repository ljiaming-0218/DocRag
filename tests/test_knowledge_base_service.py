from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from services import knowledge_base_service


def make_knowledge_base(
    kb_id: str = "kb-1",
    user_id: str = "user-1",
    name: str = "Research Papers",
) -> dict:
    now = datetime(2026, 8, 28, tzinfo=timezone.utc)
    return {
        "_id": kb_id,
        "user_id": user_id,
        "name": name,
        "description": "RAG papers",
        "created_at": now,
        "updated_at": now,
    }


@pytest.mark.asyncio
async def test_create_knowledge_base_returns_public_contract(monkeypatch):
    monkeypatch.setattr(
        knowledge_base_service,
        "get_existing_user",
        AsyncMock(return_value={"user_id": "user-1"}),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_by_user_and_name",
        AsyncMock(return_value=None),
    )
    insert = AsyncMock()
    monkeypatch.setattr(
        knowledge_base_service,
        "insert_knowledge_base",
        insert,
    )

    result = await knowledge_base_service.create_knowledge_base(
        " user-1 ",
        " Research Papers ",
        " RAG papers ",
    )

    assert result["user_id"] == "user-1"
    assert result["name"] == "Research Papers"
    assert result["description"] == "RAG papers"
    assert "kb_id" in result
    assert "_id" not in result
    insert.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_knowledge_base_rejects_duplicate_name(monkeypatch):
    monkeypatch.setattr(
        knowledge_base_service,
        "get_existing_user",
        AsyncMock(return_value={"user_id": "user-1"}),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_by_user_and_name",
        AsyncMock(return_value=make_knowledge_base()),
    )
    insert = AsyncMock()
    monkeypatch.setattr(
        knowledge_base_service,
        "insert_knowledge_base",
        insert,
    )

    with pytest.raises(ValueError, match="name already exists"):
        await knowledge_base_service.create_knowledge_base(
            "user-1",
            "Research Papers",
        )

    insert.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_knowledge_base_rejects_cross_user_access(monkeypatch):
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_by_id",
        AsyncMock(return_value=make_knowledge_base(user_id="user-1")),
    )

    with pytest.raises(PermissionError, match="does not belong"):
        await knowledge_base_service.get_knowledge_base(
            "user-2",
            "kb-1",
        )


@pytest.mark.asyncio
async def test_list_knowledge_bases_serializes_store_results(monkeypatch):
    monkeypatch.setattr(
        knowledge_base_service,
        "get_existing_user",
        AsyncMock(return_value={"user_id": "user-1"}),
    )
    find_many = AsyncMock(return_value=[make_knowledge_base()])
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_bases_by_user",
        find_many,
    )

    result = await knowledge_base_service.list_knowledge_bases(
        "user-1",
        20,
    )

    assert result[0]["kb_id"] == "kb-1"
    assert "_id" not in result[0]
    find_many.assert_awaited_once_with("user-1", 20)


@pytest.mark.asyncio
async def test_add_document_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_by_id",
        AsyncMock(return_value=make_knowledge_base()),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "get_existing_document_for_user",
        AsyncMock(return_value={
            "_id": "document-1",
            "filename": "paper.pdf",
        }),
    )
    upsert = AsyncMock(side_effect=[True, False])
    touch = AsyncMock()
    monkeypatch.setattr(
        knowledge_base_service,
        "upsert_knowledge_base_document",
        upsert,
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "touch_knowledge_base",
        touch,
    )

    first = await knowledge_base_service.add_document_to_knowledge_base(
        "user-1",
        "kb-1",
        "document-1",
    )
    second = await knowledge_base_service.add_document_to_knowledge_base(
        "user-1",
        "kb-1",
        "document-1",
    )

    assert first["linked"] is True
    assert second["linked"] is False
    assert touch.await_count == 1


@pytest.mark.asyncio
async def test_list_knowledge_base_documents_maps_joined_documents(monkeypatch):
    now = datetime(2026, 8, 28, tzinfo=timezone.utc)
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_by_id",
        AsyncMock(return_value=make_knowledge_base()),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "find_documents_in_knowledge_base",
        AsyncMock(return_value=[{
            "added_at": now,
            "document": {
                "_id": "document-1",
                "user_id": "user-1",
                "filename": "paper.pdf",
                "document_hash": "hash-1",
                "language": "en",
            },
        }]),
    )

    result = await knowledge_base_service.list_knowledge_base_documents(
        "user-1",
        "kb-1",
    )

    assert result == [{
        "document_id": "document-1",
        "user_id": "user-1",
        "kb_id": "kb-1",
        "filename": "paper.pdf",
        "document_hash": "hash-1",
        "language": "en",
        "added_at": now,
    }]


@pytest.mark.asyncio
async def test_remove_document_requires_existing_link(monkeypatch):
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_by_id",
        AsyncMock(return_value=make_knowledge_base()),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "get_existing_document_for_user",
        AsyncMock(return_value={"_id": "document-1"}),
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "delete_knowledge_base_document",
        AsyncMock(return_value=False),
    )

    with pytest.raises(LookupError, match="not linked"):
        await knowledge_base_service.remove_document_from_knowledge_base(
            "user-1",
            "kb-1",
            "document-1",
        )


@pytest.mark.asyncio
async def test_remove_knowledge_base_keeps_documents_and_deletes_links(monkeypatch):
    monkeypatch.setattr(
        knowledge_base_service,
        "find_knowledge_base_by_id",
        AsyncMock(return_value=make_knowledge_base()),
    )
    delete_base = AsyncMock(return_value=True)
    delete_links = AsyncMock(return_value=2)
    monkeypatch.setattr(
        knowledge_base_service,
        "delete_knowledge_base",
        delete_base,
    )
    monkeypatch.setattr(
        knowledge_base_service,
        "delete_knowledge_base_links",
        delete_links,
    )

    result = await knowledge_base_service.remove_knowledge_base(
        "user-1",
        "kb-1",
    )

    assert result == {
        "kb_id": "kb-1",
        "deleted": True,
        "deleted_document_links": 2,
    }
    delete_base.assert_awaited_once_with("kb-1", "user-1")
    delete_links.assert_awaited_once_with("kb-1", "user-1")
