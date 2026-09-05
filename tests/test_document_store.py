from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from stores import document_store


def install_document_collection(monkeypatch):
    collection = SimpleNamespace(
        update_one=AsyncMock(
            return_value=SimpleNamespace(matched_count=1),
        ),
    )
    monkeypatch.setattr(
        document_store,
        "get_database",
        lambda: {"documents": collection},
    )
    return collection


@pytest.mark.asyncio
async def test_processing_state_update_uses_user_and_document_filter(
    monkeypatch,
):
    collection = install_document_collection(monkeypatch)
    now = datetime(2026, 8, 31, tzinfo=timezone.utc)

    updated = await document_store.update_document_processing_state(
        "user-1",
        "document-1",
        "processing",
        "embedding",
        now,
        clear_error=True,
    )

    assert updated is True
    collection.update_one.assert_awaited_once_with(
        {
            "_id": "document-1",
            "user_id": "user-1",
        },
        {
            "$set": {
                "processing_status": "processing",
                "processing_stage": "embedding",
                "updated_at": now,
            },
            "$unset": {
                "error_stage": "",
                "error_code": "",
                "error_message": "",
            },
        },
    )


@pytest.mark.asyncio
async def test_index_state_update_atomically_marks_document_completed(
    monkeypatch,
):
    collection = install_document_collection(monkeypatch)
    indexed_at = datetime(2026, 8, 31, tzinfo=timezone.utc)
    index_config = {"index_version": "v1"}

    updated = await document_store.update_document_index_state(
        "user-1",
        "document-1",
        "fingerprint-1",
        index_config,
        indexed_at,
    )

    assert updated is True
    _, update = collection.update_one.await_args.args
    assert update["$set"] == {
        "index_fingerprint": "fingerprint-1",
        "index_config": index_config,
        "indexed_at": indexed_at,
        "processing_status": "completed",
        "processing_stage": "completed",
        "processed_at": indexed_at,
        "updated_at": indexed_at,
    }
    assert set(update["$unset"]) == {
        "error_stage",
        "error_code",
        "error_message",
    }


@pytest.mark.asyncio
async def test_index_state_update_switches_active_generation(monkeypatch):
    collection = install_document_collection(monkeypatch)
    indexed_at = datetime(2026, 8, 31, tzinfo=timezone.utc)

    updated = await document_store.update_document_index_state(
        "user-1",
        "document-1",
        "fingerprint-1",
        {"index_version": "v1"},
        indexed_at,
        active_index_generation_id="generation-new",
    )

    assert updated is True
    _, update = collection.update_one.await_args.args
    assert update["$set"]["active_index_generation_id"] == (
        "generation-new"
    )
