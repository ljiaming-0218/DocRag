from datetime import datetime, timezone
from unittest.mock import ANY, AsyncMock

import pytest

from services import document_service


def make_document(
    document_id: str = "document-1",
    user_id: str = "user-1",
) -> dict:
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    return {
        "_id": document_id,
        "user_id": user_id,
        "filename": "rag-paper.pdf",
        "document_hash": "hash-1",
        "language": "en",
        "created_at": now,
        "updated_at": now,
        "storage_internal_field": "must-not-leak",
    }


@pytest.mark.asyncio
async def test_list_user_documents_returns_response_list(monkeypatch):
    monkeypatch.setattr(
        document_service,
        "get_existing_user",
        AsyncMock(return_value={"user_id": "user-1"}),
    )
    find_documents = AsyncMock(
        return_value=[make_document()],
    )
    monkeypatch.setattr(
        document_service,
        "find_documents_by_user",
        find_documents,
    )

    result = await document_service.list_user_documents(
        " user-1 ",
        limit=20,
    )

    assert result == [
        {
            "document_id": "document-1",
            "user_id": "user-1",
            "filename": "rag-paper.pdf",
            "document_hash": "hash-1",
            "language": "en",
            "created_at": datetime(
                2026,
                7,
                30,
                tzinfo=timezone.utc,
            ),
            "updated_at": datetime(
                2026,
                7,
                30,
                tzinfo=timezone.utc,
            ),
            "processing_status": "pending",
            "processing_stage": "pending",
            "error_stage": None,
            "error_code": None,
            "error_message": None,
        }
    ]
    find_documents.assert_awaited_once_with(
        "user-1",
        20,
    )


@pytest.mark.asyncio
async def test_list_user_documents_returns_empty_list(monkeypatch):
    monkeypatch.setattr(
        document_service,
        "get_existing_user",
        AsyncMock(return_value={"user_id": "user-1"}),
    )
    monkeypatch.setattr(
        document_service,
        "find_documents_by_user",
        AsyncMock(return_value=[]),
    )

    result = await document_service.list_user_documents(
        "user-1",
    )

    assert result == []


@pytest.mark.asyncio
async def test_list_user_documents_checks_user_exists(monkeypatch):
    get_user = AsyncMock(
        side_effect=ValueError("用户不存在"),
    )
    find_documents = AsyncMock()
    monkeypatch.setattr(
        document_service,
        "get_existing_user",
        get_user,
    )
    monkeypatch.setattr(
        document_service,
        "find_documents_by_user",
        find_documents,
    )

    with pytest.raises(ValueError, match="用户不存在"):
        await document_service.list_user_documents(
            "user-404",
        )

    find_documents.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "limit", "message"),
    [
        (" ", 50, "user_id 不能为空"),
        ("user-1", 0, "limit 必须在 1 到 100 之间"),
        ("user-1", 101, "limit 必须在 1 到 100 之间"),
    ],
)
async def test_list_user_documents_validates_input(
    monkeypatch,
    user_id,
    limit,
    message,
):
    get_user = AsyncMock()
    find_documents = AsyncMock()
    monkeypatch.setattr(
        document_service,
        "get_existing_user",
        get_user,
    )
    monkeypatch.setattr(
        document_service,
        "find_documents_by_user",
        find_documents,
    )

    with pytest.raises(ValueError, match=message):
        await document_service.list_user_documents(
            user_id,
            limit,
        )

    get_user.assert_not_awaited()
    find_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_document_index_state_updates_owned_document(monkeypatch):
    update_state = AsyncMock(return_value=True)
    monkeypatch.setattr(
        document_service,
        "update_document_index_state",
        update_state,
    )
    indexed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    index_config = {
        "chunk_strategy": "recursive",
        "index_version": "v1",
    }

    await document_service.set_document_index_state(
        "user-1",
        "document-1",
        "fingerprint-1",
        index_config,
        indexed_at,
    )

    update_state.assert_awaited_once_with(
        "user-1",
        "document-1",
        "fingerprint-1",
        index_config,
        indexed_at,
    )


@pytest.mark.asyncio
async def test_set_document_index_state_rejects_missing_document(monkeypatch):
    monkeypatch.setattr(
        document_service,
        "update_document_index_state",
        AsyncMock(return_value=False),
    )

    with pytest.raises(ValueError, match="文档不存在"):
        await document_service.set_document_index_state(
            "user-1",
            "document-404",
            "fingerprint-1",
            {"index_version": "v1"},
            datetime(2026, 8, 29, tzinfo=timezone.utc),
        )


@pytest.mark.asyncio
async def test_set_document_processing_state_updates_owned_document(
    monkeypatch,
):
    update_state = AsyncMock(return_value=True)
    monkeypatch.setattr(
        document_service,
        "update_document_processing_state",
        update_state,
    )

    await document_service.set_document_processing_state(
        "user-1",
        "document-1",
        "processing",
        "parsing",
    )

    update_state.assert_awaited_once_with(
        "user-1",
        "document-1",
        "processing",
        "parsing",
        ANY,
        processing_started_at=ANY,
        error_stage=None,
        error_code=None,
        error_message=None,
        clear_error=True,
    )


@pytest.mark.asyncio
async def test_set_document_processing_state_requires_failure_details(
    monkeypatch,
):
    update_state = AsyncMock()
    monkeypatch.setattr(
        document_service,
        "update_document_processing_state",
        update_state,
    )

    with pytest.raises(ValueError, match="失败状态必须包含"):
        await document_service.set_document_processing_state(
            "user-1",
            "document-1",
            "failed",
            "embedding",
        )

    update_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_document_processing_state_records_failure(monkeypatch):
    update_state = AsyncMock(return_value=True)
    monkeypatch.setattr(
        document_service,
        "update_document_processing_state",
        update_state,
    )

    await document_service.set_document_processing_state(
        "user-1",
        "document-1",
        "failed",
        "embedding",
        error_stage="embedding",
        error_code="EMBEDDING_FAILED",
        error_message="model unavailable",
    )

    update_state.assert_awaited_once_with(
        "user-1",
        "document-1",
        "failed",
        "embedding",
        ANY,
        processing_started_at=None,
        error_stage="embedding",
        error_code="EMBEDDING_FAILED",
        error_message="model unavailable",
        clear_error=False,
    )


def test_ensure_document_ready_accepts_completed_document():
    document_service.ensure_document_ready({
        "_id": "document-1",
        "processing_status": "completed",
        "processing_stage": "completed",
    })


def test_ensure_document_ready_accepts_legacy_indexed_document():
    document_service.ensure_document_ready({
        "_id": "document-1",
        "index_fingerprint": "fingerprint-1",
    })


def test_ensure_document_ready_keeps_previous_generation_available():
    document_service.ensure_document_ready({
        "_id": "document-1",
        "processing_status": "failed",
        "processing_stage": "indexing",
        "active_index_generation_id": "generation-old",
    })


def test_legacy_index_is_blocked_during_first_versioned_reindex():
    with pytest.raises(document_service.DocumentNotReadyError):
        document_service.ensure_document_ready({
            "_id": "document-1",
            "processing_status": "processing",
            "processing_stage": "indexing",
            "index_fingerprint": "legacy-fingerprint",
        })


@pytest.mark.parametrize(
    ("status", "stage", "message"),
    [
        ("pending", "pending", "索引尚未完成"),
        ("processing", "embedding", "当前阶段: embedding"),
        ("failed", "indexing", "索引失败"),
    ],
)
def test_ensure_document_ready_rejects_unready_document(
    status,
    stage,
    message,
):
    with pytest.raises(
        document_service.DocumentNotReadyError,
        match=message,
    ) as error_info:
        document_service.ensure_document_ready({
            "_id": "document-1",
            "processing_status": status,
            "processing_stage": stage,
        })

    assert error_info.value.document_id == "document-1"
    assert error_info.value.processing_status == status
    assert error_info.value.processing_stage == stage
