from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from services import document_ingestion_service as service


def make_context(
    *,
    existing_document: bool = False,
    document_language: str = "unknown",
    index_fingerprint: str | None = None,
    processing_status: str | None = None,
    active_index_generation_id: str | None = None,
) -> dict:
    return {
        "文件名": "paper.pdf",
        "save_path": "D:/tmp/paper.pdf",
        "document_hash": "hash-1",
        "document_id": "document-1",
        "user_id": "user-1",
        "existing_document": existing_document,
        "document_language": document_language,
        "index_fingerprint": index_fingerprint,
        "index_config": None,
        "indexed_at": None,
        "processing_status": processing_status,
        "active_index_generation_id": active_index_generation_id,
    }


def install_index_dependencies(
    monkeypatch,
    *,
    context: dict | None = None,
    pages: list[dict] | None = None,
    chunks: list[dict] | None = None,
    embedded_chunks: list[dict] | None = None,
    existing_chunks: bool = False,
    document_language: str = "en",
    conversations: list[dict] | None = None,
) -> SimpleNamespace:
    context = context or make_context()
    pages = pages if pages is not None else [
        {"文本": "page 1", "提取方式": "text"},
    ]
    chunks = chunks if chunks is not None else [
        {"文本块": "chunk 1"},
    ]
    embedded_chunks = embedded_chunks if embedded_chunks is not None else [
        {"文本块": "chunk 1", "embedding": [0.1, 0.2]},
    ]
    conversations = conversations or []

    dependencies = SimpleNamespace(
        get_existing_user=AsyncMock(
            return_value={"user_id": "user-1"},
        ),
        validate_chunk_params=Mock(),
        prepare_document=AsyncMock(return_value=context),
        has_chunks=Mock(return_value=existing_chunks),
        list_conversations=AsyncMock(return_value=conversations),
        parse_document_pages=AsyncMock(return_value=pages),
        detect_document_language=Mock(
            return_value=document_language,
        ),
        set_document_language=AsyncMock(),
        activate_document_index_generation=AsyncMock(),
        set_document_processing_state=AsyncMock(),
        split_pages=Mock(return_value=chunks),
        embed_chunks=Mock(return_value=embedded_chunks),
        save_chunks=Mock(return_value=len(embedded_chunks)),
        delete_index_generation=Mock(return_value=0),
        invalidate_sparse_indexes=Mock(return_value=0),
    )

    for name, dependency in vars(dependencies).items():
        monkeypatch.setattr(service, name, dependency)

    return dependencies


@pytest.mark.asyncio
async def test_new_document_builds_index(monkeypatch):
    pages = [
        {"文本": "normal text", "提取方式": "text"},
        {"文本": "ocr text", "提取方式": "ocr"},
        {"文本": "", "提取方式": "ocr_required"},
    ]
    chunks = [
        {"文本块": "chunk 1"},
        {"文本块": "chunk 2"},
    ]
    embedded_chunks = [
        {"文本块": "chunk 1", "embedding": [0.1]},
        {"文本块": "chunk 2", "embedding": [0.2]},
    ]
    dependencies = install_index_dependencies(
        monkeypatch,
        pages=pages,
        chunks=chunks,
        embedded_chunks=embedded_chunks,
        document_language="en",
    )

    result = await service.index_document(
        "user-1",
        object(),
        chunk_size=500,
        chunk_overlap=50,
    )

    dependencies.validate_chunk_params.assert_called_once_with(500, 50)
    dependencies.detect_document_language.assert_called_once_with(pages)
    dependencies.set_document_language.assert_awaited_once_with(
        "user-1",
        "document-1",
        "en",
    )
    dependencies.split_pages.assert_called_once_with(
        pages,
        chunk_size=500,
        chunk_overlap=50,
        strategy="fixed",
    )
    dependencies.embed_chunks.assert_called_once_with(chunks)
    dependencies.save_chunks.assert_called_once_with(embedded_chunks)
    dependencies.invalidate_sparse_indexes.assert_called_once_with(
        "user-1",
        "document-1",
    )
    dependencies.activate_document_index_generation.assert_awaited_once()
    assert dependencies.set_document_processing_state.await_args_list == [
        call("user-1", "document-1", "processing", "parsing"),
        call("user-1", "document-1", "processing", "chunking"),
        call("user-1", "document-1", "processing", "embedding"),
        call("user-1", "document-1", "processing", "indexing"),
    ]
    assert result["document_id"] == "document-1"
    assert result["总页数"] == 3
    assert result["总块数"] == 2
    assert result["成功保存块数"] == 2
    assert result["ocr_page_count"] == 1
    assert result["ocr_required_page_count"] == 1
    assert result["empty_page_count"] == 1
    assert result["document_language"] == "en"
    assert result["chunk_strategy"] == "fixed"


@pytest.mark.asyncio
async def test_recursive_strategy_is_forwarded_to_splitter(monkeypatch):
    dependencies = install_index_dependencies(monkeypatch)

    result = await service.index_document(
        "user-1",
        object(),
        chunk_size=500,
        chunk_overlap=50,
        strategy="recursive",
    )

    dependencies.split_pages.assert_called_once_with(
        dependencies.parse_document_pages.return_value,
        chunk_size=500,
        chunk_overlap=50,
        strategy="recursive",
    )
    assert result["chunk_strategy"] == "recursive"


@pytest.mark.asyncio
async def test_existing_index_skips_reprocessing(monkeypatch):
    history = [{"conversation_id": "conversation-1"}]
    index_config = service.build_index_config("fixed", 500, 50)
    dependencies = install_index_dependencies(
        monkeypatch,
        context=make_context(
            existing_document=True,
            document_language="en",
            index_fingerprint=service.build_index_fingerprint(index_config),
        ),
        existing_chunks=True,
        conversations=history,
    )

    result = await service.index_document(
        "user-1",
        object(),
        chunk_size=500,
        chunk_overlap=50,
    )

    assert result["existing_document"] is True
    assert result["reindexed"] is False
    assert result["conversations"] == history
    dependencies.parse_document_pages.assert_not_called()
    dependencies.detect_document_language.assert_not_called()
    dependencies.set_document_language.assert_not_awaited()
    dependencies.split_pages.assert_not_called()
    dependencies.embed_chunks.assert_not_called()
    dependencies.save_chunks.assert_not_called()
    dependencies.invalidate_sparse_indexes.assert_not_called()
    dependencies.delete_index_generation.assert_not_called()
    dependencies.activate_document_index_generation.assert_not_awaited()
    dependencies.set_document_processing_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_index_backfills_unknown_language(monkeypatch):
    pages = [{"文本": "中文内容", "提取方式": "text"}]
    index_config = service.build_index_config("fixed", 500, 50)
    dependencies = install_index_dependencies(
        monkeypatch,
        context=make_context(
            existing_document=True,
            document_language="unknown",
            index_fingerprint=service.build_index_fingerprint(index_config),
        ),
        pages=pages,
        existing_chunks=True,
        document_language="zh",
    )

    result = await service.index_document(
        "user-1",
        object(),
        chunk_size=500,
        chunk_overlap=50,
    )

    dependencies.parse_document_pages.assert_called_once()
    dependencies.detect_document_language.assert_called_once_with(pages)
    dependencies.set_document_language.assert_awaited_once_with(
        "user-1",
        "document-1",
        "zh",
    )
    dependencies.split_pages.assert_not_called()
    assert result["document_language"] == "zh"


@pytest.mark.asyncio
async def test_language_backfill_failure_does_not_break_reuse(
    monkeypatch,
):
    index_config = service.build_index_config("fixed", 500, 50)
    dependencies = install_index_dependencies(
        monkeypatch,
        context=make_context(
            existing_document=True,
            document_language="unknown",
            index_fingerprint=service.build_index_fingerprint(index_config),
        ),
        existing_chunks=True,
    )
    dependencies.parse_document_pages.side_effect = RuntimeError(
        "解析失败",
    )

    result = await service.index_document(
        "user-1",
        object(),
        chunk_size=500,
        chunk_overlap=50,
    )

    assert result["existing_document"] is True
    assert result["document_language"] == "unknown"
    dependencies.set_document_language.assert_not_awaited()
    dependencies.save_chunks.assert_not_called()


def test_index_fingerprint_is_stable_and_config_sensitive():
    fixed_config = service.build_index_config("fixed", 500, 50)
    same_config = service.build_index_config("fixed", 500, 50)
    recursive_config = service.build_index_config("recursive", 500, 50)

    assert service.build_index_fingerprint(fixed_config) == (
        service.build_index_fingerprint(same_config)
    )
    assert service.build_index_fingerprint(fixed_config) != (
        service.build_index_fingerprint(recursive_config)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_fingerprint",
    [None, "stale-fingerprint"],
)
async def test_stale_index_fingerprint_triggers_reindex(
    monkeypatch,
    stored_fingerprint,
):
    dependencies = install_index_dependencies(
        monkeypatch,
        context=make_context(
            existing_document=True,
            document_language="en",
            index_fingerprint=stored_fingerprint,
        ),
        existing_chunks=True,
    )

    result = await service.index_document(
        "user-1",
        object(),
        chunk_size=500,
        chunk_overlap=50,
        strategy="recursive",
    )

    assert result["reindexed"] is True
    assert result["index_config"]["chunk_strategy"] == "recursive"
    dependencies.parse_document_pages.assert_awaited_once()
    dependencies.save_chunks.assert_called_once()
    dependencies.activate_document_index_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_force_reindex_rebuilds_matching_index(monkeypatch):
    index_config = service.build_index_config("fixed", 500, 50)
    dependencies = install_index_dependencies(
        monkeypatch,
        context=make_context(
            existing_document=True,
            document_language="en",
            index_fingerprint=service.build_index_fingerprint(index_config),
        ),
        existing_chunks=True,
    )

    result = await service.index_document(
        "user-1",
        object(),
        chunk_size=500,
        chunk_overlap=50,
        force_reindex=True,
    )

    assert result["reindexed"] is True
    assert result["force_reindex"] is True
    dependencies.parse_document_pages.assert_awaited_once()
    dependencies.save_chunks.assert_called_once()
    dependencies.activate_document_index_generation.assert_awaited_once()

    active_generation = result["active_index_generation_id"]
    assert active_generation
    saved_chunks = dependencies.save_chunks.call_args.args[0]
    assert {
        chunk["index_generation_id"]
        for chunk in saved_chunks
    } == {active_generation}
    assert (
        dependencies.activate_document_index_generation.await_args.kwargs[
            "new_active_generation_id"
        ]
        == active_generation
    )


@pytest.mark.asyncio
async def test_failed_reindex_reuses_previous_active_index(monkeypatch):
    index_config = service.build_index_config("fixed", 500, 50)
    dependencies = install_index_dependencies(
        monkeypatch,
        context=make_context(
            existing_document=True,
            document_language="en",
            index_fingerprint=service.build_index_fingerprint(index_config),
            processing_status="failed",
            active_index_generation_id="generation-old",
        ),
        existing_chunks=True,
    )

    result = await service.index_document(
        "user-1",
        object(),
        chunk_size=500,
        chunk_overlap=50,
    )

    assert result["reindexed"] is False
    dependencies.parse_document_pages.assert_not_awaited()
    dependencies.save_chunks.assert_not_called()
    dependencies.activate_document_index_generation.assert_not_awaited()
    dependencies.set_document_processing_state.assert_awaited_once_with(
        "user-1",
        "document-1",
        "completed",
        "completed",
    )


@pytest.mark.asyncio
async def test_empty_pages_stop_before_language_detection(monkeypatch):
    dependencies = install_index_dependencies(
        monkeypatch,
        pages=[],
    )

    with pytest.raises(ValueError, match="PDF 未解析出有效页面"):
        await service.index_document(
            "user-1",
            object(),
            chunk_size=500,
            chunk_overlap=50,
        )

    dependencies.detect_document_language.assert_not_called()
    dependencies.set_document_language.assert_not_awaited()
    dependencies.split_pages.assert_not_called()
    dependencies.embed_chunks.assert_not_called()
    dependencies.save_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_empty_chunks_stop_before_embedding(monkeypatch):
    dependencies = install_index_dependencies(
        monkeypatch,
        chunks=[],
    )

    with pytest.raises(ValueError, match="PDF 未生成有效文本块"):
        await service.index_document(
            "user-1",
            object(),
            chunk_size=500,
            chunk_overlap=50,
        )

    dependencies.set_document_language.assert_awaited_once()
    dependencies.embed_chunks.assert_not_called()
    dependencies.save_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_chunk_params_stop_before_file_preparation(
    monkeypatch,
):
    dependencies = install_index_dependencies(monkeypatch)
    dependencies.validate_chunk_params.side_effect = ValueError(
        "chunk_overlap 必须小于 chunk_size",
    )

    with pytest.raises(
        ValueError,
        match="chunk_overlap 必须小于 chunk_size",
    ):
        await service.index_document(
            "user-1",
            object(),
            chunk_size=100,
            chunk_overlap=100,
        )

    dependencies.prepare_document.assert_not_awaited()
    dependencies.has_chunks.assert_not_called()
    dependencies.parse_document_pages.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_user_stops_before_file_preparation(monkeypatch):
    dependencies = install_index_dependencies(monkeypatch)
    dependencies.get_existing_user.side_effect = ValueError(
        "用户不存在",
    )

    with pytest.raises(ValueError, match="用户不存在"):
        await service.index_document(
            "missing-user",
            object(),
            chunk_size=500,
            chunk_overlap=50,
        )

    dependencies.validate_chunk_params.assert_not_called()
    dependencies.prepare_document.assert_not_awaited()
    dependencies.has_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_vector_store_failure_is_propagated(monkeypatch):
    dependencies = install_index_dependencies(monkeypatch)
    dependencies.save_chunks.side_effect = RuntimeError(
        "向量数据库写入失败",
    )

    with pytest.raises(RuntimeError, match="向量数据库写入失败"):
        await service.index_document(
            "user-1",
            object(),
            chunk_size=500,
            chunk_overlap=50,
        )

    dependencies.embed_chunks.assert_called_once()
    dependencies.save_chunks.assert_called_once()
    dependencies.invalidate_sparse_indexes.assert_not_called()
    dependencies.delete_index_generation.assert_called_once()
    dependencies.activate_document_index_generation.assert_not_awaited()
    failure_call = dependencies.set_document_processing_state.await_args_list[-1]
    assert failure_call.args == (
        "user-1",
        "document-1",
        "failed",
        "indexing",
    )
    assert failure_call.kwargs["error_stage"] == "indexing"
    assert failure_call.kwargs["error_code"] == "VECTOR_WRITE_FAILED"


@pytest.mark.asyncio
async def test_cache_invalidation_failure_keeps_activated_generation(monkeypatch):
    dependencies = install_index_dependencies(monkeypatch)
    dependencies.invalidate_sparse_indexes.side_effect = RuntimeError(
        "cache unavailable",
    )

    result = await service.index_document(
        "user-1",
        object(),
        chunk_size=500,
        chunk_overlap=50,
    )

    dependencies.activate_document_index_generation.assert_awaited_once()
    dependencies.invalidate_sparse_indexes.assert_called_once_with(
        "user-1",
        "document-1",
    )
    dependencies.delete_index_generation.assert_not_called()
    assert result["active_index_generation_id"]


@pytest.mark.asyncio
async def test_activation_failure_cleans_uncommitted_generation(monkeypatch):
    dependencies = install_index_dependencies(monkeypatch)
    dependencies.activate_document_index_generation.side_effect = RuntimeError(
        "document state update failed",
    )

    with pytest.raises(RuntimeError, match="document state update failed"):
        await service.index_document(
            "user-1",
            object(),
            chunk_size=500,
            chunk_overlap=50,
        )

    dependencies.delete_index_generation.assert_called_once()
    dependencies.invalidate_sparse_indexes.assert_not_called()


@pytest.mark.asyncio
async def test_activation_conflict_cleans_generation_without_marking_failed(
    monkeypatch,
):
    dependencies = install_index_dependencies(
        monkeypatch,
        context=make_context(
            existing_document=True,
            active_index_generation_id="generation-old",
        ),
        existing_chunks=True,
    )
    dependencies.activate_document_index_generation.side_effect = (
        service.IndexActivationConflictError("generation changed")
    )

    with pytest.raises(
        service.IndexActivationConflictError,
        match="generation changed",
    ):
        await service.index_document(
            "user-1",
            object(),
            chunk_size=500,
            chunk_overlap=50,
            force_reindex=True,
        )

    activation_kwargs = (
        dependencies.activate_document_index_generation.await_args.kwargs
    )
    assert activation_kwargs["expected_active_generation_id"] == (
        "generation-old"
    )
    dependencies.delete_index_generation.assert_called_once()
    dependencies.invalidate_sparse_indexes.assert_not_called()
    assert all(
        state_call.args[2] != "failed"
        for state_call in dependencies.set_document_processing_state.await_args_list
    )


@pytest.mark.asyncio
async def test_embedding_failure_marks_document_failed(monkeypatch):
    dependencies = install_index_dependencies(monkeypatch)
    dependencies.embed_chunks.side_effect = RuntimeError(
        "embedding model unavailable",
    )

    with pytest.raises(RuntimeError, match="embedding model unavailable"):
        await service.index_document(
            "user-1",
            object(),
            chunk_size=500,
            chunk_overlap=50,
        )

    failure_call = dependencies.set_document_processing_state.await_args_list[-1]
    assert failure_call.args[-2:] == ("failed", "embedding")
    assert failure_call.kwargs["error_code"] == "EMBEDDING_FAILED"
    dependencies.save_chunks.assert_not_called()
    dependencies.activate_document_index_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_incomplete_vector_write_does_not_finalize_index(monkeypatch):
    embedded_chunks = [
        {"文本块": "chunk 1", "embedding": [0.1]},
        {"文本块": "chunk 2", "embedding": [0.2]},
    ]
    dependencies = install_index_dependencies(
        monkeypatch,
        chunks=[{"文本块": "chunk 1"}, {"文本块": "chunk 2"}],
        embedded_chunks=embedded_chunks,
    )
    dependencies.save_chunks.return_value = 1

    with pytest.raises(RuntimeError, match="向量写入数量不完整"):
        await service.index_document(
            "user-1",
            object(),
            chunk_size=500,
            chunk_overlap=50,
        )

    dependencies.activate_document_index_generation.assert_not_awaited()
    failure_call = dependencies.set_document_processing_state.await_args_list[-1]
    assert failure_call.args[-2:] == ("failed", "indexing")
    assert failure_call.kwargs["error_code"] == "VECTOR_WRITE_FAILED"


@pytest.mark.asyncio
async def test_preview_pdf_pages_validates_user_before_temp_file(
    monkeypatch,
):
    get_existing_user = AsyncMock(
        side_effect=ValueError("用户不存在"),
    )
    temporary_pdf = Mock()
    monkeypatch.setattr(
        service,
        "get_existing_user",
        get_existing_user,
    )
    monkeypatch.setattr(service, "temporary_pdf", temporary_pdf)

    with pytest.raises(ValueError, match="用户不存在"):
        await service.preview_pdf_pages(
            "missing-user",
            object(),
        )

    temporary_pdf.assert_not_called()


@pytest.mark.asyncio
async def test_preview_pdf_pages_does_not_persist_document(monkeypatch):
    pages = [{"页码": 1, "文本": "content"}]

    @asynccontextmanager
    async def fake_temporary_pdf(_file):
        yield {
            "文件名": "paper.pdf",
            "保存路径": "D:/tmp/preview.tmp",
            "document_hash": "hash-1",
        }

    monkeypatch.setattr(
        service,
        "get_existing_user",
        AsyncMock(return_value={"user_id": "user-1"}),
    )
    monkeypatch.setattr(
        service,
        "temporary_pdf",
        fake_temporary_pdf,
    )
    get_or_create_document = AsyncMock()
    monkeypatch.setattr(
        service,
        "get_or_create_document",
        get_or_create_document,
    )
    monkeypatch.setattr(
        service,
        "extract_pdf_pages",
        Mock(return_value=pages),
    )

    result = await service.preview_pdf_pages(
        "user-1",
        object(),
    )

    assert result["总页数"] == 1
    assert result["document_id"] is None
    assert result["persisted"] is False
    assert result["每页内容"][0]["document_id"] is None
    assert result["每页内容"][0]["user_id"] == "user-1"
    get_or_create_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_preview_pdf_pages_wraps_parser_failure(monkeypatch):
    @asynccontextmanager
    async def fake_temporary_pdf(_file):
        yield {
            "文件名": "paper.pdf",
            "保存路径": "D:/tmp/preview.tmp",
            "document_hash": "hash-1",
        }

    monkeypatch.setattr(
        service,
        "get_existing_user",
        AsyncMock(return_value={"user_id": "user-1"}),
    )
    monkeypatch.setattr(
        service,
        "temporary_pdf",
        fake_temporary_pdf,
    )
    monkeypatch.setattr(
        service,
        "extract_pdf_pages",
        Mock(side_effect=ValueError("损坏的 PDF")),
    )

    with pytest.raises(RuntimeError, match="PDF 解析失败"):
        await service.preview_pdf_pages(
            "user-1",
            object(),
        )
