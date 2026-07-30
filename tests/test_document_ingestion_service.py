from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from services import document_ingestion_service as service


def make_context(
    *,
    existing_document: bool = False,
    document_language: str = "unknown",
) -> dict:
    return {
        "文件名": "paper.pdf",
        "save_path": "D:/tmp/paper.pdf",
        "document_hash": "hash-1",
        "document_id": "document-1",
        "user_id": "user-1",
        "existing_document": existing_document,
        "document_language": document_language,
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
        parse_document_pages=Mock(return_value=pages),
        detect_document_language=Mock(
            return_value=document_language,
        ),
        set_document_language=AsyncMock(),
        split_pages=Mock(return_value=chunks),
        embed_chunks=Mock(return_value=embedded_chunks),
        save_chunks=Mock(return_value=len(embedded_chunks)),
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
    dependencies.split_pages.assert_called_once_with(pages, 500, 50)
    dependencies.embed_chunks.assert_called_once_with(chunks)
    dependencies.save_chunks.assert_called_once_with(embedded_chunks)
    assert result["document_id"] == "document-1"
    assert result["总页数"] == 3
    assert result["总块数"] == 2
    assert result["成功保存块数"] == 2
    assert result["ocr_page_count"] == 1
    assert result["ocr_required_page_count"] == 1
    assert result["empty_page_count"] == 1
    assert result["document_language"] == "en"


@pytest.mark.asyncio
async def test_existing_index_skips_reprocessing(monkeypatch):
    history = [{"conversation_id": "conversation-1"}]
    dependencies = install_index_dependencies(
        monkeypatch,
        context=make_context(
            existing_document=True,
            document_language="en",
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


@pytest.mark.asyncio
async def test_existing_index_backfills_unknown_language(monkeypatch):
    pages = [{"文本": "中文内容", "提取方式": "text"}]
    dependencies = install_index_dependencies(
        monkeypatch,
        context=make_context(
            existing_document=True,
            document_language="unknown",
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
    dependencies = install_index_dependencies(
        monkeypatch,
        context=make_context(
            existing_document=True,
            document_language="unknown",
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


@pytest.mark.asyncio
async def test_build_pdf_pages_validates_user_before_upload(
    monkeypatch,
):
    get_existing_user = AsyncMock(
        side_effect=ValueError("用户不存在"),
    )
    upload_pdf = AsyncMock()
    monkeypatch.setattr(
        service,
        "get_existing_user",
        get_existing_user,
    )
    monkeypatch.setattr(service, "upload_pdf", upload_pdf)

    with pytest.raises(ValueError, match="用户不存在"):
        await service.build_pdf_pages(
            "missing-user",
            object(),
        )

    upload_pdf.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_pdf_pages_adds_document_metadata(monkeypatch):
    pages = [{"页码": 1, "文本": "content"}]
    monkeypatch.setattr(
        service,
        "get_existing_user",
        AsyncMock(return_value={"user_id": "user-1"}),
    )
    monkeypatch.setattr(
        service,
        "upload_pdf",
        AsyncMock(return_value={
            "文件名": "paper.pdf",
            "保存路径": "D:/tmp/paper.pdf",
            "document_hash": "hash-1",
        }),
    )
    monkeypatch.setattr(
        service,
        "get_or_create_document",
        AsyncMock(return_value={
            "document_id": "document-1",
            "document_hash": "hash-1",
            "existing_document": False,
        }),
    )
    monkeypatch.setattr(
        service,
        "extract_pdf_pages",
        Mock(return_value=pages),
    )

    result = await service.build_pdf_pages(
        "user-1",
        object(),
    )

    assert result["总页数"] == 1
    assert result["每页内容"][0]["document_id"] == "document-1"
    assert result["每页内容"][0]["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_build_pdf_pages_wraps_parser_failure(monkeypatch):
    monkeypatch.setattr(
        service,
        "get_existing_user",
        AsyncMock(return_value={"user_id": "user-1"}),
    )
    monkeypatch.setattr(
        service,
        "upload_pdf",
        AsyncMock(return_value={
            "文件名": "paper.pdf",
            "保存路径": "D:/tmp/paper.pdf",
            "document_hash": "hash-1",
        }),
    )
    monkeypatch.setattr(
        service,
        "get_or_create_document",
        AsyncMock(return_value={
            "document_id": "document-1",
            "document_hash": "hash-1",
            "existing_document": False,
        }),
    )
    monkeypatch.setattr(
        service,
        "extract_pdf_pages",
        Mock(side_effect=ValueError("损坏的 PDF")),
    )

    with pytest.raises(RuntimeError, match="PDF 解析失败"):
        await service.build_pdf_pages(
            "user-1",
            object(),
        )
