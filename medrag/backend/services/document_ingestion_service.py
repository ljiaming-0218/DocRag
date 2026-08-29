import logging
import hashlib
import json
from asyncio import to_thread
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from services.chunk_validation_service import validate_chunk_params
from services.conversation_service import list_conversations
from services.document_service import (
    get_or_create_document,
    set_document_index_state,
    set_document_language,
)
from services.embedding_service import MODEL_NAME, embed_chunks
from services.language_service import detect_document_language
from services.pdf_parser_service import extract_pdf_pages
from services.pdf_storage_service import temporary_pdf, upload_pdf
from services.text_splitter_service import (
    DEFAULT_CHUNK_STRATEGY,
    get_chunk_version,
    resolve_chunk_strategy,
    split_pages,
)
from services.user_service import get_existing_user
from services.vector_store_service import has_chunks, save_chunks


logger = logging.getLogger(__name__)

INDEX_VERSION = "v1"
EMBEDDING_VERSION = "v1"


def build_index_config(
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    return {
        "chunk_strategy": chunk_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_version": get_chunk_version(chunk_strategy),
        "embedding_model": MODEL_NAME,
        "embedding_version": EMBEDDING_VERSION,
        "index_version": INDEX_VERSION,
    }


def build_index_fingerprint(index_config: dict) -> str:
    serialized_config = json.dumps(
        index_config,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_config.encode("utf-8")).hexdigest()


async def preview_pdf_pages(user_id: str, file: UploadFile) -> dict:
    user = await get_existing_user(user_id)
    async with temporary_pdf(file) as stored_file:
        safe_name = stored_file["文件名"]
        save_path = Path(stored_file["保存路径"])
        document_hash = stored_file["document_hash"]

        try:
            pages = await to_thread(extract_pdf_pages, save_path)
        except Exception as exc:
            raise RuntimeError(f"PDF 解析失败: {exc}") from exc

    for page in pages:
        page["document_id"] = None
        page["user_id"] = user_id

    return {
        "文件名": safe_name,
        "总页数": len(pages),
        "每页内容": pages,
        "document_id": None,
        "user_id": user["user_id"],
        "document_hash": document_hash,
        "persisted": False,
    }


async def prepare_document(user_id: str, file: UploadFile) -> dict:
    user = await get_existing_user(user_id)
    stored_file = await upload_pdf(file)
    safe_name = stored_file["文件名"]
    save_path = Path(stored_file["保存路径"])
    document_hash = stored_file["document_hash"]

    document_info = await get_or_create_document(
        user_id,
        safe_name,
        document_hash,
    )

    return {
        "文件名": safe_name,
        "save_path": str(save_path),
        "document_hash": document_hash,
        "document_id": document_info["document_id"],
        "user_id": user["user_id"],
        "existing_document": document_info["existing_document"],
        "document_language": document_info["language"],
        "index_fingerprint": document_info.get("index_fingerprint"),
        "index_config": document_info.get("index_config"),
        "indexed_at": document_info.get("indexed_at"),
    }


async def index_document(
    user_id: str,
    file: UploadFile,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str | None = DEFAULT_CHUNK_STRATEGY,
    force_reindex: bool = False,
) -> dict:
    user = await get_existing_user(user_id)
    validate_chunk_params(chunk_size, chunk_overlap)
    actual_strategy = resolve_chunk_strategy(strategy)
    index_config = build_index_config(
        actual_strategy,
        chunk_size,
        chunk_overlap,
    )
    index_fingerprint = build_index_fingerprint(index_config)
    context = await prepare_document(user_id, file)

    existing_chunks = False
    if context["existing_document"]:
        existing_chunks = await to_thread(
            has_chunks,
            user["user_id"],
            context["document_id"],
        )

    can_reuse_index = (
        existing_chunks
        and not force_reindex
        and context.get("index_fingerprint") == index_fingerprint
    )

    if can_reuse_index:
        document_language = context["document_language"]
        history = await list_conversations(
            user_id=user["user_id"],
            document_id=context["document_id"],
            limit=50,
        )

        if document_language == "unknown":
            try:
                pages = await parse_document_pages(
                    context["save_path"],
                    context,
                )
                document_language = await to_thread(
                    detect_document_language,
                    pages,
                )
                await set_document_language(
                    user["user_id"],
                    context["document_id"],
                    document_language,
                )
            except Exception:
                logger.exception(
                    "document_language_backfill_failed "
                    "user_id=%s document_id=%s",
                    user["user_id"],
                    context["document_id"],
                )
                document_language = "unknown"

        return {
            "message": f"文件 '{context['文件名']}' 已存在，跳过索引。",
            "文件名": context["文件名"],
            "document_id": context["document_id"],
            "user_id": user["user_id"],
            "document_hash": context["document_hash"],
            "existing_document": True,
            "reindexed": False,
            "force_reindex": False,
            "index_fingerprint": index_fingerprint,
            "index_config": index_config,
            "indexed_at": context.get("indexed_at"),
            "document_language": document_language,
            "conversations": history,
        }

    pages = await parse_document_pages(
        context["save_path"],
        context,
    )
    if not pages:
        raise ValueError("PDF 未解析出有效页面")

    document_language = await to_thread(
        detect_document_language,
        pages,
    )
    await set_document_language(
        user["user_id"],
        context["document_id"],
        document_language,
    )

    chunks = await to_thread(
        split_pages,
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=actual_strategy,
    )
    if not chunks:
        raise ValueError("PDF 未生成有效文本块")

    for chunk in chunks:
        chunk.update(index_config)
        chunk["index_fingerprint"] = index_fingerprint

    ocr_page_count = sum(
        page.get("提取方式") == "ocr"
        for page in pages
    )
    ocr_required_page_count = sum(
        page.get("提取方式") == "ocr_required"
        for page in pages
    )
    empty_page_count = sum(
        not page.get("文本", "").strip()
        for page in pages
    )

    embedded_chunks = await to_thread(embed_chunks, chunks)
    saved_count = await to_thread(save_chunks, embedded_chunks)
    indexed_at = datetime.now(timezone.utc)
    await set_document_index_state(
        user["user_id"],
        context["document_id"],
        index_fingerprint,
        index_config,
        indexed_at,
    )

    history = []
    if context["existing_document"]:
        history = await list_conversations(
            user_id=user["user_id"],
            document_id=context["document_id"],
            limit=50,
        )

    reindexed = bool(existing_chunks)

    return {
        "message": (
            f"成功处理文件 '{context['文件名']}'，共 {len(pages)} 页，"
            f"生成 {len(chunks)} 个文本块，成功保存 {saved_count} 个块到向量数据库。"
        ),
        "文件名": context["文件名"],
        "总页数": len(pages),
        "总块数": len(chunks),
        "成功保存块数": saved_count,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_strategy": actual_strategy,
        "document_id": context["document_id"],
        "user_id": user["user_id"],
        "document_hash": context["document_hash"],
        "existing_document": context["existing_document"],
        "reindexed": reindexed,
        "force_reindex": force_reindex,
        "index_fingerprint": index_fingerprint,
        "index_config": index_config,
        "indexed_at": indexed_at,
        "ocr_page_count": ocr_page_count,
        "ocr_required_page_count": ocr_required_page_count,
        "empty_page_count": empty_page_count,
        "document_language": document_language,
        "conversations": history,
    }


async def parse_document_pages(save_path, document_info) -> list[dict]:
    try:
        pages = await to_thread(extract_pdf_pages, save_path)
        for page in pages:
            page["document_id"] = document_info["document_id"]
            page["user_id"] = document_info["user_id"]
    except Exception as exc:
        raise RuntimeError(f"PDF 解析失败: {exc}") from exc

    return pages
