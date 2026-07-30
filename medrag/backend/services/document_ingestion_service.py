import logging

from asyncio import to_thread
from fastapi import UploadFile
from services.embedding_service import embed_chunks
from services.language_service import detect_document_language
from services.text_splitter_service import split_pages
from services.chunk_validation_service import validate_chunk_params
from services.conversation_service import list_conversations
from services.vector_store_service import has_chunks, save_chunks
from services.user_service import get_existing_user
from services.pdf_parser_service import extract_pdf_pages
from services.pdf_storage_service import upload_pdf
from services.document_service import get_or_create_document, set_document_language
from pathlib import Path

logger = logging.getLogger(__name__)

async def build_pdf_pages(user_id: str, file: UploadFile) -> dict:

    user = await get_existing_user(user_id)
    file = await upload_pdf(file)  # 调用上传函数，确保文件已保存
    safe_name = file["文件名"]
    save_path = Path(file["保存路径"])
    document_hash = file["document_hash"]



    # Get or create the document
    document_info = await get_or_create_document(user_id, safe_name, document_hash)

    try:
        pages = await to_thread(
            extract_pdf_pages,
            save_path,
        )
        for page in pages:
            page["document_id"] = document_info["document_id"]
            page["user_id"] = user_id
    except Exception as exc:
        raise RuntimeError(f"PDF 解析失败: {exc}") from exc

    return {
        "文件名": safe_name,
        "总页数": len(pages),
        "每页内容": pages,
        "document_id": document_info["document_id"],
        "user_id": user["user_id"],
        "document_hash": document_info["document_hash"],
        "existing_document": document_info["existing_document"],
    }


async def prepare_document(user_id: str, file: UploadFile) -> dict:
    file = await upload_pdf(file)  # 调用上传函数，确保文件已保存
    safe_name = file["文件名"]
    save_path = Path(file["保存路径"])
    document_hash = file["document_hash"]

    user = await get_existing_user(user_id)


    # Get or create the document
    document_info = await get_or_create_document(user_id, safe_name, document_hash)

    return {
        "文件名": safe_name,
        "save_path": str(save_path),
        "document_hash": document_hash,
        "document_id": document_info["document_id"],
        "user_id": user["user_id"],
        "existing_document": document_info["existing_document"],
        "document_language": document_info["language"],
    }


async def index_document(
    user_id: str,
    file: UploadFile,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:

    user = await get_existing_user(user_id)

    validate_chunk_params(chunk_size, chunk_overlap)
    context = await prepare_document(user_id, file)

    reindexed = context["existing_document"]

    existing_chunks = False

    if reindexed:
        existing_chunks = await to_thread(
            has_chunks,
            user["user_id"],
            context["document_id"],
        )
    if existing_chunks:
        document_language = context["document_language"]
        history = await list_conversations(
            user_id=user["user_id"],
            document_id=context["document_id"],
            limit=50,
        )
        if document_language == "unknown":
            try:
                pages = await to_thread(
                    parse_document_pages,
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
            "document_language": document_language,
            "conversations": history,
        }

    pages = await to_thread(
        parse_document_pages,
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
        chunk_size,
        chunk_overlap,
    )


    if not chunks:
        raise ValueError("PDF 未生成有效文本块")

    ocr_page_count = sum(page.get("提取方式") == "ocr" for page in pages)
    ocr_required_page_count = sum(page.get("提取方式") == "ocr_required" for page in pages)
    empty_page_count = sum(not page.get("文本", "").strip() for page in pages)

    embedded_chunks = await to_thread(
        embed_chunks,
        chunks,
    )

    saved_count = await to_thread(
        save_chunks,
        embedded_chunks,
    )


    return {
        "message": f"成功处理文件 '{context['文件名']}'，共 {len(pages)} 页，生成 {len(chunks)} 个文本块，成功保存 {saved_count} 个块到向量数据库。",
        "文件名": context["文件名"],
        "总页数": len(pages),
        "总块数": len(chunks),
        "成功保存块数": saved_count,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_id": context["document_id"],
        "user_id": user["user_id"],
        "document_hash": context["document_hash"],
        "existing_document": context["existing_document"],
        "reindexed": reindexed,
        "ocr_page_count": ocr_page_count,
        "ocr_required_page_count": ocr_required_page_count,
        "empty_page_count": empty_page_count,
        "document_language": document_language,
        "conversations": [],
    }

async def parse_document_pages(save_path, document_info) -> list[dict]:
    try:
        pages = await to_thread(
            extract_pdf_pages,
            save_path,
        )
        for page in pages:
            page["document_id"] = document_info["document_id"]
            page["user_id"] = document_info["user_id"]
    except Exception as exc:
        raise RuntimeError(f"PDF 解析失败: {exc}") from exc
    return pages
