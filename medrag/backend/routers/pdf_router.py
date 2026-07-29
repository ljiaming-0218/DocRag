import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.conversation_service import list_conversations
from services.language_service import detect_document_language
from services.document_service import set_document_language
from services.user_service import get_existing_user
from services.chunk_validation_service import validate_chunk_params
from services.llm_service import generate_answer
from services.prompt_service import build_rag_prompt
from services.text_splitter_service import split_pages
from services.embedding_service import embed_chunks
from services.vector_store_service import has_chunks, save_chunks
from services.document_ingestion_service import (
    build_pdf_pages,
    parse_document_pages,
    prepare_document,
)
from services.search_service import search_relevant_chunks

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/pdf",
    tags=["pdf"],
)


def raise_service_error(error: Exception) -> None:
    status_code = 400 if isinstance(error, ValueError) else 500
    raise HTTPException(status_code=status_code, detail=str(error)) from error


@router.post("/search")
async def search_pdf(user_id: str, document_id: str, query: str, n_results: int = 3) -> dict:
    try:
        results = search_relevant_chunks(user_id, document_id, query, n_results)
    except (ValueError, RuntimeError) as e:
        raise_service_error(e)

    return {
        "查询": query,
        "检索到的相关内容数": len(results),
        "相关内容": results,
        "query": query,
        "sources_count": len(results),
        "sources": results
    }

@router.post("/index")
async def index_pdfs(user_id: str, file: UploadFile = File(...), chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
    try:
        user = await get_existing_user(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    

    try:
        validate_chunk_params(chunk_size, chunk_overlap)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    context = await prepare_document(user_id, file)

    reindexed = context["existing_document"]

    existing_chunks = (
        context["existing_document"]
        and has_chunks(
            user["user_id"],
            context["document_id"],
        )
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
                pages = parse_document_pages(
                    context["save_path"],
                    context,
                )

                document_language = detect_document_language(pages)

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

    pages = parse_document_pages(context["save_path"], context)
    document_language = detect_document_language(pages)

    await set_document_language(
        user["user_id"],
        context["document_id"],
        document_language,
    )
    chunks = split_pages(pages, chunk_size, chunk_overlap)

    ocr_page_count = sum(page.get("提取方式") == "ocr" for page in pages)
    ocr_required_page_count = sum(page.get("提取方式") == "ocr_required" for page in pages)
    empty_page_count = sum(not page.get("文本", "").strip() for page in pages)


    embedded_chunks = embed_chunks(chunks)
    try:
        saved_count = save_chunks(embedded_chunks)
    except RuntimeError as e:
        raise_service_error(e)
    
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

@router.post("/parse")
async def parse_pdf(user_id: str,file: UploadFile = File(...), ) -> dict:
    return await build_pdf_pages(user_id, file)

@router.post("/chunks")
async def parse_pdf_chunks(user_id: str, file: UploadFile = File(...), chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
    
    try:
        user = await get_existing_user(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        validate_chunk_params(chunk_size, chunk_overlap)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    context = await build_pdf_pages(user_id, file)  # 调用上传函数，确保文件已保存

    chunks = split_pages(context["每页内容"], chunk_size, chunk_overlap)

    return {
        "文件名": context["文件名"],
        "总页数": context["总页数"],
        "每页内容块": chunks,
        "document_id": context["document_id"],
        "user_id": user["user_id"],
    }

@router.post("/prompt-preview")
async def preview_rag_prompt(user_id: str, document_id: str, query: str, n_results: int = 3) -> dict:
    try:
        retrieved_chunks = search_relevant_chunks(user_id, document_id, query, n_results)
    except (ValueError, RuntimeError) as e:
        raise_service_error(e)

    prompt = build_rag_prompt(query, retrieved_chunks)

    return {
        "查询": query,
        "生成的提示": prompt,
        "检索到的相关内容数": len(retrieved_chunks),
        "相关内容": retrieved_chunks,
        "query": query,
        "prompt": prompt,
        "sources_count": len(retrieved_chunks),
        "sources": retrieved_chunks
    }

@router.post("/answer")
def answer_pdf_question(user_id:str, document_id: str, query: str, n_results: int = 3) -> dict:
    try:
        retrieved_chunks = search_relevant_chunks(user_id, document_id, query, n_results)
    except (ValueError, RuntimeError) as e:
        raise_service_error(e)

    prompt = build_rag_prompt(query, retrieved_chunks)
    answer = generate_answer(prompt)

    return {
        "查询": query,
        "生成的答案": answer,
        "检索到的相关内容数": len(retrieved_chunks),
        "相关内容": retrieved_chunks,
        "query": query,
        "answer": answer,
        "sources_count": len(retrieved_chunks),
        "sources": retrieved_chunks
    }
