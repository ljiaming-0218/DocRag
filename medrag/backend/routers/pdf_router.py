import logging
from asyncio import to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile

from services.user_service import get_existing_user
from services.chunk_validation_service import validate_chunk_params
from services.llm_service import generate_answer
from services.prompt_service import build_rag_prompt
from services.text_splitter_service import split_pages

from services.document_ingestion_service import (
    build_pdf_pages,
    index_document,
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
        results = await to_thread(
            search_relevant_chunks,
            user_id,
            document_id,
            query,
            n_results,
        )
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
async def index_pdfs(
    user_id: str,
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    try:
        return await index_document(
            user_id=user_id,
            file=file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except (ValueError, RuntimeError) as error:
        raise_service_error(error)

@router.post("/parse")
async def parse_pdf(user_id: str,file: UploadFile = File(...), ) -> dict:
    try:
        return await build_pdf_pages(user_id, file)
    except (ValueError, RuntimeError) as error:
        raise_service_error(error)

@router.post("/chunks")
async def parse_pdf_chunks(user_id: str, file: UploadFile = File(...), chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
    
    try:
        user = await get_existing_user(user_id)
        validate_chunk_params(chunk_size, chunk_overlap)
        context = await build_pdf_pages(user_id, file)
    except (ValueError, RuntimeError) as error:
        raise_service_error(error)

    chunks = await to_thread(
        split_pages,
        context["每页内容"],
        chunk_size,
        chunk_overlap,
    )

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
        retrieved_chunks = await to_thread(
            search_relevant_chunks,
            user_id,
            document_id,
            query,
            n_results,
        )
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
