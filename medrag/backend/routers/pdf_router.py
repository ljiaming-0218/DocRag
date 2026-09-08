import logging
from asyncio import to_thread
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from api_errors import APIError
from dependencies.auth import get_current_user_id, require_matching_user
from services.chunk_validation_service import validate_chunk_params
from services.llm_service import LLMServiceError, generate_answer
from services.document_service import (
    DocumentNotReadyError,
    IndexActivationConflictError,
    ensure_document_ready,
    get_existing_document_for_user,
)
from services.prompt_service import build_rag_prompt
from services.text_splitter_service import (
    DEFAULT_CHUNK_STRATEGY,
    split_pages,
)

from services.document_ingestion_service import (
    index_document,
    preview_pdf_pages,
)
from services.search_service import search_relevant_chunks

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/pdf",
    tags=["pdf"],
)


def raise_pdf_error(error: Exception) -> None:
    if isinstance(error, IndexActivationConflictError):
        raise APIError(
            status_code=409,
            code="INDEX_ACTIVATION_CONFLICT",
            message=str(error),
        ) from error
    if isinstance(error, DocumentNotReadyError):
        raise APIError(
            status_code=409,
            code="DOCUMENT_NOT_READY",
            message=str(error),
        ) from error
    if isinstance(error, ValueError):
        raise APIError(
            status_code=400,
            code="INVALID_PDF_REQUEST",
            message=str(error),
        ) from error

    raise APIError(
        status_code=500,
        code="PDF_OPERATION_FAILED",
        message=str(error),
    ) from error


async def get_active_index_generation(
    user_id: str,
    document_id: str,
) -> str | None:
    document = await get_existing_document_for_user(user_id, document_id)
    ensure_document_ready(document)
    return document.get("active_index_generation_id")


@router.post("/search")
async def search_pdf(
    user_id: str,
    document_id: str,
    query: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    n_results: int = 3,
) -> dict:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        index_generation_id = await get_active_index_generation(
            user_id,
            document_id,
        )
        results = await to_thread(
            search_relevant_chunks,
            user_id,
            document_id,
            query,
            n_results,
            index_generation_id,
        )
    except (ValueError, RuntimeError) as e:
        raise_pdf_error(e)

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
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = DEFAULT_CHUNK_STRATEGY,
    force_reindex: bool = False,
) -> dict:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        return await index_document(
            user_id=user_id,
            file=file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            force_reindex=force_reindex,
        )
    except (ValueError, RuntimeError) as error:
        raise_pdf_error(error)

@router.post("/parse")
async def parse_pdf(
    user_id: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    file: UploadFile = File(...),
) -> dict:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        return await preview_pdf_pages(user_id, file)
    except (ValueError, RuntimeError) as error:
        raise_pdf_error(error)

@router.post("/chunks")
async def parse_pdf_chunks(
    user_id: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        validate_chunk_params(chunk_size, chunk_overlap)
        context = await preview_pdf_pages(user_id, file)
        chunks = await to_thread(
            split_pages,
            context["每页内容"],
            chunk_size,
            chunk_overlap,
        )
    except (ValueError, RuntimeError) as error:
        raise_pdf_error(error)

    return {
        "文件名": context["文件名"],
        "总页数": context["总页数"],
        "每页内容块": chunks,
        "document_id": context["document_id"],
        "user_id": context["user_id"],
        "persisted": False,
    }

@router.post("/prompt-preview")
async def preview_rag_prompt(
    user_id: str,
    document_id: str,
    query: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    n_results: int = 3,
) -> dict:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        index_generation_id = await get_active_index_generation(
            user_id,
            document_id,
        )
        retrieved_chunks = await to_thread(
            search_relevant_chunks,
            user_id,
            document_id,
            query,
            n_results,
            index_generation_id,
        )
        prompt = build_rag_prompt(query, retrieved_chunks)
    except (ValueError, RuntimeError) as e:
        raise_pdf_error(e)

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
async def answer_pdf_question(
    user_id: str,
    document_id: str,
    query: str,
    authenticated_user_id: Annotated[str, Depends(get_current_user_id)],
    n_results: int = 3,
) -> dict:
    try:
        user_id = require_matching_user(authenticated_user_id, user_id)
        index_generation_id = await get_active_index_generation(
            user_id,
            document_id,
        )
        retrieved_chunks = await to_thread(
            search_relevant_chunks,
            user_id,
            document_id,
            query,
            n_results,
            index_generation_id,
        )
        prompt = build_rag_prompt(query, retrieved_chunks)
        answer = await to_thread(generate_answer, prompt)
    except LLMServiceError as error:
        raise APIError(
            status_code=error.status_code,
            code=error.code,
            message=str(error),
        ) from error
    except (ValueError, RuntimeError) as e:
        raise_pdf_error(e)

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
