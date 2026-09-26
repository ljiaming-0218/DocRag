"""Validate and persist an upload before enqueueing its indexing job."""

from asyncio import to_thread
from hashlib import sha256
from pathlib import Path

from fastapi import UploadFile

from services.chunk_validation_service import validate_chunk_params
from services.document_ingestion_service import (
    build_index_config,
    build_index_fingerprint,
)
from services.document_service import (
    DocumentIndexUnavailableError,
    get_or_create_document,
)
from services.ingestion_job_service import IdempotencyConflictError, create_job
from services.pdf_storage_service import upload_pdf
from services.text_splitter_service import resolve_chunk_strategy
from services.user_service import get_existing_user
from services.vector_store_service import has_chunks
from stores.document_store import delete_unindexed_document, find_document_by_user_and_hash
from stores.ingestion_job_store import has_jobs_for_document, find_by_idempotency_key


def _key(user_id: str, content_hash: str, fingerprint: str,
         client_key: str | None, force_reindex: bool,
         active_generation: str | None) -> str:
    if client_key is not None:
        if not client_key.strip() or len(client_key) > 256:
            raise ValueError("Idempotency-Key 长度必须为 1 到 256")
        source = f"{user_id}:client:{client_key}"
    else:
        source = f"{user_id}:{content_hash}:{fingerprint}"
        if force_reindex:
            source += f":force:{active_generation or 'none'}"
    return sha256(source.encode("utf-8")).hexdigest()


async def submit_index(
    user_id: str, file: UploadFile, chunk_size: int,
    chunk_overlap: int, strategy: str | None,
    force_reindex: bool, idempotency_key: str | None,
) -> tuple[dict, int]:
    validate_chunk_params(chunk_size, chunk_overlap)
    actual_strategy = resolve_chunk_strategy(strategy)
    config = build_index_config(actual_strategy, chunk_size, chunk_overlap)
    fingerprint = build_index_fingerprint(config)
    await get_existing_user(user_id)
    if idempotency_key is not None and (
        not idempotency_key.strip() or len(idempotency_key) > 256
    ):
        raise ValueError("Idempotency-Key 长度必须为 1 到 256")

    stored = await upload_pdf(file)
    path = Path(stored["保存路径"])
    content_hash = stored["document_hash"]
    existing_before = await find_document_by_user_and_hash(user_id, content_hash)
    document_info = await get_or_create_document(
        user_id, stored["文件名"], content_hash,
    )
    document_id = document_info["document_id"]
    active = document_info.get("active_index_generation_id")
    key = _key(user_id, content_hash, fingerprint, idempotency_key,
               force_reindex, active)
    try:
        previous = await find_by_idempotency_key(key)
        if previous is not None:
            if (previous["user_id"] != user_id
                    or previous["document_id"] != document_id
                    or previous["index_fingerprint"] != fingerprint
                    or previous.get("index_config") != config):
                raise IdempotencyConflictError("Idempotency-Key 已用于其他索引请求")
            if previous["status"] == "completed" and (
                not active
                or document_info.get("index_fingerprint") != fingerprint
                or not await to_thread(has_chunks, user_id, document_id, active)
            ):
                raise DocumentIndexUnavailableError(document_id)
            return {
                "job_id": previous["job_id"], "document_id": document_id,
                "status": previous["status"], "stage": previous["stage"],
                "reused": False,
            }, 200 if previous["status"] == "completed" else 202
        # A matching active generation is usable without another index write.
        if (not force_reindex and active
                and document_info.get("index_fingerprint") == fingerprint
                and await to_thread(has_chunks, user_id, document_id, active)):
            return {
                "job_id": None, "document_id": document_id,
                "status": "completed", "stage": "completed",
                "reused": True,
            }, 200

        job, _ = await create_job(
            user_id=user_id, document_id=document_id,
            file_path=str(path), idempotency_key=key,
            index_fingerprint=fingerprint, index_config=config,
        )
        return {
            "job_id": job["job_id"], "document_id": document_id,
            "status": job["status"], "stage": job["stage"],
            "reused": False,
        }, 200 if job["status"] == "completed" else 202
    except Exception:
        if (existing_before is None
                and not document_info["existing_document"]
                and not await has_jobs_for_document(document_id)):
            removed = await delete_unindexed_document(user_id, document_id)
            if removed and stored.get("created_new_file"):
                await to_thread(path.unlink, missing_ok=True)
        raise
