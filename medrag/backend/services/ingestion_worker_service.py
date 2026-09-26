import asyncio
import hashlib
import logging
from asyncio import to_thread
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from services import document_ingestion_service, ingestion_job_service
from services.document_service import (
    IndexActivationConflictError,
    get_existing_document_for_user,
)
from services.vector_store_service import (
    delete_index_generation,
    verify_index_generation,
)
from stores import ingestion_job_store
from stores.database import get_database


logger = logging.getLogger(__name__)

ERROR_CODES = {
    "extracting": "PDF_PARSE_FAILED",
    "chunking": "CHUNK_FAILED",
    "embedding": "EMBEDDING_FAILED",
    "indexing": "VECTOR_WRITE_FAILED",
    "validating": "INDEX_VALIDATION_FAILED",
}


def _check_source(file_path: str, expected_hash: str) -> None:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        if b"%PDF-" not in source.read(1024):
            raise ValueError("Stored file is not a PDF")
        source.seek(0)
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        raise ValueError("Stored PDF hash does not match document")


async def _cleanup_inactive_generation(job: dict, generation_id: str) -> None:
    document = await get_existing_document_for_user(
        job["user_id"], job["document_id"]
    )
    if document.get("active_index_generation_id") != generation_id:
        await to_thread(
            delete_index_generation,
            job["user_id"], job["document_id"], generation_id,
        )


async def _activate_and_complete(
    job: dict,
    worker_id: str,
    lease_duration_seconds: int,
    **activation,
) -> None:
    """Fence the owner check, document CAS, and job completion together."""
    now = datetime.now(timezone.utc)
    client = get_database().client
    async with client.start_session() as session:
        async with await session.start_transaction():
            renewed = await ingestion_job_store.renew_lease(
                job["job_id"], worker_id, now,
                now + timedelta(seconds=lease_duration_seconds),
                session=session,
            )
            if not renewed:
                raise document_ingestion_service.IndexOwnershipLostError(
                    "Ingestion lease lost before activation"
                )
            await document_ingestion_service.activate_document_index_generation(
                **activation, session=session,
            )
            completed = await ingestion_job_store.mark_completed(
                job["job_id"], worker_id, now, session=session,
            )
            if not completed:
                raise document_ingestion_service.IndexOwnershipLostError(
                    "Ingestion lease lost before completion"
                )


async def run_once(
    worker_id: str,
    lease_duration_seconds: int = 60,
    heartbeat_interval_seconds: float | None = None,
) -> dict | None:
    """Claim and process at most one persisted ingestion job."""
    interval = heartbeat_interval_seconds or lease_duration_seconds / 3
    if interval <= 0 or interval >= lease_duration_seconds:
        raise ValueError("heartbeat interval must be shorter than the lease")

    job = await ingestion_job_service.claim_next_job(
        worker_id, lease_duration_seconds
    )
    if job is None:
        return None

    job_id = job["job_id"]
    stage = job["stage"]
    generation_id = job.get("index_generation_id")
    lease_lost = asyncio.Event()

    async def check_lease() -> None:
        if lease_lost.is_set() or not await ingestion_job_service.renew_lease(
            job_id, worker_id, lease_duration_seconds
        ):
            lease_lost.set()
            raise document_ingestion_service.IndexOwnershipLostError(
                "Ingestion lease lost"
            )

    async def heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                await check_lease()
        except asyncio.CancelledError:
            raise
        except Exception:
            lease_lost.set()
            logger.exception("ingestion_lease_renew_failed job_id=%s", job_id)

    async def advance(next_stage: str, progress: int) -> None:
        nonlocal stage
        await check_lease()
        changed = await ingestion_job_service.transition_job_state(
            job_id,
            expected_stage=stage,
            new_stage=next_stage,
            worker_id=worker_id,
            progress=progress,
        )
        if not changed:
            lease_lost.set()
            raise document_ingestion_service.IndexOwnershipLostError(
                "Ingestion stage ownership lost"
            )
        stage = next_stage

    async def validate(
        user_id: str, document_id: str,
        new_generation_id: str, chunks: list[dict],
    ) -> None:
        await to_thread(
            verify_index_generation,
            user_id, document_id, new_generation_id, chunks,
        )
        await check_lease()

    async def activate(**kwargs) -> None:
        await check_lease()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        await _activate_and_complete(
            job, worker_id, lease_duration_seconds, **kwargs,
        )

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        document = await get_existing_document_for_user(
            job["user_id"], job["document_id"]
        )

        if generation_id and stage == "validating" and (
            document.get("active_index_generation_id") == generation_id
        ):
            await check_lease()
            if not await ingestion_job_service.mark_completed(job_id, worker_id):
                raise document_ingestion_service.IndexOwnershipLostError(
                    "Cannot finalize recovered job"
                )
            return await ingestion_job_service.get_job(job_id)

        if generation_id:
            await _cleanup_inactive_generation(job, generation_id)
            await check_lease()
            if not await ingestion_job_service.restart_claimed_job(
                job_id, worker_id, stage
            ):
                raise document_ingestion_service.IndexOwnershipLostError(
                    "Cannot restart reclaimed job"
                )
            generation_id = None
            stage = "extracting"

        config = job.get("index_config")
        if not isinstance(config, dict):
            raise ValueError("Ingestion job has no index_config")
        strategy = config["chunk_strategy"]
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        expected_config = document_ingestion_service.build_index_config(
            strategy, chunk_size, chunk_overlap
        )
        if config != expected_config or job["index_fingerprint"] != (
            document_ingestion_service.build_index_fingerprint(config)
        ):
            raise ValueError("Ingestion job index configuration changed")

        await to_thread(_check_source, job["file_path"], document["document_hash"])
        await check_lease()
        generation_id = str(uuid4())
        if not await ingestion_job_service.assign_generation(
            job_id, worker_id, generation_id
        ):
            lease_lost.set()
            raise document_ingestion_service.IndexOwnershipLostError(
                "Cannot assign index generation"
            )

        context = {
            "文件名": document["filename"],
            "save_path": job["file_path"],
            "document_hash": document["document_hash"],
            "document_id": job["document_id"],
            "user_id": job["user_id"],
            "existing_document": True,
            "document_language": document.get("language", "unknown"),
            "index_fingerprint": document.get("index_fingerprint"),
            "active_index_generation_id": document.get(
                "active_index_generation_id"
            ),
            "index_config": document.get("index_config"),
            "indexed_at": document.get("indexed_at"),
            "processing_status": document.get("processing_status"),
        }
        await document_ingestion_service.index_document(
            job["user_id"], None, chunk_size, chunk_overlap,
            strategy=strategy,
            force_reindex=True,
            prepared_context=context,
            generation_id=generation_id,
            on_stage=advance,
            validate_generation=validate,
            before_activate=check_lease,
            activate_generation=activate,
            include_history=False,
        )
    except Exception as exc:
        logger.exception(
            "ingestion_job_failed job_id=%s stage=%s error_type=%s",
            job_id, stage, type(exc).__name__,
        )
        if generation_id:
            try:
                await _cleanup_inactive_generation(job, generation_id)
            except Exception:
                logger.exception(
                    "ingestion_generation_cleanup_failed job_id=%s generation_id=%s",
                    job_id, generation_id,
                )
        if not lease_lost.is_set() and not isinstance(
            exc, document_ingestion_service.IndexOwnershipLostError
        ):
            error_code = (
                "ACTIVE_GENERATION_CONFLICT"
                if isinstance(exc, IndexActivationConflictError)
                else "INDEX_CONFIG_INVALID"
                if isinstance(exc, (KeyError, ValueError)) and stage == "extracting"
                else ERROR_CODES[stage]
            )
            await ingestion_job_service.mark_failed(
                job_id, worker_id, stage, error_code, str(exc),
            )
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    return await ingestion_job_service.get_job(job_id)
