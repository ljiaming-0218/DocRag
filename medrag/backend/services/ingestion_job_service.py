from datetime import datetime, timedelta, timezone
from typing import TypedDict
from uuid import uuid4

from pymongo.errors import DuplicateKeyError

from stores import ingestion_job_store


JOB_STATUSES = {"queued", "running", "failed", "completed"}
JOB_STAGES = {
    "queued", "extracting", "chunking", "embedding",
    "indexing", "validating", "completed",
}
STAGE_TRANSITIONS = {
    "extracting": "chunking",
    "chunking": "embedding",
    "embedding": "indexing",
    "indexing": "validating",
}
ERROR_MESSAGE_LIMIT = 2000


class IngestionJob(TypedDict):
    job_id: str
    user_id: str
    document_id: str
    file_path: str
    idempotency_key: str | None
    index_fingerprint: str
    index_config: dict | None
    index_generation_id: str | None
    status: str
    stage: str
    progress: int
    retry_count: int
    max_retries: int
    worker_id: str | None
    lease_until: datetime | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None


class IdempotencyConflictError(ValueError):
    """An idempotency key was reused for a different indexing request."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_job(
    *,
    user_id: str,
    document_id: str,
    file_path: str,
    index_fingerprint: str,
    idempotency_key: str | None = None,
    index_config: dict | None = None,
    max_retries: int = 3,
    now: datetime | None = None,
) -> IngestionJob:
    required = {
        "user_id": user_id,
        "document_id": document_id,
        "file_path": file_path,
        "index_fingerprint": index_fingerprint,
    }
    if any(not isinstance(value, str) or not value.strip()
           for value in required.values()):
        raise ValueError("user_id, document_id, file_path 和 fingerprint 必填")
    if idempotency_key is not None and not idempotency_key.strip():
        raise ValueError("idempotency_key 不能为空字符串")
    if max_retries < 0:
        raise ValueError("max_retries 不能小于 0")

    if index_config is not None and not isinstance(index_config, dict):
        raise ValueError("index_config must be a dict")

    created_at = now or _now()
    job = {
        "job_id": str(uuid4()),
        "user_id": user_id,
        "document_id": document_id,
        "file_path": file_path,
        "idempotency_key": idempotency_key,
        "index_fingerprint": index_fingerprint,
        "index_config": index_config,
        "index_generation_id": None,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "retry_count": 0,
        "max_retries": max_retries,
        "worker_id": None,
        "lease_until": None,
        "error_code": None,
        "error_message": None,
        "created_at": created_at,
        "updated_at": created_at,
        "finished_at": None,
    }
    return job


def _matches_request(existing: dict, requested: dict) -> bool:
    return all(
        existing.get(key) == requested.get(key)
        for key in (
            "user_id", "document_id", "index_fingerprint", "index_config",
        )
    )


async def create_job(**kwargs) -> tuple[dict, bool]:
    job = build_job(**kwargs)
    key = job.get("idempotency_key")
    if key is not None:
        existing = await ingestion_job_store.find_by_idempotency_key(key)
        if existing is not None:
            if _matches_request(existing, job):
                return existing, False
            raise IdempotencyConflictError(
                "idempotency_key 已用于不同的索引请求"
            )
    try:
        await ingestion_job_store.create_job(job)
    except DuplicateKeyError:
        if key is None:
            raise
        existing = await ingestion_job_store.find_by_idempotency_key(key)
        if existing is not None and _matches_request(existing, job):
            return existing, False
        raise IdempotencyConflictError(
            "idempotency_key 已被并发请求占用"
        )
    return job, True


async def get_job(job_id: str) -> dict | None:
    return await ingestion_job_store.get_job(job_id)


async def get_job_for_user(job_id: str, user_id: str) -> dict | None:
    return await ingestion_job_store.get_job_for_user(job_id, user_id)


async def assign_generation(
    job_id: str,
    worker_id: str,
    generation_id: str,
) -> bool:
    if not generation_id.strip():
        raise ValueError("generation_id is required")
    return await ingestion_job_store.assign_generation(
        job_id, worker_id, generation_id, _now()
    )


async def restart_claimed_job(
    job_id: str,
    worker_id: str,
    expected_stage: str,
) -> bool:
    return await ingestion_job_store.restart_claimed_job(
        job_id, worker_id, expected_stage, _now()
    )


async def transition_job_state(
    job_id: str,
    *,
    expected_stage: str,
    new_stage: str,
    worker_id: str,
    progress: int,
    now: datetime | None = None,
) -> bool:
    if expected_stage not in JOB_STAGES or new_stage not in JOB_STAGES:
        raise ValueError("非法 ingestion job stage")
    if expected_stage == "validating" or (
        new_stage != expected_stage
        and STAGE_TRANSITIONS.get(expected_stage) != new_stage
    ):
        raise ValueError("非法 ingestion job 阶段迁移")
    if not 0 <= progress < 100:
        raise ValueError("处理中 progress 必须在 0 到 99 之间")
    return await ingestion_job_store.transition_job_state(
        job_id,
        "running",
        expected_stage,
        "running",
        new_stage,
        now or _now(),
        worker_id=worker_id,
        progress=progress,
    )


async def claim_next_job(
    worker_id: str,
    lease_duration_seconds: int = 60,
    now: datetime | None = None,
) -> dict | None:
    if not worker_id.strip() or lease_duration_seconds <= 0:
        raise ValueError("worker_id 与正数 lease_duration_seconds 必填")
    current = now or _now()
    return await ingestion_job_store.claim_next_job(
        worker_id,
        current,
        current + timedelta(seconds=lease_duration_seconds),
    )


async def renew_lease(
    job_id: str,
    worker_id: str,
    lease_duration_seconds: int = 60,
    now: datetime | None = None,
) -> bool:
    if lease_duration_seconds <= 0:
        raise ValueError("lease_duration_seconds 必须大于 0")
    current = now or _now()
    return await ingestion_job_store.renew_lease(
        job_id,
        worker_id,
        current,
        current + timedelta(seconds=lease_duration_seconds),
    )


async def mark_failed(
    job_id: str,
    worker_id: str,
    expected_stage: str,
    error_code: str,
    error_message: str,
    now: datetime | None = None,
) -> bool:
    if not error_code.strip():
        raise ValueError("error_code 不能为空")
    if expected_stage not in JOB_STAGES or expected_stage in {"queued", "completed"}:
        raise ValueError("失败阶段无效")
    return await ingestion_job_store.mark_failed(
        job_id,
        worker_id,
        expected_stage,
        now or _now(),
        error_code,
        (error_message or "")[:ERROR_MESSAGE_LIMIT],
    )


async def mark_completed(
    job_id: str,
    worker_id: str,
    now: datetime | None = None,
) -> bool:
    return await ingestion_job_store.mark_completed(
        job_id,
        worker_id,
        now or _now(),
    )


async def reset_for_retry(
    job_id: str,
    expected_stage: str,
    now: datetime | None = None,
) -> bool:
    if expected_stage not in JOB_STAGES or expected_stage in {"queued", "completed"}:
        raise ValueError("重试时的失败 stage 无效")
    return await ingestion_job_store.reset_for_retry(
        job_id,
        expected_stage,
        now or _now(),
    )
