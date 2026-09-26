from typing import Annotated

from fastapi import APIRouter, Depends

from api_errors import APIError
from dependencies.auth import get_current_user_id
from services.ingestion_job_service import get_job_for_user, reset_for_retry


router = APIRouter(prefix="/ingestion-jobs", tags=["ingestion-jobs"])
PUBLIC_FIELDS = (
    "job_id", "document_id", "status", "stage", "progress",
    "retry_count", "max_retries", "error_code", "error_message",
    "created_at", "updated_at", "finished_at",
)


def public_job(job: dict) -> dict:
    return {field: job.get(field) for field in PUBLIC_FIELDS}


async def owned_job(job_id: str, user_id: str) -> dict:
    job = await get_job_for_user(job_id, user_id)
    if job is None:
        raise APIError(404, "INGESTION_JOB_NOT_FOUND", "任务不存在")
    return job


@router.get("/{job_id}")
async def get_ingestion_job(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    return public_job(await owned_job(job_id, user_id))


@router.post("/{job_id}/retry")
async def retry_ingestion_job(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    job = await owned_job(job_id, user_id)
    if job["status"] != "failed" or job["retry_count"] >= job["max_retries"]:
        raise APIError(409, "INGESTION_JOB_NOT_RETRYABLE", "任务不可重试")
    if not await reset_for_retry(job_id, job["stage"]):
        raise APIError(409, "INGESTION_JOB_STATE_CHANGED", "任务状态已变化")
    return public_job(await owned_job(job_id, user_id))
