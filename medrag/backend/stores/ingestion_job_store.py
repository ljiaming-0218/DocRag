from datetime import datetime, timezone

from bson.codec_options import CodecOptions
from pymongo import ReturnDocument

from stores.database import get_database


COLLECTION_NAME = "ingestion_jobs"
JOB_CODEC_OPTIONS = CodecOptions(tz_aware=True, tzinfo=timezone.utc)


def _collection():
    return get_database()[COLLECTION_NAME].with_options(
        codec_options=JOB_CODEC_OPTIONS
    )


async def create_job(job: dict) -> None:
    await _collection().insert_one(job)


async def get_job(job_id: str) -> dict | None:
    return await _collection().find_one({"job_id": job_id})


async def get_job_for_user(job_id: str, user_id: str) -> dict | None:
    return await _collection().find_one(
        {"job_id": job_id, "user_id": user_id}
    )


async def find_by_idempotency_key(idempotency_key: str) -> dict | None:
    return await _collection().find_one(
        {"idempotency_key": idempotency_key}
    )


async def has_jobs_for_document(document_id: str) -> bool:
    return await _collection().find_one({"document_id": document_id}) is not None


async def assign_generation(
    job_id: str,
    worker_id: str,
    generation_id: str,
    now: datetime,
) -> bool:
    result = await _collection().update_one(
        {
            "job_id": job_id,
            "status": "running",
            "stage": "extracting",
            "worker_id": worker_id,
            "lease_until": {"$gt": now},
            "index_generation_id": None,
        },
        {"$set": {
            "index_generation_id": generation_id,
            "updated_at": now,
        }},
    )
    return result.matched_count == 1


async def restart_claimed_job(
    job_id: str,
    worker_id: str,
    expected_stage: str,
    now: datetime,
) -> bool:
    result = await _collection().update_one(
        {
            "job_id": job_id,
            "status": "running",
            "stage": expected_stage,
            "worker_id": worker_id,
            "lease_until": {"$gt": now},
        },
        {
            "$set": {
                "stage": "extracting",
                "progress": 0,
                "updated_at": now,
            },
            "$unset": {"index_generation_id": ""},
        },
    )
    return result.matched_count == 1


async def create_ingestion_job_indexes() -> None:
    collection = _collection()
    await collection.create_index(
        "job_id",
        name="job_id_unique_idx",
        unique=True,
    )
    await collection.create_index(
        [("user_id", 1), ("created_at", -1)],
        name="user_created_at_idx",
    )
    await collection.create_index(
        [("status", 1), ("lease_until", 1)],
        name="status_lease_until_idx",
    )
    await collection.create_index(
        [("document_id", 1), ("created_at", -1)],
        name="document_created_at_idx",
    )
    await collection.create_index(
        "idempotency_key",
        name="idempotency_key_unique_idx",
        unique=True,
        partialFilterExpression={"idempotency_key": {"$type": "string"}},
    )


async def transition_job_state(
    job_id: str,
    expected_status: str,
    expected_stage: str,
    new_status: str,
    new_stage: str,
    now: datetime,
    *,
    worker_id: str | None = None,
    progress: int | None = None,
) -> bool:
    query = {
        "job_id": job_id,
        "status": expected_status,
        "stage": expected_stage,
    }
    if progress is not None:
        query["progress"] = {"$lte": progress}
    if expected_status == "running":
        query.update({
            "worker_id": worker_id,
            "lease_until": {"$gt": now},
        })
    fields = {
        "status": new_status,
        "stage": new_stage,
        "updated_at": now,
    }
    if progress is not None:
        fields["progress"] = progress
    result = await _collection().update_one(query, {"$set": fields})
    return result.matched_count == 1


async def claim_next_job(
    worker_id: str,
    now: datetime,
    lease_until: datetime,
) -> dict | None:
    collection = _collection()
    expired = await collection.find_one_and_update(
        {
            "status": "running",
            "$or": [
                {
                    "lease_until": {"$lte": now},
                    "stage": {"$in": [
                        "extracting", "chunking", "embedding",
                        "indexing", "validating",
                    ]},
                },
                {
                    "lease_until": {"$exists": False},
                    "stage": {"$in": [
                        "extracting", "chunking", "embedding",
                        "indexing", "validating",
                    ]},
                },
            ],
        },
        {"$set": {
            "worker_id": worker_id,
            "lease_until": lease_until,
            "updated_at": now,
        }},
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )
    if expired is not None:
        return expired

    return await collection.find_one_and_update(
        {"status": "queued", "stage": "queued"},
        {"$set": {
            "status": "running",
            "stage": "extracting",
            "worker_id": worker_id,
            "lease_until": lease_until,
            "updated_at": now,
        }},
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )


async def renew_lease(
    job_id: str,
    worker_id: str,
    now: datetime,
    lease_until: datetime,
    *,
    session=None,
) -> bool:
    result = await _collection().update_one(
        {
            "job_id": job_id,
            "status": "running",
            "worker_id": worker_id,
            "lease_until": {"$gt": now},
        },
        {"$set": {"lease_until": lease_until, "updated_at": now}},
        **({"session": session} if session is not None else {}),
    )
    return result.matched_count == 1


async def mark_failed(
    job_id: str,
    worker_id: str,
    expected_stage: str,
    now: datetime,
    error_code: str,
    error_message: str,
) -> bool:
    result = await _collection().update_one(
        {
            "job_id": job_id,
            "status": "running",
            "stage": expected_stage,
            "worker_id": worker_id,
            "lease_until": {"$gt": now},
        },
        {
            "$set": {
                "status": "failed",
                "error_code": error_code,
                "error_message": error_message[:2000],
                "updated_at": now,
                "finished_at": now,
            },
            "$unset": {"worker_id": "", "lease_until": ""},
        },
    )
    return result.matched_count == 1


async def mark_completed(
    job_id: str,
    worker_id: str,
    now: datetime,
    *,
    session=None,
) -> bool:
    result = await _collection().update_one(
        {
            "job_id": job_id,
            "status": "running",
            "stage": "validating",
            "worker_id": worker_id,
            "lease_until": {"$gt": now},
        },
        {
            "$set": {
                "status": "completed",
                "stage": "completed",
                "progress": 100,
                "updated_at": now,
                "finished_at": now,
            },
            "$unset": {
                "worker_id": "",
                "lease_until": "",
                "error_code": "",
                "error_message": "",
            },
        },
        **({"session": session} if session is not None else {}),
    )
    return result.matched_count == 1


async def reset_for_retry(
    job_id: str,
    expected_stage: str,
    now: datetime,
) -> bool:
    result = await _collection().update_one(
        {
            "job_id": job_id,
            "status": "failed",
            "stage": expected_stage,
            "$expr": {"$lt": ["$retry_count", "$max_retries"]},
        },
        {
            "$set": {
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "updated_at": now,
            },
            "$inc": {"retry_count": 1},
            "$unset": {
                "worker_id": "",
                "lease_until": "",
                "error_code": "",
                "error_message": "",
                "index_generation_id": "",
                "finished_at": "",
            },
        },
    )
    return result.matched_count == 1
