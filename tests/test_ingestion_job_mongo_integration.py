import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pymongo.errors import ConnectionFailure, DuplicateKeyError

from services import ingestion_job_service as service
from services import document_ingestion_service as pipeline
from services import vector_store_service as vector_store
from services import ingestion_worker_service as worker
from services.document_service import IndexActivationConflictError
from services.document_ingestion_service import IndexOwnershipLostError
from stores import database
from stores import ingestion_job_store as store
from services import ingestion_submission_service as submission
from unittest.mock import AsyncMock


pytestmark = pytest.mark.mongo_integration
COLLECTION_PREFIX = "ingestion_jobs_c2a_test_"


@pytest.fixture
async def mongo_jobs(request, monkeypatch):
    if "mongo_integration" not in request.config.getoption("markexpr"):
        pytest.skip("Run explicitly with -m mongo_integration")

    try:
        await database.connect_database()
    except ConnectionFailure as exc:
        await database.close_database()
        pytest.skip(f"MongoDB unavailable: {type(exc).__name__}")

    collection_name = f"{COLLECTION_PREFIX}{uuid4().hex}"
    collection = database.get_database()[collection_name]
    monkeypatch.setattr(store, "COLLECTION_NAME", collection_name)
    try:
        await store.create_ingestion_job_indexes()
        yield collection
    finally:
        assert collection_name.startswith(COLLECTION_PREFIX)
        await database.get_database().drop_collection(collection_name)
        await database.close_database()


def job_args(**overrides):
    values = {
        "user_id": f"c2a-user-{uuid4().hex}",
        "document_id": f"c2a-document-{uuid4().hex}",
        "file_path": "c2a-test.pdf",
        "index_fingerprint": "c2a-fingerprint",
    }
    values.update(overrides)
    return values


@pytest.mark.asyncio
async def test_real_mongo_submission_persists_queued_job(
    mongo_jobs, monkeypatch, tmp_path,
):
    file_path = tmp_path / "paper.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    user_id = f"c2a-user-{uuid4().hex}"
    document_id = f"c2a-document-{uuid4().hex}"
    content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    monkeypatch.setattr(submission, "get_existing_user", AsyncMock(
        return_value={"user_id": user_id},
    ))
    monkeypatch.setattr(submission, "upload_pdf", AsyncMock(return_value={
        "保存路径": str(file_path), "文件名": "paper.pdf",
        "document_hash": content_hash, "created_new_file": True,
    }))
    monkeypatch.setattr(submission, "find_document_by_user_and_hash",
                        AsyncMock(return_value=None))
    monkeypatch.setattr(submission, "get_or_create_document", AsyncMock(
        return_value={
            "document_id": document_id, "existing_document": False,
            "active_index_generation_id": None, "index_fingerprint": None,
        },
    ))
    first, first_status = await submission.submit_index(
        user_id, object(), 500, 50, "fixed", False, None,
    )
    second, second_status = await submission.submit_index(
        user_id, object(), 500, 50, "fixed", False, None,
    )
    persisted = await service.get_job(first["job_id"])
    assert first_status == second_status == 202
    assert first["job_id"] == second["job_id"]
    assert persisted["status"] == "queued"
    assert persisted["file_path"] == str(file_path)
    assert await mongo_jobs.count_documents({}) == 1


async def compete(left, right):
    ready = asyncio.Event()
    start = asyncio.Event()
    waiting = 0

    async def run(action):
        nonlocal waiting
        waiting += 1
        if waiting == 2:
            ready.set()
        await start.wait()
        return await action()

    tasks = [asyncio.create_task(run(action)) for action in (left, right)]
    await ready.wait()
    start.set()
    return await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_real_mongo_persists_queued_job_with_utc_dates(mongo_jobs):
    job, created = await service.create_job(**job_args())
    saved = await service.get_job(job["job_id"])

    assert created is True
    assert saved["status"] == saved["stage"] == "queued"
    assert saved["progress"] == 0
    assert saved["job_id"] == job["job_id"]
    assert saved["created_at"].tzinfo is not None
    assert saved["created_at"].utcoffset() == timedelta(0)
    assert saved["updated_at"].tzinfo is not None


@pytest.mark.asyncio
async def test_real_mongo_partial_unique_idempotency(mongo_jobs):
    key = f"c2a-key-{uuid4().hex}"
    await store.create_job(service.build_job(**job_args(idempotency_key=key)))
    with pytest.raises(DuplicateKeyError):
        await store.create_job(service.build_job(**job_args(idempotency_key=key)))

    for _ in range(2):
        await store.create_job(service.build_job(**job_args()))
        missing = service.build_job(**job_args())
        del missing["idempotency_key"]
        await store.create_job(missing)
    assert await mongo_jobs.count_documents({}) == 5


@pytest.mark.asyncio
async def test_real_mongo_claim_renew_and_reclaim(mongo_jobs):
    job, _ = await service.create_job(**job_args())
    now = datetime.now(timezone.utc)
    claims = await compete(
        lambda: service.claim_next_job("worker-a", now=now),
        lambda: service.claim_next_job("worker-b", now=now),
    )
    winner = next(item for item in claims if item is not None)
    assert sum(item is not None for item in claims) == 1
    saved = await service.get_job(job["job_id"])
    assert saved["worker_id"] == winner["worker_id"]
    assert abs(
        (saved["lease_until"] - (now + timedelta(seconds=60))).total_seconds()
    ) < 0.001
    assert saved["lease_until"].tzinfo is not None

    owner = winner["worker_id"]
    other = "worker-b" if owner == "worker-a" else "worker-a"
    assert await service.renew_lease(job["job_id"], owner, now=now)
    assert not await service.renew_lease(job["job_id"], other, now=now)

    expired_at = now + timedelta(seconds=121)
    reclaimed = await service.claim_next_job("worker-new", now=expired_at)
    assert reclaimed["job_id"] == job["job_id"]
    assert reclaimed["worker_id"] == "worker-new"
    assert not await service.renew_lease(job["job_id"], owner, now=expired_at)


@pytest.mark.asyncio
async def test_real_mongo_stage_transition_is_cas(mongo_jobs):
    job, _ = await service.create_job(**job_args())
    now = datetime.now(timezone.utc)
    await service.claim_next_job("worker-a", now=now)

    results = await compete(
        lambda: service.transition_job_state(
            job["job_id"], expected_stage="extracting", new_stage="chunking",
            worker_id="worker-a", progress=20, now=now,
        ),
        lambda: service.transition_job_state(
            job["job_id"], expected_stage="extracting", new_stage="chunking",
            worker_id="worker-a", progress=20, now=now,
        ),
    )
    assert sorted(results) == [False, True]
    saved = await service.get_job(job["job_id"])
    assert saved["stage"] == "chunking"
    assert saved["progress"] == 20


@pytest.mark.asyncio
async def test_real_mongo_retry_resets_failed_generation(mongo_jobs):
    job, _ = await service.create_job(**job_args(max_retries=1))
    now = datetime.now(timezone.utc)
    await service.claim_next_job("worker-a", now=now)
    await mongo_jobs.update_one(
        {"job_id": job["job_id"]},
        {"$set": {"index_generation_id": "failed-generation"}},
    )
    assert await service.mark_failed(
        job["job_id"], "worker-a", "extracting", "PARSE_FAILED", "failed", now=now,
    )
    assert await service.reset_for_retry(
        job["job_id"], expected_stage="extracting", now=now,
    )
    saved = await service.get_job(job["job_id"])
    assert saved["retry_count"] == 1
    assert saved["status"] == saved["stage"] == "queued"
    for field in (
        "index_generation_id", "worker_id", "lease_until",
        "error_code", "error_message", "finished_at",
    ):
        assert field not in saved

    await service.claim_next_job("worker-b", now=now)
    assert await service.mark_failed(
        job["job_id"], "worker-b", "extracting", "PARSE_FAILED", "again", now=now,
    )
    assert not await service.reset_for_retry(
        job["job_id"], expected_stage="extracting", now=now,
    )


@pytest.mark.asyncio
async def test_real_mongo_index_metadata(mongo_jobs):
    indexes = {item["name"]: item async for item in await mongo_jobs.list_indexes()}
    assert indexes["job_id_unique_idx"]["key"] == {"job_id": 1}
    assert indexes["job_id_unique_idx"]["unique"] is True
    assert indexes["user_created_at_idx"]["key"] == {
        "user_id": 1, "created_at": -1,
    }
    assert indexes["document_created_at_idx"]["key"] == {
        "document_id": 1, "created_at": -1,
    }
    assert indexes["status_lease_until_idx"]["key"] == {
        "status": 1, "lease_until": 1,
    }
    assert indexes["idempotency_key_unique_idx"]["unique"] is True
    assert indexes["idempotency_key_unique_idx"]["partialFilterExpression"] == {
        "idempotency_key": {"$type": "string"},
    }


@pytest.mark.asyncio
async def test_real_mongo_activation_transaction_fences_owner_and_cas(mongo_jobs):
    user_id = f"c2b-user-{uuid4().hex}"
    document_id = f"c2b-document-{uuid4().hex}"
    documents = database.get_database()["documents"]
    await documents.insert_one({
        "_id": document_id,
        "user_id": user_id,
        "document_hash": f"c2b-hash-{uuid4().hex}",
        "active_index_generation_id": "generation-old",
    })

    async def ready_job(worker_id):
        job, _ = await service.create_job(**job_args(
            user_id=user_id,
            document_id=document_id,
        ))
        claimed = await service.claim_next_job(worker_id)
        assert claimed["job_id"] == job["job_id"]
        for next_stage, progress in (
            ("chunking", 20), ("embedding", 40),
            ("indexing", 70), ("validating", 95),
        ):
            assert await service.transition_job_state(
                job["job_id"],
                expected_stage=claimed["stage"],
                new_stage=next_stage,
                worker_id=worker_id,
                progress=progress,
            )
            claimed["stage"] = next_stage
        return job

    def activation(expected, new):
        return {
            "user_id": user_id,
            "document_id": document_id,
            "expected_active_generation_id": expected,
            "new_active_generation_id": new,
            "index_fingerprint": "c2b-fingerprint",
            "index_config": {"chunk_strategy": "fixed"},
            "indexed_at": datetime.now(timezone.utc),
        }

    try:
        first = await ready_job("worker-1")
        await worker._activate_and_complete(
            first, "worker-1", 60,
            **activation("generation-old", "generation-1"),
        )
        assert (await documents.find_one({"_id": document_id}))[
            "active_index_generation_id"
        ] == "generation-1"
        assert (await service.get_job(first["job_id"]))["status"] == "completed"

        stolen = await ready_job("worker-2")
        await mongo_jobs.update_one(
            {"job_id": stolen["job_id"]},
            {"$set": {"worker_id": "worker-other"}},
        )
        with pytest.raises(IndexOwnershipLostError):
            await worker._activate_and_complete(
                stolen, "worker-2", 60,
                **activation("generation-1", "generation-stolen"),
            )
        assert (await documents.find_one({"_id": document_id}))[
            "active_index_generation_id"
        ] == "generation-1"
        assert (await service.get_job(stolen["job_id"]))["status"] == "running"
        await mongo_jobs.delete_one({"job_id": stolen["job_id"]})

        conflicted = await ready_job("worker-3")
        with pytest.raises(IndexActivationConflictError):
            await worker._activate_and_complete(
                conflicted, "worker-3", 60,
                **activation("wrong-generation", "generation-conflict"),
            )
        assert (await documents.find_one({"_id": document_id}))[
            "active_index_generation_id"
        ] == "generation-1"
        assert (await service.get_job(conflicted["job_id"]))["status"] == "running"
    finally:
        await documents.delete_one({"_id": document_id, "user_id": user_id})


@pytest.mark.asyncio
async def test_worker_indexes_real_pdf_into_temporary_chroma(
    mongo_jobs, tmp_path, monkeypatch,
):
    import fitz
    from unittest.mock import AsyncMock

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "HIV risk prediction using electronic health records.")
    file_path = tmp_path / "c2b-document.pdf"
    pdf.save(file_path)
    pdf.close()

    user_id = f"c2b-user-{uuid4().hex}"
    document_id = f"c2b-document-{uuid4().hex}"
    document_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    documents = database.get_database()["documents"]
    monkeypatch.setattr(vector_store, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(
        pipeline,
        "get_existing_user",
        AsyncMock(return_value={"user_id": user_id}),
    )

    def local_embedding(chunks):
        return [{**chunk, "embedding": [0.1, 0.2, 0.3]} for chunk in chunks]

    monkeypatch.setattr(pipeline, "embed_chunks", local_embedding)
    config = pipeline.build_index_config("fixed", 500, 50)
    await documents.insert_one({
        "_id": document_id,
        "user_id": user_id,
        "filename": file_path.name,
        "document_hash": document_hash,
        "language": "unknown",
        "index_fingerprint": None,
        "active_index_generation_id": None,
        "processing_status": "pending",
        "processing_stage": "pending",
    })

    try:
        job, _ = await service.create_job(**job_args(
            user_id=user_id,
            document_id=document_id,
            file_path=str(file_path),
            index_config=config,
            index_fingerprint=pipeline.build_index_fingerprint(config),
        ))
        result = await worker.run_once("c2b-worker", lease_duration_seconds=120)
        saved_document = await documents.find_one({"_id": document_id})
        assert result["job_id"] == job["job_id"]
        assert result["status"] == result["stage"] == "completed"
        assert saved_document["active_index_generation_id"] == (
            result["index_generation_id"]
        )
        assert vector_store.has_chunks(
            user_id, document_id, result["index_generation_id"]
        )
    finally:
        await documents.delete_one({"_id": document_id, "user_id": user_id})
