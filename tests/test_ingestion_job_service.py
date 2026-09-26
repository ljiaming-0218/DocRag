from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pymongo.errors import DuplicateKeyError

from services import ingestion_job_service as service
from stores import ingestion_job_store as store


class FakeCollection:
    def __init__(self):
        self.documents = []
        self.indexes = []

    def with_options(self, *, codec_options):
        assert codec_options.tz_aware is True
        return self

    async def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    async def insert_one(self, document):
        if any(
            existing["job_id"] == document["job_id"]
            or (
                isinstance(document.get("idempotency_key"), str)
                and existing.get("idempotency_key")
                == document["idempotency_key"]
            )
            for existing in self.documents
        ):
            raise DuplicateKeyError("duplicate key")
        self.documents.append(deepcopy(document))

    async def find_one(self, query):
        return deepcopy(next(
            (doc for doc in self.documents if _matches(doc, query)),
            None,
        ))

    async def update_one(self, query, update):
        document = next(
            (doc for doc in self.documents if _matches(doc, query)),
            None,
        )
        if document is None:
            return SimpleNamespace(matched_count=0)
        _apply_update(document, update)
        return SimpleNamespace(matched_count=1)

    async def find_one_and_update(
        self, query, update, *, sort, return_document
    ):
        matches = [doc for doc in self.documents if _matches(doc, query)]
        if not matches:
            return None
        key, direction = sort[0]
        matches.sort(key=lambda doc: doc.get(key), reverse=direction < 0)
        document = matches[0]
        _apply_update(document, update)
        return deepcopy(document)


def _matches(document, query):
    for key, condition in query.items():
        if key == "$or":
            if not any(_matches(document, item) for item in condition):
                return False
            continue
        if key == "$expr":
            left, right = condition["$lt"]
            if document.get(left[1:]) >= document.get(right[1:]):
                return False
            continue
        value = document.get(key)
        if isinstance(condition, dict):
            for operator, expected in condition.items():
                if operator == "$gt" and not (value is not None and value > expected):
                    return False
                if operator == "$lte" and not (value is not None and value <= expected):
                    return False
                if operator == "$lt" and not (value is not None and value < expected):
                    return False
                if operator == "$in" and value not in expected:
                    return False
                if operator == "$exists" and ((key in document) != expected):
                    return False
                if operator == "$type" and not isinstance(value, str):
                    return False
        elif value != condition:
            return False
    return True


def _apply_update(document, update):
    for key, value in update.get("$set", {}).items():
        document[key] = value
    for key in update.get("$unset", {}):
        document.pop(key, None)
    for key, value in update.get("$inc", {}).items():
        document[key] = document.get(key, 0) + value


@pytest.fixture
def collection(monkeypatch):
    value = FakeCollection()
    monkeypatch.setattr(store, "get_database", lambda: {"ingestion_jobs": value})
    return value


def job_args(**overrides):
    values = {
        "user_id": "user-1",
        "document_id": "doc-1",
        "file_path": "D:/uploads/doc.pdf",
        "index_fingerprint": "fp-1",
        "idempotency_key": None,
    }
    values.update(overrides)
    return values


@pytest.mark.asyncio
async def test_create_job_has_queued_schema_and_unique_job_id(collection):
    job, created = await service.create_job(**job_args())

    assert created is True
    assert job["status"] == job["stage"] == "queued"
    assert job["progress"] == job["retry_count"] == 0
    assert job["index_generation_id"] is None
    assert {
        "job_id", "user_id", "document_id", "file_path",
        "idempotency_key", "index_fingerprint", "index_generation_id",
        "status", "stage", "progress", "retry_count", "max_retries",
        "worker_id", "lease_until", "error_code", "error_message",
        "created_at", "updated_at", "finished_at",
    } <= job.keys()
    with pytest.raises(DuplicateKeyError):
        await store.create_job(job)


@pytest.mark.asyncio
async def test_idempotency_reuses_same_request_and_rejects_conflict(collection):
    first, created = await service.create_job(
        **job_args(idempotency_key="request-1")
    )
    repeated, repeated_created = await service.create_job(
        **job_args(idempotency_key="request-1")
    )

    assert created is True
    assert repeated_created is False
    assert repeated["job_id"] == first["job_id"]
    same_content, same_content_created = await service.create_job(
        **job_args(file_path="D:/uploads/same-content.pdf", idempotency_key="request-1")
    )
    assert same_content_created is False
    assert same_content["job_id"] == first["job_id"]
    with pytest.raises(service.IdempotencyConflictError):
        await service.create_job(**job_args(
            document_id="doc-other",
            idempotency_key="request-1",
        ))


@pytest.mark.asyncio
async def test_idempotency_key_race_recovers_existing_job(monkeypatch, collection):
    original_create = store.create_job
    calls = 0

    async def racing_create(job):
        nonlocal calls
        calls += 1
        if calls == 1:
            await original_create(job)
            raise DuplicateKeyError("concurrent idempotent insert")
        await original_create(job)

    monkeypatch.setattr(store, "create_job", racing_create)
    job, created = await service.create_job(
        **job_args(idempotency_key="race-key")
    )
    assert created is False
    assert len(collection.documents) == 1


@pytest.mark.asyncio
async def test_indexes_include_required_partial_unique_idempotency(collection):
    await store.create_ingestion_job_indexes()
    names = {options["name"] for _, options in collection.indexes}
    assert names == {
        "job_id_unique_idx", "user_created_at_idx", "status_lease_until_idx",
        "document_created_at_idx", "idempotency_key_unique_idx",
    }
    idem = next(opts for _, opts in collection.indexes
                if opts["name"] == "idempotency_key_unique_idx")
    assert idem["unique"] is True
    assert idem["partialFilterExpression"] == {
        "idempotency_key": {"$type": "string"}
    }


@pytest.mark.asyncio
async def test_illegal_stage_transition_is_rejected(collection):
    with pytest.raises(ValueError, match="阶段迁移"):
        await service.transition_job_state(
            "job-1", expected_stage="extracting", new_stage="indexing",
            worker_id="worker-1", progress=50,
        )


@pytest.mark.asyncio
async def test_claim_is_exclusive_and_checks_worker_lease(collection):
    job, _ = await service.create_job(**job_args())
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    claimed = await service.claim_next_job("worker-1", now=now)
    second = await service.claim_next_job("worker-2", now=now)

    assert claimed["job_id"] == job["job_id"]
    assert claimed["status"] == "running"
    assert claimed["stage"] == "extracting"
    assert claimed["lease_until"] == now + timedelta(seconds=60)
    assert second is None
    assert await service.renew_lease(
        job["job_id"], "worker-1", now=now
    ) is True
    assert await service.renew_lease(
        job["job_id"], "worker-2", now=now
    ) is False


@pytest.mark.asyncio
async def test_expired_lease_is_reclaimed_without_resetting_stage(collection):
    job, _ = await service.create_job(**job_args())
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    claimed = await service.claim_next_job("worker-old", now=now)
    await service.transition_job_state(
        job["job_id"], expected_stage="extracting", new_stage="chunking",
        worker_id="worker-old", progress=20, now=now,
    )
    later = now + timedelta(seconds=121)
    reclaimed = await service.claim_next_job("worker-new", now=later)

    assert claimed["worker_id"] == "worker-old"
    assert reclaimed["worker_id"] == "worker-new"
    assert reclaimed["stage"] == "chunking"


@pytest.mark.asyncio
async def test_failed_job_can_retry_and_clears_failed_generation(collection):
    job, _ = await service.create_job(**job_args(max_retries=1))
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    await service.claim_next_job("worker-1", now=now)
    claimed = collection.documents[0]
    claimed["index_generation_id"] = "failed-generation"
    assert await service.mark_failed(
        job["job_id"], "worker-1", "extracting", "PDF_PARSE_FAILED",
        "x" * 2500, now=now,
    ) is True
    assert len(claimed["error_message"]) == 2000
    assert await service.reset_for_retry(
        job["job_id"], expected_stage="extracting", now=now
    ) is True
    assert await service.reset_for_retry(
        job["job_id"], expected_stage="extracting", now=now
    ) is False
    retried = await service.get_job(job["job_id"])
    assert retried["retry_count"] == 1
    assert retried["status"] == retried["stage"] == "queued"
    assert "index_generation_id" not in retried
    assert "worker_id" not in retried and "lease_until" not in retried
    assert "error_code" not in retried and "finished_at" not in retried


@pytest.mark.asyncio
async def test_job_lookup_is_scoped_to_user(collection):
    job, _ = await service.create_job(**job_args())
    assert await service.get_job_for_user(job["job_id"], "user-1")
    assert await service.get_job_for_user(job["job_id"], "user-other") is None


@pytest.mark.asyncio
async def test_transition_requires_matching_stage_owner_and_live_lease(collection):
    job, _ = await service.create_job(**job_args())
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    await service.claim_next_job("worker-1", now=now)

    assert await service.transition_job_state(
        job["job_id"], expected_stage="extracting", new_stage="chunking",
        worker_id="worker-other", progress=20, now=now,
    ) is False
    assert await service.transition_job_state(
        job["job_id"], expected_stage="extracting", new_stage="chunking",
        worker_id="worker-1", progress=20, now=now + timedelta(seconds=61),
    ) is False
    assert await service.transition_job_state(
        job["job_id"], expected_stage="extracting", new_stage="chunking",
        worker_id="worker-1", progress=20, now=now,
    ) is True


@pytest.mark.asyncio
async def test_mark_completed_requires_validating_and_clears_lease(collection):
    job, _ = await service.create_job(**job_args())
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    await service.claim_next_job("worker-1", now=now)
    current_stage = "extracting"
    for next_stage, progress in [
        ("chunking", 20), ("embedding", 40), ("indexing", 70),
        ("validating", 95),
    ]:
        assert await service.transition_job_state(
            job["job_id"], expected_stage=current_stage,
            new_stage=next_stage, worker_id="worker-1", progress=progress,
            now=now,
        )
        current_stage = next_stage
    assert await service.mark_completed(job["job_id"], "worker-1", now) is True
    completed = await service.get_job(job["job_id"])
    assert completed["status"] == completed["stage"] == "completed"
    assert completed["progress"] == 100
    assert completed["finished_at"] == now
    assert "worker_id" not in completed and "lease_until" not in completed
