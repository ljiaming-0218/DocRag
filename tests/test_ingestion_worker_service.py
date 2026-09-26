from datetime import datetime, timedelta, timezone
from threading import Event
from unittest.mock import AsyncMock, Mock

import pytest

from services import document_ingestion_service as pipeline
from services import ingestion_job_service as jobs
from services import ingestion_worker_service as worker
from stores import ingestion_job_store
from test_document_ingestion_service import install_index_dependencies
from test_ingestion_job_service import FakeCollection


@pytest.fixture
def environment(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(
        ingestion_job_store,
        "get_database",
        lambda: {"ingestion_jobs": collection},
    )
    config = pipeline.build_index_config("fixed", 500, 50)
    document = {
        "_id": "document-1",
        "user_id": "user-1",
        "filename": "paper.pdf",
        "document_hash": "hash-1",
        "language": "en",
        "active_index_generation_id": "generation-old",
        "index_fingerprint": "old-fingerprint",
        "index_config": None,
        "processing_status": "completed",
    }
    monkeypatch.setattr(
        worker,
        "get_existing_document_for_user",
        AsyncMock(return_value=document),
    )
    monkeypatch.setattr(worker, "_check_source", Mock())
    verifier = Mock()
    cleaner = Mock(return_value=0)
    monkeypatch.setattr(worker, "verify_index_generation", verifier)
    monkeypatch.setattr(worker, "delete_index_generation", cleaner)
    dependencies = install_index_dependencies(
        monkeypatch,
        existing_chunks=True,
    )
    order = []

    def activate(**kwargs):
        order.append("activate")
        document["active_index_generation_id"] = kwargs[
            "new_active_generation_id"
        ]

    def verify(*args):
        order.append("verify")

    dependencies.activate_document_index_generation.side_effect = activate
    verifier.side_effect = verify

    async def transactional_activate(job, worker_id, lease_duration, **activation):
        await dependencies.activate_document_index_generation(**activation)
        assert await jobs.mark_completed(job["job_id"], worker_id)

    monkeypatch.setattr(worker, "_activate_and_complete", transactional_activate)
    return collection, config, document, dependencies, verifier, cleaner, order


async def queue_job(config, *, max_retries=1):
    return (await jobs.create_job(
        user_id="user-1",
        document_id="document-1",
        file_path="paper.pdf",
        index_config=config,
        index_fingerprint=pipeline.build_index_fingerprint(config),
        max_retries=max_retries,
    ))[0]


@pytest.mark.asyncio
async def test_worker_completes_shared_pipeline_in_stage_order(environment, monkeypatch):
    collection, config, document, deps, verifier, cleaner, order = environment
    job = await queue_job(config)
    original = ingestion_job_store.transition_job_state
    transitions = AsyncMock(wraps=original)
    monkeypatch.setattr(ingestion_job_store, "transition_job_state", transitions)

    result = await worker.run_once("worker-1")

    assert result["status"] == result["stage"] == "completed"
    assert result["index_generation_id"]
    assert result["finished_at"] is not None
    assert [call.args[4] for call in transitions.await_args_list] == [
        "extracting", "chunking", "embedding", "indexing", "validating",
    ]
    assert [call.kwargs["progress"] for call in transitions.await_args_list] == [
        5, 25, 50, 75, 95,
    ]
    assert order == ["verify", "activate"]
    assert document["active_index_generation_id"] == result["index_generation_id"]
    assert deps.parse_document_pages.await_count == 1
    deps.save_chunks.assert_called_once()
    verifier.assert_called_once()
    cleaner.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_point", "failure_stage", "error_code"),
    [
        ("embedding", "embedding", "EMBEDDING_FAILED"),
        ("indexing", "indexing", "VECTOR_WRITE_FAILED"),
        ("validating", "validating", "INDEX_VALIDATION_FAILED"),
        ("activation", "validating", "ACTIVE_GENERATION_CONFLICT"),
    ],
)
async def test_worker_failure_preserves_old_active(
    environment, failure_point, failure_stage, error_code,
):
    collection, config, document, deps, verifier, cleaner, order = environment
    await queue_job(config)
    failure = RuntimeError("stage failed")
    if failure_point == "embedding":
        deps.embed_chunks.side_effect = failure
    elif failure_point == "indexing":
        deps.save_chunks.side_effect = failure
    elif failure_point == "validating":
        verifier.side_effect = failure
    else:
        deps.activate_document_index_generation.side_effect = (
            pipeline.IndexActivationConflictError("generation changed")
        )

    result = await worker.run_once("worker-1")

    assert result["status"] == "failed"
    assert result["stage"] == failure_stage
    assert result["error_code"] == error_code
    assert document["active_index_generation_id"] == "generation-old"
    deps.invalidate_sparse_indexes.assert_not_called()
    cleaner.assert_called()
    if failure_stage == "validating" and failure_point != "activation":
        failure_call = deps.set_document_processing_state.await_args_list[-1]
        assert failure_call.args[2:4] == ("failed", "validating")


@pytest.mark.asyncio
async def test_worker_lease_loss_prevents_activation_or_completion(
    environment, monkeypatch,
):
    collection, config, document, deps, verifier, cleaner, order = environment
    job = await queue_job(config)
    original = jobs.renew_lease
    renew_count = 0

    async def lose_before_activation(*args, **kwargs):
        nonlocal renew_count
        renew_count += 1
        if renew_count == 8:
            collection.documents[0]["worker_id"] = "worker-new"
            return False
        return await original(*args, **kwargs)

    monkeypatch.setattr(jobs, "renew_lease", lose_before_activation)
    result = await worker.run_once("worker-1")

    assert result["status"] == "running"
    assert result["worker_id"] == "worker-new"
    assert document["active_index_generation_id"] == "generation-old"
    deps.activate_document_index_generation.assert_not_awaited()
    cleaner.assert_called()


@pytest.mark.asyncio
async def test_heartbeat_loss_during_embedding_stops_before_vector_write(
    environment, monkeypatch,
):
    collection, config, document, deps, verifier, cleaner, order = environment
    await queue_job(config)
    embedding_started = Event()
    release_embedding = Event()
    embedded = deps.embed_chunks.return_value

    def slow_embedding(chunks):
        embedding_started.set()
        assert release_embedding.wait(timeout=2)
        return embedded

    original = jobs.renew_lease

    async def lose_during_embedding(*args, **kwargs):
        if embedding_started.is_set():
            release_embedding.set()
            return False
        return await original(*args, **kwargs)

    deps.embed_chunks.side_effect = slow_embedding
    monkeypatch.setattr(jobs, "renew_lease", lose_during_embedding)
    result = await worker.run_once(
        "worker-1", heartbeat_interval_seconds=0.01
    )

    assert result["status"] == "running"
    assert document["active_index_generation_id"] == "generation-old"
    deps.save_chunks.assert_not_called()
    deps.activate_document_index_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_lease_loss_during_embedding_error_does_not_fail_document(
    environment, monkeypatch,
):
    collection, config, document, deps, verifier, cleaner, order = environment
    await queue_job(config)
    embedding_started = Event()
    release_embedding = Event()

    def failing_embedding(chunks):
        embedding_started.set()
        assert release_embedding.wait(timeout=2)
        raise RuntimeError("embedding unavailable")

    original = jobs.renew_lease

    async def lose_during_embedding(*args, **kwargs):
        if embedding_started.is_set():
            release_embedding.set()
            return False
        return await original(*args, **kwargs)

    deps.embed_chunks.side_effect = failing_embedding
    monkeypatch.setattr(jobs, "renew_lease", lose_during_embedding)
    result = await worker.run_once(
        "worker-1", heartbeat_interval_seconds=0.01
    )

    assert result["status"] == "running"
    assert document["active_index_generation_id"] == "generation-old"
    assert all(
        call.args[2] != "failed"
        for call in deps.set_document_processing_state.await_args_list
    )
    deps.save_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_retry_uses_fresh_generation(environment):
    collection, config, document, deps, verifier, cleaner, order = environment
    job = await queue_job(config)
    deps.embed_chunks.side_effect = RuntimeError("temporary failure")
    first = await worker.run_once("worker-1")
    failed_generation = first["index_generation_id"]
    assert first["status"] == "failed"
    assert await jobs.reset_for_retry(
        job["job_id"], expected_stage="embedding"
    )

    deps.embed_chunks.side_effect = None
    second = await worker.run_once("worker-2")
    assert second["status"] == "completed"
    assert second["index_generation_id"] != failed_generation
    assert second["retry_count"] == 1
    assert document["active_index_generation_id"] == second[
        "index_generation_id"
    ]


@pytest.mark.asyncio
async def test_expired_job_restarts_with_new_generation(environment):
    collection, config, document, deps, verifier, cleaner, order = environment
    job = await queue_job(config)
    now = datetime.now(timezone.utc)
    await jobs.claim_next_job("worker-old", now=now)
    old_generation = "abandoned-generation"
    collection.documents[0]["index_generation_id"] = old_generation
    collection.documents[0]["lease_until"] = now - timedelta(seconds=1)

    result = await worker.run_once("worker-new")

    assert result["status"] == "completed"
    assert result["index_generation_id"] != old_generation
    cleaner.assert_any_call("user-1", "document-1", old_generation)
    assert document["active_index_generation_id"] == result[
        "index_generation_id"
    ]


@pytest.mark.asyncio
async def test_reclaimed_validating_job_finishes_existing_active(environment):
    collection, config, document, deps, verifier, cleaner, order = environment
    job = await queue_job(config)
    collection.documents[0].update({
        "status": "running",
        "stage": "validating",
        "progress": 95,
        "worker_id": "worker-old",
        "index_generation_id": "generation-already-active",
        "lease_until": datetime.now(timezone.utc) - timedelta(seconds=1),
    })
    document["active_index_generation_id"] = "generation-already-active"

    result = await worker.run_once("worker-new")

    assert result["status"] == "completed"
    assert result["index_generation_id"] == "generation-already-active"
    deps.parse_document_pages.assert_not_awaited()
    cleaner.assert_not_called()
