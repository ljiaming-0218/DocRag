import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api_errors import APIError, api_error_handler
from dependencies.auth import get_current_user_id
from routers import ingestion_job_router
from services import ingestion_submission_service as submission
from services import ingestion_worker_loop
import main


def make_job(**changes):
    now = datetime.now(timezone.utc)
    job = {
        "job_id": "job-1", "user_id": "user-1",
        "document_id": "doc-1", "status": "failed",
        "stage": "embedding", "progress": 40,
        "retry_count": 0, "max_retries": 2,
        "error_code": "EMBEDDING_FAILED", "error_message": "failed",
        "created_at": now, "updated_at": now, "finished_at": now,
        "worker_id": "internal-worker",
    }
    job.update(changes)
    return job


@pytest.fixture
def job_client(monkeypatch):
    app = FastAPI()
    app.include_router(ingestion_job_router.router)
    app.add_exception_handler(APIError, api_error_handler)
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    with TestClient(app) as client:
        yield client


def test_get_job_hides_worker_and_rejects_foreign_user(job_client, monkeypatch):
    monkeypatch.setattr(ingestion_job_router, "get_job_for_user",
                        AsyncMock(return_value=make_job()))
    response = job_client.get("/ingestion-jobs/job-1")
    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
    assert "worker_id" not in response.json()
    monkeypatch.setattr(ingestion_job_router, "get_job_for_user",
                        AsyncMock(return_value=None))
    assert job_client.get("/ingestion-jobs/job-1").status_code == 404


def test_retry_checks_failed_status_and_limit(job_client, monkeypatch):
    fetch = AsyncMock(side_effect=[make_job(), make_job(
        status="queued", stage="queued", retry_count=1,
    )])
    reset = AsyncMock(return_value=True)
    monkeypatch.setattr(ingestion_job_router, "get_job_for_user", fetch)
    monkeypatch.setattr(ingestion_job_router, "reset_for_retry", reset)
    response = job_client.post("/ingestion-jobs/job-1/retry")
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    reset.assert_awaited_once_with("job-1", "embedding")

    for job in (make_job(status="running"), make_job(retry_count=2)):
        reset.reset_mock()
        monkeypatch.setattr(ingestion_job_router, "get_job_for_user",
                            AsyncMock(return_value=job))
        assert job_client.post("/ingestion-jobs/job-1/retry").status_code == 409
        reset.assert_not_awaited()


@pytest.fixture
def submission_mocks(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(submission, "get_existing_user",
                        AsyncMock(return_value={"user_id": "user-1"}))
    monkeypatch.setattr(submission, "upload_pdf", AsyncMock(return_value={
        "保存路径": str(pdf), "文件名": "paper.pdf",
        "document_hash": "hash-1", "created_new_file": True,
    }))
    monkeypatch.setattr(submission, "find_document_by_user_and_hash",
                        AsyncMock(return_value=None))
    monkeypatch.setattr(submission, "get_or_create_document",
                        AsyncMock(return_value={
                            "document_id": "doc-1", "existing_document": False,
                            "index_fingerprint": None,
                            "active_index_generation_id": None,
                        }))
    monkeypatch.setattr(submission, "find_by_idempotency_key",
                        AsyncMock(return_value=None))
    create = AsyncMock(return_value=(make_job(
        status="queued", stage="queued"), True))
    monkeypatch.setattr(submission, "create_job", create)
    return create, pdf


@pytest.mark.asyncio
async def test_submit_queues_without_running_index(submission_mocks, monkeypatch):
    create, _ = submission_mocks
    result, code = await submission.submit_index(
        "user-1", object(), 500, 50, "fixed", False, None,
    )
    assert code == 202 and result["job_id"] == "job-1"
    assert create.await_args.kwargs["index_config"]["chunk_size"] == 500
    assert create.await_args.kwargs["file_path"].endswith("paper.pdf")


@pytest.mark.asyncio
async def test_automatic_idempotency_key_is_stable(submission_mocks):
    create, _ = submission_mocks
    await submission.submit_index("user-1", object(), 500, 50, "fixed", False, None)
    first_key = create.await_args.kwargs["idempotency_key"]
    await submission.submit_index("user-1", object(), 500, 50, "fixed", False, None)
    assert create.await_args.kwargs["idempotency_key"] == first_key


@pytest.mark.asyncio
async def test_submit_reuses_existing_job_and_active_index(submission_mocks, monkeypatch):
    create, _ = submission_mocks
    previous = make_job(status="running", stage="embedding")
    previous.update({
        "index_fingerprint": submission.build_index_fingerprint(
            submission.build_index_config("fixed", 500, 50)),
        "index_config": submission.build_index_config("fixed", 500, 50),
    })
    monkeypatch.setattr(submission, "find_by_idempotency_key",
                        AsyncMock(return_value=previous))
    result, code = await submission.submit_index(
        "user-1", object(), 500, 50, "fixed", False, None,
    )
    assert code == 202 and result["job_id"] == "job-1"
    create.assert_not_awaited()

    monkeypatch.setattr(submission, "find_by_idempotency_key",
                        AsyncMock(return_value=None))
    monkeypatch.setattr(submission, "get_or_create_document",
                        AsyncMock(return_value={
                            "document_id": "doc-1",
                            "existing_document": True,
                            "index_fingerprint": previous["index_fingerprint"],
                            "active_index_generation_id": "generation-1",
                        }))
    monkeypatch.setattr(submission, "has_chunks", lambda *args: True)
    result, code = await submission.submit_index(
        "user-1", object(), 500, 50, "fixed", False, None,
    )
    assert code == 200 and result["reused"] is True
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_job_requires_existing_active_index(submission_mocks, monkeypatch):
    monkeypatch.setattr(submission, "find_document_by_user_and_hash",
                        AsyncMock(return_value={"_id": "doc-1"}))
    previous = make_job(status="completed", stage="completed")
    previous.update({
        "index_fingerprint": submission.build_index_fingerprint(
            submission.build_index_config("fixed", 500, 50)),
        "index_config": submission.build_index_config("fixed", 500, 50),
    })
    monkeypatch.setattr(submission, "find_by_idempotency_key",
                        AsyncMock(return_value=previous))
    with pytest.raises(submission.DocumentIndexUnavailableError):
        await submission.submit_index(
            "user-1", object(), 500, 50, "fixed", False, None,
        )

@pytest.mark.asyncio
async def test_submit_cleans_new_unindexed_file_on_job_failure(
    submission_mocks, monkeypatch,
):
    create, pdf = submission_mocks
    create.side_effect = RuntimeError("database unavailable")
    monkeypatch.setattr(submission, "has_jobs_for_document",
                        AsyncMock(return_value=False))
    monkeypatch.setattr(submission, "delete_unindexed_document",
                        AsyncMock(return_value=True))
    with pytest.raises(RuntimeError):
        await submission.submit_index(
            "user-1", object(), 500, 50, "fixed", False, None,
        )
    assert not pdf.exists()


@pytest.mark.asyncio
async def test_worker_loop_consumes_queue_and_stops(monkeypatch):
    stop = asyncio.Event()
    calls = []

    async def once(worker_id):
        calls.append(worker_id)
        if len(calls) == 2:
            stop.set()
        return None

    monkeypatch.setattr(ingestion_worker_loop, "run_once", once)
    await ingestion_worker_loop.run_worker_loop(stop, poll_seconds=0.001)
    assert len(calls) == 2 and calls[0] == calls[1]


@pytest.mark.asyncio
async def test_worker_loop_waits_for_inflight_on_shutdown(monkeypatch):
    stop = asyncio.Event()
    started = asyncio.Event()
    release = asyncio.Event()

    async def once(worker_id):
        started.set()
        await release.wait()
        return make_job(status="completed")

    monkeypatch.setattr(ingestion_worker_loop, "run_once", once)
    task = asyncio.create_task(ingestion_worker_loop.run_worker_loop(stop))
    await started.wait()
    stop.set()
    assert not task.done()
    release.set()
    await task


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_worker(monkeypatch):
    calls = []

    async def noop(*args, **kwargs):
        return None

    async def worker(stop_event):
        calls.append("started")
        await stop_event.wait()
        calls.append("stopped")

    monkeypatch.setattr(main, "ensure_runtime_directories", lambda: None)
    for name in (
        "connect_database", "create_conversation_indexes",
        "create_message_indexes", "create_user_indexes",
        "create_document_indexes", "create_knowledge_base_indexes",
        "create_ingestion_job_indexes", "close_database",
    ):
        monkeypatch.setattr(main, name, noop)
    monkeypatch.setattr(main, "run_worker_loop", worker)
    async with main.lifespan(main.app):
        await asyncio.sleep(0)
        assert calls == ["started"]
    assert calls == ["started", "stopped"]
