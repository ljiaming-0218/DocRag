from unittest.mock import AsyncMock

import pytest

from services import health_service


def test_ensure_runtime_directories_creates_missing_directories(
    monkeypatch,
    tmp_path,
):
    chroma_dir = tmp_path / "runtime" / "chroma"
    upload_dir = tmp_path / "runtime" / "uploads"

    monkeypatch.setattr(health_service, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(health_service, "UPLOAD_DIR", upload_dir)

    health_service.ensure_runtime_directories()

    assert chroma_dir.is_dir()
    assert upload_dir.is_dir()


@pytest.mark.asyncio
async def test_readiness_returns_ready(monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma"
    upload_dir = tmp_path / "uploads"

    chroma_dir.mkdir()
    upload_dir.mkdir()

    monkeypatch.setattr(
        health_service,
        "ping_database",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        health_service,
        "CHROMA_DIR",
        chroma_dir,
    )
    monkeypatch.setattr(
        health_service,
        "UPLOAD_DIR",
        upload_dir,
    )
    monkeypatch.setattr(
        health_service,
        "OPENROUTER_API_KEY",
        "test-key",
    )
    monkeypatch.setattr(
        health_service,
        "OPENROUTER_BASE_URL",
        "https://example.com",
    )
    monkeypatch.setattr(
        health_service,
        "OPENROUTER_MODEL",
        "test-model",
    )

    result = await health_service.check_readiness()

    assert result["status"] == "ready"
    assert result["checks"]["mongodb"] == "ok"


@pytest.mark.asyncio
async def test_readiness_handles_mongodb_failure(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        health_service,
        "ping_database",
        AsyncMock(side_effect=RuntimeError("MongoDB unavailable")),
    )

    monkeypatch.setattr(
        health_service,
        "CHROMA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        health_service,
        "UPLOAD_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        health_service,
        "OPENROUTER_API_KEY",
        "test-key",
    )
    monkeypatch.setattr(
        health_service,
        "OPENROUTER_BASE_URL",
        "https://example.com",
    )
    monkeypatch.setattr(
        health_service,
        "OPENROUTER_MODEL",
        "test-model",
    )

    result = await health_service.check_readiness()

    assert result["status"] == "not_ready"
    assert result["checks"]["mongodb"] == "failed"


@pytest.mark.asyncio
async def test_readiness_detects_missing_directory(
    monkeypatch,
    tmp_path,
):
    missing_directory = tmp_path / "missing"

    monkeypatch.setattr(
        health_service,
        "ping_database",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        health_service,
        "CHROMA_DIR",
        missing_directory,
    )
    monkeypatch.setattr(
        health_service,
        "UPLOAD_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        health_service,
        "OPENROUTER_API_KEY",
        "test-key",
    )
    monkeypatch.setattr(
        health_service,
        "OPENROUTER_BASE_URL",
        "https://example.com",
    )
    monkeypatch.setattr(
        health_service,
        "OPENROUTER_MODEL",
        "test-model",
    )

    result = await health_service.check_readiness()

    assert result["status"] == "not_ready"
    assert result["checks"]["chroma_directory"] == "failed"
