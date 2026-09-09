import logging
from pathlib import Path

from config import (
    CHROMA_DIR,
    UPLOAD_DIR,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)
from stores.database import ping_database


logger = logging.getLogger("uvicorn.error")


def ensure_runtime_directories() -> None:
    for path in (CHROMA_DIR, UPLOAD_DIR):
        path.mkdir(parents=True, exist_ok=True)


def is_directory_ready(path: Path) -> bool:
    return path.exists() and path.is_dir()


def is_llm_configured() -> bool:
    required_values = [
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        OPENROUTER_MODEL,
    ]

    return all(
        isinstance(value, str) and value.strip()
        for value in required_values
    )


async def check_readiness() -> dict:
    checks = {}

    try:
        mongodb_ready = await ping_database()
        checks["mongodb"] = "ok" if mongodb_ready else "failed"
    except Exception:
        logger.exception("readiness_check_failed dependency=mongodb")
        checks["mongodb"] = "failed"

    checks["chroma_directory"] = (
        "ok" if is_directory_ready(CHROMA_DIR) else "failed"
    )

    checks["upload_directory"] = (
        "ok" if is_directory_ready(UPLOAD_DIR) else "failed"
    )

    checks["llm_config"] = (
        "ok" if is_llm_configured() else "failed"
    )

    ready = all(
        status == "ok"
        for status in checks.values()
    )

    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }
