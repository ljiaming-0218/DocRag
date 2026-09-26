import asyncio
import logging
from uuid import uuid4

from services.ingestion_worker_service import run_once


logger = logging.getLogger(__name__)


async def run_worker_loop(
    stop_event: asyncio.Event,
    *,
    worker_id: str | None = None,
    poll_seconds: float = 2.0,
) -> None:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    worker_id = worker_id or f"ingestion-{uuid4()}"
    while not stop_event.is_set():
        try:
            job = await run_once(worker_id)
        except Exception:
            logger.exception("ingestion_worker_loop_failed worker_id=%s", worker_id)
            job = None
        if job is None and not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass
