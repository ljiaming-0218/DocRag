"""Independent MongoDB research worker: python -m harness.research_worker."""

import argparse
import asyncio
import logging
from uuid import uuid4
from datetime import datetime, timezone

from harness.research_mongo_store import MongoResearchTaskStore, ResearchLeaseLostError
from harness.research_runner import run_research_task
from harness.research_tools import ResearchStoppedError, ensure_task_scope
from stores.database import close_database, connect_database


logger = logging.getLogger(__name__)


async def run_once(worker_id, *, store=None, lease_seconds=30):
    if lease_seconds <= 0:
        raise ValueError("lease duration must be positive")
    store = store or MongoResearchTaskStore()
    claimed = await store.claim(worker_id, lease_seconds)
    if claimed is None:
        return None
    task, token = claimed
    logger.info(
        "research_task_claimed task_id=%s worker_id=%s model_turns=%s tool_calls=%s pending_action=%s",
        task.task_id, worker_id, task.model_turns, task.tool_calls, bool(task.pending_action),
    )
    lock = asyncio.Lock()
    lost = asyncio.Event()

    async def checkpoint(current):
        async with lock:
            if lost.is_set():
                raise ResearchLeaseLostError("research worker no longer owns the task")
            await store.checkpoint(current, worker_id, token)

    async def execute():
        try:
            await ensure_task_scope(task)
        except ResearchStoppedError:
            task.finished_at = datetime.now(timezone.utc)
            await checkpoint(task)
            return
        await run_research_task(task, checkpoint)

    execution = asyncio.create_task(execute())

    async def heartbeat():
        try:
            while True:
                await asyncio.sleep(lease_seconds / 3)
                async with lock:
                    renewed = await store.renew(task.task_id, worker_id, token, lease_seconds)
                if not renewed:
                    lost.set()
                    execution.cancel()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("research_lease_renew_failed task_id=%s", task.task_id)
            lost.set()
            execution.cancel()

    renewal = asyncio.create_task(heartbeat())
    try:
        await execution
    except ResearchLeaseLostError:
        logger.info("research_lease_lost task_id=%s", task.task_id)
    except asyncio.CancelledError:
        if not lost.is_set():
            raise
    else:
        logger.info(
            "research_task_finished task_id=%s status=%s stop_reason=%s model_turns=%s tool_calls=%s",
            task.task_id, task.status, task.stop_reason, task.model_turns, task.tool_calls,
        )
    finally:
        execution.cancel()
        renewal.cancel()
        await asyncio.gather(execution, renewal, return_exceptions=True)
    return task.task_id


async def serve(once=False):
    await connect_database()
    store = MongoResearchTaskStore()
    worker_id = f"research-{uuid4()}"
    try:
        await store.create_indexes()
        while True:
            try:
                task_id = await run_once(worker_id, store=store)
            except Exception:
                logger.exception("research_worker_failed worker_id=%s", worker_id)
                task_id = None
            if once:
                return
            if task_id is None:
                await asyncio.sleep(2)
    finally:
        await close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Claim at most one task and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(serve(args.once))
    except KeyboardInterrupt:
        pass
