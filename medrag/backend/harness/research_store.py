from asyncio import Lock, Task
from datetime import datetime, timezone

from harness.research_task import ResearchTask


class ResearchTaskStore:
    """Bounded, process-local task storage for the H1 API prototype."""

    def __init__(self, max_tasks: int = 100) -> None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        self._tasks: dict[str, ResearchTask] = {}
        self._lock = Lock()
        self._max_tasks = max_tasks
        self._running: set[Task] = set()

    async def add(self, task: ResearchTask) -> None:
        async with self._lock:
            if len(self._tasks) >= self._max_tasks:
                raise RuntimeError("research task capacity reached")
            if task.task_id in self._tasks:
                raise ValueError("research task already exists")
            self._tasks[task.task_id] = task

    async def get_for_user(self, task_id: str, user_id: str) -> ResearchTask | None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.owner_id != user_id:
                return None
            return task

    async def start_for_user(self, task_id: str, user_id: str) -> ResearchTask | None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.owner_id != user_id:
                return None
            if task.status != "ready":
                raise ValueError("research task is not ready")
            task.status = "running"
            task.started_at = datetime.now(timezone.utc)
            return task

    def track(self, job: Task) -> None:
        self._running.add(job)
        job.add_done_callback(self._running.discard)

    async def enqueue_for_user(self, task_id: str, user_id: str) -> ResearchTask | None:
        task = await self.start_for_user(task_id, user_id)
        if task is not None:
            task.status = "queued"
        return task

    async def cancel_for_user(self, task_id: str, user_id: str) -> ResearchTask | None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.owner_id != user_id:
                return None
            if task.status in {"ready", "queued", "running"}:
                task.status = "cancelled"
                task.stop_reason = "user_cancelled"
                task.finished_at = datetime.now(timezone.utc)
            return task
