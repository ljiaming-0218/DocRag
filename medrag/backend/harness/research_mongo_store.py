"""Durable H2 storage. Task state, events and evidence commit in one transaction."""

from dataclasses import asdict, fields
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from bson.codec_options import CodecOptions
from pymongo import ReturnDocument
from pymongo.read_concern import ReadConcern

from harness.research_task import ResearchTask
from stores.database import get_database


class ResearchLeaseLostError(RuntimeError):
    pass


class MongoResearchTaskStore:
    def _collection(self, name):
        return get_database()[name].with_options(
            codec_options=CodecOptions(tz_aware=True, tzinfo=timezone.utc)
        )

    @property
    def tasks(self):
        return self._collection("research_tasks")

    @property
    def events(self):
        return self._collection("research_events")

    @property
    def evidence(self):
        return self._collection("research_evidence")

    async def create_indexes(self):
        await self.tasks.create_index("task_id", unique=True)
        await self.tasks.create_index([("owner_id", 1), ("created_at", -1)])
        await self.tasks.create_index([("status", 1), ("lease_until", 1)])
        await self.events.create_index([("task_id", 1), ("action_id", 1)], unique=True)
        await self.evidence.create_index([("task_id", 1), ("evidence_id", 1)], unique=True)

    @staticmethod
    def _state(task):
        state = asdict(task)
        state.pop("events")
        state.pop("evidence")
        return state

    async def add(self, task):
        async with get_database().client.start_session() as session:
            async with await session.start_transaction():
                await self.tasks.insert_one(
                    {**self._state(task), "updated_at": datetime.now(timezone.utc)}, session=session,
                )
                await self._write_artifacts(task, session)

    async def _read(self, query, session):
        state = await self.tasks.find_one(query, session=session)
        if state is None:
            return None
        values = {field.name: state[field.name] for field in fields(ResearchTask) if field.name in state}
        values["document_ids"] = tuple(values["document_ids"])
        values["events"] = [row["value"] async for row in self.events.find(
            {"task_id": state["task_id"]}, session=session,
        ).sort("value.sequence", 1)]
        values["evidence"] = {row["evidence_id"]: row["value"] async for row in self.evidence.find(
            {"task_id": state["task_id"]}, session=session,
        ).sort("order", 1)}
        return ResearchTask(**values)

    async def get_for_user(self, task_id, user_id):
        async with get_database().client.start_session() as session:
            async with await session.start_transaction(read_concern=ReadConcern("snapshot")):
                return await self._read({"task_id": task_id, "owner_id": user_id}, session)

    async def enqueue_for_user(self, task_id, user_id):
        now = datetime.now(timezone.utc)
        row = await self.tasks.find_one_and_update(
            {"task_id": task_id, "owner_id": user_id, "status": "ready"},
            {"$set": {"status": "queued", "started_at": now, "updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
        if row is None:
            task = await self.get_for_user(task_id, user_id)
            if task is not None:
                raise ValueError("research task is not ready")
            return None
        return await self.get_for_user(task_id, user_id)

    async def cancel_for_user(self, task_id, user_id):
        now = datetime.now(timezone.utc)
        await self.tasks.update_one(
            {"task_id": task_id, "owner_id": user_id, "status": {"$in": ["ready", "queued", "running"]}},
            {"$set": {"status": "cancelled", "stop_reason": "user_cancelled", "finished_at": now, "updated_at": now}},
        )
        return await self.get_for_user(task_id, user_id)

    async def delete_for_user(self, task_id, user_id):
        async with get_database().client.start_session() as session:
            async with await session.start_transaction():
                row = await self.tasks.find_one(
                    {"task_id": task_id, "owner_id": user_id}, session=session,
                )
                if row is None:
                    return False
                if row["status"] in {"queued", "running"}:
                    raise ValueError("Cancel the running research task before deleting it")
                await self.tasks.delete_one({"task_id": task_id, "owner_id": user_id}, session=session)
                await self.events.delete_many({"task_id": task_id}, session=session)
                await self.evidence.delete_many({"task_id": task_id}, session=session)
        return True

    async def claim(self, worker_id, lease_seconds=30):
        now = datetime.now(timezone.utc)
        row = await self.tasks.find_one_and_update(
            {"$or": [{"status": "queued"}, {"status": "running", "lease_until": {"$lte": now}}]},
            {"$set": {"status": "running", "worker_id": worker_id, "lease_token": str(uuid4()),
                      "lease_until": now + timedelta(seconds=lease_seconds), "updated_at": now}},
            sort=[("created_at", 1)], return_document=ReturnDocument.AFTER,
        )
        if row is None:
            return None
        task = await self.get_for_user(row["task_id"], row["owner_id"])
        return task, row["lease_token"]

    def _fence(self, task_id, worker_id, token):
        return {"task_id": task_id, "status": "running", "worker_id": worker_id,
                "lease_token": token, "lease_until": {"$gt": datetime.now(timezone.utc)}}

    async def renew(self, task_id, worker_id, token, lease_seconds=30):
        result = await self.tasks.update_one(
            self._fence(task_id, worker_id, token),
            {"$set": {"lease_until": datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)}},
        )
        return result.matched_count == 1

    async def checkpoint(self, task, worker_id, token):
        async with get_database().client.start_session() as session:
            async with await session.start_transaction():
                result = await self.tasks.update_one(
                    self._fence(task.task_id, worker_id, token),
                    {"$set": {**self._state(task), "updated_at": datetime.now(timezone.utc)}}, session=session,
                )
                if result.matched_count != 1:
                    raise ResearchLeaseLostError("research lease lost before checkpoint")
                await self._write_artifacts(task, session)

    async def _write_artifacts(self, task, session):
        for event in task.events:
            await self.events.update_one(
                {"task_id": task.task_id, "action_id": event["action_id"]},
                {"$set": {"value": event}}, upsert=True, session=session,
            )
        for order, evidence in enumerate(task.evidence.values()):
            await self.evidence.update_one(
                {"task_id": task.task_id, "evidence_id": evidence["evidence_id"]},
                {"$set": {"value": evidence, "order": order}}, upsert=True, session=session,
            )
