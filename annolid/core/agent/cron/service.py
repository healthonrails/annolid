from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore

CronJobHandler = Callable[[CronJob], Awaitable[Optional[str]]]


def _now_ms() -> int:
    return int(time.time() * 1000)


def compute_next_run(schedule: CronSchedule, now_ms: int) -> Optional[int]:
    if schedule.kind == "at":
        if schedule.at_ms and schedule.at_ms > now_ms:
            return int(schedule.at_ms)
        return None
    if schedule.kind == "every":
        if not schedule.every_ms or int(schedule.every_ms) <= 0:
            return None
        return int(now_ms + int(schedule.every_ms))
    if schedule.kind == "cron" and schedule.expr:
        try:
            from croniter import croniter  # type: ignore

            itr = croniter(schedule.expr, now_ms / 1000.0)
            return int(float(itr.get_next()) * 1000.0)
        except Exception:
            return None
    return None


class CronService:
    """Persistent cron scheduler for scheduled agent jobs."""

    def __init__(
        self,
        store_path: Path,
        *,
        on_job: Optional[CronJobHandler] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.store_path = Path(store_path).expanduser()
        self.on_job = on_job
        self._store: Optional[CronStore] = None
        self._running = False
        self._timer_task: Optional[asyncio.Task[Any]] = None
        self._logger = logger or logging.getLogger("annolid.agent.cron")
        self._persistence_error: Optional[str] = None

    def _load_store(self) -> CronStore:
        if self._store is not None:
            return self._store
        if not self.store_path.exists():
            self._store = CronStore()
            return self._store
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception:
            self._store = CronStore()
            return self._store
        jobs: list[CronJob] = []
        for row in list(payload.get("jobs") or []):
            try:
                job = CronJob(
                    id=str(row["id"]),
                    name=str(row.get("name") or ""),
                    enabled=bool(row.get("enabled", True)),
                    schedule=CronSchedule(
                        kind=str(row.get("schedule", {}).get("kind") or "every"),  # type: ignore[arg-type]
                        at_ms=row.get("schedule", {}).get("atMs"),
                        every_ms=row.get("schedule", {}).get("everyMs"),
                        expr=row.get("schedule", {}).get("expr"),
                        tz=row.get("schedule", {}).get("tz"),
                    ),
                    payload=CronPayload(
                        kind=str(row.get("payload", {}).get("kind") or "agent_turn"),  # type: ignore[arg-type]
                        message=str(row.get("payload", {}).get("message") or ""),
                        deliver=bool(row.get("payload", {}).get("deliver", False)),
                        channel=row.get("payload", {}).get("channel"),
                        to=row.get("payload", {}).get("to"),
                    ),
                    state=CronJobState(
                        next_run_at_ms=row.get("state", {}).get("nextRunAtMs"),
                        last_run_at_ms=row.get("state", {}).get("lastRunAtMs"),
                        last_status=row.get("state", {}).get("lastStatus"),
                        last_error=row.get("state", {}).get("lastError"),
                    ),
                    created_at_ms=int(row.get("createdAtMs") or 0),
                    updated_at_ms=int(row.get("updatedAtMs") or 0),
                    delete_after_run=bool(row.get("deleteAfterRun", False)),
                )
            except Exception:
                continue
            jobs.append(job)
        self._store = CronStore(version=int(payload.get("version") or 1), jobs=jobs)
        return self._store

    def _save_store(self) -> None:
        if self._store is None:
            return
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": self._store.version,
                "jobs": [
                    {
                        "id": j.id,
                        "name": j.name,
                        "enabled": j.enabled,
                        "schedule": {
                            "kind": j.schedule.kind,
                            "atMs": j.schedule.at_ms,
                            "everyMs": j.schedule.every_ms,
                            "expr": j.schedule.expr,
                            "tz": j.schedule.tz,
                        },
                        "payload": {
                            "kind": j.payload.kind,
                            "message": j.payload.message,
                            "deliver": j.payload.deliver,
                            "channel": j.payload.channel,
                            "to": j.payload.to,
                        },
                        "state": {
                            "nextRunAtMs": j.state.next_run_at_ms,
                            "lastRunAtMs": j.state.last_run_at_ms,
                            "lastStatus": j.state.last_status,
                            "lastError": j.state.last_error,
                        },
                        "createdAtMs": j.created_at_ms,
                        "updatedAtMs": j.updated_at_ms,
                        "deleteAfterRun": j.delete_after_run,
                    }
                    for j in self._store.jobs
                ],
            }
            self.store_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._persistence_error = None
        except OSError as exc:
            self._persistence_error = str(exc)
            self._logger.warning(
                "Cron store persistence failed path=%s error=%s",
                self.store_path,
                exc,
            )

    def _recompute_next_runs(self) -> None:
        store = self._load_store()
        now = _now_ms()
        for job in store.jobs:
            if not job.enabled:
                job.state.next_run_at_ms = None
                continue
            job.state.next_run_at_ms = compute_next_run(job.schedule, now)

    def _next_wake_ms(self) -> Optional[int]:
        store = self._load_store()
        next_values = [
            int(j.state.next_run_at_ms)
            for j in store.jobs
            if j.enabled and j.state.next_run_at_ms
        ]
        if not next_values:
            return None
        return min(next_values)

    def _arm_timer(self) -> None:
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None
        if not self._running:
            return
        wake = self._next_wake_ms()
        if wake is None:
            return
        delay = max(0.0, (wake - _now_ms()) / 1000.0)

        async def _tick() -> None:
            await asyncio.sleep(delay)
            if self._running:
                await self.on_timer()

        self._timer_task = asyncio.create_task(_tick())

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._recompute_next_runs()
        self._save_store()
        self._arm_timer()

    async def stop(self) -> None:
        self._running = False
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None

    async def on_timer(self) -> None:
        await self.run_due_jobs()
        self._save_store()
        self._arm_timer()

    async def run_due_jobs(self) -> int:
        store = self._load_store()
        now = _now_ms()
        due = [
            job
            for job in store.jobs
            if job.enabled
            and job.state.next_run_at_ms
            and now >= job.state.next_run_at_ms
        ]
        ran = 0
        for job in due:
            await self.execute_job(job)
            ran += 1
        if ran:
            self._save_store()
        return ran

    async def execute_job(self, job: CronJob) -> None:
        started = _now_ms()
        try:
            if self.on_job is not None:
                await self.on_job(job)
                job.state.last_status = "ok"
                job.state.last_error = None
            else:
                job.state.last_status = "skipped"
                job.state.last_error = "No on_job handler configured."
        except Exception as exc:
            job.state.last_status = "error"
            job.state.last_error = str(exc)
            self._logger.error("Cron job failed id=%s err=%s", job.id, exc)
        job.state.last_run_at_ms = started
        job.updated_at_ms = _now_ms()
        if job.schedule.kind == "at":
            if job.delete_after_run:
                store = self._load_store()
                store.jobs = [row for row in store.jobs if row.id != job.id]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
            return
        job.state.next_run_at_ms = compute_next_run(job.schedule, _now_ms())

    def list_jobs(self, *, include_disabled: bool = False) -> list[CronJob]:
        store = self._load_store()
        rows = store.jobs if include_disabled else [j for j in store.jobs if j.enabled]
        return sorted(rows, key=lambda j: j.state.next_run_at_ms or 2**63 - 1)

    def add_job(
        self,
        *,
        name: str,
        schedule: CronSchedule,
        payload: CronPayload,
        delete_after_run: bool = False,
    ) -> CronJob:
        store = self._load_store()
        now = _now_ms()
        job = CronJob(
            id=str(uuid.uuid4())[:12],
            name=str(name or payload.message[:30] or "job"),
            enabled=True,
            schedule=schedule,
            payload=payload,
            state=CronJobState(next_run_at_ms=compute_next_run(schedule, now)),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=bool(delete_after_run),
        )
        store.jobs.append(job)
        self._save_store()
        self._arm_timer()
        return job

    def remove_job(self, job_id: str) -> bool:
        store = self._load_store()
        before = len(store.jobs)
        store.jobs = [j for j in store.jobs if j.id != str(job_id)]
        removed = len(store.jobs) != before
        if removed:
            self._save_store()
            self._arm_timer()
        return removed

    def enable_job(self, job_id: str, enabled: bool = True) -> Optional[CronJob]:
        store = self._load_store()
        for job in store.jobs:
            if job.id != str(job_id):
                continue
            job.enabled = bool(enabled)
            job.updated_at_ms = _now_ms()
            if job.enabled:
                job.state.next_run_at_ms = compute_next_run(job.schedule, _now_ms())
            else:
                job.state.next_run_at_ms = None
            self._save_store()
            self._arm_timer()
            return job
        return None

    async def run_job(self, job_id: str, *, force: bool = False) -> bool:
        store = self._load_store()
        for job in store.jobs:
            if job.id != str(job_id):
                continue
            if not force and not job.enabled:
                return False
            await self.execute_job(job)
            self._save_store()
            self._arm_timer()
            return True
        return False

    def status(self) -> dict[str, Any]:
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._next_wake_ms(),
            "persistence_error": self._persistence_error,
        }
