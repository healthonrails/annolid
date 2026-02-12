from __future__ import annotations

import asyncio
import time
from pathlib import Path

from annolid.core.agent.cron import (
    CronPayload,
    CronSchedule,
    CronService,
    compute_next_run,
)


def test_compute_next_run_for_every_and_at() -> None:
    now = int(time.time() * 1000)
    every = CronSchedule(kind="every", every_ms=5000)
    next_every = compute_next_run(every, now)
    assert next_every is not None and next_every >= now + 5000

    at_future = CronSchedule(kind="at", at_ms=now + 2000)
    assert compute_next_run(at_future, now) == now + 2000

    at_past = CronSchedule(kind="at", at_ms=now - 2000)
    assert compute_next_run(at_past, now) is None


def test_cron_service_add_list_remove_and_enable(tmp_path: Path) -> None:
    svc = CronService(store_path=tmp_path / "jobs.json")
    job = svc.add_job(
        name="ping",
        schedule=CronSchedule(kind="every", every_ms=1000),
        payload=CronPayload(message="hello"),
    )
    rows = svc.list_jobs(include_disabled=True)
    assert len(rows) == 1
    assert rows[0].id == job.id

    disabled = svc.enable_job(job.id, enabled=False)
    assert disabled is not None and disabled.enabled is False

    assert svc.remove_job(job.id) is True
    assert svc.remove_job(job.id) is False


def test_cron_service_run_job_calls_handler(tmp_path: Path) -> None:
    seen: list[str] = []

    async def _on_job(job) -> str:
        seen.append(job.payload.message)
        return "ok"

    svc = CronService(store_path=tmp_path / "jobs.json", on_job=_on_job)
    job = svc.add_job(
        name="once",
        schedule=CronSchedule(kind="at", at_ms=int(time.time() * 1000) + 60_000),
        payload=CronPayload(message="task"),
        delete_after_run=False,
    )
    ok = asyncio.run(svc.run_job(job.id, force=True))
    assert ok is True
    assert seen == ["task"]
