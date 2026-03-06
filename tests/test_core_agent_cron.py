from __future__ import annotations

import asyncio
import io
import logging
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


def test_cron_service_preserves_at_job_on_handler_error(tmp_path: Path) -> None:
    async def _on_job(_job) -> str:
        return "Error: background bus unavailable"

    svc = CronService(store_path=tmp_path / "jobs.json", on_job=_on_job)
    job = svc.add_job(
        name="oneshot",
        schedule=CronSchedule(kind="at", at_ms=int(time.time() * 1000) + 60_000),
        payload=CronPayload(message="task"),
        delete_after_run=True,
    )
    ok = asyncio.run(svc.run_job(job.id, force=True))
    assert ok is True
    rows = svc.list_jobs(include_disabled=True)
    assert any(row.id == job.id for row in rows)
    saved = next(row for row in rows if row.id == job.id)
    assert saved.enabled is True
    assert saved.state.last_status == "error"
    assert saved.state.next_run_at_ms is not None


def test_cron_service_start_arms_poll_timer_without_jobs(tmp_path: Path) -> None:
    svc = CronService(store_path=tmp_path / "jobs.json")

    async def _run() -> None:
        await svc.start()
        assert svc.status().get("timer_armed") is True
        await svc.stop()

    asyncio.run(_run())


def test_cron_service_emits_job_audit_log(tmp_path: Path) -> None:
    async def _on_job(_job) -> str:
        return "ok"

    stream = io.StringIO()
    logger = logging.getLogger("annolid.agent.cron.test_audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    logger.handlers = [handler]

    svc = CronService(
        store_path=tmp_path / "jobs.json",
        on_job=_on_job,
        logger=logger,
    )
    job = svc.add_job(
        name="audit",
        schedule=CronSchedule(kind="every", every_ms=1000),
        payload=CronPayload(message="task"),
    )
    ok = asyncio.run(svc.run_job(job.id, force=True))
    assert ok is True
    log_text = stream.getvalue()
    assert "Cron job audit id=" in log_text
    assert f"id={job.id}" in log_text


def test_cron_service_rejects_invalid_timezone_on_add(tmp_path: Path) -> None:
    svc = CronService(store_path=tmp_path / "jobs.json")
    try:
        svc.add_job(
            name="bad-tz",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="Mars/Phobos"),
            payload=CronPayload(message="hello"),
        )
        assert False, "expected ValueError for invalid timezone"
    except ValueError as exc:
        assert "unknown timezone" in str(exc)


def test_cron_service_rejects_timezone_for_non_cron(tmp_path: Path) -> None:
    svc = CronService(store_path=tmp_path / "jobs.json")
    try:
        svc.add_job(
            name="bad-tz-kind",
            schedule=CronSchedule(kind="every", every_ms=1000, tz="UTC"),
            payload=CronPayload(message="hello"),
        )
        assert False, "expected ValueError for non-cron tz usage"
    except ValueError as exc:
        assert "tz can only be used with cron schedules" in str(exc)


def test_cron_service_rejects_invalid_every_interval(tmp_path: Path) -> None:
    svc = CronService(store_path=tmp_path / "jobs.json")
    try:
        svc.add_job(
            name="bad-every",
            schedule=CronSchedule(kind="every", every_ms=0),
            payload=CronPayload(message="hello"),
        )
        assert False, "expected ValueError for invalid every_ms"
    except ValueError as exc:
        assert "every_ms > 0" in str(exc)


def test_cron_service_rejects_past_at_schedule(tmp_path: Path) -> None:
    svc = CronService(store_path=tmp_path / "jobs.json")
    try:
        svc.add_job(
            name="past-at",
            schedule=CronSchedule(kind="at", at_ms=int(time.time() * 1000) - 1),
            payload=CronPayload(message="hello"),
        )
        assert False, "expected ValueError for past at schedule"
    except ValueError as exc:
        assert "must be in the future" in str(exc)


def test_cron_service_rejects_invalid_cron_expression(tmp_path: Path) -> None:
    svc = CronService(store_path=tmp_path / "jobs.json")
    try:
        svc.add_job(
            name="bad-cron",
            schedule=CronSchedule(kind="cron", expr="bad cron expr"),
            payload=CronPayload(message="hello"),
        )
        assert False, "expected ValueError for invalid cron expression"
    except ValueError as exc:
        assert "invalid cron schedule" in str(exc)


def test_cron_service_persists_scheduled_email_payload(tmp_path: Path) -> None:
    store_path = tmp_path / "jobs.json"
    svc = CronService(store_path=store_path)
    job = svc.add_job(
        name="email-job",
        schedule=CronSchedule(kind="every", every_ms=1000),
        payload=CronPayload(
            kind="send_email",
            message="send update",
            email_to="user@example.com",
            email_subject="Status",
            email_content="All good",
            attachment_paths=["/tmp/report.txt"],
        ),
    )

    loaded = CronService(store_path=store_path).list_jobs(include_disabled=True)
    restored = next(row for row in loaded if row.id == job.id)
    assert restored.payload.kind == "send_email"
    assert restored.payload.email_to == "user@example.com"
    assert restored.payload.email_subject == "Status"
    assert restored.payload.email_content == "All good"
    assert restored.payload.attachment_paths == ["/tmp/report.txt"]
