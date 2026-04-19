from __future__ import annotations

import asyncio
from pathlib import Path

from annolid.core.agent.cron import CronJob, CronJobState, CronPayload, CronSchedule
from annolid.core.agent.cron.service import CronService
from annolid.core.agent.tools.function_registry import FunctionToolRegistry
from annolid.gui.widgets.ai_chat_manager import (
    _execute_cron_job_payload,
    _sync_dream_cron_job,
)


class _EmailTool:
    name = "email"

    def validate_params(self, params):
        del params
        return []

    async def execute(self, **kwargs):
        return (
            f"sent:{kwargs.get('to')}:{kwargs.get('subject')}:{kwargs.get('content')}:"
            f"{len(kwargs.get('attachment_paths') or [])}"
        )


class _Bus:
    def __init__(self) -> None:
        self.rows = []

    async def publish_inbound(self, msg) -> None:
        self.rows.append(msg)


def test_execute_cron_job_payload_sends_email_directly() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EmailTool())
    job = CronJob(
        id="job1",
        name="email",
        schedule=CronSchedule(kind="every", every_ms=1000),
        payload=CronPayload(
            kind="send_email",
            message="send",
            email_to="user@example.com",
            email_subject="Hello",
            email_content="Body",
            attachment_paths=["/tmp/a.txt"],
        ),
        state=CronJobState(),
    )

    result = asyncio.run(
        _execute_cron_job_payload(job, tools=registry, background_bus=None)
    )
    assert result == "sent:user@example.com:Hello:Body:1"


def test_execute_cron_job_payload_publishes_agent_turn() -> None:
    bus = _Bus()
    job = CronJob(
        id="job2",
        name="prompt",
        schedule=CronSchedule(kind="every", every_ms=1000),
        payload=CronPayload(
            kind="agent_turn",
            message="send reminder",
            channel="email",
            to="user@example.com",
        ),
        state=CronJobState(next_run_at_ms=123),
    )

    result = asyncio.run(_execute_cron_job_payload(job, tools=None, background_bus=bus))
    assert result == "Inbound generated"
    assert len(bus.rows) == 1
    assert bus.rows[0].content == "send reminder"


def test_execute_cron_job_payload_runs_dream_job(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    history = workspace / "memory" / "HISTORY.md"
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text("# History\n\n[2026-04-17 10:00] base\n", encoding="utf-8")

    job = CronJob(
        id="job3",
        name="dream",
        schedule=CronSchedule(kind="every", every_ms=3600 * 1000),
        payload=CronPayload(kind="dream_run", message="dream"),
        state=CronJobState(next_run_at_ms=123),
    )
    first = asyncio.run(
        _execute_cron_job_payload(
            job,
            tools=None,
            background_bus=None,
            dream_workspace=str(workspace),
            dream_max_batch_entries=10,
            dream_initialize_cursor_to_end=True,
        )
    )
    second = asyncio.run(
        _execute_cron_job_payload(
            job,
            tools=None,
            background_bus=None,
            dream_workspace=str(workspace),
            dream_max_batch_entries=10,
            dream_initialize_cursor_to_end=True,
        )
    )
    assert (
        "initialized" in str(first or "").lower()
        or "cursor" in str(first or "").lower()
    )
    assert "nothing to process" in str(second or "").lower()


def test_sync_dream_cron_job_adds_and_disables_system_job(tmp_path: Path) -> None:
    service = CronService(store_path=tmp_path / "cron" / "jobs.json")
    enabled_id = _sync_dream_cron_job(
        cron_service=service,
        enabled=True,
        interval_hours=2,
    )
    assert enabled_id
    rows = service.list_jobs(include_disabled=True)
    assert len(rows) == 1
    assert rows[0].payload.kind == "dream_run"
    assert rows[0].enabled is True

    disabled_id = _sync_dream_cron_job(
        cron_service=service,
        enabled=False,
        interval_hours=2,
    )
    assert disabled_id is None
    rows = service.list_jobs(include_disabled=True)
    assert len(rows) == 1
    assert rows[0].enabled is False
