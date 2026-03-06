from __future__ import annotations

import asyncio

from annolid.core.agent.cron import CronJob, CronJobState, CronPayload, CronSchedule
from annolid.core.agent.tools.function_registry import FunctionToolRegistry
from annolid.gui.widgets.ai_chat_manager import _execute_cron_job_payload


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
