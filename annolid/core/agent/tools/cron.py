from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from annolid.core.agent.cron import CronPayload, CronSchedule, CronService
from annolid.core.agent.utils import get_agent_data_path

from .function_base import FunctionTool


class CronTool(FunctionTool):
    def __init__(
        self,
        *,
        store_path: Path | None = None,
        send_callback: Callable[[str, str, str], Awaitable[None] | None] | None = None,
    ):
        self._channel = ""
        self._chat_id = ""
        self._send_callback = send_callback
        if store_path is None:
            store_path = self._resolve_default_store_path()
        self._service = CronService(store_path=store_path, on_job=self._on_job)

    @staticmethod
    def _resolve_default_store_path() -> Path:
        data_path = get_agent_data_path()
        candidates = [
            data_path / "cron" / "jobs.json",
            Path.cwd() / ".annolid" / "cron" / "jobs.json",
            Path("/tmp") / "annolid" / "cron" / "jobs.json",
        ]
        for path in candidates:
            if CronTool._is_store_path_writable(path):
                return path
        return candidates[-1]

    @staticmethod
    def _is_store_path_writable(path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        probe = path.parent / f".cron-write-probe-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return (
            "Schedule commands, agent workflows (like sending emails), or recurring tasks. "
            "When triggered, the configured `message` string is executed as a background prompt. "
            "Actions: add, list, remove, enable, disable, run, status."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "add",
                        "list",
                        "remove",
                        "enable",
                        "disable",
                        "run",
                        "status",
                    ],
                },
                "message": {"type": "string"},
                "every_seconds": {"type": "integer"},
                "cron_expr": {"type": "string"},
                "at": {"type": "string"},
                "at_ms": {"type": "integer"},
                "deliver": {"type": "boolean"},
                "job_id": {"type": "string"},
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        at: str | None = None,
        at_ms: int | None = None,
        deliver: bool = False,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if action == "add":
            return self._add_job(
                message=message,
                every_seconds=every_seconds,
                cron_expr=cron_expr,
                at=at,
                at_ms=at_ms,
                deliver=bool(deliver),
            )
        if action == "list":
            return self._list_jobs()
        if action == "remove":
            return self._remove_job(job_id)
        if action == "enable":
            return self._enable_job(job_id, True)
        if action == "disable":
            return self._enable_job(job_id, False)
        if action == "run":
            return await self._run_job(job_id)
        if action == "status":
            return self._status()
        return f"Unknown action: {action}"

    def _add_job(
        self,
        *,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        at: str | None,
        at_ms: int | None,
        deliver: bool,
    ) -> str:
        if not message:
            return "Error: message is required for add"
        if not self._channel or not self._chat_id:
            return "Error: no session context (channel/chat_id)"
        parsed_at_ms: int | None = None
        at_text = str(at or "").strip()
        if at_text:
            parsed_at_ms = self._parse_iso_datetime_ms(at_text)
            if parsed_at_ms is None:
                return "Error: at must be an ISO datetime string (e.g., 2026-02-13T09:30:00Z)"
        if not every_seconds and not cron_expr and not at_ms and parsed_at_ms is None:
            return "Error: one of every_seconds, cron_expr, at, or at_ms is required"
        if every_seconds and int(every_seconds) <= 0:
            return "Error: every_seconds must be > 0"

        resolved_at_ms = int(at_ms) if at_ms else parsed_at_ms
        if resolved_at_ms is not None:
            schedule = CronSchedule(kind="at", at_ms=resolved_at_ms)
        elif every_seconds:
            schedule = CronSchedule(kind="every", every_ms=int(every_seconds) * 1000)
        else:
            schedule = CronSchedule(kind="cron", expr=str(cron_expr or "").strip())

        payload = CronPayload(
            kind="agent_turn",
            message=message,
            deliver=bool(deliver),
            channel=self._channel,
            to=self._chat_id,
        )
        job = self._service.add_job(
            name=message[:40],
            schedule=schedule,
            payload=payload,
            delete_after_run=(schedule.kind == "at"),
        )
        return f"Created job '{message[:30]}' (id: {job.id})"

    def _list_jobs(self) -> str:
        jobs = self._service.list_jobs(include_disabled=True)
        if not jobs:
            return "No scheduled jobs."
        lines = []
        for job in jobs:
            if job.schedule.kind == "every":
                mode = f"every={int((job.schedule.every_ms or 0) / 1000)}s"
            elif job.schedule.kind == "cron":
                mode = f"cron={job.schedule.expr}"
            else:
                mode = f"at={job.schedule.at_ms}"
            marker = "enabled" if job.enabled else "disabled"
            lines.append(
                f"- {job.payload.message[:30]} (id: {job.id}, {mode}, {marker})"
            )
        return "Scheduled jobs:\n" + "\n".join(lines)

    def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required for remove"
        if self._service.remove_job(job_id):
            return f"Removed job {job_id}"
        return f"Job {job_id} not found"

    def _enable_job(self, job_id: str | None, enabled: bool) -> str:
        if not job_id:
            return "Error: job_id is required"
        updated = self._service.enable_job(job_id, enabled=enabled)
        if updated is None:
            return f"Job {job_id} not found"
        return f"{'Enabled' if enabled else 'Disabled'} job {job_id}"

    async def _run_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required"
        ok = await self._service.run_job(job_id, force=True)
        if not ok:
            return f"Job {job_id} not found"
        return f"Ran job {job_id}"

    def _status(self) -> str:
        status = self._service.status()
        text = (
            f"Cron status: enabled={status.get('enabled')} "
            f"jobs={status.get('jobs')} next_wake_at_ms={status.get('next_wake_at_ms')}"
        )
        persistence_error = str(status.get("persistence_error") or "").strip()
        if persistence_error:
            text += f" persistence_error={persistence_error}"
        return text

    async def _on_job(self, job) -> str | None:
        message = str(job.payload.message or "")
        if job.payload.deliver and self._send_callback is not None:
            channel = str(job.payload.channel or self._channel or "")
            chat_id = str(job.payload.to or self._chat_id or "")
            if channel and chat_id and message:
                result = self._send_callback(channel, chat_id, message)
                if asyncio.iscoroutine(result):
                    await result
        return message

    @staticmethod
    def _parse_iso_datetime_ms(value: str) -> int | None:
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            when = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if when.tzinfo is None:
            when = when.astimezone()
        return int(when.timestamp() * 1000)


__all__ = ["CronTool"]
