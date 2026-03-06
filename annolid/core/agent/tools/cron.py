from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Awaitable, Callable

from annolid.core.agent.cron import (
    CronPayload,
    CronSchedule,
    CronService,
    default_cron_store_path,
)

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
        candidates = [
            default_cron_store_path(),
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
            "Actions: add, list, remove/cancel, enable, disable, run, status/check. "
            "For cron expressions, optional `tz` accepts an IANA timezone."
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
                        "cancel",
                        "enable",
                        "disable",
                        "run",
                        "status",
                        "check",
                    ],
                },
                "message": {"type": "string"},
                "every_seconds": {"type": "integer"},
                "cron_expr": {"type": "string"},
                "tz": {"type": "string"},
                "at": {"type": "string"},
                "at_ms": {"type": "integer"},
                "schedule_time": {"type": "string"},
                "schedule_time_ms": {"type": "integer"},
                "deliver": {"type": "boolean"},
                "job_id": {"type": "string"},
                "email_to": {"type": "string"},
                "email_subject": {"type": "string"},
                "email_content": {"type": "string"},
                "attachment_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        at_ms: int | None = None,
        schedule_time: str | None = None,
        schedule_time_ms: int | None = None,
        deliver: bool = False,
        job_id: str | None = None,
        email_to: str | None = None,
        email_subject: str | None = None,
        email_content: str | None = None,
        attachment_paths: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        # Compatibility aliases used by some models/runtimes.
        if not at and schedule_time:
            at = str(schedule_time)
        if at_ms is None and schedule_time_ms is not None:
            try:
                at_ms = int(schedule_time_ms)
            except Exception:
                at_ms = None
        if not at and isinstance(kwargs.get("scheduleAt"), str):
            at = str(kwargs.get("scheduleAt") or "")
        if at_ms is None and kwargs.get("scheduleAtMs") is not None:
            try:
                at_ms = int(kwargs.get("scheduleAtMs"))
            except Exception:
                at_ms = None
        if action == "add":
            return self._add_job(
                message=message,
                every_seconds=every_seconds,
                cron_expr=cron_expr,
                tz=tz,
                at=at,
                at_ms=at_ms,
                deliver=bool(deliver),
                email_to=email_to,
                email_subject=email_subject,
                email_content=email_content,
                attachment_paths=attachment_paths,
            )
        if action == "list":
            return self._list_jobs()
        if action in {"remove", "cancel"}:
            return self._remove_job(job_id)
        if action == "enable":
            return self._enable_job(job_id, True)
        if action == "disable":
            return self._enable_job(job_id, False)
        if action == "run":
            return await self._run_job(job_id)
        if action == "status":
            return self._status()
        if action == "check":
            return self._check(job_id)
        return f"Unknown action: {action}"

    def _add_job(
        self,
        *,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
        at_ms: int | None,
        deliver: bool,
        email_to: str | None,
        email_subject: str | None,
        email_content: str | None,
        attachment_paths: list[str] | None,
    ) -> str:
        normalized_email_to = (
            str(email_to or "").strip() or self._default_email_recipient()
        )
        normalized_email_content = str(email_content or "").strip()
        normalized_attachments = self._normalize_attachment_paths(attachment_paths)
        direct_email_requested = bool(
            normalized_email_to or normalized_email_content or normalized_attachments
        )
        if not message and not direct_email_requested:
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
        if tz and not cron_expr:
            return "Error: tz can only be used with cron_expr"

        resolved_at_ms = int(at_ms) if at_ms else parsed_at_ms
        if resolved_at_ms is not None:
            schedule = CronSchedule(kind="at", at_ms=resolved_at_ms)
        elif every_seconds:
            schedule = CronSchedule(kind="every", every_ms=int(every_seconds) * 1000)
        else:
            schedule = CronSchedule(
                kind="cron",
                expr=str(cron_expr or "").strip(),
                tz=(str(tz).strip() if tz else None),
            )

        payload_message = str(message or "").strip()
        if direct_email_requested:
            if not normalized_email_to:
                return (
                    "Error: email_to is required for scheduled email jobs "
                    "unless the current chat_id is an email address."
                )
            if not normalized_email_content:
                normalized_email_content = payload_message
            if not normalized_email_content:
                return "Error: email_content or message is required for scheduled email jobs"
            payload_message = payload_message or normalized_email_content
            payload = CronPayload(
                kind="send_email",
                message=payload_message,
                deliver=bool(deliver),
                channel=self._channel,
                to=self._chat_id,
                email_to=normalized_email_to,
                email_subject=str(email_subject or "").strip()
                or "Scheduled message from Annolid Bot",
                email_content=normalized_email_content,
                attachment_paths=normalized_attachments,
            )
        else:
            payload = CronPayload(
                kind="agent_turn",
                message=payload_message,
                deliver=bool(deliver),
                channel=self._channel,
                to=self._chat_id,
            )
        try:
            job = self._service.add_job(
                name=payload_message[:40],
                schedule=schedule,
                payload=payload,
                delete_after_run=(schedule.kind == "at"),
            )
        except ValueError as exc:
            return f"Error: {exc}"
        if payload.kind == "send_email":
            return f"Created scheduled email job to {payload.email_to} (id: {job.id})"
        return f"Created job '{payload_message[:30]}' (id: {job.id})"

    def _list_jobs(self) -> str:
        jobs = self._service.list_jobs(include_disabled=True)
        if not jobs:
            return "No scheduled jobs."
        lines = []
        for job in jobs:
            if job.schedule.kind == "every":
                mode = f"every={int((job.schedule.every_ms or 0) / 1000)}s"
            elif job.schedule.kind == "cron":
                tz = f", tz={job.schedule.tz}" if job.schedule.tz else ""
                mode = f"cron={job.schedule.expr}{tz}"
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

    def _check(self, job_id: str | None) -> str:
        key = str(job_id or "").strip()
        if not key:
            return self._status()
        rows = self._service.list_jobs(include_disabled=True)
        for job in rows:
            if str(job.id) != key:
                continue
            if job.schedule.kind == "every":
                mode = f"every={int((job.schedule.every_ms or 0) / 1000)}s"
            elif job.schedule.kind == "cron":
                tz = f", tz={job.schedule.tz}" if job.schedule.tz else ""
                mode = f"cron={job.schedule.expr}{tz}"
            else:
                mode = f"at={job.schedule.at_ms}"
            state = "enabled" if job.enabled else "disabled"
            return (
                "Cron job: "
                f"id={job.id} name={job.name!r} state={state} mode={mode} "
                f"next_run_at_ms={job.state.next_run_at_ms} "
                f"last_run_at_ms={job.state.last_run_at_ms} "
                f"last_status={job.state.last_status} "
                f"last_error={job.state.last_error!r}"
            )
        return f"Job {key} not found"

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

    def _default_email_recipient(self) -> str:
        _, addr = parseaddr(str(self._chat_id or "").strip())
        return addr if "@" in addr else ""

    @staticmethod
    def _normalize_attachment_paths(paths: list[str] | None) -> list[str]:
        if not isinstance(paths, list):
            return []
        normalized: list[str] = []
        for item in paths:
            text = str(item or "").strip()
            if text:
                normalized.append(text)
        return normalized

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
