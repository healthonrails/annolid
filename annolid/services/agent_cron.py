"""Service-layer orchestration for agent cron and workspace admin commands."""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path


def _default_agent_cron_store_path() -> Path:
    from annolid.core.agent.cron import default_cron_store_path

    return default_cron_store_path()


def _agent_cron_service():
    from annolid.core.agent.cron import CronService

    return CronService(store_path=_default_agent_cron_store_path())


def onboard_agent_workspace(
    *, workspace: str | None = None, overwrite: bool = False
) -> dict:
    from annolid.core.agent import bootstrap_workspace
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    outcomes = bootstrap_workspace(resolved_workspace, overwrite=bool(overwrite))
    return {
        "workspace": str(resolved_workspace),
        "overwrite": bool(overwrite),
        "files": outcomes,
    }


def get_agent_status() -> dict:
    from annolid.core.agent.utils import get_agent_data_path, get_agent_workspace_path

    data_dir = get_agent_data_path()
    workspace = get_agent_workspace_path()
    store_path = _default_agent_cron_store_path()
    cron_status = _agent_cron_service().status()
    return {
        "data_dir": str(data_dir),
        "workspace": str(workspace),
        "workspace_templates": {
            "AGENTS.md": (workspace / "AGENTS.md").exists(),
            "SOUL.md": (workspace / "SOUL.md").exists(),
            "USER.md": (workspace / "USER.md").exists(),
            "TOOLS.md": (workspace / "TOOLS.md").exists(),
            "HEARTBEAT.md": (workspace / "HEARTBEAT.md").exists(),
            "memory/MEMORY.md": (workspace / "memory" / "MEMORY.md").exists(),
            "memory/HISTORY.md": (workspace / "memory" / "HISTORY.md").exists(),
        },
        "cron_store_path": str(store_path),
        "cron": cron_status,
    }


def list_agent_cron_jobs(*, include_all: bool = False) -> list[dict]:
    jobs = _agent_cron_service().list_jobs(include_disabled=bool(include_all))
    rows = []
    for j in jobs:
        rows.append(
            {
                "id": j.id,
                "name": j.name,
                "enabled": j.enabled,
                "schedule": {
                    "kind": j.schedule.kind,
                    "at_ms": j.schedule.at_ms,
                    "every_ms": j.schedule.every_ms,
                    "expr": j.schedule.expr,
                    "tz": j.schedule.tz,
                },
                "payload": {
                    "message": j.payload.message,
                    "deliver": j.payload.deliver,
                    "channel": j.payload.channel,
                    "to": j.payload.to,
                },
                "state": {
                    "next_run_at_ms": j.state.next_run_at_ms,
                    "last_run_at_ms": j.state.last_run_at_ms,
                    "last_status": j.state.last_status,
                    "last_error": j.state.last_error,
                },
            }
        )
    return rows


def add_agent_cron_job(
    *,
    name: str,
    message: str,
    deliver: bool = False,
    channel: str | None = None,
    to: str | None = None,
    every: int | None = None,
    cron_expr: str | None = None,
    at: str | None = None,
    tz: str | None = None,
) -> dict:
    from annolid.core.agent.cron import CronPayload, CronSchedule

    def _parse_iso_datetime_ms(raw: str) -> int:
        text = str(raw or "").strip()
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = datetime.datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return int(dt.timestamp() * 1000)

    if every is None and cron_expr is None and at is None:
        raise SystemExit("Specify one of --every, --cron, or --at.")
    if tz is not None and cron_expr is None:
        raise SystemExit("--tz can only be used with --cron.")

    if at is not None:
        try:
            at_ms = _parse_iso_datetime_ms(str(at))
        except ValueError as exc:
            raise SystemExit(f"Invalid --at value: {at}") from exc
        schedule = CronSchedule(kind="at", at_ms=at_ms)
        delete_after_run = True
    elif every is not None:
        every_value = int(every)
        if every_value <= 0:
            raise SystemExit("--every must be > 0")
        schedule = CronSchedule(kind="every", every_ms=every_value * 1000)
        delete_after_run = False
    else:
        schedule = CronSchedule(
            kind="cron",
            expr=str(cron_expr),
            tz=(str(tz) if tz else None),
        )
        delete_after_run = False

    payload = CronPayload(
        kind="agent_turn",
        message=str(message),
        deliver=bool(deliver),
        channel=(str(channel) if channel else None),
        to=(str(to) if to else None),
    )
    try:
        job = _agent_cron_service().add_job(
            name=str(name),
            schedule=schedule,
            payload=payload,
            delete_after_run=delete_after_run,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return {
        "id": job.id,
        "name": job.name,
        "enabled": job.enabled,
        "next_run_at_ms": job.state.next_run_at_ms,
    }


def remove_agent_cron_job(*, job_id: str) -> tuple[dict, int]:
    ok = _agent_cron_service().remove_job(str(job_id))
    return {"removed": bool(ok), "job_id": str(job_id)}, (0 if ok else 1)


def set_agent_cron_job_enabled(*, job_id: str, enabled: bool) -> tuple[dict, int]:
    job = _agent_cron_service().enable_job(str(job_id), enabled=bool(enabled))
    if job is None:
        return {"updated": False, "job_id": str(job_id)}, 1
    return {"updated": True, "job_id": job.id, "enabled": bool(job.enabled)}, 0


def run_agent_cron_job(*, job_id: str, force: bool = False) -> tuple[dict, int]:
    async def _run() -> bool:
        return await _agent_cron_service().run_job(str(job_id), force=bool(force))

    ok = bool(asyncio.run(_run()))
    return {"ran": ok, "job_id": str(job_id)}, (0 if ok else 1)


__all__ = [
    "add_agent_cron_job",
    "get_agent_status",
    "list_agent_cron_jobs",
    "onboard_agent_workspace",
    "remove_agent_cron_job",
    "run_agent_cron_job",
    "set_agent_cron_job_enabled",
]
