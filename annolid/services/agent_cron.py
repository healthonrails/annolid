"""Service-layer orchestration for agent cron and workspace admin commands."""

from __future__ import annotations

import asyncio
import datetime
import shutil
from pathlib import Path
from typing import Dict


def _default_agent_cron_store_path() -> Path:
    from annolid.core.agent.cron import default_cron_store_path

    return default_cron_store_path()


def _agent_cron_service():
    from annolid.core.agent.cron import CronService

    return CronService(store_path=_default_agent_cron_store_path())


def _bootstrap_backup_root(workspace: Path) -> Path:
    return workspace / ".annolid" / "bootstrap-backups"


_WORKSPACE_TEMPLATE_PATHS = (
    "AGENTS.md",
    "SOUL.md",
    "USER.md",
    "IDENTITY.md",
    "TOOLS.md",
    "HEARTBEAT.md",
    "BOOT.md",
    "BOOTSTRAP.md",
    "memory/MEMORY.md",
    "memory/HISTORY.md",
)


def _workspace_template_status(workspace: Path) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for rel in _WORKSPACE_TEMPLATE_PATHS:
        out[rel] = (workspace / rel).exists()
    return out


def _looks_like_annolid_repo_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").exists()
        and (path / "annolid").is_dir()
        and (path / "tests").is_dir()
    )


def _build_workspace_guidance(
    workspace: Path, template_status: Dict[str, bool]
) -> list[str]:
    missing = [rel for rel, ok in template_status.items() if not ok]
    guidance: list[str] = []
    if missing:
        guidance.append(
            "Workspace bootstrap files are missing; run onboarding to initialize templates."
        )
    else:
        guidance.append("Workspace bootstrap templates are present.")

    if _looks_like_annolid_repo_root(workspace):
        guidance.append(
            "Selected workspace looks like an Annolid source repository root; prefer ~/.annolid/workspace to avoid mixing runtime memory with project source control."
        )
    return guidance


def _workspace_health_payload(workspace: Path) -> dict:
    templates = _workspace_template_status(workspace)
    missing = sorted([rel for rel, ok in templates.items() if not ok])
    return {
        "workspace": str(workspace),
        "template_count": len(templates),
        "template_present_count": len(templates) - len(missing),
        "template_missing_count": len(missing),
        "template_missing": missing,
        "looks_like_repo_root": bool(_looks_like_annolid_repo_root(workspace)),
        "guidance": _build_workspace_guidance(workspace, templates),
    }


def _list_backup_dirs(backup_root: Path) -> list[Path]:
    if not backup_root.exists():
        return []
    return sorted(
        [p for p in backup_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )


def onboard_agent_workspace(
    *,
    workspace: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    backup: bool = True,
    backup_dir: str | None = None,
    prune_bootstrap: bool = False,
) -> dict:
    from annolid.core.agent import bootstrap_workspace, prune_bootstrap_workspace
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    health_before = _workspace_health_payload(resolved_workspace)
    should_backup = bool(overwrite) and bool(backup)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    resolved_backup_dir = (
        (Path(backup_dir).expanduser() if backup_dir else None)
        if should_backup
        else None
    )
    if should_backup and resolved_backup_dir is None:
        resolved_backup_dir = (
            resolved_workspace / ".annolid" / "bootstrap-backups" / timestamp
        )

    prune_outcomes = (
        prune_bootstrap_workspace(
            resolved_workspace,
            dry_run=bool(dry_run),
            backup_root=resolved_backup_dir,
        )
        if prune_bootstrap
        else {}
    )
    outcomes = bootstrap_workspace(
        resolved_workspace,
        overwrite=bool(overwrite),
        dry_run=bool(dry_run),
        backup_root=resolved_backup_dir,
    )

    counts: Dict[str, int] = {}
    for status in list(outcomes.values()) + list(prune_outcomes.values()):
        counts[status] = counts.get(status, 0) + 1
    health_after = (
        _workspace_health_payload(resolved_workspace) if not dry_run else health_before
    )
    return {
        "workspace": str(resolved_workspace),
        "overwrite": bool(overwrite),
        "dry_run": bool(dry_run),
        "prune_bootstrap": bool(prune_bootstrap),
        "backup_enabled": bool(should_backup),
        "backup_dir": (str(resolved_backup_dir) if resolved_backup_dir else None),
        "summary": counts,
        "files": outcomes,
        "pruned_files": prune_outcomes,
        "workspace_health_before": health_before,
        "workspace_health_after": health_after,
    }


def get_agent_status(*, workspace: str | None = None) -> dict:
    from annolid.core.agent.utils import get_agent_data_path, get_agent_workspace_path

    data_dir = get_agent_data_path()
    resolved_workspace = get_agent_workspace_path(workspace)
    store_path = _default_agent_cron_store_path()
    cron_status = _agent_cron_service().status()
    backup_root = _bootstrap_backup_root(resolved_workspace)
    backups = _list_backup_dirs(backup_root)
    templates = _workspace_template_status(resolved_workspace)
    health = _workspace_health_payload(resolved_workspace)
    return {
        "data_dir": str(data_dir),
        "workspace": str(resolved_workspace),
        "workspace_templates": templates,
        "workspace_health": health,
        "cron_store_path": str(store_path),
        "workspace_backup_root": str(backup_root),
        "workspace_backup_count": len(backups),
        "workspace_latest_backup": (str(backups[0]) if backups else None),
        "cron": cron_status,
    }


def restore_agent_workspace_backup(
    *,
    workspace: str | None = None,
    backup_dir: str | None = None,
    latest: bool = True,
    dry_run: bool = False,
    backup_before_restore: bool = True,
) -> dict:
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    backup_root = _bootstrap_backup_root(resolved_workspace)
    selected_backup: Path | None = None
    if backup_dir:
        candidate = Path(backup_dir).expanduser()
        selected_backup = candidate if candidate.is_dir() else None
    elif latest:
        dirs = _list_backup_dirs(backup_root)
        selected_backup = dirs[0] if dirs else None

    if selected_backup is None:
        return {
            "workspace": str(resolved_workspace),
            "restored": False,
            "reason": "No backup directory found to restore from.",
            "backup_dir": None,
            "dry_run": bool(dry_run),
        }

    files = sorted([p for p in selected_backup.rglob("*") if p.is_file()])
    restore_backup_dir = None
    if backup_before_restore:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        restore_backup_dir = backup_root / f"restore-pre-{stamp}"

    outcomes: Dict[str, str] = {}
    for src in files:
        rel = src.relative_to(selected_backup).as_posix()
        dst = resolved_workspace / rel
        existed = dst.exists()
        if dry_run:
            outcomes[rel] = "would_overwrite" if existed else "would_restore"
            continue
        if existed and restore_backup_dir is not None:
            backup_target = restore_backup_dir / rel
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dst, backup_target)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        outcomes[rel] = "overwritten" if existed else "restored"

    summary: Dict[str, int] = {}
    for status in outcomes.values():
        summary[status] = summary.get(status, 0) + 1

    return {
        "workspace": str(resolved_workspace),
        "restored": True,
        "dry_run": bool(dry_run),
        "backup_dir": str(selected_backup),
        "pre_restore_backup_dir": (
            str(restore_backup_dir)
            if (restore_backup_dir is not None and not dry_run)
            else None
        ),
        "summary": summary,
        "files": outcomes,
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
    "restore_agent_workspace_backup",
    "remove_agent_cron_job",
    "run_agent_cron_job",
    "set_agent_cron_job_enabled",
]
