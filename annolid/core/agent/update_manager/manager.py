from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from annolid.core.agent.observability import emit_governance_event
from annolid.version import get_version

from .canary import CanaryPolicy, evaluate_canary
from .manifest import UpdateManifest, fetch_channel_manifest
from .rollback import build_rollback_plan, execute_rollback
from .verify import VerificationResult, verify_manifest


def _find_source_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "pyproject.toml").exists() and (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


def _detect_install_mode(source_root: Path) -> str:
    if (source_root / ".git").exists() and (source_root / "pyproject.toml").exists():
        return "source"
    return "package"


def _build_apply_commands(
    *, install_mode: str, channel: str, project: str
) -> List[List[str]]:
    if install_mode == "source":
        return [
            ["git", "fetch", "--all", "--tags"],
            ["git", "pull", "--ff-only"],
            [sys.executable, "-m", "pip", "install", "-e", "."],
        ]
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", project]
    if str(channel).lower() in {"beta", "dev"}:
        cmd.append("--pre")
    return [cmd]


def _run_step(
    *,
    name: str,
    command: List[str],
    cwd: str | None,
    execute: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "phase": name,
        "command": list(command),
        "cwd": cwd,
        "ok": True,
        "returncode": 0,
    }
    if not execute:
        row["dry_run"] = True
        return row
    proc = subprocess.run(  # noqa: S603
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    row["returncode"] = int(proc.returncode)
    row["ok"] = proc.returncode == 0
    if proc.stdout:
        row["stdout_tail"] = proc.stdout[-1000:]
    if proc.stderr:
        row["stderr_tail"] = proc.stderr[-1000:]
    return row


@dataclass(frozen=True)
class SignedUpdatePlan:
    channel: str
    project: str
    current_version: str
    manifest: UpdateManifest
    install_mode: str
    source_root: str
    commands: List[List[str]]
    verification: VerificationResult

    @property
    def update_available(self) -> bool:
        return bool(self.verification.ok)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "project": self.project,
            "current_version": self.current_version,
            "manifest": self.manifest.to_dict(),
            "install_mode": self.install_mode,
            "source_root": self.source_root,
            "commands": [list(c) for c in self.commands],
            "verification": self.verification.to_dict(),
            "update_available": self.update_available,
        }


class SignedUpdateManager:
    """Preflight → stage → verify → apply → restart → post-check + rollback."""

    def __init__(self, *, project: str = "annolid") -> None:
        self.project = str(project or "annolid").strip() or "annolid"

    def preflight(self) -> Dict[str, Any]:
        source_root = _find_source_root()
        install_mode = _detect_install_mode(source_root)
        required_bins = ["git"] if install_mode == "source" else []
        missing = [name for name in required_bins if not shutil.which(name)]
        payload = {
            "phase": "preflight",
            "ok": len(missing) == 0,
            "install_mode": install_mode,
            "source_root": str(source_root),
            "missing_binaries": missing,
        }
        emit_governance_event(
            event_type="update",
            action="preflight",
            outcome="ok" if payload["ok"] else "failed",
            actor="system",
            details=dict(payload),
        )
        return payload

    def stage(
        self,
        *,
        channel: str = "stable",
        timeout_s: float = 4.0,
        require_signature: bool = False,
    ) -> SignedUpdatePlan:
        preflight = self.preflight()
        source_root = Path(str(preflight["source_root"]))
        install_mode = str(preflight["install_mode"])
        manifest = fetch_channel_manifest(
            project=self.project,
            channel=channel,
            timeout_s=timeout_s,
        )
        verification = verify_manifest(manifest, require_signature=require_signature)
        commands = _build_apply_commands(
            install_mode=install_mode,
            channel=manifest.channel,
            project=self.project,
        )
        plan = SignedUpdatePlan(
            channel=manifest.channel,
            project=self.project,
            current_version=str(get_version() or "0.0.0"),
            manifest=manifest,
            install_mode=install_mode,
            source_root=str(source_root),
            commands=commands,
            verification=verification,
        )
        emit_governance_event(
            event_type="update",
            action="stage",
            outcome="ok" if verification.ok else "blocked",
            actor="system",
            details={
                "project": self.project,
                "channel": manifest.channel,
                "current_version": plan.current_version,
                "target_version": manifest.version,
                "require_signature": bool(require_signature),
                "verification_reason": verification.reason,
                "install_mode": install_mode,
            },
        )
        return plan

    def run(
        self,
        plan: SignedUpdatePlan,
        *,
        execute: bool = False,
        run_post_check: bool = True,
        canary_metrics: Dict[str, Any] | None = None,
        canary_policy: CanaryPolicy | None = None,
    ) -> Dict[str, Any]:
        payload = plan.to_dict()
        payload["status"] = "ok" if plan.update_available else "blocked"
        payload["executed"] = bool(execute)
        payload["pipeline"] = []
        payload["rollback"] = None
        payload["restart_required"] = False
        payload["canary"] = None

        if not plan.update_available:
            payload["reason"] = plan.verification.reason
            emit_governance_event(
                event_type="update",
                action="run",
                outcome="blocked",
                actor="operator" if execute else "system",
                details={
                    "project": plan.project,
                    "channel": plan.channel,
                    "reason": plan.verification.reason,
                    "execute": bool(execute),
                },
            )
            return payload

        rollback_plan = build_rollback_plan(
            install_mode=plan.install_mode,
            project=plan.project,
            previous_version=plan.current_version,
        )
        payload["rollback_plan"] = rollback_plan.to_dict()

        # apply
        for cmd in plan.commands:
            step = _run_step(
                name="apply",
                command=list(cmd),
                cwd=(plan.source_root if plan.install_mode == "source" else None),
                execute=execute,
            )
            payload["pipeline"].append(step)
            if not step.get("ok", False):
                payload["status"] = "failed"
                payload["rollback"] = execute_rollback(rollback_plan, execute=execute)
                emit_governance_event(
                    event_type="update",
                    action="apply",
                    outcome="failed",
                    actor="operator" if execute else "system",
                    details={
                        "project": plan.project,
                        "channel": plan.channel,
                        "execute": bool(execute),
                        "rollback_triggered": True,
                    },
                )
                return payload

        # restart marker (caller handles real restart)
        payload["pipeline"].append(
            {
                "phase": "restart",
                "ok": True,
                "executed": bool(execute),
                "action": "restart_required" if execute else "restart_planned",
            }
        )
        payload["restart_required"] = bool(execute)

        if run_post_check:
            checks = [
                ["annolid-run", "validate-agent-tools"],
                ["annolid-run", "agent-security-check"],
            ]
            for cmd in checks:
                step = _run_step(
                    name="post_check",
                    command=list(cmd),
                    cwd=None,
                    execute=execute,
                )
                payload["pipeline"].append(step)
                if execute and not step.get("ok", False):
                    payload["status"] = "failed_post_check"
                    payload["rollback"] = execute_rollback(
                        rollback_plan,
                        execute=execute,
                    )
                    emit_governance_event(
                        event_type="update",
                        action="post_check",
                        outcome="failed",
                        actor="operator",
                        details={
                            "project": plan.project,
                            "channel": plan.channel,
                            "rollback_triggered": True,
                        },
                    )
                    return payload

        if canary_metrics is not None:
            policy = canary_policy or CanaryPolicy()
            canary = evaluate_canary(canary_metrics, policy=policy)
            payload["canary"] = canary.to_dict()
            if execute and not canary.passed:
                payload["status"] = "failed_canary"
                payload["rollback"] = execute_rollback(rollback_plan, execute=execute)
                emit_governance_event(
                    event_type="update",
                    action="canary",
                    outcome="failed",
                    actor="operator",
                    details={
                        "project": plan.project,
                        "channel": plan.channel,
                        "rollback_triggered": True,
                        "reason": canary.reason,
                    },
                )
                return payload

        payload["status"] = "updated" if execute else "staged"
        payload["updated"] = bool(execute)
        emit_governance_event(
            event_type="update",
            action="run",
            outcome=payload["status"],
            actor="operator" if execute else "system",
            details={
                "project": plan.project,
                "channel": plan.channel,
                "execute": bool(execute),
                "run_post_check": bool(run_post_check),
            },
        )
        return payload
