from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from annolid.core.agent.observability import emit_governance_event
from annolid.core.agent.security_policy import require_signed_updates
from annolid.version import get_version

from .canary import CanaryPolicy, evaluate_canary
from .history import record_update_history
from .manifest import UpdateManifest, fetch_channel_manifest, release_page_url
from .rollback import build_rollback_plan, execute_rollback
from .verify import VerificationResult, verify_manifest


def _find_source_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    cur = Path(__file__).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "pyproject.toml").exists() and (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


def _detect_install_mode(source_root: Path) -> str:
    if getattr(sys, "frozen", False):
        return "frozen"
    if (source_root / ".git").exists() and (source_root / "pyproject.toml").exists():
        return "source"
    return "package"


def _use_uv_for_source_update() -> bool:
    raw = str(os.getenv("ANNOLID_UPDATE_USE_UV", "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(shutil.which("uv"))


def _has_pip_module() -> bool:
    try:
        return importlib.util.find_spec("pip") is not None
    except Exception:
        return False


def _build_apply_commands(
    *, install_mode: str, channel: str, project: str
) -> List[List[str]]:
    mode = str(install_mode or "package").strip().lower()
    if mode == "frozen":
        return []
    if mode == "source":
        use_uv = _use_uv_for_source_update()
        reinstall_cmd = (
            ["uv", "pip", "install", "-e", "."]
            if use_uv
            else [sys.executable, "-m", "pip", "install", "-e", "."]
        )
        return [
            ["git", "fetch", "--all", "--tags"],
            ["git", "pull", "--ff-only"],
            reinstall_cmd,
        ]
    if not _has_pip_module() and shutil.which("uv"):
        cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--upgrade",
            project,
        ]
        if str(channel).lower() in {"beta", "dev"}:
            cmd.append("--prerelease=allow")
        return [cmd]
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", project]
    if str(channel).lower() in {"beta", "dev"}:
        cmd.append("--pre")
    return [cmd]


def _build_post_check_commands() -> List[List[str]]:
    return [
        [
            sys.executable,
            "-c",
            "import annolid, annolid.version; print(annolid.version.get_version())",
        ],
        [sys.executable, "-m", "annolid.engine.cli", "validate-agent-tools"],
    ]


def _build_subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    exe_dir = str(Path(sys.executable).resolve().parent)
    existing_path = env.get("PATH", "")
    path_parts = [p for p in existing_path.split(os.pathsep) if p]
    if exe_dir not in path_parts:
        env["PATH"] = (
            f"{exe_dir}{os.pathsep}{existing_path}" if existing_path else exe_dir
        )
    if sys.prefix != sys.base_prefix and not env.get("VIRTUAL_ENV"):
        env["VIRTUAL_ENV"] = sys.prefix
    return env


def _probe_git_metadata(source_root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "git_branch": "",
        "git_commit": "",
        "git_dirty": False,
        "git_upstream": "",
    }
    if not shutil.which("git"):
        return info

    def _git_out(args: List[str]) -> str | None:
        try:
            proc = subprocess.run(  # noqa: S603
                ["git", *args],
                cwd=str(source_root),
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if proc.returncode == 0:
                return str(proc.stdout or "").strip()
        except Exception:
            return None
        return None

    branch = _git_out(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch is not None:
        info["git_branch"] = branch
    commit = _git_out(["rev-parse", "--short", "HEAD"])
    if commit is not None:
        info["git_commit"] = commit
    status_out = _git_out(["status", "--porcelain"])
    if status_out is not None:
        info["git_dirty"] = bool(status_out.strip())
    upstream = _git_out(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream is not None:
        info["git_upstream"] = upstream
    return info


def _summarize_step_failure(step: Dict[str, Any]) -> str:
    cmd = " ".join(str(part) for part in (step.get("command") or []))
    detail = str(
        step.get("error") or step.get("stderr_tail") or step.get("stdout_tail") or ""
    ).strip()
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    tail = "\n".join(lines[-4:]) if lines else f"exit code {step.get('returncode')}"
    if cmd:
        return f"Command failed: {cmd}\n{tail}"
    return tail


def _run_step(
    *,
    name: str,
    command: List[str],
    cwd: str | None,
    execute: bool,
    on_output: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    timeout_s: float = 300.0,
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
    if cancel_event is not None and cancel_event.is_set():
        row["ok"] = False
        row["returncode"] = -2
        row["cancelled"] = True
        row["stderr_tail"] = "Cancelled by user."
        return row

    env = _build_subprocess_env()
    if on_output is None and cancel_event is None:
        try:
            proc = subprocess.run(  # noqa: S603
                command,
                cwd=cwd,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            row["returncode"] = -1
            row["ok"] = False
            row["error"] = f"Command timed out after {int(timeout_s)}s."
            row["stderr_tail"] = row["error"]
            if exc.stdout:
                row["stdout_tail"] = str(exc.stdout)[-1000:]
            return row
        except OSError as exc:
            row["returncode"] = -1
            row["ok"] = False
            row["error"] = str(exc)
            row["stderr_tail"] = str(exc)
            return row

        row["returncode"] = int(proc.returncode)
        row["ok"] = proc.returncode == 0
        if proc.stdout:
            row["stdout_tail"] = proc.stdout[-1000:]
        if proc.stderr:
            row["stderr_tail"] = proc.stderr[-1000:]
        return row

    if on_output is not None:
        on_output(f"$ {' '.join(command)}")
    collected: List[str] = []
    try:
        proc = subprocess.Popen(  # noqa: S603
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        row["returncode"] = -1
        row["ok"] = False
        row["error"] = str(exc)
        row["stderr_tail"] = str(exc)
        if on_output is not None:
            on_output(f"Error: {exc}")
        return row

    def _drain_output() -> None:
        assert proc.stdout is not None
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\r\n")
                collected.append(line)
                if on_output is not None:
                    on_output(line)
        except Exception:
            pass

    reader = threading.Thread(target=_drain_output, daemon=True)
    reader.start()

    deadline = time.monotonic() + max(1.0, float(timeout_s))
    cancelled = False
    timed_out = False
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        if time.monotonic() > deadline:
            timed_out = True
            break
        time.sleep(0.1)

    if cancelled or timed_out:
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass
    reader.join(timeout=3.0)

    joined = "\n".join(collected)
    if cancelled:
        row["returncode"] = -2
        row["ok"] = False
        row["cancelled"] = True
        row["error"] = "Cancelled by user."
        row["stderr_tail"] = "Cancelled by user."
        if joined:
            row["stdout_tail"] = joined[-1000:]
        return row
    if timed_out:
        row["returncode"] = -1
        row["ok"] = False
        row["error"] = f"Command timed out after {int(timeout_s)}s."
        row["stderr_tail"] = row["error"]
        if joined:
            row["stdout_tail"] = joined[-1000:]
        return row

    rc = int(proc.returncode or 0)
    row["returncode"] = rc
    row["ok"] = rc == 0
    if joined:
        if rc == 0:
            row["stdout_tail"] = joined[-1000:]
        else:
            row["stderr_tail"] = joined[-1000:]
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
    preflight: Dict[str, Any] = field(default_factory=dict)

    @property
    def update_available(self) -> bool:
        return bool(self.verification.ok)

    @property
    def up_to_date(self) -> bool:
        return str(self.verification.reason or "") == "not_newer_than_current"

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
            "up_to_date": self.up_to_date,
            "release_url": release_page_url(self.project, self.manifest.version),
            "preflight": dict(self.preflight),
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
        payload: Dict[str, Any] = {
            "phase": "preflight",
            "ok": len(missing) == 0,
            "install_mode": install_mode,
            "source_root": str(source_root),
            "missing_binaries": missing,
        }
        if install_mode == "source" and not missing:
            payload.update(_probe_git_metadata(source_root))
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
        effective_require_signature = bool(
            require_signature or require_signed_updates()
        )
        preflight = self.preflight()
        source_root = Path(str(preflight["source_root"]))
        install_mode = str(preflight["install_mode"])
        manifest = fetch_channel_manifest(
            project=self.project,
            channel=channel,
            timeout_s=timeout_s,
        )
        verification = verify_manifest(
            manifest,
            require_signature=effective_require_signature,
        )
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
            preflight=preflight,
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
                "require_signature": bool(effective_require_signature),
                "require_signature_requested": bool(require_signature),
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
        on_output: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
        force_source_sync: bool = False,
    ) -> Dict[str, Any]:
        payload = plan.to_dict()
        payload["status"] = "ok" if plan.update_available else "blocked"
        payload["executed"] = bool(execute)
        payload["pipeline"] = []
        payload["rollback"] = None
        payload["restart_required"] = False
        payload["canary"] = None

        allow_source_sync = bool(
            force_source_sync and plan.install_mode == "source" and plan.up_to_date
        )

        if not plan.update_available and not allow_source_sync:
            if plan.up_to_date:
                payload["status"] = "up_to_date"
                payload["reason"] = "already_up_to_date"
                payload["message"] = (
                    f"{plan.project.title()} {plan.current_version} is already up to date "
                    f"(latest on {plan.channel}: {plan.manifest.version})."
                )
            else:
                payload["status"] = "blocked"
                payload["reason"] = plan.verification.reason
                payload["message"] = (
                    f"Update blocked by verification ({plan.verification.reason})."
                )
            emit_governance_event(
                event_type="update",
                action="run",
                outcome=payload["status"],
                actor="operator" if execute else "system",
                details={
                    "project": plan.project,
                    "channel": plan.channel,
                    "reason": payload["reason"],
                    "execute": bool(execute),
                },
            )
            return payload

        if plan.install_mode == "frozen":
            release_url = release_page_url(plan.project, plan.manifest.version)
            payload["status"] = "manual_required"
            payload["reason"] = "frozen_bundle_manual_update"
            payload["release_url"] = release_url
            payload["message"] = (
                f"Annolid {plan.manifest.version} is available. Desktop app bundles "
                f"cannot self-update in place; download the latest release from {release_url}."
            )
            emit_governance_event(
                event_type="update",
                action="run",
                outcome="manual_required",
                actor="operator" if execute else "system",
                details={
                    "project": plan.project,
                    "channel": plan.channel,
                    "reason": payload["reason"],
                    "release_url": release_url,
                },
            )
            return payload

        preflight = dict(plan.preflight or {})
        missing_bins = list(preflight.get("missing_binaries") or [])
        if execute and missing_bins:
            payload["status"] = "failed"
            payload["reason"] = "missing_required_binaries"
            payload["message"] = (
                "Required binaries are missing from PATH: "
                + ", ".join(str(b) for b in missing_bins)
            )
            return payload

        if execute and plan.install_mode == "source" and preflight.get("git_dirty"):
            payload["status"] = "failed"
            payload["reason"] = "dirty_working_tree"
            payload["message"] = (
                "Your local git checkout has uncommitted changes. "
                "Commit or stash them before running a source update."
            )
            return payload

        previous_commit = str(preflight.get("git_commit") or "")
        rollback_plan = build_rollback_plan(
            install_mode=plan.install_mode,
            project=plan.project,
            previous_version=plan.current_version,
            previous_commit=previous_commit,
            source_root=plan.source_root,
        )
        payload["rollback_plan"] = rollback_plan.to_dict()

        # apply
        for cmd in plan.commands:
            step = _run_step(
                name="apply",
                command=list(cmd),
                cwd=(plan.source_root if plan.install_mode == "source" else None),
                execute=execute,
                on_output=on_output,
                cancel_event=cancel_event,
            )
            payload["pipeline"].append(step)
            if step.get("cancelled"):
                payload["status"] = "cancelled"
                payload["reason"] = "cancelled_by_user"
                payload["message"] = "Update cancelled by user."
                return payload
            if not step.get("ok", False):
                payload["status"] = "failed"
                payload["reason"] = "apply_command_failed"
                payload["failed_step"] = dict(step)
                payload["message"] = _summarize_step_failure(step)
                payload["rollback"] = execute_rollback(
                    rollback_plan, execute=execute, on_output=on_output
                )
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
                        "reason": payload["reason"],
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
            checks = _build_post_check_commands()
            for cmd in checks:
                step = _run_step(
                    name="post_check",
                    command=list(cmd),
                    cwd=None,
                    execute=execute,
                    on_output=on_output,
                    cancel_event=cancel_event,
                )
                payload["pipeline"].append(step)
                if step.get("cancelled"):
                    payload["status"] = "cancelled"
                    payload["reason"] = "cancelled_by_user"
                    payload["message"] = "Update cancelled during post-check."
                    return payload
                if execute and not step.get("ok", False):
                    payload["status"] = "failed_post_check"
                    payload["reason"] = "post_check_failed"
                    payload["failed_step"] = dict(step)
                    payload["message"] = _summarize_step_failure(step)
                    payload["rollback"] = execute_rollback(
                        rollback_plan,
                        execute=execute,
                        on_output=on_output,
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
                            "reason": payload["reason"],
                        },
                    )
                    return payload

        if canary_metrics is not None:
            policy = canary_policy or CanaryPolicy()
            canary = evaluate_canary(canary_metrics, policy=policy)
            payload["canary"] = canary.to_dict()
            if execute and not canary.passed:
                payload["status"] = "failed_canary"
                payload["reason"] = canary.reason
                payload["message"] = f"Canary validation failed: {canary.reason}"
                payload["rollback"] = execute_rollback(
                    rollback_plan, execute=execute, on_output=on_output
                )
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
        if execute:
            payload["message"] = (
                f"Updated {plan.project} to {plan.manifest.version} successfully."
                if not allow_source_sync
                else f"Synced source checkout for {plan.project} successfully."
            )
            record_update_history(
                project=plan.project,
                channel=plan.channel,
                install_mode=plan.install_mode,
                previous_version=plan.current_version,
                target_version=plan.manifest.version,
                status=payload["status"],
                previous_commit=previous_commit,
                source_root=plan.source_root,
            )
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
