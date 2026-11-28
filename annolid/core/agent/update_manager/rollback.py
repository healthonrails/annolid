from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from annolid.core.agent.observability import emit_governance_event


@dataclass(frozen=True)
class RollbackPlan:
    install_mode: str
    project: str
    previous_version: str
    commands: List[List[str]]
    manual_required: bool
    manual_instructions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "install_mode": self.install_mode,
            "project": self.project,
            "previous_version": self.previous_version,
            "commands": [list(cmd) for cmd in self.commands],
            "manual_required": self.manual_required,
        }
        if self.manual_instructions:
            payload["manual_instructions"] = self.manual_instructions
        return payload


def _package_install_command(spec: str) -> List[str]:
    if importlib.util.find_spec("pip") is None:
        uv_bin = shutil.which("uv")
        if uv_bin:
            return ["uv", "pip", "install", "--python", sys.executable, spec]
    return [sys.executable, "-m", "pip", "install", spec]


def build_rollback_plan(
    *,
    install_mode: str,
    project: str,
    previous_version: str,
    previous_commit: str = "",
    source_root: str = "",
) -> RollbackPlan:
    mode = str(install_mode or "package").strip().lower()
    prev = str(previous_version or "").strip()
    commit = str(previous_commit or "").strip()
    root = str(source_root or "").strip()
    if mode == "package" and prev:
        return RollbackPlan(
            install_mode=mode,
            project=project,
            previous_version=prev,
            commands=[_package_install_command(f"{project}=={prev}")],
            manual_required=False,
        )
    if mode == "source":
        if commit:
            prefix = f"cd {root} && " if root else ""
            hint = f"{prefix}git checkout {commit} && uv pip install -e ."
        else:
            hint = (
                "Source checkout rollback requires git. Run `git log -n 5` and "
                "`git checkout <previous-commit>` in your Annolid repository."
            )
        return RollbackPlan(
            install_mode=mode,
            project=project,
            previous_version=prev,
            commands=[],
            manual_required=True,
            manual_instructions=hint,
        )
    if mode == "frozen":
        return RollbackPlan(
            install_mode=mode,
            project=project,
            previous_version=prev,
            commands=[],
            manual_required=True,
            manual_instructions=(
                "Desktop bundle rollback requires reinstalling the previous "
                "Annolid release archive from GitHub Releases."
            ),
        )
    return RollbackPlan(
        install_mode=mode,
        project=project,
        previous_version=prev,
        commands=[],
        manual_required=True,
        manual_instructions="Specify a previous known-good version to roll back.",
    )


def execute_rollback(
    plan: RollbackPlan,
    *,
    execute: bool = False,
    on_output: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = plan.to_dict()
    payload["executed"] = bool(execute)
    payload["ok"] = True
    payload["steps"] = []
    if plan.manual_required:
        payload["ok"] = False if execute else True
        payload["reason"] = "manual_required"
        if plan.manual_instructions:
            payload["message"] = plan.manual_instructions
        emit_governance_event(
            event_type="update",
            action="rollback",
            outcome="manual_required" if execute else "planned",
            actor="operator" if execute else "system",
            details=payload,
        )
        return payload
    for cmd in plan.commands:
        step: Dict[str, Any] = {"command": list(cmd), "ok": True, "returncode": 0}
        if execute:
            if on_output is not None:
                on_output(f"$ {' '.join(cmd)}")
            try:
                proc = subprocess.run(  # noqa: S603
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                step["returncode"] = -1
                step["ok"] = False
                step["error"] = str(exc)
                step["stderr_tail"] = str(exc)
                payload["ok"] = False
                payload["reason"] = "rollback_command_failed"
                payload["message"] = str(exc)
                if on_output is not None:
                    on_output(f"Error: {exc}")
                payload["steps"].append(step)
                break
            step["returncode"] = int(proc.returncode)
            step["ok"] = proc.returncode == 0
            if proc.stdout:
                step["stdout_tail"] = proc.stdout[-1000:]
                if on_output is not None:
                    on_output(proc.stdout.rstrip())
            if proc.stderr:
                step["stderr_tail"] = proc.stderr[-1000:]
                if on_output is not None:
                    on_output(proc.stderr.rstrip())
            if proc.returncode != 0:
                payload["ok"] = False
                payload["reason"] = "rollback_command_failed"
                detail = (proc.stderr or proc.stdout or "").strip()
                if detail:
                    payload["message"] = detail.splitlines()[-1]
        payload["steps"].append(step)
    emit_governance_event(
        event_type="update",
        action="rollback",
        outcome="ok" if bool(payload.get("ok", False)) else "failed",
        actor="operator" if execute else "system",
        details=payload,
    )
    return payload
