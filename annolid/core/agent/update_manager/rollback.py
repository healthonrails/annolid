from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

from annolid.core.agent.observability import emit_governance_event


@dataclass(frozen=True)
class RollbackPlan:
    install_mode: str
    project: str
    previous_version: str
    commands: List[List[str]]
    manual_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "install_mode": self.install_mode,
            "project": self.project,
            "previous_version": self.previous_version,
            "commands": [list(cmd) for cmd in self.commands],
            "manual_required": self.manual_required,
        }


def build_rollback_plan(
    *,
    install_mode: str,
    project: str,
    previous_version: str,
) -> RollbackPlan:
    mode = str(install_mode or "package").strip().lower()
    prev = str(previous_version or "").strip()
    if mode == "package" and prev:
        return RollbackPlan(
            install_mode=mode,
            project=project,
            previous_version=prev,
            commands=[[sys.executable, "-m", "pip", "install", f"{project}=={prev}"]],
            manual_required=False,
        )
    return RollbackPlan(
        install_mode=mode,
        project=project,
        previous_version=prev,
        commands=[],
        manual_required=True,
    )


def execute_rollback(plan: RollbackPlan, *, execute: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = plan.to_dict()
    payload["executed"] = bool(execute)
    payload["ok"] = True
    payload["steps"] = []
    if plan.manual_required:
        payload["ok"] = False if execute else True
        payload["reason"] = "manual_required"
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
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603
            step["returncode"] = int(proc.returncode)
            step["ok"] = proc.returncode == 0
            if proc.stdout:
                step["stdout_tail"] = proc.stdout[-1000:]
            if proc.stderr:
                step["stderr_tail"] = proc.stderr[-1000:]
            if proc.returncode != 0:
                payload["ok"] = False
                payload["reason"] = "rollback_command_failed"
        payload["steps"].append(step)
    emit_governance_event(
        event_type="update",
        action="rollback",
        outcome="ok" if bool(payload.get("ok", False)) else "failed",
        actor="operator" if execute else "system",
        details=payload,
    )
    return payload
