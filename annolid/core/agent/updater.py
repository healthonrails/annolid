from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from annolid.core.agent.update_manager import SignedUpdateManager


@dataclass(frozen=True)
class UpdateReport:
    status: str
    project: str
    channel: str
    current_version: str
    latest_version: str
    update_available: bool
    install_mode: str
    source_root: str
    check_url: str
    commands: List[List[str]]
    require_signature: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "project": self.project,
            "channel": self.channel,
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "update_available": self.update_available,
            "install_mode": self.install_mode,
            "source_root": self.source_root,
            "check_url": self.check_url,
            "commands": [list(cmd) for cmd in self.commands],
            "require_signature": bool(self.require_signature),
        }


def check_for_updates(
    *,
    channel: str = "stable",
    project: str = "annolid",
    timeout_s: float = 4.0,
    require_signature: bool = False,
) -> UpdateReport:
    manager = SignedUpdateManager(project=project)
    plan = manager.stage(
        channel=channel,
        timeout_s=timeout_s,
        require_signature=bool(require_signature),
    )
    return UpdateReport(
        status="update_available" if plan.update_available else "ok",
        project=plan.project,
        channel=plan.channel,
        current_version=plan.current_version,
        latest_version=plan.manifest.version,
        update_available=bool(plan.update_available),
        install_mode=plan.install_mode,
        source_root=plan.source_root,
        check_url=plan.manifest.source,
        commands=[list(c) for c in plan.commands],
        require_signature=bool(require_signature),
    )


def apply_update(
    report: UpdateReport,
    *,
    execute: bool = False,
    run_doctor: bool = True,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = report.to_dict()
    payload["executed"] = bool(execute)
    payload["updated"] = False
    payload["steps"] = []
    if not report.update_available:
        return payload

    manager = SignedUpdateManager(project=report.project)
    plan = manager.stage(
        channel=report.channel,
        timeout_s=4.0,
        require_signature=bool(report.require_signature),
    )
    run_payload = manager.run(
        plan,
        execute=execute,
        run_post_check=bool(run_doctor),
    )

    # Compatibility projection for existing callers/tests
    pipeline = run_payload.get("pipeline")
    if isinstance(pipeline, list):
        payload["steps"] = list(pipeline)
    payload["status"] = str(run_payload.get("status") or payload.get("status") or "ok")
    payload["updated"] = bool(run_payload.get("updated", execute))
    payload["restart_required"] = bool(run_payload.get("restart_required", execute))
    if "rollback" in run_payload:
        payload["rollback"] = run_payload.get("rollback")
    if "rollback_plan" in run_payload:
        payload["rollback_plan"] = run_payload.get("rollback_plan")
    if "verification" in run_payload:
        payload["verification"] = run_payload.get("verification")
    return payload
