"""Service-layer orchestration for agent update workflows."""

from __future__ import annotations

import json
from pathlib import Path


def bot_update_requires_operator_consent() -> bool:
    from annolid.core.agent.security_policy import (
        bot_update_requires_operator_consent as _bot_update_requires_operator_consent,
    )

    return bool(_bot_update_requires_operator_consent())


def operator_consent_phrase() -> str:
    from annolid.core.agent.security_policy import (
        operator_consent_phrase as _operator_consent_phrase,
    )

    return str(_operator_consent_phrase() or "")


def has_operator_consent(value: str) -> bool:
    from annolid.core.agent.security_policy import (
        has_operator_consent as _has_operator_consent,
    )

    return bool(_has_operator_consent(value))


def check_for_agent_update(
    *,
    project: str = "annolid",
    channel: str = "stable",
    timeout_s: float = 4.0,
    require_signature: bool = False,
) -> dict:
    from annolid.core.agent.update_manager.manager import SignedUpdateManager

    manager = SignedUpdateManager(project=str(project or "annolid"))
    plan = manager.stage(
        channel=str(channel or "stable"),
        timeout_s=float(timeout_s),
        require_signature=bool(require_signature),
    )
    return plan.to_dict()


def run_agent_update(
    *,
    project: str = "annolid",
    channel: str = "stable",
    timeout_s: float = 4.0,
    require_signature: bool = False,
    execute: bool = False,
    skip_post_check: bool = False,
    canary_metrics: str | Path | None = None,
    canary_min_samples: int = 20,
    canary_max_failure_rate: float = 0.05,
    canary_max_regressions: int = 0,
) -> tuple[dict, int]:
    from annolid.core.agent.update_manager.canary import CanaryPolicy
    from annolid.core.agent.update_manager.manager import SignedUpdateManager

    manager = SignedUpdateManager(project=str(project or "annolid"))
    plan = manager.stage(
        channel=str(channel or "stable"),
        timeout_s=float(timeout_s),
        require_signature=bool(require_signature),
    )
    metrics_payload = None
    if canary_metrics:
        metrics_payload = json.loads(
            Path(canary_metrics).expanduser().resolve().read_text(encoding="utf-8")
        )
    canary_policy = CanaryPolicy(
        min_samples=int(canary_min_samples),
        max_failure_rate=float(canary_max_failure_rate),
        max_regressions=int(canary_max_regressions),
    )
    payload = manager.run(
        plan,
        execute=bool(execute),
        run_post_check=not bool(skip_post_check),
        canary_metrics=metrics_payload,
        canary_policy=canary_policy,
    )
    status = str(payload.get("status") or "").strip().lower()
    return payload, (0 if status in {"staged", "updated"} else 1)


def rollback_agent_update(
    *,
    install_mode: str = "package",
    project: str = "annolid",
    previous_version: str = "",
    execute: bool = False,
) -> tuple[dict, int]:
    from annolid.core.agent.update_manager.rollback import (
        build_rollback_plan,
        execute_rollback,
    )

    plan = build_rollback_plan(
        install_mode=str(install_mode or "package"),
        project=str(project or "annolid"),
        previous_version=str(previous_version or ""),
    )
    payload = execute_rollback(plan, execute=bool(execute))
    return payload, (0 if bool(payload.get("ok", False)) else 1)


__all__ = [
    "bot_update_requires_operator_consent",
    "check_gui_agent_update",
    "execute_gui_agent_rollback",
    "has_operator_consent",
    "operator_consent_phrase",
    "check_for_agent_update",
    "run_legacy_agent_update",
    "rollback_agent_update",
    "run_agent_update",
]


def check_gui_agent_update(
    *,
    project: str = "annolid",
    channel: str = "stable",
    timeout_s: float = 4.0,
    require_signature: bool = False,
) -> dict:
    from annolid.core.agent.update_manager.service import UpdateManagerService

    service = UpdateManagerService(project=str(project or "annolid"))
    plan = service.check(
        channel=str(channel or "stable"),
        timeout_s=float(timeout_s),
        require_signature=bool(require_signature),
    )
    return {
        "current_version": str(plan.current_version),
        "target_version": str(plan.manifest.version),
        "channel": str(plan.channel),
        "update_available": bool(plan.update_available),
        "verification_reason": str(plan.verification.reason),
    }


def execute_gui_agent_rollback(
    *,
    project: str = "annolid",
    previous_version: str,
) -> dict:
    from annolid.core.agent.update_manager.rollback import (
        build_rollback_plan,
        execute_rollback,
    )
    from annolid.core.agent.update_manager.service import UpdateManagerService

    service = UpdateManagerService(project=str(project or "annolid"))
    preflight = service.manager.preflight()
    plan = build_rollback_plan(
        install_mode=str(preflight.get("install_mode") or "package"),
        project=str(project or "annolid"),
        previous_version=str(previous_version or ""),
    )
    return execute_rollback(plan, execute=True)


def run_legacy_agent_update(
    *,
    channel: str = "stable",
    timeout_s: float = 4.0,
    apply: bool = False,
    execute: bool = False,
    skip_doctor: bool = False,
    require_signature: bool = False,
) -> tuple[dict, int]:
    from annolid.core.agent.updater import apply_update, check_for_updates

    report = check_for_updates(
        channel=str(channel or "stable").strip().lower(),
        timeout_s=float(timeout_s),
        require_signature=bool(require_signature),
    )
    payload = report.to_dict()
    if bool(apply):
        payload = apply_update(
            report,
            execute=bool(execute and apply),
            run_doctor=not bool(skip_doctor),
        )
    exit_code = 0
    if bool(apply) and bool(execute and apply):
        steps = payload.get("steps")
        if isinstance(steps, list):
            for item in steps:
                if isinstance(item, dict) and not bool(item.get("ok", True)):
                    exit_code = 1
                    break
    return payload, exit_code
