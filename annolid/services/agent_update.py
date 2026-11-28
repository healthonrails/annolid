"""Service-layer orchestration for agent and desktop update workflows."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

_REASON_MESSAGES: dict[str, str] = {
    "ok": "A verified update is ready to install.",
    "not_newer_than_current": "Annolid is already up to date.",
    "already_up_to_date": "Annolid is already up to date.",
    "invalid_manifest_version": "The release server returned an unrecognized version string.",
    "missing_artifact_sha256": "The release manifest did not include a SHA-256 package digest.",
    "signature_required_missing": (
        "Signed manifests are required by your current settings or production policy, "
        "but PyPI release metadata does not include a detached signature. "
        "Disable 'Require signed manifest' for PyPI updates or configure a signed channel."
    ),
    "unsupported_signature_alg": "The release manifest uses an unsupported signature algorithm.",
    "missing_signing_key": (
        "ANNOLID_UPDATE_SIGNING_KEY is not configured to verify the manifest signature."
    ),
    "signature_verification_failed": "Manifest signature verification failed.",
    "frozen_bundle_manual_update": (
        "Desktop app bundles cannot self-update in place. Download the latest installer "
        "or archive from GitHub Releases."
    ),
    "dirty_working_tree": (
        "Your local git checkout has uncommitted changes. Commit or stash them before updating."
    ),
    "missing_required_binaries": "A required tool (such as git) was not found on PATH.",
    "apply_command_failed": "An update command failed while applying the update.",
    "post_check_failed": "Post-update verification failed and rollback was attempted.",
    "cancelled_by_user": "The update was cancelled.",
    "manual_required": "Automatic rollback is not available for this install mode.",
    "rollback_command_failed": "The rollback command failed.",
}


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


def resolve_update_runtime_options(
    *, agent_config: Any = None
) -> tuple[str, float, bool]:
    """Resolve effective (channel, timeout_s, require_signature) from config + env."""
    channel = "stable"
    timeout_s = 8.0
    require_signature = False

    cfg_obj = agent_config
    if cfg_obj is None:
        try:
            from annolid.infrastructure.agent_config import load_agent_config

            cfg_obj = load_agent_config()
        except Exception:
            cfg_obj = None

    try:
        auto_cfg = getattr(getattr(cfg_obj, "update", None), "auto", None)
        if auto_cfg is not None:
            channel = str(getattr(auto_cfg, "channel", channel) or channel).strip()
            timeout_s = float(getattr(auto_cfg, "timeout_s", timeout_s) or timeout_s)
            require_signature = bool(
                getattr(auto_cfg, "require_signature", require_signature)
            )
    except Exception:
        pass

    env_channel = str(os.environ.get("ANNOLID_AUTO_UPDATE_CHANNEL") or "").strip()
    if env_channel:
        channel = env_channel
    env_timeout = str(os.environ.get("ANNOLID_AUTO_UPDATE_TIMEOUT_S") or "").strip()
    if env_timeout:
        try:
            timeout_s = float(env_timeout)
        except Exception:
            pass
    env_sig = str(os.environ.get("ANNOLID_AUTO_UPDATE_REQUIRE_SIGNATURE") or "").strip()
    if env_sig:
        require_signature = env_sig.lower() in {"1", "true", "yes", "on"}

    channel = channel.lower()
    if channel not in {"stable", "beta", "dev"}:
        channel = "stable"
    timeout_s = max(1.0, float(timeout_s))
    return channel, timeout_s, bool(require_signature)


def format_update_reason(reason: str, payload: Mapping[str, Any] | None = None) -> str:
    """Return a clear user-facing explanation for an update status/reason."""
    if payload is not None:
        explicit = str(payload.get("message") or "").strip()
        if explicit:
            return explicit
    code = str(reason or "").strip().lower()
    if code in _REASON_MESSAGES:
        return _REASON_MESSAGES[code]
    if code:
        return code.replace("_", " ").capitalize() + "."
    return "Unknown update status."


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
    on_output: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    force_source_sync: bool = False,
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
    run_kwargs: dict[str, Any] = {
        "execute": bool(execute),
        "run_post_check": not bool(skip_post_check),
        "canary_metrics": metrics_payload,
        "canary_policy": canary_policy,
    }
    if on_output is not None:
        run_kwargs["on_output"] = on_output
    if cancel_event is not None:
        run_kwargs["cancel_event"] = cancel_event
    if force_source_sync:
        run_kwargs["force_source_sync"] = True
    payload = manager.run(plan, **run_kwargs)
    status = str(payload.get("status") or "").strip().lower()
    return payload, (0 if status in {"staged", "updated", "up_to_date"} else 1)


def rollback_agent_update(
    *,
    install_mode: str = "package",
    project: str = "annolid",
    previous_version: str = "",
    execute: bool = False,
    on_output: Callable[[str], None] | None = None,
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
    exec_kwargs: dict[str, Any] = {"execute": bool(execute)}
    if on_output is not None:
        exec_kwargs["on_output"] = on_output
    payload = execute_rollback(plan, **exec_kwargs)
    return payload, (0 if bool(payload.get("ok", False)) else 1)


def get_update_history(*, project: str = "annolid", limit: int = 10) -> dict:
    from annolid.core.agent.update_manager.history import (
        last_known_good_version,
        load_update_history,
    )

    proj = str(project or "annolid").strip() or "annolid"
    return {
        "project": proj,
        "last_known_good_version": last_known_good_version(project=proj),
        "entries": load_update_history(limit=limit),
    }


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


def inspect_gui_update_status(
    *,
    project: str = "annolid",
    channel: str = "stable",
    timeout_s: float = 8.0,
    require_signature: bool = False,
) -> dict:
    """Return comprehensive update metadata for the GUI Update Dialog."""
    from annolid.core.agent.update_manager.history import (
        last_known_good_version,
        load_update_history,
    )
    from annolid.core.agent.update_manager.manifest import release_page_url
    from annolid.core.agent.update_manager.service import UpdateManagerService

    proj = str(project or "annolid").strip() or "annolid"
    service = UpdateManagerService(project=proj)
    plan = service.check(
        channel=str(channel or "stable"),
        timeout_s=float(timeout_s),
        require_signature=bool(require_signature),
    )
    preflight = dict(getattr(plan, "preflight", {}) or {})
    reason = str(plan.verification.reason or "")
    up_to_date = bool(getattr(plan, "up_to_date", reason == "not_newer_than_current"))
    install_mode = str(getattr(plan, "install_mode", "package") or "package")
    target_ver = str(plan.manifest.version or "")
    return {
        "project": proj,
        "current_version": str(plan.current_version),
        "target_version": target_ver,
        "channel": str(plan.channel),
        "update_available": bool(plan.update_available),
        "up_to_date": up_to_date,
        "verification_ok": bool(plan.verification.ok),
        "verification_reason": reason,
        "verification_message": format_update_reason(reason),
        "install_mode": install_mode,
        "source_root": str(getattr(plan, "source_root", "") or ""),
        "release_date": str(getattr(plan.manifest, "release_date", "") or ""),
        "release_url": release_page_url(proj, target_ver),
        "commands": [list(cmd) for cmd in (getattr(plan, "commands", []) or [])],
        "preflight": preflight,
        "git_branch": str(preflight.get("git_branch") or ""),
        "git_commit": str(preflight.get("git_commit") or ""),
        "git_dirty": bool(preflight.get("git_dirty", False)),
        "git_upstream": str(preflight.get("git_upstream") or ""),
        "missing_binaries": list(preflight.get("missing_binaries") or []),
        "last_known_good_version": last_known_good_version(project=proj),
        "history": load_update_history(limit=5),
    }


def check_auto_update_if_due(
    *,
    project: str = "annolid",
    now_epoch_s: float | None = None,
) -> dict:
    """Perform a non-intrusive, read-only scheduled update check when enabled and due."""
    from annolid.core.agent.update_manager.auto_update import AutoUpdatePolicy
    from annolid.core.agent.update_manager.history import (
        load_auto_update_state,
        save_auto_update_state,
    )

    policy = AutoUpdatePolicy.from_config_and_env()
    if not policy.enabled:
        return {"status": "disabled", "update_available": False}

    now = float(now_epoch_s if now_epoch_s is not None else time.time())
    state = load_auto_update_state()
    last_raw = state.get("last_check_epoch_s")
    last_epoch = float(last_raw) if isinstance(last_raw, (int, float)) else None
    due = policy.is_due(last_check_epoch_s=last_epoch, now_epoch_s=now)
    if not due.get("due", False):
        return {
            "status": "not_due",
            "update_available": False,
            "next_due_epoch_s": due.get("next_due_epoch_s"),
        }

    info = inspect_gui_update_status(
        project=project,
        channel=policy.channel,
        timeout_s=policy.timeout_s,
        require_signature=policy.require_signature,
    )
    save_auto_update_state(
        {
            "last_check_epoch_s": round(now, 3),
            "channel": policy.channel,
            "current_version": info.get("current_version", ""),
            "target_version": info.get("target_version", ""),
            "update_available": bool(info.get("update_available", False)),
        }
    )
    return {
        "status": "update_available" if info.get("update_available") else "up_to_date",
        **info,
    }


def execute_gui_agent_rollback(
    *,
    project: str = "annolid",
    previous_version: str,
    on_output: Callable[[str], None] | None = None,
) -> dict:
    from annolid.core.agent.update_manager.rollback import (
        build_rollback_plan,
        execute_rollback,
    )
    from annolid.core.agent.update_manager.service import UpdateManagerService

    service = UpdateManagerService(project=str(project or "annolid"))
    preflight = service.manager.preflight()
    plan_kwargs: dict[str, Any] = {
        "install_mode": str(preflight.get("install_mode") or "package"),
        "project": str(project or "annolid"),
        "previous_version": str(previous_version or ""),
    }
    if preflight.get("git_commit"):
        plan_kwargs["previous_commit"] = str(preflight.get("git_commit"))
    if preflight.get("source_root"):
        plan_kwargs["source_root"] = str(preflight.get("source_root"))
    plan = build_rollback_plan(**plan_kwargs)
    exec_kwargs: dict[str, Any] = {"execute": True}
    if on_output is not None:
        exec_kwargs["on_output"] = on_output
    return execute_rollback(plan, **exec_kwargs)


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


__all__ = [
    "bot_update_requires_operator_consent",
    "check_auto_update_if_due",
    "check_for_agent_update",
    "check_gui_agent_update",
    "execute_gui_agent_rollback",
    "format_update_reason",
    "get_update_history",
    "has_operator_consent",
    "inspect_gui_update_status",
    "operator_consent_phrase",
    "resolve_update_runtime_options",
    "rollback_agent_update",
    "run_agent_update",
    "run_legacy_agent_update",
]
