from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from annolid.core.agent.config import load_config
from annolid.core.agent.config.secrets import (
    SecretsConfig,
    get_secret_store_path,
    inspect_secret_posture,
    load_secret_store,
    read_raw_agent_config,
)
from annolid.core.agent.security_policy import (
    require_signed_skills,
    require_signed_updates,
)
from annolid.core.agent.tools.policy import TOOL_GROUPS, resolve_allowed_tools


_RUNTIME_EXEC_TOOLS = frozenset({"exec", "exec_start", "exec_process"})
_MESSAGING_TOOLS = frozenset({"email", "list_emails", "read_email", "message"})
_AUTOMATION_TOOLS = frozenset({"cron", "automation_schedule", "spawn"})
_WEB_TOOLS = frozenset(
    {
        "web_search",
        "web_fetch",
        "mcp_browser",
        "mcp_browser_navigate",
        "mcp_browser_click",
        "mcp_browser_type",
        "mcp_browser_snapshot",
        "mcp_browser_screenshot",
        "mcp_browser_scroll",
        "mcp_browser_wait",
        "download_url",
        "download_pdf",
        "box",
        "gui_open_in_browser",
    }
)


def _coerce_group_values(values: Iterable[Iterable[str]]) -> list[set[str]]:
    return [set(value) for value in values]


_AUDIT_TOOL_NAMES = sorted(
    set().union(
        *_coerce_group_values(TOOL_GROUPS.values()),
        _RUNTIME_EXEC_TOOLS,
        _MESSAGING_TOOLS,
        _AUTOMATION_TOOLS,
        _WEB_TOOLS,
    )
)


@dataclass(frozen=True)
class SecurityFinding:
    check_id: str
    severity: str
    summary: str
    details: str
    recommendation: str
    fixable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "summary": self.summary,
            "details": self.details,
            "recommendation": self.recommendation,
            "fixable": self.fixable,
        }


def _is_private_dir_mode(path: Path) -> bool:
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return False
    return mode == 0o700


def _is_private_file_mode(path: Path) -> bool:
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return False
    return mode == 0o600


def _mode_octal(path: Path) -> str | None:
    try:
        return oct(path.stat().st_mode & 0o777)
    except OSError:
        return None


def _chmod_if_exists(path: Path, mode: int) -> bool:
    if not path.exists():
        return False
    try:
        path.chmod(mode)
        return True
    except OSError:
        return False


def _enabled_external_channels(agent_cfg: Any) -> list[tuple[str, Any]]:
    tools = getattr(agent_cfg, "tools", None)
    if tools is None:
        return []
    channels: list[tuple[str, Any]] = []
    for name in ("email", "whatsapp", "zulip"):
        cfg = getattr(tools, name, None)
        if cfg is not None and bool(getattr(cfg, "enabled", False)):
            channels.append((name, cfg))
    return channels


def run_agent_security_audit(
    *,
    config_path: Path,
    fix: bool = False,
) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser()
    agent_cfg = load_config(config_path)
    raw_payload = read_raw_agent_config(config_path)
    secrets_cfg = SecretsConfig.from_dict(raw_payload.get("secrets"))
    secret_store_path = get_secret_store_path()
    secret_store = load_secret_store(secret_store_path)
    sessions_dir = config_path.parent / "sessions"
    posture = inspect_secret_posture(
        raw_payload,
        secrets_cfg.refs,
        store=secret_store,
    )
    findings: List[SecurityFinding] = []
    fixes_applied: list[str] = []

    if posture["plaintext_paths"]:
        findings.append(
            SecurityFinding(
                check_id="plaintext-config-secrets",
                severity="high",
                summary="Plaintext agent secrets are still stored in config.json.",
                details=", ".join(posture["plaintext_paths"]),
                recommendation=(
                    "Run `annolid-run agent-secrets-migrate --apply` or attach "
                    "explicit secret refs with `agent-secrets-set`."
                ),
            )
        )
    if posture["unresolved_ref_paths"]:
        findings.append(
            SecurityFinding(
                check_id="unresolved-secret-refs",
                severity="high",
                summary="Some configured secret refs cannot be resolved.",
                details=", ".join(posture["unresolved_ref_paths"]),
                recommendation=(
                    "Set the required environment variables or populate the local "
                    "secret store before enabling the affected integrations."
                ),
            )
        )

    config_dir = config_path.parent
    if not _is_private_dir_mode(config_dir):
        if fix and _chmod_if_exists(config_dir, 0o700):
            fixes_applied.append(f"chmod 700 {config_dir}")
        findings.append(
            SecurityFinding(
                check_id="config-dir-permissions",
                severity="medium",
                summary="Agent config directory is not private.",
                details=f"{config_dir} mode={_mode_octal(config_dir)}",
                recommendation="Restrict the directory to mode 700.",
                fixable=True,
            )
        )
    if config_path.exists() and not _is_private_file_mode(config_path):
        if fix and _chmod_if_exists(config_path, 0o600):
            fixes_applied.append(f"chmod 600 {config_path}")
        findings.append(
            SecurityFinding(
                check_id="config-file-permissions",
                severity="medium",
                summary="Agent config file is not private.",
                details=f"{config_path} mode={_mode_octal(config_path)}",
                recommendation="Restrict the file to mode 600.",
                fixable=True,
            )
        )
    if secret_store_path.exists() and not _is_private_file_mode(secret_store_path):
        if fix and _chmod_if_exists(secret_store_path, 0o600):
            fixes_applied.append(f"chmod 600 {secret_store_path}")
        findings.append(
            SecurityFinding(
                check_id="secret-store-permissions",
                severity="medium",
                summary="Local secret store is not private.",
                details=f"{secret_store_path} mode={_mode_octal(secret_store_path)}",
                recommendation="Restrict the file to mode 600.",
                fixable=True,
            )
        )
    if sessions_dir.exists() and not _is_private_dir_mode(sessions_dir):
        if fix and _chmod_if_exists(sessions_dir, 0o700):
            fixes_applied.append(f"chmod 700 {sessions_dir}")
        findings.append(
            SecurityFinding(
                check_id="sessions-dir-permissions",
                severity="medium",
                summary="Agent sessions directory is not private.",
                details=f"{sessions_dir} mode={_mode_octal(sessions_dir)}",
                recommendation="Restrict the directory to mode 700.",
                fixable=True,
            )
        )

    enabled_channels = _enabled_external_channels(agent_cfg)
    dm_scope = str(
        getattr(getattr(agent_cfg.agents.defaults, "session", None), "dm_scope", "main")
        or "main"
    ).strip()
    if enabled_channels and dm_scope == "main":
        findings.append(
            SecurityFinding(
                check_id="dm-scope-main",
                severity="high",
                summary="Direct-message session scope is set to `main`.",
                details=(
                    "Enabled channels: "
                    + ", ".join(name for name, _ in enabled_channels)
                ),
                recommendation=(
                    "Use `per-peer`, `per-channel-peer`, or preferably "
                    "`per-account-channel-peer` for shared messaging channels."
                ),
            )
        )
    for name, cfg in enabled_channels:
        allow_from = list(getattr(cfg, "allow_from", []) or [])
        if not allow_from:
            findings.append(
                SecurityFinding(
                    check_id=f"channel-allowlist-{name}",
                    severity="medium",
                    summary=f"{name} channel accepts messages from any sender.",
                    details=f"{name}.allow_from is empty.",
                    recommendation=(
                        "Set `allow_from` to trusted senders or channels before "
                        "enabling the integration in production."
                    ),
                )
            )

    if not bool(agent_cfg.agents.defaults.strict_runtime_tool_guard):
        findings.append(
            SecurityFinding(
                check_id="strict-runtime-tool-guard-disabled",
                severity="high",
                summary="Strict runtime tool guard is disabled.",
                details="agents.defaults.strict_runtime_tool_guard = false",
                recommendation="Re-enable the runtime tool guard unless you have a narrow, reviewed exception.",
            )
        )

    allow_patterns = {
        str(item or "").strip() for item in list(agent_cfg.tools.allow or [])
    }
    requested_runtime = (
        str(agent_cfg.tools.profile or "").strip().lower() in {"full", "coding"}
        or "group:runtime" in allow_patterns
        or bool(_RUNTIME_EXEC_TOOLS.intersection(allow_patterns))
    )
    requested_messaging_or_automation = (
        str(agent_cfg.tools.profile or "").strip().lower() in {"coding", "messaging"}
        or "group:automation" in allow_patterns
        or "group:messaging" in allow_patterns
        or bool((_MESSAGING_TOOLS | _AUTOMATION_TOOLS).intersection(allow_patterns))
    )
    if (
        not bool(agent_cfg.agents.defaults.strict_runtime_tool_guard)
        and requested_runtime
        and requested_messaging_or_automation
    ):
        findings.append(
            SecurityFinding(
                check_id="runtime-with-messaging-or-automation",
                severity="high",
                summary="Configured policy requests runtime execution together with messaging or automation while guard rails are disabled.",
                details=(
                    f"profile={agent_cfg.tools.profile}; allow={sorted(allow_patterns)}"
                ),
                recommendation="Split runtime execution away from messaging/automation or re-enable strict_runtime_tool_guard.",
            )
        )

    resolved_policy = resolve_allowed_tools(
        all_tool_names=_AUDIT_TOOL_NAMES,
        tools_cfg=agent_cfg.tools,
        provider="audit",
        model="audit",
    )
    allowed = set(resolved_policy.allowed_tools)
    if allowed & _RUNTIME_EXEC_TOOLS and allowed & (
        _MESSAGING_TOOLS | _AUTOMATION_TOOLS
    ):
        findings.append(
            SecurityFinding(
                check_id="runtime-with-messaging-or-automation",
                severity="high",
                summary="Tool policy allows process execution alongside messaging or automation primitives.",
                details=", ".join(
                    sorted(
                        allowed
                        & (_RUNTIME_EXEC_TOOLS | _MESSAGING_TOOLS | _AUTOMATION_TOOLS)
                    )
                ),
                recommendation="Split runtime execution away from messaging/automation profiles unless the workflow is explicitly reviewed.",
            )
        )
    elif allowed & _RUNTIME_EXEC_TOOLS and allowed & _WEB_TOOLS:
        findings.append(
            SecurityFinding(
                check_id="runtime-with-web-tools",
                severity="medium",
                summary="Tool policy allows process execution alongside web/browser tools.",
                details=", ".join(sorted(allowed & (_RUNTIME_EXEC_TOOLS | _WEB_TOOLS))),
                recommendation="Prefer a narrower tool profile or deny runtime execution for browsing-focused agents.",
            )
        )

    if "clawhub_install_skill" in allowed and not require_signed_skills():
        findings.append(
            SecurityFinding(
                check_id="unsigned-skill-install",
                severity="medium",
                summary="Skill installation is allowed without signed-skill enforcement.",
                details="clawhub_install_skill is enabled and ANNOLID_REQUIRE_SIGNED_SKILLS is not enforced.",
                recommendation="Set ANNOLID_REQUIRE_SIGNED_SKILLS=1 in production or remove skill-install capabilities.",
            )
        )

    if bool(agent_cfg.update.auto.enabled):
        if (
            not bool(agent_cfg.update.auto.require_signature)
            or not require_signed_updates()
        ):
            findings.append(
                SecurityFinding(
                    check_id="unsigned-auto-update",
                    severity="medium",
                    summary="Automatic agent updates are enabled without strict signature requirements.",
                    details=(
                        f"config.require_signature={bool(agent_cfg.update.auto.require_signature)} "
                        f"env.require_signed_updates={bool(require_signed_updates())}"
                    ),
                    recommendation="Enable signature enforcement in config and environment before using automatic updates in production.",
                )
            )

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    findings_sorted = sorted(
        findings,
        key=lambda item: severity_rank.get(item.severity, 0),
        reverse=True,
    )
    status = "ok" if not findings_sorted else "warning"
    return {
        "status": status,
        "config_path": str(config_path),
        "secret_store_path": str(secret_store_path),
        "sessions_dir": str(sessions_dir),
        "dm_scope": dm_scope,
        "enabled_channels": [name for name, _cfg in enabled_channels],
        "resolved_tool_policy": {
            "profile": resolved_policy.profile,
            "source": resolved_policy.source,
            "allowed_tools": sorted(allowed),
        },
        "findings": [item.to_dict() for item in findings_sorted],
        "fixes_applied": fixes_applied,
    }
