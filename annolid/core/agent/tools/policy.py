from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Set

from annolid.core.agent.config.schema import ToolPolicyConfig, ToolsConfig


_CAPABILITY_FILESYSTEM: Set[str] = {
    "read_file",
    "write_file",
    "edit_file",
    "rename_file",
    "list_dir",
    "code_search",
    "code_explain",
}
_CAPABILITY_EMAIL: Set[str] = {
    "email",
    "list_emails",
    "read_email",
    "message",
}
_CAPABILITY_REALTIME: Set[str] = {
    "camera_snapshot",
    "gui_start_realtime_stream",
    "gui_stop_realtime_stream",
    "gui_get_realtime_status",
    "gui_check_stream_source",
    "gui_list_realtime_models",
    "gui_list_realtime_logs",
}
_CAPABILITY_GUI: Set[str] = {
    "gui_context",
    "gui_shared_image_path",
    "gui_open_video",
    "gui_open_url",
    "gui_open_in_browser",
    "gui_open_threejs",
    "gui_open_threejs_example",
    "gui_open_pdf",
    "gui_pdf_get_state",
    "gui_pdf_get_text",
    "gui_pdf_find_sections",
    "gui_web_get_dom_text",
    "gui_web_click",
    "gui_web_type",
    "gui_web_scroll",
    "gui_web_find_forms",
    "gui_web_run_steps",
    "gui_set_frame",
    "gui_set_chat_prompt",
    "gui_send_chat_prompt",
    "gui_set_chat_model",
    "gui_select_annotation_model",
    "gui_track_next_frames",
    "gui_set_ai_text_prompt",
    "gui_run_ai_text_segmentation",
    "gui_segment_track_video",
    "gui_label_behavior_segments",
    "gui_save_citation",
    "gui_generate_annolid_tutorial",
}


TOOL_PROFILE_BASE: Mapping[str, Set[str] | None] = {
    "full": None,
    "minimal": {"gui_context", "gui_shared_image_path"},
    # Explicit capability profiles.
    "filesystem": set(_CAPABILITY_FILESYSTEM),
    "email": set(_CAPABILITY_EMAIL),
    "realtime": set(_CAPABILITY_REALTIME),
    "gui": set(_CAPABILITY_GUI),
    "capability_filesystem": set(_CAPABILITY_FILESYSTEM),
    "capability_email": set(_CAPABILITY_EMAIL),
    "capability_realtime": set(_CAPABILITY_REALTIME),
    "capability_gui": set(_CAPABILITY_GUI),
    "coding": {
        *_CAPABILITY_FILESYSTEM,
        "git_status",
        "git_diff",
        "git_log",
        "github_pr_status",
        "github_pr_checks",
        "exec",
        "video_info",
        "video_sample_frames",
        "video_segment",
        "video_process_segments",
        "camera_snapshot",
        "google_calendar",
        "extract_pdf_text",
        "open_pdf",
        "extract_pdf_images",
        "download_url",
        "download_pdf",
        "clawhub_search_skills",
        "clawhub_install_skill",
        *_CAPABILITY_GUI,
        *_CAPABILITY_REALTIME,
        *_CAPABILITY_EMAIL,
    },
    "messaging": {
        *_CAPABILITY_EMAIL,
        "spawn",
        "cron",
        "automation_schedule",
        "camera_snapshot",
    },
}

_CAPABILITY_PROFILE_MAP: Mapping[str, Set[str]] = {
    "filesystem": _CAPABILITY_FILESYSTEM,
    "email": _CAPABILITY_EMAIL,
    "realtime": _CAPABILITY_REALTIME,
    "gui": _CAPABILITY_GUI,
}


TOOL_GROUPS: Mapping[str, Set[str]] = {
    "group:runtime": {"exec"},
    "group:fs": {
        "read_file",
        "write_file",
        "edit_file",
        "rename_file",
        "list_dir",
        "code_search",
        "code_explain",
    },
    "group:vcs": {
        "git_status",
        "git_diff",
        "git_log",
        "github_pr_status",
        "github_pr_checks",
    },
    "group:web": {
        "web_search",
        "web_fetch",
        "download_url",
        "download_pdf",
        "clawhub_search_skills",
        "clawhub_install_skill",
        # MCP browser tools (require Playwright MCP server configured)
        "mcp_browser_navigate",
        "mcp_browser_click",
        "mcp_browser_type",
        "mcp_browser_snapshot",
        "mcp_browser_screenshot",
        "mcp_browser_scroll",
        "mcp_browser_close",
        "mcp_browser_wait",
    },
    "group:ui": {
        "gui_context",
        "gui_shared_image_path",
        "gui_open_video",
        "gui_open_url",
        "gui_open_in_browser",
        "gui_open_threejs",
        "gui_open_threejs_example",
        "gui_web_get_dom_text",
        "gui_web_click",
        "gui_web_type",
        "gui_web_scroll",
        "gui_web_find_forms",
        "gui_web_run_steps",
        "gui_open_pdf",
        "gui_pdf_get_state",
        "gui_pdf_get_text",
        "gui_pdf_find_sections",
        "gui_set_frame",
        "gui_set_chat_prompt",
        "gui_send_chat_prompt",
        "gui_set_chat_model",
        "gui_select_annotation_model",
        "gui_track_next_frames",
        "gui_set_ai_text_prompt",
        "gui_run_ai_text_segmentation",
        "gui_segment_track_video",
        "gui_label_behavior_segments",
        "gui_start_realtime_stream",
        "gui_stop_realtime_stream",
        "gui_get_realtime_status",
        "gui_list_realtime_models",
        "gui_list_realtime_logs",
        "gui_check_stream_source",
        "gui_save_citation",
        "gui_generate_annolid_tutorial",
    },
    "group:automation": {
        "cron",
        "automation_schedule",
        "spawn",
        "google_calendar",
        "email",
        "list_emails",
        "read_email",
    },
    "group:messaging": {
        "message",
        "email",
        "list_emails",
        "read_email",
        "camera_snapshot",
    },
    "group:video": {
        "video_info",
        "video_sample_frames",
        "video_segment",
        "video_process_segments",
        "camera_snapshot",
    },
    "group:pdf": {"extract_pdf_text", "open_pdf", "extract_pdf_images"},
    "group:skills": {"clawhub_search_skills", "clawhub_install_skill"},
}


@dataclass(frozen=True)
class ResolvedToolPolicy:
    allowed_tools: Set[str]
    profile: str
    allow_patterns: Sequence[str]
    deny_patterns: Sequence[str]
    source: str


_HIGH_RISK_INTENT_MARKERS = {
    "intent:high-risk",
    "intent:high_risk",
    "allow:high-risk",
    "allow_high_risk",
    "unsafe:high-risk",
}


def _has_explicit_high_risk_intent(markers: Iterable[str]) -> bool:
    for marker in markers:
        value = str(marker or "").strip().lower()
        if value in _HIGH_RISK_INTENT_MARKERS:
            return True
    return False


def _apply_high_risk_guards(
    *,
    allowed: Set[str],
    explicit_high_risk_intent: bool,
) -> Set[str]:
    if explicit_high_risk_intent:
        return set(allowed)
    resolved = set(allowed)
    automation_or_messaging = {
        "email",
        "list_emails",
        "read_email",
        "message",
        "camera_snapshot",
        "cron",
        "automation_schedule",
        "spawn",
    }
    # Deny-by-default for process execution mixed with messaging/automation primitives.
    if "exec" in resolved and any(t in resolved for t in automation_or_messaging):
        resolved.discard("exec")
    # Deny-by-default for automated file exfiltration chains.
    if (
        "read_file" in resolved
        and ("email" in resolved or "message" in resolved)
        and any(t in resolved for t in {"cron", "automation_schedule", "spawn"})
    ):
        resolved.discard("read_file")
    # Deny-by-default for subagent spawning mixed with scheduling + direct messaging.
    if (
        "spawn" in resolved
        and ("cron" in resolved or "automation_schedule" in resolved)
        and "message" in resolved
    ):
        resolved.discard("spawn")
    return resolved


def _expand_entries(entries: Iterable[str], all_tool_names: Set[str]) -> Set[str]:
    expanded: Set[str] = set()
    for raw in entries:
        item = str(raw or "").strip()
        if not item:
            continue
        if item in TOOL_GROUPS:
            expanded.update(
                name for name in TOOL_GROUPS[item] if name in all_tool_names
            )
            continue
        if item in all_tool_names:
            expanded.add(item)
            continue
        if any(ch in item for ch in ("*", "?", "[")):
            expanded.update(
                name for name in all_tool_names if fnmatch.fnmatchcase(name, item)
            )
    return expanded


def _apply_policy(
    *,
    base_allowed: Set[str],
    policy: ToolPolicyConfig,
    all_tool_names: Set[str],
) -> Set[str]:
    resolved = set(base_allowed)
    profile = str(policy.profile or "").strip().lower()
    if profile:
        profile_base = _resolve_profile_base(profile)
        if profile_base is None:
            resolved = set(all_tool_names)
        else:
            resolved = set(profile_base).intersection(all_tool_names)
    allow_set = _expand_entries(policy.allow, all_tool_names)
    if allow_set:
        resolved.update(allow_set)
    deny_set = _expand_entries(policy.deny, all_tool_names)
    if deny_set:
        resolved.difference_update(deny_set)
    return resolved


def resolve_allowed_tools(
    *,
    all_tool_names: Sequence[str],
    tools_cfg: ToolsConfig,
    provider: str,
    model: str,
) -> ResolvedToolPolicy:
    all_names = {str(name).strip() for name in all_tool_names if str(name).strip()}
    base_profile = str(tools_cfg.profile or "full").strip().lower() or "full"
    profile_base = _resolve_profile_base(base_profile)
    if profile_base is None:
        allowed = set(all_names)
    else:
        allowed = set(profile_base).intersection(all_names)

    base_policy = ToolPolicyConfig(
        profile="",
        allow=list(tools_cfg.allow),
        deny=list(tools_cfg.deny),
    )
    allowed = _apply_policy(
        base_allowed=allowed,
        policy=base_policy,
        all_tool_names=all_names,
    )

    source = "global"
    explicit_markers: list[str] = list(tools_cfg.allow)
    provider_key = str(provider or "").strip().lower()
    model_key = str(model or "").strip().lower()
    overrides: Dict[str, ToolPolicyConfig] = dict(
        getattr(tools_cfg, "by_provider", {}) or {}
    )
    candidates = [
        f"{provider_key}:{model_key}",
        provider_key,
    ]
    for key in candidates:
        override = overrides.get(key)
        if override is None:
            continue
        allowed = _apply_policy(
            base_allowed=allowed,
            policy=override,
            all_tool_names=all_names,
        )
        explicit_markers.extend(list(override.allow))
        source = key
        break

    allowed = _apply_high_risk_guards(
        allowed=allowed,
        explicit_high_risk_intent=_has_explicit_high_risk_intent(explicit_markers),
    )

    return ResolvedToolPolicy(
        allowed_tools=allowed,
        profile=base_profile,
        allow_patterns=list(tools_cfg.allow),
        deny_patterns=list(tools_cfg.deny),
        source=source,
    )


def _resolve_profile_base(profile: str) -> Set[str] | None:
    """Resolve profile names and explicit capability profile expressions.

    Supported explicit capability expressions:
    - capability:gui
    - capability:gui,email
    - capability:gui+realtime
    """
    key = str(profile or "").strip().lower()
    direct = TOOL_PROFILE_BASE.get(key)
    if direct is not None or key in TOOL_PROFILE_BASE:
        return direct
    if not key.startswith("capability:"):
        return None
    raw_caps = key.split(":", 1)[1]
    caps: Set[str] = set()
    for token in raw_caps.replace("+", ",").split(","):
        cap = str(token or "").strip().lower()
        if not cap:
            continue
        base = _CAPABILITY_PROFILE_MAP.get(cap)
        if base:
            caps.update(base)
    return caps
