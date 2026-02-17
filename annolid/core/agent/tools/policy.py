from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Set

from annolid.core.agent.config.schema import ToolPolicyConfig, ToolsConfig


TOOL_PROFILE_BASE: Mapping[str, Set[str] | None] = {
    "full": None,
    "minimal": {"gui_context", "gui_shared_image_path"},
    "coding": {
        "read_file",
        "write_file",
        "edit_file",
        "rename_file",
        "list_dir",
        "code_search",
        "code_explain",
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
        "google_calendar",
        "extract_pdf_text",
        "open_pdf",
        "extract_pdf_images",
        "download_url",
        "download_pdf",
        "gui_context",
        "gui_shared_image_path",
        "gui_open_video",
        "gui_open_url",
        "gui_open_in_browser",
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
    },
    "messaging": {"message", "spawn", "cron"},
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
    },
    "group:automation": {"cron", "spawn", "google_calendar"},
    "group:messaging": {"message"},
    "group:video": {
        "video_info",
        "video_sample_frames",
        "video_segment",
        "video_process_segments",
    },
    "group:pdf": {"extract_pdf_text", "open_pdf", "extract_pdf_images"},
}


@dataclass(frozen=True)
class ResolvedToolPolicy:
    allowed_tools: Set[str]
    profile: str
    allow_patterns: Sequence[str]
    deny_patterns: Sequence[str]
    source: str


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
        profile_base = TOOL_PROFILE_BASE.get(profile)
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
    profile_base = TOOL_PROFILE_BASE.get(base_profile)
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
        source = key
        break

    return ResolvedToolPolicy(
        allowed_tools=allowed,
        profile=base_profile,
        allow_patterns=list(tools_cfg.allow),
        deny_patterns=list(tools_cfg.deny),
        source=source,
    )
