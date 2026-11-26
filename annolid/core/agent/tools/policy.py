from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Set

from annolid.core.agent.config.schema import ToolPolicyConfig, ToolsConfig

logger = logging.getLogger(__name__)


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
    "gui_list_logs",
    "gui_open_log_folder",
    "gui_remove_log_folder",
    "gui_list_log_files",
    "gui_read_log_file",
    "gui_search_logs",
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
    "gui_pdf_summarize",
    "gui_pdf_find_sections",
    "gui_web_get_dom_text",
    "gui_web_capture_screenshot",
    "gui_web_describe_view",
    "gui_web_extract_structured",
    "gui_web_click",
    "gui_web_type",
    "gui_web_scroll",
    "gui_web_find_forms",
    "gui_web_save_current",
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
    "gui_correct_tracking_ndjson",
    "gui_label_behavior_segments",
    "gui_process_video_behaviors",
    "gui_score_aggression_bouts",
    "gui_behavior_catalog",
    "gui_analyze_tracking_stats",
    "gui_arxiv_search",
    "gui_list_pdfs",
    "gui_list_shapes",
    "gui_select_shapes",
    "gui_set_selected_shape_label",
    "gui_delete_selected_shapes",
    "gui_list_shapes_in_annotation",
    "gui_relabel_shapes_in_annotation",
    "gui_delete_shapes_in_annotation",
    "gui_save_citation",
    "gui_verify_citations",
    "gui_generate_annolid_tutorial",
    "gui_self_update",
    "gui_list_logs",
    "gui_open_log_folder",
    "gui_remove_log_folder",
    "gui_list_log_files",
    "gui_read_log_file",
    "gui_search_logs",
}

_CAPABILITY_ANALYSIS: Set[str] = {
    "annolid_run",
    "annolid_tracking_stats",
    "annolid_track_qc",
    "annolid_kinematics_stats",
    "annolid_zone_analysis",
    "annolid_behavior_time_budget",
    "annolid_behavior_qc",
    "annolid_behavior_irr",
    "annolid_behavior_cohort",
    "annolid_generate_report",
    "annolid_dataset_inspect",
    "annolid_dataset_prepare",
    "annolid_eval_report",
    "annolid_eval_start",
    "annolid_novelty_check",
    "annolid_paper_run_report",
    "build_research_packet",
    "assess_paper_draft",
    "literature_search",
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
    "analysis": set(_CAPABILITY_ANALYSIS),
    "capability_analysis": set(_CAPABILITY_ANALYSIS),
    "coding": {
        *_CAPABILITY_FILESYSTEM,
        *_CAPABILITY_ANALYSIS,
        "git_status",
        "git_cli",
        "git_diff",
        "git_log",
        "github_pr_status",
        "gh_cli",
        "github_pr_checks",
        "exec",
        "exec_start",
        "exec_process",
        "video_info",
        "video_list_inference_models",
        "video_run_model_inference",
        "sam3_agent_video_track",
        "video_ffmpeg_process",
        "video_sample_frames",
        "video_segment",
        "video_process_segments",
        "video_segment_frame_grid",
        "camera_snapshot",
        "google_calendar",
        "extract_pdf_text",
        "open_pdf",
        "extract_pdf_images",
        "download_url",
        "download_pdf",
        "box",
        "clawhub_search_skills",
        "clawhub_install_skill",
        *_CAPABILITY_GUI,
        *_CAPABILITY_REALTIME,
        *_CAPABILITY_EMAIL,
    },
    "messaging": {
        *_CAPABILITY_EMAIL,
        "spawn",
        "spawn_behavior_subagent",
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
    "analysis": _CAPABILITY_ANALYSIS,
}


TOOL_GROUPS: Mapping[str, Set[str]] = {
    "group:runtime": {"exec", "exec_start", "exec_process"},
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
        "git_cli",
        "git_diff",
        "git_log",
        "github_pr_status",
        "gh_cli",
        "github_pr_checks",
    },
    "group:web": {
        "web_search",
        "web_fetch",
        "download_url",
        "download_pdf",
        "box",
        "clawhub_search_skills",
        "clawhub_install_skill",
        # MCP browser tools (require Playwright MCP server configured)
        "mcp_browser",
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
        "gui_web_capture_screenshot",
        "gui_web_describe_view",
        "gui_web_extract_structured",
        "gui_web_click",
        "gui_web_type",
        "gui_web_scroll",
        "gui_web_find_forms",
        "gui_web_save_current",
        "gui_web_run_steps",
        "gui_open_pdf",
        "gui_pdf_get_state",
        "gui_pdf_get_text",
        "gui_pdf_summarize",
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
        "gui_correct_tracking_ndjson",
        "gui_label_behavior_segments",
        "gui_process_video_behaviors",
        "gui_score_aggression_bouts",
        "gui_behavior_catalog",
        "gui_analyze_tracking_stats",
        "gui_arxiv_search",
        "gui_list_pdfs",
        "gui_list_shapes",
        "gui_select_shapes",
        "gui_set_selected_shape_label",
        "gui_delete_selected_shapes",
        "gui_list_shapes_in_annotation",
        "gui_relabel_shapes_in_annotation",
        "gui_delete_shapes_in_annotation",
        "gui_start_realtime_stream",
        "gui_stop_realtime_stream",
        "gui_get_realtime_status",
        "gui_list_realtime_models",
        "gui_list_realtime_logs",
        "gui_list_logs",
        "gui_open_log_folder",
        "gui_remove_log_folder",
        "gui_list_log_files",
        "gui_read_log_file",
        "gui_search_logs",
        "gui_check_stream_source",
        "gui_save_citation",
        "gui_verify_citations",
        "gui_generate_annolid_tutorial",
        "gui_self_update",
    },
    "group:automation": {
        "cron",
        "automation_schedule",
        "spawn",
        "spawn_behavior_subagent",
        "google_calendar",
        "box",
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
        "video_list_inference_models",
        "video_run_model_inference",
        "sam3_agent_video_track",
        "video_ffmpeg_process",
        "video_sample_frames",
        "video_segment",
        "video_process_segments",
        "video_segment_frame_grid",
        "camera_snapshot",
    },
    "group:pdf": {"extract_pdf_text", "open_pdf", "extract_pdf_images"},
    "group:skills": {"clawhub_search_skills", "clawhub_install_skill"},
    "group:analysis": set(_CAPABILITY_ANALYSIS),
    "group:admin": {
        "admin_skills_refresh",
        "admin_memory_flush",
        "admin_eval_run",
        "admin_update_run",
    },
    "group:memory": {"memory_search", "memory_get"},
    "group:training": {
        "annolid_dataset_inspect",
        "annolid_dataset_prepare",
        "annolid_train_start",
        "annolid_eval_start",
        "annolid_eval_report",
    },
    "group:citations": {
        "literature_search",
        "gui_arxiv_search",
        "gui_save_citation",
        "gui_verify_citations",
    },
    "group:coding": {
        *_CAPABILITY_FILESYSTEM,
        "git_status",
        "git_cli",
        "git_diff",
        "git_log",
        "github_pr_status",
        "gh_cli",
        "github_pr_checks",
        "exec",
        "exec_start",
        "exec_process",
    },
}

LEGACY_TOOL_NAME_ALIASES: Dict[str, str] = {
    "skills.refresh": "admin_skills_refresh",
    "memory.flush": "admin_memory_flush",
    "eval.run": "admin_eval_run",
    "update.run": "admin_update_run",
}


def canonicalize_tool_name(tool_name: str) -> str:
    lowered = str(tool_name or "").strip().lower()
    if not lowered:
        return ""
    return LEGACY_TOOL_NAME_ALIASES.get(lowered, lowered)


def classify_destructive_tool_call(
    tool_name: str,
    params: Dict[str, object] | None = None,
) -> str | None:
    """Return a human-readable risk summary if the tool call is destructive."""
    canonical = canonicalize_tool_name(tool_name)
    args = params if isinstance(params, dict) else {}
    if canonical in {"exec", "exec_start"}:
        return "Execute a shell command on this machine"
    if canonical == "exec_process":
        action = str(args.get("action") or "").strip().lower()
        if action in {"write", "submit", "kill"}:
            return "Send input to or terminate a running background process"
        return None
    if canonical in {"gui_delete_selected_shapes", "gui_delete_shapes_in_annotation"}:
        return "Delete annotation shapes"
    if canonical == "gui_remove_log_folder":
        return "Delete a log folder from disk"
    if canonical in {"rename_file", "gui_rename_file"}:
        return "Rename or overwrite a file on disk"
    if canonical in {"gui_self_update", "admin_update_run"}:
        if bool(args.get("execute", False)) or bool(args.get("rollback", False)):
            return "Modify or update the Annolid installation"
        return None
    if canonical == "git_cli":
        if bool(args.get("allow_mutation", False)):
            return "Run a mutating Git command"
        return None
    if canonical == "gh_cli":
        if bool(args.get("allow_mutation", False)):
            return "Run a mutating GitHub CLI command"
        return None
    if canonical in {"annolid_run", "annolid_dataset_prepare", "annolid_train_start"}:
        if bool(args.get("allow_mutation", False)):
            return "Run a mutating Annolid workflow command"
        return None
    return None


@dataclass(frozen=True)
class ResolvedToolPolicy:
    allowed_tools: Set[str]
    profile: str
    allow_patterns: Sequence[str]
    deny_patterns: Sequence[str]
    source: str


@dataclass(frozen=True)
class ToolPermissionContext:
    deny_names: frozenset[str]
    deny_prefixes: tuple[str, ...] = ()

    @classmethod
    def from_iterables(
        cls,
        *,
        deny_names: Iterable[str] = (),
        deny_prefixes: Iterable[str] = (),
    ) -> "ToolPermissionContext":
        deny_set = set()
        for name in deny_names:
            lowered = str(name or "").strip().lower()
            if not lowered:
                continue
            deny_set.add(lowered)
            canonical = canonicalize_tool_name(lowered)
            if canonical:
                deny_set.add(canonical)
        return cls(
            deny_names=frozenset(deny_set),
            deny_prefixes=tuple(
                str(prefix or "").strip().lower()
                for prefix in deny_prefixes
                if str(prefix or "").strip()
            ),
        )

    def blocks(self, tool_name: str) -> bool:
        lowered = str(tool_name or "").strip().lower()
        if not lowered:
            return False
        canonical = canonicalize_tool_name(lowered)
        candidates = (lowered, canonical) if canonical != lowered else (lowered,)
        for candidate in candidates:
            if candidate in self.deny_names:
                return True
            if any(candidate.startswith(prefix) for prefix in self.deny_prefixes):
                return True
        return False

    def to_dict(self) -> Dict[str, object]:
        return {
            "deny_names": sorted(self.deny_names),
            "deny_prefixes": list(self.deny_prefixes),
        }


HIGH_RISK_INTENT_MARKERS: frozenset[str] = frozenset(
    {
        "intent:high-risk",
        "intent:high_risk",
        "allow:high-risk",
        "allow_high_risk",
        "unsafe:high-risk",
    }
)
_HIGH_RISK_INTENT_MARKERS: Set[str] = set(HIGH_RISK_INTENT_MARKERS)

HIGH_RISK_TOOL_MESSAGING: frozenset[str] = frozenset(
    {
        "email",
        "list_emails",
        "read_email",
        "message",
        "camera_snapshot",
    }
)
HIGH_RISK_TOOL_RUNTIME_EXEC: frozenset[str] = frozenset(
    {"exec", "exec_start", "exec_process"}
)
HIGH_RISK_TOOL_AUTOMATION: frozenset[str] = frozenset(
    {"cron", "automation_schedule", "spawn", "spawn_behavior_subagent"}
)
HIGH_RISK_TOOL_RUNTIME_AUTOMATION: frozenset[str] = (
    HIGH_RISK_TOOL_MESSAGING | HIGH_RISK_TOOL_AUTOMATION
)

CONCURRENCY_SAFE_READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "code_search",
        "code_explain",
        "git_status",
        "git_diff",
        "git_log",
        "github_pr_status",
        "github_pr_checks",
        "memory_search",
        "memory_get",
        "web_search",
        "web_fetch",
        "literature_search",
        "video_info",
        "video_list_inference_models",
        "list_emails",
        "read_email",
        "annolid_dataset_inspect",
        "annolid_tracking_stats",
        "annolid_track_qc",
        "annolid_kinematics_stats",
        "annolid_zone_analysis",
        "annolid_behavior_time_budget",
        "annolid_behavior_qc",
        "annolid_behavior_irr",
        "annolid_behavior_cohort",
        "annolid_eval_report",
        "annolid_novelty_check",
        "calculator",
    }
)


def _has_explicit_high_risk_intent(markers: Iterable[str]) -> bool:
    for marker in markers:
        value = str(marker or "").strip().lower()
        if value in HIGH_RISK_INTENT_MARKERS:
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
    # Deny-by-default for process execution mixed with messaging/automation primitives.
    if any(t in resolved for t in HIGH_RISK_TOOL_RUNTIME_EXEC) and any(
        t in resolved for t in HIGH_RISK_TOOL_RUNTIME_AUTOMATION
    ):
        resolved.difference_update(HIGH_RISK_TOOL_RUNTIME_EXEC)
    # Deny-by-default for automated file exfiltration chains.
    if (
        "read_file" in resolved
        and ("email" in resolved or "message" in resolved)
        and any(t in resolved for t in HIGH_RISK_TOOL_AUTOMATION)
    ):
        resolved.discard("read_file")
    # Deny-by-default for subagent spawning mixed with scheduling + direct messaging.
    if (
        ("spawn" in resolved or "spawn_behavior_subagent" in resolved)
        and ("cron" in resolved or "automation_schedule" in resolved)
        and "message" in resolved
    ):
        resolved.discard("spawn")
        resolved.discard("spawn_behavior_subagent")
    return resolved


def _expand_entries(entries: Iterable[str], all_tool_names: Set[str]) -> Set[str]:
    canonical_names = {
        str(name).strip().lower(): str(name).strip()
        for name in all_tool_names
        if str(name).strip()
    }
    canonical_groups = {
        str(group).strip().lower(): {
            str(name).strip().lower() for name in names if str(name).strip()
        }
        for group, names in TOOL_GROUPS.items()
    }
    expanded: Set[str] = set()
    for raw in entries:
        item = str(raw or "").strip().lower()
        if not item:
            continue
        if item in canonical_groups:
            expanded.update(
                canonical_names[name]
                for name in canonical_groups[item]
                if name in canonical_names
            )
            continue
        if item in canonical_names:
            expanded.add(canonical_names[item])
            continue
        aliased = canonicalize_tool_name(item)
        if aliased in canonical_names:
            expanded.add(canonical_names[aliased])
            continue
        # Reverse check when all_tool_names contains legacy dotted names
        # and policy references the canonical underscored name.
        reverse_matches = [
            canonical_names[legacy]
            for legacy, target in LEGACY_TOOL_NAME_ALIASES.items()
            if target == item and legacy in canonical_names
        ]
        if reverse_matches:
            expanded.update(reverse_matches)
            continue
        if any(ch in item for ch in ("*", "?", "[")):
            expanded.update(
                canonical
                for name, canonical in canonical_names.items()
                if fnmatch.fnmatchcase(name, item)
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
    effective_allow_patterns: list[str] = list(tools_cfg.allow)
    effective_deny_patterns: list[str] = list(tools_cfg.deny)
    provider_key = str(provider or "").strip().lower()
    model_key = str(model or "").strip().lower()
    overrides: Dict[str, ToolPolicyConfig] = {
        str(key or "").strip().lower(): value
        for key, value in dict(getattr(tools_cfg, "by_provider", {}) or {}).items()
        if str(key or "").strip()
    }
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
        effective_allow_patterns.extend(list(override.allow))
        effective_deny_patterns.extend(list(override.deny))
        source = key
        break

    allowed = _apply_high_risk_guards(
        allowed=allowed,
        explicit_high_risk_intent=_has_explicit_high_risk_intent(explicit_markers),
    )

    return ResolvedToolPolicy(
        allowed_tools=allowed,
        profile=base_profile,
        allow_patterns=effective_allow_patterns,
        deny_patterns=effective_deny_patterns,
        source=source,
    )


def build_tool_permission_context(
    *,
    all_tool_names: Sequence[str],
    resolved_policy: ResolvedToolPolicy,
    extra_deny_patterns: Sequence[str] = (),
) -> ToolPermissionContext:
    all_names = {
        str(name or "").strip().lower() for name in all_tool_names if str(name).strip()
    }
    allowed = {
        str(name or "").strip().lower()
        for name in resolved_policy.allowed_tools
        if str(name).strip()
    }
    deny_names = all_names.difference(allowed)

    deny_prefixes: list[str] = []
    policy_deny_patterns = list(getattr(resolved_policy, "deny_patterns", ()) or ())
    for raw in [*policy_deny_patterns, *list(extra_deny_patterns)]:
        pattern = str(raw or "").strip().lower()
        if not pattern:
            continue
        # Keep a lightweight prefix set for simple fast checks.
        if pattern.endswith("*") and pattern.count("*") == 1 and "?" not in pattern:
            prefix = pattern[:-1].strip()
            if prefix:
                deny_prefixes.append(prefix)
    return ToolPermissionContext.from_iterables(
        deny_names=sorted(deny_names),
        deny_prefixes=deny_prefixes,
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
        # Fail closed to minimal profile on unrecognized/typo profile names
        # instead of silently escalating to full tool access.
        logger.warning(
            "Unrecognized tool policy profile %r; failing closed to 'minimal'.",
            profile,
        )
        return set(TOOL_PROFILE_BASE["minimal"] or set())
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
