from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .web_intents import (
    LIVE_WEB_INTENT_TOKENS,
    WEATHER_INTENT_TOKENS,
    has_fast_web_data_intent,
)

_TOKENIZE_RE = re.compile(r"[a-zA-Z0-9_]+")
_CAMEL_SPLIT_RE = re.compile(r"([a-z0-9])([A-Z])")

TOOL_SELECTOR_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "use",
        "with",
        "you",
        "your",
    }
)
DEFAULT_TOOL_SELECTION_MAX = 6

WEB_INTENT_TOKENS: frozenset[str] = frozenset(
    set(LIVE_WEB_INTENT_TOKENS)
    | {
        "search",
        "web",
        "website",
        "threejs",
        "three",
        "3d",
        "url",
    }
)
VCS_INTENT_TOKENS: frozenset[str] = frozenset(
    {
        "git",
        "github",
        "gh",
        "commit",
        "commits",
        "branch",
        "branches",
        "diff",
        "staged",
        "unstaged",
        "pull",
        "request",
        "requests",
        "pr",
        "merge",
        "rebase",
        "checkout",
    }
)
VCS_TOOL_NAMES: tuple[str, ...] = (
    "git_status",
    "git_diff",
    "git_log",
    "github_pr_status",
    "github_pr_checks",
    "git_cli",
    "gh_cli",
)
ANALYTICS_INTENT_TOKENS: frozenset[str] = frozenset(
    {
        "stats",
        "statistics",
        "statistical",
        "qc",
        "quality",
        "kinematics",
        "velocity",
        "speed",
        "freezing",
        "immobility",
        "tortuosity",
        "zone",
        "zones",
        "occupancy",
        "preference",
        "ethogram",
        "budget",
        "irr",
        "kappa",
        "krippendorff",
        "rater",
        "raters",
        "cohort",
        "anova",
        "pvalue",
        "fdr",
        "transitions",
        "markov",
        "report",
        "gaps",
    }
)
ANALYTICS_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "annolid_generate_report",
        "annolid_track_qc",
        "annolid_kinematics_stats",
        "annolid_zone_analysis",
        "annolid_behavior_time_budget",
        "annolid_behavior_qc",
        "annolid_behavior_irr",
        "annolid_behavior_cohort",
        "annolid_tracking_stats",
        "gui_analyze_tracking_stats",
        "annolid_eval_report",
    }
)
BROWSER_WORKFLOW_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "mcp_browser",
        "mcp_browser_navigate",
        "mcp_browser_snapshot",
        "mcp_browser_type",
        "mcp_browser_click",
        "mcp_browser_wait",
        "mcp_browser_screenshot",
        "mcp_browser_scroll",
    }
)
DEFAULT_TOOL_PRIORITY: tuple[str, ...] = (
    "mcp_browser",
    "mcp_browser_navigate",
    "mcp_browser_snapshot",
    "mcp_browser_type",
    "mcp_browser_click",
    "mcp_browser_wait",
    "gui_start_realtime_stream",
    "gui_stop_realtime_stream",
    "gui_label_behavior_segments",
    "gui_segment_track_video",
    "sam3_agent_video_track",
    "gui_open_video",
    "gui_open_url",
    "gui_open_in_browser",
    "gui_open_threejs",
    "gui_open_threejs_example",
    "clawhub_search_skills",
    "clawhub_install_skill",
    "gui_web_get_dom_text",
    "gui_web_capture_screenshot",
    "gui_web_describe_view",
    "gui_web_extract_structured",
    "gui_web_click",
    "gui_web_type",
    "gui_web_scroll",
    "gui_web_find_forms",
    "gui_web_run_steps",
    "gui_open_pdf",
    "gui_pdf_summarize",
    "gui_set_frame",
    "gui_set_ai_text_prompt",
    "gui_run_ai_text_segmentation",
    "gui_track_next_frames",
    "gui_context",
    "run_swarm",
    "read_file",
    "list_dir",
    "video_info",
    "video_sample_frames",
    "open_pdf",
    "extract_pdf_text",
    "download_pdf",
    "web_search",
)

# Natural next-step companions retained once a tool has been invoked in the turn.
COMPANION_TOOLS: Mapping[str, tuple[str, ...]] = {
    "list_dir": ("read_file", "code_search"),
    "code_search": ("read_file", "edit_file"),
    "read_file": ("edit_file", "write_file"),
    "git_status": ("git_diff", "git_log"),
    "git_diff": ("git_status", "read_file"),
    "mcp_browser_navigate": (
        "mcp_browser_snapshot",
        "mcp_browser_click",
        "mcp_browser_type",
    ),
    "mcp_browser_snapshot": (
        "mcp_browser_click",
        "mcp_browser_type",
        "mcp_browser_scroll",
    ),
    "web_search": ("web_fetch",),
    "memory_search": ("memory_get",),
}


@dataclass(frozen=True)
class ToolSchemaIndex:
    schema: Dict[str, Any]
    name: str
    desc: str
    tokens: frozenset[str]
    name_tokens: frozenset[str] = frozenset()


def tokenize_text(text: str) -> List[str]:
    """Tokenize free text and identifiers, splitting snake_case and camelCase cleanly."""
    source = str(text or "")
    if not source:
        return []
    tokens: List[str] = []
    seen: set[str] = set()
    for raw in _TOKENIZE_RE.findall(source):
        lowered = raw.lower()
        # Keep whole snake_case identifier (e.g. `read_file`) for exact tool mentions.
        if "_" in lowered and len(lowered) > 1 and lowered not in seen:
            seen.add(lowered)
            tokens.append(lowered)
        # Also emit split camelCase / snake_case sub-tokens.
        expanded = _CAMEL_SPLIT_RE.sub(r"\1 \2", raw).replace("_", " ").lower()
        for part in expanded.split():
            if (
                len(part) > 1
                and part not in TOOL_SELECTOR_STOPWORDS
                and part not in seen
            ):
                seen.add(part)
                tokens.append(part)
    return tokens


def _name_tokens_for_tool(name: str) -> frozenset[str]:
    lowered = str(name or "").strip().lower()
    if not lowered:
        return frozenset()
    parts = {lowered}
    expanded = _CAMEL_SPLIT_RE.sub(r"\1 \2", name).replace("_", " ").lower()
    for part in expanded.split():
        if len(part) > 1:
            parts.add(part)
    return frozenset(parts)


def build_tool_tokens(schema: Mapping[str, Any]) -> set[str]:
    fn = schema.get("function")
    if not isinstance(fn, Mapping):
        return set()
    parts: List[str] = [
        str(fn.get("name") or ""),
        str(fn.get("description") or ""),
    ]
    params = fn.get("parameters")
    if isinstance(params, Mapping):
        props = params.get("properties")
        if isinstance(props, Mapping):
            for key, value in props.items():
                parts.append(str(key))
                if isinstance(value, Mapping):
                    parts.append(str(value.get("description") or ""))
                    enum_values = value.get("enum")
                    if isinstance(enum_values, list):
                        parts.extend(str(item) for item in enum_values)
    tokens: set[str] = set()
    for part in parts:
        tokens.update(tokenize_text(part))
    return tokens


def compile_tool_index(
    all_tool_definitions: Sequence[Mapping[str, Any]],
) -> List[ToolSchemaIndex]:
    compiled: List[ToolSchemaIndex] = []
    for schema in all_tool_definitions:
        schema_dict = dict(schema)
        fn = schema_dict.get("function")
        if not isinstance(fn, Mapping):
            continue
        raw_name = str(fn.get("name") or "")
        compiled.append(
            ToolSchemaIndex(
                schema=schema_dict,
                name=raw_name.lower(),
                desc=str(fn.get("description") or "").lower(),
                tokens=frozenset(build_tool_tokens(schema_dict)),
                name_tokens=_name_tokens_for_tool(raw_name),
            )
        )
    return compiled


def tool_signature(
    all_tool_definitions: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str, str], ...]:
    triples: List[tuple[str, str, str]] = []
    for schema in all_tool_definitions:
        fn = schema.get("function")
        if not isinstance(fn, Mapping):
            continue
        params = fn.get("parameters")
        params_sig = (
            json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
            if isinstance(params, Mapping)
            else ""
        )
        triples.append(
            (
                str(fn.get("name") or ""),
                str(fn.get("description") or ""),
                params_sig,
            )
        )
    return tuple(sorted(triples))


def score_tool_schema(
    tool_entry: ToolSchemaIndex,
    query_tokens: Sequence[str],
    *,
    exact_mentions: Optional[set[str]] = None,
) -> int:
    if not query_tokens:
        return 0
    score = 0
    name_tokens = tool_entry.name_tokens or _name_tokens_for_tool(tool_entry.name)
    if exact_mentions and tool_entry.name in exact_mentions:
        score += 18
    for token in query_tokens:
        if token in tool_entry.tokens:
            score += 2
        if token in name_tokens:
            score += 3
        elif len(token) >= 4 and token in tool_entry.name:
            # Allow long stems (e.g. "search", "browser") without matching short
            # false positives like "id" in "annolid" or "pr" in "prompt".
            score += 2
        if len(token) >= 3 and token in tool_entry.desc:
            score += 1
    return score


def select_default_tool_definitions(
    tools: Sequence[Mapping[str, Any]],
    *,
    max_tools: int = DEFAULT_TOOL_SELECTION_MAX,
) -> List[Dict[str, Any]]:
    cap = max(1, int(max_tools))
    if len(tools) <= cap:
        return [dict(t) for t in tools]

    by_name: Dict[str, Dict[str, Any]] = {}
    for schema in tools:
        fn = schema.get("function")
        if not isinstance(fn, Mapping):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        by_name.setdefault(name, dict(schema))

    selected: List[Dict[str, Any]] = []
    selected_names: set[str] = set()
    for name in DEFAULT_TOOL_PRIORITY:
        schema = by_name.get(name)
        if schema is None:
            continue
        selected.append(schema)
        selected_names.add(name)
        if len(selected) >= cap:
            return selected

    remaining_names = sorted(
        name for name in by_name.keys() if name not in selected_names
    )
    for name in remaining_names:
        selected.append(by_name[name])
        if len(selected) >= cap:
            break
    return selected or [dict(t) for t in tools[:cap]]


def _extract_recent_tool_names(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    recent: set[str] = set()
    for msg in messages[-8:]:
        role = str(msg.get("role") or "").strip().lower()
        if role == "tool":
            name = str(msg.get("name") or "").strip().lower()
            if name:
                recent.add(name)
        elif role == "assistant":
            for tc in list(msg.get("tool_calls") or []):
                if not isinstance(tc, Mapping):
                    continue
                fn = tc.get("function")
                name = (
                    str(
                        (fn.get("name") if isinstance(fn, Mapping) else tc.get("name"))
                        or ""
                    )
                    .strip()
                    .lower()
                )
                if name:
                    recent.add(name)
    return recent


def resolve_max_selected_tools(explicit: Optional[int] = None) -> int:
    if explicit is not None:
        return max(3, min(32, int(explicit)))
    raw = str(os.getenv("ANNOLID_AGENT_MAX_SELECTED_TOOLS", "")).strip()
    if raw:
        try:
            return max(3, min(32, int(raw)))
        except ValueError:
            pass
    return DEFAULT_TOOL_SELECTION_MAX


def select_relevant_tool_definitions(
    *,
    all_tool_definitions: Sequence[Mapping[str, Any]],
    tool_index: Sequence[ToolSchemaIndex],
    default_tool_definitions: Sequence[Mapping[str, Any]],
    user_message_text: str,
    messages: Sequence[Mapping[str, Any]],
    registered_tool_names: Optional[set[str]] = None,
    browser_first_for_web: bool = True,
    max_selected_tools: Optional[int] = None,
) -> List[Dict[str, Any]]:
    tools = [entry.schema for entry in tool_index] or [
        dict(t) for t in all_tool_definitions
    ]
    if len(tools) <= 2:
        return [dict(t) for t in tools]
    user_tokens = tokenize_text(user_message_text)
    if not user_tokens:
        return [dict(t) for t in default_tool_definitions]

    # Use recent interaction context to improve follow-up turns.
    tail_text_parts: List[str] = []
    for msg in messages[-6:]:
        role = str(msg.get("role") or "")
        if role in {"assistant", "tool"}:
            tail_text_parts.append(str(msg.get("content") or ""))
    tail_tokens = tokenize_text(" ".join(tail_text_parts))
    query_tokens = list(dict.fromkeys([*user_tokens, *tail_tokens]))
    exact_mentions = set(user_tokens)
    recent_tools = _extract_recent_tool_names(messages)
    companion_boost_names: set[str] = set()
    for recent_name in recent_tools:
        companion_boost_names.update(COMPANION_TOOLS.get(recent_name, ()))

    scored: List[tuple[int, Dict[str, Any]]] = []
    entries = list(tool_index) or compile_tool_index(tools)
    web_intent = bool(WEB_INTENT_TOKENS.intersection(query_tokens))
    fast_web_data_intent = has_fast_web_data_intent(query_tokens)
    weather_intent = bool(WEATHER_INTENT_TOKENS.intersection(query_tokens))
    vcs_intent = bool(VCS_INTENT_TOKENS.intersection(query_tokens))
    analytics_intent = bool(ANALYTICS_INTENT_TOKENS.intersection(query_tokens))

    for entry in entries:
        score = score_tool_schema(
            entry,
            query_tokens,
            exact_mentions=exact_mentions,
        )
        if entry.name in recent_tools:
            score += 6
        elif entry.name in companion_boost_names:
            score += 4
        if analytics_intent and entry.name in ANALYTICS_TOOL_NAMES:
            score += 14
        if browser_first_for_web and web_intent:
            if fast_web_data_intent and entry.name in {"web_search", "web_fetch"}:
                score += 22
            elif entry.name in BROWSER_WORKFLOW_TOOL_NAMES:
                score += 24
            elif "browser" in entry.name or "browser" in entry.desc:
                score += 12
            if fast_web_data_intent and (
                entry.name in BROWSER_WORKFLOW_TOOL_NAMES or "browser" in entry.name
            ):
                score = max(1, score - 12)
            if entry.name == "web_search" and not fast_web_data_intent:
                score = max(1, score - 6)
        if score > 0:
            scored.append((score, entry.schema))
    if not scored:
        return [dict(t) for t in default_tool_definitions]

    scored.sort(
        key=lambda item: (
            -item[0],
            str((item[1].get("function") or {}).get("name") or ""),
        )
    )
    configured_cap = resolve_max_selected_tools(max_selected_tools)
    max_tools = min(configured_cap, max(3, len(scored)))
    selected = [schema for _, schema in scored[:max_tools]]
    selected_names = {
        str((schema.get("function") or {}).get("name") or "") for schema in selected
    }
    has_read_file = (
        "read_file" in registered_tool_names
        if registered_tool_names is not None
        else any(
            str((schema.get("function") or {}).get("name") or "") == "read_file"
            for schema in tools
        )
    )
    if has_read_file and "read_file" not in selected_names:
        read_file_schema = next(
            (
                schema
                for schema in tools
                if str((schema.get("function") or {}).get("name") or "") == "read_file"
            ),
            None,
        )
        if read_file_schema is not None:
            if weather_intent and selected:
                selected[-1] = read_file_schema
                selected_names.add("read_file")
            elif len(selected) < max_tools:
                selected.append(read_file_schema)
                selected_names.add("read_file")
    if vcs_intent:
        tool_by_name = {
            str((schema.get("function") or {}).get("name") or ""): schema
            for schema in tools
        }
        preferred_vcs: List[str] = ["git_status", "git_diff"]
        if "github" in query_tokens or "gh" in query_tokens or "pr" in query_tokens:
            preferred_vcs.extend(["github_pr_status", "github_pr_checks"])
        for name in VCS_TOOL_NAMES:
            if name in tool_by_name and name not in preferred_vcs:
                preferred_vcs.append(name)
        protected_vcs_names: set[str] = {"read_file"}
        for name in preferred_vcs:
            schema = tool_by_name.get(name)
            if schema is None:
                continue
            if name in selected_names:
                protected_vcs_names.add(name)
                continue
            if len(selected) < max_tools:
                selected.append(schema)
                selected_names.add(name)
                protected_vcs_names.add(name)
                continue
            replace_idx = len(selected) - 1
            while replace_idx >= 0:
                existing_name = str(
                    (selected[replace_idx].get("function") or {}).get("name") or ""
                )
                if existing_name not in protected_vcs_names:
                    break
                replace_idx -= 1
            if replace_idx < 0:
                break
            selected[replace_idx] = schema
            protected_vcs_names.add(name)
            selected_names = {
                str((entry.get("function") or {}).get("name") or "")
                for entry in selected
            }
    return selected or [dict(t) for t in default_tool_definitions]


__all__ = [
    "ANALYTICS_INTENT_TOKENS",
    "ANALYTICS_TOOL_NAMES",
    "BROWSER_WORKFLOW_TOOL_NAMES",
    "COMPANION_TOOLS",
    "DEFAULT_TOOL_PRIORITY",
    "DEFAULT_TOOL_SELECTION_MAX",
    "TOOL_SELECTOR_STOPWORDS",
    "ToolSchemaIndex",
    "VCS_INTENT_TOKENS",
    "VCS_TOOL_NAMES",
    "WEB_INTENT_TOKENS",
    "build_tool_tokens",
    "compile_tool_index",
    "resolve_max_selected_tools",
    "score_tool_schema",
    "select_default_tool_definitions",
    "select_relevant_tool_definitions",
    "tokenize_text",
    "tool_signature",
]
