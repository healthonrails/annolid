from __future__ import annotations

from typing import Any, Dict


def fast_mode_timeout_seconds(settings: Dict[str, Any]) -> float:
    agent_cfg = settings.get("agent", {})
    raw = 60.0
    if isinstance(agent_cfg, dict):
        raw = agent_cfg.get("fast_mode_timeout_seconds", raw)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 60.0
    # Keep a sane range to avoid accidental extreme values.
    return max(10.0, min(600.0, value))


def agent_loop_llm_timeout_seconds(
    settings: Dict[str, Any], *, prompt_needs_tools: bool
) -> float:
    agent_cfg = settings.get("agent", {})
    key = (
        "loop_llm_timeout_seconds"
        if prompt_needs_tools
        else "loop_llm_timeout_seconds_no_tools"
    )
    default = 600.0 if prompt_needs_tools else 180.0
    raw = default
    if isinstance(agent_cfg, dict):
        raw = agent_cfg.get(key, agent_cfg.get("loop_llm_timeout_seconds", default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(5.0, min(1800.0, value))


def ollama_agent_tool_timeout_seconds(settings: Dict[str, Any]) -> float:
    agent_cfg = settings.get("agent", {})
    raw = 120.0
    if isinstance(agent_cfg, dict):
        raw = agent_cfg.get("ollama_tool_timeout_seconds", raw)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 45.0
    return max(5.0, min(600.0, value))


def agent_loop_tool_timeout_seconds(
    settings: Dict[str, Any], *, provider: str
) -> float:
    agent_cfg = settings.get("agent", {})
    if provider == "ollama":
        return ollama_agent_tool_timeout_seconds(settings)
    raw = 600.0
    if isinstance(agent_cfg, dict):
        raw = agent_cfg.get("loop_tool_timeout_seconds", raw)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 20.0
    return max(3.0, min(1800.0, value))


def browser_first_for_web(settings: Dict[str, Any]) -> bool:
    agent_cfg = settings.get("agent", {})
    if not isinstance(agent_cfg, dict):
        return True
    return bool(agent_cfg.get("browser_first_for_web", True))


def ollama_agent_plain_timeout_seconds(settings: Dict[str, Any]) -> float:
    agent_cfg = settings.get("agent", {})
    raw = 25.0
    if isinstance(agent_cfg, dict):
        raw = agent_cfg.get("ollama_plain_timeout_seconds", raw)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 25.0
    return max(5.0, min(600.0, value))


def ollama_plain_recovery_timeout_seconds(settings: Dict[str, Any]) -> float:
    agent_cfg = settings.get("agent", {})
    raw = 12.0
    if isinstance(agent_cfg, dict):
        raw = agent_cfg.get("ollama_plain_recovery_timeout_seconds", raw)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 12.0
    return max(3.0, min(90.0, value))


def ollama_plain_recovery_nudge_timeout_seconds(settings: Dict[str, Any]) -> float:
    agent_cfg = settings.get("agent", {})
    raw = 8.0
    if isinstance(agent_cfg, dict):
        raw = agent_cfg.get("ollama_plain_recovery_nudge_timeout_seconds", raw)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 8.0
    return max(2.0, min(90.0, value))


def fallback_retry_timeout_seconds(settings: Dict[str, Any]) -> float:
    agent_cfg = settings.get("agent", {})
    raw = 60.0
    if isinstance(agent_cfg, dict):
        raw = agent_cfg.get("fallback_retry_timeout_seconds", raw)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 20.0
    # Keep retry short so total wall time doesn't balloon after a timeout.
    return max(5.0, min(300.0, value))


def fallback_timeout_retry_seconds(
    settings: Dict[str, Any], *, prompt_needs_tools: bool
) -> float:
    """Retry timeout after loop timeout; never shorter than current loop limit."""
    base = agent_loop_llm_timeout_seconds(
        settings,
        prompt_needs_tools=prompt_needs_tools,
    )
    retry = fallback_retry_timeout_seconds(settings)
    return max(retry, base)
