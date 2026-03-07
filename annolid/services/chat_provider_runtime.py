"""Service helpers for GUI chat provider/runtime policy and execution."""

from __future__ import annotations

from typing import Any

from annolid.core.agent.gui_backend.provider_dependencies import (
    format_dependency_error as gui_format_dependency_error,
    provider_dependency_error as gui_provider_dependency_error,
)
from annolid.core.agent.gui_backend.provider_fallback import (
    format_provider_config_error as gui_format_provider_config_error,
    is_provider_config_error as gui_is_provider_config_error,
    is_provider_timeout_error as gui_is_provider_timeout_error,
    run_provider_fallback as gui_run_provider_fallback,
)
from annolid.core.agent.gui_backend.provider_runtime import (
    has_image_context as gui_has_image_context,
    run_fast_mode as gui_run_fast_mode,
    run_fast_provider_chat as gui_run_fast_provider_chat,
    run_gemini_provider_chat as gui_run_gemini_provider_chat,
    run_ollama_chat as gui_run_ollama_chat,
    run_openai_chat as gui_run_openai_chat,
)
from annolid.core.agent.gui_backend.runtime_config import (
    agent_loop_llm_timeout_seconds as gui_agent_loop_llm_timeout_seconds,
    agent_loop_tool_timeout_seconds as gui_agent_loop_tool_timeout_seconds,
    browser_first_for_web as gui_browser_first_for_web,
    fallback_retry_timeout_seconds as gui_fallback_retry_timeout_seconds,
    fallback_timeout_retry_seconds as gui_fallback_timeout_retry_seconds,
    fast_mode_timeout_seconds as gui_fast_mode_timeout_seconds,
    ollama_agent_plain_timeout_seconds as gui_ollama_agent_plain_timeout_seconds,
    ollama_agent_tool_timeout_seconds as gui_ollama_agent_tool_timeout_seconds,
    ollama_plain_recovery_nudge_timeout_seconds as gui_ollama_plain_recovery_nudge_timeout_seconds,
    ollama_plain_recovery_timeout_seconds as gui_ollama_plain_recovery_timeout_seconds,
)
from annolid.core.agent.gui_backend.telemetry import (
    log_runtime_timeouts as gui_log_runtime_timeouts,
)
from annolid.core.agent.loop import AgentLoop
from annolid.utils.llm_settings import provider_kind as gui_provider_kind


def is_chat_provider_config_error(exc: Exception | str) -> bool:
    return gui_is_provider_config_error(exc)


def format_chat_provider_config_error(raw_error: str, *, provider: str) -> str:
    return gui_format_provider_config_error(raw_error, provider=provider)


def is_chat_provider_timeout_error(exc: Exception | str) -> bool:
    return gui_is_provider_timeout_error(exc)


def chat_provider_dependency_error(
    *, settings: dict[str, Any], provider: str
) -> str | None:
    return gui_provider_dependency_error(settings=settings, provider=provider)


def format_chat_dependency_error(
    *, raw_error: str, settings: dict[str, Any], provider: str
) -> str:
    return gui_format_dependency_error(
        raw_error=raw_error,
        settings=settings,
        provider=provider,
    )


def resolve_chat_provider_kind(*, settings: dict[str, Any], provider: str) -> str:
    return str(gui_provider_kind(settings, provider) or "")


def has_chat_image_context(image_path: str | None) -> bool:
    return gui_has_image_context(image_path)


def run_chat_fast_mode(
    *,
    chat_mode: str,
    run_fast_provider_chat: Any,
) -> None:
    gui_run_fast_mode(
        chat_mode=chat_mode,
        run_fast_provider_chat=run_fast_provider_chat,
    )


def run_chat_fast_provider_chat(**kwargs: Any) -> None:
    gui_run_fast_provider_chat(**kwargs)


def chat_fast_mode_timeout_seconds(settings: dict[str, Any]) -> float:
    return gui_fast_mode_timeout_seconds(settings)


def chat_agent_loop_llm_timeout_seconds(
    *, settings: dict[str, Any], provider: str, prompt_needs_tools: bool
) -> float:
    base = gui_agent_loop_llm_timeout_seconds(
        settings,
        prompt_needs_tools=prompt_needs_tools,
    )
    if str(provider or "").strip().lower() == "nvidia":
        return max(base, 420.0 if prompt_needs_tools else 180.0)
    return base


def chat_ollama_agent_tool_timeout_seconds(settings: dict[str, Any]) -> float:
    return gui_ollama_agent_tool_timeout_seconds(settings)


def chat_agent_loop_tool_timeout_seconds(
    *, settings: dict[str, Any], provider: str, prompt: str
) -> float:
    base = gui_agent_loop_tool_timeout_seconds(settings, provider=provider)
    if prompt and ("swarm" in prompt.lower() or "agents" in prompt.lower()):
        return max(base, 600.0)
    return base


def chat_browser_first_for_web(settings: dict[str, Any]) -> bool:
    return gui_browser_first_for_web(settings)


def chat_ollama_agent_plain_timeout_seconds(settings: dict[str, Any]) -> float:
    return gui_ollama_agent_plain_timeout_seconds(settings)


def chat_ollama_plain_recovery_timeout_seconds(settings: dict[str, Any]) -> float:
    return gui_ollama_plain_recovery_timeout_seconds(settings)


def chat_ollama_plain_recovery_nudge_timeout_seconds(settings: dict[str, Any]) -> float:
    return gui_ollama_plain_recovery_nudge_timeout_seconds(settings)


def chat_fallback_retry_timeout_seconds(
    *, settings: dict[str, Any], provider: str
) -> float:
    base = gui_fallback_retry_timeout_seconds(settings)
    if str(provider or "").strip().lower() == "nvidia":
        return max(base, 60.0)
    return base


def chat_fallback_timeout_retry_seconds(
    *, settings: dict[str, Any], prompt_needs_tools: bool
) -> float:
    return gui_fallback_timeout_retry_seconds(
        settings,
        prompt_needs_tools=prompt_needs_tools,
    )


def log_chat_runtime_timeouts(
    *,
    logger: Any,
    session_id: str,
    model: str,
    loop_llm_s: float,
    loop_tool_s: float,
    ollama_tool_s: float,
    ollama_plain_s: float,
    recover_s: float,
    recover_nudge_s: float,
) -> None:
    gui_log_runtime_timeouts(
        logger=logger,
        session_id=session_id,
        model=model,
        loop_llm_s=loop_llm_s,
        loop_tool_s=loop_tool_s,
        ollama_tool_s=ollama_tool_s,
        ollama_plain_s=ollama_plain_s,
        recover_s=recover_s,
        recover_nudge_s=recover_nudge_s,
    )


def run_chat_provider_fallback(**kwargs: Any) -> None:
    gui_run_provider_fallback(**kwargs)


def run_chat_ollama(**kwargs: Any) -> None:
    gui_run_ollama_chat(**kwargs)


def run_chat_openai(**kwargs: Any) -> tuple[str, str]:
    return gui_run_openai_chat(**kwargs)


def run_chat_gemini(**kwargs: Any) -> tuple[str, str]:
    return gui_run_gemini_provider_chat(**kwargs)


def build_chat_agent_loop(
    *,
    tools: Any,
    llm_callable: Any,
    provider: str,
    model: str,
    memory_store: Any,
    workspace: str,
    allowed_read_roots: list[str],
    mcp_servers: dict[str, Any],
    llm_timeout_seconds: float,
    tool_timeout_seconds: float,
    browser_first_for_web: bool,
    strict_runtime_tool_guard: bool,
) -> AgentLoop:
    return AgentLoop(
        tools=tools,
        llm_callable=llm_callable,
        provider=provider,
        model=model,
        profile="playground",
        memory_store=memory_store,
        workspace=workspace,
        allowed_read_roots=allowed_read_roots,
        mcp_servers=mcp_servers,
        llm_timeout_seconds=llm_timeout_seconds,
        tool_timeout_seconds=tool_timeout_seconds,
        browser_first_for_web=browser_first_for_web,
        strict_runtime_tool_guard=strict_runtime_tool_guard,
    )


__all__ = [
    "build_chat_agent_loop",
    "chat_agent_loop_llm_timeout_seconds",
    "chat_agent_loop_tool_timeout_seconds",
    "chat_browser_first_for_web",
    "chat_fallback_retry_timeout_seconds",
    "chat_fallback_timeout_retry_seconds",
    "chat_fast_mode_timeout_seconds",
    "chat_ollama_agent_plain_timeout_seconds",
    "chat_ollama_agent_tool_timeout_seconds",
    "chat_ollama_plain_recovery_nudge_timeout_seconds",
    "chat_ollama_plain_recovery_timeout_seconds",
    "chat_provider_dependency_error",
    "format_chat_dependency_error",
    "format_chat_provider_config_error",
    "has_chat_image_context",
    "is_chat_provider_config_error",
    "is_chat_provider_timeout_error",
    "log_chat_runtime_timeouts",
    "resolve_chat_provider_kind",
    "run_chat_fast_mode",
    "run_chat_fast_provider_chat",
    "run_chat_gemini",
    "run_chat_ollama",
    "run_chat_openai",
    "run_chat_provider_fallback",
]
