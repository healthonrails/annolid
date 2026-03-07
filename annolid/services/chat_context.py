"""Service helpers for GUI chat execution-context assembly."""

from __future__ import annotations

from typing import Any, Callable

from annolid.core.agent.gui_backend.context_setup import (
    load_execution_prerequisites as load_gui_execution_prerequisites,
    prepare_context_tools as prepare_gui_context_tools,
)
from annolid.core.agent.gui_backend.tool_registration import register_chat_gui_tools


def load_chat_execution_prerequisites():
    return load_gui_execution_prerequisites()


async def prepare_chat_context_tools(
    *,
    include_tools: bool,
    workspace: Any,
    allowed_read_roots: list[str],
    agent_cfg: Any,
    register_gui_tools: Callable[[Any], None],
    provider: str,
    model: str,
    enable_web_tools: bool,
    always_disabled_tools: set[str],
    web_tools: set[str],
    resolve_policy: Callable[..., Any],
):
    return await prepare_gui_context_tools(
        include_tools=include_tools,
        workspace=workspace,
        allowed_read_roots=allowed_read_roots,
        agent_cfg=agent_cfg,
        register_gui_tools=register_gui_tools,
        provider=provider,
        model=model,
        enable_web_tools=enable_web_tools,
        always_disabled_tools=always_disabled_tools,
        web_tools=web_tools,
        resolve_policy=resolve_policy,
    )


def register_chat_gui_toolset(
    tools: Any,
    *,
    context_callback: Callable[..., Any],
    image_path_callback: Callable[..., Any],
    wrap_tool_callback: Callable[[str, Callable[..., Any]], Callable[..., Any]],
    handlers: dict[str, Callable[..., Any]],
) -> None:
    register_chat_gui_tools(
        tools,
        context_callback=context_callback,
        image_path_callback=image_path_callback,
        wrap_tool_callback=wrap_tool_callback,
        handlers=handlers,
    )


__all__ = [
    "load_chat_execution_prerequisites",
    "prepare_chat_context_tools",
    "register_chat_gui_toolset",
]
