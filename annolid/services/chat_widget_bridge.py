"""Service helpers for GUI chat widget bridge and direct-command execution."""

from __future__ import annotations

from typing import Any

from annolid.core.agent.gui_backend.direct_commands import (
    execute_direct_gui_command as gui_execute_direct_gui_command,
    run_awaitable_sync as gui_run_awaitable_sync,
)
from annolid.core.agent.gui_backend.router import (
    execute_direct_gui_command as gui_route_direct_gui_command,
)
from annolid.core.agent.gui_backend.widget_bridge import (
    build_gui_context_payload as gui_build_gui_context_payload,
    get_widget_action_result as gui_get_widget_action_result,
    invoke_widget_json_slot as gui_invoke_widget_json_slot,
    invoke_widget_slot as gui_invoke_widget_slot,
)


def build_chat_gui_context_payload(**kwargs: Any) -> dict[str, Any]:
    return gui_build_gui_context_payload(**kwargs)


def invoke_chat_widget_slot(**kwargs: Any) -> bool:
    return bool(gui_invoke_widget_slot(**kwargs))


def invoke_chat_widget_json_slot(**kwargs: Any) -> dict[str, Any]:
    return gui_invoke_widget_json_slot(**kwargs)


def get_chat_widget_action_result(*, widget: Any, action_name: str) -> dict[str, Any]:
    return gui_get_widget_action_result(widget=widget, action_name=action_name)


def run_chat_awaitable_sync(awaitable: Any) -> Any:
    return gui_run_awaitable_sync(awaitable)


async def execute_chat_direct_gui_command(
    *,
    prompt: str,
    parse_direct_gui_command: Any,
    handlers: dict[str, Any],
) -> dict[str, Any]:
    return await gui_execute_direct_gui_command(
        prompt=prompt,
        parse_direct_gui_command=parse_direct_gui_command,
        route_direct_gui_command=gui_route_direct_gui_command,
        handlers=handlers,
    )


__all__ = [
    "build_chat_gui_context_payload",
    "execute_chat_direct_gui_command",
    "get_chat_widget_action_result",
    "invoke_chat_widget_json_slot",
    "invoke_chat_widget_slot",
    "run_chat_awaitable_sync",
]
