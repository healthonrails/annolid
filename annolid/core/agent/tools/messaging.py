from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .function_base import FunctionTool


class MessageTool(FunctionTool):
    def __init__(
        self,
        send_callback: Callable[[str, str, str], Awaitable[None] | None] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id

    def set_context(self, channel: str, chat_id: str) -> None:
        self._default_channel = channel
        self._default_chat_id = chat_id

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "channel": {"type": "string"},
                "chat_id": {"type": "string"},
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if self._send_callback is None:
            return "Error: Message sending not configured"
        resolved_channel = channel or self._default_channel
        resolved_chat_id = chat_id or self._default_chat_id
        if not resolved_channel or not resolved_chat_id:
            return "Error: No target channel/chat specified"
        ret = self._send_callback(resolved_channel, resolved_chat_id, content)
        if asyncio.iscoroutine(ret):
            await ret
        return f"Message sent to {resolved_channel}:{resolved_chat_id}"


class SpawnTool(FunctionTool):
    def __init__(
        self,
        spawn_callback: Callable[..., Awaitable[str] | str] | None = None,
    ):
        self._spawn_callback = spawn_callback
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    def set_spawn_callback(
        self, callback: Callable[..., Awaitable[str] | str] | None
    ) -> None:
        self._spawn_callback = callback

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return "Spawn a subagent/background task."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        del kwargs
        if self._spawn_callback is None:
            return "Error: spawn callback not configured"
        try:
            ret = self._spawn_callback(
                task=task,
                label=label,
                origin_channel=self._origin_channel,
                origin_chat_id=self._origin_chat_id,
            )
        except TypeError:
            ret = self._spawn_callback(task, label)
        if asyncio.iscoroutine(ret):
            return str(await ret)
        return str(ret)


class ListTasksTool(FunctionTool):
    def __init__(
        self,
        list_tasks_callback: Callable[[], dict[str, Any]] | None = None,
    ):
        self._list_tasks_callback = list_tasks_callback

    @property
    def name(self) -> str:
        return "list_tasks"

    @property
    def description(self) -> str:
        return "List all background subagent tasks and their current statuses."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        if self._list_tasks_callback is None:
            return "Error: list_tasks callback not configured"
        try:
            tasks = self._list_tasks_callback()
        except Exception as exc:
            return f"Error listing tasks: {exc}"
        if not tasks:
            return "No background tasks currently tracked."
        lines = []
        for tid, meta in tasks.items():
            status = getattr(meta, "status", "unknown").upper()
            label = getattr(meta, "label", "Unnamed Task")
            lines.append(f"- [{status}] ID: {tid} | Label: {label}")
        return "\n".join(lines)


class CancelTaskTool(FunctionTool):
    def __init__(
        self,
        cancel_callback: Callable[[str], bool] | None = None,
    ):
        self._cancel_callback = cancel_callback

    @property
    def name(self) -> str:
        return "cancel_task"

    @property
    def description(self) -> str:
        return "Cancel/terminate a running background subagent task by its ID."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The ID of the task to cancel (obtained from list_tasks)",
                },
            },
            "required": ["task_id"],
        }

    async def execute(self, task_id: str, **kwargs: Any) -> str:
        del kwargs
        if self._cancel_callback is None:
            return "Error: cancel callback not configured"

        task_id = str(task_id).strip()
        if not task_id:
            return "Error: task_id cannot be empty"

        try:
            success = self._cancel_callback(task_id)
        except Exception as exc:
            return f"Error cancelling task {task_id}: {exc}"

        if success:
            return f"Successfully initiated cancellation for task {task_id}."
        return f"Could not cancel task {task_id} (not found or not running)."


__all__ = ["MessageTool", "SpawnTool", "ListTasksTool", "CancelTaskTool"]
