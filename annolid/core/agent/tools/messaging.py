from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from annolid.services.behavior_agent import (
    list_behavior_subagent_profiles,
    resolve_behavior_subagent_profile,
)

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
        return "Spawn a background task using either the native subagent runtime or the ACP runtime."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "label": {"type": "string"},
                "runtime": {
                    "type": "string",
                    "enum": ["subagent", "acp"],
                },
                "provider": {"type": "string"},
                "model": {"type": "string"},
                "workspace": {"type": "string"},
                "profile": {
                    "type": "string",
                    "description": (
                        "Optional subagent profile name (for example "
                        "behavior_assay_inference)."
                    ),
                },
                "skill_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit skill names to prioritize.",
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        label: str | None = None,
        runtime: str = "subagent",
        provider: str = "",
        model: str = "",
        workspace: str = "",
        profile: str = "",
        skill_names: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if self._spawn_callback is None:
            return "Error: spawn callback not configured"
        try:
            ret = self._spawn_callback(
                task=task,
                label=label,
                runtime=runtime,
                provider=provider,
                model=model,
                workspace=workspace,
                profile=profile,
                skill_names=skill_names,
                origin_channel=self._origin_channel,
                origin_chat_id=self._origin_chat_id,
            )
        except TypeError:
            ret = self._spawn_callback(task, label)
        if asyncio.iscoroutine(ret):
            return str(await ret)
        return str(ret)


class SpawnBehaviorSubagentTool(FunctionTool):
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
        return "spawn_behavior_subagent"

    @property
    def description(self) -> str:
        return (
            "Spawn a specialized behavior-analysis subagent profile with optional "
            "explicit skills."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        profiles = [row.name for row in list_behavior_subagent_profiles()]
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "profile": {
                    "type": "string",
                    "enum": profiles,
                },
                "label": {"type": "string"},
                "skill_names": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["profile"],
        }

    async def execute(
        self,
        profile: str,
        task: str = "",
        label: str | None = None,
        skill_names: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        resolved_profile = resolve_behavior_subagent_profile(profile)
        if resolved_profile is None:
            names = ", ".join(row.name for row in list_behavior_subagent_profiles())
            return (
                f"Error: unknown behavior subagent profile '{profile}'. "
                f"Available profiles: {names}"
            )
        if self._spawn_callback is None:
            return "Error: spawn callback not configured"

        task_text = str(task or "").strip()
        if not task_text:
            task_text = (
                f"{resolved_profile.description} "
                f"Follow profile guidance and return concise results."
            )
        profile_label = str(label or "").strip() or resolved_profile.name
        try:
            ret = self._spawn_callback(
                task=task_text,
                label=profile_label,
                runtime="subagent",
                profile=resolved_profile.name,
                skill_names=skill_names,
                origin_channel=self._origin_channel,
                origin_chat_id=self._origin_chat_id,
            )
        except TypeError:
            ret = self._spawn_callback(task_text, profile_label)
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
        if asyncio.iscoroutine(success):
            success = await success

        if success:
            return f"Successfully initiated cancellation for task {task_id}."
        return f"Could not cancel task {task_id} (not found or not running)."


__all__ = [
    "MessageTool",
    "SpawnTool",
    "SpawnBehaviorSubagentTool",
    "ListTasksTool",
    "CancelTaskTool",
]
