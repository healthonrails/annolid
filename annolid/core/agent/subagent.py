from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional

from .loop import AgentLoop
from .tools import FunctionToolRegistry, register_nanobot_style_tools

AnnounceCallback = Callable[[str, str, str, str, str], Awaitable[None] | None]


@dataclass
class SubagentTask:
    task_id: str
    label: str
    task: str
    origin_channel: str
    origin_chat_id: str
    status: str = "running"
    result: str = ""
    error: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str = ""


class SubagentManager:
    """Background subagent execution manager for long-running tool tasks."""

    def __init__(
        self,
        *,
        loop_factory: Callable[[], AgentLoop],
        announce_callback: Optional[AnnounceCallback] = None,
        workspace: Optional[Path] = None,
        max_iterations: int = 15,
    ) -> None:
        self._loop_factory = loop_factory
        self._announce_callback = announce_callback
        self._workspace = Path(workspace) if workspace is not None else None
        self._max_iterations = max(1, int(max_iterations))
        self._running: Dict[str, asyncio.Task[None]] = {}
        self._tasks: Dict[str, SubagentTask] = {}

    async def spawn(
        self,
        task: str,
        label: Optional[str] = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        task_id = str(uuid.uuid4())[:8]
        display_label = label or (task[:30] + ("..." if len(task) > 30 else ""))
        meta = SubagentTask(
            task_id=task_id,
            label=display_label,
            task=task,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
        )
        self._tasks[task_id] = meta
        bg = asyncio.create_task(self._run_subagent(meta))
        self._running[task_id] = bg
        bg.add_done_callback(lambda _: self._running.pop(task_id, None))
        return (
            f"Subagent [{display_label}] started (id: {task_id}). "
            "I will notify you when it completes."
        )

    def get_task(self, task_id: str) -> Optional[SubagentTask]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> Dict[str, SubagentTask]:
        return dict(self._tasks)

    def get_running_count(self) -> int:
        return len(self._running)

    async def wait(self, task_id: str, timeout: Optional[float] = None) -> bool:
        task = self._running.get(task_id)
        if task is None:
            return task_id in self._tasks
        try:
            await asyncio.wait_for(task, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def cancel(self, task_id: str) -> bool:
        task = self._running.get(task_id)
        if task is None:
            return False
        task.cancel()
        return True

    async def _run_subagent(self, meta: SubagentTask) -> None:
        try:
            loop = self._loop_factory()
            prompt = self._build_subagent_prompt(meta.task)
            result = await loop.run(
                meta.task,
                session_id=f"subagent:{meta.task_id}",
                system_prompt=prompt,
                use_memory=False,
            )
            meta.status = "ok"
            meta.result = result.content
        except asyncio.CancelledError:
            meta.status = "cancelled"
            meta.error = "cancelled"
        except Exception as exc:  # pragma: no cover - defensive
            meta.status = "error"
            meta.error = str(exc)
            meta.result = f"Error: {exc}"
        finally:
            meta.finished_at = datetime.now(timezone.utc).isoformat()
            await self._announce(meta)

    async def _announce(self, meta: SubagentTask) -> None:
        if self._announce_callback is None:
            return
        ret = self._announce_callback(
            meta.task_id,
            meta.status,
            meta.label,
            meta.result or meta.error,
            f"{meta.origin_channel}:{meta.origin_chat_id}",
        )
        if asyncio.iscoroutine(ret):
            await ret

    def _build_subagent_prompt(self, task: str) -> str:
        workspace = str(self._workspace) if self._workspace is not None else "."
        return (
            "# Subagent\n\n"
            "You are a focused background subagent for a single task.\n\n"
            f"## Task\n{task}\n\n"
            "## Rules\n"
            "1. Stay focused on this task only.\n"
            "2. Use tools when needed.\n"
            "3. Return concise findings.\n"
            "4. Do not start side conversations.\n\n"
            f"## Workspace\n{workspace}\n"
        )


def build_subagent_tools_registry(
    workspace: Optional[Path] = None,
) -> FunctionToolRegistry:
    """Create a default function-tool registry suitable for subagents."""

    registry = FunctionToolRegistry()
    register_nanobot_style_tools(
        registry,
        allowed_dir=Path(workspace) if workspace is not None else None,
    )
    # Subagents should stay focused and not recursively spawn or send messages.
    registry.unregister("spawn")
    registry.unregister("message")
    registry.unregister("cron")
    return registry
