from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


SchedulerCallback = Callable[["ScheduledTask"], Awaitable[Optional[str]]]


@dataclass
class ScheduledTask:
    id: str
    name: str
    session_id: str
    channel: str
    chat_id: str
    prompt: str
    interval_seconds: float
    enabled: bool = True
    run_immediately: bool = True
    max_runs: Optional[int] = None
    runs: int = 0
    last_run_at: float = 0.0
    next_run_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "session_id": self.session_id,
            "channel": self.channel,
            "chat_id": self.chat_id,
            "prompt": self.prompt,
            "interval_seconds": self.interval_seconds,
            "enabled": self.enabled,
            "run_immediately": self.run_immediately,
            "max_runs": self.max_runs,
            "runs": self.runs,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "metadata": dict(self.metadata),
        }


class TaskScheduler:
    """Lightweight in-memory scheduler for periodic automation tasks."""

    def __init__(
        self,
        *,
        on_run: SchedulerCallback,
        tick_seconds: float = 0.25,
    ) -> None:
        self._on_run = on_run
        self._tick_seconds = max(0.05, float(tick_seconds))
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        task = self._loop_task
        self._loop_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def add_task(
        self,
        *,
        name: str,
        session_id: str,
        channel: str,
        chat_id: str,
        prompt: str,
        interval_seconds: float,
        run_immediately: bool = True,
        max_runs: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduledTask:
        now = time.monotonic()
        interval = max(0.1, float(interval_seconds))
        task = ScheduledTask(
            id=f"task_{uuid.uuid4().hex[:12]}",
            name=str(name or "task").strip() or "task",
            session_id=str(session_id or "").strip(),
            channel=str(channel or "").strip() or "automation",
            chat_id=str(chat_id or "").strip() or "automation",
            prompt=str(prompt or "").strip(),
            interval_seconds=interval,
            run_immediately=bool(run_immediately),
            max_runs=int(max_runs) if max_runs is not None else None,
            next_run_at=now if run_immediately else (now + interval),
            metadata=dict(metadata or {}),
        )
        async with self._lock:
            self._tasks[task.id] = task
        return task

    async def remove_task(self, task_id: str) -> bool:
        key = str(task_id or "").strip()
        if not key:
            return False
        async with self._lock:
            return self._tasks.pop(key, None) is not None

    async def list_tasks(self) -> List[ScheduledTask]:
        async with self._lock:
            return [task for task in self._tasks.values()]

    async def run_task_now(self, task_id: str) -> bool:
        key = str(task_id or "").strip()
        if not key:
            return False
        async with self._lock:
            task = self._tasks.get(key)
            if task is None or not task.enabled:
                return False
            task.next_run_at = time.monotonic()
        return True

    async def _run_loop(self) -> None:
        while self._running:
            now = time.monotonic()
            due: List[ScheduledTask] = []
            async with self._lock:
                for task in self._tasks.values():
                    if not task.enabled:
                        continue
                    if task.next_run_at <= now:
                        due.append(task)
            for task in due:
                await self._execute_task(task)
            await asyncio.sleep(self._tick_seconds)

    async def _execute_task(self, task: ScheduledTask) -> None:
        now = time.monotonic()
        try:
            await self._on_run(task)
        except Exception:
            # Keep scheduler resilient; failed runs still count and reschedule.
            pass
        async with self._lock:
            existing = self._tasks.get(task.id)
            if existing is None:
                return
            existing.runs += 1
            existing.last_run_at = now
            if existing.max_runs is not None and existing.runs >= existing.max_runs:
                existing.enabled = False
                return
            existing.next_run_at = now + max(0.1, float(existing.interval_seconds))


__all__ = ["ScheduledTask", "TaskScheduler", "SchedulerCallback"]
