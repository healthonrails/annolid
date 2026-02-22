from __future__ import annotations

from typing import Any

from annolid.core.agent.scheduler import TaskScheduler

from .function_base import FunctionTool


class AutomationSchedulerTool(FunctionTool):
    """Schedule common bot automation prompts on the lightweight task scheduler."""

    def __init__(self, *, scheduler: TaskScheduler | None = None) -> None:
        self._scheduler = scheduler
        self._channel = ""
        self._chat_id = ""

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = str(channel or "").strip()
        self._chat_id = str(chat_id or "").strip()

    @property
    def name(self) -> str:
        return "automation_schedule"

    @property
    def description(self) -> str:
        return (
            "Schedule lightweight recurring bot automations. Supported task types: "
            "camera_check, periodic_report, email_summary. "
            "Actions: add, list, remove, run, status."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove", "run", "status"],
                },
                "task_id": {"type": "string"},
                "name": {"type": "string"},
                "task_type": {
                    "type": "string",
                    "enum": ["camera_check", "periodic_report", "email_summary"],
                },
                "every_seconds": {"type": "number", "minimum": 0.1},
                "camera_source": {"type": "string"},
                "email_to": {"type": "string"},
                "notes": {"type": "string"},
                "run_immediately": {"type": "boolean"},
                "max_runs": {"type": "integer", "minimum": 1},
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        task_id: str = "",
        name: str = "",
        task_type: str = "",
        every_seconds: float = 0.0,
        camera_source: str = "",
        email_to: str = "",
        notes: str = "",
        run_immediately: bool = True,
        max_runs: int | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if self._scheduler is None:
            return "Error: automation scheduler is not available."

        action_norm = str(action or "").strip().lower()
        if action_norm == "status":
            tasks = await self._scheduler.list_tasks()
            enabled = sum(1 for t in tasks if t.enabled)
            return f"Automation scheduler: tasks={len(tasks)} enabled={enabled}"
        if action_norm == "list":
            tasks = await self._scheduler.list_tasks()
            if not tasks:
                return "No automation tasks scheduled."
            lines = []
            for task in tasks:
                state = "enabled" if task.enabled else "disabled"
                lines.append(
                    f"- {task.name} (id: {task.id}, every={task.interval_seconds:.1f}s, runs={task.runs}, {state})"
                )
            return "Automation tasks:\n" + "\n".join(lines)
        if action_norm == "remove":
            key = str(task_id or "").strip()
            if not key:
                return "Error: task_id is required for remove."
            ok = await self._scheduler.remove_task(key)
            return f"Removed task {key}" if ok else f"Task {key} not found"
        if action_norm == "run":
            key = str(task_id or "").strip()
            if not key:
                return "Error: task_id is required for run."
            ok = await self._scheduler.run_task_now(key)
            return f"Triggered task {key}" if ok else f"Task {key} not found"
        if action_norm != "add":
            return f"Unknown action: {action}"

        if not self._channel or not self._chat_id:
            return "Error: no session context (channel/chat_id)."
        task_type_norm = str(task_type or "").strip().lower()
        if task_type_norm not in {"camera_check", "periodic_report", "email_summary"}:
            return "Error: task_type is required for add."
        interval = float(every_seconds or 0.0)
        if interval <= 0:
            return "Error: every_seconds must be > 0 for add."

        prompt = self._build_prompt(
            task_type=task_type_norm,
            camera_source=str(camera_source or "").strip(),
            email_to=str(email_to or "").strip(),
            notes=str(notes or "").strip(),
        )
        task_name = str(name or "").strip() or task_type_norm
        task = await self._scheduler.add_task(
            name=task_name,
            session_id=f"{self._channel}:{self._chat_id}",
            channel=self._channel,
            chat_id=self._chat_id,
            prompt=prompt,
            interval_seconds=interval,
            run_immediately=bool(run_immediately),
            max_runs=int(max_runs) if max_runs is not None else None,
            metadata={
                "task_type": task_type_norm,
                "camera_source": str(camera_source or "").strip(),
                "email_to": str(email_to or "").strip(),
                "notes": str(notes or "").strip(),
            },
        )
        return f"Created automation task '{task.name}' (id: {task.id})"

    @staticmethod
    def _build_prompt(
        *,
        task_type: str,
        camera_source: str,
        email_to: str,
        notes: str,
    ) -> str:
        email_clause = f" and email to {email_to}" if email_to else ""
        note_clause = f" Notes: {notes}" if notes else ""
        source_clause = f" Source: {camera_source}." if camera_source else ""
        if task_type == "camera_check":
            return (
                f"Check streaming camera.{source_clause} Save a snapshot{email_clause}."
                + note_clause
            )
        if task_type == "periodic_report":
            return (
                f"Analyze recent detections and write a concise periodic report{email_clause}."
                + note_clause
            )
        return (
            f"Write and send a summary email{email_clause} for recent activity."
            + note_clause
        )


__all__ = ["AutomationSchedulerTool"]
