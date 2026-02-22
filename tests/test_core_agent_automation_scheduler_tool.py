from __future__ import annotations

import asyncio

from annolid.core.agent.scheduler import TaskScheduler
from annolid.core.agent.tools.automation_scheduler import AutomationSchedulerTool


def test_automation_scheduler_tool_add_list_run_remove() -> None:
    enqueued: list[str] = []

    async def _on_run(task) -> str | None:
        enqueued.append(str(task.prompt))
        return "ok"

    async def _run() -> None:
        scheduler = TaskScheduler(on_run=_on_run, tick_seconds=0.02)
        await scheduler.start()
        try:
            tool = AutomationSchedulerTool(scheduler=scheduler)
            tool.set_context("email", "alice@example.com")

            created = await tool.execute(
                action="add",
                task_type="camera_check",
                every_seconds=0.2,
                run_immediately=False,
                camera_source="http://camera.local/mjpeg",
                email_to="alice@example.com",
            )
            assert "Created automation task" in created

            listing = await tool.execute(action="list")
            assert "camera_check" in listing

            tasks = await scheduler.list_tasks()
            assert tasks
            task_id = tasks[0].id
            run_result = await tool.execute(action="run", task_id=task_id)
            assert "Triggered task" in run_result
            await asyncio.sleep(0.06)
            assert enqueued
            assert "snapshot" in enqueued[-1].lower()

            remove_result = await tool.execute(action="remove", task_id=task_id)
            assert "Removed task" in remove_result
        finally:
            await scheduler.stop()

    asyncio.run(_run())


def test_automation_scheduler_tool_requires_scheduler_and_context() -> None:
    async def _run() -> None:
        tool = AutomationSchedulerTool(scheduler=None)
        out = await tool.execute(action="status")
        assert "not available" in out.lower()

        async def _noop(_task):
            return "ok"

        scheduler = TaskScheduler(on_run=_noop, tick_seconds=0.05)
        await scheduler.start()
        try:
            tool2 = AutomationSchedulerTool(scheduler=scheduler)
            missing_ctx = await tool2.execute(
                action="add",
                task_type="email_summary",
                every_seconds=10,
            )
            assert "no session context" in missing_ctx.lower()
        finally:
            await scheduler.stop()

    asyncio.run(_run())
