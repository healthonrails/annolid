from __future__ import annotations

import asyncio

from annolid.core.agent.scheduler import TaskScheduler


def test_task_scheduler_runs_periodic_tasks_and_stops_at_max_runs() -> None:
    runs: list[str] = []

    async def _on_run(task):
        runs.append(task.name)
        return "ok"

    async def _run() -> None:
        scheduler = TaskScheduler(on_run=_on_run, tick_seconds=0.02)
        await scheduler.start()
        try:
            task = await scheduler.add_task(
                name="camera-check",
                session_id="s1",
                channel="automation",
                chat_id="bot",
                prompt="Check camera",
                interval_seconds=0.03,
                run_immediately=True,
                max_runs=2,
            )
            await asyncio.sleep(0.15)
            tasks = await scheduler.list_tasks()
            assert tasks
            current = [t for t in tasks if t.id == task.id][0]
            assert current.enabled is False
            assert int(current.runs) == 2
            assert runs.count("camera-check") == 2
        finally:
            await scheduler.stop()

    asyncio.run(_run())


def test_task_scheduler_run_now_and_error_resilience() -> None:
    calls = {"n": 0}

    async def _on_run(_task):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return "ok"

    async def _run() -> None:
        scheduler = TaskScheduler(on_run=_on_run, tick_seconds=0.02)
        await scheduler.start()
        try:
            task = await scheduler.add_task(
                name="report",
                session_id="s1",
                channel="automation",
                chat_id="bot",
                prompt="Generate report",
                interval_seconds=10.0,
                run_immediately=False,
                max_runs=2,
            )
            assert await scheduler.run_task_now(task.id) is True
            await asyncio.sleep(0.05)
            # First run fails but scheduler remains alive.
            assert await scheduler.run_task_now(task.id) is True
            await asyncio.sleep(0.05)
            tasks = await scheduler.list_tasks()
            current = [t for t in tasks if t.id == task.id][0]
            assert int(current.runs) >= 2
            assert calls["n"] >= 2
        finally:
            await scheduler.stop()

    asyncio.run(_run())
