from __future__ import annotations

import asyncio
from dataclasses import dataclass

from annolid.core.agent.swarm import SwarmAgent, SwarmManager


@dataclass
class _Result:
    content: str


class _Loop:
    def __init__(self, content: str, *, fail: bool = False) -> None:
        self._content = content
        self._fail = fail

    async def run(self, *args, **kwargs) -> _Result:
        del args, kwargs
        if self._fail:
            raise RuntimeError("loop failed")
        return _Result(self._content)


def test_swarm_completes_when_agent_reports_task_complete() -> None:
    manager = SwarmManager()
    manager.register_agent(
        SwarmAgent(
            name="planner",
            role="planner",
            system_prompt="plan",
            loop_factory=lambda: _Loop("Draft plan. TASK COMPLETE"),
        )
    )

    output = asyncio.run(manager.run_swarm("Solve X", max_turns=3))
    assert "Swarm reached consensus: TASK COMPLETE." in output


def test_swarm_accepts_awaitable_loop_factory() -> None:
    manager = SwarmManager()

    async def _factory():
        return _Loop("awaitable loop response")

    manager.register_agent(
        SwarmAgent(
            name="async_agent",
            role="analyst",
            system_prompt="analyze",
            loop_factory=_factory,
        )
    )

    output = asyncio.run(manager.run_swarm("Task", max_turns=1))
    assert "awaitable loop response" in output


def test_swarm_isolates_agent_failures_and_continues() -> None:
    manager = SwarmManager()
    manager.register_agent(
        SwarmAgent(
            name="broken",
            role="executor",
            system_prompt="exec",
            loop_factory=lambda: _Loop("", fail=True),
        )
    )
    manager.register_agent(
        SwarmAgent(
            name="healthy",
            role="reviewer",
            system_prompt="review",
            loop_factory=lambda: _Loop("healthy output"),
        )
    )

    output = asyncio.run(manager.run_swarm("Task", max_turns=1))
    assert "[ERROR] loop failed" in output
    assert "healthy output" in output
