from __future__ import annotations

import importlib
import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

import annolid.core.agent.loop as loop_module
from annolid.core.agent.swarm import SwarmAgent, SwarmManager
import annolid.core.agent.swarm_budget as swarm_budget_module
from annolid.core.agent.swarm_budget import (
    record_swarm_budget_observation,
    reset_swarm_budget_history,
    resolve_swarm_turn_budget,
)
from annolid.gui.widgets import threejs_viewer_server as srv


@dataclass
class _Result:
    content: str


class _Loop:
    def __init__(
        self,
        content: str,
        *,
        fail: bool = False,
        prompt_log: list[str] | None = None,
    ) -> None:
        self._content = content
        self._fail = fail
        self._prompt_log = prompt_log
        self.closed = 0

    async def run(self, *args, **kwargs) -> _Result:
        if args and self._prompt_log is not None:
            self._prompt_log.append(str(args[0]))
        del kwargs
        if self._fail:
            raise RuntimeError("loop failed")
        return _Result(self._content)

    async def close(self) -> None:
        self.closed += 1


@pytest.fixture(autouse=True)
def _reset_swarm_budget_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        "ANNOLID_SWARM_BUDGET_HISTORY_PATH",
        str(tmp_path / "swarm_budget_history.json"),
    )
    reset_swarm_budget_history()
    yield
    reset_swarm_budget_history()


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


def test_swarm_reuses_resolved_loops_and_closes_them_once() -> None:
    manager = SwarmManager()
    loop = _Loop("still working")
    calls = {"factory": 0}

    def _factory() -> _Loop:
        calls["factory"] += 1
        return loop

    manager.register_agent(
        SwarmAgent(
            name="worker",
            role="worker",
            system_prompt="work",
            loop_factory=_factory,
        )
    )

    output = asyncio.run(manager.run_swarm("Task", max_turns=3))
    assert "Swarm max turns reached." in output
    assert calls["factory"] == 1
    assert loop.closed == 1


def test_swarm_compacts_context_with_recent_turns_and_agent_state() -> None:
    manager = SwarmManager()
    manager.max_context_chars = 1200
    prompt_log: list[str] = []

    manager.register_agent(
        SwarmAgent(
            name="planner",
            role="planner",
            system_prompt="plan",
            loop_factory=lambda: _Loop(
                "Planner insight " + ("A" * 1800),
            ),
        )
    )
    manager.register_agent(
        SwarmAgent(
            name="reviewer",
            role="reviewer",
            system_prompt="review",
            loop_factory=lambda: _Loop(
                "TASK COMPLETE",
                prompt_log=prompt_log,
            ),
        )
    )

    output = asyncio.run(manager.run_swarm("Analyze X", max_turns=1))
    assert "TASK COMPLETE" in output
    assert prompt_log
    prompt = prompt_log[0]
    assert "# Swarm Task" in prompt
    assert "# Recent Swarm Turns" in prompt
    assert "# Condensed Agent State" in prompt
    assert "planner" in prompt
    assert "latest=Planner insight" in prompt or "latest=Planner insight ..." in prompt


def test_swarm_records_turn_latency_for_visualizer() -> None:
    manager = SwarmManager()

    manager.register_agent(
        SwarmAgent(
            name="worker",
            role="worker",
            system_prompt="work",
            loop_factory=lambda: _Loop("done"),
        )
    )

    original_perf_counter = srv.time.perf_counter
    try:
        # Keep the latency deterministic so we verify the telemetry wiring, not wall clock jitter.
        counter = iter([100.0, 100.275])
        import annolid.core.agent.swarm as swarm_module

        swarm_module.time.perf_counter = lambda: next(counter)

        output = asyncio.run(manager.run_swarm("Task", max_turns=1))
        assert "done" in output
        state = srv.get_swarm_state()
        assert state["worker"]["turn_latency_ms"] == pytest.approx(275.0)
        assert state["worker"]["status"] == "idle"
    finally:
        import annolid.core.agent.swarm as swarm_module

        swarm_module.time.perf_counter = original_perf_counter


def test_subagent_loop_factory_uses_roomier_iteration_budget(monkeypatch) -> None:
    class _ToolRegistry:
        def get(self, _name):
            return None

        def values(self):
            return []

    async def _llm_callable(messages, tools, model, on_token=None):
        del messages, tools, model, on_token
        return {"content": "", "finish_reason": "stop"}

    parent_loop = loop_module.AgentLoop(
        tools=_ToolRegistry(),
        llm_callable=_llm_callable,
        model="test-model",
        max_iterations=9,
    )
    factory = parent_loop._build_subagent_loop_factory()

    captured: dict[str, object] = {}

    class _DummyLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(loop_module, "AgentLoop", _DummyLoop)

    built = factory()
    assert built is not None
    assert captured["max_iterations"] == 5


def test_resolve_swarm_turn_budget_is_adaptive() -> None:
    assert resolve_swarm_turn_budget("quick task", 8) == 8
    assert (
        resolve_swarm_turn_budget(
            "Draft a research paper with literature review and citations",
            8,
            agent_count=3,
            paper_context=True,
        )
        > 8
    )
    assert resolve_swarm_turn_budget("Short explicit cap", 3) == 3


def test_swarm_budget_history_increases_budget_after_timeout() -> None:
    task = "Draft a research paper with literature review and citations"
    baseline = swarm_budget_module.resolve_swarm_turn_budget(
        task,
        8,
        agent_count=3,
        paper_context=True,
    )
    record_swarm_budget_observation(
        task,
        requested_turns=8,
        used_turns=8,
        completed=False,
        timed_out=True,
        agent_count=3,
        paper_context=True,
    )

    reloaded = importlib.reload(swarm_budget_module)
    boosted = reloaded.resolve_swarm_turn_budget(
        task,
        8,
        agent_count=3,
        paper_context=True,
    )
    assert boosted > baseline


def test_swarm_budget_history_recovers_from_backup_after_primary_corruption(
    tmp_path, monkeypatch
) -> None:
    history_path = tmp_path / "swarm_budget_history.json"
    backup_path = Path(f"{history_path}.bak")
    monkeypatch.setenv(
        "ANNOLID_SWARM_BUDGET_HISTORY_PATH",
        str(history_path),
    )
    history_path.write_text("{not valid json", encoding="utf-8")
    backup_path.write_text(
        """
        {
          "version": 1,
          "observations": {
            "paper": [
              {
                "task_key": "paper",
                "requested_turns": 8,
                "used_turns": 8,
                "completed": false,
                "timed_out": true,
                "agent_count": 3,
                "paper_context": true
              }
            ]
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    module = importlib.reload(swarm_budget_module)
    recovered = module.resolve_swarm_turn_budget(
        "Draft a research paper with literature review and citations",
        8,
        agent_count=3,
        paper_context=True,
    )
    assert recovered > 8
