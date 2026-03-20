from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.meta_learning import AgentMetaLearner
from annolid.core.agent.tools.function_base import FunctionTool
from annolid.core.agent.tools.function_registry import FunctionToolRegistry


class _FailingTool(FunctionTool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Simulate a repeated read_file error."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return '{"error":"File not found: /tmp/missing.txt"}'


def test_meta_learner_evolves_skill_after_repeated_failures(tmp_path: Path) -> None:
    learner = AgentMetaLearner(
        tmp_path,
        enabled=True,
        auto_evolve_skills=True,
        failure_threshold=2,
    )
    payload = [{"name": "read_file", "result": '{"error":"File not found: x"}'}]
    first = learner.record_turn(
        session_id="s1",
        user_text="open file",
        assistant_text="failed",
        tool_runs=payload,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=12.0,
    )
    assert first["recorded"] is True
    assert first["evolved_skills"] == []

    second = learner.record_turn(
        session_id="s1",
        user_text="open file again",
        assistant_text="failed again",
        tool_runs=payload,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=11.0,
        total_ms=13.0,
    )
    assert len(second["evolved_skills"]) == 1
    skill_name = str(second["evolved_skills"][0])
    skill_file = tmp_path / "skills" / skill_name / "SKILL.md"
    assert skill_file.exists()
    assert "Recover `read_file` Failures" in skill_file.read_text(encoding="utf-8")

    events = (tmp_path / "memory" / "meta_learning" / "events.jsonl").read_text(
        encoding="utf-8"
    )
    assert "session_id" in events


def test_agent_loop_meta_learning_integration_writes_events(tmp_path: Path) -> None:
    os.environ["ANNOLID_AGENT_META_LEARNING_ENABLED"] = "1"
    os.environ["ANNOLID_AGENT_META_LEARNING_AUTO_EVOLVE"] = "1"
    os.environ["ANNOLID_AGENT_META_LEARNING_FAILURE_THRESHOLD"] = "2"
    try:
        registry = FunctionToolRegistry()
        registry.register(_FailingTool())

        state = {"n": 0}

        async def fake_llm(
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]],
            model: str,
            on_token: Optional[Callable[[str], None]] = None,
        ) -> Mapping[str, Any]:
            del messages, tools, model, on_token
            state["n"] += 1
            if state["n"] % 2 == 1:
                return {
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "read_file", "arguments": {}}],
                }
            return {"content": "done", "tool_calls": []}

        loop = AgentLoop(
            tools=registry,
            llm_callable=fake_llm,
            model="fake",
            workspace=str(tmp_path),
        )
        asyncio.run(loop.run("do read", session_id="meta:s1"))
        asyncio.run(loop.run("do read again", session_id="meta:s1"))

        events_path = tmp_path / "memory" / "meta_learning" / "events.jsonl"
        assert events_path.exists()
        assert events_path.read_text(encoding="utf-8").count("\n") >= 2
        skills_root = tmp_path / "skills"
        evolved = [
            p
            for p in skills_root.iterdir()
            if p.is_dir() and p.name.startswith("meta-recover-")
        ]
        assert evolved
    finally:
        os.environ.pop("ANNOLID_AGENT_META_LEARNING_ENABLED", None)
        os.environ.pop("ANNOLID_AGENT_META_LEARNING_AUTO_EVOLVE", None)
        os.environ.pop("ANNOLID_AGENT_META_LEARNING_FAILURE_THRESHOLD", None)


def test_meta_learner_llm_evolver_generates_custom_skill(tmp_path: Path) -> None:
    class _FakeLLM:
        def chat_complete(self, prompt: str) -> str:
            assert "Tool: read_file" in prompt
            return (
                '{"name":"meta-open-safe-file",'
                '"description":"Use when file opens fail repeatedly due to path issues.",'
                '"content":"## Recover File Opens\\n\\n1. Normalize the path.\\n2. Verify existence before read.\\n3. Retry once with corrected path.\\n\\n**Anti-pattern:** blind retries."}'
            )

    learner = AgentMetaLearner(
        tmp_path,
        enabled=True,
        auto_evolve_skills=True,
        failure_threshold=2,
        llm_evolver_enabled=True,
        llm_client=_FakeLLM(),
    )
    payload = [{"name": "read_file", "result": '{"error":"File not found: x"}'}]
    learner.record_turn(
        session_id="s2",
        user_text="first",
        assistant_text="",
        tool_runs=payload,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    second = learner.record_turn(
        session_id="s2",
        user_text="second",
        assistant_text="",
        tool_runs=payload,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    assert second["evolved_skills"] == ["meta-open-safe-file"]
    skill_file = tmp_path / "skills" / "meta-open-safe-file" / "SKILL.md"
    assert skill_file.exists()
    text = skill_file.read_text(encoding="utf-8")
    assert "Recover File Opens" in text


def test_meta_learner_llm_evolver_parses_fenced_json(tmp_path: Path) -> None:
    class _FakeLLM:
        def chat_complete(self, prompt: str) -> str:
            del prompt
            return (
                "```json\n"
                '{"name":"meta-safe-web-fetch","description":"Use when web fetch errors repeat.","content":"## Recover Web Fetch\\n\\n1. Check URL syntax.\\n2. Retry with shorter timeout.\\n\\n**Anti-pattern:** same request spam."}\n'
                "```"
            )

    learner = AgentMetaLearner(
        tmp_path,
        enabled=True,
        auto_evolve_skills=True,
        failure_threshold=2,
        llm_evolver_enabled=True,
        llm_client=_FakeLLM(),
    )
    payload = [
        {
            "name": "web_fetch",
            "result": '{"error":"Timeout while fetching https://a.com"}',
        }
    ]
    learner.record_turn(
        session_id="s3",
        user_text="first",
        assistant_text="",
        tool_runs=payload,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    second = learner.record_turn(
        session_id="s3",
        user_text="second",
        assistant_text="",
        tool_runs=payload,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    assert second["evolved_skills"] == ["meta-safe-web-fetch"]
    assert (tmp_path / "skills" / "meta-safe-web-fetch" / "SKILL.md").exists()


def test_meta_learner_normalizes_paths_for_stable_signatures(tmp_path: Path) -> None:
    learner = AgentMetaLearner(
        tmp_path,
        enabled=True,
        auto_evolve_skills=True,
        failure_threshold=2,
    )
    payload_a = [
        {"name": "read_file", "result": '{"error":"File not found: /tmp/a.txt"}'}
    ]
    payload_b = [
        {"name": "read_file", "result": '{"error":"File not found: /tmp/b.txt"}'}
    ]
    learner.record_turn(
        session_id="s4",
        user_text="a",
        assistant_text="",
        tool_runs=payload_a,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    second = learner.record_turn(
        session_id="s4",
        user_text="b",
        assistant_text="",
        tool_runs=payload_b,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    assert len(second["evolved_skills"]) == 1


def test_meta_learner_reward_window_triggers_evolution(tmp_path: Path) -> None:
    learner = AgentMetaLearner(
        tmp_path,
        enabled=True,
        auto_evolve_skills=True,
        failure_threshold=99,
    )
    payload = [
        {"name": "read_file", "result": '{"error":"File not found: /tmp/x.txt"}'}
    ]
    for i in range(3):
        out = learner.record_turn(
            session_id="reward:s1",
            user_text=f"turn {i}",
            assistant_text="",
            tool_runs=payload,
            stopped_reason="max_iterations",
            empty_repair_used=True,
            llm_total_ms=10.0,
            total_ms=11.0,
        )
    assert out["evolved_skills"]
    history_path = tmp_path / "memory" / "meta_learning" / "evolution_history.jsonl"
    assert history_path.exists()
    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(str(r.get("trigger") or "") == "reward_window" for r in rows)


def test_meta_learner_idle_scheduler_queues_and_runs_on_force(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANNOLID_AGENT_META_LEARNING_IDLE_SCHEDULER_ENABLED", "1")
    monkeypatch.setenv("ANNOLID_AGENT_META_LEARNING_IDLE_MIN_SECONDS", "86400")
    monkeypatch.setenv("ANNOLID_AGENT_META_LEARNING_IDLE_SLEEP_START", "invalid")
    learner = AgentMetaLearner(
        tmp_path,
        enabled=True,
        auto_evolve_skills=True,
        failure_threshold=2,
    )
    payload = [
        {"name": "read_file", "result": '{"error":"File not found: /tmp/a.txt"}'}
    ]
    learner.record_turn(
        session_id="idle:s1",
        user_text="a",
        assistant_text="",
        tool_runs=payload,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    second = learner.record_turn(
        session_id="idle:s1",
        user_text="b",
        assistant_text="",
        tool_runs=payload,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    assert second["evolved_skills"] == []
    maintenance = learner.run_idle_maintenance(force=True, max_jobs=5)
    assert maintenance["processed_jobs"] == 1
    assert maintenance["evolved_skills"]


def test_meta_learner_generation_increments_on_evolution(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANNOLID_AGENT_META_LEARNING_IDLE_SCHEDULER_ENABLED", "0")
    learner = AgentMetaLearner(
        tmp_path,
        enabled=True,
        auto_evolve_skills=True,
        failure_threshold=2,
    )
    payload = [
        {"name": "read_file", "result": '{"error":"File not found: /tmp/a.txt"}'}
    ]
    learner.record_turn(
        session_id="gen:s1",
        user_text="a",
        assistant_text="",
        tool_runs=payload,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    second = learner.record_turn(
        session_id="gen:s1",
        user_text="b",
        assistant_text="",
        tool_runs=payload,
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    assert second["evolved_skills"]
    generation_state = json.loads(
        (tmp_path / "memory" / "meta_learning" / "generation_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert int(generation_state.get("current_generation") or 0) >= 1


def test_meta_learner_discards_stale_jobs_by_generation(tmp_path: Path) -> None:
    learner = AgentMetaLearner(
        tmp_path,
        enabled=True,
        auto_evolve_skills=True,
        failure_threshold=99,
    )
    meta_dir = tmp_path / "memory" / "meta_learning"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "generation_state.json").write_text(
        json.dumps({"current_generation": 3}),
        encoding="utf-8",
    )
    (meta_dir / "pending_evolution_jobs.json").write_text(
        json.dumps(
            {
                "read_file:s1": {
                    "signature": "read_file:s1",
                    "tool_name": "read_file",
                    "reason": "File not found: /tmp/x.txt",
                    "trigger": "failure_threshold",
                    "signature_count": 3,
                    "outcome_score": 0.2,
                    "reward_window_avg": 0.2,
                    "queued_generation": 1,
                    "queued_at": "2026-03-20T12:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )
    result = learner.run_idle_maintenance(force=True, max_jobs=5)
    assert result["processed_jobs"] == 0
    assert result["discarded_stale_jobs"] == 1
    pending = json.loads(
        (meta_dir / "pending_evolution_jobs.json").read_text(encoding="utf-8")
    )
    assert pending == {}


def test_meta_learner_prm_majority_scoring(tmp_path: Path) -> None:
    class _FakePRM:
        def __init__(self) -> None:
            self.calls = 0

        def chat_complete(self, prompt: str) -> str:
            assert "Return ONLY one line in format: Score:" in prompt
            self.calls += 1
            if self.calls <= 2:
                return "Score: -1"
            return "Score: 1"

    prm = _FakePRM()
    learner = AgentMetaLearner(
        tmp_path,
        enabled=True,
        auto_evolve_skills=False,
        prm_enabled=True,
        prm_client=prm,
        prm_votes=3,
    )
    out = learner.record_turn(
        session_id="prm:s1",
        user_text="read file",
        assistant_text="not useful",
        tool_runs=[{"name": "read_file", "result": '{"error":"x"}'}],
        stopped_reason="done",
        empty_repair_used=False,
        llm_total_ms=10.0,
        total_ms=11.0,
    )
    assert out["recorded"] is True
    events_path = tmp_path / "memory" / "meta_learning" / "events.jsonl"
    row = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])
    assert float(row.get("outcome_score") or 0.0) == 0.0
