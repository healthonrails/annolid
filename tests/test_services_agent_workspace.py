from __future__ import annotations

import json
from pathlib import Path

from annolid.services.agent_workspace import (
    add_agent_feedback,
    flush_agent_memory,
    inspect_agent_meta_learning,
    inspect_agent_meta_learning_history,
    inspect_agent_meta_learning_maintenance_status,
    inspect_agent_memory,
    inspect_agent_skills,
    refresh_agent_skills,
    run_agent_meta_learning_maintenance,
    shadow_agent_skills,
)


def test_refresh_agent_skills_emits_event(monkeypatch, tmp_path: Path) -> None:
    import annolid.core.agent.observability as obs_mod
    import annolid.core.agent.skills as skills_mod
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )

    calls: dict[str, object] = {}

    class _Loader:
        def __init__(self, *, workspace):
            calls["workspace"] = workspace
            self._refreshed = False

        def list_skills(self, filter_unavailable=False):
            if not self._refreshed:
                return [{"name": "alpha"}, {"name": "beta"}]
            return [{"name": "alpha"}, {"name": "gamma"}]

        def refresh_snapshot(self):
            self._refreshed = True

    monkeypatch.setattr(skills_mod, "AgentSkillsLoader", _Loader)
    monkeypatch.setattr(
        obs_mod,
        "emit_governance_event",
        lambda **kwargs: calls.setdefault("event", kwargs),
    )

    payload = refresh_agent_skills(workspace=str(workspace))

    assert payload["added"] == ["gamma"]
    assert payload["removed"] == ["beta"]
    assert calls["workspace"] == workspace
    assert calls["event"]["event_type"] == "skills"


def test_inspect_and_shadow_agent_skills(monkeypatch, tmp_path: Path) -> None:
    import annolid.core.agent.skill_registry as registry_mod
    import annolid.core.agent.skills as skills_mod
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )

    class _Loader:
        def __init__(self, *, workspace):
            self.workspace = workspace

        def list_skills(self, filter_unavailable=False):
            return [
                {
                    "name": "ok",
                    "source": "workspace",
                    "manifest_valid": True,
                    "path": "ok/SKILL.md",
                },
                {
                    "name": "broken",
                    "source": "workspace",
                    "manifest_valid": False,
                    "manifest_errors": ["missing description"],
                    "path": "broken/SKILL.md",
                },
            ]

    monkeypatch.setattr(skills_mod, "AgentSkillsLoader", _Loader)
    monkeypatch.setattr(
        registry_mod,
        "flatten_skills_by_name",
        lambda rows: {"ok": rows[0], "broken": rows[1]},
    )
    monkeypatch.setattr(
        registry_mod,
        "compare_skill_pack_shadow",
        lambda *, active_skills, candidate_pack_dir: {
            "candidate_pack_dir": str(candidate_pack_dir),
            "overlap": sorted(active_skills.keys()),
        },
    )

    inspect_payload = inspect_agent_skills(workspace=str(workspace))
    shadow_payload = shadow_agent_skills(
        workspace=str(workspace), candidate_pack=tmp_path / "candidate"
    )

    assert inspect_payload["invalid_manifest_count"] == 1
    assert inspect_payload["invalid_skills"][0]["name"] == "broken"
    assert shadow_payload["workspace"] == str(workspace)
    assert shadow_payload["overlap"] == ["broken", "ok"]


def test_add_feedback_and_memory_ops(monkeypatch, tmp_path: Path) -> None:
    import annolid.core.agent.eval.telemetry as telemetry_mod
    import annolid.core.agent.memory as memory_mod
    import annolid.core.agent.observability as obs_mod
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )

    captured: dict[str, object] = {}

    class _RunTraceStore:
        def __init__(self, workspace_path):
            captured["feedback_workspace"] = workspace_path

        def capture_feedback(self, **kwargs):
            captured["feedback_args"] = kwargs
            return {"rating": kwargs["rating"], "trace_id": kwargs["trace_id"]}

    class _MemoryStore:
        def __init__(self, workspace_path):
            captured["memory_workspace"] = workspace_path
            self.memory_dir = workspace_path / "memory"
            self.memory_file = self.memory_dir / "MEMORY.md"
            self.history_file = self.memory_dir / "HISTORY.md"
            self.retrieval_plugin_name = "lexical"
            self.today_entries: list[str] = []
            self.history_entries: list[str] = []

        def append_today(self, entry: str) -> None:
            self.today_entries.append(entry)
            captured["today_entry"] = entry

        def append_history(self, entry: str) -> None:
            self.history_entries.append(entry)
            captured["history_entry"] = entry

        def get_today_file(self):
            return self.memory_dir / "today.md"

    monkeypatch.setattr(telemetry_mod, "RunTraceStore", _RunTraceStore)
    monkeypatch.setattr(memory_mod, "AgentMemoryStore", _MemoryStore)
    monkeypatch.setattr(
        obs_mod,
        "emit_governance_event",
        lambda **kwargs: captured.setdefault("memory_event", kwargs),
    )

    feedback_payload = add_agent_feedback(
        workspace=str(workspace),
        session_id="s1",
        rating=1,
        comment="good",
        trace_id="t1",
        expected_substring="done",
    )
    flush_payload = flush_agent_memory(
        workspace=str(workspace),
        session_id="s1",
        note="manual flush",
    )
    inspect_payload = inspect_agent_memory(workspace=str(workspace))

    assert feedback_payload["saved"] is True
    assert captured["feedback_args"]["trace_id"] == "t1"
    assert "manual flush" in flush_payload["entry"]
    assert "(session_id=s1)" in flush_payload["entry"]
    assert inspect_payload["retrieval_plugin"] == "lexical"
    assert captured["memory_event"]["event_type"] == "memory"


def test_inspect_agent_meta_learning(tmp_path: Path, monkeypatch) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    meta_dir = workspace / "memory" / "meta_learning"
    skills_dir = workspace / "skills"
    meta_dir.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "meta-recover-read-file-12345678").mkdir()
    (meta_dir / "events.jsonl").write_text(
        (
            '{"session_id":"s1","failures":[{"tool":"read_file","signature":"read_file:a1","count":1}]}\n'
            '{"session_id":"s2","failures":[{"tool":"read_file","signature":"read_file:a1","count":1},{"tool":"web_fetch","signature":"web_fetch:b2","count":1}]}\n'
        ),
        encoding="utf-8",
    )
    (meta_dir / "failure_patterns.json").write_text(
        '{"read_file:a1": 3, "web_fetch:b2": 1}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )

    payload = inspect_agent_meta_learning(workspace=str(workspace), limit=5)
    assert payload["events_count"] == 2
    assert payload["evolved_skill_count"] == 1
    assert payload["evolved_skills"] == ["meta-recover-read-file-12345678"]
    assert payload["top_patterns"][0]["signature"] == "read_file:a1"
    assert payload["top_patterns"][0]["count"] == 3
    assert payload["top_failure_tools"][0]["tool"] == "read_file"


def test_inspect_agent_meta_learning_brief(tmp_path: Path, monkeypatch) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    meta_dir = workspace / "memory" / "meta_learning"
    skills_dir = workspace / "skills"
    meta_dir.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "meta-recover-read-file-12345678").mkdir()
    (meta_dir / "events.jsonl").write_text(
        '{"session_id":"s1","failures":[{"tool":"read_file","signature":"read_file:a1","count":1}]}\n',
        encoding="utf-8",
    )
    (meta_dir / "failure_patterns.json").write_text(
        '{"read_file:a1": 3}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )

    payload = inspect_agent_meta_learning(workspace=str(workspace), brief=True)
    assert payload["events_count"] == 1
    assert payload["evolved_skill_count"] == 1
    assert "events_path" not in payload
    assert "patterns_path" not in payload
    assert "recent_events" not in payload
    assert "evolved_skills" not in payload


def test_inspect_agent_meta_learning_history(tmp_path: Path, monkeypatch) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    history_path = workspace / "memory" / "meta_learning" / "evolution_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        (
            '{"ts":"2026-03-20T12:00:00+00:00","trigger":"failure_threshold","tool":"read_file","skill_name":"meta-recover-read-file-a1"}\n'
            '{"ts":"2026-03-20T12:01:00+00:00","trigger":"reward_window","tool":"read_file","skill_name":"meta-recover-read-file-a2"}\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )

    payload = inspect_agent_meta_learning_history(workspace=str(workspace), limit=1)
    assert payload["events_count"] == 2
    assert len(payload["events"]) == 1
    assert payload["events"][0]["trigger"] == "reward_window"
    assert payload["skills_generated_in_window"] == 1
    assert payload["top_triggers"][0]["trigger"] == "reward_window"


def test_run_agent_meta_learning_maintenance(tmp_path: Path, monkeypatch) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    pending_path = (
        workspace / "memory" / "meta_learning" / "pending_evolution_jobs.json"
    )
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(
        json.dumps(
            {
                "read_file:testsig": {
                    "signature": "read_file:testsig",
                    "tool_name": "read_file",
                    "reason": "File not found: /tmp/x.txt",
                    "trigger": "failure_threshold",
                    "signature_count": 3,
                    "outcome_score": 0.2,
                    "reward_window_avg": 0.3,
                    "queued_at": "2026-03-20T12:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )

    payload = run_agent_meta_learning_maintenance(
        workspace=str(workspace), force=True, max_jobs=1
    )
    assert payload["processed_jobs"] == 1
    assert isinstance(payload["evolved_skills"], list)


def test_inspect_agent_meta_learning_maintenance_status(
    tmp_path: Path, monkeypatch
) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    meta_dir = workspace / "memory" / "meta_learning"
    pending_path = meta_dir / "pending_evolution_jobs.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(
        json.dumps(
            {
                "sig1": {
                    "signature": "sig1",
                    "tool_name": "read_file",
                    "reason": "x",
                    "queued_at": "2026-03-20T12:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )
    payload = inspect_agent_meta_learning_maintenance_status(workspace=str(workspace))
    assert payload["pending_jobs_count"] == 1
    assert "window_open" in payload
