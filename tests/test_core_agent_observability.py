from __future__ import annotations

import json
from pathlib import Path

from annolid.core.agent.memory_store.store import WorkspaceMemoryStore
from annolid.core.agent.observability.store import emit_governance_event
from annolid.core.agent.skill_registry.registry import SkillRegistry
from annolid.core.agent.update_manager.rollback import (
    build_rollback_plan,
    execute_rollback,
)
from annolid.engine.cli import main as annolid_run


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def test_emit_governance_event_writes_ndjson(tmp_path: Path, monkeypatch) -> None:
    events_path = tmp_path / "events.ndjson"
    monkeypatch.setenv("ANNOLID_GOVERNANCE_EVENTS_PATH", str(events_path))
    row = emit_governance_event(
        event_type="skills",
        action="refresh",
        outcome="ok",
        actor="operator",
        details={"count": 1},
    )
    assert row.get("event_type") == "skills"
    rows = _read_events(events_path)
    assert len(rows) == 1
    assert rows[0]["action"] == "refresh"


def test_workspace_memory_store_logs_memory_writes(tmp_path: Path, monkeypatch) -> None:
    events_path = tmp_path / "events.ndjson"
    monkeypatch.setenv("ANNOLID_GOVERNANCE_EVENTS_PATH", str(events_path))
    store = WorkspaceMemoryStore(tmp_path)
    store.append_today("note")
    store.append_history("history")
    store.write_long_term("long-term")

    rows = _read_events(events_path)
    actions = [
        str(r.get("action") or "") for r in rows if r.get("event_type") == "memory"
    ]
    assert "append_today" in actions
    assert "append_history" in actions
    assert "write_long_term" in actions


def test_operator_skills_refresh_logs_skill_event(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    events_path = tmp_path / "events.ndjson"
    monkeypatch.setenv("ANNOLID_GOVERNANCE_EVENTS_PATH", str(events_path))
    workspace = tmp_path / "workspace"
    skill_dir = workspace / "skills" / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: demo\n---\ndemo\n",
        encoding="utf-8",
    )

    rc = annolid_run(["agent", "skills", "refresh", "--workspace", str(workspace)])
    assert rc == 0
    _ = capsys.readouterr().out

    rows = _read_events(events_path)
    assert any(
        r.get("event_type") == "skills" and r.get("action") == "refresh" for r in rows
    )


def test_rollback_logs_governance_event(tmp_path: Path, monkeypatch) -> None:
    events_path = tmp_path / "events.ndjson"
    monkeypatch.setenv("ANNOLID_GOVERNANCE_EVENTS_PATH", str(events_path))

    plan = build_rollback_plan(
        install_mode="source",
        project="annolid",
        previous_version="1.0.0",
    )
    payload = execute_rollback(plan, execute=True)
    assert payload["reason"] == "manual_required"

    rows = _read_events(events_path)
    assert any(
        r.get("event_type") == "update" and r.get("action") == "rollback" for r in rows
    )


def test_skill_registry_snapshot_logs_changes(tmp_path: Path, monkeypatch) -> None:
    events_path = tmp_path / "events.ndjson"
    monkeypatch.setenv("ANNOLID_GOVERNANCE_EVENTS_PATH", str(events_path))

    workspace = tmp_path / "workspace"
    skill_root = workspace / "skills"
    skill_root.mkdir(parents=True, exist_ok=True)
    (skill_root / "a").mkdir(parents=True, exist_ok=True)
    (skill_root / "a" / "SKILL.md").write_text("description: a", encoding="utf-8")

    registry = SkillRegistry(
        workspace=workspace,
        builtin_skills_dir=tmp_path / "builtin",
        managed_skills_dir=tmp_path / "managed",
        parse_meta=lambda meta: dict(meta or {}),
        read_frontmatter_from_path=lambda _: {},
        get_config_path=lambda: tmp_path / "missing.json",
        watch=False,
    )
    registry.refresh()

    (skill_root / "b").mkdir(parents=True, exist_ok=True)
    (skill_root / "b" / "SKILL.md").write_text("description: b", encoding="utf-8")
    registry.refresh()

    rows = _read_events(events_path)
    assert any(
        r.get("event_type") == "skills"
        and r.get("action") == "snapshot"
        and "b" in (r.get("details") or {}).get("added", [])
        for r in rows
    )
