from __future__ import annotations

import json
from pathlib import Path

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


def test_agent_memory_flush_writes_memory_files_and_event(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    workspace = tmp_path / "workspace"
    events_path = tmp_path / "events.ndjson"
    monkeypatch.setenv("ANNOLID_GOVERNANCE_EVENTS_PATH", str(events_path))

    rc = annolid_run(
        [
            "agent",
            "memory",
            "flush",
            "--workspace",
            str(workspace),
            "--session-id",
            "s1",
            "--note",
            "checkpoint",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["flushed"] is True

    today_file = Path(payload["today_file"])
    history_file = Path(payload["history_file"])
    assert today_file.exists()
    assert history_file.exists()
    assert "checkpoint" in today_file.read_text(encoding="utf-8")
    assert "checkpoint" in history_file.read_text(encoding="utf-8")

    rows = _read_events(events_path)
    assert any(
        r.get("event_type") == "memory" and r.get("action") == "operator_flush"
        for r in rows
    )
