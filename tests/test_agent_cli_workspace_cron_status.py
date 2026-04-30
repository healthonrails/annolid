from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from annolid.engine.cli import main as annolid_run


def test_agent_onboard_creates_workspace_templates(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    rc = annolid_run(["agent-onboard", "--workspace", str(workspace)])
    assert rc == 0
    assert (workspace / "AGENTS.md").exists()
    assert (workspace / "SOUL.md").exists()
    assert (workspace / "USER.md").exists()
    assert (workspace / "IDENTITY.md").exists()
    assert (workspace / "TOOLS.md").exists()
    assert (workspace / "HEARTBEAT.md").exists()
    assert (workspace / "BOOT.md").exists()
    assert (workspace / "BOOTSTRAP.md").exists()
    assert (workspace / "memory" / "MEMORY.md").exists()
    assert (workspace / "memory" / "HISTORY.md").exists()


def test_agent_onboard_dry_run_does_not_write(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    rc = annolid_run(["agent-onboard", "--workspace", str(workspace), "--dry-run"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["files"]["AGENTS.md"] == "would_created"
    assert not (workspace / "AGENTS.md").exists()


def test_agent_onboard_update_creates_backup(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    rc = annolid_run(["agent-onboard", "--workspace", str(workspace)])
    assert rc == 0
    _ = capsys.readouterr()

    agents = workspace / "AGENTS.md"
    agents.write_text("custom", encoding="utf-8")
    rc = annolid_run(["agent-onboard", "--workspace", str(workspace), "--update"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overwrite"] is True
    assert payload["backup_enabled"] is True
    assert payload["files"]["AGENTS.md"] == "overwritten"
    backup_dir = Path(payload["backup_dir"])
    assert (backup_dir / "AGENTS.md").exists()


def test_agent_onboard_prune_bootstrap_removes_stale_file(
    tmp_path: Path, capsys
) -> None:
    workspace = tmp_path / "workspace"
    rc = annolid_run(["agent-onboard", "--workspace", str(workspace)])
    assert rc == 0
    _ = capsys.readouterr()

    stale_rel = "legacy/OLD.md"
    stale_path = workspace / stale_rel
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("legacy", encoding="utf-8")

    manifest_path = workspace / ".annolid" / "bootstrap-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["managed_files"] = list(payload.get("managed_files") or []) + [stale_rel]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    rc = annolid_run(
        [
            "agent-onboard",
            "--workspace",
            str(workspace),
            "--prune-bootstrap",
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["prune_bootstrap"] is True
    assert result["pruned_files"][stale_rel] == "removed"
    assert not stale_path.exists()


def test_agent_status_uses_patched_paths(tmp_path: Path, monkeypatch, capsys) -> None:
    import annolid.services.agent_cron as cron_mod
    import annolid.core.agent.utils as utils_mod

    data_dir = tmp_path / "data"
    workspace = data_dir / "workspace"
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser()
        if workspace
        else (data_dir / "workspace"),
    )
    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
    monkeypatch.setattr(
        cron_mod,
        "_default_agent_cron_store_path",
        lambda: data_dir / "cron" / "jobs.json",
    )

    # create templates so status reflects true values
    _ = annolid_run(["agent-onboard"])
    _ = capsys.readouterr()
    rc = annolid_run(["agent-status"])
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["workspace"] == str(workspace)
    assert payload["data_dir"] == str(data_dir)
    assert payload["workspace_templates"]["AGENTS.md"] is True


def test_agent_tool_pool_cli(monkeypatch, capsys) -> None:
    import annolid.services.agent_tooling as tooling_mod

    monkeypatch.setattr(
        tooling_mod,
        "describe_agent_tool_pool",
        lambda **kwargs: {
            "workspace": str(kwargs.get("workspace") or ""),
            "provider": str(kwargs.get("provider") or ""),
            "model": str(kwargs.get("model") or ""),
            "counts": {"registered": 2, "allowed": 1, "denied": 1},
            "allowed_tools": ["read_file"],
            "denied_tools": ["write_file"],
            "permission_context": {"deny_names": ["write_file"], "deny_prefixes": []},
        },
    )
    rc = annolid_run(
        [
            "agent-tool-pool",
            "--workspace",
            "/tmp/ws",
            "--provider",
            "ollama",
            "--model",
            "qwen3",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace"] == "/tmp/ws"
    assert payload["counts"]["allowed"] == 1
    assert payload["denied_tools"] == ["write_file"]


def test_agent_skill_pool_cli(monkeypatch, capsys) -> None:
    import annolid.services.agent_tooling as tooling_mod

    monkeypatch.setattr(
        tooling_mod,
        "describe_agent_skill_pool",
        lambda **kwargs: {
            "workspace": str(kwargs.get("workspace") or ""),
            "task_hint": str(kwargs.get("task_hint") or ""),
            "skill_pool": {
                "counts": {"total": 3, "available": 2, "unavailable": 1, "always": 1}
            },
            "suggested_skills": [
                {
                    "name": "weather",
                    "strategy": "lexical",
                    "score": 1.0,
                    "source": "workspace",
                }
            ],
        },
    )
    rc = annolid_run(
        [
            "agent-skill-pool",
            "--workspace",
            "/tmp/ws",
            "--task-hint",
            "weather forecast",
            "--top-k",
            "3",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace"] == "/tmp/ws"
    assert payload["task_hint"] == "weather forecast"
    assert payload["skill_pool"]["counts"]["total"] == 3
    assert payload["suggested_skills"][0]["name"] == "weather"


def test_agent_capabilities_cli(monkeypatch, capsys) -> None:
    import annolid.services.agent_tooling as tooling_mod

    monkeypatch.setattr(
        tooling_mod,
        "describe_agent_capabilities",
        lambda **kwargs: {
            "workspace": str(kwargs.get("workspace") or ""),
            "provider": str(kwargs.get("provider") or ""),
            "model": str(kwargs.get("model") or ""),
            "tool_pool": {"counts": {"registered": 4, "allowed": 3, "denied": 1}},
            "skill_pool": {
                "skill_pool": {"counts": {"total": 2, "available": 2, "unavailable": 0}}
            },
            "summary": {
                "registered_tools": 4,
                "available_tools": 3,
                "available_skills": 2,
                "suggested_skills": 1,
            },
        },
    )
    rc = annolid_run(
        [
            "agent-capabilities",
            "--workspace",
            "/tmp/ws",
            "--provider",
            "ollama",
            "--model",
            "qwen3",
            "--task-hint",
            "weather forecast",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace"] == "/tmp/ws"
    assert payload["tool_pool"]["counts"]["registered"] == 4
    assert payload["skill_pool"]["skill_pool"]["counts"]["available"] == 2


def test_agent_turn_snapshots_cli(tmp_path: Path, capsys) -> None:
    from annolid.core.agent.session_manager import AgentSessionManager

    workspace = tmp_path / "workspace"
    manager = AgentSessionManager(workspace=workspace)
    _ = manager.append_snapshot(
        "email:test@example.com",
        {"turn_id": "turn-1", "stopped_reason": "done", "tool_call_count": 1},
        max_entries=20,
    )
    _ = manager.append_snapshot(
        "email:test@example.com",
        {"turn_id": "turn-2", "stopped_reason": "max_iterations", "tool_call_count": 2},
        max_entries=20,
    )

    rc = annolid_run(
        [
            "agent-turn-snapshots",
            "--workspace",
            str(workspace),
            "--session-id",
            "email:test@example.com",
            "--limit",
            "5",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace"] == str(workspace)
    assert payload["session_id"] == "email:test@example.com"
    assert payload["count"] == 2
    assert payload["snapshots"][-1]["turn_id"] == "turn-2"


def test_agent_meta_learning_status_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    meta_dir = workspace / "memory" / "meta_learning"
    skills_dir = workspace / "skills"
    meta_dir.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "meta-recover-read-file-deadbeef").mkdir()
    (meta_dir / "events.jsonl").write_text(
        '{"session_id":"s1","failures":[{"tool":"read_file","signature":"read_file:x1","count":1}]}\n',
        encoding="utf-8",
    )
    (meta_dir / "failure_patterns.json").write_text(
        '{"read_file:x1": 2}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else workspace,
    )

    rc = annolid_run(["agent-meta-learning-status", "--workspace", str(workspace)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace"] == str(workspace)
    assert payload["events_count"] == 1
    assert payload["evolved_skill_count"] == 1
    assert payload["top_patterns"][0]["signature"] == "read_file:x1"


def test_agent_meta_learning_status_operator_alias(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    meta_dir = workspace / "memory" / "meta_learning"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "events.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else workspace,
    )

    rc = annolid_run(
        [
            "agent",
            "meta-learning",
            "status",
            "--workspace",
            str(workspace),
            "--limit",
            "5",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace"] == str(workspace)
    assert payload["events_count"] == 0


def test_agent_meta_learning_status_operator_alias_without_options(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    ws = tmp_path / "workspace"
    meta_dir = ws / "memory" / "meta_learning"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "events.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else ws,
    )

    rc = annolid_run(["agent", "meta-learning", "status"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "events_count" in payload


def test_agent_meta_learning_status_cli_brief(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    meta_dir = workspace / "memory" / "meta_learning"
    skills_dir = workspace / "skills"
    meta_dir.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "meta-recover-read-file-deadbeef").mkdir()
    (meta_dir / "events.jsonl").write_text(
        '{"session_id":"s1","failures":[{"tool":"read_file","signature":"read_file:x1","count":1}]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else workspace,
    )

    rc = annolid_run(
        ["agent-meta-learning-status", "--workspace", str(workspace), "--brief"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["events_count"] == 1
    assert "recent_events" not in payload
    assert "evolved_skills" not in payload


def test_agent_meta_learning_status_operator_alias_brief(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    ws = tmp_path / "workspace"
    meta_dir = ws / "memory" / "meta_learning"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "events.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else ws,
    )

    rc = annolid_run(["agent", "meta-learning", "status", "--brief"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "events_count" in payload
    assert "recent_events" not in payload


def test_agent_meta_learning_history_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    history_path = workspace / "memory" / "meta_learning" / "evolution_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        '{"ts":"2026-03-20T12:00:00+00:00","trigger":"reward_window","tool":"read_file"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else workspace,
    )

    rc = annolid_run(
        ["agent-meta-learning-history", "--workspace", str(workspace), "--limit", "5"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["events_count"] == 1
    assert payload["events"][0]["trigger"] == "reward_window"


def test_agent_meta_learning_history_operator_alias(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    ws = tmp_path / "workspace"
    history_path = ws / "memory" / "meta_learning" / "evolution_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else ws,
    )

    rc = annolid_run(["agent", "meta-learning", "history"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["events_count"] == 0


def test_agent_meta_learning_history_cli_full(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    history_path = workspace / "memory" / "meta_learning" / "evolution_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        '{"ts":"2026-03-20T12:00:00+00:00","trigger":"reward_window","tool":"read_file","skill_name":"s1","skill":{"name":"s1","content_excerpt":"abc"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else workspace,
    )

    rc = annolid_run(
        [
            "agent-meta-learning-history",
            "--workspace",
            str(workspace),
            "--full",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["events"][0]["skill"]["content_excerpt"] == "abc"


def test_agent_meta_learning_maintenance_run_cli(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    pending_path = (
        workspace / "memory" / "meta_learning" / "pending_evolution_jobs.json"
    )
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(
        json.dumps(
            {
                "read_file:abc123": {
                    "signature": "read_file:abc123",
                    "tool_name": "read_file",
                    "reason": "File not found: /tmp/a.txt",
                    "trigger": "failure_threshold",
                    "signature_count": 2,
                    "outcome_score": 0.2,
                    "reward_window_avg": 0.2,
                    "queued_at": "2026-03-20T12:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else workspace,
    )
    rc = annolid_run(
        [
            "agent-meta-learning-maintenance-run",
            "--workspace",
            str(workspace),
            "--force",
            "--max-jobs",
            "1",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["processed_jobs"] == 1


def test_agent_meta_learning_maintenance_run_operator_alias(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    ws = tmp_path / "workspace"
    pending_path = ws / "memory" / "meta_learning" / "pending_evolution_jobs.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(
        json.dumps(
            {
                "read_file:def456": {
                    "signature": "read_file:def456",
                    "tool_name": "read_file",
                    "reason": "File not found: /tmp/b.txt",
                    "trigger": "failure_threshold",
                    "signature_count": 2,
                    "outcome_score": 0.2,
                    "reward_window_avg": 0.2,
                    "queued_at": "2026-03-20T12:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else ws,
    )

    rc = annolid_run(["agent", "meta-learning", "maintenance", "run", "--force"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["processed_jobs"] == 1


def test_agent_meta_learning_maintenance_status_cli(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    pending_path = (
        workspace / "memory" / "meta_learning" / "pending_evolution_jobs.json"
    )
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(
        json.dumps(
            {
                "sig2": {
                    "signature": "sig2",
                    "tool_name": "read_file",
                    "reason": "x",
                    "queued_at": "2026-03-20T12:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else workspace,
    )
    rc = annolid_run(
        ["agent-meta-learning-maintenance-status", "--workspace", str(workspace)]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pending_jobs_count"] == 1


def test_agent_meta_learning_maintenance_status_operator_alias(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    ws = tmp_path / "workspace"
    pending_path = ws / "memory" / "meta_learning" / "pending_evolution_jobs.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else ws,
    )

    rc = annolid_run(["agent", "meta-learning", "maintenance", "status"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "window_open" in payload


def test_agent_meta_learning_maintenance_next_window_cli(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    pending_path = (
        workspace / "memory" / "meta_learning" / "pending_evolution_jobs.json"
    )
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else workspace,
    )
    rc = annolid_run(
        ["agent-meta-learning-maintenance-next-window", "--workspace", str(workspace)]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "next_window_eta_seconds" in payload


def test_agent_meta_learning_maintenance_next_window_operator_alias(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    ws = tmp_path / "workspace"
    pending_path = ws / "memory" / "meta_learning" / "pending_evolution_jobs.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else ws,
    )

    rc = annolid_run(["agent", "meta-learning", "maintenance", "next-window"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "next_window_eta_seconds" in payload


def test_agent_skills_import_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    source_root = tmp_path / "metaclaw_skills"
    skill_dir = source_root / "debug-systematically"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: debug-systematically\n"
        "description: Structured debugging workflow.\n"
        "category: coding\n"
        "---\n\n"
        "# Debug\n\n"
        "Test body\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else workspace,
    )
    rc = annolid_run(
        [
            "agent-skills-import",
            "--workspace",
            str(workspace),
            "--source-dir",
            str(source_root),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["imported_count"] == 1


def test_agent_skills_import_operator_alias(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod

    ws = tmp_path / "workspace"
    source_root = tmp_path / "metaclaw_skills"
    skill_dir = source_root / "debug-systematically"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: debug-systematically\n"
        "description: Structured debugging workflow.\n"
        "---\n\n"
        "# Debug\n\n"
        "Test body\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        utils_mod,
        "get_agent_workspace_path",
        lambda workspace=None: Path(workspace).expanduser() if workspace else ws,
    )
    rc = annolid_run(
        [
            "agent",
            "skills",
            "import",
            "--workspace",
            str(ws),
            "--source-dir",
            str(source_root),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["imported_count"] == 1


def test_agent_cron_add_list_remove(tmp_path: Path, monkeypatch, capsys) -> None:
    import annolid.services.agent_cron as cron_mod

    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        cron_mod,
        "_default_agent_cron_store_path",
        lambda: data_dir / "cron" / "jobs.json",
    )

    rc_add = annolid_run(
        [
            "agent-cron-add",
            "--name",
            "ping",
            "--message",
            "hello",
            "--every",
            "30",
        ]
    )
    assert rc_add == 0
    add_payload = json.loads(capsys.readouterr().out)
    assert add_payload["name"] == "ping"
    job_id = add_payload["id"]
    assert job_id

    rc_list = annolid_run(["agent-cron-list"])
    assert rc_list == 0
    rows = json.loads(capsys.readouterr().out)
    assert any(r["id"] == job_id for r in rows)

    rc_remove = annolid_run(["agent-cron-remove", job_id])
    assert rc_remove == 0
    remove_payload = json.loads(capsys.readouterr().out)
    assert remove_payload["removed"] is True


def test_agent_cron_add_with_timezone(tmp_path: Path, monkeypatch, capsys) -> None:
    import annolid.services.agent_cron as cron_mod

    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        cron_mod,
        "_default_agent_cron_store_path",
        lambda: data_dir / "cron" / "jobs.json",
    )

    rc_add = annolid_run(
        [
            "agent-cron-add",
            "--name",
            "weekday",
            "--message",
            "daily check",
            "--cron",
            "0 9 * * 1-5",
            "--tz",
            "America/New_York",
        ]
    )
    assert rc_add == 0
    _ = json.loads(capsys.readouterr().out)

    rc_list = annolid_run(["agent-cron-list"])
    assert rc_list == 0
    rows = json.loads(capsys.readouterr().out)
    assert any(
        (row.get("schedule") or {}).get("tz") == "America/New_York" for row in rows
    )


def test_agent_cron_add_accepts_at_with_z_suffix(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.services.agent_cron as cron_mod

    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        cron_mod,
        "_default_agent_cron_store_path",
        lambda: data_dir / "cron" / "jobs.json",
    )

    at_value = (
        (datetime.now(timezone.utc) + timedelta(minutes=5))
        .isoformat()
        .replace("+00:00", "Z")
    )
    rc_add = annolid_run(
        [
            "agent-cron-add",
            "--name",
            "oneshot-z",
            "--message",
            "run once",
            "--at",
            at_value,
        ]
    )
    assert rc_add == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "oneshot-z"


def test_agent_security_check_ok_for_private_clean_settings(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from annolid.core.agent import config as agent_config_mod
    import annolid.core.agent.utils as utils_mod
    from annolid.utils import llm_settings as settings_mod

    data_dir = tmp_path / "data"
    settings_dir = data_dir / "settings"
    settings_file = settings_dir / "llm_settings.json"
    agent_config_path = data_dir / "config.json"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(settings_dir, 0o700)
    settings_file.write_text(
        json.dumps(
            {
                "provider": "ollama",
                "openai": {"base_url": "https://api.openai.com/v1"},
                "gemini": {},
            }
        ),
        encoding="utf-8",
    )
    os.chmod(settings_file, 0o600)
    agent_config_path.write_text(json.dumps({}), encoding="utf-8")

    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
    monkeypatch.setattr(agent_config_mod, "get_config_path", lambda: agent_config_path)
    monkeypatch.setattr(settings_mod, "_SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(settings_mod, "_SETTINGS_FILE", settings_file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    rc = annolid_run(["agent-security-check"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["checks"]["persisted_secrets_found"] is False
    assert payload["checks"]["settings_dir_private"] is True
    assert payload["checks"]["settings_file_private"] is True


def test_agent_security_check_detects_persisted_secrets_and_permissions(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod
    from annolid.core.agent import config as agent_config_mod
    from annolid.utils import llm_settings as settings_mod

    data_dir = tmp_path / "data"
    settings_dir = data_dir / "settings"
    settings_file = settings_dir / "llm_settings.json"
    agent_config_path = data_dir / "config.json"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(settings_dir, 0o755)
    settings_file.write_text(
        json.dumps(
            {
                "provider": "openai",
                "openai": {"api_key": "sk-test"},
                "profiles": {"a": {"token": "plain-secret"}},
            }
        ),
        encoding="utf-8",
    )
    os.chmod(settings_file, 0o644)
    agent_config_path.write_text(
        json.dumps(
            {
                "tools": {
                    "zulip": {
                        "enabled": True,
                        "serverUrl": "https://zulip.example.com",
                        "user": "annolid@example.com",
                        "apiKey": "plain-zulip-secret",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
    monkeypatch.setattr(agent_config_mod, "get_config_path", lambda: agent_config_path)
    monkeypatch.setattr(settings_mod, "_SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(settings_mod, "_SETTINGS_FILE", settings_file)

    rc = annolid_run(["agent-security-check"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "warning"
    assert payload["checks"]["persisted_secrets_found"] is True
    assert payload["checks"]["agent_plaintext_config_secrets_found"] is True
    assert payload["checks"]["settings_dir_private"] is False
    assert payload["checks"]["settings_file_private"] is False
    assert "openai.api_key" in payload["persisted_secret_keys"]
    assert "tools.zulip.apiKey" in payload["agent_plaintext_secret_keys"]


def test_agent_secrets_migrate_and_audit(tmp_path: Path, monkeypatch, capsys) -> None:
    import annolid.core.agent.utils as utils_mod
    from annolid.core.agent import config as agent_config_mod

    data_dir = tmp_path / "data"
    config_path = data_dir / "config.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "tools": {
                    "zulip": {
                        "enabled": True,
                        "serverUrl": "https://zulip.example.com",
                        "user": "annolid@example.com",
                        "apiKey": "zulip-plain",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
    monkeypatch.setattr(agent_config_mod, "get_config_path", lambda: config_path)

    rc_preview = annolid_run(["agent-secrets-migrate"])
    assert rc_preview == 1
    preview = json.loads(capsys.readouterr().out)
    assert preview["candidate_paths"] == ["tools.zulip.apiKey"]

    rc_apply = annolid_run(["agent-secrets-migrate", "--apply"])
    assert rc_apply == 0
    migrated = json.loads(capsys.readouterr().out)
    assert migrated["migrated"] == [
        {
            "path": "tools.zulip.apiKey",
            "local_key": "tools.zulip.apiKey",
        }
    ]

    rc_audit = annolid_run(["agent-secrets-audit"])
    assert rc_audit == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["plaintext_paths"] == []
    assert audit["resolved_ref_paths"] == ["tools.zulip.apiKey"]


def test_agent_config_migrate_updates_legacy_gws(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod
    from annolid.core.agent import config as agent_config_mod

    data_dir = tmp_path / "data"
    config_path = data_dir / "config.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "tools": {
                    "gws": {"enabled": True, "services": ["drive", "calendar"]},
                    "calendar": {
                        "credentialsFile": "~/legacy_creds.json",
                        "tokenFile": "~/legacy_token.json",
                        "allowInteractiveAuth": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
    monkeypatch.setattr(agent_config_mod, "get_config_path", lambda: config_path)

    rc_preview = annolid_run(["agent-config-migrate"])
    assert rc_preview == 1
    preview = json.loads(capsys.readouterr().out)
    assert preview["needs_migration"] is True
    assert preview["has_legacy_google_workspace"] is True
    assert preview["has_legacy_gws"] is True

    rc_apply = annolid_run(["agent-config-migrate", "--apply"])
    assert rc_apply == 0
    migrated = json.loads(capsys.readouterr().out)
    assert migrated["migrated"] is True

    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    tools = persisted.get("tools") or {}
    assert "gws" not in tools
    assert (tools.get("googleDrive") or {}).get("enabled") is True
    assert (tools.get("googleAuth") or {}).get(
        "credentialsFile"
    ) == "~/legacy_creds.json"


def test_agent_security_audit_reports_risky_config(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod
    from annolid.core.agent import config as agent_config_mod

    data_dir = tmp_path / "data"
    config_path = data_dir / "config.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "session": {"dmScope": "main"},
                        "strictRuntimeToolGuard": False,
                    }
                },
                "tools": {
                    "profile": "coding",
                    "allow": ["group:automation"],
                    "zulip": {
                        "enabled": True,
                        "serverUrl": "https://zulip.example.com",
                        "user": "annolid@example.com",
                        "apiKey": "plain-zulip-secret",
                        "allowFrom": [],
                    },
                },
                "update": {"auto": {"enabled": True, "requireSignature": False}},
            }
        ),
        encoding="utf-8",
    )
    os.chmod(data_dir, 0o755)
    os.chmod(config_path, 0o644)

    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
    monkeypatch.setattr(agent_config_mod, "get_config_path", lambda: config_path)
    monkeypatch.setenv("ANNOLID_REQUIRE_SIGNED_SKILLS", "0")
    monkeypatch.setenv("ANNOLID_REQUIRE_SIGNED_UPDATES", "0")

    rc = annolid_run(["agent-security-audit"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    finding_ids = {item["check_id"] for item in payload["findings"]}
    assert "plaintext-config-secrets" in finding_ids
    assert "dm-scope-main" in finding_ids
    assert "channel-allowlist-zulip" in finding_ids
    assert "strict-runtime-tool-guard-disabled" in finding_ids
    assert "runtime-with-messaging-or-automation" in finding_ids
    assert "unsigned-auto-update" in finding_ids


def test_agent_security_audit_fix_repairs_permissions(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod
    from annolid.core.agent import config as agent_config_mod

    data_dir = tmp_path / "data"
    config_path = data_dir / "config.json"
    sessions_dir = data_dir / "sessions"
    data_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({}), encoding="utf-8")
    os.chmod(data_dir, 0o755)
    os.chmod(config_path, 0o644)
    os.chmod(sessions_dir, 0o755)

    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
    monkeypatch.setattr(agent_config_mod, "get_config_path", lambda: config_path)

    rc = annolid_run(["agent-security-audit", "--fix"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert any("chmod 700" in item for item in payload["fixes_applied"])
    assert any("chmod 600" in item for item in payload["fixes_applied"])
    assert oct(config_path.stat().st_mode & 0o777) == "0o600"
    assert oct(data_dir.stat().st_mode & 0o777) == "0o700"
    assert oct(sessions_dir.stat().st_mode & 0o777) == "0o700"


def test_agent_update_check_command(monkeypatch, capsys) -> None:
    import annolid.core.agent.updater as updater_mod
    from annolid.core.agent.updater import UpdateReport

    monkeypatch.setattr(
        updater_mod,
        "check_for_updates",
        lambda channel="stable", timeout_s=4.0, require_signature=False: UpdateReport(
            status="update_available",
            project="annolid",
            channel=channel,
            current_version="1.0.0",
            latest_version="1.1.0",
            update_available=True,
            install_mode="package",
            source_root=".",
            check_url="https://pypi.org/pypi/annolid/json",
            commands=[["python", "-m", "pip", "install", "--upgrade", "annolid"]],
        ),
    )

    rc = annolid_run(["agent-update", "--channel", "stable"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "update_available"
    assert payload["latest_version"] == "1.1.0"


def test_agent_update_apply_dry_run(monkeypatch, capsys) -> None:
    import annolid.core.agent.updater as updater_mod
    from annolid.core.agent.updater import UpdateReport

    report = UpdateReport(
        status="update_available",
        project="annolid",
        channel="stable",
        current_version="1.0.0",
        latest_version="1.1.0",
        update_available=True,
        install_mode="package",
        source_root=".",
        check_url="https://pypi.org/pypi/annolid/json",
        commands=[["python", "-m", "pip", "install", "--upgrade", "annolid"]],
    )
    monkeypatch.setattr(
        updater_mod,
        "check_for_updates",
        lambda channel="stable", timeout_s=4.0, require_signature=False: report,
    )
    monkeypatch.setattr(
        updater_mod,
        "apply_update",
        lambda rep, execute=False, run_doctor=True: {
            **rep.to_dict(),
            "executed": bool(execute),
            "doctor": bool(run_doctor),
            "updated": False,
            "steps": [],
        },
    )

    rc = annolid_run(["agent-update", "--apply"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["updated"] is False


def test_agent_update_require_signature_flag(monkeypatch, capsys) -> None:
    import annolid.core.agent.updater as updater_mod
    from annolid.core.agent.updater import UpdateReport

    observed = {"require_signature": None}

    def _fake_check(
        channel="stable",
        timeout_s=4.0,
        require_signature=False,
    ):
        del timeout_s
        observed["require_signature"] = bool(require_signature)
        return UpdateReport(
            status="ok",
            project="annolid",
            channel=channel,
            current_version="1.0.0",
            latest_version="1.0.0",
            update_available=False,
            install_mode="package",
            source_root=".",
            check_url="https://pypi.org/pypi/annolid/json",
            commands=[["python", "-m", "pip", "install", "--upgrade", "annolid"]],
            require_signature=bool(require_signature),
        )

    monkeypatch.setattr(updater_mod, "check_for_updates", _fake_check)
    rc = annolid_run(["agent-update", "--require-signature"])
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)
    assert observed["require_signature"] is True


def test_agent_skills_refresh_operator_command(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    skill_dir = workspace / "skills" / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: demo skill\n---\ndemo content\n",
        encoding="utf-8",
    )
    rc = annolid_run(["agent", "skills", "refresh", "--workspace", str(workspace)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["refreshed"] is True
    assert payload["count"] >= 1
    assert "demo" in payload["names"]


def test_agent_skills_inspect_reports_invalid_manifest(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    good = workspace / "skills" / "good"
    bad = workspace / "skills" / "bad"
    good.mkdir(parents=True, exist_ok=True)
    bad.mkdir(parents=True, exist_ok=True)
    (good / "SKILL.md").write_text(
        "---\ndescription: good skill\n---\nok\n",
        encoding="utf-8",
    )
    (bad / "SKILL.md").write_text(
        '---\ndescription: "bad skill"\nalways: "sometimes"\n---\nbad\n',
        encoding="utf-8",
    )
    rc = annolid_run(["agent", "skills", "inspect", "--workspace", str(workspace)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace_skill_count"] >= 2
    assert payload["invalid_manifest_count"] == 1
    assert payload["invalid_skills"][0]["name"] == "bad"


def test_agent_skills_shadow_operator_command(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    active = workspace / "skills" / "demo"
    active.mkdir(parents=True, exist_ok=True)
    (active / "SKILL.md").write_text("---\ndescription: demo\n---\n", encoding="utf-8")
    candidate_pack = tmp_path / "candidate_pack"
    candidate_demo = candidate_pack / "demo"
    candidate_new = candidate_pack / "new_skill"
    candidate_demo.mkdir(parents=True, exist_ok=True)
    candidate_new.mkdir(parents=True, exist_ok=True)
    (candidate_demo / "SKILL.md").write_text(
        "---\ndescription: demo v2\n---\n", encoding="utf-8"
    )
    (candidate_new / "SKILL.md").write_text(
        "---\ndescription: new\n---\n", encoding="utf-8"
    )
    rc = annolid_run(
        [
            "agent",
            "skills",
            "shadow",
            "--workspace",
            str(workspace),
            "--candidate-pack",
            str(candidate_pack),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "new_skill" in payload["added"]
    assert "demo" in payload["overridden"]


def test_agent_memory_flush_operator_command(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
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
    assert "checkpoint" in payload["entry"]
    assert Path(payload["today_file"]).exists()
    assert Path(payload["history_file"]).exists()


def test_agent_memory_inspect_operator_command(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    rc = annolid_run(["agent", "memory", "inspect", "--workspace", str(workspace)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workspace"] == str(workspace)
    assert payload["retrieval_plugin"] == "workspace_semantic_keyword_v1"
    assert payload["long_term_file"].endswith("memory/MEMORY.md")
    assert payload["history_file"].endswith("memory/HISTORY.md")


def test_agent_eval_run_operator_command(tmp_path: Path, capsys) -> None:
    traces = tmp_path / "traces.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    out = tmp_path / "report.json"
    traces.write_text(
        '{"trace_id":"t1","user_message":"u","expected_substring":"ok"}\n',
        encoding="utf-8",
    )
    baseline.write_text('{"trace_id":"t1","content":"ok"}\n', encoding="utf-8")
    candidate.write_text('{"trace_id":"t1","content":"bad"}\n', encoding="utf-8")

    rc = annolid_run(
        [
            "agent",
            "eval",
            "run",
            "--traces",
            str(traces),
            "--baseline-responses",
            str(baseline),
            "--candidate-responses",
            str(candidate),
            "--out",
            str(out),
            "--max-regressions",
            "0",
        ]
    )
    assert rc == 1
    _payload = json.loads(capsys.readouterr().out)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["regression_gate"]["passed"] is False


def test_agent_feedback_add_and_build_regression_dataset(
    tmp_path: Path, capsys
) -> None:
    workspace = tmp_path / "workspace"
    # Create trace by running a flush/inspect cycle and explicit feedback.
    rc_feedback = annolid_run(
        [
            "agent",
            "feedback",
            "add",
            "--workspace",
            str(workspace),
            "--session-id",
            "s1",
            "--trace-id",
            "trace-manual",
            "--rating",
            "1",
            "--comment",
            "good",
            "--expected-substring",
            "ok",
        ]
    )
    assert rc_feedback == 0
    payload_feedback = json.loads(capsys.readouterr().out)
    assert payload_feedback["saved"] is True

    # write synthetic trace directly to enable dataset builder row.
    trace_path = workspace / "eval" / "run_traces.ndjson"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "trace_id": "trace-manual",
                "user_message_preview": "u",
                "assistant_response_preview": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "regression.jsonl"
    rc_build = annolid_run(
        [
            "agent",
            "eval",
            "build-regression",
            "--workspace",
            str(workspace),
            "--out",
            str(out),
        ]
    )
    assert rc_build == 0
    payload_build = json.loads(capsys.readouterr().out)
    assert payload_build["count"] >= 1
    assert out.exists()


def test_agent_eval_gate_operator_command(tmp_path: Path, capsys) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text("annolid/core/agent/skills.py\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"regressions": [], "candidate": {"pass_rate": 1.0}}),
        encoding="utf-8",
    )
    rc = annolid_run(
        [
            "agent",
            "eval",
            "gate",
            "--changed-files",
            str(changed),
            "--report",
            str(report),
            "--max-regressions",
            "0",
            "--min-pass-rate",
            "0.8",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["required"] is True
    assert payload["gate"]["passed"] is True


def test_update_check_run_rollback_operator_commands(monkeypatch, capsys) -> None:
    import annolid.engine.cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_cmd_update_check",
        lambda args: (
            print(json.dumps({"action": "check", "channel": args.channel})),
            0,
        )[1],
    )
    monkeypatch.setattr(
        cli_mod,
        "_cmd_update_run",
        lambda args: (
            print(
                json.dumps(
                    {
                        "action": "run",
                        "execute": bool(args.execute),
                        "canary_metrics": args.canary_metrics,
                    }
                )
            ),
            0,
        )[1],
    )
    monkeypatch.setattr(
        cli_mod,
        "_cmd_update_rollback",
        lambda args: (
            print(
                json.dumps(
                    {"action": "rollback", "previous_version": args.previous_version}
                )
            ),
            0,
        )[1],
    )

    rc_check = annolid_run(["update", "check", "--channel", "stable"])
    assert rc_check == 0
    payload_check = json.loads(capsys.readouterr().out)
    assert payload_check["action"] == "check"

    rc_run = annolid_run(
        ["update", "run", "--execute", "--canary-metrics", "/tmp/metrics.json"]
    )
    assert rc_run == 0
    payload_run = json.loads(capsys.readouterr().out)
    assert payload_run["action"] == "run"
    assert payload_run["execute"] is True
    assert payload_run["canary_metrics"] == "/tmp/metrics.json"

    rc_rb = annolid_run(["update", "rollback", "--previous-version", "1.0.0"])
    assert rc_rb == 0
    payload_rb = json.loads(capsys.readouterr().out)
    assert payload_rb["action"] == "rollback"
    assert payload_rb["previous_version"] == "1.0.0"


def test_agent_box_auth_url_cli(monkeypatch, capsys) -> None:
    import annolid.services.agent_box as box_mod

    def _fake_get_box_oauth_authorize_url(**kwargs):
        assert kwargs["authorize_base_url"] == "https://my_org_xxx.account.box.com"
        return {
            "ok": True,
            "authorize_url": (
                "https://my_org_xxx.account.box.com/api/oauth2/authorize?client_id=cid"
            ),
        }

    monkeypatch.setattr(
        box_mod, "get_box_oauth_authorize_url", _fake_get_box_oauth_authorize_url
    )
    rc = annolid_run(
        [
            "agent-box-auth-url",
            "--authorize-base-url",
            "https://my_org_xxx.account.box.com",
            "--redirect-uri",
            "https://example.com/callback",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "authorize_url" in payload


def test_agent_box_auth_url_cli_open_browser(monkeypatch, capsys) -> None:
    import annolid.services.agent_box as box_mod
    import webbrowser

    opened: list[str] = []

    def _fake_get_box_oauth_authorize_url(**kwargs):
        return {
            "ok": True,
            "authorize_url": "https://my_org_xxx.account.box.com/api/oauth2/authorize?client_id=cid",
        }

    def _fake_open_new_tab(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(
        box_mod, "get_box_oauth_authorize_url", _fake_get_box_oauth_authorize_url
    )
    monkeypatch.setattr(webbrowser, "open_new_tab", _fake_open_new_tab)
    rc = annolid_run(
        [
            "agent-box-auth-url",
            "--open-browser",
            "--redirect-uri",
            "https://example.com/callback",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["opened_in_browser"] is True
    assert opened


def test_agent_box_auth_url_cli_uses_config_redirect_uri(
    tmp_path: Path, capsys
) -> None:
    from annolid.core.agent.config import AgentConfig, save_config

    cfg = AgentConfig()
    cfg.tools.box.client_id = "cid"
    cfg.tools.box.redirect_uri = "https://localhost:8765/oauth/callback"
    cfg.tools.box.authorize_base_url = "https://my_org_xxx.account.box.com"
    cfg_path = tmp_path / "config.json"
    save_config(cfg, cfg_path)

    rc = annolid_run(["agent-box-auth-url", "--config", str(cfg_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["redirect_uri"] == "https://localhost:8765/oauth/callback"
    assert payload["authorize_url"].startswith(
        "https://my_org_xxx.account.box.com/api/oauth2/authorize?"
    )


def test_agent_box_auth_exchange_cli(monkeypatch, capsys) -> None:
    import annolid.services.agent_box as box_mod

    def _fake_exchange_box_oauth_code(**kwargs):
        assert kwargs["authorize_base_url"] == "https://my_org_xxx.account.box.com"
        return ({"ok": True, "persisted": False, "access_token": "a1"}, 0)

    monkeypatch.setattr(
        box_mod, "exchange_box_oauth_code", _fake_exchange_box_oauth_code
    )
    rc = annolid_run(
        [
            "agent-box-auth-exchange",
            "--authorize-base-url",
            "https://my_org_xxx.account.box.com",
            "--redirect-uri",
            "https://example.com/callback",
            "--code",
            "abc123",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["access_token"] == "a1"


def test_agent_box_token_refresh_cli(monkeypatch, capsys) -> None:
    import annolid.services.agent_box as box_mod

    def _fake_refresh_box_oauth_token(**kwargs):
        return ({"ok": True, "persisted": True, "access_token": "new-a1"}, 0)

    monkeypatch.setattr(
        box_mod, "refresh_box_oauth_token", _fake_refresh_box_oauth_token
    )
    rc = annolid_run(["agent-box-token-refresh", "--persist"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["persisted"] is True


def test_llm_settings_dialog_box_auth_uses_browser_flow(monkeypatch) -> None:
    import annolid.gui.widgets.llm_settings_dialog as dialog_mod

    calls: dict[str, object] = {}

    def _fake_complete_box_oauth_browser_flow(**kwargs):
        calls.update(kwargs)
        return ({"ok": True, "persisted": True, "access_token": "token-a"}, 0)

    class _FakeThread:
        def __init__(self, target=None, name=None, daemon=None):  # noqa: D401
            self._target = target
            self.name = name
            self.daemon = daemon

        def start(self) -> None:
            if self._target is not None:
                self._target()

    class _FakeEdit:
        def __init__(self, value: str) -> None:
            self._value = value

        def text(self) -> str:
            return self._value

        def setText(self, value: str) -> None:
            self._value = value

    class _FakeButton:
        def __init__(self) -> None:
            self.enabled = True
            self.text_value = "Grant Box Access"

        def setEnabled(self, value: bool) -> None:
            self.enabled = bool(value)

        def setText(self, value: str) -> None:
            self.text_value = value

    monkeypatch.setattr(
        "annolid.services.agent_box.complete_box_oauth_browser_flow",
        _fake_complete_box_oauth_browser_flow,
    )
    monkeypatch.setattr(dialog_mod.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        dialog_mod.QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        dialog_mod.QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(dialog_mod, "load_config", lambda: None)

    fake_dialog = type(
        "FakeDialog",
        (),
        {
            "box_authorize_base_url_edit": _FakeEdit(
                "https://my_org_xxx.account.box.com"
            ),
            "box_client_id_edit": _FakeEdit("cid"),
            "box_client_secret_edit": _FakeEdit("sec"),
            "box_redirect_uri_edit": _FakeEdit("http://localhost:8765/oauth/callback"),
            "box_auth_button": _FakeButton(),
            "_agent_config": None,
            "_on_box_auth_completed": dialog_mod.LLMSettingsDialog._on_box_auth_completed,
            "boxAuthCompleted": type(
                "_FakeSignal",
                (),
                {
                    "emit": lambda self,
                    payload: dialog_mod.LLMSettingsDialog._on_box_auth_completed(
                        fake_dialog, payload
                    )
                },
            )(),
        },
    )()

    dialog_mod.LLMSettingsDialog._open_box_auth_url(fake_dialog)

    assert calls["client_id"] == "cid"
    assert calls["client_secret"] == "sec"
    assert calls["redirect_uri"] == "http://localhost:8765/oauth/callback"
    assert calls["authorize_base_url"] == "https://my_org_xxx.account.box.com"
    assert calls["persist"] is True
    assert calls["open_browser"] is True
    assert fake_dialog.box_auth_button.enabled is True
    assert fake_dialog.box_auth_button.text_value == "Grant Box Access"
