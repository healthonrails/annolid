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
    assert (workspace / "TOOLS.md").exists()
    assert (workspace / "HEARTBEAT.md").exists()
    assert (workspace / "memory" / "MEMORY.md").exists()
    assert (workspace / "memory" / "HISTORY.md").exists()


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
