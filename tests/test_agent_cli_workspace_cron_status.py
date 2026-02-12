from __future__ import annotations

import json
import os
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


def test_agent_status_uses_patched_paths(tmp_path: Path, monkeypatch, capsys) -> None:
    import annolid.engine.cli as cli_mod
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
        cli_mod,
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


def test_agent_cron_add_list_remove(tmp_path: Path, monkeypatch, capsys) -> None:
    import annolid.engine.cli as cli_mod

    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        cli_mod,
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


def test_agent_security_check_ok_for_private_clean_settings(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import annolid.core.agent.utils as utils_mod
    from annolid.utils import llm_settings as settings_mod

    data_dir = tmp_path / "data"
    settings_dir = data_dir / "settings"
    settings_file = settings_dir / "llm_settings.json"
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

    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
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
    from annolid.utils import llm_settings as settings_mod

    data_dir = tmp_path / "data"
    settings_dir = data_dir / "settings"
    settings_file = settings_dir / "llm_settings.json"
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

    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
    monkeypatch.setattr(settings_mod, "_SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(settings_mod, "_SETTINGS_FILE", settings_file)

    rc = annolid_run(["agent-security-check"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "warning"
    assert payload["checks"]["persisted_secrets_found"] is True
    assert payload["checks"]["settings_dir_private"] is False
    assert payload["checks"]["settings_file_private"] is False
    assert "openai.api_key" in payload["persisted_secret_keys"]
