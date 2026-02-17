from __future__ import annotations

import json
from pathlib import Path

from annolid.core.agent.config import (
    AgentConfig,
    ProviderConfig,
    SessionRoutingConfig,
    ToolPolicyConfig,
    load_config,
    save_config,
)


def test_agent_config_load_creates_default_template(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    loaded = load_config(cfg_path)
    assert cfg_path.exists()
    assert loaded.tools.calendar.enabled is False
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    tools = payload.get("tools") or {}
    calendar = tools.get("calendar") or {}
    assert "enabled" in calendar
    assert "provider" in calendar


def test_agent_config_load_save_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg = AgentConfig()
    cfg.agents.defaults.workspace = str(tmp_path / "workspace")
    cfg.agents.defaults.model = "gemini-1.5-flash"
    cfg.agents.defaults.session = SessionRoutingConfig(
        dm_scope="per-account-channel-peer",
        main_session_key="main",
    )
    cfg.providers["gemini"] = ProviderConfig(api_key="secret", api_base="")
    cfg.tools.restrict_to_workspace = True
    cfg.tools.allowed_read_roots = [
        str(tmp_path / "videos"),
        str(tmp_path / "datasets"),
    ]
    cfg.tools.profile = "coding"
    cfg.tools.allow = ["group:ui", "web_search"]
    cfg.tools.deny = ["exec"]
    cfg.tools.by_provider["ollama:glm-5:cloud"] = ToolPolicyConfig(
        profile="minimal",
        allow=["gui_open_video"],
        deny=["gui_set_chat_model"],
    )
    cfg.tools.whatsapp.enabled = True
    cfg.tools.whatsapp.auto_start = False
    cfg.tools.whatsapp.bridge_mode = "python"
    cfg.tools.whatsapp.bridge_url = "ws://127.0.0.1:3001"
    cfg.tools.whatsapp.bridge_host = "127.0.0.1"
    cfg.tools.whatsapp.bridge_port = 3001
    cfg.tools.whatsapp.bridge_token = "bridge-secret"
    cfg.tools.whatsapp.bridge_session_dir = "~/.annolid/whatsapp-web-session"
    cfg.tools.whatsapp.bridge_headless = False
    cfg.tools.whatsapp.phone_number_id = "123456"
    cfg.tools.whatsapp.verify_token = "verify"
    cfg.tools.whatsapp.preview_url = True
    cfg.tools.whatsapp.webhook_enabled = True
    cfg.tools.whatsapp.webhook_host = "127.0.0.1"
    cfg.tools.whatsapp.webhook_port = 18081
    cfg.tools.whatsapp.webhook_path = "/whatsapp/webhook"
    cfg.tools.whatsapp.ingest_outgoing_messages = True
    cfg.tools.calendar.enabled = True
    cfg.tools.calendar.provider = "google"
    cfg.tools.calendar.credentials_file = "~/calendar_credentials.json"
    cfg.tools.calendar.token_file = "~/calendar_token.json"
    cfg.tools.calendar.calendar_id = "primary"
    cfg.tools.calendar.timezone = "America/Los_Angeles"
    cfg.tools.calendar.default_event_duration_minutes = 45

    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)

    assert loaded.agents.defaults.workspace == str(tmp_path / "workspace")
    assert loaded.agents.defaults.model == "gemini-1.5-flash"
    assert loaded.agents.defaults.session.dm_scope == "per-account-channel-peer"
    assert loaded.agents.defaults.session.main_session_key == "main"
    assert loaded.providers["gemini"].api_key == "secret"
    assert loaded.tools.restrict_to_workspace is True
    assert loaded.tools.allowed_read_roots == [
        str(tmp_path / "videos"),
        str(tmp_path / "datasets"),
    ]
    assert loaded.tools.profile == "coding"
    assert loaded.tools.allow == ["group:ui", "web_search"]
    assert loaded.tools.deny == ["exec"]
    assert "ollama:glm-5:cloud" in loaded.tools.by_provider
    assert loaded.tools.by_provider["ollama:glm-5:cloud"].profile == "minimal"
    assert loaded.tools.whatsapp.enabled is True
    assert loaded.tools.whatsapp.auto_start is False
    assert loaded.tools.whatsapp.bridge_mode == "python"
    assert loaded.tools.whatsapp.bridge_url == "ws://127.0.0.1:3001"
    assert loaded.tools.whatsapp.bridge_host == "127.0.0.1"
    assert loaded.tools.whatsapp.bridge_port == 3001
    assert loaded.tools.whatsapp.bridge_token == "bridge-secret"
    assert loaded.tools.whatsapp.bridge_session_dir == "~/.annolid/whatsapp-web-session"
    assert loaded.tools.whatsapp.bridge_headless is False
    assert loaded.tools.whatsapp.phone_number_id == "123456"
    assert loaded.tools.whatsapp.verify_token == "verify"
    assert loaded.tools.whatsapp.preview_url is True
    assert loaded.tools.whatsapp.webhook_enabled is True
    assert loaded.tools.whatsapp.webhook_host == "127.0.0.1"
    assert loaded.tools.whatsapp.webhook_port == 18081
    assert loaded.tools.whatsapp.webhook_path == "/whatsapp/webhook"
    assert loaded.tools.whatsapp.ingest_outgoing_messages is True
    assert loaded.tools.calendar.enabled is True
    assert loaded.tools.calendar.provider == "google"
    assert loaded.tools.calendar.credentials_file == "~/calendar_credentials.json"
    assert loaded.tools.calendar.token_file == "~/calendar_token.json"
    assert loaded.tools.calendar.calendar_id == "primary"
    assert loaded.tools.calendar.timezone == "America/Los_Angeles"
    assert loaded.tools.calendar.default_event_duration_minutes == 45


def test_agent_config_migrates_legacy_restrict_to_workspace(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    legacy_payload = {
        "tools": {
            "exec": {"timeout": 10, "restrictToWorkspace": True},
        }
    }
    cfg_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    loaded = load_config(cfg_path)
    assert loaded.tools.exec.timeout == 10
    assert loaded.tools.restrict_to_workspace is True


def test_agent_config_loads_legacy_session_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    payload = {
        "agents": {
            "defaults": {
                "sessionDmScope": "per-peer",
                "mainSessionKey": "main",
            }
        }
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_config(cfg_path)
    assert loaded.agents.defaults.session.dm_scope == "per-peer"
    assert loaded.agents.defaults.session.main_session_key == "main"
