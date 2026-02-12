from __future__ import annotations

import json
from pathlib import Path


def test_load_llm_settings_migrates_camel_case_and_legacy_agent_keys(
    tmp_path: Path, monkeypatch
) -> None:
    from annolid.utils import llm_settings as mod

    monkeypatch.setattr(mod, "_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(mod, "_SETTINGS_FILE", tmp_path / "llm_settings.json")

    raw = {
        "provider": "openai",
        "lastModels": {"openai": "gpt-4.1-mini"},
        "openai": {
            "baseUrl": "https://api.openai.com/v1",
            "preferredModels": "gpt-4o-mini",
        },
        "agent": {"maxIterations": 9},
        "profiles": {"playground": {"maxIterations": 6}},
    }
    (tmp_path / "llm_settings.json").write_text(json.dumps(raw), encoding="utf-8")

    settings = mod.load_llm_settings()
    assert settings["last_models"]["openai"] == "gpt-4.1-mini"
    assert settings["openai"]["preferred_models"] == ["gpt-4o-mini"]
    assert settings["agent"]["max_tool_iterations"] == 9
    assert settings["profiles"]["playground"]["max_tool_iterations"] == 6


def test_resolve_agent_runtime_config_prefers_profile_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    from annolid.utils import llm_settings as mod

    monkeypatch.setattr(mod, "_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(mod, "_SETTINGS_FILE", tmp_path / "llm_settings.json")

    mod.save_llm_settings(
        {
            "agent": {
                "temperature": 0.3,
                "max_tool_iterations": 15,
                "max_history_messages": 50,
                "memory_window": 80,
            },
            "profiles": {
                "analysis_agent": {
                    "temperature": 0.4,
                    "max_tool_iterations": 7,
                    "max_history_messages": 30,
                    "memory_window": 20,
                }
            },
        }
    )

    cfg = mod.resolve_agent_runtime_config(profile="analysis_agent")
    assert cfg.temperature == 0.4
    assert cfg.max_tool_iterations == 7
    assert cfg.max_history_messages == 30
    assert cfg.memory_window == 20

    global_cfg = mod.resolve_agent_runtime_config()
    assert global_cfg.temperature == 0.3
    assert global_cfg.max_tool_iterations == 15
    assert global_cfg.max_history_messages == 50
    assert global_cfg.memory_window == 80


def test_save_llm_settings_scrubs_nested_secrets(tmp_path: Path, monkeypatch) -> None:
    from annolid.utils import llm_settings as mod

    monkeypatch.setattr(mod, "_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(mod, "_SETTINGS_FILE", tmp_path / "llm_settings.json")

    settings = {
        "openai": {"api_key": "sk-xxx", "base_url": "https://api.openai.com/v1"},
        "gemini": {"api_key": "gm-xxx"},
        "profiles": {
            "custom": {"provider": "openai", "token": "secret-token", "model": "gpt-4o"}
        },
        "nested": {"child": {"access_token": "abc123", "keep": "ok"}},
    }
    mod.save_llm_settings(settings)

    persisted = json.loads((tmp_path / "llm_settings.json").read_text(encoding="utf-8"))
    assert "api_key" not in persisted.get("openai", {})
    assert "api_key" not in persisted.get("gemini", {})
    assert "token" not in persisted.get("profiles", {}).get("custom", {})
    assert "access_token" not in persisted.get("nested", {}).get("child", {})
    assert persisted.get("nested", {}).get("child", {}).get("keep") == "ok"


def test_has_provider_api_key_supports_env(monkeypatch) -> None:
    from annolid.utils import llm_settings as mod

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert mod.has_provider_api_key({"openai": {}}, "openai") is False
    assert mod.has_provider_api_key({"gemini": {}}, "gemini") is False

    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    monkeypatch.setenv("GOOGLE_API_KEY", "gm-live")
    assert mod.has_provider_api_key({"openai": {}}, "openai") is True
    assert mod.has_provider_api_key({"gemini": {}}, "gemini") is True

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-live")
    assert (
        mod.has_provider_api_key(
            {"openai": {"base_url": "https://openrouter.ai/api/v1"}}, "openai"
        )
        is True
    )
    assert mod.has_provider_api_key({"openrouter": {}}, "openrouter") is True


def test_save_llm_settings_applies_secure_permissions(
    tmp_path: Path, monkeypatch
) -> None:
    from annolid.utils import llm_settings as mod

    monkeypatch.setattr(mod, "_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(mod, "_SETTINGS_FILE", tmp_path / "llm_settings.json")
    mod.save_llm_settings({"provider": "ollama"})

    settings_mode = (tmp_path / "llm_settings.json").stat().st_mode & 0o777
    dir_mode = tmp_path.stat().st_mode & 0o777
    assert settings_mode == 0o600
    assert dir_mode == 0o700
