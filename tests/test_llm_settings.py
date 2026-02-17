from __future__ import annotations

import json
import os
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


def test_default_settings_include_agent_runtime_timeout_keys() -> None:
    from annolid.utils import llm_settings as mod

    settings = mod.default_settings()
    agent = dict(settings.get("agent") or {})
    assert agent.get("fast_mode_timeout_seconds") == 60
    assert agent.get("fallback_retry_timeout_seconds") == 20
    assert agent.get("loop_llm_timeout_seconds") == 60
    assert agent.get("loop_llm_timeout_seconds_no_tools") == 40
    assert agent.get("ollama_tool_timeout_seconds") == 45
    assert agent.get("ollama_plain_timeout_seconds") == 25
    assert agent.get("ollama_plain_recovery_timeout_seconds") == 12
    assert agent.get("ollama_plain_recovery_nudge_timeout_seconds") == 8


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


def test_load_llm_settings_reads_dotenv_without_overriding_parent_env(
    tmp_path: Path, monkeypatch
) -> None:
    from annolid.utils import llm_settings as mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(mod, "_SETTINGS_DIR", tmp_path / ".annolid")
    monkeypatch.setattr(
        mod, "_SETTINGS_FILE", tmp_path / ".annolid" / "llm_settings.json"
    )
    monkeypatch.setattr(mod, "_GLOBAL_DOTENV_FILE", tmp_path / ".annolid" / ".env")
    (workspace / ".env").write_text(
        "OPENAI_API_KEY=workspace-key\nNVIDIA_API_KEY=nvidia-key\n",
        encoding="utf-8",
    )
    (tmp_path / ".annolid").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".annolid" / ".env").write_text(
        "OPENAI_API_KEY=global-key\nGEMINI_API_KEY=global-gemini\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "parent-key")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    mod.load_llm_settings()

    assert os.getenv("OPENAI_API_KEY") == "parent-key"
    assert os.getenv("NVIDIA_API_KEY") == "nvidia-key"
    assert os.getenv("GEMINI_API_KEY") == "global-gemini"


def test_load_llm_settings_applies_inline_env_when_unset(
    tmp_path: Path, monkeypatch
) -> None:
    from annolid.utils import llm_settings as mod

    monkeypatch.setattr(mod, "_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(mod, "_SETTINGS_FILE", tmp_path / "llm_settings.json")
    monkeypatch.setattr(mod, "_GLOBAL_DOTENV_FILE", tmp_path / ".env")
    (tmp_path / "llm_settings.json").write_text(
        json.dumps(
            {
                "env": {
                    "OPENROUTER_API_KEY": "inline-openrouter",
                    "vars": {"NVIDIA_API_KEY": "inline-nvidia"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    mod.load_llm_settings()
    assert os.getenv("OPENROUTER_API_KEY") == "inline-openrouter"
    assert os.getenv("NVIDIA_API_KEY") == "inline-nvidia"


def test_persist_global_env_vars_updates_global_dotenv(
    tmp_path: Path, monkeypatch
) -> None:
    from annolid.utils import llm_settings as mod

    monkeypatch.setattr(mod, "_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(mod, "_SETTINGS_FILE", tmp_path / "llm_settings.json")
    monkeypatch.setattr(mod, "_GLOBAL_DOTENV_FILE", tmp_path / ".env")
    (tmp_path / ".env").write_text("EXISTING_KEY=1\n", encoding="utf-8")

    mod.persist_global_env_vars(
        {
            "OPENAI_API_KEY": "sk-test-123",
            "NVIDIA_API_KEY": "nv key with space",
            "EMPTY_SKIP": "",
        }
    )

    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "EXISTING_KEY=1" in content
    assert "OPENAI_API_KEY=sk-test-123" in content
    assert 'NVIDIA_API_KEY="nv key with space"' in content
    assert "EMPTY_SKIP" not in content
    mode = (tmp_path / ".env").stat().st_mode & 0o777
    assert mode == 0o600


def test_save_llm_settings_preserves_provider_api_key_env_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    from annolid.utils import llm_settings as mod

    monkeypatch.setattr(mod, "_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(mod, "_SETTINGS_FILE", tmp_path / "llm_settings.json")
    monkeypatch.setattr(mod, "_GLOBAL_DOTENV_FILE", tmp_path / ".env")

    mod.save_llm_settings(
        {
            "provider_definitions": {
                "nvidia": {
                    "label": "Nvidia",
                    "kind": "openai_compat",
                    "env_keys": ["NVIDIA_API_KEY"],
                    "api_key_env": ["NVIDIA_API_KEY"],
                    "base_url_default": "https://integrate.api.nvidia.com/v1",
                }
            },
            "nvidia": {"preferred_models": ["moonshotai/kimi-k2.5"]},
        }
    )

    settings = mod.load_llm_settings()
    provider_defs = settings.get("provider_definitions", {})
    nvidia = provider_defs.get("nvidia", {})
    assert nvidia.get("api_key_env") == ["NVIDIA_API_KEY"]
