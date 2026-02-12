from __future__ import annotations

import json
from pathlib import Path

from annolid.core.agent.config import (
    AgentConfig,
    ProviderConfig,
    load_config,
    save_config,
)


def test_agent_config_load_save_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg = AgentConfig()
    cfg.agents.defaults.workspace = str(tmp_path / "workspace")
    cfg.agents.defaults.model = "gemini-1.5-flash"
    cfg.providers["gemini"] = ProviderConfig(api_key="secret", api_base="")
    cfg.tools.restrict_to_workspace = True

    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)

    assert loaded.agents.defaults.workspace == str(tmp_path / "workspace")
    assert loaded.agents.defaults.model == "gemini-1.5-flash"
    assert loaded.providers["gemini"].api_key == "secret"
    assert loaded.tools.restrict_to_workspace is True


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
