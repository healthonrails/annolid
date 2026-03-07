from __future__ import annotations

import json
from pathlib import Path

from annolid.services.agent_admin import (
    run_agent_security_check,
    set_agent_secret,
)


def test_run_agent_security_check_reports_warning(monkeypatch, tmp_path: Path) -> None:
    import annolid.core.agent.config as config_mod
    import annolid.core.agent.config.secrets as secrets_mod
    import annolid.core.agent.utils as utils_mod
    import annolid.utils.llm_settings as llm_mod

    data_dir = tmp_path / "agent-data"
    data_dir.mkdir()
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    settings_path = settings_dir / "llm_settings.json"
    settings_path.write_text(
        json.dumps({"providers": {"openai": {"api_key": "sk-test"}}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "agent.json"
    config_path.write_text("{}", encoding="utf-8")
    secret_store_path = tmp_path / "secrets.json"
    secret_store_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
    monkeypatch.setattr(llm_mod, "settings_path", lambda: settings_path)
    monkeypatch.setattr(
        llm_mod,
        "has_provider_api_key",
        lambda settings, provider: bool(
            (settings.get("providers") or {}).get(provider, {}).get("api_key")
        ),
    )
    monkeypatch.setattr(config_mod, "get_config_path", lambda: config_path)
    monkeypatch.setattr(
        secrets_mod,
        "read_raw_agent_config",
        lambda path: {"providers": {"openai": {"api_key": "plain-secret"}}},
    )
    monkeypatch.setattr(
        secrets_mod,
        "get_secret_store_path",
        lambda: secret_store_path,
    )
    monkeypatch.setattr(secrets_mod, "load_secret_store", lambda path: {})
    monkeypatch.setattr(
        secrets_mod,
        "inspect_secret_posture",
        lambda payload, refs, store: {
            "plaintext_paths": ["providers.openai.api_key"],
            "shadowed_plaintext_paths": [],
            "unresolved_ref_paths": [],
            "resolved_ref_paths": [],
        },
    )

    class _SecretsConfig:
        refs = {}

        @classmethod
        def from_dict(cls, _value):
            return cls()

    monkeypatch.setattr(secrets_mod, "SecretsConfig", _SecretsConfig)

    payload = run_agent_security_check()

    assert payload["status"] == "warning"
    assert payload["checks"]["persisted_secrets_found"] is True
    assert payload["checks"]["agent_plaintext_config_secrets_found"] is True
    assert payload["provider_key_presence"]["openai"] is True


def test_set_agent_secret_local_updates_store(monkeypatch, tmp_path: Path) -> None:
    import annolid.core.agent.config as config_mod
    import annolid.core.agent.config.secrets as secrets_mod

    config_path = tmp_path / "agent.json"
    config_path.write_text("{}", encoding="utf-8")
    store_path = tmp_path / "secret-store.json"
    captured: dict[str, object] = {}

    monkeypatch.setattr(config_mod, "get_config_path", lambda: config_path)
    monkeypatch.setattr(secrets_mod, "read_raw_agent_config", lambda path: {})
    monkeypatch.setattr(secrets_mod, "get_secret_store_path", lambda: store_path)
    monkeypatch.setattr(secrets_mod, "load_secret_store", lambda path: {})

    def _save_secret_store(secrets, path):
        captured["store"] = dict(secrets)
        captured["store_path"] = path

    monkeypatch.setattr(secrets_mod, "save_secret_store", _save_secret_store)

    class _SecretRefConfig:
        def __init__(self, *, source: str, name: str) -> None:
            self.source = source
            self.name = name

        def to_dict(self) -> dict[str, str]:
            return {"source": self.source, "name": self.name}

    monkeypatch.setattr(secrets_mod, "SecretRefConfig", _SecretRefConfig)

    def _apply_secret_ref(raw_payload, *, path: str, ref) -> dict:
        captured["apply_path"] = path
        captured["ref"] = ref.to_dict()
        return {"secrets": {path: ref.to_dict()}}

    monkeypatch.setattr(secrets_mod, "apply_secret_ref", _apply_secret_ref)

    payload = set_agent_secret(
        path="providers.openai.api_key",
        local="openai_api_key",
        value="sk-local",
    )

    written = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["updated"] is True
    assert payload["ref"] == {"source": "local", "name": "openai_api_key"}
    assert captured["store"] == {"openai_api_key": "sk-local"}
    assert captured["apply_path"] == "providers.openai.api_key"
    assert written["secrets"]["providers.openai.api_key"]["source"] == "local"
