from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from annolid.core.agent.utils import get_agent_data_path

from .schema import AgentConfig
from .secrets import (
    SecretsConfig,
    load_secret_store,
    resolve_secret_refs,
    scrub_ref_backed_secrets,
)


def get_config_path() -> Path:
    return get_agent_data_path() / "config.json"


def _camel_to_snake(name: str) -> str:
    out = []
    for i, ch in enumerate(str(name)):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _snake_to_camel(name: str) -> str:
    parts = str(name).split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def convert_keys_to_snake(data: Any) -> Any:
    if isinstance(data, dict):
        return {_camel_to_snake(k): convert_keys_to_snake(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys_to_snake(item) for item in data]
    return data


def convert_keys_to_camel(data: Any) -> Any:
    if isinstance(data, dict):
        return {_snake_to_camel(k): convert_keys_to_camel(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys_to_camel(item) for item in data]
    return data


def _migrate_config(data: Dict[str, Any]) -> Dict[str, Any]:
    migrated = dict(data)
    tools = migrated.get("tools")
    if isinstance(tools, dict):
        exec_cfg = tools.get("exec")
        if isinstance(exec_cfg, dict):
            key = "restrict_to_workspace"
            camel = "restrictToWorkspace"
            if key in exec_cfg and key not in tools and camel not in tools:
                tools[key] = bool(exec_cfg.pop(key))
            elif camel in exec_cfg and key not in tools and camel not in tools:
                tools[key] = bool(exec_cfg.pop(camel))

        # Legacy Google Workspace CLI migration:
        # tools.gws -> tools.google_drive (+ shared tools.google_auth from calendar paths)
        legacy_gws = tools.get("gws")
        if isinstance(legacy_gws, dict):
            if "google_drive" not in tools:
                services = legacy_gws.get("services", [])
                enabled = bool(legacy_gws.get("enabled", False))
                drive_enabled = enabled
                if isinstance(services, (list, tuple)):
                    normalized = {str(item).strip().lower() for item in services}
                    drive_enabled = enabled and (
                        not normalized or "drive" in normalized
                    )
                tools["google_drive"] = {"enabled": bool(drive_enabled)}

            if "google_auth" not in tools:
                calendar_cfg = tools.get("calendar")
                credentials_file = "~/.annolid/agent/google_oauth_credentials.json"
                token_file = "~/.annolid/agent/google_oauth_token.json"
                allow_interactive_auth = False
                if isinstance(calendar_cfg, dict):
                    credentials_file = str(
                        calendar_cfg.get("credentials_file")
                        or calendar_cfg.get("credentialsFile")
                        or credentials_file
                    )
                    token_file = str(
                        calendar_cfg.get("token_file")
                        or calendar_cfg.get("tokenFile")
                        or token_file
                    )
                    allow_interactive_auth = bool(
                        calendar_cfg.get("allow_interactive_auth")
                        if "allow_interactive_auth" in calendar_cfg
                        else calendar_cfg.get("allowInteractiveAuth", False)
                    )
                tools["google_auth"] = {
                    "credentials_file": credentials_file,
                    "token_file": token_file,
                    "allow_interactive_auth": allow_interactive_auth,
                }

            tools.pop("gws", None)
    return migrated


def load_config(config_path: Path | None = None) -> AgentConfig:
    path = (
        Path(config_path).expanduser() if config_path is not None else get_config_path()
    )
    if not path.exists():
        cfg = AgentConfig()
        try:
            save_config(cfg, path)
        except OSError:
            # Keep startup resilient even when config path is not writable.
            pass
        return cfg
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return AgentConfig()
        migrated = _migrate_config(convert_keys_to_snake(raw))
        secrets = SecretsConfig.from_dict(migrated.get("secrets"))
        resolved_payload, _ = resolve_secret_refs(
            migrated,
            secrets.refs,
            store=load_secret_store(),
        )
        return AgentConfig.from_dict(resolved_payload)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return AgentConfig()


def save_config(config: AgentConfig, config_path: Path | None = None) -> None:
    path = (
        Path(config_path).expanduser() if config_path is not None else get_config_path()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    refs = getattr(getattr(config, "secrets", None), "refs", {}) or {}
    if refs:
        payload = scrub_ref_backed_secrets(payload, refs)
    payload = convert_keys_to_camel(payload)
    payload["secrets"] = config.secrets.to_dict()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
