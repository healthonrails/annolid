"""Service-layer orchestration for agent admin and secret management commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _mode_octal(path: Path) -> Optional[str]:
    try:
        return oct(path.stat().st_mode & 0o777)
    except OSError:
        return None


def _is_private_dir_mode(path: Path) -> bool:
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return False
    return (mode & 0o077) == 0


def _is_private_file_mode(path: Path) -> bool:
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return False
    return mode == 0o600


def _find_persisted_secret_keys(data: object, prefix: str = "") -> list[str]:
    secret_names = {
        "api_key",
        "apikey",
        "access_token",
        "client_secret",
        "refresh_token",
        "token",
        "secret",
        "password",
    }
    if isinstance(data, dict):
        hits: list[str] = []
        for key, value in data.items():
            key_text = str(key or "").strip().lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            if key_text in secret_names:
                hits.append(path)
            hits.extend(_find_persisted_secret_keys(value, path))
        return hits
    if isinstance(data, list):
        hits: list[str] = []
        for idx, item in enumerate(data):
            item_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            hits.extend(_find_persisted_secret_keys(item, item_prefix))
        return hits
    return []


def _find_agent_config_plaintext_secret_paths(
    data: object, prefix: str = ""
) -> list[str]:
    secret_names = {
        "access_token",
        "accesstoken",
        "api_key",
        "apikey",
        "client_secret",
        "clientsecret",
        "bridge_token",
        "bridgetoken",
        "password",
        "refresh_token",
        "refreshtoken",
        "secret",
        "token",
        "verify_token",
        "verifytoken",
    }
    if isinstance(data, dict):
        hits: list[str] = []
        for key, value in data.items():
            key_text = str(key or "").strip().lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            if path == "secrets" or path.startswith("secrets."):
                continue
            if key_text in secret_names and str(value or "").strip():
                hits.append(path)
            hits.extend(_find_agent_config_plaintext_secret_paths(value, path))
        return hits
    if isinstance(data, list):
        hits: list[str] = []
        for idx, item in enumerate(data):
            item_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            hits.extend(_find_agent_config_plaintext_secret_paths(item, item_prefix))
        return hits
    return []


def run_agent_security_check() -> dict:
    from annolid.core.agent.config import get_config_path
    from annolid.core.agent.config.secrets import (
        SecretsConfig,
        get_secret_store_path,
        inspect_secret_posture,
        load_secret_store,
        read_raw_agent_config,
    )
    from annolid.core.agent.utils import get_agent_data_path
    from annolid.utils.llm_settings import has_provider_api_key, settings_path

    data_dir = get_agent_data_path()
    settings_file = settings_path()
    settings_dir = settings_file.parent

    persisted_payload: dict = {}
    parse_error: Optional[str] = None
    if settings_file.exists():
        try:
            persisted_payload = json.loads(settings_file.read_text(encoding="utf-8"))
            if not isinstance(persisted_payload, dict):
                persisted_payload = {}
                parse_error = "llm_settings.json content is not a JSON object."
        except Exception as exc:
            parse_error = str(exc)

    persisted_secret_keys = _find_persisted_secret_keys(persisted_payload)
    settings = persisted_payload if isinstance(persisted_payload, dict) else {}
    agent_config_path = get_config_path()
    agent_raw_payload = read_raw_agent_config(agent_config_path)
    agent_secrets = SecretsConfig.from_dict(agent_raw_payload.get("secrets"))
    secret_store_path = get_secret_store_path()
    secret_store = load_secret_store(secret_store_path)
    agent_secret_posture = inspect_secret_posture(
        agent_raw_payload,
        agent_secrets.refs,
        store=secret_store,
    )

    checks = {
        "settings_dir_exists": settings_dir.exists(),
        "settings_file_exists": settings_file.exists(),
        "settings_dir_private": _is_private_dir_mode(settings_dir),
        "settings_file_private": _is_private_file_mode(settings_file),
        "persisted_secrets_found": bool(persisted_secret_keys),
        "settings_json_parse_ok": parse_error is None,
        "agent_config_exists": agent_config_path.exists(),
        "agent_secret_store_exists": secret_store_path.exists(),
        "agent_secret_store_private": _is_private_file_mode(secret_store_path),
        "agent_plaintext_config_secrets_found": bool(
            agent_secret_posture["plaintext_paths"]
        ),
        "agent_secret_refs_unresolved": bool(
            agent_secret_posture["unresolved_ref_paths"]
        ),
    }
    status = "ok"
    if not all(
        [
            checks["settings_dir_exists"],
            checks["settings_file_exists"],
            checks["settings_dir_private"],
            checks["settings_file_private"],
            checks["settings_json_parse_ok"],
        ]
    ):
        status = "warning"
    if checks["persisted_secrets_found"]:
        status = "warning"
    if (
        checks["agent_plaintext_config_secrets_found"]
        or checks["agent_secret_refs_unresolved"]
    ):
        status = "warning"

    payload = {
        "status": status,
        "data_dir": str(data_dir),
        "llm_settings_path": str(settings_file),
        "agent_config_path": str(agent_config_path),
        "agent_secret_store_path": str(secret_store_path),
        "llm_settings_dir_mode": _mode_octal(settings_dir),
        "llm_settings_file_mode": _mode_octal(settings_file),
        "agent_secret_store_mode": _mode_octal(secret_store_path),
        "checks": checks,
        "persisted_secret_keys": persisted_secret_keys,
        "agent_plaintext_secret_keys": agent_secret_posture["plaintext_paths"],
        "agent_shadowed_plaintext_secret_keys": agent_secret_posture[
            "shadowed_plaintext_paths"
        ],
        "agent_unresolved_secret_refs": agent_secret_posture["unresolved_ref_paths"],
        "agent_resolved_secret_refs": agent_secret_posture["resolved_ref_paths"],
        "provider_key_presence": {
            "openai": bool(has_provider_api_key(settings, "openai")),
            "gemini": bool(has_provider_api_key(settings, "gemini")),
        },
    }
    if parse_error is not None:
        payload["settings_json_error"] = parse_error
    return payload


def run_agent_security_audit(
    *, config_path: str | Path | None = None, fix: bool = False
) -> dict:
    from annolid.core.agent.config import get_config_path
    from annolid.core.agent.security_audit import run_agent_security_audit as _run

    resolved_config = (
        Path(config_path).expanduser() if config_path else get_config_path()
    )
    return _run(
        config_path=resolved_config,
        fix=bool(fix),
    )


def audit_agent_secrets(*, config_path: str | Path | None = None) -> dict:
    from annolid.core.agent.config import get_config_path
    from annolid.core.agent.config.secrets import (
        SecretsConfig,
        get_secret_store_path,
        inspect_secret_posture,
        load_secret_store,
        read_raw_agent_config,
    )

    resolved_config = (
        Path(config_path).expanduser() if config_path else get_config_path()
    )
    raw_payload = read_raw_agent_config(resolved_config)
    secrets = SecretsConfig.from_dict(raw_payload.get("secrets"))
    store_path = get_secret_store_path()
    store = load_secret_store(store_path)
    posture = inspect_secret_posture(raw_payload, secrets.refs, store=store)
    status = "ok"
    if posture["plaintext_paths"] or posture["unresolved_ref_paths"]:
        status = "warning"
    return {
        "config_path": str(resolved_config),
        "secret_store_path": str(store_path),
        "secret_store_mode": _mode_octal(store_path),
        "ref_count": posture["ref_count"],
        "resolved_ref_paths": posture["resolved_ref_paths"],
        "unresolved_ref_paths": posture["unresolved_ref_paths"],
        "plaintext_paths": posture["plaintext_paths"],
        "shadowed_plaintext_paths": posture["shadowed_plaintext_paths"],
        "status": status,
    }


def set_agent_secret(
    *,
    path: str,
    env: str | None = None,
    local: str | None = None,
    value: str | None = None,
    config_path: str | Path | None = None,
) -> dict:
    from annolid.core.agent.config import get_config_path
    from annolid.core.agent.config.secrets import (
        SecretRefConfig,
        apply_secret_ref,
        get_secret_store_path,
        load_secret_store,
        read_raw_agent_config,
        save_secret_store,
    )

    source_count = int(bool(env)) + int(bool(local))
    if source_count != 1:
        raise SystemExit("Choose exactly one of --env or --local.")
    resolved_config = (
        Path(config_path).expanduser() if config_path else get_config_path()
    )
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    raw_payload = read_raw_agent_config(resolved_config)
    target_path = str(path or "").strip()
    if not target_path:
        raise SystemExit("--path is required.")
    if env:
        ref = SecretRefConfig(source="env", name=str(env).strip())
        next_payload = apply_secret_ref(raw_payload, path=target_path, ref=ref)
    else:
        local_name = str(local).strip()
        if not local_name:
            raise SystemExit("--local requires a non-empty key.")
        store_path = get_secret_store_path()
        secrets = load_secret_store(store_path)
        secrets[local_name] = str(value or "")
        save_secret_store(secrets, store_path)
        ref = SecretRefConfig(source="local", name=local_name)
        next_payload = apply_secret_ref(raw_payload, path=target_path, ref=ref)
    resolved_config.write_text(json.dumps(next_payload, indent=2), encoding="utf-8")
    return {
        "updated": True,
        "config_path": str(resolved_config),
        "path": target_path,
        "ref": ref.to_dict(),
    }


def remove_agent_secret(
    *,
    path: str,
    config_path: str | Path | None = None,
    delete_local_value: bool = False,
) -> dict:
    from annolid.core.agent.config import get_config_path
    from annolid.core.agent.config.secrets import (
        SecretsConfig,
        get_secret_store_path,
        load_secret_store,
        read_raw_agent_config,
        remove_secret_ref,
        save_secret_store,
    )

    resolved_config = (
        Path(config_path).expanduser() if config_path else get_config_path()
    )
    raw_payload = read_raw_agent_config(resolved_config)
    secrets = SecretsConfig.from_dict(raw_payload.get("secrets"))
    target_path = str(path or "").strip()
    ref = secrets.refs.get(target_path)
    next_payload = remove_secret_ref(raw_payload, path=target_path)
    resolved_config.write_text(json.dumps(next_payload, indent=2), encoding="utf-8")
    deleted_local_value = False
    if bool(delete_local_value) and ref and ref.source == "local" and ref.name:
        store_path = get_secret_store_path()
        store = load_secret_store(store_path)
        if ref.name in store:
            store.pop(ref.name, None)
            save_secret_store(store, store_path)
            deleted_local_value = True
    return {
        "updated": True,
        "config_path": str(resolved_config),
        "path": target_path,
        "deleted_local_value": deleted_local_value,
    }


def migrate_agent_secrets(
    *, config_path: str | Path | None = None, apply: bool = False
) -> tuple[dict, int]:
    from annolid.core.agent.config import get_config_path
    from annolid.core.agent.config.secrets import (
        SecretRefConfig,
        SecretsConfig,
        apply_secret_ref,
        get_secret_store_path,
        load_secret_store,
        read_raw_agent_config,
        save_secret_store,
    )

    resolved_config = (
        Path(config_path).expanduser() if config_path else get_config_path()
    )
    raw_payload = read_raw_agent_config(resolved_config)
    secrets_cfg = SecretsConfig.from_dict(raw_payload.get("secrets"))
    existing_refs = secrets_cfg.refs
    plaintext_paths = [
        path
        for path in _find_agent_config_plaintext_secret_paths(raw_payload)
        if path != "secrets" and path not in existing_refs
    ]
    payload = {
        "config_path": str(resolved_config),
        "candidate_paths": plaintext_paths,
        "apply": bool(apply),
        "migrated": [],
    }
    if not apply:
        return payload, (0 if not plaintext_paths else 1)

    store_path = get_secret_store_path()
    store = load_secret_store(store_path)
    next_payload = dict(raw_payload)
    for path in plaintext_paths:
        current = raw_payload
        for part in path.split("."):
            if not isinstance(current, dict):
                current = {}
                break
            current = current.get(part)
        value = str(current or "")
        if not value:
            continue
        local_name = path
        store[local_name] = value
        next_payload = apply_secret_ref(
            next_payload,
            path=path,
            ref=SecretRefConfig(source="local", name=local_name),
        )
        payload["migrated"].append({"path": path, "local_key": local_name})
    save_secret_store(store, store_path)
    resolved_config.write_text(json.dumps(next_payload, indent=2), encoding="utf-8")
    return payload, 0


__all__ = [
    "audit_agent_secrets",
    "migrate_agent_secrets",
    "remove_agent_secret",
    "run_agent_security_audit",
    "run_agent_security_check",
    "set_agent_secret",
]
