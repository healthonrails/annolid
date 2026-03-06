from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

SENSITIVE_LEAF_KEYS = frozenset(
    {
        "access_token",
        "accesstoken",
        "api_key",
        "apikey",
        "bridge_token",
        "bridgetoken",
        "password",
        "secret",
        "token",
        "verify_token",
        "verifytoken",
    }
)


@dataclass
class SecretRefConfig:
    source: str = "env"
    name: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "SecretRefConfig":
        payload = dict(data or {})
        source = str(payload.get("source") or "env").strip().lower() or "env"
        name = str(
            payload.get("name")
            or payload.get("key")
            or payload.get("id")
            or payload.get("secret_name")
            or ""
        ).strip()
        return cls(source=source, name=name)

    def to_dict(self) -> Dict[str, str]:
        return {"source": self.source, "name": self.name}


@dataclass
class SecretsConfig:
    refs: Dict[str, SecretRefConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "SecretsConfig":
        payload = dict(data or {})
        refs_raw = payload.get("refs") or {}
        refs: Dict[str, SecretRefConfig] = {}
        if isinstance(refs_raw, dict):
            for key, value in refs_raw.items():
                path = str(key or "").strip()
                if not path:
                    continue
                refs[path] = SecretRefConfig.from_dict(
                    value if isinstance(value, Mapping) else None
                )
        return cls(refs=refs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refs": {
                path: ref.to_dict()
                for path, ref in sorted(self.refs.items())
                if path and ref.name
            }
        }


def get_secret_store_path() -> Path:
    from annolid.core.agent.utils import get_agent_data_path

    return get_agent_data_path() / "agent_secrets.json"


def load_secret_store(store_path: Path | None = None) -> Dict[str, str]:
    path = (
        Path(store_path).expanduser()
        if store_path is not None
        else get_secret_store_path()
    )
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in payload.items()
        if str(key or "").strip() and value is not None
    }


def save_secret_store(
    secrets: Mapping[str, str],
    store_path: Path | None = None,
) -> Path:
    path = (
        Path(store_path).expanduser()
        if store_path is not None
        else get_secret_store_path()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    payload = {
        str(key): str(value)
        for key, value in sorted(secrets.items())
        if str(key or "").strip() and str(value or "")
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def is_sensitive_path(path: str) -> bool:
    leaf = str(path or "").split(".")[-1].strip().lower()
    return leaf in SENSITIVE_LEAF_KEYS


def iter_sensitive_config_values(
    data: Mapping[str, Any] | Iterable[Any] | Any,
    prefix: str = "",
):
    if isinstance(data, Mapping):
        for key, value in data.items():
            key_text = str(key or "").strip()
            path = f"{prefix}.{key_text}" if prefix else key_text
            if path == "secrets" or path.startswith("secrets."):
                continue
            if is_sensitive_path(path):
                yield path, value
            if isinstance(value, (Mapping, list, tuple)):
                yield from iter_sensitive_config_values(value, path)
        return
    if isinstance(data, (list, tuple)):
        for idx, value in enumerate(data):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            if isinstance(value, (Mapping, list, tuple)):
                yield from iter_sensitive_config_values(value, path)


def _get_path(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in str(path or "").split("."):
        if not part:
            return None
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _set_path(data: Dict[str, Any], path: str, value: Any) -> bool:
    parts = [part for part in str(path or "").split(".") if part]
    if not parts:
        return False
    current: Dict[str, Any] = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value
    return True


def _delete_path(data: Dict[str, Any], path: str) -> bool:
    parts = [part for part in str(path or "").split(".") if part]
    if not parts:
        return False
    current: Dict[str, Any] = data
    parents: list[tuple[Dict[str, Any], str]] = []
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            return False
        parents.append((current, part))
        current = next_value
    removed = current.pop(parts[-1], None) is not None
    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
    return removed


def resolve_secret_refs(
    raw_payload: Mapping[str, Any],
    refs: Mapping[str, SecretRefConfig],
    *,
    environ: Optional[Mapping[str, str]] = None,
    store: Optional[Mapping[str, str]] = None,
) -> tuple[Dict[str, Any], list[str]]:
    payload = deepcopy(dict(raw_payload))
    env = environ if environ is not None else os.environ
    secret_store = dict(store or {})
    unresolved: list[str] = []
    for path, ref in refs.items():
        source = str(ref.source or "env").strip().lower() or "env"
        name = str(ref.name or "").strip()
        if not path or not name:
            unresolved.append(path)
            continue
        resolved = ""
        if source == "env":
            resolved = str(env.get(name) or "")
        elif source in {"local", "store", "file"}:
            resolved = str(secret_store.get(name) or "")
        else:
            unresolved.append(path)
            continue
        if not resolved:
            unresolved.append(path)
            continue
        _set_path(payload, path, resolved)
    return payload, sorted(set(unresolved))


def scrub_ref_backed_secrets(
    raw_payload: Mapping[str, Any],
    refs: Mapping[str, SecretRefConfig],
) -> Dict[str, Any]:
    payload = deepcopy(dict(raw_payload))
    for path, ref in refs.items():
        if not path or not str(ref.name or "").strip():
            continue
        _set_path(payload, path, "")
    return payload


def inspect_secret_posture(
    raw_payload: Mapping[str, Any],
    refs: Mapping[str, SecretRefConfig],
    *,
    environ: Optional[Mapping[str, str]] = None,
    store: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    env = environ if environ is not None else os.environ
    secret_store = dict(store or {})
    plaintext_paths: list[str] = []
    shadowed_plaintext_paths: list[str] = []
    for path, value in iter_sensitive_config_values(raw_payload):
        text = str(value or "").strip()
        if not text:
            continue
        if path in refs:
            shadowed_plaintext_paths.append(path)
        else:
            plaintext_paths.append(path)

    unresolved_refs: list[str] = []
    resolved_refs: list[str] = []
    for path, ref in refs.items():
        source = str(ref.source or "env").strip().lower() or "env"
        name = str(ref.name or "").strip()
        if not name:
            unresolved_refs.append(path)
            continue
        if source == "env":
            if str(env.get(name) or "").strip():
                resolved_refs.append(path)
            else:
                unresolved_refs.append(path)
        elif source in {"local", "store", "file"}:
            if str(secret_store.get(name) or "").strip():
                resolved_refs.append(path)
            else:
                unresolved_refs.append(path)
        else:
            unresolved_refs.append(path)

    return {
        "plaintext_paths": sorted(set(plaintext_paths)),
        "shadowed_plaintext_paths": sorted(set(shadowed_plaintext_paths)),
        "resolved_ref_paths": sorted(set(resolved_refs)),
        "unresolved_ref_paths": sorted(set(unresolved_refs)),
        "ref_count": len([1 for ref in refs.values() if str(ref.name or "").strip()]),
    }


def apply_secret_ref(
    raw_payload: Mapping[str, Any],
    *,
    path: str,
    ref: SecretRefConfig,
    clear_plaintext: bool = True,
) -> Dict[str, Any]:
    payload = deepcopy(dict(raw_payload))
    secrets_payload = dict(payload.get("secrets") or {})
    refs_payload = dict(secrets_payload.get("refs") or {})
    refs_payload[path] = ref.to_dict()
    secrets_payload["refs"] = refs_payload
    payload["secrets"] = secrets_payload
    if clear_plaintext:
        _set_path(payload, path, "")
    return payload


def remove_secret_ref(
    raw_payload: Mapping[str, Any],
    *,
    path: str,
) -> Dict[str, Any]:
    payload = deepcopy(dict(raw_payload))
    secrets_payload = payload.get("secrets")
    if isinstance(secrets_payload, dict):
        refs_payload = secrets_payload.get("refs")
        if isinstance(refs_payload, dict):
            refs_payload.pop(path, None)
            if not refs_payload:
                secrets_payload.pop("refs", None)
        if not secrets_payload:
            payload.pop("secrets", None)
    return payload


def read_raw_agent_config(config_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}
