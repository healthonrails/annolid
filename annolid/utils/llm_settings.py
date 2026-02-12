from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_SETTINGS_DIR = Path.home() / ".annolid"
_SETTINGS_FILE = _SETTINGS_DIR / "llm_settings.json"
_GLOBAL_DOTENV_FILE = _SETTINGS_DIR / ".env"
_SETTINGS_DIR_MODE = 0o700
_SETTINGS_FILE_MODE = 0o600
_SECRET_KEY_NAMES = {
    "api_key",
    "apikey",
    "access_token",
    "token",
    "secret",
    "password",
}
_SECRET_KEY_MARKERS = ("api_key", "token", "secret", "password", "access_key")
_SAFE_METADATA_KEYS = {"api_key_env"}

_DEFAULT_MODEL_FALLBACKS: Dict[str, str] = {
    "ollama": "qwen3-vl",  # prefer a tool-friendly local VLM
    "openai": "gpt-4o-mini",
    "openrouter": "openai/gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
}

_DEFAULT_PROVIDER_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "label": "Ollama (local)",
        "kind": "ollama",
        "env_keys": ["OLLAMA_HOST"],
        "host_default": "http://localhost:11434",
        "model_placeholder": "Type model name (e.g. qwen3-vl) and press Add",
    },
    "openai": {
        "label": "OpenAI GPT",
        "kind": "openai_compat",
        "env_keys": ["OPENAI_API_KEY"],
        "api_key_env": ["OPENAI_API_KEY"],
        "base_url_default": "https://api.openai.com/v1",
        "base_url_env": "OPENAI_BASE_URL",
        "model_placeholder": "Type model name (e.g. gpt-4o-mini) and press Add",
    },
    "openrouter": {
        "label": "OpenRouter",
        "kind": "openai_compat",
        "env_keys": ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
        "api_key_env": ["OPENROUTER_API_KEY"],
        "base_url_default": "https://openrouter.ai/api/v1",
        "base_url_env": "",
        "model_placeholder": (
            "Type model name (e.g. openai/gpt-4o-mini) and press Add"
        ),
    },
    "gemini": {
        "label": "Google Gemini",
        "kind": "gemini",
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "api_key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "model_placeholder": "Type model name (e.g. gemini-1.5-flash) and press Add",
    },
}


def _build_default_settings() -> Dict[str, Any]:
    return {
        "provider": "ollama",
        "provider_definitions": deepcopy(_DEFAULT_PROVIDER_DEFINITIONS),
        "env": {"vars": {}},
        "ollama": {
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "preferred_models": ["qwen3-vl", "llama3.2-vision:latest"],
        },
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "preferred_models": [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4.1-mini",
                "gpt-3.5-turbo",
            ],
        },
        "openrouter": {
            "api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "base_url": "https://openrouter.ai/api/v1",
            "preferred_models": [
                "openai/gpt-4o-mini",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-2.0-flash-001",
            ],
        },
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
            "preferred_models": [
                "gemini-2.0-flash-thinking-exp-01-21",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ],
        },
        "last_models": {},
        "agent": {
            "temperature": 0.7,
            "max_tool_iterations": 12,
            "max_history_messages": 24,
            "memory_window": 50,
        },
        "profiles": {
            "caption": {"provider": "ollama"},
            "behavior_agent": {
                "provider": "gemini",
                "model": "gemini-2.0-flash-thinking-exp-01-21",
            },
            "polygon_agent": {
                "provider": "gemini",
                "model": "gemini-2.0-flash-thinking-exp-01-21",
            },
            "frame_agent": {"provider": "ollama", "model": "llama3.2"},
            "research_agent": {"provider": "ollama", "model": "llama3.2"},
            "playground": {"provider": "openai", "model": "gpt-4o"},
            "sam3_agent": {"provider": "ollama", "model": "qwen3-vl"},
        },
    }


@dataclass
class LLMConfig:
    """Resolved configuration describing the target LLM provider and model."""

    provider: str
    model: str
    profile: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.params = dict(self.params or {})

    @property
    def host(self) -> Optional[str]:
        return self.params.get("host")

    @property
    def api_key(self) -> Optional[str]:
        return self.params.get("api_key")

    @property
    def base_url(self) -> Optional[str]:
        return self.params.get("base_url")

    @property
    def preferred_models(self) -> List[str]:
        return list(self.params.get("preferred_models", []) or [])


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Resolved defaults used by the Annolid tool-calling agent loop."""

    temperature: float = 0.7
    max_tool_iterations: int = 12
    max_history_messages: int = 24
    memory_window: int = 50


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(deepcopy(base[key]), value)
        else:
            base[key] = value
    return base


def _normalize_provider_id(value: Any) -> str:
    provider = str(value or "").strip().lower()
    if not provider:
        return ""
    out: List[str] = []
    for ch in provider:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _normalize_provider_definitions(
    settings: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {
        key: dict(value) for key, value in _DEFAULT_PROVIDER_DEFINITIONS.items()
    }
    raw_defs = settings.get("provider_definitions")
    if isinstance(raw_defs, dict):
        for raw_key, raw_value in raw_defs.items():
            provider_id = _normalize_provider_id(raw_key)
            if not provider_id:
                continue
            block = dict(raw_value) if isinstance(raw_value, dict) else {}
            base = dict(merged.get(provider_id, {}))
            base.update(block)
            base["label"] = str(base.get("label") or provider_id.title()).strip()
            kind = str(base.get("kind") or "openai_compat").strip().lower()
            if kind not in {"ollama", "openai_compat", "gemini"}:
                kind = "openai_compat"
            base["kind"] = kind

            env_keys = base.get("env_keys", [])
            if isinstance(env_keys, str):
                env_keys = [env_keys]
            if not isinstance(env_keys, list):
                env_keys = []
            base["env_keys"] = [str(k).strip() for k in env_keys if str(k).strip()]

            api_key_env = base.get("api_key_env", [])
            if isinstance(api_key_env, str):
                api_key_env = [api_key_env]
            if not isinstance(api_key_env, list):
                api_key_env = []
            base["api_key_env"] = [
                str(k).strip() for k in api_key_env if str(k).strip()
            ]
            if kind != "ollama" and not base["api_key_env"]:
                base["api_key_env"] = list(base["env_keys"])
            base["base_url_default"] = str(base.get("base_url_default") or "").strip()
            base["base_url_env"] = str(base.get("base_url_env") or "").strip()
            base["model_placeholder"] = str(
                base.get("model_placeholder") or "Type model name and press Add"
            ).strip()
            if kind == "ollama" and not str(base.get("host_default") or "").strip():
                base["host_default"] = "http://localhost:11434"
            merged[provider_id] = base

    for provider_id, spec in list(merged.items()):
        if provider_id not in settings or not isinstance(
            settings.get(provider_id), dict
        ):
            settings[provider_id] = {}
        if provider_id == "ollama":
            settings[provider_id].setdefault(
                "host",
                str(spec.get("host_default") or "http://localhost:11434"),
            )
    return merged


def _normalise_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    settings.setdefault("last_models", {})
    settings.setdefault("profiles", {})
    settings.setdefault("agent", {})
    provider_definitions = _normalize_provider_definitions(settings)
    settings["provider_definitions"] = provider_definitions
    settings.setdefault("env", {})
    env_block = settings.get("env")
    if not isinstance(env_block, dict):
        env_block = {}
        settings["env"] = env_block
    vars_block = env_block.get("vars")
    if not isinstance(vars_block, dict):
        env_block["vars"] = {}

    for key in provider_definitions:
        settings.setdefault(key, {})
        block = settings.get(key)
        if not isinstance(block, dict):
            settings[key] = {}
            block = settings[key]
        preferred = block.get("preferred_models", [])
        if isinstance(preferred, str):
            block["preferred_models"] = [preferred]
        elif isinstance(preferred, list):
            block["preferred_models"] = [
                str(item).strip() for item in preferred if str(item).strip()
            ]
        else:
            block["preferred_models"] = []
    if not isinstance(settings.get("profiles"), dict):
        settings["profiles"] = {}
    if not isinstance(settings.get("last_models"), dict):
        settings["last_models"] = {}
    if not isinstance(settings.get("agent"), dict):
        settings["agent"] = {}
    return settings


def _camel_to_snake(name: str) -> str:
    result: List[str] = []
    for i, char in enumerate(str(name)):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _convert_keys_to_snake(data: Any) -> Any:
    if isinstance(data, dict):
        return {_camel_to_snake(k): _convert_keys_to_snake(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_keys_to_snake(item) for item in data]
    return data


def _migrate_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply backward-compatible migrations for persisted llm_settings.

    Known migrations:
    - camelCase keys to snake_case
    - `agent.max_iterations` -> `agent.max_tool_iterations`
    - profile-level `max_iterations` -> `max_tool_iterations`
    """
    original_env = data.get("env") if isinstance(data.get("env"), dict) else None
    migrated = _convert_keys_to_snake(data)
    if isinstance(original_env, dict):
        restored_env: Dict[str, Any] = {}
        for key, value in original_env.items():
            if str(key) == "vars" and isinstance(value, dict):
                restored_env["vars"] = {str(k): v for k, v in value.items()}
            else:
                restored_env[str(key)] = value
        migrated["env"] = restored_env

    agent_cfg = migrated.get("agent")
    if isinstance(agent_cfg, dict):
        if "max_iterations" in agent_cfg and "max_tool_iterations" not in agent_cfg:
            agent_cfg["max_tool_iterations"] = agent_cfg.pop("max_iterations")

    profiles = migrated.get("profiles")
    if isinstance(profiles, dict):
        for _, profile_cfg in profiles.items():
            if not isinstance(profile_cfg, dict):
                continue
            if (
                "max_iterations" in profile_cfg
                and "max_tool_iterations" not in profile_cfg
            ):
                profile_cfg["max_tool_iterations"] = profile_cfg.pop("max_iterations")
    return migrated


def _fallback_model_for(provider: str) -> str:
    return _DEFAULT_MODEL_FALLBACKS.get(provider, "")


def provider_definitions(settings: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return dict(_normalise_settings(settings).get("provider_definitions", {}) or {})


def list_providers(settings: Dict[str, Any]) -> List[str]:
    defs = provider_definitions(settings)
    return list(defs.keys())


def provider_label(settings: Dict[str, Any], provider: str) -> str:
    defs = provider_definitions(settings)
    spec = defs.get(str(provider or "").strip().lower(), {})
    return str(spec.get("label") or str(provider or "").strip() or "Provider")


def provider_kind(settings: Dict[str, Any], provider: str) -> str:
    defs = provider_definitions(settings)
    spec = defs.get(str(provider or "").strip().lower(), {})
    kind = str(spec.get("kind") or "").strip().lower()
    if kind in {"ollama", "openai_compat", "gemini"}:
        return kind
    return "openai_compat"


def _provider_block(settings: Dict[str, Any], provider: str) -> Dict[str, Any]:
    block = settings.get(provider)
    if isinstance(block, dict):
        return dict(block)
    return {}


def _inject_env_defaults(
    settings: Dict[str, Any], provider: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    params = {k: v for k, v in params.items() if v not in (None, "")}
    p_kind = provider_kind(settings, provider)
    defs = provider_definitions(settings)
    spec = defs.get(provider, {})
    if p_kind == "ollama":
        if not params.get("host"):
            env_host = os.getenv("OLLAMA_HOST")
            if env_host:
                params["host"] = env_host
        return params

    env_keys = spec.get("env_keys", [])
    if not isinstance(env_keys, list):
        env_keys = []
    if not params.get("api_key"):
        for env_name in env_keys:
            value = str(os.getenv(str(env_name).strip()) or "").strip()
            if value:
                params["api_key"] = value
                break
    if (
        provider == "openai"
        and not params.get("api_key")
        and "openrouter" in str(params.get("base_url") or "").lower()
    ):
        openrouter_key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
        if openrouter_key:
            params["api_key"] = openrouter_key
    if p_kind == "openai_compat" and not params.get("base_url"):
        env_base_url = str(spec.get("base_url_env") or "").strip()
        if env_base_url:
            value = str(os.getenv(env_base_url) or "").strip()
            if value:
                params["base_url"] = value
        if not params.get("base_url"):
            default_base = str(spec.get("base_url_default") or "").strip()
            if default_base:
                params["base_url"] = default_base
    if p_kind == "gemini" and not params.get("api_key"):
        if not params.get("api_key"):
            env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if env_key:
                params["api_key"] = env_key
    return params


def _parse_dotenv_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for raw_line in lines:
        line = str(raw_line or "").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_name = str(key).strip()
        if not env_name:
            continue
        parsed = str(value).strip()
        if (
            parsed.startswith(("'", '"'))
            and parsed.endswith(("'", '"'))
            and len(parsed) >= 2
        ):
            parsed = parsed[1:-1]
        out[env_name] = parsed
    return out


def _apply_env_files() -> None:
    dotenv_sources = [
        Path.cwd() / ".env",
        _GLOBAL_DOTENV_FILE,
    ]
    for source in dotenv_sources:
        values = _parse_dotenv_file(source)
        for key, value in values.items():
            if key not in os.environ:
                os.environ[key] = value


def _iter_inline_env(settings: Dict[str, Any]) -> List[Tuple[str, str]]:
    env_cfg = settings.get("env")
    if not isinstance(env_cfg, dict):
        return []
    pairs: List[Tuple[str, str]] = []
    for key, value in env_cfg.items():
        if key == "vars":
            continue
        env_name = str(key or "").strip()
        if env_name:
            pairs.append((env_name, str(value or "")))
    vars_cfg = env_cfg.get("vars")
    if isinstance(vars_cfg, dict):
        for key, value in vars_cfg.items():
            env_name = str(key or "").strip()
            if env_name:
                pairs.append((env_name, str(value or "")))
    return pairs


def _apply_inline_env(settings: Dict[str, Any]) -> None:
    for env_name, env_value in _iter_inline_env(settings):
        if env_name in os.environ:
            continue
        if env_value:
            os.environ[env_name] = env_value


def _format_dotenv_value(value: str) -> str:
    text = str(value or "")
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or "#" in text or '"' in text or "\\" in text:
        return json.dumps(text)
    return text


def global_env_path() -> Path:
    """Return the global dotenv path used by Annolid for env fallbacks."""
    return _GLOBAL_DOTENV_FILE


def persist_global_env_vars(values: Dict[str, str]) -> None:
    """
    Upsert environment variables into ~/.annolid/.env.

    Existing keys are updated, unknown keys are appended, and file mode is
    hardened to 600. Empty keys/values are ignored.
    """
    filtered: Dict[str, str] = {}
    for raw_key, raw_value in dict(values or {}).items():
        key = str(raw_key or "").strip()
        value = str(raw_value or "").strip()
        if not key or not value:
            continue
        filtered[key] = value
    if not filtered:
        return

    _ensure_secure_storage_permissions()
    existing = _parse_dotenv_file(_GLOBAL_DOTENV_FILE)
    merged = dict(existing)
    merged.update(filtered)
    lines = [f"{key}={_format_dotenv_value(merged[key])}" for key in sorted(merged)]
    _GLOBAL_DOTENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _secure_path_permissions(_GLOBAL_DOTENV_FILE, _SETTINGS_FILE_MODE)


def load_llm_settings() -> Dict[str, Any]:
    """Load saved LLM settings, falling back to defaults when unavailable."""
    _apply_env_files()
    settings = deepcopy(_build_default_settings())
    _ensure_secure_storage_permissions()
    if not _SETTINGS_FILE.exists():
        normalized = _normalise_settings(settings)
        _apply_inline_env(normalized)
        return deepcopy(normalized)

    try:
        with _SETTINGS_FILE.open("r", encoding="utf-8") as fh:
            persisted = json.load(fh)
    except (json.JSONDecodeError, OSError):
        normalized = _normalise_settings(settings)
        _apply_inline_env(normalized)
        return deepcopy(normalized)

    if not isinstance(persisted, dict):
        normalized = _normalise_settings(settings)
        _apply_inline_env(normalized)
        return deepcopy(normalized)
    persisted = _migrate_settings(persisted)
    merged = _deep_update(settings, persisted)
    normalized = _normalise_settings(merged)
    _apply_inline_env(normalized)
    return normalized


def save_llm_settings(settings: Dict[str, Any]) -> None:
    """Persist LLM settings to disk."""

    sanitized = scrub_secrets_for_persistence(settings)
    _ensure_secure_storage_permissions()
    with _SETTINGS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(sanitized, fh, indent=2)
    _secure_path_permissions(_SETTINGS_FILE, _SETTINGS_FILE_MODE)


def settings_path() -> Path:
    """Return the absolute path to the stored settings file."""
    return _SETTINGS_FILE


def _secure_path_permissions(path: Path, mode: int) -> None:
    try:
        if path.exists():
            os.chmod(path, mode)
    except OSError:
        return


def _ensure_secure_storage_permissions() -> None:
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    _secure_path_permissions(_SETTINGS_DIR, _SETTINGS_DIR_MODE)
    _secure_path_permissions(_SETTINGS_FILE, _SETTINGS_FILE_MODE)


def _scrub_secrets_recursive(data: Any) -> Any:
    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in data.items():
            key_text = str(key or "").strip().lower()
            if key_text in _SAFE_METADATA_KEYS:
                cleaned[key] = _scrub_secrets_recursive(value)
                continue
            if key_text in _SECRET_KEY_NAMES or any(
                marker in key_text for marker in _SECRET_KEY_MARKERS
            ):
                continue
            cleaned[key] = _scrub_secrets_recursive(value)
        return cleaned
    if isinstance(data, list):
        return [_scrub_secrets_recursive(item) for item in data]
    return data


def scrub_secrets_for_persistence(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied settings payload with secret/token fields removed."""
    payload = deepcopy(settings)
    cleaned = _scrub_secrets_recursive(payload)
    if isinstance(cleaned, dict):
        return cleaned
    return {}


def has_provider_api_key(settings: Dict[str, Any], provider: str) -> bool:
    provider_key = str(provider or "").strip().lower()
    block = _provider_block(settings, provider_key)
    if isinstance(block, dict):
        value = str(block.get("api_key") or "").strip()
        if value:
            return True
    defs = provider_definitions(settings)
    spec = defs.get(provider_key, {})
    env_keys = spec.get("env_keys", [])
    if isinstance(env_keys, list):
        for env_name in env_keys:
            if bool(str(os.getenv(str(env_name).strip()) or "").strip()):
                return True
    if provider_key == "openai":
        base_url = str(block.get("base_url") or "").lower()
        if "openrouter" in base_url:
            return bool(str(os.getenv("OPENROUTER_API_KEY") or "").strip())
    return False


def default_settings() -> Dict[str, Any]:
    """Return a copy of the default settings."""
    _apply_env_files()
    defaults = _normalise_settings(_build_default_settings())
    _apply_inline_env(defaults)
    return deepcopy(defaults)


def resolve_llm_config(
    profile: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    persist: bool = True,
) -> LLMConfig:
    """
    Resolve the effective LLM configuration for a given profile/provider/model triple.

    If a model is not supplied, the resolution order is:
        1. A profile-specific override (if a profile is provided).
        2. The last used model for the resolved provider.
        3. The provider's preferred_models list.
        4. A global fallback for the provider.
    """
    settings = load_llm_settings()
    profiles = settings.get("profiles", {})
    profile_settings = profiles.get(profile or "", {})

    resolved_provider = (
        provider
        or profile_settings.get("provider")
        or settings.get("provider")
        or "ollama"
    )

    provider_block = _provider_block(settings, resolved_provider)
    provider_block = _inject_env_defaults(settings, resolved_provider, provider_block)

    last_models = settings.get("last_models", {})
    resolved_model = (
        model or last_models.get(resolved_provider) or profile_settings.get("model")
    )

    if not resolved_model:
        preferred = provider_block.get("preferred_models") or []
        if preferred:
            resolved_model = preferred[0]

    if not resolved_model:
        resolved_model = _fallback_model_for(resolved_provider)

    if not resolved_model:
        raise ValueError(
            f"No model configured for provider '{resolved_provider}'. "
            "Update the AI Model Settings dialog to continue."
        )

    if persist:
        last_models = settings.setdefault("last_models", {})
        if last_models.get(resolved_provider) != resolved_model:
            last_models[resolved_provider] = resolved_model
            save_llm_settings(settings)

    return LLMConfig(
        provider=resolved_provider,
        model=resolved_model,
        profile=profile,
        params=provider_block,
    )


def resolve_agent_runtime_config(profile: Optional[str] = None) -> AgentRuntimeConfig:
    """
    Resolve agent runtime defaults, allowing profile-level overrides.

    This keeps settings backward-compatible while exposing a typed view for loop code.
    """
    settings = load_llm_settings()
    agent_defaults = dict(settings.get("agent") or {})
    profile_cfg = dict((settings.get("profiles") or {}).get(profile or "", {}) or {})

    def _as_int(value: Any, fallback: int, minimum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = fallback
        return max(minimum, parsed)

    def _as_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = fallback
        return min(maximum, max(minimum, parsed))

    temperature = _as_float(
        profile_cfg.get("temperature", agent_defaults.get("temperature", 0.7)),
        0.7,
        0.0,
        2.0,
    )
    max_tool_iterations = _as_int(
        profile_cfg.get(
            "max_tool_iterations",
            profile_cfg.get(
                "max_iterations", agent_defaults.get("max_tool_iterations", 12)
            ),
        ),
        12,
        1,
    )
    max_history_messages = _as_int(
        profile_cfg.get(
            "max_history_messages", agent_defaults.get("max_history_messages", 24)
        ),
        24,
        2,
    )
    memory_window = _as_int(
        profile_cfg.get("memory_window", agent_defaults.get("memory_window", 50)),
        50,
        4,
    )
    return AgentRuntimeConfig(
        temperature=temperature,
        max_tool_iterations=max_tool_iterations,
        max_history_messages=max_history_messages,
        memory_window=memory_window,
    )


def update_last_model(provider: str, model: str) -> None:
    """Record the most recently used model for a provider."""
    settings = load_llm_settings()
    last_models = settings.setdefault("last_models", {})
    if last_models.get(provider) == model:
        return
    last_models[provider] = model
    save_llm_settings(settings)


def ensure_provider_env(config: LLMConfig) -> None:
    """Apply environment variables required by third-party SDKs."""
    settings = load_llm_settings()
    p_kind = provider_kind(settings, config.provider)
    defs = provider_definitions(settings)
    spec = defs.get(config.provider, {})
    if p_kind == "ollama":
        host = config.params.get("host")
        if host:
            os.environ["OLLAMA_HOST"] = host
    elif p_kind == "openai_compat":
        api_key = config.params.get("api_key")
        env_vars = spec.get("api_key_env", [])
        if isinstance(env_vars, str):
            env_vars = [env_vars]
        if not isinstance(env_vars, list):
            env_vars = []
        if api_key:
            for env_name in env_vars:
                env_key = str(env_name or "").strip()
                if env_key:
                    os.environ[env_key] = str(api_key)
        base_url = config.params.get("base_url")
        base_env = str(spec.get("base_url_env") or "").strip()
        if base_url and base_env:
            os.environ[base_env] = str(base_url)
        if base_url and config.provider == "openai":
            os.environ["OPENAI_BASE_URL"] = str(base_url)
    elif p_kind == "gemini":
        api_key = config.params.get("api_key")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key


def fetch_available_models(provider: str) -> List[str]:
    """
    Fetch available models for the provider using local/system settings.

    Ollama queries the local server, while OpenAI/Gemini return preferred presets.
    """
    settings = load_llm_settings()
    provider_block = _provider_block(settings, provider)
    kind = provider_kind(settings, provider)

    if kind == "ollama":
        host = provider_block.get("host")
        prev_host = os.environ.get("OLLAMA_HOST")
        if host:
            os.environ["OLLAMA_HOST"] = host

        try:
            import ollama  # type: ignore

            response = ollama.list()
        except Exception:
            fallback = _fallback_model_for(provider) or _fallback_model_for("ollama")
            return provider_block.get("preferred_models", []) or [fallback]
        finally:
            if host and prev_host is not None:
                os.environ["OLLAMA_HOST"] = prev_host
            elif host:
                os.environ.pop("OLLAMA_HOST", None)

        models = []
        payload = response.get("models") if isinstance(response, dict) else None
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("name"):
                    models.append(item["name"])
        return models or provider_block.get("preferred_models", [])

    return provider_block.get("preferred_models", [])
