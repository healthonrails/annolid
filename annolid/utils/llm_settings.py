from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


_SETTINGS_DIR = Path.home() / ".annolid"
_SETTINGS_FILE = _SETTINGS_DIR / "llm_settings.json"

_DEFAULT_MODEL_FALLBACKS: Dict[str, str] = {
    "ollama": "llama3.2-vision:latest",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
}

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "provider": "ollama",
    "ollama": {
        "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "preferred_models": [],
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
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY", ""),
        "preferred_models": [
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
    },
    "last_models": {},
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


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(deepcopy(base[key]), value)
        else:
            base[key] = value
    return base


def _normalise_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    settings.setdefault("last_models", {})
    settings.setdefault("profiles", {})
    for key in ("ollama", "openai", "gemini"):
        settings.setdefault(key, {})
    return settings


def _fallback_model_for(provider: str) -> str:
    return _DEFAULT_MODEL_FALLBACKS.get(provider, "")


def _inject_env_defaults(provider: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = {k: v for k, v in params.items() if v not in (None, "")}
    if provider == "ollama":
        if not params.get("host"):
            env_host = os.getenv("OLLAMA_HOST")
            if env_host:
                params["host"] = env_host
    elif provider == "openai":
        if not params.get("api_key"):
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                params["api_key"] = env_key
        if not params.get("base_url"):
            env_url = os.getenv("OPENAI_BASE_URL")
            if env_url:
                params["base_url"] = env_url
    elif provider == "gemini":
        if not params.get("api_key"):
            env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if env_key:
                params["api_key"] = env_key
    return params


def load_llm_settings() -> Dict[str, Any]:
    """Load saved LLM settings, falling back to defaults when unavailable."""
    settings = deepcopy(_DEFAULT_SETTINGS)
    if not _SETTINGS_FILE.exists():
        return deepcopy(settings)

    try:
        with _SETTINGS_FILE.open("r", encoding="utf-8") as fh:
            persisted = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return deepcopy(settings)

    merged = _deep_update(settings, persisted)
    return _normalise_settings(merged)


def save_llm_settings(settings: Dict[str, Any]) -> None:
    """Persist LLM settings to disk."""
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with _SETTINGS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)


def settings_path() -> Path:
    """Return the absolute path to the stored settings file."""
    return _SETTINGS_FILE


def default_settings() -> Dict[str, Any]:
    """Return a copy of the default settings."""
    return deepcopy(_DEFAULT_SETTINGS)


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

    provider_block = dict(settings.get(resolved_provider, {}))
    provider_block = _inject_env_defaults(resolved_provider, provider_block)

    last_models = settings.get("last_models", {})
    resolved_model = (
        model
        or last_models.get(resolved_provider)
        or profile_settings.get("model")
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
    if config.provider == "ollama":
        host = config.params.get("host")
        if host:
            os.environ["OLLAMA_HOST"] = host
    elif config.provider == "openai":
        api_key = config.params.get("api_key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        base_url = config.params.get("base_url")
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
    elif config.provider == "gemini":
        api_key = config.params.get("api_key")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key


def create_phi_model(config: LLMConfig):
    """
    Instantiate a Phi model for the resolved configuration.

    The corresponding environment variables are ensured before model creation.
    """
    ensure_provider_env(config)

    if config.provider == "ollama":
        from phi.model.ollama import Ollama

        return Ollama(id=config.model)

    if config.provider == "openai":
        from phi.model.openai import OpenAIChat

        kwargs: Dict[str, Any] = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return OpenAIChat(id=config.model, **kwargs)

    if config.provider == "gemini":
        from phi.model.google import Gemini

        kwargs: Dict[str, Any] = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return Gemini(id=config.model, **kwargs)

    raise ValueError(f"Unsupported provider '{config.provider}'.")


def fetch_available_models(provider: str) -> List[str]:
    """
    Fetch available models for the provider using local/system settings.

    Ollama queries the local server, while OpenAI/Gemini return preferred presets.
    """
    settings = load_llm_settings()
    provider_block = settings.get(provider, {})

    if provider == "ollama":
        host = provider_block.get("host")
        prev_host = os.environ.get("OLLAMA_HOST")
        if host:
            os.environ["OLLAMA_HOST"] = host

        try:
            import ollama  # type: ignore

            response = ollama.list()
        except Exception:
            return provider_block.get("preferred_models", []) or [
                _fallback_model_for("ollama")
            ]
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
