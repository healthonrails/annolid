import os
from typing import Any, Callable, Dict, List, Optional

from annolid.utils.llm_settings import (
    list_providers,
    provider_definitions,
    provider_kind,
)


class ProviderRegistry:
    """Manage provider selection and available model lists."""

    _DEFAULT_MODELS: Dict[str, str] = {
        "ollama": "llama3.2-vision:latest",
        "openai": "gpt-4o-mini",
        "openrouter": "openai/gpt-4o-mini",
        "gemini": "gemini-2.5-pro",
    }

    def __init__(
        self,
        settings: Dict[str, Any],
        save_settings: Callable[[Dict[str, Any]], None],
    ) -> None:
        self._settings = settings
        self._save_settings = save_settings
        self._ollama_error_reported = False
        self._settings.setdefault("last_models", {})

    # ------------------------------------------------------------------ basic i/o
    def current_provider(self) -> str:
        provider = str(self._settings.get("provider", "ollama")).strip().lower()
        if provider and provider in self.available_providers():
            return provider
        providers = self.available_providers()
        return providers[0] if providers else "ollama"

    def set_current_provider(self, provider: str) -> None:
        self._settings["provider"] = str(provider or "").strip().lower()
        self._save_settings(self._settings)

    def available_providers(self) -> List[str]:
        return list_providers(self._settings)

    def labels(self) -> Dict[str, str]:
        defs = provider_definitions(self._settings)
        labels: Dict[str, str] = {}
        for key in self.available_providers():
            spec = defs.get(key, {})
            labels[key] = str(spec.get("label") or key)
        return labels

    # ------------------------------------------------------------------ models
    def available_models(self, provider: str) -> List[str]:
        provider_settings = self._settings.get(provider, {}) or {}
        preferred = self._normalize_model_list(
            provider_settings.get("preferred_models", [])
        )
        custom = self._normalize_model_list(provider_settings.get("custom_models", []))

        if provider_kind(self._settings, provider) == "ollama":
            models = self._fetch_ollama_models()
            for model in preferred:
                if model and model not in models:
                    models.append(model)
            for model in custom:
                if model and model not in models:
                    models.append(model)
            if not models:
                fallback = self._DEFAULT_MODELS.get("ollama")
                if fallback:
                    models.append(fallback)
            elif models != preferred:
                self._settings.setdefault("ollama", {})["preferred_models"] = models
                self._save_settings(self._settings)
            return models

        models = list(preferred)
        for model in custom:
            if model and model not in models:
                models.append(model)
        return models

    def resolve_initial_model(self, provider: str, available_models: List[str]) -> str:
        last_models = self._settings.get("last_models", {})
        last_model = last_models.get(provider)
        if last_model:
            if last_model not in available_models:
                available_models.append(last_model)
            return last_model
        if available_models:
            return available_models[0]

        provider_settings = self._settings.get(provider, {}) or {}
        preferred = self._normalize_model_list(
            provider_settings.get("preferred_models", [])
        )
        fallback = self._DEFAULT_MODELS.get(provider, "")
        if not fallback and preferred:
            fallback = preferred[0]
        if fallback and fallback not in available_models:
            available_models.append(fallback)
        return fallback

    def remember_last_model(self, provider: str, model: Optional[str]) -> None:
        if not model:
            return
        model_text = str(model).strip()
        if not model_text:
            return
        prev_model = self._settings.setdefault("last_models", {}).get(provider)
        provider_settings = self._settings.setdefault(provider, {})
        preferred = self._normalize_model_list(
            provider_settings.get("preferred_models", [])
        )
        custom = self._normalize_model_list(provider_settings.get("custom_models", []))
        changed = False
        if model_text not in preferred and model_text not in custom:
            custom.append(model_text)
            provider_settings["custom_models"] = custom
            changed = True
        self._settings.setdefault("last_models", {})[provider] = model_text
        if changed or prev_model != model_text:
            self._save_settings(self._settings)

    @staticmethod
    def _normalize_model_list(raw_models: Any) -> List[str]:
        if isinstance(raw_models, str):
            return [raw_models.strip()] if raw_models.strip() else []
        if not isinstance(raw_models, list):
            return []
        seen = set()
        out: List[str] = []
        for item in raw_models:
            model = str(item or "").strip()
            if not model or model in seen:
                continue
            seen.add(model)
            out.append(model)
        return out

    # ------------------------------------------------------------------ helpers
    def _fetch_ollama_models(self) -> List[str]:
        """Fetch available Ollama models respecting the configured host."""
        try:
            import ollama
        except ImportError:
            return []

        host = self._settings.get("ollama", {}).get("host")
        prev_host_present = "OLLAMA_HOST" in os.environ
        prev_host_value = os.environ.get("OLLAMA_HOST")

        try:
            if host:
                os.environ["OLLAMA_HOST"] = host
            else:
                os.environ.pop("OLLAMA_HOST", None)

            model_list = ollama.list()

            raw_models = None
            if isinstance(model_list, dict):
                raw_models = model_list.get("models")
            else:
                raw_models = getattr(model_list, "models", None)

            models: List[str] = []
            if isinstance(raw_models, list):
                if all(isinstance(model, dict) for model in raw_models):
                    models = [model["name"] for model in raw_models]
                elif all(hasattr(model, "model") for model in raw_models):
                    models = [model.model for model in raw_models]
            if models:
                self._ollama_error_reported = False
                return models

            raise ValueError("Unexpected model format in response.")
        except Exception as exc:
            if not self._ollama_error_reported:
                friendly_host = host or prev_host_value or "http://localhost:11434"
                print(
                    "Unable to reach the Ollama server to list models. "
                    f"Check that Ollama is running at {friendly_host}. "
                    f"Original error: {exc}"
                )
                self._ollama_error_reported = True
            return []
        finally:
            if prev_host_present and prev_host_value is not None:
                os.environ["OLLAMA_HOST"] = prev_host_value
            else:
                os.environ.pop("OLLAMA_HOST", None)
