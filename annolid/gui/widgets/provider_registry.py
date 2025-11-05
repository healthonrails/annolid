import os
from typing import Any, Callable, Dict, List, Optional


class ProviderRegistry:
    """Manage provider selection and available model lists."""

    _DEFAULT_MODELS: Dict[str, str] = {
        "ollama": "llama3.2-vision:latest",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-pro",
    }

    def __init__(
        self,
        settings: Dict[str, Any],
        save_settings: Callable[[Dict[str, Any]], None],
        *,
        ollama_module: Optional[Any] = None,
    ) -> None:
        self._settings = settings
        self._save_settings = save_settings
        self._ollama = ollama_module
        self._ollama_error_reported = False
        self._settings.setdefault("last_models", {})

    # ------------------------------------------------------------------ basic i/o
    def current_provider(self) -> str:
        return self._settings.get("provider", "ollama")

    def set_current_provider(self, provider: str) -> None:
        self._settings["provider"] = provider

    # ------------------------------------------------------------------ models
    def available_models(self, provider: str) -> List[str]:
        if provider == "ollama":
            models = self._fetch_ollama_models()
            pinned = self._settings.get("ollama", {}).get("preferred_models", [])
            for model in pinned:
                if model and model not in models:
                    models.append(model)
            if not models:
                fallback = self._DEFAULT_MODELS.get("ollama")
                if fallback:
                    models.append(fallback)
            elif models != pinned:
                self._settings.setdefault("ollama", {})["preferred_models"] = models
                self._save_settings(self._settings)
            return models

        provider_settings = self._settings.get(provider, {})
        return list(provider_settings.get("preferred_models", []))

    def resolve_initial_model(
        self, provider: str, available_models: List[str]
    ) -> str:
        last_models = self._settings.get("last_models", {})
        last_model = last_models.get(provider)
        if last_model and last_model in available_models:
            return last_model
        if available_models:
            return available_models[0]

        fallback = self._DEFAULT_MODELS.get(provider, "")
        if fallback and fallback not in available_models:
            available_models.append(fallback)
        return fallback

    def remember_last_model(self, provider: str, model: Optional[str]) -> None:
        if not model:
            return
        self._settings.setdefault("last_models", {})[provider] = model
        self._save_settings(self._settings)

    # ------------------------------------------------------------------ helpers
    def _fetch_ollama_models(self) -> List[str]:
        """Fetch available Ollama models respecting the configured host."""
        if self._ollama is None:
            return []

        host = self._settings.get("ollama", {}).get("host")
        prev_host_present = "OLLAMA_HOST" in os.environ
        prev_host_value = os.environ.get("OLLAMA_HOST")

        try:
            if host:
                os.environ["OLLAMA_HOST"] = host
            else:
                os.environ.pop("OLLAMA_HOST", None)

            model_list = self._ollama.list()

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
