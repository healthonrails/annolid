from __future__ import annotations

from annolid.gui.widgets.provider_registry import ProviderRegistry


def test_provider_registry_remembers_custom_model_per_provider() -> None:
    settings = {
        "provider": "openrouter",
        "last_models": {},
        "openrouter": {
            "preferred_models": ["openai/gpt-4o-mini"],
            "custom_models": [],
        },
    }
    saves = {"count": 0}

    def _save(_settings):
        saves["count"] += 1

    registry = ProviderRegistry(settings, _save)
    registry.remember_last_model("openrouter", "anthropic/claude-3.7-sonnet")

    assert settings["last_models"]["openrouter"] == "anthropic/claude-3.7-sonnet"
    assert "anthropic/claude-3.7-sonnet" in settings["openrouter"]["custom_models"]
    models = registry.available_models("openrouter")
    assert "openai/gpt-4o-mini" in models
    assert "anthropic/claude-3.7-sonnet" in models
    assert saves["count"] >= 1


def test_provider_registry_uses_last_model_even_if_not_predefined() -> None:
    settings = {
        "provider": "openrouter",
        "last_models": {"openrouter": "meta-llama/llama-3.3-70b-instruct"},
        "openrouter": {"preferred_models": ["openai/gpt-4o-mini"]},
    }
    registry = ProviderRegistry(settings, lambda _settings: None)

    available = registry.available_models("openrouter")
    selected = registry.resolve_initial_model("openrouter", available)

    assert selected == "meta-llama/llama-3.3-70b-instruct"
    assert "meta-llama/llama-3.3-70b-instruct" in available
