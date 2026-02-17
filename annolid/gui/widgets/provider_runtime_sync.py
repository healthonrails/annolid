from __future__ import annotations

from typing import Any, Callable, Optional

from annolid.gui.widgets.provider_registry import ProviderRegistry
from annolid.utils.llm_settings import load_llm_settings, save_llm_settings


def refresh_runtime_llm_settings(
    owner: Any,
    *,
    after_refresh: Optional[Callable[[], None]] = None,
) -> None:
    """Reload provider/model settings and apply saved selection immediately."""
    try:
        latest = load_llm_settings()
    except Exception:
        return
    if not isinstance(latest, dict):
        return

    prev_provider = str(getattr(owner, "selected_provider", "") or "").strip().lower()
    prev_model = str(getattr(owner, "selected_model", "") or "").strip()

    owner.llm_settings = latest
    owner._providers = ProviderRegistry(owner.llm_settings, save_llm_settings)

    populate = getattr(owner, "_populate_provider_selector", None)
    if callable(populate):
        populate()

    providers = owner._providers.available_providers()
    saved_provider = str(latest.get("provider", "") or "").strip().lower()
    if saved_provider and saved_provider in providers:
        next_provider = saved_provider
    elif prev_provider and prev_provider in providers:
        next_provider = prev_provider
    else:
        next_provider = owner._providers.current_provider()

    setattr(owner, "_suppress_provider_updates", True)
    try:
        selector = getattr(owner, "provider_selector", None)
        if selector is not None:
            provider_index = selector.findData(next_provider)
            if provider_index != -1:
                selector.setCurrentIndex(provider_index)
        owner.selected_provider = next_provider
    finally:
        setattr(owner, "_suppress_provider_updates", False)

    owner.available_models = owner._providers.available_models(owner.selected_provider)
    if prev_model and prev_provider == next_provider:
        owner.selected_model = prev_model
        if prev_model not in owner.available_models:
            owner.available_models.append(prev_model)
    else:
        owner.selected_model = owner._providers.resolve_initial_model(
            owner.selected_provider,
            owner.available_models,
        )

    update_model_selector = getattr(owner, "_update_model_selector", None)
    if callable(update_model_selector):
        update_model_selector()
    selector = getattr(owner, "model_selector", None)
    if selector is not None and str(owner.selected_model or "").strip():
        selector.setCurrentText(owner.selected_model)

    if callable(after_refresh):
        after_refresh()
