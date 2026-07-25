from __future__ import annotations


_KNOWN_TEXT_ONLY_MODEL_MARKERS = (
    "nemotron-3-super",
    "nemotron-3-ultra",
)
_PROVIDER_TEXT_ONLY_MODEL_MARKERS = {
    # These gateway variants have returned explicit "not a multimodal model"
    # responses even when similarly named models may support images elsewhere.
    "nvidia": ("glm-5", "kimi-k2"),
}


def is_known_text_only_model(*, provider: str = "", model: str = "") -> bool:
    """Return whether a model family is documented or observed as text-only."""
    provider_text = str(provider or "").strip().lower()
    model_text = str(model or "").strip().lower()
    if not model_text:
        return False
    if any(marker in model_text for marker in _KNOWN_TEXT_ONLY_MODEL_MARKERS):
        return True
    provider_markers = _PROVIDER_TEXT_ONLY_MODEL_MARKERS.get(provider_text, ())
    return any(marker in model_text for marker in provider_markers)


__all__ = ["is_known_text_only_model"]
