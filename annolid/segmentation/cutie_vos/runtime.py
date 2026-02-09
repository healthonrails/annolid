from __future__ import annotations

from typing import Any

from annolid.tracker.processor_registry import (
    CUTIE_BACKEND,
    TRACKING_PROCESSOR_REGISTRY,
    register_tracking_backend,
    resolve_tracking_processor_class,
)


def is_cutie_model_name(model_name: str | None) -> bool:
    """Return True when the model identifier routes to CUTIE tracking."""
    return TRACKING_PROCESSOR_REGISTRY.has_match(model_name, CUTIE_BACKEND)


def resolve_tracking_video_processor_class(model_name: str | None):
    """Resolve the appropriate video processor class for the selected model."""
    return resolve_tracking_processor_class(model_name)


def build_tracking_video_processor(
    video_path: str,
    *,
    model_name: str | None,
    **kwargs: Any,
):
    """Create a tracking video processor with lazy backend selection."""
    kwargs_model_name = kwargs.pop("model_name", None)
    kwargs.pop("video_path", None)
    effective_model_name = (
        model_name if model_name is not None else kwargs_model_name
    )
    processor_cls = resolve_tracking_video_processor_class(effective_model_name)
    return processor_cls(
        video_path=video_path,
        model_name=effective_model_name,
        **kwargs,
    )


__all__ = [
    "build_tracking_video_processor",
    "is_cutie_model_name",
    "register_tracking_backend",
    "resolve_tracking_video_processor_class",
]
