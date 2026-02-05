from __future__ import annotations

from typing import Any


def is_cutie_model_name(model_name: str | None) -> bool:
    """Return True when the model identifier routes to CUTIE tracking."""
    return "cutie" in str(model_name or "").lower()


def resolve_tracking_video_processor_class(model_name: str | None):
    """Resolve the appropriate video processor class for the selected model."""
    if is_cutie_model_name(model_name):
        from annolid.segmentation.cutie_vos.video_processor import CutieVideoProcessor

        return CutieVideoProcessor

    from annolid.segmentation.SAM.edge_sam_bg import VideoProcessor

    return VideoProcessor


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
