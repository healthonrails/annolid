from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional


def build_tracking_video_processor(
    *,
    video_path: str,
    model_name: str,
    device: Optional[object] = None,
    results_folder: Optional[str] = None,
    **kwargs: Any,
):
    """Service-layer facade for Cutie/runtime tracking processor construction."""
    from annolid.segmentation.cutie_vos.runtime import (
        build_tracking_video_processor as _build_runtime_processor,
    )

    return _build_runtime_processor(
        video_path=video_path,
        model_name=model_name,
        device=device,
        results_folder=results_folder,
        **kwargs,
    )


def is_cutie_model_name(model_name: str) -> bool:
    from annolid.segmentation.cutie_vos.runtime import (
        is_cutie_model_name as _is_cutie_model_name,
    )

    return bool(_is_cutie_model_name(model_name))


def resolve_tracking_video_processor_class(model_name: str):
    from annolid.segmentation.cutie_vos.runtime import (
        resolve_tracking_video_processor_class as _resolve_tracking_video_processor_class,
    )

    return _resolve_tracking_video_processor_class(model_name)


def create_cutie_engine(
    *,
    cutie_config_overrides: Optional[Mapping[str, Any]] = None,
    device: Optional[object] = None,
):
    from annolid.segmentation.cutie_vos.engine import CutieEngine

    return CutieEngine(
        cutie_config_overrides=dict(cutie_config_overrides or {}),
        device=device,
    )


def create_segmented_cutie_executor(
    *,
    video_path_str: str,
    segment_annotated_frame: int,
    segment_start_frame: int,
    segment_end_frame: int,
    processing_config: Mapping[str, Any],
    pred_worker: object,
    device: object,
    cutie_engine: Optional[object] = None,
):
    from annolid.segmentation.cutie_vos.processor import SegmentedCutieExecutor

    return SegmentedCutieExecutor(
        video_path_str=video_path_str,
        segment_annotated_frame=int(segment_annotated_frame),
        segment_start_frame=int(segment_start_frame),
        segment_end_frame=int(segment_end_frame),
        processing_config=dict(processing_config),
        pred_worker=pred_worker,
        device=device,
        cutie_engine=cutie_engine,
    )


def run_tracking_video_frames(
    *,
    processor: object,
    start_frame: int,
    end_frame: int,
    config: Mapping[str, Any],
) -> str:
    """Run tracking over a frame range via a shared service API."""
    return processor.process_video_frames(
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        step=1,
        is_cutie=True,
        mem_every=int(config.get("mem_every", 5)),
        point_tracking=False,
        has_occlusion=bool(config.get("has_occlusion", True)),
        save_video_with_color_mask=bool(
            config.get("save_video_with_color_mask", False)
        ),
    )


def run_video_processor_frames(
    *,
    processor: object,
    start_frame: int,
    end_frame: int,
    is_cutie: bool,
    is_new_segment: bool = False,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    kwargs = dict(extra_kwargs or {})
    return processor.process_video_frames(
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        is_cutie=bool(is_cutie),
        is_new_segment=bool(is_new_segment),
        **kwargs,
    )


def has_tracking_completion_artifacts(
    *,
    video_path: str,
    output_folder: str,
    total_frames: int,
) -> bool:
    """Check whether tracking outputs already exist for a video."""
    import glob

    video_name = Path(video_path).stem
    output_path = Path(output_folder)
    csv_pattern = str(output_path / f"{video_name}*_tracking.csv")
    if glob.glob(csv_pattern):
        return True
    last_frame = int(total_frames) - 1
    json_filename = output_path / f"{video_name}_{last_frame:09d}.json"
    return json_filename.exists()


__all__ = [
    "build_tracking_video_processor",
    "create_cutie_engine",
    "create_segmented_cutie_executor",
    "has_tracking_completion_artifacts",
    "is_cutie_model_name",
    "resolve_tracking_video_processor_class",
    "run_video_processor_frames",
    "run_tracking_video_frames",
]
