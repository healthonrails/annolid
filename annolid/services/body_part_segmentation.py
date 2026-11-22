"""Service-layer orchestration for intra-instance body-part segmentation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from annolid.annotation.pose_schema import PoseSchema, find_pose_schema_in_dir
from annolid.segmentation.body_parts import (
    BodyPartSegmentationConfig,
    InstanceBodyPartVideoProcessor,
)
from annolid.utils.atomic_io import atomic_write_json
from annolid.version import get_version

__all__ = [
    "run_body_part_segmentation",
    "run_body_part_segmentation_cli",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_skeleton_edges(
    pose_schema_path: Optional[Path | str],
    *,
    annotation_dir: Optional[Path] = None,
) -> Tuple[Tuple[str, str], ...]:
    if pose_schema_path is not None:
        resolved = Path(pose_schema_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Pose schema file not found: {resolved}")
        schema = PoseSchema.load(resolved)
        return tuple(schema.base_edges())
    if annotation_dir is not None:
        discovered = find_pose_schema_in_dir(annotation_dir)
        if discovered is not None:
            try:
                schema = PoseSchema.load(discovered)
                return tuple(schema.base_edges())
            except Exception:
                return ()
    return ()


def _parse_keypoint_part_map(
    raw_map: Optional[Mapping[str, str] | str],
) -> Dict[str, str]:
    if raw_map is None:
        return {}
    if isinstance(raw_map, Mapping):
        return {
            str(k).strip().lower(): str(v).strip()
            for k, v in raw_map.items()
            if str(k).strip() and str(v).strip()
        }
    text = str(raw_map).strip()
    if not text:
        return {}
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, Mapping):
            raise ValueError("--keypoint-part-map JSON must be an object.")
        return {
            str(k).strip().lower(): str(v).strip()
            for k, v in parsed.items()
            if str(k).strip() and str(v).strip()
        }
    mapping: Dict[str, str] = {}
    for entry in text.split(","):
        item = entry.strip()
        if not item:
            continue
        if "=" not in item and ":" not in item:
            raise ValueError(
                f"Invalid --keypoint-part-map item '{item}'. Expected 'keypoint=part'."
            )
        sep = "=" if "=" in item else ":"
        kp, part = item.split(sep, 1)
        if kp.strip() and part.strip():
            mapping[kp.strip().lower()] = part.strip()
    return mapping


def run_body_part_segmentation(
    video_path: Path | str,
    *,
    annotation_dir: Optional[Path | str] = None,
    config: Optional[BodyPartSegmentationConfig] = None,
    pose_schema_path: Optional[Path | str] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    export_csv: bool = False,
    export_ethogram: bool = False,
    camera_view: str = "auto",
    fps: Optional[float] = None,
    extractor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Segment body parts inside tracked instances across ``video_path``."""
    resolved_video = Path(video_path).expanduser().resolve()
    if not resolved_video.exists():
        raise FileNotFoundError(f"Video file not found: {resolved_video}")

    resolved_dir = (
        Path(annotation_dir).expanduser().resolve()
        if annotation_dir is not None
        else resolved_video.with_suffix("")
    )
    if not resolved_dir.exists() or not resolved_dir.is_dir():
        raise FileNotFoundError(
            f"Tracked annotation directory not found: {resolved_dir}"
        )

    skeleton_edges = _load_skeleton_edges(pose_schema_path, annotation_dir=resolved_dir)
    effective_config = config or BodyPartSegmentationConfig()

    processor = InstanceBodyPartVideoProcessor(
        str(resolved_video),
        results_folder=str(resolved_dir),
        body_part_config=effective_config,
        skeleton_edges=skeleton_edges,
        extractor=extractor,
    )
    try:
        message = processor.process_video_frames(
            start_frame=max(0, int(start_frame)),
            end_frame=end_frame,
            step=max(1, int(step)),
        )
        summary = dict(processor.last_summary)
    finally:
        processor.cleanup()

    csv_path_out: Optional[str] = None
    if export_csv and summary.get("frames_processed", 0) > 0:
        from annolid.annotation.labelme2csv import convert_json_to_csv

        csv_target = resolved_dir.parent / f"{resolved_dir.name}_tracking.csv"
        convert_json_to_csv(
            str(resolved_dir),
            csv_file=str(csv_target),
            force_rewrite_tracking_csv=True,
        )
        if csv_target.exists():
            csv_path_out = str(csv_target)

    ethogram_json_out: Optional[str] = None
    ethogram_summary: Optional[Dict[str, Any]] = None
    if export_ethogram and summary.get("frames_processed", 0) > 0:
        from annolid.data.videos import CV2Video
        from annolid.postprocessing.body_part_ethogram import (
            analyze_body_part_ethogram,
            export_body_part_ethogram_csvs,
        )

        if fps is not None and float(fps) > 0.0:
            vid_fps = float(fps)
        else:
            vid_fps = 30.0
            vloader = CV2Video(str(resolved_video))
            try:
                raw_fps = float(vloader.get_fps() or 30.0)
                if raw_fps > 0.0:
                    vid_fps = raw_fps
            finally:
                vloader.release()

        eth_report = analyze_body_part_ethogram(
            resolved_dir, fps=vid_fps, camera_view=camera_view
        )
        ethogram_summary = export_body_part_ethogram_csvs(
            eth_report,
            resolved_dir.parent,
            stem=resolved_dir.name,
        )
        ethogram_json_out = str(ethogram_summary["ethogram_json"])

    return {
        "video_path": str(resolved_video),
        "annotation_dir": str(resolved_dir),
        "message": message,
        "tracking_csv": csv_path_out,
        "ethogram_json": ethogram_json_out,
        "ethogram": ethogram_summary,
        **summary,
        "provenance": {
            "schema": "body_part_segmentation/v1",
            "annolid_version": str(get_version() or "0.0.0"),
            "generated_at_utc": _utc_now_iso(),
            "config": asdict(effective_config),
        },
    }


def run_body_part_segmentation_cli(
    *,
    video_path: Path | str,
    annotation_dir: Optional[Path | str] = None,
    mode: str = "auto",
    feature_backend: str = "fast",
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    crop_size: int = 224,
    crop_padding_ratio: float = 0.12,
    num_parts: int = 3,
    part_names: Optional[Sequence[str] | str] = None,
    keypoint_granularity: str = "coarse",
    keypoint_part_map: Optional[Mapping[str, str] | str] = None,
    partition_full_instance: Optional[bool] = None,
    share_seed_across_instances: bool = True,
    max_part_area_growth: float = 2.5,
    min_part_area_px: int = 6,
    boundary_refine: bool = True,
    crf_refine: bool = False,
    polygon_epsilon: float = 1.0,
    split_merged_parent_masks: bool = True,
    recover_missing_merged_instances: bool = True,
    qualify_label_with_instance: bool = False,
    persist_json: bool = False,
    device: Optional[str] = None,
    pose_schema_path: Optional[Path | str] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    export_csv: bool = False,
    export_ethogram: bool = False,
    camera_view: str = "auto",
    json_stdout: bool = False,
    out_json: Optional[Path | str] = None,
) -> int:
    """Execute ``annolid-run segment-body-parts`` and return a process exit code."""
    try:
        if isinstance(part_names, str):
            parsed_names = tuple(p.strip() for p in part_names.split(",") if p.strip())
        else:
            parsed_names = tuple(
                str(p).strip() for p in (part_names or ()) if str(p).strip()
            )
        parsed_kp_map = _parse_keypoint_part_map(keypoint_part_map)
        cfg = BodyPartSegmentationConfig(
            mode=mode,
            feature_backend=feature_backend,
            model_name=model_name,
            crop_size=crop_size,
            crop_padding_ratio=crop_padding_ratio,
            num_parts=num_parts,
            part_names=parsed_names,
            keypoint_granularity=keypoint_granularity,
            keypoint_part_map=parsed_kp_map,
            partition_full_instance=partition_full_instance,
            share_seed_across_instances=share_seed_across_instances,
            max_part_area_growth=max_part_area_growth,
            min_part_area_px=min_part_area_px,
            boundary_refine=boundary_refine,
            crf_refine=crf_refine,
            polygon_epsilon=polygon_epsilon,
            split_merged_parent_masks=split_merged_parent_masks,
            recover_missing_merged_instances=recover_missing_merged_instances,
            qualify_label_with_instance=qualify_label_with_instance,
            persist_json=persist_json,
            device=device,
        )
        report = run_body_part_segmentation(
            video_path,
            annotation_dir=annotation_dir,
            config=cfg,
            pose_schema_path=pose_schema_path,
            start_frame=start_frame,
            end_frame=end_frame,
            step=step,
            export_csv=export_csv,
            export_ethogram=export_ethogram,
            camera_view=camera_view,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    if out_json is not None:
        atomic_write_json(Path(out_json), report, indent=2, ensure_ascii=False)

    if json_stdout:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(report.get("message", "Body-part segmentation completed."))
        part_counts = report.get("part_counts") or {}
        if part_counts:
            formatted = ", ".join(f"{k}={v}" for k, v in part_counts.items())
            print(f"Parts: {formatted}")
        if report.get("tracking_csv"):
            print(f"Tracking CSV: {report['tracking_csv']}")

    if int(report.get("frames_processed", 0) or 0) == 0:
        return 2
    return 0
