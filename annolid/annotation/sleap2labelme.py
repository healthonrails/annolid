#!/usr/bin/env python3
"""Convert SLEAP HDF5 files (.slp, .pkg.slp, .analysis.h5) to Annolid LabelMe JSON + PoseSchema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from annolid.annotation.pose_interop.converters import import_pose_to_labelme
from annolid.annotation.pose_interop.labelme import build_label_mapper
from annolid.annotation.pose_interop.sleap import (
    HDF5_MAGIC,
    INSTANCE_TYPE_PREDICTED,
    INSTANCE_TYPE_USER,
    InstanceRecord,
    SkeletonSpec,
    SleapH5Reader,
    ZIP_MAGIC,
    detect_sleap_container,
    detect_sleap_h5_kind,
    safe_json_loads,
)
from annolid.annotation.pose_schema import PoseSchema
from annolid.utils.annotation_compat import __version__ as LABELME_VERSION


def make_labelme_json(
    image_relpath: str, image_size: Tuple[int, int], shapes: List[dict]
) -> dict:
    w, h = image_size
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_relpath,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }


def _infer_symmetry_pairs_extended(keypoints: Sequence[str]) -> List[Tuple[str, str]]:
    try:
        return list(PoseSchema.infer_symmetry_pairs(keypoints))
    except Exception:
        return []


def write_pose_schema(out_dir: Path, skeleton: SkeletonSpec, *, filename: str) -> Path:
    """Write pose schema JSON to ``out_dir / filename``."""
    sym = skeleton.symmetry_pairs or _infer_symmetry_pairs_extended(skeleton.names)
    schema = PoseSchema(
        version="1.0",
        keypoints=list(skeleton.names),
        edges=list(skeleton.edges) if skeleton.edges else [],
        symmetry_pairs=sym,
        flip_idx=None,
        instances=[],
        instance_separator="_",
    )
    out_path = Path(out_dir) / filename
    schema.save(out_path)
    return out_path


def convert_sleap_h5_to_labelme_summary(
    sleap_path: str | Path,
    out_dir: str | Path,
    *,
    save_frames: bool = True,
    video_index: Optional[int] = None,
    print_every: int = 200,
    pose_schema_path: Optional[str | Path] = None,
    video_path: Optional[str | Path] = None,
    prefer_user_instances: bool = True,
    include_predicted: bool = True,
    include_occluded: bool = True,
    min_score: Optional[float] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Convert SLEAP ``.slp`` / ``.pkg.slp`` / ``.analysis.h5`` to LabelMe JSON and return summary."""
    del print_every  # kept for signature compatibility
    src_path = Path(sleap_path).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sleap_kind = detect_sleap_h5_kind(src_path)
    summary = import_pose_to_labelme(
        src_path,
        out_root,
        source_format="sleap_analysis_h5"
        if sleap_kind == "analysis_h5"
        else "sleap_slp",
        video_path=video_path,
        pose_schema_path=pose_schema_path,
        save_frames=save_frames,
        video_index=video_index,
        min_likelihood=min_score,
        prefer_user_instances=prefer_user_instances,
        include_predicted=include_predicted,
        include_occluded=include_occluded,
        write_pose_schema=True,
        pose_schema_preset="auto",
        overwrite=overwrite,
    )

    # Preserve multi-skeleton pose_schema_<i>.json exports for .slp files with >1 skeleton.
    schema_files: List[str] = []
    if summary.get("pose_schema"):
        schema_files.append(str(summary["pose_schema"]))
    if sleap_kind == "slp":
        with SleapH5Reader(src_path) as reader:
            skeletons = reader.read_skeletons()
            if len(skeletons) > 1:
                for idx, sk in enumerate(skeletons):
                    p = write_pose_schema(
                        out_root, sk, filename=f"pose_schema_{idx}.json"
                    )
                    schema_files.append(str(p))

    summary["sleap_kind"] = sleap_kind
    summary["pose_schemas"] = schema_files
    return summary


def convert_sleap_h5_to_labelme(
    sleap_path: str | Path,
    out_dir: str | Path,
    *,
    save_frames: bool = True,
    video_index: Optional[int] = None,
    print_every: int = 200,
    pose_schema_path: Optional[str | Path] = None,
    video_path: Optional[str | Path] = None,
    prefer_user_instances: bool = True,
    include_predicted: bool = True,
    include_occluded: bool = True,
    min_score: Optional[float] = None,
    overwrite: bool = True,
) -> Path:
    """Convert SLEAP ``.slp`` / ``.pkg.slp`` / ``.analysis.h5`` to LabelMe JSON and return ``out_dir``."""
    summary = convert_sleap_h5_to_labelme_summary(
        sleap_path,
        out_dir,
        save_frames=save_frames,
        video_index=video_index,
        print_every=print_every,
        pose_schema_path=pose_schema_path,
        video_path=video_path,
        prefer_user_instances=prefer_user_instances,
        include_predicted=include_predicted,
        include_occluded=include_occluded,
        min_score=min_score,
        overwrite=overwrite,
    )
    print(
        f"[sleap2labelme] wrote {summary['json_written']} LabelMe JSONs to: {summary['output_dir']}"
    )
    if save_frames and summary.get("frames_extracted"):
        print(
            f"[sleap2labelme] saved {summary['frames_extracted']} frames to: {summary['output_dir']}"
        )
    return Path(summary["output_dir"])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert SLEAP (.slp, .pkg.slp, .analysis.h5) to Annolid LabelMe JSON + PoseSchema."
    )
    ap.add_argument(
        "--sleap",
        required=True,
        help="Path to .slp, .pkg.slp, or SLEAP .analysis.h5 file",
    )
    ap.add_argument(
        "--out", required=True, help="Output directory for LabelMe JSON + images"
    )
    ap.add_argument(
        "--video",
        default=None,
        help="Optional external video path (used for non-packaged .slp or .analysis.h5).",
    )
    ap.add_argument(
        "--no-save-frames", action="store_true", help="Do not extract frames."
    )
    ap.add_argument(
        "--video-index",
        type=int,
        default=None,
        help="Optional: process a single video index (video0, video1, ...) for multi-video .slp files.",
    )
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument(
        "--pose-schema",
        default=None,
        help="Optional: pose schema JSON/YAML to map skeleton names.",
    )
    ap.add_argument(
        "--user-only",
        action="store_true",
        help="Skip predicted instances and import user-labeled instances only.",
    )
    ap.add_argument(
        "--keep-both-user-and-predicted",
        action="store_true",
        help="Keep predicted instances even on frames that already have user labels.",
    )
    ap.add_argument(
        "--skip-occluded",
        action="store_true",
        help="Drop occluded keypoints (visible=False with finite coordinates).",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum confidence score for predicted keypoints.",
    )
    args = ap.parse_args(argv)

    try:
        convert_sleap_h5_to_labelme(
            args.sleap,
            args.out,
            save_frames=not args.no_save_frames,
            video_index=args.video_index,
            print_every=args.print_every,
            pose_schema_path=args.pose_schema,
            video_path=args.video,
            prefer_user_instances=not args.keep_both_user_and_predicted,
            include_predicted=not args.user_only,
            include_occluded=not args.skip_occluded,
            min_score=args.min_score,
        )
        return 0
    except Exception as e:
        print(f"[sleap2labelme] ERROR: {e}", file=sys.stderr)
        return 2


__all__ = [
    "HDF5_MAGIC",
    "INSTANCE_TYPE_PREDICTED",
    "INSTANCE_TYPE_USER",
    "InstanceRecord",
    "SkeletonSpec",
    "SleapH5Reader",
    "ZIP_MAGIC",
    "build_label_mapper",
    "convert_sleap_h5_to_labelme",
    "convert_sleap_h5_to_labelme_summary",
    "detect_sleap_container",
    "detect_sleap_h5_kind",
    "make_labelme_json",
    "safe_json_loads",
    "write_pose_schema",
]


if __name__ == "__main__":
    raise SystemExit(main())
