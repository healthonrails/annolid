"""Convert DeepLabCut / Lightning Pose analysis CSV or H5 files to LabelMe JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from annolid.annotation.pose_interop.converters import import_pose_to_labelme
from annolid.annotation.pose_interop.deeplabcut import find_dlc_analysis_file_for_video


def deeplabcut_to_labelme_json(
    video_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    is_multi_animal: Optional[bool] = None,
    *,
    csv_path: Optional[str | Path] = None,
    min_likelihood: Optional[float] = None,
    write_pose_schema: bool = True,
    pose_schema_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    default_instance: str = "mouse",
    filename_prefix: Optional[str] = None,
    prefix_labels_with_instance: bool = False,
    save_frames: bool = False,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Convert DeepLabCut / Lightning Pose CSV or H5 output to LabelMe JSON files.

    Args:
        video_path: Optional path to the video file. When ``csv_path`` is omitted,
            Annolid looks for ``<video_stem>.csv``, ``<video_stem>DLC_*.csv``,
            ``<video_stem>.h5``, or ``<video_stem>DLC_*.h5`` beside the video.
        output_dir: Directory to write LabelMe JSON files. Defaults to
            ``<video_stem>/`` (or ``<csv_stem>_labelme/`` when no video is supplied).
        is_multi_animal: Optional hint; single- vs multi-animal headers (3 vs 4 rows)
            are auto-detected from the file.
        csv_path: Explicit path to a DeepLabCut / Lightning Pose ``.csv`` or ``.h5`` file.
        min_likelihood: Optional minimum likelihood threshold (points below are skipped).
        write_pose_schema: Write ``pose_schema.json`` into ``output_dir``.
        pose_schema_path: Optional existing ``PoseSchema`` to map keypoint names against.
        config_path: Optional DeepLabCut ``config.yaml`` for skeleton edges & ordering.
        default_instance: Instance label used for single-animal files (default: ``"mouse"``).
        filename_prefix: Optional filename prefix for generated JSON files
            (e.g. ``video_stem`` -> ``<video_stem>_000000000.json``).
        prefix_labels_with_instance: If True, write shape labels as ``<instance>_<bodypart>``.
        save_frames: If True and ``video_path`` is provided, also extract sidecar PNG frames.
        overwrite: Overwrite existing JSON files (default: True).

    Returns:
        Structured summary dict with counts, resolved paths, keypoints, and instances.
    """
    resolved_video = Path(video_path).expanduser().resolve() if video_path else None
    resolved_table: Optional[Path] = (
        Path(csv_path).expanduser().resolve() if csv_path else None
    )

    if resolved_table is None:
        if resolved_video is None:
            raise ValueError("Either video_path or csv_path must be provided.")
        resolved_table = find_dlc_analysis_file_for_video(resolved_video)
        if resolved_table is None:
            expected_csv = resolved_video.with_suffix(".csv")
            raise FileNotFoundError(
                f"DeepLabCut analysis CSV/H5 not found for video '{resolved_video.name}'. "
                f"Expected '{expected_csv.name}' or '{resolved_video.stem}DLC_*.csv/.h5' "
                "in the same folder, or pass csv_path explicitly."
            )
    elif not resolved_table.is_file():
        raise FileNotFoundError(f"DeepLabCut analysis file not found: {resolved_table}")

    if output_dir is None:
        if resolved_video is not None:
            resolved_out = resolved_video.with_suffix("")
        else:
            resolved_out = resolved_table.parent / f"{resolved_table.stem}_labelme"
    else:
        resolved_out = Path(output_dir).expanduser().resolve()

    summary = import_pose_to_labelme(
        resolved_table,
        resolved_out,
        source_format="auto",
        video_path=resolved_video,
        pose_schema_path=pose_schema_path,
        config_path=config_path,
        save_frames=save_frames,
        min_likelihood=min_likelihood,
        is_multi_animal=is_multi_animal,
        default_instance=default_instance,
        write_pose_schema=write_pose_schema,
        pose_schema_preset="auto",
        filename_prefix=filename_prefix,
        prefix_labels_with_instance=prefix_labels_with_instance,
        overwrite=overwrite,
    )
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert DeepLabCut / Lightning Pose analysis CSV or H5 to LabelMe JSON."
    )
    parser.add_argument(
        "--csv",
        "--input",
        dest="csv_path",
        default=None,
        help="Path to DeepLabCut/Lightning Pose .csv or .h5 analysis file.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Optional path to matching video file (used for image size & auto-discovery).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for LabelMe JSON files.",
    )
    parser.add_argument(
        "--multi-animal",
        action="store_true",
        default=None,
        help="Hint that the input is multi-animal (auto-detected by default).",
    )
    parser.add_argument(
        "--min-likelihood",
        type=float,
        default=None,
        help="Drop keypoints with likelihood below this threshold.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional DeepLabCut config.yaml for skeleton edges.",
    )
    parser.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema JSON/YAML for canonical keypoint name mapping.",
    )
    parser.add_argument(
        "--instance-label",
        default="mouse",
        help="Default instance label for single-animal files (default: mouse).",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Extract matching PNG frames from --video alongside LabelMe JSON files.",
    )
    parser.add_argument(
        "--no-pose-schema",
        action="store_true",
        help="Do not write pose_schema.json into the output directory.",
    )
    args = parser.parse_args(argv)

    try:
        summary = deeplabcut_to_labelme_json(
            video_path=args.video,
            output_dir=args.out,
            is_multi_animal=args.multi_animal,
            csv_path=args.csv_path,
            min_likelihood=args.min_likelihood,
            write_pose_schema=not args.no_pose_schema,
            pose_schema_path=args.pose_schema,
            config_path=args.config,
            default_instance=args.instance_label,
            save_frames=args.save_frames,
        )
        print(
            f"[deeplabcut2labelme] wrote {summary['json_written']} LabelMe JSONs to: {summary['output_dir']}"
        )
        return 0
    except Exception as exc:
        print(f"[deeplabcut2labelme] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
