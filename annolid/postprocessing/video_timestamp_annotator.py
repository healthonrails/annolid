import subprocess
import logging
import math
import os
import tempfile
from pathlib import Path
import argparse
import pandas as pd
import json
from typing import List, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".flv",
    ".mpeg",
    ".mpg",
    ".m4v",
    ".mts",
}
CSV_EXTENSION = ".csv"


def extract_frame_timestamps(video_path: Path) -> List[float]:
    """
    Use ffprobe with JSON output to extract frame presentation timestamps (PTS) reliably.
    Supports both 'pts_time' (newer FFmpeg versions) and 'pkt_pts_time' (older versions).
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pts_time,pkt_pts_time",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parsing error: {e}")

    timestamps: List[float] = []
    has_finite = False
    for frame in data.get("frames", []):
        # Prefer pts_time (newer FFmpeg), fall back to pkt_pts_time (older FFmpeg)
        ts = frame.get("pts_time")
        if ts is None or ts == "":
            ts = frame.get("pkt_pts_time")
        if ts is not None and ts != "":
            try:
                parsed = float(ts)
            except (TypeError, ValueError):
                parsed = float("nan")
            if math.isfinite(parsed):
                has_finite = True
            timestamps.append(parsed)
        else:
            warnings.warn(
                f"No valid timestamp (pts_time or pkt_pts_time) found in frame: {frame}"
            )
            timestamps.append(float("nan"))
    if not has_finite:
        warnings.warn(
            "No valid timestamps extracted. Check FFmpeg version or video input."
        )
        return []
    return timestamps


def find_assets(root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Recursively locate video and CSV files under a root directory.

    Args:
        root: Top-level directory to scan.

    Returns:
        Tuple of (video_files, csv_files).
    """
    video_files = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
    csv_files = [p for p in root.rglob(f"*{CSV_EXTENSION}")]
    return video_files, csv_files


def match_video_for_csv(csv_path: Path, video_files: List[Path]) -> Optional[Path]:
    """
    Match a CSV file to a video, preferring same-directory exact/prefix stem matches
    and longest video stems first so ``trial_1`` never shadows ``trial_10``.
    """
    csv_key = csv_path.stem.lower()
    csv_parent = csv_path.parent
    ordered = sorted(
        video_files,
        key=lambda p: (p.parent == csv_parent, len(p.stem)),
        reverse=True,
    )
    for video in ordered:
        v_stem = video.stem.lower()
        if (
            csv_key == v_stem
            or csv_key.startswith(f"{v_stem}_")
            or csv_key.startswith(f"{v_stem}-")
        ):
            return video
    for video in ordered:
        if video.stem.lower() in csv_key:
            return video
    return None


def find_frame_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find a column in df corresponding to frame number, handling variations.
    Returns the original column name if found, else None.
    """
    for col in df.columns:
        normalized = col.strip().lower().replace(" ", "_")
        if normalized == "frame_number":
            return col
    return None


def annotate_csv(csv_path: Path, video_path: Path) -> None:
    """
    Load a tracking CSV, append real frame timestamps, and save a new annotated CSV.

    If no frame column, logs a warning and skips.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read {csv_path.name}: {e}")
        return

    frame_col = find_frame_column(df)
    if not frame_col:
        logger.warning(f"Skipping {csv_path.name}: no frame_number column found.")
        return

    try:
        timestamps = extract_frame_timestamps(video_path)
    except Exception as e:
        logger.error(f"Failed to extract timestamps for {video_path.name}: {e}")
        return

    if df.empty:
        df["real_timestamp_sec"] = pd.Series(dtype="float64")
    else:
        numeric_frames = pd.to_numeric(df[frame_col], errors="coerce")
        valid_frames = numeric_frames.dropna()
        if valid_frames.empty:
            logger.warning(
                f"Skipping {csv_path.name}: {frame_col} contains no valid integers."
            )
            return
        max_frame = int(valid_frames.max())
        if max_frame >= len(timestamps):
            logger.warning(
                f"{csv_path.name}: Frame index {max_frame} exceeds available timestamps ({len(timestamps)}). "
                "Truncating unavailable timestamps."
            )
        elif len(timestamps) > max_frame + 1:
            logger.info(
                f"{csv_path.name}: Video has {len(timestamps)} frame timestamps but CSV uses only {max_frame + 1} frames. "
                "Consider checking for dropped or skipped frames."
            )

        def safe_lookup(val):
            if pd.isna(val):
                return float("nan")
            idx = int(val)
            if 0 <= idx < len(timestamps):
                return timestamps[idx]
            return float("nan")

        df["real_timestamp_sec"] = numeric_frames.map(safe_lookup)

    tmp_path: Optional[Path] = None
    try:
        fd, raw_tmp = tempfile.mkstemp(
            prefix=f".{csv_path.name}.",
            suffix=".tmp",
            dir=str(csv_path.parent),
        )
        os.close(fd)
        tmp_path = Path(raw_tmp)
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)
        tmp_path = None
        logger.info(f"Annotated CSV saved: {csv_path.name}")
    except Exception as e:
        logger.error(f"Failed to write {csv_path.name}: {e}")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    try:
        # run gap analysis for the video
        run_gap_analysis_for_video(str(video_path))
    except Exception as e:
        logger.error(f"Gap analysis failed for {video_path.name}: {e}")


def run_gap_analysis_for_video(video_path: str) -> bool:
    """
    Analyzes a video's tracking results to find and report gaps in tracked instances.

    This function serves as a single, convenient entry point. It takes a video file,
    infers the associated JSON results directory, calculates tracking gaps based on the
    video's true frame count, and saves both a human-readable Markdown report and a
    machine-readable CSV file in the results directory.

    Args:
        video_path (str): The full path to the video file to be analyzed.

    Returns:
        bool: True if the reports were successfully generated, False otherwise.
    """
    from annolid.postprocessing.tracking_reports import (
        find_tracking_gaps,
        generate_reports,
    )

    try:
        # --- Core Logic: Find Gaps ---
        gaps = find_tracking_gaps(video_path)

        # --- Reporting ---
        generate_reports(gaps, video_path)

        logger.info("Gap analysis completed successfully.")
        return True
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during analysis: {e}", exc_info=True
        )
        return False


def process_directory(root: Path) -> None:
    """
    Recursively process all CSVs, matching them to videos and annotating.
    """
    video_files, csv_files = find_assets(root)
    if not video_files:
        logger.warning("No video files found.")
    if not csv_files:
        logger.warning("No CSV files found.")

    for csv_path in csv_files:
        video_path = match_video_for_csv(csv_path, video_files)
        if not video_path:
            logger.warning(f"No matching video for {csv_path.name}")
            continue
        annotate_csv(csv_path, video_path)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate tracking CSVs with real frame timestamps."
    )
    parser.add_argument(
        "root_folder", type=Path, help="Root folder containing videos and CSVs"
    )
    args = parser.parse_args()

    if not args.root_folder.is_dir():
        logger.error(f"Not a directory: {args.root_folder}")
        return

    process_directory(args.root_folder)


if __name__ == "__main__":
    main()
