import subprocess
import logging
from pathlib import Path
import argparse
import pandas as pd
import json
from typing import List, Optional, Tuple
import warnings

# Configure logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov',
                    '.flv', '.mpeg', '.mpg', '.m4v', '.mts'}
CSV_EXTENSION = '.csv'


def extract_frame_timestamps(video_path: Path) -> List[float]:
    """
    Use ffprobe with JSON output to extract frame presentation timestamps (PTS) reliably.
    Supports both 'pts_time' (newer FFmpeg versions) and 'pkt_pts_time' (older versions).
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_frames',
        '-show_entries', 'frame=pts_time,pkt_pts_time',
        '-of', 'json', str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parsing error: {e}")

    timestamps = []
    for frame in data.get("frames", []):
        # Prefer pts_time (newer FFmpeg), fall back to pkt_pts_time (older FFmpeg)
        ts = frame.get("pts_time") or frame.get("pkt_pts_time")
        if ts is not None:
            try:
                timestamps.append(float(ts))
            except ValueError:
                continue
        else:
            warnings.warn(
                f"No valid timestamp (pts_time or pkt_pts_time) found in frame: {frame}")
    if not timestamps:
        warnings.warn(
            "No valid timestamps extracted. Check FFmpeg version or video input.")
    return timestamps


def find_assets(root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Recursively locate video and CSV files under a root directory.

    Args:
        root: Top-level directory to scan.

    Returns:
        Tuple of (video_files, csv_files).
    """
    video_files = [p for p in root.rglob(
        '*') if p.suffix.lower() in VIDEO_EXTENSIONS]
    csv_files = [p for p in root.rglob(f'*{CSV_EXTENSION}')]
    return video_files, csv_files


def match_video_for_csv(csv_path: Path, video_files: List[Path]) -> Optional[Path]:
    """
    Match a CSV file to a video by checking if video stem is substring of CSV stem.
    """
    csv_key = csv_path.stem.lower()
    for video in video_files:
        if video.stem.lower() in csv_key:
            return video
    return None


def find_frame_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find a column in df corresponding to frame number, handling variations.
    Returns the original column name if found, else None.
    """
    for col in df.columns:
        normalized = col.strip().lower().replace(' ', '_')
        if normalized == 'frame_number':
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
        logger.warning(
            f"Skipping {csv_path.name}: no frame_number column found.")
        return

    try:
        timestamps = extract_frame_timestamps(video_path)
    except Exception as e:
        logger.error(
            f"Failed to extract timestamps for {video_path.name}: {e}")
        return

    max_frame = int(df[frame_col].max())
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
    # Map frame numbers to timestamps with bounds check

    def safe_lookup(i):
        if 0 <= i < len(timestamps):
            return timestamps[i]
        else:
            return float('nan')

    df['real_timestamp_sec'] = df[frame_col].astype(int).map(safe_lookup)

    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"Annotated CSV saved: {csv_path.name}")
    except Exception as e:
        logger.error(f"Failed to write {csv_path.name}: {e}")


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
        description="Annotate tracking CSVs with real frame timestamps.")
    parser.add_argument('root_folder', type=Path,
                        help='Root folder containing videos and CSVs')
    args = parser.parse_args()

    if not args.root_folder.is_dir():
        logger.error(f"Not a directory: {args.root_folder}")
        return

    process_directory(args.root_folder)


if __name__ == '__main__':
    main()
