import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime
import re
from annolid.utils.logger import logger

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning(
        "OpenCV not found. Video metadata (frame count) cannot be read. "
        "Analysis will be based on existing JSON files only."
    )


def _tracking_csv_for_video(video_file: Path) -> Path | None:
    """Return a tracking CSV with frame/instance rows for the video, if present."""
    candidates = [
        video_file.with_name(f"{video_file.stem}_tracking.csv"),
        video_file.with_name(f"{video_file.stem}_tracked.csv"),
    ]
    for csv_path in candidates:
        if not csv_path.is_file():
            continue
        try:
            columns = set(pd.read_csv(csv_path, nrows=0).columns)
        except Exception as exc:
            logger.warning("Could not inspect tracking CSV %s: %s", csv_path, exc)
            continue
        if {"frame_number", "instance_name"}.issubset(columns):
            return csv_path
    return None


def _presence_from_tracking_csv(csv_path: Path) -> tuple[dict[int, set], set, int]:
    usecols = ["frame_number", "instance_name"]
    df = pd.read_csv(csv_path, usecols=usecols)
    if df.empty:
        return {}, set(), -1

    df = df.dropna(subset=usecols)
    df["frame_number"] = df["frame_number"].astype(int)
    frame_presence = {
        int(frame_number): set(group["instance_name"].astype(str))
        for frame_number, group in df.groupby("frame_number", sort=True)
    }
    all_instance_labels = set(df["instance_name"].astype(str))
    return frame_presence, all_instance_labels, int(df["frame_number"].max())


def _resolve_gap_inputs(
    source_path: str | Path,
) -> tuple[Path, Path | None, Path, str]:
    """Normalize a video, tracking CSV, or frame JSON directory path for gap analysis."""
    raw = Path(source_path)
    if raw.is_file() and raw.suffix.lower() == ".csv":
        stem = raw.stem
        for suffix in ("_tracking", "_tracked"):
            if stem.endswith(suffix) and len(stem) > len(suffix):
                stem = stem[: -len(suffix)]
                break
        json_dir = raw.parent / stem
        try:
            cols = set(pd.read_csv(raw, nrows=0).columns)
        except Exception:
            cols = set()
        csv_path = raw if {"frame_number", "instance_name"}.issubset(cols) else None
        return raw, csv_path, json_dir, stem

    if raw.is_dir():
        stem = raw.name
        pseudo_video = raw.parent / f"{stem}.mp4"
        csv_path = _tracking_csv_for_video(pseudo_video)
        return pseudo_video, csv_path, raw, stem

    csv_path = _tracking_csv_for_video(raw)
    json_dir = raw.with_suffix("")
    return raw, csv_path, json_dir, raw.stem


def find_tracking_gaps(video_path: str) -> dict:
    """
    Scans for tracking gaps based on a video file, tracking CSV, or frame JSON directory.

    Args:
        video_path (str): Path to the video file, tracking CSV, or frame JSON directory.

    Returns:
        A dictionary where keys are instance labels and values are lists of gap tuples.
    """
    video_file, tracking_csv, json_directory, _ = _resolve_gap_inputs(video_path)
    has_video_file = video_file.is_file() and video_file.suffix.lower() != ".csv"
    if not has_video_file and tracking_csv is None and not json_directory.is_dir():
        logger.error(f"Video or tracking data not found: {video_file}")
        return {}

    if tracking_csv is None and not json_directory.is_dir():
        logger.error(f"Associated JSON directory not found at: {json_directory}")
        return {}

    logger.info(f"Analyzing source: {Path(video_path).name}")
    if tracking_csv is not None:
        logger.info(f"Reading tracking CSV from: {tracking_csv}")
    else:
        logger.info(f"Reading JSON files from: {json_directory}")

    # --- 1. Determine the Full Frame Range (The Ground Truth) ---
    min_frame = 0
    max_frame = -1
    if OPENCV_AVAILABLE and has_video_file:
        cap = cv2.VideoCapture(str(video_file))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count > 0:
                max_frame = frame_count - 1
                logger.info(f"Video metadata: Total frames = {max_frame + 1}.")
            else:
                logger.warning(
                    "Video metadata reported 0 frames for %s; inferring frame range from annotations.",
                    video_file,
                )
        else:
            logger.warning(
                "Could not open video file %s with OpenCV; falling back to CSV/JSON frame range.",
                video_file,
            )

    if max_frame < 0 and tracking_csv is not None:
        _, _, max_frame = _presence_from_tracking_csv(tracking_csv)
        if max_frame < 0:
            logger.warning("No tracking rows found to infer frame range.")
            return {}
        logger.info("Inferred max frame is %s from tracking CSV.", max_frame)
    elif max_frame < 0:
        if not json_directory.is_dir():
            logger.error(f"Could not determine frame range for {video_file}")
            return {}
        # Fallback if OpenCV is unavailable or video metadata unreadable: infer range from JSON files or AnnotationStore.
        from annolid.utils.annotation_store import AnnotationStore

        frame_candidates: list[int] = []
        for p in json_directory.glob("*.json"):
            if p.name.endswith("_tracking_stats.json") or p.name.endswith(
                "_zones.json"
            ):
                continue
            m = re.search(r"(\d+)(?=\.json$)", p.name)
            if m:
                frame_candidates.append(int(m.group(1)))
        try:
            store = AnnotationStore.for_frame_path(
                json_directory / f"{json_directory.name}_000000000.json"
            )
            if store.store_path.is_file():
                frame_candidates.extend(int(f) for f in store.iter_frames())
        except Exception:
            pass
        if not frame_candidates:
            logger.warning("No JSON files found to infer frame range.")
            return {}
        max_frame = max(frame_candidates)
        logger.warning(f"Inferred max frame is {max_frame} from JSON files.")

    master_frame_index = pd.RangeIndex(
        start=min_frame, stop=max_frame + 1, name="frame_number"
    )

    # --- 2. Scan and Parse Existing Files ---
    if tracking_csv is not None:
        frame_presence, all_instance_labels, _ = _presence_from_tracking_csv(
            tracking_csv
        )
    else:
        from annolid.utils.annotation_store import AnnotationStore
        from annolid.utils.body_part_shapes import (
            format_shape_csv_tracking_name,
            is_body_part_shape_payload,
        )
        from annolid.utils.zone_shapes import is_zone_shape_payload

        frame_presence: dict[int, set] = {}
        all_instance_labels = set()

        def _extract_shape_names(shapes_list) -> set:
            names = set()
            for shape in shapes_list or []:
                if not isinstance(shape, dict) or is_zone_shape_payload(shape):
                    continue
                if is_body_part_shape_payload(shape):
                    tname = format_shape_csv_tracking_name(shape)
                    if tname:
                        names.add(tname)
                elif (
                    "label" in shape
                    and shape["label"] is not None
                    and shape["label"] != ""
                ):
                    names.add(shape["label"])
            return names

        for file_path in json_directory.glob("*.json"):
            if file_path.name.endswith(
                "_tracking_stats.json"
            ) or file_path.name.endswith("_zones.json"):
                continue
            try:
                match = re.search(r"(\d+)(?=\.json$)", file_path.name)
                if not match:
                    continue

                frame_number = int(match.group(1))
                with open(file_path, "r") as f:
                    data = json.load(f)

                labels_in_frame = _extract_shape_names(data.get("shapes", []))
                frame_presence.setdefault(frame_number, set()).update(labels_in_frame)
                all_instance_labels.update(labels_in_frame)
            except (ValueError, IndexError, json.JSONDecodeError) as e:
                logger.warning(f"Could not parse file {file_path.name}: {e}")

        try:
            store = AnnotationStore.for_frame_path(
                json_directory / f"{json_directory.name}_000000000.json"
            )
            if store.store_path.is_file():
                for fno in store.iter_frames():
                    frame_number = int(fno)
                    rec = store.get_frame(frame_number)
                    if not isinstance(rec, dict):
                        continue
                    labels_in_frame = _extract_shape_names(rec.get("shapes", []))
                    frame_presence.setdefault(frame_number, set()).update(
                        labels_in_frame
                    )
                    all_instance_labels.update(labels_in_frame)
        except Exception as e:
            logger.warning(f"Could not read AnnotationStore in {json_directory}: {e}")

    if not all_instance_labels:
        logger.info("No labeled instances found in any JSON file.")
        return {}

    sorted_labels = sorted(list(all_instance_labels), key=lambda x: str(x))

    # --- 3. Build Presence DataFrame and Merge with Master Timeline ---
    if not frame_presence:
        df_presence_sparse = pd.DataFrame(
            columns=["frame_number"] + sorted_labels
        ).set_index("frame_number")
    else:
        records = []
        for frame_number in sorted(frame_presence):
            labels_in_frame = frame_presence[frame_number]
            frame_record = {"frame_number": frame_number}
            for label in sorted_labels:
                frame_record[label] = label in labels_in_frame
            records.append(frame_record)
        df_presence_sparse = pd.DataFrame(records).set_index("frame_number")

    # Reindex directly with a fill value so we avoid DataFrame.fillna downcasting
    # warnings on object-dtype columns in newer pandas versions.
    df_full = df_presence_sparse.reindex(master_frame_index, fill_value=False)

    # --- 4. Identify Gaps for Each Animal ---
    gap_report = {}
    for label in sorted_labels:
        # Be defensive about non-bool column dtypes (e.g. ints from legacy CSV/JSON):
        # using `~series` on integer values yields -1/-2 and breaks DataFrame indexing.
        present_mask = df_full[label].eq(True)
        missing_mask = ~present_mask
        missing_frames = df_full.loc[missing_mask].copy()
        if missing_frames.empty:
            continue

        gaps = []
        # Find contiguous blocks by checking where the frame number difference is > 1
        missing_frames["gap_block"] = (
            missing_frames.index.to_series().diff() > 1
        ).cumsum()

        for _, group in missing_frames.groupby("gap_block"):
            start_frame = group.index.min()
            end_frame = group.index.max()
            duration = end_frame - start_frame + 1
            gaps.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": duration,
                }
            )

        if gaps:
            gap_report[label] = gaps

    return gap_report


def summarize_tracking_gaps(
    video_path: str | Path,
    *,
    fps: float | None = None,
) -> dict:
    """Return structured per-instance coverage and gap statistics for a video, CSV, or directory."""
    video_file, tracking_csv, json_dir, _ = _resolve_gap_inputs(video_path)
    gaps_by_label = find_tracking_gaps(str(video_path))
    total_frames = 0
    eff_fps = float(fps) if fps and fps > 0 else None
    has_video_file = video_file.is_file() and video_file.suffix.lower() != ".csv"
    if OPENCV_AVAILABLE and has_video_file:
        cap = cv2.VideoCapture(str(video_file))
        if cap.isOpened():
            total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            if eff_fps is None:
                raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if raw_fps > 0:
                    eff_fps = raw_fps
            cap.release()

    all_labels: set[str] = set(str(k) for k in gaps_by_label.keys())
    if tracking_csv is not None:
        _, csv_labels, max_f = _presence_from_tracking_csv(tracking_csv)
        all_labels.update(str(k) for k in csv_labels)
        if total_frames <= 0 and max_f >= 0:
            total_frames = max_f + 1
    elif json_dir.is_dir():
        max_f = -1
        for p in json_dir.glob("*.json"):
            if p.name.endswith("_tracking_stats.json") or p.name.endswith(
                "_zones.json"
            ):
                continue
            m = re.search(r"(\d+)(?=\.json$)", p.name)
            if m:
                max_f = max(max_f, int(m.group(1)))
                try:
                    payload = json.loads(p.read_text(encoding="utf-8"))
                    for sh in payload.get("shapes", []) or []:
                        if isinstance(sh, dict) and sh.get("label"):
                            all_labels.add(str(sh["label"]))
                except Exception:
                    pass
        if total_frames <= 0 and max_f >= 0:
            total_frames = max_f + 1

    instances_summary = []
    total_gap_events = 0
    for label in sorted(all_labels):
        label_gaps = gaps_by_label.get(
            label, gaps_by_label.get(_coerce_gap_key(gaps_by_label, label), [])
        )
        missing_frames = sum(int(g.get("duration_frames", 0)) for g in label_gaps)
        total_gap_events += len(label_gaps)
        eff_total = max(total_frames, missing_frames)
        observed_frames = max(0, eff_total - missing_frames)
        coverage_ratio = (
            float(observed_frames) / float(eff_total) if eff_total > 0 else 1.0
        )
        longest_gap = max(
            (int(g.get("duration_frames", 0)) for g in label_gaps), default=0
        )
        enriched_gaps = []
        for g in label_gaps:
            sf = int(g["start_frame"])
            ef = int(g["end_frame"])
            dur = int(g["duration_frames"])
            enriched_gaps.append(
                {
                    "start_frame": sf,
                    "end_frame": ef,
                    "duration_frames": dur,
                    "duration_seconds": (round(dur / eff_fps, 4) if eff_fps else None),
                }
            )
        instances_summary.append(
            {
                "instance": label,
                "observed_frames": observed_frames,
                "missing_frames": missing_frames,
                "coverage_ratio": round(coverage_ratio, 6),
                "gap_count": len(label_gaps),
                "longest_gap_frames": longest_gap,
                "longest_gap_seconds": (
                    round(longest_gap / eff_fps, 4) if eff_fps else None
                ),
                "gaps": enriched_gaps,
            }
        )

    return {
        "schema": "tracking_gap_report/v1",
        "video_path": str(Path(video_path)),
        "total_frames": int(total_frames),
        "fps": round(eff_fps, 4) if eff_fps else None,
        "total_gap_events": int(total_gap_events),
        "instances": instances_summary,
    }


def _coerce_gap_key(gaps_by_label: dict, label: str):
    if label in gaps_by_label:
        return label
    for key in gaps_by_label:
        if str(key) == label:
            return key
    return label


def generate_reports(gap_report: dict, video_path: str):
    """
    Generates and saves both a human-readable Markdown report and a machine-readable CSV report.
    """
    _, _, json_dir, stem = _resolve_gap_inputs(video_path)
    output_path = json_dir.parent / stem

    # --- 1. Generate Machine-Readable CSV Report ---
    csv_records = []
    for label, gaps in gap_report.items():
        for gap in gaps:
            record = {"instance_label": label, **gap}
            csv_records.append(record)

    csv_filename = output_path.parent / f"{output_path.stem}_gaps_report.csv"
    if csv_records:
        csv_df = pd.DataFrame(csv_records)
        csv_df.to_csv(csv_filename, index=False)
        logger.info(f"Machine-readable gap report saved to: {csv_filename}")
    elif csv_filename.exists():
        csv_filename.unlink()
        logger.info("Removed stale machine-readable gap report: %s", csv_filename)

    # --- 2. Generate Human-Readable Markdown Report ---
    md_filename = output_path.parent / f"{output_path.stem}_tracking_gaps_report.md"
    with open(md_filename, "w") as f:
        f.write("# Tracking Gap Analysis Report\n\n")
        f.write(f"**Analyzed Directory:** `{output_path}`\n")
        f.write(
            f"**Report Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        if not gap_report:
            f.write("## Analysis Complete: No Tracking Gaps Found\n\n")
            f.write("All tracked instances were present in all analyzed frames.\n")
            logger.info(f"Human-readable report saved to: {md_filename}")
            return md_filename

        f.write("## Summary of Detected Gaps\n\n")
        f.write(
            "This report lists all time periods where a tracked instance was not found. "
            "These 'gaps' may indicate that the animal left the frame, was occluded, "
            "or that no JSON file was generated for that frame (no detections).\n\n"
        )

        for label, gaps in gap_report.items():
            f.write(f"### Instance: `{label}`\n")
            f.write(f"* **Total Gaps Found:** {len(gaps)}\n\n")

            f.write("| Gap # | Start Frame | End Frame | Duration (frames) |\n")
            f.write("|:-----:|:-----------:|:---------:|:------------------|\n")
            for i, gap in enumerate(gaps, 1):
                f.write(
                    f"| {i} | {gap['start_frame']} | {gap['end_frame']} | {gap['duration_frames']} |\n"
                )
            f.write("\n")

        f.write("## Actionable Recommendations\n\n")
        f.write(
            "To fix these gaps, especially in cases where the animal was present but not tracked:\n\n"
        )
        f.write("1.  **Navigate to the `Start Frame`** of a gap listed above.\n")
        f.write(
            "2.  **Manually re-label** the animal with its correct instance label.\n"
        )
        f.write(
            "3.  Open the **'Define Video Segments'** tool from the 'Video Tools' menu.\n"
        )
        f.write(
            "4.  Create a new segment, setting its start to the `Start Frame` and its end to the `End Frame` of the gap.\n"
        )
        f.write(
            "5.  **Run tracking for just that segment.** This will efficiently fill in the missing annotations without re-processing the entire video.\n\n"
        )

    logger.info(f"Human-readable report saved to: {md_filename}")
    return md_filename


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a directory of tracked LabelMe JSON files to find and report tracking gaps."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file. The script will look for JSON files in a directory "
        "with the same name (e.g., 'my_video/' for 'my_video.mp4').",
    )
    args = parser.parse_args()

    if not OPENCV_AVAILABLE:
        logger.error(
            "OpenCV is required to read video metadata for accurate frame counts. "
            "Please install it using: pip install opencv-python"
        )
        # Decide if you want to exit or allow the fallback
        return

    gaps = find_tracking_gaps(args.video_path)
    generate_reports(gaps, args.video_path)


if __name__ == "__main__":
    main()
