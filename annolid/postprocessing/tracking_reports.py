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


def find_tracking_gaps(video_path: str) -> dict:
    """
    Scans for tracking gaps based on a video file and its associated JSON directory.

    Args:
        video_path (str): The path to the video file.

    Returns:
        A dictionary where keys are instance labels and values are lists of gap tuples.
    """
    video_file = Path(video_path)
    if not video_file.is_file():
        logger.error(f"Video file not found: {video_file}")
        return {}

    # Infer the JSON directory path from the video filename
    json_directory = video_file.with_suffix("")
    if not json_directory.is_dir():
        logger.error(f"Associated JSON directory not found at: {json_directory}")
        return {}

    logger.info(f"Analyzing video: {video_file.name}")
    logger.info(f"Reading JSON files from: {json_directory}")

    # --- 1. Determine the Full Frame Range (The Ground Truth) ---
    min_frame = 0
    max_frame = -1
    if OPENCV_AVAILABLE:
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            logger.error(f"Could not open video file {video_file} with OpenCV.")
            return {}
        max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()
        logger.info(f"Video metadata: Total frames = {max_frame + 1}.")
    else:
        # Fallback if OpenCV is not installed: infer range from JSON files.
        json_files_for_range = sorted(
            [p for p in json_directory.glob("*.json")],
            key=lambda p: int(re.search(r"(\d+)(?=\.json$)", p.name).group(1)),
        )
        if not json_files_for_range:
            logger.warning("No JSON files found to infer frame range.")
            return {}
        max_frame = int(
            re.search(r"(\d+)(?=\.json$)", json_files_for_range[-1].name).group(1)
        )
        logger.warning(
            f"OpenCV not found. Inferred max frame is {max_frame} from JSON files."
        )

    master_frame_index = pd.RangeIndex(
        start=min_frame, stop=max_frame + 1, name="frame_number"
    )

    # --- 2. Scan and Parse Existing Files ---
    presence_data = []
    all_instance_labels = set()
    for file_path in json_directory.glob("*.json"):
        try:
            # Use robust regex to handle filenames like 'video_00001.json'
            match = re.search(r"(\d+)(?=\.json$)", file_path.name)
            if not match:
                continue

            frame_number = int(match.group(1))
            with open(file_path, "r") as f:
                data = json.load(f)

            labels_in_frame = {shape["label"] for shape in data.get("shapes", [])}
            presence_data.append(
                {"frame_number": frame_number, "labels": labels_in_frame}
            )
            all_instance_labels.update(labels_in_frame)
        except (ValueError, IndexError, json.JSONDecodeError) as e:
            logger.warning(f"Could not parse file {file_path.name}: {e}")

    if not all_instance_labels:
        logger.info("No labeled instances found in any JSON file.")
        return {}

    sorted_labels = sorted(list(all_instance_labels))

    # --- 3. Build Presence DataFrame and Merge with Master Timeline ---
    if not presence_data:
        df_presence_sparse = pd.DataFrame(
            columns=["frame_number"] + sorted_labels
        ).set_index("frame_number")
    else:
        records = []
        for item in presence_data:
            frame_record = {"frame_number": item["frame_number"]}
            for label in sorted_labels:
                frame_record[label] = label in item["labels"]
            records.append(frame_record)
        df_presence_sparse = pd.DataFrame(records).set_index("frame_number")

    df_master = pd.DataFrame(index=master_frame_index)
    df_full = df_master.join(df_presence_sparse).fillna(False)

    # --- 4. Identify Gaps for Each Animal ---
    gap_report = {}
    for label in sorted_labels:
        missing_frames = df_full[~df_full[label]]
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


def generate_reports(gap_report: dict, video_path: str):
    """
    Generates and saves both a human-readable Markdown report and a machine-readable CSV report.
    (This function remains the same as the previous version)
    """
    video_file = Path(video_path)
    output_path = video_file.with_suffix("")

    # --- 1. Generate Machine-Readable CSV Report ---
    csv_records = []
    for label, gaps in gap_report.items():
        for gap in gaps:
            record = {"instance_label": label, **gap}
            csv_records.append(record)

    if csv_records:
        csv_df = pd.DataFrame(csv_records)
        csv_filename = output_path.parent / f"{output_path.stem}_gaps_report.csv"
        csv_df.to_csv(csv_filename, index=False)
        logger.info(f"Machine-readable gap report saved to: {csv_filename}")

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
            return

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
