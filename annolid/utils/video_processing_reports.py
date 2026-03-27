from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import cv2

from annolid.utils.videos import collect_video_metadata, save_metadata_to_csv

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".mpeg",
    ".mpg",
    ".m4v",
    ".mts",
)


def collect_video_metadata_for_paths(
    folder: str, video_paths: Sequence[str] | None = None
) -> list[dict]:
    """Collect video metadata for either a folder or an explicit file list."""
    if video_paths is None:
        return collect_video_metadata(folder)

    metadata_list: list[dict] = []
    for video_path in video_paths:
        path = Path(video_path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            continue
        try:
            metadata_list.append(
                {
                    "video_name": path.name,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "codec": cap.get(cv2.CAP_PROP_FOURCC),
                }
            )
        finally:
            cap.release()
    return metadata_list


def save_processing_summary(
    folder: str,
    video_paths: Sequence[str] | None = None,
    *,
    scale_factor: float | None = None,
    fps: float | None = None,
    apply_denoise: bool | None = None,
    auto_contrast: bool | None = None,
    auto_contrast_strength: float | None = None,
    crop_params: tuple[int, int, int, int] | None = None,
    command_log: Mapping[str, str] | None = None,
) -> None:
    """Write metadata.csv and per-video markdown summaries for processed videos."""
    metadata_list = collect_video_metadata_for_paths(folder, video_paths=video_paths)
    csv_all_path = os.path.join(folder, "metadata.csv")
    save_metadata_to_csv(metadata_list, csv_all_path)

    if video_paths is None:
        selected_video_names = {
            video_file.lower()
            for video_file in os.listdir(folder)
            if video_file.lower().endswith(VIDEO_EXTENSIONS)
        }
    else:
        selected_video_names = {
            Path(video_path).name.lower() for video_path in video_paths
        }

    for video_file in os.listdir(folder):
        if video_file.lower() not in selected_video_names:
            continue

        base, _ = os.path.splitext(video_file)
        md_path = os.path.join(folder, base + ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Metadata and Processing Info for {video_file}\n\n")
            if scale_factor is not None:
                f.write("**Processing Parameters:**\n")
                f.write(f"- Scale Factor: {scale_factor}\n")
                f.write(
                    f"- FPS: {fps if fps is not None else 'original per-video FPS'}\n"
                )
                f.write(f"- Apply Denoise: {apply_denoise}\n")
                f.write(f"- Auto Contrast: {auto_contrast}\n")
                if auto_contrast:
                    f.write(f"- Auto Contrast Strength: {auto_contrast_strength}\n")
                if crop_params is not None:
                    crop_x, crop_y, crop_width, crop_height = crop_params
                    f.write(
                        f"- Crop Region: x={crop_x}, y={crop_y}, width={crop_width}, height={crop_height}\n"
                    )
                f.write("\n")
            if command_log is not None:
                command_used = command_log.get(video_file, "N/A")
                f.write("**FFmpeg Command:**\n")
                f.write("```\n")
                f.write(f"{command_used}\n")
                f.write("```\n\n")
            f.write("**Video Metadata:**\n")
            video_metadata = [
                m
                for m in metadata_list
                if m.get("video_name", "").lower() == video_file.lower()
            ]
            for entry in video_metadata:
                for key, value in entry.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
