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
    input_mode: str | None = None,
    input_source: str | None = None,
    output_folder: str | None = None,
    scale_factor: float | None = None,
    fps: float | None = None,
    apply_denoise: bool | None = None,
    auto_contrast: bool | None = None,
    auto_contrast_strength: float | None = None,
    crop_params: tuple[int, int, int, int] | None = None,
    per_video_overrides: Mapping[str, Mapping[str, object]] | None = None,
    command_log: Mapping[str, str] | None = None,
    progress_callback=None,
    cancel_callback=None,
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

    target_files = [
        video_file
        for video_file in sorted(os.listdir(folder))
        if video_file.lower() in selected_video_names
    ]
    overrides_by_source_name: dict[str, Mapping[str, object]] = {}
    if per_video_overrides:
        for source_path, override_config in per_video_overrides.items():
            source_name = Path(source_path).name.lower()
            overrides_by_source_name[source_name] = override_config
    total_targets = len(target_files)

    def _is_cancelled() -> bool:
        return bool(cancel_callback and cancel_callback())

    def _raise_if_cancelled() -> None:
        if _is_cancelled():
            raise RuntimeError("Processing cancelled by user.")

    if progress_callback is not None:
        progress_callback(
            0,
            max(total_targets, 1),
            "Writing summaries",
        )

    for index, video_file in enumerate(target_files, start=1):
        _raise_if_cancelled()
        if progress_callback is not None:
            progress_callback(
                index - 1,
                max(total_targets, 1),
                f"Writing {index}/{max(total_targets, 1)} - {video_file}",
            )

        base, _ = os.path.splitext(video_file)
        md_path = os.path.join(folder, base + ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Metadata and Processing Info for {video_file}\n\n")
            if (
                input_mode is not None
                or input_source is not None
                or output_folder is not None
                or scale_factor is not None
                or fps is not None
                or apply_denoise is not None
                or auto_contrast is not None
                or crop_params is not None
            ):
                f.write("**Downsample Parameters:**\n")
                if input_mode is not None:
                    f.write(f"- Input Mode: {input_mode}\n")
                if input_source is not None:
                    f.write(f"- Input Source: {input_source}\n")
                if output_folder is not None:
                    f.write(f"- Output Folder: {output_folder}\n")
                if scale_factor is not None:
                    f.write(f"- Scale Factor: {scale_factor}\n")
                if fps is not None:
                    f.write(f"- FPS: {fps}\n")
                else:
                    f.write("- FPS: original per-video FPS\n")
                if apply_denoise is not None:
                    f.write(f"- Apply Denoise: {apply_denoise}\n")
                if auto_contrast is not None:
                    f.write(f"- Auto Contrast: {auto_contrast}\n")
                if auto_contrast:
                    f.write(f"- Auto Contrast Strength: {auto_contrast_strength}\n")
                if crop_params is not None:
                    crop_x, crop_y, crop_width, crop_height = crop_params
                    f.write(
                        f"- Crop Region: x={crop_x}, y={crop_y}, width={crop_width}, height={crop_height}\n"
                    )
                source_name = video_file.lower()
                if source_name.endswith("_fix.mp4"):
                    source_name = source_name[: -len("_fix.mp4")] + ".mp4"
                override_config = overrides_by_source_name.get(source_name)
                f.write(
                    f"- Per-video review selection: {'yes' if override_config else 'no'}\n"
                )
                if override_config:
                    f.write(
                        f"  - Effective Scale Factor: {override_config.get('scale_factor')}\n"
                    )
                    override_fps = override_config.get("fps")
                    if override_fps is None:
                        f.write("  - Effective FPS: source FPS\n")
                    else:
                        f.write(f"  - Effective FPS: {override_fps}\n")
                    f.write(
                        "  - Effective Apply Denoise: "
                        f"{override_config.get('apply_denoise')}\n"
                    )
                    f.write(
                        "  - Effective Auto Contrast: "
                        f"{override_config.get('auto_contrast')}\n"
                    )
                    f.write(
                        "  - Effective Auto Contrast Strength: "
                        f"{override_config.get('auto_contrast_strength')}\n"
                    )
                    override_crop = override_config.get("crop_params")
                    if override_crop is None:
                        f.write("  - Effective Crop Region: disabled\n")
                    else:
                        f.write(
                            "  - Effective Crop Region: "
                            f"x={override_crop[0]}, y={override_crop[1]}, "
                            f"width={override_crop[2]}, height={override_crop[3]}\n"
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

        if progress_callback is not None:
            progress_callback(
                index,
                max(total_targets, 1),
                f"Wrote {index}/{max(total_targets, 1)} - {video_file}",
            )

    if progress_callback is not None:
        _raise_if_cancelled()
        progress_callback(
            max(total_targets, 1), max(total_targets, 1), "Summary complete"
        )
