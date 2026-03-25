import cv2
import os
import subprocess
import csv
import argparse
from dataclasses import dataclass
from annolid.core.media.video import get_video_fps
from annolid.utils.logger import logger


def extract_frames_with_opencv(video_path, output_dir=None, start_number=0, quality=95):
    """
    Extract frames from a video file and save them as high-quality JPEG images using OpenCV.
    The saved file names start with the video name followed by an underscore and a 9-digit number.

    If the output directory already contains the same or more frames than the video, extraction is skipped.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str, optional): Directory where the extracted frames will be saved.
                                    Defaults to the video path without its file extension.
        start_number (int): Starting number for naming the output JPEG files.
        quality (int): JPEG quality factor (0 to 100, where 100 is the best quality).

    Returns:
        str: The output directory path where frames are saved.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video file cannot be opened.
    """
    # Check if the video file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Set the default output directory to be the video path without the file extension
    if output_dir is None:
        output_dir = os.path.splitext(video_path)[0]

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Count existing JPEG files in the output directory
    existing_frames = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])

    # Check if the output directory already has the required number of frames
    if existing_frames >= total_frames:
        logger.info(
            f"Output directory already contains {existing_frames} frames, skipping extraction."
        )
        cap.release()
        return output_dir

    # Extract frames
    frame_number = start_number
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frame is returned (end of video)

        # Construct the output file path with the video name and a 9-digit zero-padded numbering
        output_filename = os.path.join(output_dir, f"{frame_number:09d}.jpg")

        # Save the frame as a JPEG image with specified quality
        cv2.imwrite(output_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

        frame_number += 1

    # Release the video capture object
    cap.release()

    logger.info(f"Frames extracted and saved in: {output_dir}")
    return output_dir


def frame_to_timestamp(frames, fps):
    timestamps = []
    for frame in frames:
        timestamp = frame / fps
        timestamps.append(timestamp)
    return timestamps


def collect_video_metadata(input_folder):
    """
    Collects metadata for each video file in the specified input folder.

    Args:
    - input_folder (str): Path to the folder containing video files.

    Returns:
    - metadata (list of dicts): List of dictionaries containing metadata for each video.
    """
    metadata = []

    if not os.path.exists(input_folder):
        logger.info("Input folder does not exist.")
        return metadata

    video_files = [
        f for f in os.listdir(input_folder) if f.endswith((".mp4", ".avi", ".mkv"))
    ]

    if not video_files:
        logger.info("No video files found in the input folder.")
        return metadata

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            logger.info(f"Error opening video file: {video_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cap.get(cv2.CAP_PROP_FOURCC)

        metadata_entry = {
            "video_name": video_file,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "codec": codec,
        }

        metadata.append(metadata_entry)

        logger.info(f"Collected metadata for {video_file}")

        cap.release()

    return metadata


def save_metadata_to_csv(metadata, output_csv):
    """
    Saves video metadata to a CSV file.

    Args:
    - metadata (list of dicts): List of dictionaries containing video metadata.
    - output_csv (str): Path to the output CSV file.
    """
    if not metadata:
        logger.info("No metadata to save.")
        return

    fieldnames = ["video_name", "width", "height", "fps", "frame_count", "codec"]
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)


def compress_and_rescale_video(
    input_folder,
    output_folder,
    scale_factor,
    fps=None,
    apply_denoise=False,
    auto_contrast=False,
    auto_contrast_strength=1.0,
    crop_x=None,
    crop_y=None,
    crop_width=None,
    crop_height=None,
):
    """
    Compresses and rescales video files in the input folder using ffmpeg.
    Optionally applies spacetempo smoothing (denoising), adjusts FPS, and
    crops to a specified region.

    Args:
        input_folder (str): Path to the folder containing video files.
        output_folder (str): Path to the folder for processed videos.
        scale_factor (float): Scale factor for resizing videos (e.g., 0.25 for 25%).
        fps (int): Frames per second for the output video.
        apply_denoise (bool): If True, applies spacetempo smoothing using the hqdn3d filter.
        auto_contrast (bool): If True, apply light auto-contrast enhancement.
        auto_contrast_strength (float): Strength multiplier for auto-contrast.
        crop_x (int, optional): X coordinate of the top-left corner for cropping.
        crop_y (int, optional): Y coordinate of the top-left corner for cropping.
        crop_width (int, optional): Width of the crop area.
        crop_height (int, optional): Height of the crop area.

    Returns:
        dict: A mapping from each output video filename to the executed FFmpeg command.
    """
    if not os.path.exists(input_folder):
        logger.info("Input folder does not exist.")
        return {}
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_extensions = (
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
    video_files = [
        f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)
    ]

    if not video_files:
        logger.info("No video files found in the input folder.")
        return {}

    command_log = {}

    @dataclass(frozen=True)
    class _EncoderProfile:
        name: str
        args: list[str]

    def _resolve_ffmpeg_binary() -> str:
        binary = str(os.environ.get("FFMPEG_BINARY", "ffmpeg") or "ffmpeg").strip()
        if binary == "ffmpeg-imageio":
            # imageio helper binaries can be feature-limited; prefer system ffmpeg.
            return "ffmpeg"
        return binary

    def _candidate_profiles() -> list[_EncoderProfile]:
        return [
            _EncoderProfile(
                name="h264_videotoolbox",
                args=["-c:v", "h264_videotoolbox", "-b:v", "4M"],
            ),
            _EncoderProfile(
                name="libopenh264",
                args=["-c:v", "libopenh264", "-b:v", "2M"],
            ),
            _EncoderProfile(
                name="mpeg4",
                args=["-c:v", "mpeg4", "-q:v", "5"],
            ),
            _EncoderProfile(
                name="libx264",
                args=["-c:v", "libx264", "-crf", "23"],
            ),
        ]

    def _compute_even_scaled_dimensions(
        video_path: str, scale: float
    ) -> tuple[int, int] | None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
        if width <= 0 or height <= 0:
            return None
        out_w = max(2, int(round(width * float(scale))))
        out_h = max(2, int(round(height * float(scale))))
        if out_w % 2:
            out_w -= 1
        if out_h % 2:
            out_h -= 1
        out_w = max(out_w, 2)
        out_h = max(out_h, 2)
        return out_w, out_h

    def _candidate_filter_chains(
        *,
        base_filter: str,
        apply_denoise: bool,
        auto_contrast: bool,
        auto_contrast_strength: float,
    ) -> list[tuple[str, str]]:
        contrast_strength = max(0.0, float(auto_contrast_strength))
        contrast = 1.0 + min(contrast_strength, 3.0) * 0.25
        brightness = min(max((contrast_strength - 1.0) * 0.03, -0.1), 0.1)
        saturation = 1.0 + min(contrast_strength, 3.0) * 0.06
        contrast_filter = (
            f"eq=contrast={contrast:.3f}:brightness={brightness:.3f}:"
            f"saturation={saturation:.3f}"
        )
        enhanced_base = (
            f"{base_filter},{contrast_filter}" if auto_contrast else base_filter
        )

        if not apply_denoise:
            return [("base", enhanced_base)]
        # Prefer the compact positional hqdn3d args first, then named args for
        # ffmpeg builds that reject positional parsing. Final fallback disables
        # denoise so scaling/compression can still complete.
        return [
            ("denoise_positional", f"{enhanced_base},hqdn3d=4.0:3.0:6.0:4.5"),
            (
                "denoise_named",
                (
                    f"{enhanced_base},"
                    "hqdn3d=luma_spatial=4.0:chroma_spatial=3.0:"
                    "luma_tmp=6.0:chroma_tmp=4.5"
                ),
            ),
            ("no_denoise_fallback", enhanced_base),
        ]

    def _run_ffmpeg_with_fallback(
        *,
        ffmpeg_bin: str,
        input_path: str,
        output_path: str,
        filter_chains: list[tuple[str, str]],
        copy_audio: bool,
        fps_value: float,
        scale_factor_value: float,
    ) -> tuple[list[str] | None, str | None, str | None, str | None]:
        audio_args = ["-c:a", "copy"] if copy_audio else ["-c:a", "aac", "-b:a", "128k"]
        attempts: list[tuple[str, str, str]] = []
        saw_filtergraph_error = False

        for filter_name, filter_chain in filter_chains:
            for profile in _candidate_profiles():
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    input_path,
                    "-vf",
                    filter_chain,
                    *profile.args,
                    *audio_args,
                    output_path,
                ]
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    return cmd, profile.name, filter_name, None
                except subprocess.CalledProcessError as exc:
                    stderr = (exc.stderr or "").strip()
                    summary_line = (
                        stderr.splitlines()[-1] if stderr else "unknown error"
                    )
                    lowered = stderr.lower()
                    if (
                        "filter not found" in lowered
                        or "error parsing filterchain" in lowered
                        or "error initializing a simple filtergraph" in lowered
                    ):
                        saw_filtergraph_error = True
                    attempts.append((filter_name, profile.name, summary_line))
                    # Keep trying fallback profiles and filter variants.
                    continue

        # Last-resort fallback: bypass filtergraph and use plain ffmpeg options.
        if saw_filtergraph_error:
            dims = _compute_even_scaled_dimensions(input_path, scale_factor_value)
            for profile in _candidate_profiles():
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    input_path,
                    "-r",
                    str(fps_value),
                ]
                if dims is not None:
                    cmd.extend(["-s", f"{dims[0]}x{dims[1]}"])
                cmd.extend([*profile.args, *audio_args, output_path])
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    return cmd, profile.name, "simple_io_fallback", None
                except subprocess.CalledProcessError as exc:
                    stderr = (exc.stderr or "").strip()
                    summary_line = (
                        stderr.splitlines()[-1] if stderr else "unknown error"
                    )
                    attempts.append(("simple_io_fallback", profile.name, summary_line))
                    continue

        if not attempts:
            return None, None, None, "No ffmpeg profiles were attempted."

        last_filter, last_profile, last_error = attempts[-1]
        summary = "; ".join(f"{flt}/{name}: {err}" for flt, name, err in attempts)
        return (
            None,
            last_profile,
            last_filter,
            f"All ffmpeg profiles failed ({summary}). Last error: {last_error}",
        )

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)

        video_fps = fps if fps is not None else get_video_fps(input_path)
        if video_fps is None:
            logger.info(f"Warning: Failed to detect FPS for {video_file}. Skipping.")
            continue

        root, _ = os.path.splitext(video_file)
        output_filename = f"{root}_fix.mp4" if apply_denoise else f"{root}.mp4"
        output_path = os.path.join(output_folder, output_filename)

        # Build filter chain.
        filter_chain = ""
        # Optionally add crop filter if all crop parameters are provided.
        if (
            crop_x is not None
            and crop_y is not None
            and crop_width is not None
            and crop_height is not None
        ):
            filter_chain += f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
        # Append the FPS and scaling filters.
        # Force even dimensions for broader encoder compatibility.
        dims = _compute_even_scaled_dimensions(input_path, scale_factor)
        if dims is not None:
            filter_chain += f"fps={video_fps},scale={dims[0]}:{dims[1]}"
        else:
            filter_chain += (
                f"fps={video_fps},"
                f"scale=trunc(iw*{scale_factor}/2)*2:trunc(ih*{scale_factor}/2)*2"
            )
        filter_chains = _candidate_filter_chains(
            base_filter=filter_chain,
            apply_denoise=bool(apply_denoise),
            auto_contrast=bool(auto_contrast),
            auto_contrast_strength=float(auto_contrast_strength),
        )

        ffmpeg_bin = _resolve_ffmpeg_binary()
        cmd, profile_name, filter_name, error_text = _run_ffmpeg_with_fallback(
            ffmpeg_bin=ffmpeg_bin,
            input_path=input_path,
            output_path=output_path,
            filter_chains=filter_chains,
            copy_audio=bool(apply_denoise),
            fps_value=float(video_fps),
            scale_factor_value=float(scale_factor),
        )
        if cmd is not None:
            command_str = " ".join(cmd)
            logger.info(
                (
                    "Compressed and rescaled %s to %s using '%s' with filter mode "
                    "'%s' (denoise=%s, auto_contrast=%s, contrast_strength=%.2f)."
                ),
                video_file,
                output_path,
                profile_name,
                filter_name,
                bool(apply_denoise),
                bool(auto_contrast),
                float(auto_contrast_strength),
            )
            command_log[output_filename] = command_str
        else:
            logger.warning(
                "Failed to compress and rescale %s. %s",
                video_file,
                error_text,
            )

    return command_log


def main(args):
    """
    Examples
    Suppose you have some .mkv files in a folder named input_videos,
    and you want to convert them to .mp4 format with a scale factor of 0.5 for resizing.
      Additionally, you want to save the metadata to a CSV file named video_metadata.csv.
      Here's how you would run the script:
    ```python script.py input_videos --output_folder output_videos -
        -output_csv video_metadata.csv --scale_factor 0.5```
    In this command:
    input_videos is the input folder containing the .mkv files.
    --output_folder output_videos specifies the output folder
    where the converted .mp4 files will be saved.
    --output_csv video_metadata.csv specifies the output CSV file path for storing the metadata.
    --scale_factor 0.5 specifies the scale factor for resizing the videos during conversion.

    If you only want to collect metadata without performing the conversion,
    you can add the --collect_only flag:
    ```python script.py input_videos --output_csv video_metadata.csv --collect_only```
    This command will collect metadata for the .mkv files in the input_videos folder and
    save it to video_metadata.csv. No conversion will be performed in this case.

    """
    if args.collect_only:
        metadata = collect_video_metadata(args.input_folder)
        if args.output_csv:
            save_metadata_to_csv(metadata, args.output_csv)
    else:
        compress_and_rescale_video(
            args.input_folder, args.output_folder, args.scale_factor
        )
        if args.output_csv:
            metadata = collect_video_metadata(args.output_folder)
            if not metadata:
                logger.info(
                    "No processed videos were generated in %s; saving input metadata instead.",
                    args.output_folder,
                )
                metadata = collect_video_metadata(args.input_folder)
            save_metadata_to_csv(metadata, args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress and rescale video files and collect metadata."
    )
    parser.add_argument(
        "input_folder", type=str, help="Input folder path containing video files."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder path for compressed and rescaled video files.",
    )
    parser.add_argument(
        "--output_csv", type=str, help="Output CSV file path for metadata."
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.5,
        help="Scale factor for resizing videos.",
    )
    parser.add_argument(
        "--collect_only",
        action="store_true",
        help="Collect metadata only, do not compress and rescale.",
    )

    args = parser.parse_args()
    main(args)
