import cv2
import os
import queue
import select
import subprocess
import csv
import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from annolid.core.media.video import get_video_fps
from annolid.utils.logger import logger

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


def _ffmpeg_cancel_poll_interval() -> float:
    raw_value = os.environ.get("ANNOLID_FFMPEG_CANCEL_POLL_INTERVAL", "0.2")
    try:
        interval = float(raw_value)
    except (TypeError, ValueError):
        return 0.2
    if interval <= 0:
        return 0.2
    return max(0.05, min(interval, 2.0))


def _is_windows_platform() -> bool:
    return os.name == "nt"


def _probe_video_duration_ms(video_path: str) -> int | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0 or fps <= 0:
            return None
        return int((frame_count / fps) * 1000)
    finally:
        cap.release()


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

    for input_path_obj in (Path(input_folder) / f for f in video_files):
        input_path = str(input_path_obj)
        video_file = input_path_obj.name

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
    input_video_path=None,
    fps=None,
    apply_denoise=False,
    auto_contrast=False,
    auto_contrast_strength=1.0,
    crop_x=None,
    crop_y=None,
    crop_width=None,
    crop_height=None,
    progress_callback=None,
    cancel_callback=None,
):
    """
    Compresses and rescales video files in the input folder using ffmpeg.
    Optionally applies spacetempo smoothing (denoising), adjusts FPS, and
    crops to a specified region.

    Args:
        input_folder (str): Path to the folder containing video files.
        output_folder (str): Path to the folder for processed videos.
        scale_factor (float): Scale factor for resizing videos (e.g., 0.25 for 25%).
        input_video_path (str, optional): If provided, process only this video file.
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
    if input_video_path:
        if not os.path.isfile(input_video_path):
            logger.info("Input video does not exist: %s", input_video_path)
            return {}
    elif not os.path.exists(input_folder):
        logger.info("Input folder does not exist.")
        return {}
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if input_video_path:
        video_paths = [Path(input_video_path)]
    else:
        video_paths = [
            Path(input_folder) / f
            for f in os.listdir(input_folder)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]

    if not video_paths:
        logger.info("No video files found in the input folder.")
        return {}

    command_log = {}
    total_videos = len(video_paths)
    video_durations_ms = [_probe_video_duration_ms(str(path)) for path in video_paths]
    total_duration_ms = sum(
        duration for duration in video_durations_ms if duration is not None
    )
    use_duration_progress = total_duration_ms > 0 and all(
        duration is not None and duration > 0 for duration in video_durations_ms
    )

    def _emit_progress(current_index: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(
                max(0, min(int(current_index), total_videos)),
                total_videos,
                str(message),
            )

    def _emit_percent_progress(
        completed_ms: int, current_ms: int, message: str
    ) -> None:
        if progress_callback is None:
            return
        if use_duration_progress and total_duration_ms > 0:
            overall_ms = max(0, min(completed_ms + current_ms, total_duration_ms))
            percent = max(0, min(int((overall_ms / total_duration_ms) * 100), 100))
            progress_callback(percent, 100, str(message))
            return
        _emit_progress(current_ms, message)

    def _is_cancelled() -> bool:
        return bool(cancel_callback and cancel_callback())

    def _raise_if_cancelled() -> None:
        if _is_cancelled():
            raise RuntimeError("Processing cancelled by user.")

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
        video_path: str,
        scale: float,
        crop_w: int | None = None,
        crop_h: int | None = None,
    ) -> tuple[int, int] | None:
        if crop_w is not None and crop_h is not None and crop_w > 0 and crop_h > 0:
            width = int(crop_w)
            height = int(crop_h)
        else:
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

    def _parse_ffmpeg_progress_line(line: str) -> tuple[str, int | str] | None:
        if "=" not in line:
            return None
        key, value = line.strip().split("=", 1)
        if key in {"out_time_ms", "out_time_us"}:
            try:
                numeric = int(value)
            except ValueError:
                return None
            if key == "out_time_us":
                numeric = numeric // 1000
            return key, max(0, numeric)
        if key == "progress":
            return key, value.strip()
        return None

    def _run_ffmpeg_with_fallback(
        *,
        ffmpeg_bin: str,
        input_path: str,
        output_path: str,
        filter_chains: list[tuple[str, str]],
        copy_audio: bool,
        fps_value: float,
        scale_factor_value: float,
        crop_x_value: int | None = None,
        crop_y_value: int | None = None,
        crop_width_value: int | None = None,
        crop_height_value: int | None = None,
        completed_ms: int = 0,
        total_ms: int = 0,
        progress_prefix: str = "",
        streaming_label: str | None = None,
    ) -> tuple[list[str] | None, str | None, str | None, str | None]:
        audio_args = ["-c:a", "copy"] if copy_audio else ["-c:a", "aac", "-b:a", "128k"]
        attempts: list[tuple[str, str, str]] = []
        saw_filtergraph_error = False

        def _run_ffmpeg_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            if cancel_callback is None:
                return subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            try:
                stdout_stream = proc.stdout
                stderr_stream = proc.stderr
                stdout_lines: list[str] = []
                stderr_lines: list[str] = []
                last_reported_ms = -1

                if _is_windows_platform():
                    event_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
                    finished_streams: set[str] = set()

                    def _reader(stream, stream_name: str) -> None:
                        try:
                            while True:
                                line = stream.readline()
                                if not line:
                                    break
                                event_queue.put((stream_name, line))
                        finally:
                            event_queue.put((stream_name, None))

                    threads: list[threading.Thread] = []
                    for stream_name, stream in (
                        ("stdout", stdout_stream),
                        ("stderr", stderr_stream),
                    ):
                        if stream is None:
                            continue
                        thread = threading.Thread(
                            target=_reader,
                            args=(stream, stream_name),
                            daemon=True,
                        )
                        thread.start()
                        threads.append(thread)

                    try:
                        while True:
                            if _is_cancelled():
                                proc.terminate()
                                try:
                                    proc.wait(timeout=2)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    proc.wait(timeout=2)
                                raise RuntimeError("Processing cancelled by user.")
                            try:
                                stream_name, line = event_queue.get(
                                    timeout=_ffmpeg_cancel_poll_interval()
                                )
                            except queue.Empty:
                                returncode = proc.poll()
                                if returncode is not None and len(
                                    finished_streams
                                ) >= len(threads):
                                    break
                                continue
                            if line is None:
                                finished_streams.add(stream_name)
                                returncode = proc.poll()
                                if returncode is not None and len(
                                    finished_streams
                                ) >= len(threads):
                                    break
                                continue
                            if stream_name == "stdout":
                                stdout_lines.append(line)
                                parsed = _parse_ffmpeg_progress_line(line)
                                if (
                                    progress_callback is not None
                                    and parsed is not None
                                    and parsed[0] in {"out_time_ms", "out_time_us"}
                                ):
                                    current_ms = int(parsed[1])
                                    if total_ms > 0 and current_ms != last_reported_ms:
                                        last_reported_ms = current_ms
                                        _emit_percent_progress(
                                            completed_ms,
                                            current_ms,
                                            f"Encoding {streaming_label}"
                                            if streaming_label
                                            else progress_prefix,
                                        )
                            else:
                                stderr_lines.append(line)
                            returncode = proc.poll()
                            if returncode is not None and len(finished_streams) >= len(
                                threads
                            ):
                                break
                    finally:
                        for thread in threads:
                            thread.join(timeout=2)
                    returncode = proc.poll()
                    stdout = "".join(stdout_lines)
                    stderr = "".join(stderr_lines)
                    if returncode is None:
                        returncode = proc.wait()
                    if returncode != 0:
                        raise subprocess.CalledProcessError(
                            returncode=returncode,
                            cmd=cmd,
                            output=stdout,
                            stderr=stderr,
                        )
                    return subprocess.CompletedProcess(
                        cmd, returncode=returncode, stdout=stdout, stderr=stderr
                    )

                stdout_stream = stdout_stream
                stderr_stream = stderr_stream
                streams = [
                    stream for stream in (stdout_stream, stderr_stream) if stream
                ]
                while True:
                    if _is_cancelled():
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=2)
                        raise RuntimeError("Processing cancelled by user.")
                    ready, _, _ = select.select(
                        streams, [], [], _ffmpeg_cancel_poll_interval()
                    )
                    for stream in ready:
                        line = stream.readline()
                        if not line:
                            continue
                        if stream is stdout_stream:
                            stdout_lines.append(line)
                            parsed = _parse_ffmpeg_progress_line(line)
                            if (
                                progress_callback is not None
                                and parsed is not None
                                and parsed[0] in {"out_time_ms", "out_time_us"}
                            ):
                                current_ms = int(parsed[1])
                                if total_ms > 0 and current_ms != last_reported_ms:
                                    last_reported_ms = current_ms
                                    _emit_percent_progress(
                                        completed_ms,
                                        current_ms,
                                        f"Encoding {streaming_label}"
                                        if streaming_label
                                        else progress_prefix,
                                    )
                        else:
                            stderr_lines.append(line)
                    returncode = proc.poll()
                    if returncode is not None:
                        stdout_rest, stderr_rest = proc.communicate()
                        stdout = "".join(stdout_lines) + (stdout_rest or "")
                        stderr = "".join(stderr_lines) + (stderr_rest or "")
                        if returncode != 0:
                            raise subprocess.CalledProcessError(
                                returncode=returncode,
                                cmd=cmd,
                                output=stdout,
                                stderr=stderr,
                            )
                        return subprocess.CompletedProcess(
                            cmd, returncode=returncode, stdout=stdout, stderr=stderr
                        )
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=2)

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
                    _run_ffmpeg_command(cmd)
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
            simple_filter_parts = []
            if (
                crop_x_value is not None
                and crop_y_value is not None
                and crop_width_value is not None
                and crop_height_value is not None
            ):
                simple_filter_parts.append(
                    f"crop={crop_width_value}:{crop_height_value}:{crop_x_value}:{crop_y_value}"
                )
            simple_filter_parts.append(f"fps={fps_value}")
            dims = _compute_even_scaled_dimensions(
                input_path,
                scale_factor_value,
                crop_w=crop_width_value,
                crop_h=crop_height_value,
            )
            if dims is not None:
                simple_filter_parts.append(f"scale={dims[0]}:{dims[1]}")
            for profile in _candidate_profiles():
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    input_path,
                ]
                if simple_filter_parts:
                    cmd.extend(["-vf", ",".join(simple_filter_parts)])
                cmd.extend([*profile.args, *audio_args, output_path])
                try:
                    _run_ffmpeg_command(cmd)
                    return cmd, profile.name, "simple_filter_fallback", None
                except subprocess.CalledProcessError as exc:
                    stderr = (exc.stderr or "").strip()
                    summary_line = (
                        stderr.splitlines()[-1] if stderr else "unknown error"
                    )
                    attempts.append(
                        ("simple_filter_fallback", profile.name, summary_line)
                    )
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

    _emit_progress(0, "Starting")

    completed_duration_ms = 0
    for index, input_path_obj in enumerate(video_paths, start=1):
        _raise_if_cancelled()
        input_path = str(input_path_obj)
        video_file = input_path_obj.name
        _emit_progress(
            index - 1,
            f"Processing {index}/{total_videos} - {video_file}",
        )

        video_fps = fps if fps is not None else get_video_fps(input_path)
        if video_fps is None:
            logger.info(f"Warning: Failed to detect FPS for {video_file}. Skipping.")
            _emit_progress(index, f"Skipped {index}/{total_videos} - {video_file}")
            continue

        root, _ = os.path.splitext(video_file)
        output_filename = f"{root}_fix.mp4" if apply_denoise else f"{root}.mp4"
        output_path = os.path.join(output_folder, output_filename)

        # Build filter chain.
        filter_parts = []
        # Optionally add crop filter if all crop parameters are provided.
        if (
            crop_x is not None
            and crop_y is not None
            and crop_width is not None
            and crop_height is not None
        ):
            filter_parts.append(f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}")
        # Append the FPS and scaling filters.
        filter_parts.append(f"fps={video_fps}")
        # Force even dimensions for broader encoder compatibility.
        dims = _compute_even_scaled_dimensions(
            input_path,
            scale_factor,
            crop_w=crop_width,
            crop_h=crop_height,
        )
        if dims is not None:
            filter_parts.append(f"scale={dims[0]}:{dims[1]}")
        else:
            filter_parts.append(
                f"scale=trunc(iw*{scale_factor}/2)*2:trunc(ih*{scale_factor}/2)*2"
            )
        filter_chain = ",".join(filter_parts)
        filter_chains = _candidate_filter_chains(
            base_filter=filter_chain,
            apply_denoise=bool(apply_denoise),
            auto_contrast=bool(auto_contrast),
            auto_contrast_strength=float(auto_contrast_strength),
        )

        ffmpeg_bin = _resolve_ffmpeg_binary()
        current_video_duration_ms = video_durations_ms[index - 1] or 0
        cmd, profile_name, filter_name, error_text = _run_ffmpeg_with_fallback(
            ffmpeg_bin=ffmpeg_bin,
            input_path=input_path,
            output_path=output_path,
            filter_chains=filter_chains,
            copy_audio=bool(apply_denoise),
            fps_value=float(video_fps),
            scale_factor_value=float(scale_factor),
            crop_width_value=crop_width,
            crop_height_value=crop_height,
            crop_x_value=crop_x,
            crop_y_value=crop_y,
            completed_ms=completed_duration_ms,
            total_ms=total_duration_ms if use_duration_progress else 0,
            progress_prefix=f"Processing {index}/{total_videos} - {video_file}",
            streaming_label=video_file,
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
        if use_duration_progress:
            completed_duration_ms += current_video_duration_ms
            _emit_percent_progress(
                completed_duration_ms,
                0,
                f"Processed {index}/{total_videos} - {video_file}",
            )
        _emit_progress(index, f"Processed {index}/{total_videos} - {video_file}")

    if total_videos > 0:
        _raise_if_cancelled()
        _emit_progress(total_videos, "Complete")
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
