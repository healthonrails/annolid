from __future__ import annotations

import base64
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Sequence

import cv2
import numpy as np

from annolid.utils.logger import logger

DEFAULT_SUBPROCESS_TIMEOUT = 10.0


@dataclass(frozen=True)
class SegmentFrameGridResult:
    """Model-ready frame grid plus source-frame metadata."""

    image: np.ndarray
    frame_indices: list[int]
    timestamps_sec: list[float | None]
    start_frame: int
    end_frame: int
    fps: float
    total_frames: int
    rows: int
    columns: int
    tile_width: int
    tile_height: int

    def metadata(self) -> dict[str, object]:
        return {
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "fps": float(self.fps),
            "total_frames": int(self.total_frames),
            "rows": int(self.rows),
            "columns": int(self.columns),
            "tile_width": int(self.tile_width),
            "tile_height": int(self.tile_height),
            "frames": [
                {
                    "tile_index": int(idx),
                    "frame_index": int(frame_index),
                    "timestamp_sec": timestamp,
                }
                for idx, (frame_index, timestamp) in enumerate(
                    zip(self.frame_indices, self.timestamps_sec)
                )
            ],
        }


def get_video_fps(
    video_path: str, *, timeout: float = DEFAULT_SUBPROCESS_TIMEOUT
) -> float | None:
    """Return FPS using ffprobe when available, falling back to OpenCV."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            rate_str = result.stdout.strip()
            if "/" in rate_str:
                num, denom = map(int, rate_str.split("/"))
                if denom != 0:
                    return round(num / denom, 2)
            elif rate_str.replace(".", "", 1).isdigit():
                return round(float(rate_str), 2)
        else:
            logger.warning(
                "ffprobe returned a non-zero exit code while reading FPS for %s: %s",
                video_path,
                result.stderr.strip(),
            )
    except subprocess.TimeoutExpired:
        logger.warning(
            "ffprobe timed out after %.1fs while reading FPS for %s; falling back to OpenCV.",
            timeout,
            video_path,
        )
    except FileNotFoundError:
        # ffprobe is optional
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("[get_video_fps] ffprobe failed for %s: %s", video_path, exc)

    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0:
                return round(float(fps), 2)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[get_video_fps] OpenCV fallback failed for %s: %s", video_path, exc
        )

    return None


def get_keyframe_timestamps(
    video_path: str,
    *,
    timeout: float = DEFAULT_SUBPROCESS_TIMEOUT,
) -> List[float]:
    """Return keyframe timestamps (seconds) via ffprobe; empty list when unavailable."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_frames",
        "-show_entries",
        "frame=pkt_pts_time",
        "-of",
        "csv=p=0",
        video_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        logger.warning(
            "ffprobe returned a non-zero exit code while reading keyframes for %s: %s",
            video_path,
            result.stderr.strip(),
        )
        return []

    timestamps: List[float] = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        if "," in value:
            value = value.split(",")[-1]
        try:
            ts = float(value)
        except ValueError:
            continue
        if ts >= 0:
            timestamps.append(ts)
    return timestamps


def clamp_frame_index(index: int, total_frames: int) -> int:
    """Clamp a frame index to the valid inclusive frame range."""
    if total_frames <= 0:
        return 0
    return max(0, min(int(index), int(total_frames) - 1))


def resolve_segment_frame_bounds(
    *,
    total_frames: int,
    fps: float,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[int, int]:
    """Resolve an inclusive segment frame range from frame or time inputs."""
    max_index = max(0, int(total_frames) - 1)
    if start_frame is not None or end_frame is not None:
        start_idx = int(start_frame or 0)
        end_idx = int(end_frame if end_frame is not None else max_index)
    elif start_sec is not None or end_sec is not None:
        effective_fps = float(fps or 0.0)
        if effective_fps <= 0:
            effective_fps = 30.0
        start_idx = int(round(float(start_sec or 0.0) * effective_fps))
        end_idx = int(
            round(
                float(end_sec if end_sec is not None else (max_index / effective_fps))
                * effective_fps
            )
        )
    else:
        start_idx, end_idx = 0, max_index

    start_idx = clamp_frame_index(start_idx, max(total_frames, 1))
    end_idx = clamp_frame_index(end_idx, max(total_frames, 1))
    return start_idx, end_idx


def sample_segment_frame_indices(
    *,
    start_frame: int,
    end_frame: int,
    sample_count: int,
) -> list[int]:
    """Return deterministic, unique frame indices across an inclusive segment."""
    start = int(start_frame)
    end = int(end_frame)
    if end < start:
        raise ValueError(
            "Segment end_frame must be greater than or equal to start_frame."
        )
    count = max(1, int(sample_count))
    available = (end - start) + 1
    if count >= available:
        return list(range(start, end + 1))
    if count == 1:
        return [start]

    raw = np.linspace(start, end, num=count)
    selected: list[int] = []
    seen: set[int] = set()
    for value in raw:
        index = int(round(float(value)))
        index = max(start, min(index, end))
        if index in seen:
            continue
        seen.add(index)
        selected.append(index)

    candidate = start
    while len(selected) < count and candidate <= end:
        if candidate not in seen:
            selected.append(candidate)
            seen.add(candidate)
        candidate += 1
    return sorted(selected)


def _resize_frame_to_tile(
    frame_rgb: np.ndarray,
    *,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Frame has invalid dimensions.")
    scale = min(float(tile_width) / float(width), float(tile_height) / float(height))
    resized_width = max(1, int(round(float(width) * scale)))
    resized_height = max(1, int(round(float(height) * scale)))
    resized = cv2.resize(
        frame_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    tile = np.zeros((int(tile_height), int(tile_width), 3), dtype=np.uint8)
    y_offset = (int(tile_height) - resized_height) // 2
    x_offset = (int(tile_width) - resized_width) // 2
    tile[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return tile


def build_frame_grid_image(
    frames_rgb: Sequence[np.ndarray],
    *,
    frame_indices: Sequence[int],
    timestamps_sec: Sequence[float | None] | None = None,
    columns: int | None = None,
    tile_width: int = 224,
    tile_height: int | None = None,
    annotate: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Stack RGB frames into a deterministic tiled RGB grid."""
    if not frames_rgb:
        raise ValueError("At least one frame is required to build a frame grid.")
    tile_w = max(16, int(tile_width))
    if tile_height is None:
        first = frames_rgb[0]
        height, width = first.shape[:2]
        tile_h = max(16, int(round(tile_w * (float(height) / float(width)))))
    else:
        tile_h = max(16, int(tile_height))

    tile_count = len(frames_rgb)
    cols = int(columns or math.ceil(math.sqrt(tile_count)))
    cols = max(1, min(cols, tile_count))
    rows = int(math.ceil(float(tile_count) / float(cols)))
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    timestamps = list(timestamps_sec or [None] * tile_count)

    for idx, frame in enumerate(frames_rgb):
        row = idx // cols
        col = idx % cols
        tile = _resize_frame_to_tile(
            frame,
            tile_width=tile_w,
            tile_height=tile_h,
        )
        if annotate:
            frame_index = int(frame_indices[idx]) if idx < len(frame_indices) else idx
            timestamp = timestamps[idx] if idx < len(timestamps) else None
            label = f"f{frame_index}"
            if timestamp is not None:
                label = f"{label} {float(timestamp):.2f}s"
            cv2.rectangle(tile, (0, 0), (min(tile_w - 1, 150), 22), (0, 0, 0), -1)
            cv2.putText(
                tile,
                label,
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        y0 = row * tile_h
        x0 = col * tile_w
        grid[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return grid, rows, cols


def build_segment_frame_grid(
    video_file: str | Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    sample_count: int = 8,
    columns: int | None = None,
    tile_width: int = 224,
    tile_height: int | None = None,
    annotate: bool = True,
) -> SegmentFrameGridResult:
    """Sample a video segment and stack frames into one RGB image grid."""
    video = CV2Video(video_file)
    try:
        total_frames = int(video.total_frames())
        fps = float(video.get_fps() or 0.0)
        resolved_start, resolved_end = resolve_segment_frame_bounds(
            total_frames=total_frames,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        if resolved_end < resolved_start:
            raise ValueError(
                "Segment end_frame must be greater than or equal to start_frame."
            )
        frame_indices = sample_segment_frame_indices(
            start_frame=resolved_start,
            end_frame=resolved_end,
            sample_count=sample_count,
        )
        frames: list[np.ndarray] = []
        timestamps: list[float | None] = []
        for index in frame_indices:
            frames.append(video.load_frame(index))
            timestamp = video.last_timestamp_sec()
            if timestamp is None and fps > 0:
                timestamp = float(index) / float(fps)
            timestamps.append(timestamp)
    finally:
        video.release()

    image, rows, cols = build_frame_grid_image(
        frames,
        frame_indices=frame_indices,
        timestamps_sec=timestamps,
        columns=columns,
        tile_width=tile_width,
        tile_height=tile_height,
        annotate=annotate,
    )
    tile_h = int(image.shape[0] // rows)
    tile_w = int(image.shape[1] // cols)
    return SegmentFrameGridResult(
        image=image,
        frame_indices=frame_indices,
        timestamps_sec=timestamps,
        start_frame=resolved_start,
        end_frame=resolved_end,
        fps=fps,
        total_frames=total_frames,
        rows=rows,
        columns=cols,
        tile_width=tile_w,
        tile_height=tile_h,
    )


def save_rgb_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save an RGB image using OpenCV while preserving caller-visible paths."""
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(out_path), image_bgr):
        raise RuntimeError(f"Failed to save image: {out_path}")
    return out_path


def encode_rgb_image_data_uri(
    image: np.ndarray,
    *,
    image_format: str = "png",
) -> str:
    """Encode an RGB image as a data URI for multimodal model requests."""
    ext = "." + str(image_format or "png").strip().lower().lstrip(".")
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("image_format must be one of: png, jpg, jpeg, webp.")
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(ext, image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame grid image.")
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext.lstrip('.')}"
    data = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class CV2Video:
    """Lightweight OpenCV video reader that returns RGB frames."""

    def __init__(
        self,
        video_file: str | Path,
        use_decord: bool = False,
        *,
        cache_first_frame: bool = False,
    ):
        _ = use_decord  # kept for backward compatibility with older signature
        self.video_file = Path(video_file).expanduser().resolve()
        if not self.video_file.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_file}")
        self.cap = cv2.VideoCapture(str(self.video_file))
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_file}")

        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.frame_count = int(self._frame_count)
        self._fps: Optional[float] = None
        self.current_frame_timestamp_msec: Optional[float] = None
        self.current_frame_timestamp: Optional[float] = None
        self._last_frame_index: Optional[int] = None
        self._cache_first_frame = bool(cache_first_frame)
        self._first_frame = None
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._width: Optional[int] = width if width > 0 else None
        self._height: Optional[int] = height if height > 0 else None

    def total_frames(self) -> int:
        return int(self._frame_count)

    def fps(self) -> float:
        """Backwards-compatible alias for `get_fps()`."""
        return self.get_fps()

    def get_fps(self) -> float:
        if self._fps is None:
            fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self._fps = fps if fps > 0 else 0.0
        return float(self._fps)

    def load_frame(self, frame_number: int):
        if frame_number < 0 or frame_number >= self.total_frames():
            raise KeyError(f"Frame index out of bounds: {frame_number}")

        expected_next = (
            self._last_frame_index + 1 if self._last_frame_index is not None else None
        )
        if expected_next is None or frame_number != expected_next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))

        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            raise KeyError(f"Cannot load frame number: {frame_number}")

        self._last_frame_index = int(frame_number)
        ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.current_frame_timestamp_msec = float(ts) if ts is not None else None
        self.current_frame_timestamp = self.current_frame_timestamp_msec

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def last_timestamp_sec(self) -> Optional[float]:
        if self.current_frame_timestamp_msec is None:
            return None
        ts = float(self.current_frame_timestamp_msec) / 1000.0
        return ts if ts >= 0 else None

    def release(self) -> None:
        try:
            if getattr(self, "cap", None) is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Backwards-compatible helpers
    # ------------------------------------------------------------------

    def get_first_frame(self):
        if self._cache_first_frame:
            if self._first_frame is None:
                self._first_frame = self.load_frame(0)
            return self._first_frame
        return self.load_frame(0)

    def get_width(self) -> int:
        if self._width is None:
            frame = self.get_first_frame()
            self._width = int(frame.shape[1])
        return int(self._width)

    def get_height(self) -> int:
        if self._height is None:
            frame = self.get_first_frame()
            self._height = int(frame.shape[0])
        return int(self._height)

    def get_time_stamp(self) -> Optional[float]:
        return self.current_frame_timestamp_msec

    def get_frames_in_batches(
        self, start_frame: int, end_frame: int, batch_size: int
    ) -> Generator[np.ndarray, None, None]:
        if start_frame >= end_frame:
            raise ValueError("Start frame must be less than end frame.")
        total_frames = self.total_frames()
        end_frame = min(int(end_frame), int(total_frames))
        batch_size = max(1, int(batch_size))
        for batch_start in range(int(start_frame), int(end_frame), int(batch_size)):
            batch_end = min(batch_start + batch_size, int(end_frame))
            for frame_number in range(batch_start, batch_end):
                yield self.load_frame(frame_number)

    def get_frames_between(self, start_frame: int, end_frame: int) -> np.ndarray:
        if start_frame >= end_frame:
            raise ValueError("Start frame must be less than end frame.")
        total_frames = self.total_frames()
        end_frame = min(int(end_frame), int(total_frames))
        frames: List[np.ndarray] = []
        for frame_number in range(int(start_frame), int(end_frame)):
            frames.append(self.load_frame(frame_number))
        return np.stack(frames) if frames else np.zeros((0, 0, 0, 0), dtype=np.uint8)

    def __del__(self) -> None:
        self.release()
