from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Generator, List, Optional

import cv2
import numpy as np

from annolid.utils.logger import logger

DEFAULT_SUBPROCESS_TIMEOUT = 10.0


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


class CV2Video:
    """Lightweight OpenCV video reader that returns RGB frames."""

    def __init__(self, video_file: str | Path, use_decord: bool = False):
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
        self._first_frame = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None

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
        if self._first_frame is None:
            self._first_frame = self.load_frame(0)
        return self._first_frame

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
