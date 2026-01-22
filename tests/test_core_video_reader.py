from pathlib import Path

import cv2
import numpy as np
import pytest

from annolid.core.media.video import CV2Video


def _write_test_video(path: Path, *, fps: float = 10.0, frames: int = 5) -> None:
    width, height = 64, 48
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment.")
    try:
        for idx in range(int(frames)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = idx * 10
            writer.write(frame)
    finally:
        writer.release()


def test_core_cv2video_reads_frames_and_timestamps(tmp_path: Path):
    video_path = tmp_path / "test.avi"
    _write_test_video(video_path, fps=10.0, frames=4)

    video = CV2Video(video_path)
    try:
        assert video.total_frames() == 4
        assert video.get_fps() > 0

        video.load_frame(0)
        ts0 = video.last_timestamp_sec()
        video.load_frame(1)
        ts1 = video.last_timestamp_sec()

        assert ts0 is None or ts0 >= 0
        assert ts1 is None or ts1 >= 0
        if ts0 is not None and ts1 is not None:
            assert ts1 >= ts0
    finally:
        video.release()
