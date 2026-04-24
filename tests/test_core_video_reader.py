from pathlib import Path

import cv2
import numpy as np
import pytest

from annolid.core.media.video import (
    CV2Video,
    build_segment_frame_grid,
    sample_segment_frame_indices,
)


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


def test_sample_segment_frame_indices_are_deterministic_and_inclusive():
    assert sample_segment_frame_indices(
        start_frame=2,
        end_frame=8,
        sample_count=4,
    ) == [2, 4, 6, 8]
    assert sample_segment_frame_indices(
        start_frame=2,
        end_frame=4,
        sample_count=10,
    ) == [2, 3, 4]


def test_build_segment_frame_grid_returns_metadata(tmp_path: Path):
    video_path = tmp_path / "test.avi"
    _write_test_video(video_path, fps=10.0, frames=6)

    result = build_segment_frame_grid(
        video_path,
        start_frame=1,
        end_frame=5,
        sample_count=3,
        columns=3,
        tile_width=32,
        tile_height=24,
        annotate=False,
    )

    assert result.frame_indices == [1, 3, 5]
    assert result.rows == 1
    assert result.columns == 3
    assert result.image.shape[:2] == (24, 96)
    assert result.metadata()["frames"][0]["frame_index"] == 1
