from __future__ import annotations

from pathlib import Path

import numpy as np

from annolid.realtime.config import Config
from annolid.realtime.perception import DetectionSegmentRecorder


class _FakeVideoWriter:
    def __init__(self, path: str, _fourcc: int, _fps: float, _size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"")
        self._open = True
        self.frames_written = 0

    def isOpened(self) -> bool:  # noqa: N802
        return self._open

    def write(self, _frame) -> None:
        self.frames_written += 1
        self.path.write_bytes(f"frames={self.frames_written}".encode("utf-8"))

    def release(self) -> None:
        self._open = False


def test_detection_segment_recorder_matches_animal_alias() -> None:
    cfg = Config(
        save_detection_segments=True,
        detection_segment_targets=["animal", "person"],
    )
    recorder = DetectionSegmentRecorder(cfg)
    assert recorder._matches_target("dog") is True
    assert recorder._matches_target("person") is True
    assert recorder._matches_target("car") is False


def test_detection_segment_recorder_writes_segment_on_event(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("annolid.realtime.perception.cv2.VideoWriter", _FakeVideoWriter)
    monkeypatch.setattr(
        "annolid.realtime.perception.cv2.VideoWriter_fourcc", lambda *_args: 0
    )

    cfg = Config(
        save_detection_segments=True,
        detection_segment_targets=["person"],
        detection_segment_output_dir=str(tmp_path),
        detection_segment_prebuffer_sec=1.0,
        detection_segment_postbuffer_sec=1.0,
        detection_segment_min_duration_sec=0.0,
        detection_segment_max_duration_sec=10.0,
        max_fps=5.0,
    )
    recorder = DetectionSegmentRecorder(cfg)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    recorder.update(frame, 0.0, [])
    recorder.update(frame, 0.4, [])
    recorder.update(frame, 0.8, ["person"])
    recorder.update(frame, 1.2, [])
    recorder.update(frame, 1.9, [])
    recorder.close()

    saved = list(tmp_path.glob("segment_*.mp4"))
    assert len(saved) == 1
    assert saved[0].read_bytes().startswith(b"frames=")
