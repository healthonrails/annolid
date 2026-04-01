from pathlib import Path

import cv2
import numpy as np
from qtpy import QtWidgets

from annolid.core.media.video import CV2Video
from annolid.gui.workers import LoadFrameThread


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _write_test_video(path: Path, *, fps: float = 10.0, frames: int = 3) -> None:
    width, height = 64, 48
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter is not available in this environment.")
    try:
        for idx in range(int(frames)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = idx * 10
            writer.write(frame)
    finally:
        writer.release()


def test_frame_loader_keeps_latest_request_only():
    _ensure_qapp()

    class _DummyVideoLoader:
        def __init__(self):
            self.calls = []

        def load_frame(self, frame_number: int):
            self.calls.append(frame_number)
            return np.zeros((8, 8, 3), dtype=np.uint8)

    worker = LoadFrameThread()
    worker.video_loader = _DummyVideoLoader()

    emitted = []
    worker.res_frame.connect(lambda idx, _img: emitted.append(idx))

    worker.request(1)
    worker.request(8)
    worker.request(15)
    worker.load()

    assert worker.video_loader.calls == [15]
    assert emitted == [15]


def test_frame_loader_shutdown_releases_video_loader():
    _ensure_qapp()

    class _DummyVideoLoader:
        def __init__(self):
            self.released = False

        def load_frame(self, _frame_number: int):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    video_loader = _DummyVideoLoader()
    worker = LoadFrameThread()
    worker.video_loader = video_loader
    worker.shutdown()

    assert video_loader.released is True


def test_cv2video_get_dimensions_without_decoding_first_frame(tmp_path: Path):
    video_path = tmp_path / "test.avi"
    _write_test_video(video_path, fps=10.0, frames=3)

    video = CV2Video(video_path)
    try:
        video.load_frame = lambda _frame_number: (_ for _ in ()).throw(
            RuntimeError("load_frame should not be used for dimensions")
        )
        assert video.get_width() == 64
        assert video.get_height() == 48
    finally:
        video.release()
