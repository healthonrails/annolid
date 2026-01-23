from __future__ import annotations
from annolid.utils.annotation_store import AnnotationStore, load_labelme_json
from annolid.engine.cli import main as annolid_run

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")


def _write_tiny_video(path: Path, *, frames: int = 2, fps: int = 5) -> None:
    import cv2  # type: ignore
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (16, 16))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment.")
    try:
        for idx in range(frames):
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            img[:, :, 1] = idx * 80
            writer.write(img)
    finally:
        writer.release()


def test_annolid_run_agent_writes_store(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_tiny_video(video_path)

    rc = annolid_run(["agent", "--video", str(video_path), "--max-frames", "2"])
    assert rc == 0

    results_dir = video_path.with_suffix("")
    store_stub = results_dir / f"{results_dir.name}_000000000.json"
    store = AnnotationStore.for_frame_path(store_stub)
    assert store.store_path.exists()

    frames = list(store.iter_frames())
    assert frames
    candidate = results_dir / f"{results_dir.name}_{frames[0]:09}.json"
    payload = load_labelme_json(candidate)
    assert payload["imageHeight"] == 16
    assert payload["imageWidth"] == 16
