from __future__ import annotations
from annolid.utils.annotation_store import AnnotationStore, load_labelme_json
from annolid.core.behavior.spec import default_behavior_spec, save_behavior_spec
from annolid.core.agent.service import run_agent_to_results

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")


def _write_tiny_video(path: Path, *, frames: int = 3, fps: int = 5) -> None:
    import cv2  # type: ignore
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (16, 16))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment.")
    try:
        for idx in range(frames):
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            img[:, :, 2] = idx * 50
            writer.write(img)
    finally:
        writer.release()


def test_agent_service_writes_annotation_store(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_tiny_video(video_path)

    schema = default_behavior_spec()
    schema.behaviors[0].code = "digging"
    save_behavior_spec(schema, tmp_path / "project.annolid.json")

    result = run_agent_to_results(video_path=video_path)
    assert result.records_written > 0
    assert result.store_path.exists()
    assert result.ndjson_path.exists()

    store_stub = result.results_dir / f"{result.results_dir.name}_000000000.json"
    store = AnnotationStore.for_frame_path(store_stub)
    frames = list(store.iter_frames())
    assert frames

    candidate = result.results_dir / f"{result.results_dir.name}_{frames[0]:09}.json"
    payload = load_labelme_json(candidate)
    assert payload["imageHeight"] == 16
    assert payload["imageWidth"] == 16
    assert isinstance(payload["shapes"], list)

    first = json.loads(result.ndjson_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["video_name"] == "tiny.avi"
