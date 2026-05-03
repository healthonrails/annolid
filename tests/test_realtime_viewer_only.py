from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np

from annolid.realtime.config import Config
from annolid.realtime.perception import PerceptionProcess


def test_viewer_only_loop_publishes_raw_frame_without_inference() -> None:
    config = Config(
        camera_index=0,
        camera_id="arena_a",
        model_base_name="",
        viewer_only=True,
        publish_frames=True,
        publish_annotated_frames=True,
        save_detection_segments=True,
        max_fps=30.0,
    )
    process = PerceptionProcess(config)
    original_publisher = process.publisher

    class FakeSource:
        state = SimpleNamespace(name="LOCAL")

        def pop_status_events(self):
            return []

        async def get_frame(self):
            return np.zeros((4, 4, 3), dtype=np.uint8), {"source": "unit-test"}

    class FakePublisher:
        def __init__(self):
            self.frames = []

        async def publish_frame(self, frame, metadata, **kwargs):
            self.frames.append((frame, metadata, kwargs))
            process.request_stop()

        async def publish_status(self, payload):
            return None

    fake_publisher = FakePublisher()

    async def fail_if_inference_runs(_frame):
        raise AssertionError("viewer-only mode must not run inference")

    process.video_source = FakeSource()
    process.publisher = fake_publisher
    process._run_inference = fail_if_inference_runs

    try:
        asyncio.run(process._run_viewer_only_loop(frame_interval=0.0))
    finally:
        original_publisher.context.destroy(linger=0)

    assert len(fake_publisher.frames) == 1
    frame, metadata, kwargs = fake_publisher.frames[0]
    assert frame.shape == (4, 4, 3)
    assert metadata["camera_id"] == "arena_a"
    assert metadata["source"] == "unit-test"
    assert metadata["viewer_only"] is True
    assert metadata["processing"] is False
    assert metadata["detection_count"] == 0
    assert metadata["inference_ms"] == 0.0
    assert kwargs["encoding"] == "jpg"
