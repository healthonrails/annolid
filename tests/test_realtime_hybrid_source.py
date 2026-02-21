from __future__ import annotations

import asyncio
import time

import numpy as np

from annolid.realtime.config import Config
from annolid.realtime.perception import (
    HybridVideoSource,
    RecordingStateManager,
    SourceState,
)


class _FakeRemote:
    def __init__(self, connect_ok: bool = False):
        self.connect_ok = connect_ok
        self.connect_calls = 0
        self.last_error = "remote unavailable"
        self._is_active = False

    async def connect(self) -> bool:
        self.connect_calls += 1
        self._is_active = self.connect_ok
        return self.connect_ok

    async def get_frame(self):
        return None

    async def disconnect(self) -> None:
        self._is_active = False


class _FakeLocal:
    def __init__(self, connect_ok: bool = True):
        self.connect_ok = connect_ok
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.last_error = "local unavailable"
        self.frames: list[object] = []

    async def connect(self) -> bool:
        self.connect_calls += 1
        return self.connect_ok

    async def get_frame(self):
        if self.frames:
            return self.frames.pop(0)
        return None

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


def test_hybrid_source_prefers_explicit_local_stream_over_remote() -> None:
    cfg = Config(camera_index="http://camera.local/img/video.mjpeg")
    src = HybridVideoSource(cfg, RecordingStateManager(cfg))
    src.remote = _FakeRemote(connect_ok=False)  # type: ignore[assignment]
    src.local = _FakeLocal(connect_ok=True)  # type: ignore[assignment]

    asyncio.run(src.connect())

    assert src.state == SourceState.USING_LOCAL
    assert src.remote.connect_calls == 0


def test_hybrid_source_local_no_frame_tolerance_before_reset() -> None:
    cfg = Config(camera_index="http://camera.local/img/video.mjpeg")
    cfg.local_no_frame_tolerance = 3
    src = HybridVideoSource(cfg, RecordingStateManager(cfg))
    src.remote = _FakeRemote(connect_ok=False)  # type: ignore[assignment]
    src.local = _FakeLocal(connect_ok=True)  # type: ignore[assignment]
    src.state = SourceState.USING_LOCAL
    src._next_local_reconnect_time = time.time() + 60.0

    assert asyncio.run(src.get_frame()) is None
    assert src.state == SourceState.USING_LOCAL
    assert asyncio.run(src.get_frame()) is None
    assert src.state == SourceState.USING_LOCAL

    assert asyncio.run(src.get_frame()) is None
    assert src.state == SourceState.DISCONNECTED
    assert src.local.disconnect_calls == 1


def test_hybrid_source_remote_reconnect_backoff_grows_on_failures() -> None:
    cfg = Config(camera_index=0)
    cfg.remote_retry_cooldown = 1.0
    cfg.remote_retry_max_cooldown = 8.0
    src = HybridVideoSource(cfg, RecordingStateManager(cfg))
    src.remote = _FakeRemote(connect_ok=False)  # type: ignore[assignment]
    local = _FakeLocal(connect_ok=True)
    local.frames.append((np.zeros((8, 8, 3), dtype=np.uint8), {"source": "local"}))
    src.local = local  # type: ignore[assignment]
    src.state = SourceState.USING_LOCAL
    src.last_remote_attempt = 0.0

    result = asyncio.run(src.get_frame())

    assert result is not None
    assert src.remote.connect_calls == 1
    assert src._remote_retry_cooldown == 2.0
