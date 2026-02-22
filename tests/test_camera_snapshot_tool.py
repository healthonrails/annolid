from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import numpy as np

from annolid.core.agent.tools.camera import (
    CameraSnapshotTool,
    _normalize_stream_source,
    _unwrap_safelinks_url,
)


def test_unwrap_safelinks_and_normalize_source() -> None:
    wrapped = (
        "https://nam12.safelinks.protection.outlook.com/"
        "?url=http%3A%2F%2F192.168.1.21%2Fimg%2Fmain.cgi%3Fnext_file%3Dmain.htm"
        "&data=foo"
    )
    unwrapped = _unwrap_safelinks_url(wrapped)
    assert unwrapped.startswith("http://192.168.1.21/")
    normalized = _normalize_stream_source(wrapped)
    assert str(normalized) == "http://192.168.1.21/img/video.mjpeg"


def test_camera_snapshot_tool_rejects_invalid_source(tmp_path: Path) -> None:
    tool = CameraSnapshotTool(allowed_dir=tmp_path)
    result = asyncio.run(tool.execute(camera_source="../../etc/passwd"))
    payload = json.loads(result)
    assert payload["ok"] is False
    assert "camera_source must be a camera index or stream URL" in payload["error"]


def test_camera_snapshot_tool_captures_frame_with_fake_cv2(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeCapture:
        def __init__(self) -> None:
            self._opened = False

        def set(self, *_args) -> bool:
            return True

        def open(self, source) -> bool:
            del source
            self._opened = True
            return True

        def get(self, key) -> float:
            if key == 3:
                return 640.0
            if key == 4:
                return 480.0
            if key == 5:
                return 25.0
            return 0.0

        def read(self):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return True, frame

        def release(self) -> None:
            self._opened = False

    def _imwrite(path: str, _frame) -> bool:
        Path(path).write_bytes(b"fake-jpg")
        return True

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=_FakeCapture,
        imwrite=_imwrite,
        CAP_PROP_OPEN_TIMEOUT_MSEC=10001,
        CAP_PROP_READ_TIMEOUT_MSEC=10002,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    tool = CameraSnapshotTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            camera_source="http://camera.local/img/video.mjpeg",
            filename="probe.jpg",
        )
    )
    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["snapshot_path"].endswith("camera_snapshots/probe.jpg")
    assert Path(payload["snapshot_path"]).exists()
