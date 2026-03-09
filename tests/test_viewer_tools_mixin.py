from __future__ import annotations

import json
import os
import time
from pathlib import Path

from qtpy import QtWidgets

from annolid.gui.mixins.viewer_tools_mixin import (
    ViewerToolsMixin,
    _is_recent_live_flybody_payload,
)


def test_is_recent_live_flybody_payload_accepts_recent_live_payload(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "flybody_live_rollout.json"
    payload_path.write_text(
        json.dumps({"kind": "annolid-simulation-v1", "adapter": "flybody-live"}),
        encoding="utf-8",
    )

    assert _is_recent_live_flybody_payload(payload_path, max_age_seconds=60.0) is True


def test_is_recent_live_flybody_payload_rejects_stale_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "flybody_live_rollout.json"
    payload_path.write_text(
        json.dumps({"kind": "annolid-simulation-v1", "adapter": "flybody-live"}),
        encoding="utf-8",
    )
    stale_time = time.time() - 120.0
    os.utime(payload_path, (stale_time, stale_time))

    assert _is_recent_live_flybody_payload(payload_path, max_age_seconds=5.0) is False


def test_is_recent_live_flybody_payload_rejects_non_live_payload(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "flybody_live_rollout.json"
    payload_path.write_text(
        json.dumps({"kind": "annolid-simulation-v1", "adapter": "flybody"}),
        encoding="utf-8",
    )

    assert _is_recent_live_flybody_payload(payload_path, max_age_seconds=60.0) is False


class _DummyStatusBar:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def showMessage(self, text: str, timeout: int = 0) -> None:
        self.messages.append((text, timeout))


class _DummyManager:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def show_simulation_in_viewer(self, path) -> bool:
        self.paths.append(Path(path))
        return True


class _DummyViewerHost(ViewerToolsMixin):
    def __init__(self) -> None:
        self.threejs_manager = _DummyManager()
        self._status = _DummyStatusBar()

    def statusBar(self):
        return self._status

    def tr(self, text: str) -> str:
        return text


def test_open_threejs_example_flybody_stays_on_fast_example_path(
    tmp_path: Path, monkeypatch
) -> None:
    widget = _DummyViewerHost()
    example_path = tmp_path / "flybody.json"
    example_path.write_text(
        json.dumps(
            {"kind": "annolid-simulation-v1", "adapter": "flybody", "frames": []}
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin.generate_threejs_example",
        lambda example_id, out_dir: example_path,
    )
    monkeypatch.setattr(
        widget,
        "_start_live_flybody_example",
        lambda manager: (_ for _ in ()).throw(
            AssertionError("live path should not run")
        ),
    )

    widget.open_threejs_example("flybody_simulation_json")

    assert widget.threejs_manager.paths == [example_path]
    assert widget.statusBar().messages[-1][0].startswith("Loaded FlyBody 3D example.")


def test_start_live_flybody_example_shows_static_example_first_when_runtime_missing(
    tmp_path: Path, monkeypatch
) -> None:
    widget = _DummyViewerHost()
    example_path = tmp_path / "flybody.json"
    example_path.write_text(
        json.dumps(
            {"kind": "annolid-simulation-v1", "adapter": "flybody", "frames": []}
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin.generate_threejs_example",
        lambda example_id, out_dir: example_path,
    )
    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin.pick_ready_flybody_runtime",
        lambda: (None, {}),
    )
    shown: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: shown.append("info"),
    )

    widget.start_live_flybody_example()

    assert widget.threejs_manager.paths == [example_path]
    assert shown == ["info"]
    assert any(
        "Loaded FlyBody 3D example." in message
        for message, _timeout in widget.statusBar().messages
    )
