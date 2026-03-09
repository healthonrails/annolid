from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from qtpy import QtWidgets

from annolid.gui.widgets.threejs_manager import ThreeJsManager


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


class _DummyStatusBar:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def showMessage(self, text: str, timeout: int = 0) -> None:
        self.messages.append((text, timeout))


class _DummyWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._status = _DummyStatusBar()
        self.actions = SimpleNamespace(close=None)

    def statusBar(self):
        return self._status

    def set_unrelated_docks_visible(self, _visible: bool) -> None:
        return None

    def _set_active_view(self, _view: str) -> None:
        return None


class _DummyViewer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str | None]] = []

    def load_simulation_payload(
        self, path: str | Path, title: str | None = None
    ) -> None:
        self.calls.append((Path(path), title))


def test_show_simulation_in_viewer_accepts_prebuilt_payload(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))
    viewer = _DummyViewer()
    monkeypatch.setattr(manager, "ensure_threejs_viewer", lambda: viewer)

    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps({"kind": "annolid-simulation-v1", "frames": []}),
        encoding="utf-8",
    )

    assert manager.show_simulation_in_viewer(payload_path) is True
    assert viewer.calls == [(payload_path, "payload")]


def test_show_simulation_in_viewer_exports_non_payload_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))
    viewer = _DummyViewer()
    monkeypatch.setattr(manager, "ensure_threejs_viewer", lambda: viewer)

    source_path = tmp_path / "simulation.ndjson"
    source_path.write_text("{}", encoding="utf-8")
    payload_path = tmp_path / "exported.json"
    payload_path.write_text(
        json.dumps({"kind": "annolid-simulation-v1", "frames": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.threejs_manager.export_simulation_view_payload",
        lambda path: payload_path,
    )

    assert manager.show_simulation_in_viewer(source_path) is True
    assert viewer.calls == [(payload_path, "simulation")]
