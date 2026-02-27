from __future__ import annotations

import os
from types import SimpleNamespace
from pathlib import Path

from qtpy import QtWidgets

from annolid.gui.file_dock import FileDockMixin
from annolid.gui.mixins.file_browser_mixin import FileBrowserMixin


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


class _DummyFileBrowserWindow(FileBrowserMixin, FileDockMixin, QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.actions = SimpleNamespace(
            openNextImg=QtWidgets.QAction("next", self),
            openPrevImg=QtWidgets.QAction("prev", self),
        )
        self.imageList: list[str] = []
        self.lastOpenDir: str | None = None
        self.filename = None
        self.annotation_dir = ""
        self._open_next_calls = 0
        self._init_file_dock_ui()
        self.setStatusBar(QtWidgets.QStatusBar(self))

    def mayContinue(self) -> bool:  # noqa: N802
        return True

    def _getLabelFile(self, filename):  # noqa: N802
        return str(Path(filename).with_suffix(".json"))

    def _annotation_store_has_frame(self, _label_file) -> bool:
        return False

    def openNextImg(self, _value=False, load=True):  # noqa: N802
        if load:
            self._open_next_calls += 1


def test_import_dir_images_scans_asynchronously_and_keeps_image_mode(tmp_path: Path):
    app = _ensure_qapp()
    window = _DummyFileBrowserWindow()
    try:
        for idx in range(4):
            (tmp_path / f"frame_{idx:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (tmp_path / f"frame_{idx:04d}.json").write_text("{}", encoding="utf-8")

        window.importDirImages(str(tmp_path), load=True)

        for _ in range(200):
            app.processEvents()
            if not bool(getattr(window, "_dir_scan_running", False)):
                break

        assert not bool(getattr(window, "_dir_scan_running", False))
        assert len(window.imageList) == 4
        assert all(path.endswith(".png") for path in window.imageList)
        assert window._open_next_calls == 1
        assert not window.fileScanStatusLabel.isHidden()
        assert window.fileScanStatusLabel.text() == "Loaded 4"
    finally:
        window.close()
