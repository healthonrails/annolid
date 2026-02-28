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
        self._loaded_files: list[str] = []
        self._selected_file_items: list[str] = []
        self.caption_widget = None
        self._pending_last_worked_file = ""
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

    def loadFile(self, filename: str):  # noqa: N802
        self.filename = filename
        self._loaded_files.append(filename)

    def _set_current_file_item(self, filename: str):
        self._selected_file_items.append(filename)

    def _pending_last_worked_file_for_directory(self, directory: str) -> str:
        pending = str(getattr(self, "_pending_last_worked_file", "") or "").strip()
        if not pending:
            return ""
        pending_path = Path(pending).expanduser()
        if not pending_path.exists() or not pending_path.is_file():
            self._pending_last_worked_file = ""
            return ""
        try:
            Path(directory).expanduser().resolve()
            pending_path.resolve().relative_to(Path(directory).expanduser().resolve())
        except Exception:
            return ""
        return str(pending_path)

    def _clear_pending_last_worked_file(self) -> None:
        self._pending_last_worked_file = ""


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


def test_import_dir_images_restores_pending_last_worked_file_only_in_opened_dir(
    tmp_path: Path,
):
    app = _ensure_qapp()
    window = _DummyFileBrowserWindow()
    try:
        wrong_dir = tmp_path / "wrong"
        right_dir = tmp_path / "right"
        wrong_dir.mkdir()
        right_dir.mkdir()

        wrong_png = wrong_dir / "other.png"
        wrong_json = wrong_dir / "other.json"
        wrong_png.write_bytes(b"\x89PNG\r\n\x1a\n")
        wrong_json.write_text("{}", encoding="utf-8")

        target_png = right_dir / "target.png"
        target_json = right_dir / "target.json"
        target_png.write_bytes(b"\x89PNG\r\n\x1a\n")
        target_json.write_text("{}", encoding="utf-8")

        window._pending_last_worked_file = str(target_png)

        window.importDirImages(str(wrong_dir), load=True)
        for _ in range(200):
            app.processEvents()
            if not bool(getattr(window, "_dir_scan_running", False)):
                break
        assert window._open_next_calls == 1
        assert window._loaded_files == []
        assert window._pending_last_worked_file == str(target_png)

        window.importDirImages(str(right_dir), load=True)
        for _ in range(200):
            app.processEvents()
            if not bool(getattr(window, "_dir_scan_running", False)):
                break
        assert window._loaded_files == [str(target_png)]
        assert window._selected_file_items == [str(target_png)]
        assert window._open_next_calls == 1
        assert window._pending_last_worked_file == ""
    finally:
        window.close()


def test_import_dir_images_restores_when_pending_file_is_json_but_dock_has_images(
    tmp_path: Path,
):
    app = _ensure_qapp()
    window = _DummyFileBrowserWindow()
    try:
        target_png = tmp_path / "target.png"
        target_json = tmp_path / "target.json"
        target_png.write_bytes(b"\x89PNG\r\n\x1a\n")
        target_json.write_text("{}", encoding="utf-8")

        # Session can save JSON as last worked file; restore should still open image row.
        window._pending_last_worked_file = str(target_json)

        window.importDirImages(str(tmp_path), load=True)
        for _ in range(200):
            app.processEvents()
            if not bool(getattr(window, "_dir_scan_running", False)):
                break

        assert window._loaded_files == [str(target_png)]
        assert window._selected_file_items == [str(target_png)]
        assert window._pending_last_worked_file == ""
        assert window._open_next_calls == 0
    finally:
        window.close()


def test_import_dir_images_waits_for_pending_restore_across_scan_batches(
    tmp_path: Path,
):
    _ensure_qapp()
    window = _DummyFileBrowserWindow()
    try:
        target_png = tmp_path / "frame_0500.png"
        target_png.write_bytes(b"\x89PNG\r\n\x1a\n")
        window._pending_last_worked_file = str(target_png)

        # Keep this test deterministic by bypassing background worker execution.
        window._start_import_dir_scan_worker = lambda _dir: None
        window.importDirImages(str(tmp_path), load=True)

        first_batch = []
        for idx in range(1, 321):
            path = tmp_path / f"frame_{idx:04d}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            first_batch.append(str(path))
        window._consume_import_dir_scan_paths(first_batch)

        # While pending target is not seen yet, scan should wait instead of opening first file.
        assert window._open_next_calls == 0
        assert window._loaded_files == []

        second_batch = [str(target_png)]
        window._consume_import_dir_scan_paths(second_batch)

        assert window._loaded_files == [str(target_png)]
        assert window._selected_file_items == [str(target_png)]
        assert window._open_next_calls == 0
        assert window._pending_last_worked_file == ""
    finally:
        window.close()
