from __future__ import annotations

import gc
import os
import threading
import time
from types import SimpleNamespace
from pathlib import Path
import weakref

from qtpy import QtCore, QtWidgets

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


def _wait_for_scan_idle(
    app: QtWidgets.QApplication,
    window: _DummyFileBrowserWindow,
    *,
    timeout_ms: int = 5000,
) -> None:
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    while time.monotonic() < deadline:
        app.processEvents()
        if not bool(getattr(window, "_dir_scan_running", False)):
            app.processEvents()
            return
        QtCore.QThread.msleep(2)
    app.processEvents()


def test_import_dir_images_scans_asynchronously_and_keeps_image_mode(tmp_path: Path):
    app = _ensure_qapp()
    window = _DummyFileBrowserWindow()
    try:
        for idx in range(4):
            (tmp_path / f"frame_{idx:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (tmp_path / f"frame_{idx:04d}.json").write_text("{}", encoding="utf-8")

        window.importDirImages(str(tmp_path), load=True)
        _wait_for_scan_idle(app, window)

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
        _wait_for_scan_idle(app, window)
        assert not bool(getattr(window, "_dir_scan_running", False))
        assert window._open_next_calls == 1
        assert window._loaded_files == []
        assert window._pending_last_worked_file == str(target_png)

        window.importDirImages(str(right_dir), load=True)
        _wait_for_scan_idle(app, window)
        assert not bool(getattr(window, "_dir_scan_running", False))
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
        _wait_for_scan_idle(app, window)

        assert not bool(getattr(window, "_dir_scan_running", False))
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


def test_import_dir_images_handles_finished_before_batch_signal_order(
    tmp_path: Path,
):
    app = _ensure_qapp()
    window = _DummyFileBrowserWindow()
    try:
        target_png = tmp_path / "target.png"
        target_json = tmp_path / "target.json"
        target_png.write_bytes(b"\x89PNG\r\n\x1a\n")
        target_json.write_text("{}", encoding="utf-8")

        window._pending_last_worked_file = str(target_json)
        window._start_import_dir_scan_worker = lambda _dir: None
        window.importDirImages(str(tmp_path), load=True)

        token = int(getattr(window, "_dir_scan_token", 0))
        # Simulate finished arriving before a queued final batch.
        window._on_import_dir_scan_finished(token)
        window._on_import_dir_scan_batch(token, [str(target_png)])

        for _ in range(50):
            app.processEvents()

        assert window._loaded_files == [str(target_png)]
        assert window._selected_file_items == [str(target_png)]
        assert window._open_next_calls == 0
        assert window._pending_last_worked_file == ""
    finally:
        window.close()


def test_import_dir_images_callbacks_execute_on_gui_thread_without_qt_warnings(
    tmp_path: Path,
):
    app = _ensure_qapp()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    for idx in range(6):
        (first_dir / f"a_{idx:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    for idx in range(3):
        (second_dir / f"b_{idx:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    qt_messages: list[str] = []

    def _qt_handler(_mode, _context, message: str) -> None:
        qt_messages.append(str(message))

    prev_handler = QtCore.qInstallMessageHandler(_qt_handler)
    try:
        window = _DummyFileBrowserWindow()
        win_ref = weakref.ref(window)
        main_ident = threading.get_ident()
        consume_idents: set[int] = set()
        finalize_idents: set[int] = set()

        def _tracked_consume(paths: list[str]) -> None:
            consume_idents.add(threading.get_ident())
            owner = win_ref()
            if owner is not None:
                _DummyFileBrowserWindow._consume_import_dir_scan_paths(owner, paths)

        def _tracked_finalize(token: int) -> None:
            finalize_idents.add(threading.get_ident())
            owner = win_ref()
            if owner is not None:
                _DummyFileBrowserWindow._finalize_import_dir_scan(owner, token)

        window._consume_import_dir_scan_paths = _tracked_consume  # type: ignore[method-assign]
        window._finalize_import_dir_scan = _tracked_finalize  # type: ignore[method-assign]

        # Back-to-back scan triggers cancel + restart followed by async completion.
        window.importDirImages(str(first_dir), load=True)
        window.importDirImages(str(second_dir), load=True)
        _wait_for_scan_idle(app, window)

        assert not bool(getattr(window, "_dir_scan_running", False))
        assert len(window.imageList) == 3
        assert window.fileScanStatusLabel.text() == "Loaded 3"
        assert consume_idents == {main_ident}
        assert finalize_idents == {main_ident}

        del window._consume_import_dir_scan_paths
        del window._finalize_import_dir_scan
        window.close()
        window.deleteLater()
        del window

        deferred_delete = getattr(
            getattr(QtCore.QEvent, "Type", QtCore.QEvent),
            "DeferredDelete",
            QtCore.QEvent.DeferredDelete,
        )
        for _ in range(3):
            app.sendPostedEvents(None, deferred_delete)
            app.processEvents()
        gc.collect()
        app.sendPostedEvents(None, deferred_delete)
        app.processEvents()
        gc.collect()

        assert win_ref() is None
    finally:
        QtCore.qInstallMessageHandler(prev_handler)

    thread_warnings = [
        msg
        for msg in qt_messages
        if "different thread" in msg or "moveToThread" in msg or "QThread" in msg
    ]
    assert thread_warnings == []
