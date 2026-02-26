from __future__ import annotations

import os
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.file_dock import FileDockMixin


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


class _DummyPdfManager:
    def __init__(self) -> None:
        self.last_opened = ""

    def show_pdf_in_viewer(self, path: str) -> None:
        self.last_opened = str(path or "")


class _DummyFileDockWindow(FileDockMixin, QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.imageList: list[str] = []
        self.filename: str | None = None
        self.imagePath: str | None = None
        self.lastOpenDir: str | None = None
        self.loaded_paths: list[str] = []
        self.import_calls: list[tuple[str, bool]] = []
        self.errors: list[tuple[str, str]] = []
        self.reset_calls = 0
        self._may_continue = True
        self._pdf_manager = _DummyPdfManager()
        self._init_file_dock_ui()
        self.setStatusBar(QtWidgets.QStatusBar(self))

    def loadFile(self, path: str) -> None:  # noqa: N802
        self.loaded_paths.append(str(path))
        self.filename = str(path)

    def importDirImages(self, dirpath: str, load: bool = True):  # noqa: N802
        self.import_calls.append((str(dirpath), bool(load)))

    def _addItem(self, filename, label_file, *, apply_updates=True):  # noqa: N802
        del label_file, apply_updates
        self.fileListWidget.addItem(QtWidgets.QListWidgetItem(str(filename)))

    def errorMessage(self, title: str, message: str) -> None:  # noqa: N802
        self.errors.append((str(title), str(message)))

    def mayContinue(self) -> bool:  # noqa: N802
        return bool(self._may_continue)

    def resetState(self) -> None:  # noqa: N802
        self.reset_calls += 1


def test_file_dock_open_file_and_pdf_dispatch(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _DummyFileDockWindow()
    try:
        image_path = tmp_path / "sample.json"
        image_path.write_text("{}", encoding="utf-8")
        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\n")

        image_item = QtWidgets.QListWidgetItem(str(image_path))
        window.fileListWidget.addItem(image_item)
        window.fileListWidget.setCurrentItem(image_item)
        window._open_selected_file_from_dock()
        assert window.loaded_paths and window.loaded_paths[-1] == str(image_path)

        pdf_item = QtWidgets.QListWidgetItem(pdf_path.name)
        pdf_item.setData(QtCore.Qt.UserRole, str(pdf_path))
        pdf_item.setData(QtCore.Qt.UserRole + 1, "pdf")
        window.fileListWidget.addItem(pdf_item)
        window.fileListWidget.setCurrentItem(pdf_item)
        window._open_selected_file_from_dock()
        assert window._pdf_manager.last_opened == str(pdf_path)
    finally:
        window.close()


def test_file_dock_open_file_dispatches_pdf_by_suffix_without_role(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyFileDockWindow()
    try:
        pdf_path = tmp_path / "untagged.PDF"
        pdf_path.write_bytes(b"%PDF-1.7\n")

        pdf_item = QtWidgets.QListWidgetItem(str(pdf_path))
        window.fileListWidget.addItem(pdf_item)
        window.fileListWidget.setCurrentItem(pdf_item)
        window._open_selected_file_from_dock()

        assert window._pdf_manager.last_opened == str(pdf_path)
        assert not window.loaded_paths
    finally:
        window.close()


def test_file_dock_rename_selected_file(tmp_path: Path, monkeypatch) -> None:
    _ensure_qapp()
    window = _DummyFileDockWindow()
    try:
        source = tmp_path / "before.json"
        source.write_text("{}", encoding="utf-8")
        item = QtWidgets.QListWidgetItem(str(source))
        window.fileListWidget.addItem(item)
        window.fileListWidget.setCurrentItem(item)
        window.imageList = [str(source)]
        window.filename = str(source)

        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getText",
            lambda *args, **kwargs: ("after.json", True),
        )
        window._rename_selected_file_from_dock()

        target = tmp_path / "after.json"
        assert target.exists()
        assert not source.exists()
        assert window.filename == str(target)
        assert window.imageList == [str(target)]
        assert window.fileListWidget.currentItem().text() == str(target)
    finally:
        window.close()


def test_file_dock_delete_selected_file(tmp_path: Path, monkeypatch) -> None:
    _ensure_qapp()
    window = _DummyFileDockWindow()
    try:
        path = tmp_path / "delete_me.json"
        path.write_text("{}", encoding="utf-8")
        item = QtWidgets.QListWidgetItem(str(path))
        window.fileListWidget.addItem(item)
        window.fileListWidget.setCurrentItem(item)
        window.imageList = [str(path)]
        window.filename = str(path)

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.Yes,
        )
        window._delete_selected_file_from_dock()

        assert not path.exists()
        assert window.fileListWidget.count() == 0
        assert window.imageList == []
        assert window.reset_calls == 1
    finally:
        window.close()


def test_file_dock_refresh_and_shortcuts(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _DummyFileDockWindow()
    try:
        window.lastOpenDir = str(tmp_path)
        window.filename = str(tmp_path / "existing.json")
        item = QtWidgets.QListWidgetItem(window.filename)
        window.fileListWidget.addItem(item)
        window._refresh_file_dock_listing()
        assert window.import_calls == [(str(tmp_path), False)]

        keys = {shortcut.key().toString() for shortcut in window._file_dock_shortcuts}
        assert QtGui.QKeySequence(QtCore.Qt.Key_F2).toString() in keys
        assert QtGui.QKeySequence(QtCore.Qt.Key_Delete).toString() in keys
    finally:
        window.close()


def test_file_dock_sort_by_name_and_date(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _DummyFileDockWindow()
    try:
        alpha = tmp_path / "alpha.json"
        beta = tmp_path / "beta.json"
        alpha.write_text("{}", encoding="utf-8")
        beta.write_text("{}", encoding="utf-8")
        os.utime(alpha, (1_000_000_000, 1_000_000_000))
        os.utime(beta, (1_000_000_100, 1_000_000_100))

        window.fileListWidget.addItem(QtWidgets.QListWidgetItem(str(beta)))
        window.fileListWidget.addItem(QtWidgets.QListWidgetItem(str(alpha)))

        window.fileSortCombo.setCurrentIndex(window.fileSortCombo.findData("name"))
        window.fileSortOrderButton.setChecked(False)
        window._apply_file_dock_sort()
        ordered = [window.fileListWidget.item(i).text() for i in range(2)]
        assert ordered == [str(alpha), str(beta)]

        window.fileSortCombo.setCurrentIndex(window.fileSortCombo.findData("date"))
        window.fileSortOrderButton.setChecked(True)
        window._apply_file_dock_sort()
        ordered = [window.fileListWidget.item(i).text() for i in range(2)]
        assert ordered == [str(beta), str(alpha)]
    finally:
        window.close()


def test_file_dock_incremental_queue_loads_in_batches(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _DummyFileDockWindow()
    try:
        window._file_dock_batch_size = 2
        entries: list[tuple[str, str]] = []
        for i in range(5):
            p = tmp_path / f"f{i}.json"
            p.write_text("{}", encoding="utf-8")
            entries.append((str(p), str(p)))

        window._queue_file_dock_entries(entries)
        assert window.fileListWidget.count() == 5  # initial eager chunk is 300 min
        assert len(window._file_dock_pending_entries) == 0

        # Re-test with manual bounded chunk.
        window.fileListWidget.clear()
        window._reset_file_dock_incremental_state()
        window._file_dock_pending_entries = list(entries)
        window._load_more_file_dock_items(batch_size=2)
        assert window.fileListWidget.count() == 2
        assert len(window._file_dock_pending_entries) == 3
    finally:
        window.close()
