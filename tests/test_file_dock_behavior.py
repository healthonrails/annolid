from __future__ import annotations

import os
from collections import deque

from qtpy import QtCore, QtWidgets

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


class _FileDockProbe(QtWidgets.QMainWindow, FileDockMixin):
    def __init__(self) -> None:
        super().__init__()
        self._known_file_paths = set()
        self._init_file_dock_ui()

    def _addItem(self, filename, _label_file, *, apply_updates=True):
        if filename in self._known_file_paths:
            return
        item = QtWidgets.QListWidgetItem(filename)
        item.setFlags(
            QtCore.Qt.ItemIsEnabled
            | QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsUserCheckable
        )
        item.setCheckState(QtCore.Qt.Checked)
        self.fileListWidget.addItem(item)
        self._known_file_paths.add(filename)
        if apply_updates:
            self._apply_file_search_filter()


def test_file_dock_does_not_auto_sort_until_user_requests() -> None:
    _ensure_qapp()
    probe = _FileDockProbe()
    try:
        probe._append_file_dock_entries(
            [
                ("/tmp/b.png", "/tmp/b.json"),
                ("/tmp/a.png", "/tmp/a.json"),
            ]
        )
        probe._flush_file_dock_pending()

        assert probe.fileListWidget.count() == 2
        assert probe.fileListWidget.item(0).text().endswith("b.png")
        assert probe.fileListWidget.item(1).text().endswith("a.png")
    finally:
        probe.close()


def test_file_selection_counter_includes_pending_items() -> None:
    _ensure_qapp()
    probe = _FileDockProbe()
    try:
        probe._reset_file_dock_incremental_state()
        probe.fileListWidget.clear()
        probe._file_dock_pending_entries = deque(
            [
                ("/tmp/f1.png", "/tmp/f1.json"),
                ("/tmp/f2.png", "/tmp/f2.json"),
                ("/tmp/f3.png", "/tmp/f3.json"),
            ]
        )
        probe._load_more_file_dock_items(batch_size=1)

        probe.fileListWidget.setCurrentRow(0)
        probe._update_file_selection_counter()
        assert probe.fileSelectionLabel.text() == "1/3"
    finally:
        probe.close()
