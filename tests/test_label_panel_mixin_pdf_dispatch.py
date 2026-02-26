from __future__ import annotations

import os
from pathlib import Path

from qtpy import QtCore, QtWidgets

from annolid.gui.mixins.label_panel_mixin import LabelPanelMixin


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


class _DummyWindow(LabelPanelMixin, QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.fileListWidget = QtWidgets.QListWidget(self)
        self.video_loader = None
        self.filename = ""
        self.caption_widget = None
        self._pdf_manager = _DummyPdfManager()
        self.loaded: list[str] = []
        self.frame_updates = 0
        self.reset_calls = 0
        self.continue_ok = True

    def mayContinue(self) -> bool:  # noqa: N802
        return bool(self.continue_ok)

    def loadFile(self, path: str) -> None:  # noqa: N802
        self.loaded.append(str(path))
        self.filename = str(path)

    def resetState(self) -> None:  # noqa: N802
        self.reset_calls += 1

    def _update_frame_display_and_emit_update(self) -> None:
        self.frame_updates += 1


def test_file_list_current_item_routes_pdf_to_pdf_viewer(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    try:
        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\n")

        item = QtWidgets.QListWidgetItem(str(pdf_path))
        item.setCheckState(QtCore.Qt.Checked)
        window.fileListWidget.addItem(item)

        window._on_file_list_current_item_changed(item, None)

        assert window._pdf_manager.last_opened == str(pdf_path)
        assert window.loaded == []
        assert window.frame_updates == 0
    finally:
        window.close()


def test_file_list_current_item_routes_pdf_via_pdf_manager_attribute(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    try:
        pdf_path = tmp_path / "paper2.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\n")

        manager = _DummyPdfManager()
        window.pdf_manager = manager  # real app attribute
        window._pdf_manager = None

        item = QtWidgets.QListWidgetItem(str(pdf_path))
        item.setCheckState(QtCore.Qt.Checked)
        window.fileListWidget.addItem(item)

        window._on_file_list_current_item_changed(item, None)

        assert manager.last_opened == str(pdf_path)
        assert window.loaded == []
        assert window.frame_updates == 0
    finally:
        window.close()
