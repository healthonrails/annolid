from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt

from annolid.gui.widgets.pdf_viewer import PdfViewerWidget
from annolid.gui.widgets.pdf_controls import PdfControlsWidget
from annolid.gui.widgets.tts_controls import TtsControlsWidget
from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class PdfManager(QtCore.QObject):
    """Encapsulates PDF viewer, controls, and docks wiring for the main window."""

    def __init__(
        self, window: "AnnolidWindow", viewer_stack: QtWidgets.QStackedWidget
    ) -> None:
        super().__init__(window)
        self.window = window
        self.viewer_stack = viewer_stack
        self.pdf_viewer: Optional[PdfViewerWidget] = None
        self.pdf_tts_dock: Optional[QtWidgets.QDockWidget] = None
        self.pdf_tts_controls: Optional[TtsControlsWidget] = None
        self.pdf_controls_dock: Optional[QtWidgets.QDockWidget] = None
        self.pdf_controls_widget: Optional[PdfControlsWidget] = None
        self._pdf_files: list[str] = []
        self._pdf_file_signals_connected = False
        self._hidden_docks: list[QtWidgets.QDockWidget] = []
        self._labelme_file_selection_disabled = False

    # ------------------------------------------------------------------ setup
    def ensure_pdf_viewer(self) -> PdfViewerWidget:
        if self.pdf_viewer is None:
            viewer = PdfViewerWidget(self.window)
            viewer.page_changed.connect(self._on_page_changed)
            viewer.page_changed.connect(self._update_pdf_controls_page)
            viewer.controls_enabled_changed.connect(
                self._on_pdf_controls_enabled_changed
            )
            self.viewer_stack.addWidget(viewer)
            self.pdf_viewer = viewer
        return self.pdf_viewer

    def ensure_pdf_tts_dock(self) -> None:
        if self.pdf_tts_dock is None:
            dock = QtWidgets.QDockWidget(
                self.window.tr("PDF Speech"), self.window)
            dock.setObjectName("PdfTtsDock")
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            controls = TtsControlsWidget(dock)
            container = QtWidgets.QWidget(dock)
            lay = QtWidgets.QVBoxLayout(container)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(6)
            lay.addWidget(controls, alignment=Qt.AlignTop)
            lay.addStretch(1)
            container.setLayout(lay)
            container.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
            )
            dock.setWidget(container)
            self.window.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.pdf_tts_dock = dock
            self.pdf_tts_controls = controls
        if self.pdf_tts_dock is not None:
            self.pdf_tts_dock.show()
            self.pdf_tts_dock.raise_()
        self._connect_file_list_signals()

    def ensure_pdf_controls_dock(self) -> None:
        if self.pdf_controls_dock is None:
            dock = QtWidgets.QDockWidget(
                self.window.tr("PDF Controls"), self.window)
            dock.setObjectName("PdfControlsDock")
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            controls = PdfControlsWidget(dock)
            container = QtWidgets.QWidget(dock)
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(4)
            layout.addWidget(controls, alignment=Qt.AlignTop)
            container.setLayout(layout)
            container.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
            )
            dock.setWidget(container)
            self.window.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.pdf_controls_dock = dock
            self.pdf_controls_widget = controls
            controls.previous_requested.connect(self._pdf_prev_page)
            controls.next_requested.connect(self._pdf_next_page)
            controls.reset_zoom_requested.connect(self._pdf_reset_zoom)
            controls.zoom_changed.connect(self._pdf_set_zoom)

        if self.pdf_controls_dock is not None:
            self.pdf_controls_dock.show()
            self.pdf_controls_dock.raise_()
        self._sync_pdf_controls_state()

    def _connect_file_list_signals(self) -> None:
        if self._pdf_file_signals_connected:
            return
        widget = getattr(self.window, "fileListWidget", None)
        if widget is None:
            return
        try:
            widget.itemActivated.connect(self._handle_file_list_activation)
            self._pdf_file_signals_connected = True
        except Exception:
            self._pdf_file_signals_connected = False

    # ------------------------------------------------------------------ actions
    def show_pdf_in_viewer(self, pdf_path: str) -> None:
        """Load a PDF into the viewer and display it in place of the canvas."""
        viewer = self.ensure_pdf_viewer()
        try:
            viewer.load_pdf(pdf_path)
        except Exception as exc:  # pragma: no cover - user-facing dialog
            logger.error("Failed to open PDF %s: %s",
                         pdf_path, exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self.window,
                self.window.tr("Failed to Open PDF"),
                self.window.tr("Could not open the selected PDF:\n%1").replace(
                    "%1", str(exc)
                ),
            )
            self.window._set_active_view("canvas")
            return

        self.window.video_loader = None
        self.window.filename = None
        self.window._set_active_view("pdf")
        self._disable_labelme_file_selection()
        self._close_unrelated_docks_for_pdf()
        self.ensure_pdf_tts_dock()
        self.ensure_pdf_controls_dock()
        self._record_pdf_entry(str(pdf_path))
        self.window.lastOpenDir = str(Path(pdf_path).parent)
        self.window.statusBar().showMessage(
            self.window.tr("Loaded PDF %1").replace(
                "%1", Path(pdf_path).name), 3000
        )

    def close_pdf(self) -> None:
        """Close PDF view, restore docks, and return to canvas."""
        # Restore docks hidden for PDF.
        self._restore_hidden_docks()
        self._restore_labelme_file_selection()
        # Hide PDF docks.
        for dock in (self.pdf_tts_dock, self.pdf_controls_dock):
            try:
                if dock is not None:
                    dock.hide()
            except Exception:
                pass
        # Switch back to canvas.
        try:
            self.window._set_active_view("canvas")
        except Exception:
            pass

    # ------------------------------------------------------------------ slots/helpers
    def _on_page_changed(self, current: int, total: int) -> None:
        try:
            self.window.statusBar().showMessage(
                self.window.tr("PDF page %1 of %2")
                .replace("%1", str(current + 1))
                .replace("%2", str(total)),
                3000,
            )
        except Exception:
            pass

    def _close_unrelated_docks_for_pdf(self) -> None:
        """Hide docks not useful when viewing PDFs to reduce clutter."""
        self._hidden_docks.clear()
        docks = [
            getattr(self.window, "behavior_log_dock", None),
            getattr(self.window, "behavior_controls_dock", None),
            getattr(self.window, "flag_dock", None),
            getattr(self.window, "audio_dock", None),
            getattr(self.window, "florence_dock", None),
            getattr(self.window, "video_dock", None),
            getattr(self.window, "label_dock", None),
            getattr(self.window, "shape_dock", None),
        ]
        for dock in docks:
            try:
                if dock is not None:
                    dock.hide()
                    self._hidden_docks.append(dock)
            except Exception:
                continue

        # Keep the file list visible for PDF navigation.
        file_dock = getattr(self.window, "file_dock", None)
        if file_dock is not None:
            file_dock.show()
            file_dock.raise_()

    def _restore_hidden_docks(self) -> None:
        """Show docks that were hidden for PDF viewing."""
        for dock in self._hidden_docks:
            try:
                dock.show()
            except Exception:
                continue
        self._hidden_docks.clear()

    def _disable_labelme_file_selection(self) -> None:
        """Prevent LabelMe's file list selection handler from firing on PDF items."""
        if self._labelme_file_selection_disabled:
            return
        widget = getattr(self.window, "fileListWidget", None)
        handler = getattr(self.window, "fileSelectionChanged", None)
        if widget is None or handler is None:
            return
        try:
            widget.itemSelectionChanged.disconnect(handler)
            self._labelme_file_selection_disabled = True
        except Exception:
            self._labelme_file_selection_disabled = False

    def _restore_labelme_file_selection(self) -> None:
        """Restore LabelMe's file list selection handler after closing PDFs."""
        if not self._labelme_file_selection_disabled:
            return
        widget = getattr(self.window, "fileListWidget", None)
        handler = getattr(self.window, "fileSelectionChanged", None)
        if widget is None or handler is None:
            self._labelme_file_selection_disabled = False
            return
        try:
            widget.itemSelectionChanged.connect(handler)
        except Exception:
            pass
        self._labelme_file_selection_disabled = False

    def _record_pdf_entry(self, pdf_path: str) -> None:
        try:
            resolved = str(Path(pdf_path).resolve())
        except Exception:
            resolved = pdf_path
        if resolved not in self._pdf_files:
            self._pdf_files.append(resolved)
        self._populate_pdf_file_list()

    def _populate_pdf_file_list(self) -> None:
        widget = getattr(self.window, "fileListWidget", None)
        if widget is None:
            return
        widget.clear()
        for path in self._pdf_files:
            name = Path(path).name
            item = QtWidgets.QListWidgetItem(name)
            item.setData(Qt.UserRole, path)
            item.setData(Qt.UserRole + 1, "pdf")
            item.setToolTip(path)
            widget.addItem(item)
        if widget.count() > 0:
            widget.setCurrentRow(widget.count() - 1)

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def _handle_file_list_activation(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is None:
            return
        kind = item.data(Qt.UserRole + 1)
        if kind != "pdf":
            return
        path = item.data(Qt.UserRole)
        if not path:
            return
        self.show_pdf_in_viewer(str(path))

    def _pdf_prev_page(self) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.previous_page()

    def _pdf_next_page(self) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.next_page()

    def _pdf_reset_zoom(self) -> None:
        if self.pdf_viewer is None:
            return
        self.pdf_viewer.reset_zoom()
        self._sync_pdf_controls_state()

    def _pdf_set_zoom(self, percent: float) -> None:
        if self.pdf_viewer is None:
            return
        self.pdf_viewer.set_zoom_percent(percent)
        self._sync_pdf_controls_state()

    def _update_pdf_controls_page(self, current: int, total: int) -> None:
        if self.pdf_controls_widget is not None:
            self.pdf_controls_widget.set_page_info(current, total)

    def _on_pdf_controls_enabled_changed(self, enabled: bool) -> None:
        if self.pdf_controls_widget is not None:
            reason = (
                self.window.tr("Navigation disabled in embedded browser mode.")
                if not enabled
                else ""
            )
            self.pdf_controls_widget.set_controls_enabled(
                enabled, reason=reason)

    def _sync_pdf_controls_state(self) -> None:
        viewer = self.pdf_viewer
        controls = self.pdf_controls_widget
        if viewer is None or controls is None:
            return
        controls.set_controls_enabled(viewer.controls_enabled())
        controls.set_zoom_percent(viewer.current_zoom_percent())
        total = viewer.page_count()
        controls.set_page_info(viewer.current_page_index(), total)

    # ------------------------------------------------------------------ helpers
    def pdf_widget(self) -> Optional[PdfViewerWidget]:
        return self.pdf_viewer
