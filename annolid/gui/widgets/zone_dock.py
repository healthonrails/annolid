from __future__ import annotations

from typing import Optional

from qtpy import QtWidgets

from annolid.gui.widgets.zone_panel import ZonePanelWidget


class ZoneDockWidget(QtWidgets.QDockWidget):
    """Dock widget that hosts the zone authoring workflow on the live canvas."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        zone_path: str | None = None,
    ) -> None:
        title = (
            parent.tr("Zones")
            if parent is not None and hasattr(parent, "tr")
            else "Zones"
        )
        super().__init__(title, parent)
        self.setObjectName("zoneDock")
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self._zone_panel = ZonePanelWidget(parent=parent, zone_path=zone_path)
        self._zone_panel.close_button.setText("Hide Dock")
        try:
            self._zone_panel.close_button.clicked.disconnect(self._zone_panel.close)
        except Exception:
            pass
        self._zone_panel.close_button.clicked.connect(self.close)
        self.setWidget(self._zone_panel)
        self.visibilityChanged.connect(self._on_visibility_changed)

    @property
    def zone_panel(self) -> ZonePanelWidget:
        return self._zone_panel

    def refresh_from_current_canvas(self) -> None:
        self._zone_panel.refresh_from_current_canvas()

    def _on_visibility_changed(self, visible: bool) -> None:
        if visible:
            self._zone_panel.refresh_from_current_canvas()

    def closeEvent(self, event) -> None:  # noqa: N802
        if bool(getattr(self._zone_panel, "_dirty", False)):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved changes",
                "Save zone changes before closing the dock?",
                QtWidgets.QMessageBox.Yes
                | QtWidgets.QMessageBox.No
                | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes,
            )
            if reply == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            if (
                reply == QtWidgets.QMessageBox.Yes
                and not self._zone_panel.save_zone_file()
            ):
                event.ignore()
                return
        if bool(getattr(self._zone_panel, "_policy_dirty", False)):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved policy changes",
                "Save zone occupancy policy changes before closing the dock?",
                QtWidgets.QMessageBox.Yes
                | QtWidgets.QMessageBox.No
                | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes,
            )
            if reply == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            if (
                reply == QtWidgets.QMessageBox.Yes
                and not self._zone_panel.save_zone_policy_file()
            ):
                event.ignore()
                return
        try:
            self._zone_panel._clear_zone_defaults()
        except Exception:
            pass
        super().closeEvent(event)
