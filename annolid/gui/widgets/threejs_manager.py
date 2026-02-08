from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from qtpy import QtCore, QtWidgets

from annolid.gui.threejs_support import supports_threejs_canvas
from annolid.gui.widgets.threejs_viewer import ThreeJsViewerWidget
from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class ThreeJsManager(QtCore.QObject):
    """Manage an embedded Three.js viewer in the shared stacked viewer area."""

    def __init__(
        self, window: "AnnolidWindow", viewer_stack: QtWidgets.QStackedWidget
    ) -> None:
        super().__init__(window)
        self.window = window
        self.viewer_stack = viewer_stack
        self.threejs_viewer: Optional[ThreeJsViewerWidget] = None

    def ensure_threejs_viewer(self) -> ThreeJsViewerWidget:
        if self.threejs_viewer is None:
            viewer = ThreeJsViewerWidget(self.window)
            viewer.status_changed.connect(
                lambda msg: self.window.statusBar().showMessage(msg, 3000)
            )
            self.viewer_stack.addWidget(viewer)
            self.threejs_viewer = viewer
        return self.threejs_viewer

    def is_supported(self, path: str | Path) -> bool:
        return supports_threejs_canvas(path)

    def show_model_in_viewer(self, model_path: str | Path) -> bool:
        path = Path(model_path)
        if not self.is_supported(path):
            return False
        viewer = self.ensure_threejs_viewer()
        try:
            viewer.load_model(path)
        except Exception as exc:
            logger.warning("Failed to load model in Three.js viewer: %s", exc)
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("Three.js Viewer"),
                self.window.tr("Unable to load model in Three.js canvas:\n%1").replace(
                    "%1", str(exc)
                ),
            )
            return False
        # Hide unrelated docks when switching to the 3D canvas.
        self.window.set_unrelated_docks_visible(False)
        self.window._set_active_view("threejs")
        self.window.statusBar().showMessage(
            self.window.tr("Loaded 3D model %1").replace("%1", path.name), 3000
        )
        return True

    def viewer_widget(self) -> Optional[ThreeJsViewerWidget]:
        return self.threejs_viewer

    def close_threejs(self) -> None:
        """Close Three.js 3D view and return to canvas."""
        # Switch back to canvas.
        try:
            self.window._set_active_view("canvas")
        except Exception:
            pass
