from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from qtpy import QtCore, QtWidgets

from annolid.gui.threejs_support import supports_threejs_canvas
from annolid.gui.widgets.threejs_viewer import ThreeJsViewerWidget
from annolid.simulation import export_simulation_view_payload
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
            viewer.flybody_command_requested.connect(
                self._handle_flybody_viewer_command
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
        try:
            close_action = getattr(getattr(self.window, "actions", None), "close", None)
            if close_action is not None:
                close_action.setEnabled(True)
        except Exception:
            pass
        self.window.statusBar().showMessage(
            self.window.tr("Loaded 3D model %1").replace("%1", path.name), 3000
        )
        return True

    def show_url_in_viewer(self, url: str) -> bool:
        viewer = self.ensure_threejs_viewer()
        try:
            viewer.load_url(url)
        except Exception as exc:
            logger.warning("Failed to load URL in Three.js viewer: %s", exc)
            return False
        self.window.set_unrelated_docks_visible(False)
        self.window._set_active_view("threejs")
        try:
            close_action = getattr(getattr(self.window, "actions", None), "close", None)
            if close_action is not None:
                close_action.setEnabled(True)
        except Exception:
            pass
        return True

    def show_simulation_in_viewer(self, simulation_path: str | Path) -> bool:
        path = Path(simulation_path)
        if not path.exists():
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("Simulation Viewer"),
                self.window.tr("Simulation file not found:\n%1").replace(
                    "%1", str(path)
                ),
            )
            return False
        viewer = self.ensure_threejs_viewer()
        started = time.perf_counter()
        payload_path = path
        try:
            payload_path = self._resolve_simulation_payload_path(path)
            viewer.load_simulation_payload(payload_path, title=path.stem)
        except Exception as exc:
            logger.warning("Failed to load simulation in Three.js viewer: %s", exc)
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("Simulation Viewer"),
                self.window.tr(
                    "Unable to open FlyBody/simulation output in the 3D viewer:\n%1"
                ).replace("%1", str(exc)),
            )
            return False
        logger.info(
            "Prepared Three.js simulation view for %s using %s in %.1fms",
            path,
            payload_path,
            (time.perf_counter() - started) * 1000.0,
        )
        self.window.set_unrelated_docks_visible(False)
        self.window._set_active_view("threejs")
        self.window.statusBar().showMessage(
            self.window.tr("Loaded simulation view %1").replace("%1", path.name), 3000
        )
        return True

    def update_simulation_in_viewer(
        self, simulation_path: str | Path, *, title: str | None = None
    ) -> bool:
        path = Path(simulation_path)
        if not path.exists():
            return False
        viewer = self.ensure_threejs_viewer()
        started = time.perf_counter()
        payload_path = path
        try:
            payload_path = self._resolve_simulation_payload_path(path)
            viewer.update_simulation_payload(payload_path, title=title or path.stem)
        except Exception as exc:
            logger.warning("Failed to update simulation in Three.js viewer: %s", exc)
            return False
        logger.info(
            "Updated Three.js simulation view for %s using %s in %.1fms",
            path,
            payload_path,
            (time.perf_counter() - started) * 1000.0,
        )
        return True

    def _resolve_simulation_payload_path(self, path: Path) -> Path:
        if self._is_prebuilt_simulation_payload(path):
            return path
        return export_simulation_view_payload(path)

    @staticmethod
    def _is_prebuilt_simulation_payload(path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            isinstance(payload, dict) and payload.get("kind") == "annolid-simulation-v1"
        )

    def viewer_widget(self) -> Optional[ThreeJsViewerWidget]:
        return self.threejs_viewer

    def close_threejs(self) -> None:
        """Close Three.js 3D view and return to canvas."""
        # Switch back to canvas.
        try:
            self.window._set_active_view("canvas")
        except Exception:
            pass

    def _handle_flybody_viewer_command(self, action: str, behavior: str) -> None:
        handler = getattr(self.window, "handle_flybody_viewer_command", None)
        if callable(handler):
            handler(action, behavior)
