from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from qtpy import QtCore, QtGui, QtWidgets

try:
    from qtpy import QtWebEngineWidgets  # type: ignore

    _WEBENGINE_AVAILABLE = True
except Exception:
    QtWebEngineWidgets = None  # type: ignore
    _WEBENGINE_AVAILABLE = False

from annolid.gui.widgets.threejs_viewer_server import (
    _ensure_threejs_http_server,
    _register_threejs_http_model,
)
from annolid.gui.threejs_support import supports_threejs_canvas
from annolid.utils.logger import logger


class ThreeJsViewerWidget(QtWidgets.QWidget):
    """Embedded Three.js viewer for mesh and point-cloud files."""

    status_changed = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._web_view = None
        self._current_path: Optional[Path] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if not _WEBENGINE_AVAILABLE:
            placeholder = QtWidgets.QLabel(
                "Qt WebEngine is unavailable. Three.js canvas is disabled.", self
            )
            placeholder.setWordWrap(True)
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(placeholder, 1)
            return

        self._web_view = QtWebEngineWidgets.QWebEngineView(self)
        try:
            settings = self._web_view.settings()
            local_remote_attr = getattr(
                QtWebEngineWidgets.QWebEngineSettings,
                "LocalContentCanAccessRemoteUrls",
                None,
            )
            local_file_attr = getattr(
                QtWebEngineWidgets.QWebEngineSettings,
                "LocalContentCanAccessFileUrls",
                None,
            )
            if local_remote_attr is not None:
                settings.setAttribute(local_remote_attr, True)
            if local_file_attr is not None:
                settings.setAttribute(local_file_attr, True)
        except Exception:
            pass
        layout.addWidget(self._web_view, 1)

    def init_viewer(
        self, enable_eye_control: bool = False, enable_hand_control: bool = False
    ) -> None:
        """Initialize the Three.js viewer HTML without loading a model."""
        if self._web_view is None:
            return
        if self._current_path is not None:
            # If model is already loaded, just update control states
            self.set_eye_control(enable_eye_control)
            self.set_hand_control(enable_hand_control)
            return

        base = _ensure_threejs_http_server()
        base_url = QtCore.QUrl(base + "/")
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="threejs/annolid_threejs_viewer.css" />
  <script>
    window.__annolidThreeTitle = "Real-time";
    window.__annolidThreeModelUrl = "";
    window.__annolidThreeModelExt = "";
    window.__annolidEnableEyeControl = {str(enable_eye_control).lower()};
    window.__annolidEnableHandControl = {str(enable_hand_control).lower()};
  </script>
</head>
<body>
  <div id="annolidThreeRoot">
    <canvas id="annolidThreeCanvas"></canvas>
  </div>
  <div id="annolidThreeStatus">Starting real-time viewer…</div>
  <div id="annolidThreeHints">Real-time inference mode active.</div>
  <script type="module" src="threejs/annolid_threejs_viewer.js"></script>
</body>
</html>
        """.strip()
        self._web_view.setHtml(html, base_url)
        logger.info("Initialized empty Three.js viewer for real-time mode")

    def update_realtime_data(
        self, qimage: QtGui.QImage, detections: List[dict]
    ) -> None:
        """Stream base64-encoded frame and detections to the Three.js canvas."""
        if self._web_view is None:
            return

        import base64
        from qtpy.QtCore import QBuffer, QIODevice

        base64_data = ""
        if qimage is not None and not qimage.isNull():
            # Convert QImage to base64 JPEG
            buffer = QBuffer()
            buffer.open(QIODevice.WriteOnly)
            qimage.save(buffer, "JPG", 80)
            base64_data = base64.b64encode(buffer.data().data()).decode("utf-8")

        # Prepare detections JSON
        # We only pass keys that Three.js logic expects: behavior, confidence, keypoints, keypoints_pixels
        minimal_detections = []
        for det in detections:
            minimal_detections.append(
                {
                    "behavior": det.get("behavior"),
                    "confidence": det.get("confidence"),
                    "keypoints": det.get("keypoints"),
                    "keypoints_pixels": det.get("keypoints_pixels"),
                    "metadata": det.get("metadata", {}),
                }
            )

        js_code = (
            f"if (window.updateRealtimeData) {{ "
            f"window.updateRealtimeData('{base64_data}', {json.dumps(minimal_detections)});"
            f"}}"
        )
        self._web_view.page().runJavaScript(js_code)

    def set_eye_control(self, enabled: bool) -> None:
        """Dynamically enable/disable eye control in the JS viewer."""
        if self._web_view is None:
            return
        js_code = f"window.__annolidEnableEyeControl = {str(enabled).lower()};"
        self._web_view.page().runJavaScript(js_code)

    def set_hand_control(self, enabled: bool) -> None:
        """Dynamically enable/disable hand control in the JS viewer."""
        if self._web_view is None:
            return
        js_code = f"window.__annolidEnableHandControl = {str(enabled).lower()};"
        self._web_view.page().runJavaScript(js_code)

    def load_model(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"3D model not found: {path}")
        if not supports_threejs_canvas(path):
            raise ValueError(
                f"Unsupported Three.js canvas format: {path.suffix.lower() or '<none>'}"
            )
        if self._web_view is None:
            raise RuntimeError("Qt WebEngine is unavailable")

        self._current_path = path
        base = _ensure_threejs_http_server()
        model_url = _register_threejs_http_model(path)
        base_url = QtCore.QUrl(base + "/")
        title = path.name.replace('"', '\\"')
        model_ext = path.suffix.lower().replace('"', '\\"')
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="threejs/annolid_threejs_viewer.css" />
  <script>
    window.__annolidThreeTitle = "{title}";
    window.__annolidThreeModelUrl = "{model_url}";
    window.__annolidThreeModelExt = "{model_ext}";
    window.__annolidEnableEyeControl = false;
    window.__annolidEnableHandControl = false;
  </script>
</head>
<body>
  <div id="annolidThreeRoot">
    <canvas id="annolidThreeCanvas"></canvas>
  </div>
  <div id="annolidThreeStatus">Loading {title}…</div>
  <div id="annolidThreeHints">Drag: rotate | Wheel: zoom | Right-drag: pan</div>
  <script type="module" src="threejs/annolid_threejs_viewer.js"></script>
</body>
</html>
        """.strip()
        self._web_view.setHtml(html, base_url)
        self.status_changed.emit(f"Loaded 3D model: {path.name}")
        logger.info("Loading Three.js canvas model: %s", path)

    def current_model_path(self) -> Optional[str]:
        if self._current_path is None:
            return None
        return str(self._current_path)

    def clear_model(self) -> None:
        self._current_path = None
        if self._web_view is not None:
            try:
                self._web_view.setHtml("")
            except Exception:
                pass
