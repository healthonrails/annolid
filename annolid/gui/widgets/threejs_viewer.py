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


if _WEBENGINE_AVAILABLE:

    class _ThreeJsWebEnginePage(QtWebEngineWidgets.QWebEnginePage):
        def createWindow(
            self, windowType: QtWebEngineWidgets.QWebEnginePage.WebWindowType
        ) -> QtWebEngineWidgets.QWebEnginePage:
            page = _ThreeJsWebEnginePage(self)

            def handle_url_changed(url: QtCore.QUrl) -> None:
                if url.isValid() and not url.isEmpty():
                    QtGui.QDesktopServices.openUrl(url)
                    # We can't easily close/delete the page immediately here
                    # as it's still in the middle of being created.
                    # QTimer.singleShot(0, ...) is a common trick.
                    QtCore.QTimer.singleShot(0, page.deleteLater)

            page.urlChanged.connect(handle_url_changed)
            return page


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
            self._web_view.setPage(_ThreeJsWebEnginePage(self._web_view))
        except Exception:
            pass
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

  <div id="annolidThreeToolbar">
    <div class="tool-group">
      <button class="tool-btn" id="btnHome" title="Reset Camera">
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8h5z"/></svg>
      </button>
      <button class="tool-btn" id="btnViewX" title="Sagittal (X) View">X</button>
      <button class="tool-btn" id="btnViewY" title="Coronal (Y) View">Y</button>
      <button class="tool-btn" id="btnViewZ" title="Axial (Z) View">Z</button>
    </div>
    <div class="tool-sep"></div>
    <div class="tool-group">
      <button class="tool-btn" id="btnFlipX" title="Flip X">fX</button>
      <button class="tool-btn" id="btnFlipY" title="Flip Y">fY</button>
      <button class="tool-btn" id="btnFlipZ" title="Flip Z">fZ</button>
    </div>
    <div class="tool-sep"></div>
    <div class="tool-group">
      <button class="tool-btn" id="btnRotateX" title="Rotate X 90°">rX</button>
      <button class="tool-btn" id="btnRotateY" title="Rotate Y 90°">rY</button>
      <button class="tool-btn" id="btnRotateZ" title="Rotate Z 90°">rZ</button>
    </div>
    <div class="tool-sep"></div>
    <div class="tool-group">
      <button class="tool-btn" id="btnToggleAxes" title="Toggle Axes">
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M13 1.07V9h7c.55 0 1 .45 1 1v2c0 .55-.45 1-1 1h-7v7.93c0 .55-.45 1-1 1h-2c-.55 0-1-.45-1-1V13H3c-.55 0-1-.45-1-1v-2c0-.55.45-1 1-1h7V1.07c0-.55.45-1 1-1h2c.55 0 1 .45 1 1z"/></svg>
      </button>
      <button class="tool-btn" id="btnToggleTheme" title="Toggle Theme">
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M12 22c5.52 0 10-4.48 10-10S17.52 2 12 2 2 6.48 2 12s4.48 10 10 10zm1-17.93c3.94.49 7 3.85 7 7.93s-3.06 7.44-7 7.93V4.07z"/></svg>
      </button>
      <button class="tool-btn" id="btnToggleAutoRotate" title="Auto Rotate">
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46A7.93 7.93 0 0020 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74A7.93 7.93 0 004 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z"/></svg>
      </button>
    </div>
    <div class="tool-sep"></div>
    <button class="tool-btn" id="btnToggleRealtime" title="Toggle Real-time Updates">
      <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/></svg>
    </button>
  </div>

  <div id="annolidThreeIndicators">
    <div class="indicator" id="indRotate">Rotating</div>
    <div class="indicator" id="indPan">Panning</div>
    <div class="indicator" id="indZoom">Zooming</div>
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

  <div id="annolidThreeToolbar">
    <div class="tool-group">
      <button class="tool-btn" id="btnHome" title="Reset Camera">
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8h5z"/></svg>
      </button>
      <button class="tool-btn" id="btnViewX" title="Sagittal (X) View">X</button>
      <button class="tool-btn" id="btnViewY" title="Coronal (Y) View">Y</button>
      <button class="tool-btn" id="btnViewZ" title="Axial (Z) View">Z</button>
    </div>
    <div class="tool-sep"></div>
    <div class="tool-group">
      <button class="tool-btn" id="btnFlipX" title="Flip X">fX</button>
      <button class="tool-btn" id="btnFlipY" title="Flip Y">fY</button>
      <button class="tool-btn" id="btnFlipZ" title="Flip Z">fZ</button>
    </div>
    <div class="tool-sep"></div>
    <div class="tool-group">
      <button class="tool-btn" id="btnRotateX" title="Rotate X 90°">rX</button>
      <button class="tool-btn" id="btnRotateY" title="Rotate Y 90°">rY</button>
      <button class="tool-btn" id="btnRotateZ" title="Rotate Z 90°">rZ</button>
    </div>
    <div class="tool-sep"></div>
    <div class="tool-group">
      <button class="tool-btn" id="btnToggleAxes" title="Toggle Axes">
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M13 1.07V9h7c.55 0 1 .45 1 1v2c0 .55-.45 1-1 1h-7v7.93c0 .55-.45 1-1 1h-2c-.55 0-1-.45-1-1V13H3c-.55 0-1-.45-1-1v-2c0-.55.45-1 1-1h7V1.07c0-.55.45-1 1-1h2c.55 0 1 .45 1 1z"/></svg>
      </button>
      <button class="tool-btn" id="btnToggleTheme" title="Toggle Theme">
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M12 22c5.52 0 10-4.48 10-10S17.52 2 12 2 2 6.48 2 12s4.48 10 10 10zm1-17.93c3.94.49 7 3.85 7 7.93s-3.06 7.44-7 7.93V4.07z"/></svg>
      </button>
      <button class="tool-btn" id="btnToggleAutoRotate" title="Auto Rotate">
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46A7.93 7.93 0 0020 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74A7.93 7.93 0 004 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z"/></svg>
      </button>
    </div>
    <div class="tool-sep"></div>
    <button class="tool-btn" id="btnToggleRealtime" title="Toggle Real-time Updates">
      <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/></svg>
    </button>
  </div>

  <div id="annolidThreeIndicators">
    <div class="indicator" id="indRotate">Rotating</div>
    <div class="indicator" id="indPan">Panning</div>
    <div class="indicator" id="indZoom">Zooming</div>
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

    def load_url(self, url: str) -> None:
        if self._web_view is None:
            raise RuntimeError("Qt WebEngine is unavailable")
        self._web_view.setUrl(QtCore.QUrl(url))
        self.status_changed.emit(f"Loading URL: {url}")

    def clear_model(self) -> None:
        self._current_path = None
        if self._web_view is not None:
            try:
                self._web_view.setHtml("")
            except Exception:
                pass
