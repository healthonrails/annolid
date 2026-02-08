from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy import QtCore, QtWidgets

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
