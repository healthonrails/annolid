from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import unquote, urlparse

from qtpy import QtCore, QtGui, QtWidgets

try:
    from qtpy import QtWebEngineWidgets  # type: ignore

    _WEBENGINE_AVAILABLE = True
except Exception:
    QtWebEngineWidgets = None  # type: ignore
    _WEBENGINE_AVAILABLE = False

try:
    from qtpy import QtWebChannel  # type: ignore

    _WEBCHANNEL_AVAILABLE = True
except Exception:
    QtWebChannel = None  # type: ignore
    _WEBCHANNEL_AVAILABLE = False

from annolid.utils.tts_settings import default_tts_settings, load_tts_settings
from annolid.utils.audio_playback import play_audio_buffer
from annolid.utils.logger import logger


if _WEBENGINE_AVAILABLE:
    # type: ignore[misc]
    class _AnnolidWebEnginePage(QtWebEngineWidgets.QWebEnginePage):
        def javaScriptConsoleMessage(  # noqa: N802 - Qt override
            self,
            # type: ignore[name-defined]
            level: "QtWebEngineWidgets.QWebEnginePage.JavaScriptConsoleMessageLevel",
            message: str,
            lineNumber: int,
            sourceID: str,
        ) -> None:
            try:
                logger.info(
                    f"QtWebEngine js: {message} ({sourceID}:{lineNumber})")
            except Exception:
                pass


_PDFJS_HTTP_SERVER: Optional[ThreadingHTTPServer] = None
_PDFJS_HTTP_PORT: Optional[int] = None
_PDFJS_HTTP_THREAD: Optional[threading.Thread] = None
_PDFJS_HTTP_LOCK = threading.Lock()
_PDFJS_HTTP_TOKENS: dict[str, Path] = {}
_PDFJS_HTTP_ASSET_CACHE: dict[str, bytes] = {}


def _pdfjs_asset_path(filename: str) -> Optional[Path]:
    try:
        root = Path(__file__).resolve().parents[1] / "assets" / "pdfjs"
        candidate = (root / filename).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    except Exception:
        return None
    return None


def _ensure_pdfjs_http_server() -> str:
    global _PDFJS_HTTP_SERVER, _PDFJS_HTTP_PORT, _PDFJS_HTTP_THREAD
    with _PDFJS_HTTP_LOCK:
        if _PDFJS_HTTP_SERVER is not None and _PDFJS_HTTP_PORT is not None:
            return f"http://127.0.0.1:{_PDFJS_HTTP_PORT}"

        class _Handler(BaseHTTPRequestHandler):
            server_version = "AnnolidPdfServer/1.0"

            def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
                # Silence default HTTP server logs.
                return

            def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
                self._serve(send_body=False)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
                self._serve(send_body=True)

            def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods",
                                 "GET, HEAD, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Range, Content-Type")
                self.send_header(
                    "Access-Control-Expose-Headers",
                    "Accept-Ranges, Content-Range, Content-Length",
                )
                self.end_headers()

            def _serve(self, *, send_body: bool) -> None:
                try:
                    parsed = urlparse(self.path)
                    path = parsed.path or ""
                except Exception:
                    self.send_error(400)
                    return

                if path.startswith("/pdfjs/"):
                    name = unquote(path[len("/pdfjs/"):]).strip().lstrip("/")
                    if name not in {
                        "pdf.worker.min.js",
                        "pdf.min.js",
                        "annolid.worker.js",
                    }:
                        self.send_error(404)
                        return
                    asset = _pdfjs_asset_path(name)
                    if asset is None:
                        self.send_error(404)
                        return
                    try:
                        data = _PDFJS_HTTP_ASSET_CACHE.get(name)
                        if data is None:
                            data = asset.read_bytes()
                            _PDFJS_HTTP_ASSET_CACHE[name] = data
                    except Exception:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/javascript")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    if send_body:
                        try:
                            self.wfile.write(data)
                        except Exception:
                            pass
                    return

                if not path.startswith("/pdf/"):
                    self.send_error(404)
                    return
                token = unquote(path[len("/pdf/"):]).strip().split("/", 1)[0]
                file_path = _PDFJS_HTTP_TOKENS.get(token)
                if file_path is None or not file_path.exists():
                    self.send_error(404)
                    return
                try:
                    size = file_path.stat().st_size
                except Exception:
                    self.send_error(404)
                    return

                range_header = self.headers.get("Range", "")
                start = 0
                end = max(0, size - 1)
                status = 200
                if range_header.startswith("bytes="):
                    try:
                        value = range_header[len("bytes="):].strip()
                        start_str, end_str = (value.split("-", 1) + [""])[:2]
                        if start_str == "" and end_str:
                            # Suffix range: last N bytes.
                            length = int(end_str)
                            length = max(0, min(size, length))
                            start = max(0, size - length)
                            end = max(0, size - 1)
                        else:
                            start = int(start_str) if start_str else 0
                            end = int(end_str) if end_str else end
                        start = max(0, min(start, max(0, size - 1)))
                        end = max(start, min(end, max(0, size - 1)))
                        status = 206
                    except Exception:
                        start = 0
                        end = max(0, size - 1)
                        status = 200

                length = max(0, end - start + 1)
                self.send_response(status)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header(
                    "Access-Control-Expose-Headers",
                    "Accept-Ranges, Content-Range, Content-Length",
                )
                self.send_header("Content-Length", str(length))
                if status == 206:
                    self.send_header(
                        "Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if not send_body:
                    return
                try:
                    with open(file_path, "rb") as f:
                        if start:
                            f.seek(start, os.SEEK_SET)
                        remaining = length
                        while remaining > 0:
                            chunk = f.read(min(1024 * 256, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except Exception:
                    return

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        _PDFJS_HTTP_SERVER = httpd
        _PDFJS_HTTP_PORT = int(getattr(httpd, "server_port", 0) or 0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        _PDFJS_HTTP_THREAD = thread
        logger.info(
            f"PDF.js local HTTP server started on 127.0.0.1:{_PDFJS_HTTP_PORT}")
        return f"http://127.0.0.1:{_PDFJS_HTTP_PORT}"


def _register_pdfjs_http_pdf(path: Path) -> str:
    base = _ensure_pdfjs_http_server()
    token = uuid.uuid4().hex
    _PDFJS_HTTP_TOKENS[token] = path
    logger.debug(f"PDF.js serving {path} via token {token}")
    return f"{base}/pdf/{token}"


class _SpeakToken:
    def __init__(self) -> None:
        self.cancelled = False


class _PdfReaderBridge(QtCore.QObject):
    def __init__(self, viewer: "PdfViewerWidget") -> None:
        super().__init__(viewer)
        self._viewer = viewer

    @QtCore.Slot("QVariant")
    def onParagraphClicked(self, payload: object) -> None:
        self._viewer._handle_reader_click(payload)


class PdfViewerWidget(QtWidgets.QWidget):
    """PDF viewer that prefers an embedded browser (if available) with fallback rendering."""

    selection_ready = QtCore.Signal(str)
    page_changed = QtCore.Signal(int, int)
    controls_enabled_changed = QtCore.Signal(bool)
    bookmarks_changed = QtCore.Signal(list)
    reader_state_changed = QtCore.Signal(str, int, int)
    reader_availability_changed = QtCore.Signal(bool, str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._doc = None
        self._pdf_path: Optional[Path] = None
        self._current_page = 0
        self._zoom = 1.5
        self._rotation = 0
        self._thread_pool = QtCore.QThreadPool(self)
        self._web_view = None
        self._web_container = None
        self._web_loading_path: Optional[Path] = None
        self._web_pdf_capable = False
        self._use_web_engine = False
        self._pdfjs_active = False
        self._web_mode_active = False
        self._speaking = False
        self._selection_cache = ""
        self._selection_cache_time = 0.0
        self._highlight_mode: Optional[str] = None
        self._selection_anchor_start: Optional[int] = None
        self._selection_anchor_end: Optional[int] = None
        self._text_sentence_spans: list[tuple[int, int]] = []
        self._web_sentence_span_groups: list[list[int]] = []
        self._web_selected_span_text: Dict[int, str] = {}
        self._word_highlight_timer: Optional[QtCore.QTimer] = None
        self._word_highlight_units: list[object] = []
        self._word_highlight_durations_ms: list[int] = []
        self._word_highlight_index = 0
        self._active_text_sentence_span: Optional[tuple[int, int]] = None
        # Prefer PDF.js by default (Chromium <embed> is not scriptable).
        self._force_pdfjs = True
        self._bookmarks: list[dict[str, object]] = []
        self._web_channel = None
        self._reader_bridge = None
        self._reader_enabled = True
        self._reader_state = "idle"
        self._reader_queue: list[str] = []
        self._reader_spans: list[list[int]] = []
        self._reader_pages: list[int] = []
        self._reader_queue_offset = 0
        self._reader_current_index = 0
        self._reader_total = 0
        self._reader_chunk_base = 0
        self._reader_pause_requested = False
        self._reader_stop_requested = False
        self._reader_pending_restart: Optional[int] = None
        self._active_speak_token: Optional[_SpeakToken] = None
        self._build_ui()
        logger.info(
            f"QtWebEngine available={bool(_WEBENGINE_AVAILABLE)}, "
            f"pdf_capable={bool(self._web_pdf_capable)}"
        )

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._stack = QtWidgets.QStackedWidget(self)

        # Fallback: PyMuPDF-rendered pages + selectable text.
        fallback_container = QtWidgets.QWidget(self)
        fallback_layout = QtWidgets.QVBoxLayout(fallback_container)
        fallback_layout.setContentsMargins(0, 0, 0, 0)
        fallback_layout.setSpacing(6)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setBackgroundRole(QtGui.QPalette.Base)
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.image_label.setMinimumSize(200, 200)

        image_scroll = QtWidgets.QScrollArea(self)
        image_scroll.setWidgetResizable(True)
        image_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        image_scroll.setWidget(self.image_label)
        self._image_scroll = image_scroll
        fallback_layout.addWidget(image_scroll, 3)

        self.text_view = QtWidgets.QTextEdit(self)
        self.text_view.setReadOnly(True)
        self.text_view.setPlaceholderText(
            "Select text on this page, then right-click to speak it.")
        self.text_view.selectionChanged.connect(
            self._on_text_selection_changed)
        self.text_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.text_view.customContextMenuRequested.connect(
            self._show_context_menu)
        fallback_layout.addWidget(self.text_view, 1)

        self._stack.addWidget(fallback_container)

        # Preferred: embedded web-based PDF viewer (Chromium).
        if _WEBENGINE_AVAILABLE:
            self._web_view = QtWebEngineWidgets.QWebEngineView(self)
            try:
                self._web_view.setPage(_AnnolidWebEnginePage(self._web_view))
            except Exception:
                pass
            if _WEBCHANNEL_AVAILABLE:
                try:
                    self._web_channel = QtWebChannel.QWebChannel(
                        self._web_view.page()
                    )
                    self._reader_bridge = _PdfReaderBridge(self)
                    self._web_channel.registerObject(
                        "annolidBridge", self._reader_bridge
                    )
                    self._web_view.page().setWebChannel(self._web_channel)
                except Exception as exc:
                    logger.info("QtWebChannel unavailable: %s", exc)
                    self._web_channel = None
                    self._reader_bridge = None
            self._web_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self._web_view.customContextMenuRequested.connect(
                self._show_web_context_menu
            )
            try:
                settings = self._web_view.settings()
                pdf_attr = getattr(
                    QtWebEngineWidgets.QWebEngineSettings, "PdfViewerEnabled", None)
                plugins_attr = getattr(
                    QtWebEngineWidgets.QWebEngineSettings, "PluginsEnabled", None)
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
                if plugins_attr is not None:
                    settings.setAttribute(plugins_attr, True)
                if pdf_attr is not None:
                    settings.setAttribute(pdf_attr, True)
                    self._web_pdf_capable = settings.testAttribute(pdf_attr)
                if local_remote_attr is not None:
                    settings.setAttribute(local_remote_attr, True)
                if local_file_attr is not None:
                    settings.setAttribute(local_file_attr, True)
                logger.info(
                    "QtWebEngine settings: "
                    f"PluginsEnabled={settings.testAttribute(plugins_attr) if plugins_attr else 'n/a'} "
                    f"PdfViewerEnabled={settings.testAttribute(pdf_attr) if pdf_attr else 'n/a'} "
                    f"LocalContentCanAccessRemoteUrls={settings.testAttribute(local_remote_attr) if local_remote_attr else 'n/a'} "
                    f"LocalContentCanAccessFileUrls={settings.testAttribute(local_file_attr) if local_file_attr else 'n/a'}"
                )
            except Exception:
                self._web_pdf_capable = False
            self._web_view.loadFinished.connect(self._on_web_load_finished)
            self._web_view.loadStarted.connect(
                lambda: logger.info("QtWebEngine load started")
            )
            try:
                self._web_view.renderProcessTerminated.connect(
                    lambda *_: logger.warning(
                        "QtWebEngine render process terminated")
                )
            except Exception:
                pass
            self._web_container = QtWidgets.QWidget(self)
            web_layout = QtWidgets.QVBoxLayout(self._web_container)
            web_layout.setContentsMargins(0, 0, 0, 0)
            web_layout.setSpacing(0)
            web_layout.addWidget(self._web_view)
            self._stack.addWidget(self._web_container)

        layout.addWidget(self._stack, 1)
        # Even when Chromium's built-in PDF plugin is unavailable, we can still
        # render PDFs via the bundled PDF.js viewer.
        self._use_web_engine = bool(
            _WEBENGINE_AVAILABLE and self._web_view is not None)
        if _WEBENGINE_AVAILABLE and not self._web_pdf_capable:
            logger.info(
                "QtWebEngine PDF plugin support appears disabled; PDF.js will be used instead."
            )
        self._emit_reader_availability()

    def load_pdf(self, pdf_path: str) -> None:
        """Load a PDF file and render the first page."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._rotation = 0

        # Preferred: QtWebEngine <embed> plugin when available; fallback to PDF.js.
        if self._use_web_engine and self._web_view is not None:
            self._stack.setCurrentWidget(self._web_container)
            self._set_controls_for_web(True)
            self._web_loading_path = path
            if self._web_pdf_capable and not self._force_pdfjs:
                logger.info(f"Loading PDF with QtWebEngine plugin: {path}")
                self._pdfjs_active = False
                self._load_web_embed_pdf(path)
                self._load_bookmarks_from_path(path)
                self._emit_reader_availability()
            else:
                logger.info(f"Loading PDF with PDF.js viewer: {path}")
                self._pdfjs_active = True
                loaded = self._load_pdfjs_viewer(path)
                if not loaded:
                    logger.warning(
                        "Failed to load PDF.js viewer; falling back to PyMuPDF."
                    )
                    self._use_web_engine = False
                    self._pdfjs_active = False
                    self._clear_bookmarks()
                    self._open_with_pymupdf(path)
                    self._web_loading_path = None
                    return
                self._clear_bookmarks()
                self._emit_reader_availability()
            if self._doc is not None:
                self._doc.close()
                self._doc = None
            return

        # Fallback: PyMuPDF rendering.
        logger.info(f"Loading PDF with PyMuPDF fallback: {path}")
        self._pdfjs_active = False
        self._clear_bookmarks()
        self._open_with_pymupdf(path)

    def _load_web_embed_pdf(self, path: Path) -> None:
        """Load a PDF in WebEngine via an HTML wrapper <embed>."""
        if self._web_view is None:
            return
        base_url = QtCore.QUrl.fromLocalFile(str(path.parent) + "/")
        pdf_url = QtCore.QUrl.fromLocalFile(str(path)).toString()
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:#202124; }}
    #pdf {{ width:100%; height:100%; border:0; }}
  </style>
</head>
<body>
  <embed id="pdf" src="{pdf_url}" type="application/pdf" />
</body>
</html>
        """.strip()
        self._web_view.setHtml(html, base_url)

    def _open_with_pymupdf(self, path: Path) -> None:
        """Render PDF pages using PyMuPDF fallback."""
        try:
            import fitz  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - user-facing dialog
            raise RuntimeError(
                "PyMuPDF (pymupdf) is required to view PDF files."
            ) from exc

        if self._doc is not None:
            self._doc.close()
            self._doc = None

        self._doc = fitz.open(str(path))
        if self._doc.page_count == 0:
            self._doc.close()
            self._doc = None
            raise ValueError("The selected PDF does not contain any pages.")

        self._pdf_path = path
        self._current_page = 0
        self._stack.setCurrentIndex(0)
        self._set_controls_for_web(False)
        self._emit_reader_availability()
        QtCore.QTimer.singleShot(0, self.fit_to_width)
        self._render_current_page()

    def fit_to_width(self) -> None:
        """Fit the current PDF view to the available width."""
        if self._web_view is not None and self._pdfjs_active:
            try:
                self._web_view.page().runJavaScript(
                    "window.__annolidZoomFitWidth && window.__annolidZoomFitWidth();"
                )
            except Exception:
                pass
            return
        if self._doc is None:
            return
        scroll = getattr(self, "_image_scroll", None)
        if scroll is None:
            return
        viewport = scroll.viewport()
        if viewport is None:
            return
        view_w = max(0, int(viewport.width()))
        if view_w <= 0:
            QtCore.QTimer.singleShot(0, self.fit_to_width)
            return
        try:
            page = self._doc.load_page(self._current_page)
            rect = page.rect
            page_w = float(rect.width)
            page_h = float(rect.height)
            if self._rotation % 180:
                page_w = page_h
            gutter = 24.0
            target = max(1.0, float(view_w) - gutter)
            if page_w <= 0:
                return
            scale = target / page_w
            self._zoom = max(0.5, min(3.0, float(scale)))
            self._render_current_page()
        except Exception:
            return

    def _on_web_load_finished(self, ok: bool) -> None:
        """Check web load result; fallback to PyMuPDF on failure/blank page."""
        path = self._web_loading_path
        if path is None:
            return
        if not ok and not self._pdfjs_active:
            self._fallback_from_web(path, "loadFinished returned False")
            self._web_loading_path = None
            return

        if self._pdfjs_active:
            # Verify PDF.js actually rendered; otherwise fall back.
            # Allow extra time for larger PDFs to parse.
            def probe_pdfjs(attempts_left: int = 120) -> None:
                def _after_pdfjs_probe(result: object) -> None:
                    # Abort if another PDF load started meanwhile.
                    if (
                        self._web_loading_path is not None
                        and self._web_loading_path != path
                    ):
                        return

                    err = ""
                    spans = 0
                    rendered_pages = 0
                    pdf_loaded = False
                    pdfjs_ready = False
                    has_pdfjs = False
                    ready_state = ""
                    try:
                        if isinstance(result, dict):
                            err = str(result.get("err", "") or "")
                            spans = int(result.get("spans", 0))
                            rendered_pages = int(
                                result.get("renderedPages", 0) or 0
                            )
                            pdf_loaded = bool(result.get("pdfLoaded", False))
                            pdfjs_ready = bool(result.get("ready", False))
                            has_pdfjs = bool(result.get("hasPdfjs", False))
                            ready_state = str(result.get("state", "") or "")
                    except Exception:
                        err = ""
                        spans = 0
                        rendered_pages = 0
                        pdf_loaded = False
                        pdfjs_ready = False
                        has_pdfjs = False
                        ready_state = ""
                    if err:
                        self._fallback_from_web(path, f"PDF.js error: {err}")
                        return
                    if not pdfjs_ready and attempts_left > 0:
                        QtCore.QTimer.singleShot(
                            250, lambda: probe_pdfjs(attempts_left - 1)
                        )
                        return
                    if not pdfjs_ready:
                        self._fallback_from_web(
                            path,
                            "PDF.js bootstrap not running "
                            f"(readyState={ready_state!r} hasPdfjs={has_pdfjs})",
                        )
                        return
                    if not pdf_loaded and attempts_left > 0:
                        QtCore.QTimer.singleShot(
                            250, lambda: probe_pdfjs(attempts_left - 1)
                        )
                        return
                    # If the document loaded successfully, consider PDF.js
                    # active even if rendering is still in progress.
                    if not pdf_loaded:
                        self._fallback_from_web(
                            path,
                            "PDF.js did not load "
                            f"(readyState={ready_state!r} spans={spans} pages={rendered_pages})",
                        )
                        return
                    self._pdf_path = path
                    self._web_loading_path = None
                    logger.info(f"QtWebEngine PDF.js viewer active for {path}")
                    self._apply_reader_enabled_to_web()
                    self.fit_to_width()
                    self._emit_reader_availability()

                try:
                    self._web_view.page().runJavaScript(
                        """(() => {
  const err = document.body ? (document.body.getAttribute("data-pdfjs-error") || "") : "";
  const spans = (window.__annolidSpans || []).length;
  const renderedPages = (window.__annolidRenderedPages || 0);
  const pdfLoaded = !!window.__annolidPdfLoaded;
  const ready = !!window.__annolidPdfjsReady;
  const hasPdfjs = (typeof pdfjsLib !== 'undefined') && !!pdfjsLib;
  const state = document && document.readyState ? document.readyState : '';
  return {err, spans, renderedPages, pdfLoaded, ready, hasPdfjs, state};
})()""",
                        _after_pdfjs_probe,
                    )
                except Exception:
                    self._fallback_from_web(path, "PDF.js probe failed")

            probe_pdfjs()
            return

        def _after_probe(result: object) -> None:
            ok_result = False
            try:
                if isinstance(result, dict):
                    ok_result = bool(result.get("hasPdf", False))
                    plugins = str(result.get("plugins", ""))
                    ua = str(result.get("ua", ""))
                    logger.info(
                        "QtWebEngine PDF probe: "
                        f"hasPdf={ok_result} plugins={plugins!r} ua={ua!r}"
                    )
                else:
                    ok_result = bool(result)
            except Exception:
                ok_result = False
            if not ok_result:
                self._fallback_from_web(
                    path, "PDF mimeType not available in WebEngine")
                return
            self._pdf_path = path
            self._web_loading_path = None
            logger.info(f"QtWebEngine PDF plugin detected for {path}")
            self._emit_reader_availability()

        try:
            self._web_view.page().runJavaScript(
                """(() => {
  const mt = (navigator.mimeTypes && navigator.mimeTypes['application/pdf']) ? true : false;
  const plugins = navigator.plugins ? Array.from(navigator.plugins).map(p => p.name).join('|') : '';
  const hasPdf = mt || /pdf/i.test(plugins);
  return {hasPdf, plugins, ua: navigator.userAgent};
})()""",
                _after_probe,
            )
        except Exception:
            self._fallback_from_web(path, "JS mimeType probe threw")

    def _fallback_from_web(self, path: Path, reason: str) -> None:
        """Switch from web view to PyMuPDF and log why."""
        # Try an in-place PDF.js viewer before giving up on the web engine.
        pdfjs_loaded = False
        if self._web_view is not None and not self._pdfjs_active:
            pdfjs_loaded = self._load_pdfjs_viewer(path)
            if pdfjs_loaded:
                logger.warning(
                    f"QtWebEngine PDF plugin missing; using PDF.js viewer for {path} ({reason})"
                )
                self._pdf_path = path
                self._web_loading_path = None
                self._use_web_engine = True
                self._pdfjs_active = True
                self._set_controls_for_web(True)
                self._stack.setCurrentWidget(self._web_container)
                self._apply_reader_enabled_to_web()
                self._emit_reader_availability()
                return
        self._use_web_engine = False
        self._pdfjs_active = False
        self._open_with_pymupdf(path)
        self._web_loading_path = None
        logger.warning(f"Falling back to PyMuPDF for {path} ({reason})")

    def _load_pdfjs_viewer(self, path: Path) -> bool:
        """Load a lightweight PDF.js viewer into the web view."""
        if self._web_view is None:
            return False
        # Prefer a bundled PDF.js to avoid network/CSP issues in QtWebEngine.
        # Fall back to CDN if the local asset is missing.
        pdfjs_version = "2.16.105"  # Compatible with older Chromium in Qt 5.15
        pdfjs_src = f"https://cdnjs.cloudflare.com/ajax/libs/pdf.js/{pdfjs_version}/pdf.min.js"
        pdfjs_inline = ""
        pdfjs_tag = f'<script src="{pdfjs_src}"></script>'
        try:
            local_pdfjs = (
                Path(__file__).resolve().parents[1]
                / "assets"
                / "pdfjs"
                / "pdf.min.js"
            )
            if local_pdfjs.exists():
                # Inline script avoids file:// cross-origin restrictions in some
                # QtWebEngine configurations.
                pdfjs_inline = local_pdfjs.read_text(encoding="utf-8")
        except Exception:
            pass
        if pdfjs_inline:
            pdfjs_tag = "<script>\n" + pdfjs_inline + "\n</script>"
        # Serve the PDF over a local HTTP endpoint so PDF.js can fetch it
        # reliably (fetch/XHR against file:// can hang in QtWebEngine).
        base = _ensure_pdfjs_http_server()
        base_url = QtCore.QUrl(base + "/")
        pdf_url = _register_pdfjs_http_pdf(path)
        pdf_b64 = ""
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #1e1e1e;
      color: #eee;
      overflow: hidden;
      font: 13px "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    }}
    #viewerContainer {{
      position: absolute;
      top: 56px; left: 0; right: 0; bottom: 0;
      overflow: auto;
      background: #1e1e1e;
    }}
	    #annolidToolbar {{
	      position: fixed;
	      top: 0;
	      left: 0;
	      right: 0;
	      z-index: 9999;
	      display: flex;
	      align-items: center;
	      gap: 10px;
	      padding: 8px 12px;
	      background: #3a3a3a;
	      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
	      color: #f5f5f5;
	      box-sizing: border-box;
	      overflow: visible;
	      user-select: none;
	      -webkit-user-select: none;
	    }}
	    #annolidToolbar button {{
	      background: #2f2f2f;
	      color: #f5f5f5;
	      border: 1px solid rgba(255, 255, 255, 0.12);
	      border-radius: 6px;
	      padding: 5px 8px;
	      cursor: pointer;
	      min-width: 32px;
	    }}
	    #annolidToolbar button:focus {{
	      outline: none;
	      box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.45);
	    }}
    #annolidToolbar button.annolid-active {{
      background: #1976d2;
      border-color: #1976d2;
      color: white;
    }}
    #annolidToolbar .annolid-sep {{
      width: 1px;
      height: 24px;
      background: rgba(255, 255, 255, 0.14);
      margin: 0 6px;
    }}
    #annolidToolbar label {{
      opacity: 0.9;
      font-size: 12px;
    }}
    #annolidToolbar input[type="color"] {{
      width: 28px;
      height: 28px;
      padding: 0;
      border: 0;
      background: transparent;
      cursor: pointer;
    }}
	    #annolidToolbar input[type="range"] {{
	      width: 32px;
	    }}
	    @media (max-width: 980px) {{
	      .annolid-title {{ max-width: 150px; }}
	      #annolidZoomReset {{ display: none; }}
	      #annolidZoomFit {{ display: none; }}
	    }}
	    @media (max-width: 780px) {{
	      .annolid-title {{ display: none; }}
	      #annolidZoomLabel {{ min-width: 54px; }}
	    }}
	    .annolid-toolbar-left {{
	      display: flex;
	      align-items: center;
	      gap: 10px;
	      min-width: 0;
	      flex: 0 1 160px;
	    }}
	    .annolid-title {{
	      font-weight: 600;
	      font-size: 13px;
	      color: #f5f5f5;
	      white-space: nowrap;
	      overflow: hidden;
	      text-overflow: ellipsis;
	      max-width: 140px;
	    }}
	    .annolid-nav {{
	      display: flex;
	      align-items: center;
	      gap: 6px;
	      justify-content: center;
	      flex-wrap: nowrap;
	      min-width: 0;
	      flex: 0 0 auto;
	    }}
	    .annolid-actions {{
	      display: flex;
	      align-items: center;
	      gap: 8px;
	      justify-content: flex-end;
	      flex-wrap: nowrap;
	      overflow: hidden;
	      min-width: 0;
	      flex: 1 1 0;
	    }}
    .annolid-group {{
      display: inline-flex;
      align-items: center;
      gap: 0;
      border: 1px solid rgba(255, 255, 255, 0.12);
	      border-radius: 8px;
	      overflow: hidden;
	      background: rgba(0, 0, 0, 0.12);
	    }}
	    .annolid-group button {{
	      border: 0;
	      border-right: 1px solid rgba(255, 255, 255, 0.10);
	      border-radius: 0;
	      background: transparent;
	      padding: 6px 10px;
	      min-width: 28px;
	    }}
	    .annolid-group button:last-child {{
      border-right: 0;
    }}
    .annolid-mark-options {{
      gap: 8px;
      padding: 6px 10px;
    }}
    .annolid-mark-options input[type="color"] {{
      width: 30px;
      height: 28px;
      padding: 0;
      border: 0;
      background: transparent;
      cursor: pointer;
    }}
    .annolid-mark-options input[type="range"] {{
      width: 80px;
    }}
    .annolid-option-label {{
      font-size: 12px;
      opacity: 0.85;
      padding: 0 2px;
      white-space: nowrap;
    }}
    .annolid-overflow {{
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
	    }}
	    #annolidOverflowMenu {{
	      position: absolute;
	      top: calc(100% + 6px);
	      right: 0;
	      background: #2b2b2b;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 6px;
      box-shadow: 0 4px 18px rgba(0,0,0,0.4);
	      display: none;
      flex-direction: column;
      gap: 6px;
      z-index: 10000;
      min-width: 120px;
    }}
	    #annolidOverflowMenu button {{
	      width: 100%;
	      text-align: left;
	    }}
	    #annolidOverflowMenu .annolid-sep {{
	      display: none;
	    }}
	    .annolid-menu-row {{
	      display: flex;
	      align-items: center;
	      gap: 8px;
	      padding: 6px;
	      border-radius: 6px;
	      background: rgba(255, 255, 255, 0.04);
	    }}
	    .annolid-menu-label {{
	      min-width: 44px;
	      font-size: 12px;
	      opacity: 0.9;
	    }}
	    #annolidOverflowMenu input[type="color"] {{
	      width: 32px;
	      height: 28px;
	      padding: 0;
	      border: 0;
	      background: transparent;
	    }}
	    #annolidOverflowMenu input[type="range"] {{
	      width: 140px;
	    }}
    #annolidOverflowMenu.annolid-open {{
      display: flex;
    }}
    #annolidMenuPanel {{
      position: absolute;
      top: calc(100% + 6px);
      left: 0;
      background: #2b2b2b;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 6px;
      box-shadow: 0 4px 18px rgba(0,0,0,0.4);
      display: none;
      flex-direction: column;
      gap: 6px;
      z-index: 10000;
      min-width: 140px;
    }}
    #annolidMenuPanel.annolid-open {{
      display: flex;
    }}
    #annolidMenuPanel button {{
      width: 100%;
      text-align: left;
    }}
    #annolidPageInput {{
      width: 48px;
      background: #1e1e1e;
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 6px;
      padding: 5px 7px;
      text-align: center;
    }}
    #annolidZoomLabel {{
      padding: 5px 8px;
      background: #1e1e1e;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      min-width: 48px;
      text-align: center;
    }}
    .annolid-icon-btn {{
      font-size: 14px;
      line-height: 1;
    }}
    .annolid-disabled {{
      opacity: 0.45;
      pointer-events: none;
    }}
    .page {{
      position: relative;
      margin: 14px auto;
      background: #2b2b2b;
      box-shadow: 0 2px 18px rgba(0, 0, 0, 0.45);
    }}
    canvas {{
      display: block;
    }}
    .textLayer {{
      position: absolute;
      inset: 0;
      overflow: hidden;
      opacity: 1;
      user-select: text;
      -webkit-user-select: text;
      cursor: text;
      z-index: 10;
    }}
    .textLayer span {{
      position: absolute;
      white-space: pre;
      transform-origin: 0% 0%;
      color: transparent;
      -webkit-text-fill-color: transparent;
    }}
    .textLayer ::selection {{
      background: rgba(0, 120, 215, 0.35);
    }}
    .annolid-tts-layer {{
      position: absolute;
      inset: 0;
      z-index: 20;
      pointer-events: none;
    }}
    .annolid-mark-layer {{
      position: absolute;
      inset: 0;
      z-index: 30;
      pointer-events: none;
      touch-action: none;
    }}
    .annotationLayer {{
      position: absolute;
      inset: 0;
      z-index: 15;
      pointer-events: none;
    }}
    .annotationLayer a {{
      position: absolute;
      display: block;
      pointer-events: auto;
      background: rgba(0, 0, 0, 0);
      text-decoration: none;
    }}
    .annotationLayer a:hover {{
      outline: 2px solid rgba(25, 118, 210, 0.55);
      outline-offset: -2px;
      background: rgba(25, 118, 210, 0.12);
    }}
    body.annolid-drawing .textLayer {{
      user-select: none;
      -webkit-user-select: none;
    }}
    body.annolid-reader-enabled .textLayer {{
      cursor: pointer;
    }}
    body.annolid-reader-enabled .textLayer span:hover {{
      background: rgba(255, 255, 255, 0.12);
    }}
    .annolid-modal {{
      position: fixed;
      inset: 0;
      z-index: 20000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(0, 0, 0, 0.55);
      box-sizing: border-box;
    }}
    .annolid-modal.annolid-open {{
      display: flex;
    }}
    .annolid-modal-content {{
      width: min(1100px, 96vw);
      height: min(880px, 92vh);
      display: flex;
      flex-direction: column;
      background: #242424;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
      overflow: hidden;
    }}
    .annolid-modal-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      background: #2f2f2f;
      border-bottom: 1px solid rgba(255, 255, 255, 0.10);
    }}
    .annolid-modal-title {{
      font-size: 13px;
      font-weight: 600;
      color: #f5f5f5;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }}
    .annolid-modal-actions {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .annolid-modal-actions button {{
      background: #1e1e1e;
      color: #f5f5f5;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      min-width: 36px;
    }}
    .annolid-modal-actions button:hover {{
      background: rgba(255, 255, 255, 0.06);
    }}
    .annolid-modal-body {{
      position: relative;
      flex: 1 1 auto;
      overflow: auto;
      background: #1a1a1a;
    }}
    .annolid-preview-grid {{
      display: grid;
      grid-template-columns: minmax(260px, 380px) minmax(320px, 1fr);
      gap: 12px;
      padding: 12px;
      box-sizing: border-box;
      align-items: start;
    }}
    .annolid-preview-grid.annolid-preview-citation {{
      grid-template-columns: 1fr;
    }}
    @media (max-width: 980px) {{
      .annolid-preview-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    .annolid-preview-text {{
      background: #202020;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 10px;
      padding: 10px 12px;
      color: #f0f0f0;
      font-size: 13px;
      line-height: 1.35;
      white-space: pre-wrap;
      overflow: auto;
      max-height: 100%;
      min-height: 120px;
    }}
    .annolid-muted {{
      opacity: 0.72;
    }}
    .annolid-modal-canvas-wrap {{
      position: relative;
      margin: 12px auto;
      width: max-content;
      background: #2b2b2b;
      box-shadow: 0 2px 18px rgba(0, 0, 0, 0.45);
    }}
    .annolid-cite-popover {{
      position: fixed;
      z-index: 25000;
      display: none;
      max-width: 420px;
      min-width: 240px;
      pointer-events: auto;
    }}
    .annolid-cite-popover.annolid-open {{
      display: block;
    }}
    .annolid-cite-card {{
      background: #242424;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 10px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      padding: 10px 12px;
      color: #f5f5f5;
      font-size: 12px;
      line-height: 1.35;
    }}
    .annolid-cite-title {{
      font-weight: 600;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .annolid-cite-body {{
      max-height: 240px;
      overflow: auto;
      white-space: pre-wrap;
    }}
    #annolidPreviewHighlight {{
      position: absolute;
      border: 2px solid rgba(255, 193, 7, 0.95);
      background: rgba(255, 193, 7, 0.20);
      border-radius: 4px;
      pointer-events: none;
      display: none;
    }}
  </style>
  <script>
	    // Polyfill `.at()` for older Chromium (QtWebEngine 5.15).
	    function _defineNonEnumerableAt(proto, fn) {{
	      if (!proto || proto.at) return;
	      try {{
	        Object.defineProperty(proto, "at", {{
	          value: fn,
	          writable: true,
	          configurable: true,
	          enumerable: false,
	        }});
	      }} catch (e) {{
	        try {{ proto.at = fn; }} catch (e2) {{}}
	      }}
	    }}
	    function _atPolyfill(n) {{
	      n = Math.trunc(n) || 0;
	      if (n < 0) n += this.length;
	      if (n < 0 || n >= this.length) return undefined;
	      return this[n];
	    }}
	    _defineNonEnumerableAt(Array.prototype, _atPolyfill);
	    if (!String.prototype.at) {{
	      _defineNonEnumerableAt(String.prototype, function(n) {{
	        n = Math.trunc(n) || 0;
	        if (n < 0) n += this.length;
	        if (n < 0 || n >= this.length) return undefined;
	        return this.charAt(n);
	      }});
	    }}
	    const _typed = [
	      typeof Int8Array !== "undefined" ? Int8Array : null,
	      typeof Uint8Array !== "undefined" ? Uint8Array : null,
      typeof Uint8ClampedArray !== "undefined" ? Uint8ClampedArray : null,
      typeof Int16Array !== "undefined" ? Int16Array : null,
      typeof Uint16Array !== "undefined" ? Uint16Array : null,
      typeof Int32Array !== "undefined" ? Int32Array : null,
      typeof Uint32Array !== "undefined" ? Uint32Array : null,
      typeof Float32Array !== "undefined" ? Float32Array : null,
      typeof Float64Array !== "undefined" ? Float64Array : null,
      typeof BigInt64Array !== "undefined" ? BigInt64Array : null,
      typeof BigUint64Array !== "undefined" ? BigUint64Array : null,
	    ];
	    for (const T of _typed) {{
	      if (T && T.prototype) {{
	        _defineNonEnumerableAt(T.prototype, _atPolyfill);
	      }}
	    }}
    // Basic structuredClone polyfill (PDF.js may reference it in older Chromium).
    if (typeof structuredClone === "undefined") {{
      window.structuredClone = function(obj) {{
        try {{
          return JSON.parse(JSON.stringify(obj));
        }} catch (e) {{
          return obj;
        }}
      }};
    }}
  </script>
  <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
  {pdfjs_tag}
  <script>
    const pdfUrl = "{pdf_url}";
    const pdfBase64 = "{pdf_b64}";
    const pdfTitle = "{path.name}";
    document.addEventListener("DOMContentLoaded", async () => {{
      try {{
        if (typeof pdfjsLib === "undefined" || !pdfjsLib) {{
          document.body.setAttribute("data-pdfjs-error", "pdfjsLib not loaded");
          return;
        }}
        try {{
          const workerUrl = (window.location && window.location.origin)
            ? (window.location.origin + "/pdfjs/annolid.worker.js")
            : "/pdfjs/annolid.worker.js";
          if (pdfjsLib.GlobalWorkerOptions) {{
            pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;
          }}
        }} catch (e) {{}}
        window.__annolidPdfjsReady = true;

        window.__annolidSpans = [];
        window.__annolidSpanCounter = 0;
        window.__annolidSelectionSpans = [];
        window.__annolidSpanMeta = {{}};
        window.__annolidPages = {{}};
        window.__annolidTts = {{ sentenceIndices: [], wordIndex: null, lastPages: [] }};
        window.__annolidMarks = {{ tool: "select", color: "#ffb300", size: 10, undo: [], drawing: null }};
        window.__annolidReaderEnabled = window.__annolidReaderEnabled || false;
	        window.__annolidParagraphsByPage = {{}};
	        window.__annolidParagraphs = [];
	        window.__annolidParagraphOffsets = {{}};
	        window.__annolidParagraphTotal = 0;
	        window.__annolidSplitTextIntoSentenceRanges = function(text) {{
	          const s = String(text || "");
	          if (!s) return [];
	          const END = new Set([".", "!", "?", "。", "！", "？"]);
	          const isCjkLead = (ch) => /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/.test(ch || "");
	          const isNarrationLead = (ch) => {{
	            const c = String(ch || "");
	            return isCjkLead(c) || c === "（" || c === "【" || c === "“" || c === "‘";
	          }};
	          const QUOTE_CLOSERS = new Set(["”", "’", "\\\"", "'"]);
	          const CLOSERS = new Set([
	            "”", "’", "\\\"", "'", ")", "]", "）", "】", "》", "」", "』", "〉",
	          ]);
	          const ranges = [];
	          let start = 0;
	          for (let i = 0; i < s.length; i++) {{
	            const ch = s[i];
	            if (!END.has(ch)) continue;
	            let end = i + 1;
	            while (end < s.length && END.has(s[end])) end++;
	            const punctEnd = end;
	            // Attach trailing quotes/brackets, even if PDF extraction inserted whitespace
	            // between end punctuation and the closer (e.g. "。”" or "。 ”").
	            let probe = end;
	            for (let guard = 0; guard < 8; guard++) {{
	              let ws = probe;
	              while (ws < s.length && /\\s/.test(s[ws])) ws++;
	              if (ws < s.length && CLOSERS.has(s[ws])) {{
	                probe = ws + 1;
	                while (probe < s.length && CLOSERS.has(s[probe])) probe++;
	                end = probe;
	                continue;
	              }}
	              break;
	            }}
	            // Fix skipped patterns around Chinese quote endings like:
	            //   "。”说完..." / "。”小鹊..."
	            // When a quoted sentence ends and narration continues immediately, some PDFs place the
	            // continuation spans slightly off-baseline; splitting here makes the next part prone
	            // to being reordered and "skipped". Treat these as a single sentence chunk.
	            try {{
	              let hasQuoteCloser = false;
	              for (let j = punctEnd; j < end; j++) {{
	                const c = s[j];
	                if (/\\s/.test(c)) continue;
	                if (QUOTE_CLOSERS.has(c)) {{
	                  hasQuoteCloser = true;
	                  break;
	                }}
	              }}
	              if (hasQuoteCloser) {{
	                let k = end;
	                // PDFs may insert multiple spaces/newlines between the quote closer and narration.
	                // Skip a reasonable amount of whitespace and merge when narration continues.
	                while (k < s.length && /\\s/.test(s[k]) && (k - end) < 48) k++;
	                if (k < s.length && isNarrationLead(s[k])) {{
	                  continue;
	                }}
	              }}
	            }} catch (e) {{}}
	            ranges.push([start, end]);
	            start = end;
	          }}
	          if (start < s.length) ranges.push([start, s.length]);
	          // Trim whitespace around ranges.
	          const out = [];
	          for (const pair of ranges) {{
	            let a = pair[0];
	            let b = pair[1];
	            while (a < b && /\\s/.test(s[a])) a++;
	            while (b > a && /\\s/.test(s[b - 1])) b--;
	            if (b > a) out.push([a, b]);
	          }}
	          return out;
	        }};

	        window.__annolidSplitParagraphIntoSentences = function(para) {{
	          try {{
	            const spansRaw = (para && Array.isArray(para.spans)) ? para.spans.filter((n) => Number.isInteger(n)) : [];
	            if (!spansRaw.length) return [];
	            const spans = (typeof _annolidOrderSpanIndicesForReading === "function")
	              ? _annolidOrderSpanIndicesForReading(spansRaw)
	              : spansRaw.slice();
	            const nodes = window.__annolidSpans || [];
	            const isCjkLike = (ch) => /[\\u3040-\\u30ff\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff\\uac00-\\ud7af，。！？、；：“”‘’（）《》「」『』]/.test(ch || "");
	            const parts = [];
	            let combined = "";
	            spans.forEach((idx) => {{
	              const node = nodes[idx];
	              if (!node) return;
	              const raw = String(node.textContent || "");
	              const cleaned = raw.replace(/\\u00ad/g, "").replace(/\\u00a0/g, " ");
	              const text = cleaned.replace(/\\s+/g, " ").trim();
	              if (!text) return;
	              let sep = "";
	              if (combined) {{
	                const prev = combined[combined.length - 1] || "";
	                const next = text[0] || "";
	                if (!/\\s/.test(prev) && !(isCjkLike(prev) && isCjkLike(next))) {{
	                  sep = " ";
	                }}
	              }}
	              const start = combined.length + sep.length;
	              combined += sep + text;
	              const end = combined.length;
	              parts.push({{ idx, start, end }});
	            }});
	            // Use the exact combined span text for range computation so indices
	            // align with span boundaries; normalize only per sentence output.
	            const paraText = combined;
	            const normalizedText = _annolidNormalizeText(paraText);
	            const resolvedPageNum = (parseInt(para.pageNum || para.page || 0, 10) || 0);
	            if (!normalizedText) return [];
	            const shortCharLimit = 120;
	            const shortSpanLimit = 4;
	            const shortParagraph = (normalizedText.length <= shortCharLimit || spans.length <= shortSpanLimit);
	            if (shortParagraph) {{
	              return [{{
	                text: normalizedText,
	                spans: spans.slice(),
	                pageNum: resolvedPageNum,
	              }}];
	            }}
	            const sentences = [];
	            const ranges = (typeof window.__annolidSplitTextIntoSentenceRanges === "function")
	              ? (window.__annolidSplitTextIntoSentenceRanges(paraText) || [])
	              : [];
	            if (!ranges.length) {{
	              return [{{
	                text: normalizedText,
	                spans: spans.slice(),
	                pageNum: resolvedPageNum,
	              }}];
	            }}
	            for (const r of ranges) {{
	              const start = r[0];
	              const end = r[1];
	              const outText = _annolidNormalizeText(paraText.slice(start, end));
	              if (!outText) continue;
	              const group = parts
	                .filter((p) => p.end > start && p.start < end)
	                .map((p) => p.idx);
	              sentences.push({{
	                text: outText,
	                spans: group.length ? group : spans.slice(),
	                pageNum: resolvedPageNum,
	              }});
	            }}
	            // Post-merge edge cases where a quoted sentence ends and narration continues immediately,
	            // e.g., "。”说完..." / "。”小鹊..." which otherwise may split and reorder.
	            if (sentences.length > 1) {{
	              const merged = [];
	              const endQuoteRe = /[.!?。！？]+[”’"'][\)\]）】》」』〉]*$/;
	              const cjkLeadRe = /^[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff（【“‘]/;
	              let i = 0;
	              while (i < sentences.length) {{
	                let cur = sentences[i];
	                if (i + 1 < sentences.length && endQuoteRe.test(cur.text || "")) {{
	                  const next = sentences[i + 1];
	                  if (cjkLeadRe.test((next.text || "").trim())) {{
	                    const nextText = String(next.text || "");
	                    const joiner = cjkLeadRe.test(nextText) ? "" : " ";
	                    const mergedText = _annolidNormalizeText((cur.text || "") + joiner + nextText);
	                    const mergedSpansRaw = []
	                      .concat(Array.isArray(cur.spans) ? cur.spans : [])
	                      .concat(Array.isArray(next.spans) ? next.spans : []);
	                    const mergedSpans = (typeof _annolidOrderSpanIndicesForReading === "function")
	                      ? _annolidOrderSpanIndicesForReading(mergedSpansRaw)
	                      : mergedSpansRaw;
	                    merged.push({{
	                      text: mergedText,
	                      spans: mergedSpans,
	                      pageNum: resolvedPageNum,
	                    }});
	                    i += 2;
	                    continue;
	                  }}
	                }}
	                merged.push(cur);
	                i += 1;
	              }}
	              sentences.length = 0;
	              merged.forEach((m) => sentences.push(m));
	            }}
	            if (sentences.length > 1) {{
	              const totalLen = sentences.reduce((s, item) => s + (item.text || "").length, 0);
	              const avgLen = totalLen / Math.max(1, sentences.length);
	              if (avgLen < 45 && normalizedText.length <= 180) {{
	                return [{{
	                  text: normalizedText,
	                  spans: spans.slice(),
	                  pageNum: resolvedPageNum,
	                }}];
	              }}
	            }}
	            if (!sentences.length && paraText) {{
	              sentences.push({{
	                text: normalizedText,
	                spans: spans.slice(),
	                pageNum: resolvedPageNum,
	              }});
	            }}
	            return sentences;
	          }} catch (e) {{
	            return [];
	          }}
	        }};
        window.__annolidBridge = null;
        window.__annolidRenderedPages = 0;
        window.__annolidPdfLoaded = false;
        window.__annolidLinkTargets = {{}};
        window.__annolidLinkTargetCounter = 0;

        if (window.qt && window.qt.webChannelTransport && typeof QWebChannel !== "undefined") {{
          try {{
            new QWebChannel(window.qt.webChannelTransport, function(channel) {{
              window.__annolidBridge = channel.objects.annolidBridge || null;
            }});
          }} catch (e) {{
            window.__annolidBridge = null;
          }}
        }}

        function _annolidHexToRgba(hex, alpha) {{
          const m = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(hex || "");
          if (!m) return `rgba(255, 179, 0, ${{alpha}})`;
          const r = parseInt(m[1], 16);
          const g = parseInt(m[2], 16);
          const b = parseInt(m[3], 16);
          return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
        }}

        function _annolidSetupHiDpiCanvas(canvas, cssWidth, cssHeight) {{
          const dpr = (window.devicePixelRatio || 1);
          const w = Math.max(1, Math.round(cssWidth * dpr));
          const h = Math.max(1, Math.round(cssHeight * dpr));
          canvas.width = w;
          canvas.height = h;
          canvas.style.width = cssWidth + "px";
          canvas.style.height = cssHeight + "px";
          const ctx = canvas.getContext("2d");
          if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          return {{ ctx, dpr }};
        }}

        function _annolidClearCanvas(canvas, ctx, dpr) {{
          if (!canvas || !ctx) return;
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }}

        function _annolidGetPageState(pageNum) {{
          const key = String(pageNum);
          let state = window.__annolidPages[key];
          if (!state) {{
            state = {{
              pageNum,
              pageDiv: null,
              width: 0,
              height: 0,
              dpr: 1,
              ttsCanvas: null,
              ttsCtx: null,
              markCanvas: null,
              markCtx: null,
              marks: {{ highlights: [], strokes: [] }},
            }};
            window.__annolidPages[key] = state;
          }}
          if (!state.marks) state.marks = {{ highlights: [], strokes: [] }};
          if (!state.marks.highlights) state.marks.highlights = [];
          if (!state.marks.strokes) state.marks.strokes = [];
          return state;
        }}

        function _annolidPointFromEvent(ev, canvas) {{
          const rect = canvas.getBoundingClientRect();
          const x = (ev.clientX - rect.left) * (canvas.width / rect.width) / (window.devicePixelRatio || 1);
          const y = (ev.clientY - rect.top) * (canvas.height / rect.height) / (window.devicePixelRatio || 1);
          return {{ x, y }};
        }}

        function _annolidStrokeAlpha(tool) {{
          if (tool === "highlighter") return 0.28;
          return 0.92;
        }}

        function _annolidDrawStrokeSegment(ctx, from, to, tool, color, size) {{
          if (!ctx || !from || !to) return;
          ctx.save();
          ctx.globalAlpha = _annolidStrokeAlpha(tool);
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(1, size);
          ctx.lineJoin = "round";
          ctx.lineCap = "round";
          if (tool === "highlighter") {{
            try {{ ctx.globalCompositeOperation = "multiply"; }} catch (e) {{}}
          }}
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.stroke();
          ctx.restore();
        }}

        function _annolidRenderMarks(pageNum) {{
          const state = _annolidGetPageState(pageNum);
          if (!state || !state.markCanvas || !state.markCtx) return;
          _annolidClearCanvas(state.markCanvas, state.markCtx, state.dpr || 1);
          for (const hl of (state.marks.highlights || [])) {{
            const fill = _annolidHexToRgba(hl.color || "#ffb300", hl.alpha || 0.28);
            const rects = hl.rects || [];
            state.markCtx.save();
            state.markCtx.fillStyle = fill;
            for (const r of rects) {{
              const pad = 1.5;
              state.markCtx.fillRect(
                Math.max(0, r.x - pad),
                Math.max(0, r.y - pad),
                Math.max(0, r.w + pad * 2),
                Math.max(0, r.h + pad * 2),
              );
            }}
            state.markCtx.restore();
          }}
          for (const stroke of (state.marks.strokes || [])) {{
            const points = stroke.points || [];
            for (let i = 1; i < points.length; i++) {{
              _annolidDrawStrokeSegment(
                state.markCtx,
                points[i - 1],
                points[i],
                stroke.tool,
                stroke.color,
                stroke.size,
              );
            }}
          }}
        }}

		        function _annolidGetSpanMeta(idx) {{
		          if (window.__annolidSpanMeta && window.__annolidSpanMeta[idx]) {{
		            const cached = window.__annolidSpanMeta[idx];
		            if (cached && isFinite(cached.x) && isFinite(cached.y) && isFinite(cached.w) && isFinite(cached.h) && cached.w > 0 && cached.h > 0) {{
		              return cached;
		            }}
		          }}
		          const spans = window.__annolidSpans || [];
		          const span = spans[idx];
		          if (!span) return null;
		          const pageDiv = span.closest ? span.closest(".page") : null;
		          if (!pageDiv) return null;
		          const pageNumRaw = pageDiv.getAttribute("data-page-number") || "0";
		          const pageNum = parseInt(pageNumRaw, 10) || 0;
		          const pageRect = pageDiv.getBoundingClientRect();
		          let spanRect = null;
		          try {{ spanRect = span.getBoundingClientRect(); }} catch (e) {{ spanRect = null; }}
		          if (!spanRect || !isFinite(spanRect.width) || !isFinite(spanRect.height) || spanRect.width <= 0 || spanRect.height <= 0) {{
		            try {{
		              const rects = span.getClientRects ? span.getClientRects() : [];
		              for (let i = 0; i < rects.length; i++) {{
		                const r = rects[i];
		                if (r && isFinite(r.width) && isFinite(r.height) && r.width > 0 && r.height > 0) {{
		                  spanRect = r;
		                  break;
		                }}
		              }}
		            }} catch (e) {{
		              spanRect = null;
		            }}
		          }}
		          if (!spanRect || !isFinite(spanRect.width) || !isFinite(spanRect.height) || spanRect.width <= 0 || spanRect.height <= 0) return null;
		          const meta = {{
		            pageNum,
		            x: spanRect.left - pageRect.left,
		            y: spanRect.top - pageRect.top,
		            w: spanRect.width,
	            h: spanRect.height,
	          }};
	          if (window.__annolidSpanMeta && isFinite(meta.w) && isFinite(meta.h) && meta.w > 0 && meta.h > 0) {{
	            window.__annolidSpanMeta[idx] = meta;
	          }}
	          return meta;
	        }}

        function _annolidRenderTts() {{
          const tts = window.__annolidTts || {{ sentenceIndices: [], wordIndex: null, lastPages: [] }};
          const sentence = tts.sentenceIndices || [];
          const wordIndex = (tts.wordIndex === 0 || tts.wordIndex) ? tts.wordIndex : null;
          const pagesToDraw = {{}};
          function addMeta(kind, idx) {{
            const meta = _annolidGetSpanMeta(idx);
            if (!meta) return;
            const key = String(meta.pageNum);
            if (!pagesToDraw[key]) pagesToDraw[key] = {{ sentence: [], word: null }};
            if (kind === "sentence") pagesToDraw[key].sentence.push(meta);
            if (kind === "word") pagesToDraw[key].word = meta;
          }}
          for (const idx of sentence) addMeta("sentence", idx);
          if (wordIndex !== null) addMeta("word", wordIndex);

          const pageKeys = new Set([...(tts.lastPages || []), ...Object.keys(pagesToDraw)]);
          for (const key of pageKeys) {{
            const state = window.__annolidPages[key];
            if (!state || !state.ttsCanvas || !state.ttsCtx) continue;
            _annolidClearCanvas(state.ttsCanvas, state.ttsCtx, state.dpr || 1);
          }}
          for (const key of Object.keys(pagesToDraw)) {{
            const state = window.__annolidPages[key];
            if (!state || !state.ttsCanvas || !state.ttsCtx) continue;
            const entry = pagesToDraw[key];
            // Sentence highlight (soft)
            state.ttsCtx.save();
            state.ttsCtx.fillStyle = "rgba(255, 210, 80, 0.30)";
            for (const meta of (entry.sentence || [])) {{
              const pad = 1.0;
              state.ttsCtx.fillRect(
                Math.max(0, meta.x - pad),
                Math.max(0, meta.y - pad),
                Math.max(0, meta.w + pad * 2),
                Math.max(0, meta.h + pad * 2),
              );
            }}
            state.ttsCtx.restore();
            // Word highlight (strong)
            if (entry.word) {{
              const meta = entry.word;
              state.ttsCtx.save();
              state.ttsCtx.fillStyle = "rgba(255, 140, 0, 0.55)";
              const pad = 1.6;
              state.ttsCtx.fillRect(
                Math.max(0, meta.x - pad),
                Math.max(0, meta.y - pad),
                Math.max(0, meta.w + pad * 2),
                Math.max(0, meta.h + pad * 2),
              );
              state.ttsCtx.restore();
            }}
          }}
          tts.lastPages = Object.keys(pagesToDraw);
          window.__annolidTts = tts;
        }}

        window.__annolidRenderTts = _annolidRenderTts;
        window.__annolidClearSentenceHighlight = function() {{
          if (!window.__annolidTts) window.__annolidTts = {{ sentenceIndices: [], wordIndex: null, lastPages: [] }};
          window.__annolidTts.sentenceIndices = [];
          _annolidRenderTts();
        }};
        window.__annolidClearWordHighlight = function() {{
          if (!window.__annolidTts) window.__annolidTts = {{ sentenceIndices: [], wordIndex: null, lastPages: [] }};
          window.__annolidTts.wordIndex = null;
          _annolidRenderTts();
        }};
        window.__annolidClearHighlight = function() {{
          if (!window.__annolidTts) window.__annolidTts = {{ sentenceIndices: [], wordIndex: null, lastPages: [] }};
          window.__annolidTts.sentenceIndices = [];
          window.__annolidTts.wordIndex = null;
          _annolidRenderTts();
        }};
        window.__annolidHighlightSentenceIndices = function(indices) {{
          if (!window.__annolidTts) window.__annolidTts = {{ sentenceIndices: [], wordIndex: null, lastPages: [] }};
          window.__annolidTts.sentenceIndices = (indices && indices.length) ? indices : [];
          window.__annolidTts.wordIndex = null;
          _annolidRenderTts();
        }};
        window.__annolidHighlightWordIndex = function(idx) {{
          if (!window.__annolidTts) window.__annolidTts = {{ sentenceIndices: [], wordIndex: null, lastPages: [] }};
          window.__annolidTts.wordIndex = idx;
          _annolidRenderTts();
        }};
        window.__annolidHighlightSelection = function() {{
          const indices = window.__annolidSelectionSpans || [];
          window.__annolidHighlightSentenceIndices(indices);
        }};
        window.__annolidHighlightParagraphIndices = function(indices) {{
          window.__annolidHighlightSentenceIndices(indices);
        }};
        window.__annolidSetReaderEnabled = function(enabled) {{
          window.__annolidReaderEnabled = !!enabled;
          try {{
            document.body.classList.toggle("annolid-reader-enabled", window.__annolidReaderEnabled);
          }} catch (e) {{}}
        }};
	        window.__annolidScrollToPage = function(pageNum) {{
	          try {{
	            const n = parseInt(pageNum, 10) || 1;
	            if (typeof _annolidGoToPage === "function") {{
	              _annolidGoToPage(n);
	              return;
	            }}
	          }} catch (e) {{}}
	          const container = document.getElementById("viewerContainer");
	          if (!container) return;
	          const target = document.querySelector(`.page[data-page-number='${{pageNum}}']`);
	          if (!target) return;
	          const offset = Math.max(0, target.offsetTop - 60);
	          try {{
	            container.scrollTo({{ top: offset, behavior: "smooth" }});
	          }} catch (e) {{
	            container.scrollTop = offset;
	          }}
	        }};
        window.__annolidScrollToSentence = function(indices, pageNum) {{
          try {{
            const spans = Array.isArray(indices) ? indices : [];
            if (!spans.length) return;
            const container = document.getElementById("viewerContainer");
            if (!container) return;
            const requestedPage = (parseInt(pageNum || 0, 10) || 0);

            const now = Date.now();
            const key = String(requestedPage) + ":" + spans.slice(0, 10).join(",");
            const last = window.__annolidLastSentenceScroll || null;
            if (last && last.key === key && (now - (last.t || 0)) < 120) {{
              return;
            }}
            window.__annolidLastSentenceScroll = {{ key, t: now }};

            const scrollBehavior = "smooth";
            const minMovePx = 18;
            const safeHeight = Math.max(1, container.clientHeight || 1);
            const comfortMargin = Math.min(220, Math.max(80, safeHeight * 0.25));

            const clampScroll = (value) => {{
              const maxScroll = Math.max(0, (container.scrollHeight || 0) - safeHeight);
              return Math.max(0, Math.min(maxScroll, value));
            }};

            const scrollToTop = (target) => {{
              const clamped = clampScroll(target);
              if (Math.abs((container.scrollTop || 0) - clamped) < minMovePx) return;
              try {{
                container.scrollTo({{ top: clamped, behavior: scrollBehavior }});
              }} catch (e) {{
                container.scrollTop = clamped;
              }}
            }};

            const scrollToPageCenter = (pageNumToUse) => {{
              const page = document.querySelector(`.page[data-page-number='${{pageNumToUse}}']`);
              if (!page) return;
              const center = (page.offsetTop || 0) + (page.clientHeight || 0) / 2;
              scrollToTop(center - safeHeight / 2);
            }};

            const groupMetasByPage = (metas) => {{
              const groups = {{}};
              metas.forEach((m) => {{
                const p = m && (m.pageNum || 0);
                if (!p) return;
                const k = String(p);
                if (!groups[k]) groups[k] = [];
                groups[k].push(m);
              }});
              return groups;
            }};

            const pickPageGroup = (groups) => {{
              const keys = Object.keys(groups || {{}});
              if (!keys.length) return {{ pageNum: 0, metas: [] }};
              if (requestedPage > 0 && groups[String(requestedPage)] && groups[String(requestedPage)].length) {{
                return {{ pageNum: requestedPage, metas: groups[String(requestedPage)] }};
              }}
              let bestKey = keys[0];
              let bestCount = (groups[bestKey] || []).length;
              keys.forEach((k) => {{
                const c = (groups[k] || []).length;
                if (c > bestCount) {{
                  bestCount = c;
                  bestKey = k;
                }}
              }});
              return {{ pageNum: (parseInt(bestKey, 10) || 0), metas: groups[bestKey] || [] }};
            }};

            const computeSentenceBox = (metas, pageEl) => {{
              if (!metas || !metas.length || !pageEl) return null;
              let minY = Infinity;
              let maxY = -Infinity;
              metas.forEach((m) => {{
                const y0 = (m.y || 0);
                const y1 = (m.y || 0) + (m.h || 0);
                if (isFinite(y0)) minY = Math.min(minY, y0);
                if (isFinite(y1)) maxY = Math.max(maxY, y1);
              }});
              if (!isFinite(minY) || !isFinite(maxY)) return null;
              const absTop = (pageEl.offsetTop || 0) + minY;
              const absBottom = (pageEl.offsetTop || 0) + maxY;
              const absCenter = (absTop + absBottom) / 2;
              return {{ absTop, absBottom, absCenter }};
            }};

            const isComfortablyVisible = (box) => {{
              if (!box) return false;
              const topBound = (container.scrollTop || 0) + comfortMargin;
              const bottomBound = (container.scrollTop || 0) + safeHeight - comfortMargin;
              return box.absCenter >= topBound && box.absCenter <= bottomBound;
            }};

            const ensureRenderedThroughIfNeeded = (pageNumToUse) => {{
              if (pageNumToUse > 0 && typeof _annolidEnsureRenderedThrough === "function") {{
                const lastEnsure = window.__annolidLastEnsurePage || 0;
                if (pageNumToUse > lastEnsure) {{
                  window.__annolidLastEnsurePage = pageNumToUse;
                  return _annolidEnsureRenderedThrough(pageNumToUse);
                }}
              }}
              return Promise.resolve();
            }};

            const runScrollAttempt = (attempt) => {{
              const metas = spans
                .map((idx) => _annolidGetSpanMeta(idx))
                .filter((m) => m && Number.isFinite(m.y) && Number.isFinite(m.h) && (m.pageNum || 0) > 0);

              if (!metas.length) {{
                if (requestedPage > 0) scrollToPageCenter(requestedPage);
                if (attempt < 8) requestAnimationFrame(() => runScrollAttempt(attempt + 1));
                return;
              }}

              const groups = groupMetasByPage(metas);
              const picked = pickPageGroup(groups);
              const pageNumToUse = picked.pageNum || requestedPage || 0;
              const pageEl = pageNumToUse
                ? document.querySelector(`.page[data-page-number='${{pageNumToUse}}']`)
                : null;

              if (!pageEl) {{
                if (pageNumToUse > 0 && typeof _annolidGoToPage === "function") {{
                  _annolidGoToPage(pageNumToUse);
                }}
                if (attempt < 8) requestAnimationFrame(() => runScrollAttempt(attempt + 1));
                return;
              }}

              const box = computeSentenceBox(picked.metas, pageEl);
              if (!box) {{
                if (pageNumToUse > 0) scrollToPageCenter(pageNumToUse);
                return;
              }}
              if (isComfortablyVisible(box)) return;
              scrollToTop(box.absCenter - safeHeight / 2);
            }};

            const initialPage = requestedPage || 0;
            Promise.resolve(ensureRenderedThroughIfNeeded(initialPage))
              .then(() => requestAnimationFrame(() => runScrollAttempt(0)))
              .catch(() => requestAnimationFrame(() => runScrollAttempt(0)));
          }} catch (e) {{}}
        }};

        function _annolidSetTool(tool) {{
          window.__annolidMarks.tool = tool;
          const drawing = tool !== "select";
          document.body.classList.toggle("annolid-drawing", drawing);
          if (drawing) {{
            _annolidHideCitationPopover();
          }}
          const btns = [
            ["select", document.getElementById("annolidToolSelect")],
            ["pen", document.getElementById("annolidToolPen")],
            ["highlighter", document.getElementById("annolidToolHighlighter")],
          ];
          btns.forEach(([name, el]) => {{
            if (!el) return;
            if (name === tool) el.classList.add("annolid-active");
            else el.classList.remove("annolid-active");
          }});
          const pages = window.__annolidPages || {{}};
          Object.keys(pages).forEach((key) => {{
            const state = pages[key];
            if (!state || !state.markCanvas) return;
            state.markCanvas.style.pointerEvents = drawing ? "auto" : "none";
            state.markCanvas.style.cursor = drawing ? "crosshair" : "default";
          }});
        }}

        function _annolidBindMarkCanvas(state) {{
          const canvas = state && state.markCanvas;
          if (!canvas || canvas.__annolidBound) return;
          canvas.__annolidBound = true;

          canvas.addEventListener("pointerdown", (ev) => {{
            if ((window.__annolidMarks.tool || "select") === "select") return;
            if (ev.button !== 0) return;
            const pt = _annolidPointFromEvent(ev, canvas);
            const tool = window.__annolidMarks.tool;
            const stroke = {{
              tool,
              color: window.__annolidMarks.color || "#ffb300",
              size: window.__annolidMarks.size || 10,
              points: [pt],
            }};
            window.__annolidMarks.drawing = {{ pageNum: state.pageNum, stroke, pointerId: ev.pointerId }};
            try {{ canvas.setPointerCapture(ev.pointerId); }} catch (e) {{}}
            ev.preventDefault();
          }});

          canvas.addEventListener("pointermove", (ev) => {{
            const drawing = window.__annolidMarks.drawing;
            if (!drawing || drawing.pageNum !== state.pageNum) return;
            if (!state.markCtx) return;
            const stroke = drawing.stroke;
            const pts = stroke.points || [];
            const pt = _annolidPointFromEvent(ev, canvas);
            if (pts.length) {{
              _annolidDrawStrokeSegment(state.markCtx, pts[pts.length - 1], pt, stroke.tool, stroke.color, stroke.size);
            }}
            pts.push(pt);
            stroke.points = pts;
            drawing.stroke = stroke;
            window.__annolidMarks.drawing = drawing;
            ev.preventDefault();
          }});

          function finishStroke(ev) {{
            const drawing = window.__annolidMarks.drawing;
            if (!drawing || drawing.pageNum !== state.pageNum) return;
            const stroke = drawing.stroke;
            window.__annolidMarks.drawing = null;
            if (stroke && stroke.points && stroke.points.length > 1) {{
              state.marks.strokes.push(stroke);
              window.__annolidMarks.undo.push({{ type: "stroke", pageNum: state.pageNum }});
              _annolidRenderMarks(state.pageNum);
            }}
            try {{ canvas.releasePointerCapture(ev.pointerId); }} catch (e) {{}}
            ev.preventDefault();
          }}

          canvas.addEventListener("pointerup", finishStroke);
          canvas.addEventListener("pointercancel", finishStroke);
        }}

        function _annolidManualHighlightSelection() {{
          const indices = window.__annolidSelectionSpans || [];
          if (!indices.length) return;
          const grouped = {{}};
          for (const idx of indices) {{
            const meta = _annolidGetSpanMeta(idx);
            if (!meta) continue;
            const key = String(meta.pageNum);
            if (!grouped[key]) grouped[key] = [];
            grouped[key].push({{ x: meta.x, y: meta.y, w: meta.w, h: meta.h }});
          }}
          const color = window.__annolidMarks.color || "#ffb300";
          for (const key of Object.keys(grouped)) {{
            const pageNum = parseInt(key, 10) || 0;
            const state = _annolidGetPageState(pageNum);
            state.marks.highlights.push({{
              rects: grouped[key],
              color,
              alpha: 0.28,
            }});
            window.__annolidMarks.undo.push({{ type: "highlight", pageNum }});
            _annolidRenderMarks(pageNum);
          }}
        }}

        function _annolidUndo() {{
          const op = (window.__annolidMarks.undo || []).pop();
          if (!op) return;
          const state = _annolidGetPageState(op.pageNum);
          if (op.type === "stroke") state.marks.strokes.pop();
          if (op.type === "highlight") state.marks.highlights.pop();
          _annolidRenderMarks(op.pageNum);
        }}

        function _annolidClearMarks() {{
          window.__annolidMarks.undo = [];
          const pages = window.__annolidPages || {{}};
          Object.keys(pages).forEach((key) => {{
            const state = pages[key];
            if (!state || !state.marks) return;
            state.marks.strokes = [];
            state.marks.highlights = [];
            _annolidRenderMarks(state.pageNum);
          }});
        }}

        // Toolbar bindings.
        const selectBtn = document.getElementById("annolidToolSelect");
        const penBtn = document.getElementById("annolidToolPen");
        const hiBtn = document.getElementById("annolidToolHighlighter");
        const highlightBtn = document.getElementById("annolidHighlightSelection");
        const undoBtn = document.getElementById("annolidUndo");
        const clearBtn = document.getElementById("annolidClear");
        const colorInput = document.getElementById("annolidColor");
        const sizeInput = document.getElementById("annolidSize");
        const colorInline = document.getElementById("annolidColorInline");
        const sizeInline = document.getElementById("annolidSizeInline");

        if (selectBtn) selectBtn.addEventListener("click", () => _annolidSetTool("select"));
        if (penBtn) penBtn.addEventListener("click", () => _annolidSetTool("pen"));
        if (hiBtn) hiBtn.addEventListener("click", () => _annolidSetTool("highlighter"));
        if (highlightBtn) highlightBtn.addEventListener("click", _annolidManualHighlightSelection);
        if (undoBtn) undoBtn.addEventListener("click", _annolidUndo);
        if (clearBtn) clearBtn.addEventListener("click", _annolidClearMarks);

        function _annolidSetStrokeColor(value) {{
          const color = value || "#ffb300";
          window.__annolidMarks.color = color;
          if (colorInput && colorInput.value !== color) colorInput.value = color;
          if (colorInline && colorInline.value !== color) colorInline.value = color;
        }}

        function _annolidSetStrokeSize(value) {{
          const v = parseFloat(value);
          const clamped = isFinite(v) ? Math.max(2, Math.min(24, v)) : 10;
          window.__annolidMarks.size = clamped;
          const str = String(clamped);
          if (sizeInput && sizeInput.value !== str) sizeInput.value = str;
          if (sizeInline && sizeInline.value !== str) sizeInline.value = str;
        }}

        if (colorInput) {{
          colorInput.addEventListener("input", (ev) => {{
            _annolidSetStrokeColor(ev.target.value);
          }});
        }}
        if (colorInline) {{
          colorInline.addEventListener("input", (ev) => {{
            _annolidSetStrokeColor(ev.target.value);
          }});
        }}
        if (sizeInput) {{
          sizeInput.addEventListener("input", (ev) => {{
            _annolidSetStrokeSize(ev.target.value);
          }});
        }}
        if (sizeInline) {{
          sizeInline.addEventListener("input", (ev) => {{
            _annolidSetStrokeSize(ev.target.value);
          }});
        }}

        _annolidSetStrokeColor(window.__annolidMarks.color || "#ffb300");
        _annolidSetStrokeSize(window.__annolidMarks.size || 10);

        _annolidSetTool("select");
        window.__annolidSetReaderEnabled(window.__annolidReaderEnabled);

        function _annolidNormalizeText(text) {{
          // Normalize whitespace and drop discretionary (soft) hyphens.
          return String(text || "").replace(/\\u00ad/g, "").replace(/\\s+/g, " ").trim();
        }}

        function _annolidMergeHyphenated(prevText, nextText) {{
          const prev = String(prevText || "");
          const next = String(nextText || "");
          const nextTrim = next.trimStart();
          if (!nextTrim) return _annolidNormalizeText(prev);
          const m = /([A-Za-z]{1,})[-\\u2010\\u2011\\u2012\\u2013]$/.exec(prev.trimEnd());
          if (!m) {{
            return _annolidNormalizeText(prev + " " + nextTrim);
          }}
          // Only treat as line-wrap continuation when the next token starts with lowercase.
          if (!/^[a-z]/.test(nextTrim)) {{
            return _annolidNormalizeText(prev + " " + nextTrim);
          }}
          // Prefer removing the hyphen when it is used only for line wrapping:
          // "pri-" + "marily" -> "primarily", "eluci-" + "date" -> "elucidate".
          // This may also turn "non-" + "linear" into "nonlinear", which is better for TTS.
          return _annolidNormalizeText(
            prev.trimEnd().replace(/[-\\u2010\\u2011\\u2012\\u2013]$/, "") + nextTrim
          );
        }}

        function _annolidMedian(values) {{
          if (!values || !values.length) return 0;
          const sorted = values.slice().sort((a, b) => a - b);
          return sorted[Math.floor(sorted.length / 2)];
        }}

	        function _annolidGroupIntoColumns(entries, pageWidth) {{
	          if (!entries || !entries.length) return [entries];
	          const pw = isFinite(pageWidth) ? Math.max(1, pageWidth) : 1;
	          // Use left edges for column clustering; centers can be noisy for long spans.
	          const centers = entries.map((e) => e.x).filter((x) => isFinite(x));
	          const sorted = centers.slice().sort((a, b) => a - b);
	          if (sorted.length < 12) return [entries];

	          const quantile = (arr, q) => {{
	            if (!arr.length) return 0;
	            const pos = Math.max(0, Math.min(arr.length - 1, Math.round(q * (arr.length - 1))));
	            return arr[pos];
	          }};
          const mean = (arr) => {{
            if (!arr.length) return 0;
            let s = 0;
            for (let i = 0; i < arr.length; i++) s += arr[i];
            return s / arr.length;
          }};
	          const std = (arr, m) => {{
	            if (!arr.length) return 0;
	            let s = 0;
	            for (let i = 0; i < arr.length; i++) {{
	              const d = arr[i] - m;
	              s += d * d;
	            }}
	            return Math.sqrt(s / arr.length);
	          }};

	          // Fast gap-based split (more robust than sepScore when indent noise is high).
	          const q10 = quantile(sorted, 0.10);
	          const q90 = quantile(sorted, 0.90);
	          let bestGap = 0;
	          let bestCut = null;
	          for (let i = 0; i < sorted.length - 1; i++) {{
	            const a0 = sorted[i];
	            const b0 = sorted[i + 1];
	            if (a0 < q10 || b0 > q90) continue;
	            const gap = b0 - a0;
	            if (gap <= bestGap) continue;
	            const cut = (a0 + b0) * 0.5;
	            if (cut < pw * 0.30 || cut > pw * 0.70) continue;
	            bestGap = gap;
	            bestCut = cut;
	          }}
	          const minGap = Math.max(24, pw * 0.06);
	          if (bestCut != null && bestGap >= minGap) {{
	            const left = [];
	            const right = [];
	            entries.forEach((e) => {{
	              const x = e.x;
	              if (!isFinite(x)) return;
	              if (x <= bestCut) left.push(e);
	              else right.push(e);
	            }});
	            const minCount = Math.max(6, Math.floor(entries.length * 0.12));
	            if (left.length >= minCount && right.length >= minCount) {{
	              return [left, right];
	            }}
	          }}

	          // 1D k-means (k=2) to detect a stable two-column split.
	          let m1 = quantile(sorted, 0.2);
	          let m2 = quantile(sorted, 0.8);
	          if (Math.abs(m2 - m1) < 1) return [entries];
          let a = [];
          let b = [];
          for (let iter = 0; iter < 10; iter++) {{
            a = [];
            b = [];
            for (let i = 0; i < centers.length; i++) {{
            const c = centers[i];
            if (Math.abs(c - m1) <= Math.abs(c - m2)) a.push(c);
            else b.push(c);
          }}
            const next1 = mean(a);
            const next2 = mean(b);
            if (Math.abs(next1 - m1) < 0.5 && Math.abs(next2 - m2) < 0.5) break;
            m1 = next1;
            m2 = next2;
          }}
          if (!a.length || !b.length) return [entries];
          const leftMean = Math.min(m1, m2);
          const rightMean = Math.max(m1, m2);
	          const leftStd = std(a, m1);
	          const rightStd = std(b, m2);
	          const separation = rightMean - leftMean;

	          const minSeparation = Math.max(24, pw * 0.08);
	          const denom = Math.max(1e-6, leftStd + rightStd);
	          const sepScore = separation / denom;
	          if (separation < minSeparation) return [entries];
	          if (sepScore < 1.6) return [entries];
	          const cut = (leftMean + rightMean) * 0.5;
	          if (cut < pw * 0.30 || cut > pw * 0.70) return [entries];

	          const left = [];
	          const right = [];
	          entries.forEach((e) => {{
	            const c = e.x;
	            if (Math.abs(c - leftMean) <= Math.abs(c - rightMean)) left.push(e);
	            else right.push(e);
	          }});
	          const minCount = Math.max(6, Math.floor(entries.length * 0.12));
	          if (left.length < minCount || right.length < minCount) return [entries];
	          return [left, right];
	        }}

		        function _annolidGroupLinesIntoColumns(lines, pageWidth) {{
		          const list = Array.isArray(lines)
		            ? lines.filter((l) => l && isFinite(l.xMin) && (isFinite(l.yCenter) || isFinite(l.yMin)))
		            : [];
		          if (!list.length) return [lines || []];
		          const pw = isFinite(pageWidth) ? Math.max(1, pageWidth) : 1;

		          const widthOf = (l) => {{
		            if (!l || !isFinite(l.xMin) || !isFinite(l.xMax)) return 0;
		            return Math.max(0, l.xMax - l.xMin);
		          }};
		          const heightOf = (l) => {{
		            if (!l) return 0;
		            if (isFinite(l.h) && l.h > 0) return l.h;
		            if (isFinite(l.yMax) && isFinite(l.yMin)) return Math.max(1, l.yMax - l.yMin);
		            return 0;
		          }};
		          const yCenterOf = (l) => {{
		            if (!l) return 0;
		            if (isFinite(l.yCenter)) return l.yCenter;
		            if (isFinite(l.yMin) && isFinite(l.yMax)) return (l.yMin + l.yMax) * 0.5;
		            return 0;
		          }};
		          const xCenterOf = (l) => {{
		            if (!l || !isFinite(l.xMin)) return 0;
		            const w = widthOf(l);
		            return l.xMin + w * 0.5;
		          }};

		          const sortLines = (arr) => {{
		            const out = (arr || []).slice();
		            const hs = out.map((l) => heightOf(l)).filter((h) => isFinite(h) && h > 0);
		            const medH = _annolidMedian(hs);
		            const sameRowTol = Math.max(2, (medH || 0) * 0.35);
		            out.sort((a, b) => {{
		              const ay = yCenterOf(a);
		              const by = yCenterOf(b);
		              const dy = ay - by;
		              if (Math.abs(dy) <= sameRowTol) {{
		                const ax = (a && isFinite(a.xMin)) ? a.xMin : 0;
		                const bx = (b && isFinite(b.xMin)) ? b.xMin : 0;
		                if (Math.abs(ax - bx) > 0.5) return ax - bx;
		                return dy;
		              }}
		              return dy;
		            }});
		            return out;
		          }};

		          // Exclude very wide lines (titles, centered headers) from column detection.
		          const usable = list.filter((l) => {{
		            const w = widthOf(l);
		            return w > 0 && w <= pw * 0.78;
		          }});
		          const xsSource = (usable.length >= 6) ? usable : ((usable.length >= 4) ? usable : list);
		          const xs = xsSource.map((l) => xCenterOf(l)).filter((x) => isFinite(x));
		          if (xs.length < 6) return [sortLines(list)];
		          xs.sort((a, b) => a - b);

		          const q = (arr, t) => arr[Math.max(0, Math.min(arr.length - 1, Math.round(t * (arr.length - 1))))];
		          const mean = (arr) => {{
		            if (!arr.length) return 0;
		            let s = 0;
		            for (let i = 0; i < arr.length; i++) s += arr[i];
		            return s / arr.length;
		          }};
		          const std = (arr, m) => {{
		            if (!arr.length) return 0;
		            let s = 0;
		            for (let i = 0; i < arr.length; i++) {{
		              const d = arr[i] - m;
		              s += d * d;
		            }}
		            return Math.sqrt(s / arr.length);
		          }};

		          const minCount = Math.max(3, Math.min(8, Math.floor(xsSource.length * 0.22)));

		          function classifyByCut(cut, leftMean, rightMean) {{
		            const left = [];
		            const right = [];
		            list.forEach((l) => {{
		              if (!l) return;
		              const w = widthOf(l);
		              // Keep wide lines with the left column so headings stay before body text.
		              if (w > pw * 0.85) {{
		                left.push(l);
		                return;
		              }}
		              const cx = xCenterOf(l);
		              if (!isFinite(cx)) return;
		              if (leftMean != null && rightMean != null) {{
		                if (Math.abs(cx - leftMean) <= Math.abs(cx - rightMean)) left.push(l);
		                else right.push(l);
		                return;
		              }}
		              if (cx <= cut) left.push(l);
		              else right.push(l);
		            }});
		            if (left.length >= minCount && right.length >= minCount) {{
		              return [sortLines(left), sortLines(right)];
		            }}
		            return null;
		          }}

		          // Fast gap-based split on x centers.
		          const q10 = q(xs, 0.10);
		          const q90 = q(xs, 0.90);
		          let bestGap = 0;
		          let bestCut = null;
		          for (let i = 0; i < xs.length - 1; i++) {{
		            const a0 = xs[i];
		            const b0 = xs[i + 1];
		            if (a0 < q10 || b0 > q90) continue;
		            const gap = b0 - a0;
		            if (gap <= bestGap) continue;
		            const cut = (a0 + b0) * 0.5;
		            if (cut < pw * 0.35 || cut > pw * 0.65) continue;
		            bestGap = gap;
		            bestCut = cut;
		          }}
		          const minGap = Math.max(28, pw * 0.10);
		          if (bestCut != null && bestGap >= minGap) {{
		            const res = classifyByCut(bestCut, null, null);
		            if (res) return res;
		          }}

		          // 1D k-means (k=2) fallback for indented/variable line starts.
		          let m1 = q(xs, 0.25);
		          let m2 = q(xs, 0.75);
		          if (Math.abs(m2 - m1) < 1) return [sortLines(list)];
		          let a = [];
		          let b = [];
		          for (let iter = 0; iter < 10; iter++) {{
		            a = [];
		            b = [];
		            for (let i = 0; i < xs.length; i++) {{
		              const x = xs[i];
		              if (Math.abs(x - m1) <= Math.abs(x - m2)) a.push(x);
		              else b.push(x);
		            }}
		            const nm1 = mean(a);
		            const nm2 = mean(b);
		            if (Math.abs(nm1 - m1) < 0.5 && Math.abs(nm2 - m2) < 0.5) break;
		            m1 = nm1;
		            m2 = nm2;
		          }}
		          if (!a.length || !b.length) return [sortLines(list)];
		          const leftMean = Math.min(m1, m2);
		          const rightMean = Math.max(m1, m2);
		          const separation = rightMean - leftMean;
		          const minSeparation = Math.max(48, pw * 0.18);
		          if (separation < minSeparation) return [sortLines(list)];
		          const score = separation / Math.max(1e-6, std(a, m1) + std(b, m2));
		          if (score < 1.25) return [sortLines(list)];
		          const cut = (leftMean + rightMean) * 0.5;
		          if (cut < pw * 0.35 || cut > pw * 0.65) return [sortLines(list)];
		          const res = classifyByCut(cut, leftMean, rightMean);
		          if (res) return res;
		          return [sortLines(list)];
		        }}

        function _annolidBuildLinesFromEntries(entries) {{
          const list = Array.isArray(entries) ? entries.filter((e) => e && isFinite(e.x) && isFinite(e.y)) : [];
          if (!list.length) return [];
          const typicalH = _annolidMedian(list.map((e) => e.h).filter((h) => isFinite(h) && h > 0));
          // Use a slightly larger tolerance than the raw bbox height since some PDFs
          // jitter punctuation/quotes vertically (e.g. around "。”), which can split lines.
          const yTol = Math.max(2, (typicalH || 0) * 1.05);
          // Prevent merging different columns into the same "line" when y matches.
          // This is essential for correct two-column reading/highlighting.
          let minX = Infinity;
          let maxX = -Infinity;
          for (const e of list) {{
            const x0 = e.x;
            const x1 = e.x + (e.w || 0);
            if (isFinite(x0)) minX = Math.min(minX, x0);
            if (isFinite(x1)) maxX = Math.max(maxX, x1);
          }}
          const spanWidth = (isFinite(minX) && isFinite(maxX) && maxX > minX) ? (maxX - minX) : 0;
          // Use a conservative horizontal gap threshold so left/right columns never merge.
          // (~2% of page text width, clamped)
          let joinGapX = Math.max(10, Math.min(28, spanWidth * 0.02));
          // Optional column cut hint from a max-gap in x positions (helps when some spans have
          // oversized bounding boxes that reduce the apparent inter-column gap).
          let columnCut = null;
          try {{
            const xs = list.map((e) => e.x).filter((x) => isFinite(x)).sort((a, b) => a - b);
            if (xs.length >= 32 && spanWidth > 0) {{
              const q = (arr, t) => arr[Math.max(0, Math.min(arr.length - 1, Math.round(t * (arr.length - 1))))];
              const q10 = q(xs, 0.10);
              const q90 = q(xs, 0.90);
              let bestGap = 0;
              let bestCut = null;
              for (let i = 0; i < xs.length - 1; i++) {{
                const a0 = xs[i];
                const b0 = xs[i + 1];
                if (a0 < q10 || b0 > q90) continue;
                const gap = b0 - a0;
                if (gap <= bestGap) continue;
                const cut = (a0 + b0) * 0.5;
                const frac = (cut - minX) / spanWidth;
                if (!isFinite(frac) || frac < 0.30 || frac > 0.70) continue;
                bestGap = gap;
                bestCut = cut;
              }}
              const minGap = Math.max(14, spanWidth * 0.025);
              if (bestCut != null && bestGap >= minGap) {{
                columnCut = bestCut;
                // Ensure our join threshold is comfortably smaller than the inferred column gap.
                joinGapX = Math.min(joinGapX, Math.max(8, bestGap * 0.45));
              }}
            }}
          }} catch (e) {{
            columnCut = null;
          }}
          const withCenter = list.map((e) => {{
            const h = isFinite(e.h) ? e.h : 0;
            const yCenter = e.y + h * 0.5;
            return Object.assign({{}}, e, {{ yCenter }});
          }});
          withCenter.sort((a, b) => {{
            const dy = a.yCenter - b.yCenter;
            if (Math.abs(dy) < Math.max(0.5, yTol * 0.15)) return a.x - b.x;
            return dy;
          }});

          const lines = [];
          for (const entry of withCenter) {{
            const y = entry.yCenter;
            let best = null;
            let bestOverlap = 0;
            let bestDist = Infinity;
            const scanStart = Math.max(0, lines.length - 6);
            for (let i = lines.length - 1; i >= scanStart; i--) {{
              const cand = lines[i];
              if (!cand) continue;
              const dist = Math.abs(y - cand.yCenter);
              const overlap = Math.min(entry.y + entry.h, cand.yMax) - Math.max(entry.y, cand.yMin);
              const candH = Math.max(1, cand.yMax - cand.yMin);
              const minH = Math.max(1, Math.min(entry.h || candH, candH));
              const overlapRatio = (overlap > 0) ? (overlap / minH) : 0;
              const gapX = (() => {{
                const eLeft = entry.x;
                const eRight = entry.x + (entry.w || 0);
                const cLeft = cand.xMin;
                const cRight = cand.xMax;
                if (eRight < cLeft) return cLeft - eRight;
                if (eLeft > cRight) return eLeft - cRight;
                return 0;
              }})();
              // Hard guard against cross-column merges when a column split is detected.
              if (columnCut != null) {{
                const margin = Math.max(4, (typicalH || 0) * 0.35);
                const candIsLeft = (cand.xMax <= (columnCut - margin));
                const candIsRight = (cand.xMin >= (columnCut + margin));
                if (candIsLeft || candIsRight) {{
                  const entryIsLeft = entry.x <= columnCut;
                  if ((candIsLeft && !entryIsLeft) || (candIsRight && entryIsLeft)) {{
                    continue;
                  }}
                }}
              }}
              if ((dist <= yTol || overlapRatio >= 0.65) && gapX <= joinGapX) {{
                if (
                  overlapRatio > bestOverlap + 0.05 ||
                  (Math.abs(overlapRatio - bestOverlap) <= 0.05 && dist < bestDist)
                ) {{
                  best = cand;
                  bestOverlap = overlapRatio;
                  bestDist = dist;
                }}
              }}
            }}
            let line = best;
            if (!line) {{
              line = {{
                yCenter: y,
                items: [],
                spans: [],
                texts: [],
                yMin: entry.y,
                yMax: entry.y + entry.h,
                xMin: entry.x,
                xMax: entry.x + entry.w,
              }};
              lines.push(line);
            }}
            line.items.push(entry);
            const n = line.items.length;
            line.yCenter = (line.yCenter * (n - 1) + y) / n;
            line.yMin = Math.min(line.yMin, entry.y);
            line.yMax = Math.max(line.yMax, entry.y + entry.h);
            line.xMin = Math.min(line.xMin, entry.x);
            line.xMax = Math.max(line.xMax, entry.x + entry.w);
          }}

          function _annolidLineFromItems(items) {{
            const out = {{
              yCenter: 0,
              spans: [],
              texts: [],
              yMin: Infinity,
              yMax: -Infinity,
              xMin: Infinity,
              xMax: -Infinity,
              h: 1,
            }};
            let n = 0;
            for (const it of items) {{
              if (!it) continue;
              out.spans.push(it.idx);
              out.texts.push(it.text);
              out.yMin = Math.min(out.yMin, it.y);
              out.yMax = Math.max(out.yMax, it.y + (it.h || 0));
              out.xMin = Math.min(out.xMin, it.x);
              out.xMax = Math.max(out.xMax, it.x + (it.w || 0));
              out.yCenter += (it.yCenter || (it.y + (it.h || 0) * 0.5));
              n += 1;
            }}
            if (!n) {{
              out.yCenter = 0;
              out.yMin = 0;
              out.yMax = 1;
              out.xMin = 0;
              out.xMax = 1;
              out.h = 1;
              return out;
            }}
            out.yCenter = out.yCenter / n;
            out.h = Math.max(1, out.yMax - out.yMin);
            return out;
          }}

          const outLines = [];
          const splitGapX = Math.max(joinGapX * 1.6, Math.max(24, spanWidth * 0.04));
          for (const line of lines) {{
            const items = Array.isArray(line.items) ? line.items.slice() : [];
            items.sort((a, b) => a.x - b.x);
            let bestSplit = -1;
            let bestGap = 0;
            for (let i = 0; i < items.length - 1; i++) {{
              const a0 = items[i];
              const b0 = items[i + 1];
              const gap = (b0.x - (a0.x + (a0.w || 0)));
              if (gap > bestGap) {{
                bestGap = gap;
                bestSplit = i;
              }}
            }}
            if (bestSplit >= 0 && bestGap >= splitGapX) {{
              const leftItems = items.slice(0, bestSplit + 1);
              const rightItems = items.slice(bestSplit + 1);
              outLines.push(_annolidLineFromItems(leftItems));
              outLines.push(_annolidLineFromItems(rightItems));
              continue;
            }}
            const out = _annolidLineFromItems(items);
            outLines.push(out);
          }}

          const outHeights = outLines
            .map((l) => (l && isFinite(l.h)) ? l.h : 0)
            .filter((h) => isFinite(h) && h > 0);
          const outMedianH = _annolidMedian(outHeights);
          const sameRowTol = Math.max(2, (outMedianH || typicalH || 0) * 0.35);
          outLines.sort((a, b) => {{
            const ay = (a && isFinite(a.yCenter)) ? a.yCenter : ((a && isFinite(a.yMin) && isFinite(a.yMax)) ? (a.yMin + a.yMax) * 0.5 : 0);
            const by = (b && isFinite(b.yCenter)) ? b.yCenter : ((b && isFinite(b.yMin) && isFinite(b.yMax)) ? (b.yMin + b.yMax) * 0.5 : 0);
            const dy = ay - by;
            if (Math.abs(dy) <= sameRowTol) {{
              const ax = (a && isFinite(a.xMin)) ? a.xMin : 0;
              const bx = (b && isFinite(b.xMin)) ? b.xMin : 0;
              if (Math.abs(ax - bx) > 0.5) return ax - bx;
              return dy;
            }}
            return dy;
          }});
          return outLines;
        }}

        function _annolidLinesToParagraphs(lines, pageNum) {{
          const paragraphs = [];
          let current = null;
          lines.forEach((line) => {{
            const lineText = _annolidNormalizeText(line.texts.join(" "));
            if (!lineText) return;
            if (!current) {{
              current = {{
                pageNum,
                text: lineText,
                spans: [].concat(line.spans),
                yMin: line.yMin,
                yMax: line.yMax,
                xMin: line.xMin,
                xMax: line.xMax,
              }};
              return;
            }}
            const gap = line.yMin - current.yMax;
            const gapLimit = Math.max(8, (current.yMax - current.yMin) * 0.9);
            if (gap > gapLimit) {{
              paragraphs.push(current);
              current = {{
                pageNum,
                text: lineText,
                spans: [].concat(line.spans),
                yMin: line.yMin,
                yMax: line.yMax,
                xMin: line.xMin,
                xMax: line.xMax,
              }};
            }} else {{
              current.text = _annolidMergeHyphenated(current.text, lineText);
              current.spans = current.spans.concat(line.spans);
              current.yMax = Math.max(current.yMax, line.yMax);
              current.xMin = Math.min(current.xMin, line.xMin);
              current.xMax = Math.max(current.xMax, line.xMax);
            }}
          }});
          if (current) paragraphs.push(current);
          return paragraphs;
        }}

        function _annolidBuildParagraphsForPage(pageNum) {{
          const state = _annolidGetPageState(pageNum);
          if (!state || !state.pageDiv) return [];
          const pageDiv = state.pageDiv;
          const spans = Array.from(pageDiv.querySelectorAll(".textLayer span"));
          if (!spans.length) return [];
          const pageRect = pageDiv.getBoundingClientRect();
          const entries = [];
          spans.forEach((span) => {{
            const text = _annolidNormalizeText(span.textContent || "");
            if (!text) return;
            const rect = span.getBoundingClientRect();
            const idx = parseInt(span.dataset.annolidIndex || "-1", 10);
            entries.push({{
              idx,
              text,
              x: rect.left - pageRect.left,
              y: rect.top - pageRect.top,
              w: rect.width,
              h: rect.height,
            }});
          }});
          if (!entries.length) return [];
          const pageWidth = Math.max(1, pageRect.width || 1);
          // Build lines first, then detect columns using line starts (xMin).
          // Detecting columns from *per-span* x positions can wrongly split within a line,
          // which is a common root cause of "skipping" right after punctuation like "。”.
          const rawLines = _annolidBuildLinesFromEntries(entries);
          const colLines = _annolidGroupLinesIntoColumns(rawLines, pageWidth);
          // Ensure columns are ordered left->right (median xMin).
          colLines.sort((a, b) => {{
            const ax = _annolidMedian((a || []).map((l) => l.xMin).filter((x) => isFinite(x)));
            const bx = _annolidMedian((b || []).map((l) => l.xMin).filter((x) => isFinite(x)));
            return ax - bx;
          }});

          const paragraphs = [];
          colLines.forEach((lines) => {{
            const colParas = _annolidLinesToParagraphs(lines, pageNum);
            colParas.forEach((p) => paragraphs.push(p));
          }});

          window.__annolidParagraphsByPage[String(pageNum)] = paragraphs;
          return paragraphs;
        }}

        function _annolidRebuildParagraphIndex() {{
          const totalPages = window.__annolidTotalPages || 0;
          const paragraphs = [];
          const offsets = {{}};
          let offset = 0;
          for (let p = 1; p <= totalPages; p++) {{
            offsets[p] = offset;
            const pageList = window.__annolidParagraphsByPage[String(p)] || [];
            pageList.forEach((para) => paragraphs.push(para));
            offset += pageList.length;
          }}
          window.__annolidParagraphs = paragraphs;
          window.__annolidParagraphOffsets = offsets;
          window.__annolidParagraphTotal = offset;
        }}

        function _annolidFindParagraphIndexBySpan(pageNum, spanIdx) {{
          const list = window.__annolidParagraphsByPage[String(pageNum)] || [];
          for (let i = 0; i < list.length; i++) {{
            const spans = list[i].spans || [];
            if (spans.indexOf(spanIdx) >= 0) return i;
          }}
          return -1;
        }}

        function _annolidFindParagraphIndexByPoint(pageNum, x, y) {{
          const list = window.__annolidParagraphsByPage[String(pageNum)] || [];
          let best = -1;
          let bestDist = Infinity;
          for (let i = 0; i < list.length; i++) {{
            const para = list[i];
            if (para.yMin == null || para.yMax == null) continue;
            const xMin = (para.xMin == null) ? -Infinity : para.xMin;
            const xMax = (para.xMax == null) ? Infinity : para.xMax;
            const yMin = para.yMin;
            const yMax = para.yMax;
            const dx = (x < xMin) ? (xMin - x) : ((x > xMax) ? (x - xMax) : 0);
            const dy = (y < yMin) ? (yMin - y) : ((y > yMax) ? (y - yMax) : 0);
            if (dx === 0 && dy === 0) return i;
            const dist = dx + dy;
            if (dist < bestDist) {{
              best = i;
              bestDist = dist;
            }}
          }}
          return best;
        }}

	        async function _annolidBuildTextParagraphsForPage(pageNum) {{
	          if (!window.__annolidPdf) return;
	          if (window.__annolidParagraphsByPage[String(pageNum)]) return;
	          const page = await window.__annolidPdf.getPage(pageNum);
	          const textContent = await page.getTextContent();
	          const viewport1 = page.getViewport({{ scale: 1, rotation: 0 }});
	          const linesRaw = _annolidExtractLinesFromTextContent(textContent);
	          const ordered = _annolidOrderLinesForReading(linesRaw, viewport1.width || 1);
	          const paragraphs = [];
	          let current = null;
	          let lastYMin = null;
	          let lastYMax = null;
	          for (const line of ordered) {{
	            const lineText = _annolidNormalizeText(line.text || "");
	            if (!lineText) continue;
	            if (!current) {{
	              current = {{
	                pageNum,
	                text: lineText,
	                spans: [],
	                yMin: line.yMin,
	                yMax: line.yMax,
	                xMin: line.xMin,
	                xMax: line.xMax,
	              }};
	              lastYMin = line.yMin;
	              lastYMax = line.yMax;
	              continue;
	            }}
	            const gap = (lastYMin != null) ? (lastYMin - line.yMax) : 0;
	            const lastH = (lastYMax != null && lastYMin != null) ? Math.abs(lastYMax - lastYMin) : 0;
	            const gapLimit = Math.max(4, lastH * 1.15);
	            if (gap > gapLimit) {{
	              paragraphs.push(current);
	              current = {{
	                pageNum,
	                text: lineText,
	                spans: [],
	                yMin: line.yMin,
	                yMax: line.yMax,
	                xMin: line.xMin,
	                xMax: line.xMax,
	              }};
	            }} else {{
	              current.text = _annolidMergeHyphenated(current.text, lineText);
	              current.yMin = Math.min(current.yMin, line.yMin);
	              current.yMax = Math.max(current.yMax, line.yMax);
	              current.xMin = Math.min(current.xMin, line.xMin);
	              current.xMax = Math.max(current.xMax, line.xMax);
	            }}
	            lastYMin = line.yMin;
	            lastYMax = line.yMax;
	          }}
	          if (current) paragraphs.push(current);
	          window.__annolidParagraphsByPage[String(pageNum)] = paragraphs;
	        }}

        async function _annolidEnsureParagraphsFrom(pageNum) {{
          const totalPages = window.__annolidTotalPages || 0;
          for (let p = pageNum; p <= totalPages; p++) {{
            if (!window.__annolidParagraphsByPage[String(p)]) {{
              await _annolidBuildTextParagraphsForPage(p);
              await new Promise(r => setTimeout(r, 0));
            }}
          }}
          _annolidRebuildParagraphIndex();
        }}

        document.addEventListener("selectionchange", () => {{
          try {{
            const sel = window.getSelection ? window.getSelection() : null;
            window.__annolidSelection = sel ? sel.toString() : "";
            window.__annolidSelectionSpans = [];
            if (sel && !sel.isCollapsed && window.__annolidSpans.length) {{
              const ranges = [];
              for (let i = 0; i < sel.rangeCount; i++) {{
                ranges.push(sel.getRangeAt(i));
              }}
              window.__annolidSpans.forEach((span, idx) => {{
                for (const range of ranges) {{
                  if (range.intersectsNode(span)) {{
                    window.__annolidSelectionSpans.push(idx);
                    break;
                  }}
                }}
              }});
            }}
          }} catch (e) {{
            window.__annolidSelection = "";
            window.__annolidSelectionSpans = [];
          }}
        }});
        let loadingTask = null;
        if (pdfBase64 && pdfBase64.length > 0) {{
          const raw = atob(pdfBase64);
          const bytes = new Uint8Array(raw.length);
          for (let i = 0; i < raw.length; i++) {{
            bytes[i] = raw.charCodeAt(i) & 0xff;
          }}
          loadingTask = pdfjsLib.getDocument({{
            data: bytes,
          }});
        }} else {{
          loadingTask = pdfjsLib.getDocument({{
            url: pdfUrl,
          }});
        }}
	        const pdf = await loadingTask.promise;
	        window.__annolidPdf = pdf;
	        window.__annolidPdfLoaded = true;
	        const container = document.getElementById("viewerContainer");
	        const MIN_SCALE = 0.25;
	        const MAX_SCALE = 4.0;
	        const DEFAULT_SCALE = 1.25;
	        let scale = DEFAULT_SCALE;
	        let rotation = 0;
	        let nextPage = 1;
	        let renderEpoch = 0;
	        let renderChain = Promise.resolve();
	        let zoomBusy = false;
	        let pendingZoom = null;
	        const total = pdf.numPages || 1;
	        window.__annolidTotalPages = total;
        let pdfObjectUrl = null;
        let pdfObjectUrlPromise = null;
	
	        const titleEl = document.getElementById("annolidTitle");
	        const prevPageBtn = document.getElementById("annolidPrevPage");
	        const nextPageBtn = document.getElementById("annolidNextPage");
	        const pageInput = document.getElementById("annolidPageInput");
	        const totalPagesEl = document.getElementById("annolidTotalPages");
	        const zoomOutBtn = document.getElementById("annolidZoomOut");
	        const zoomInBtn = document.getElementById("annolidZoomIn");
	        const zoomResetBtn = document.getElementById("annolidZoomReset");
	        const zoomFitBtn = document.getElementById("annolidZoomFit");
        const rotateBtn = document.getElementById("annolidRotate");
        const zoomLabel = document.getElementById("annolidZoomLabel");
        const printBtn = document.getElementById("annolidPrint");
        const menuBtn = document.getElementById("annolidMenuBtn");
        const menuPanel = document.getElementById("annolidMenuPanel");
        const previewModal = document.getElementById("annolidPreviewModal");
        const previewCloseBtn = document.getElementById("annolidPreviewClose");
        const previewZoomOutBtn = document.getElementById("annolidPreviewZoomOut");
        const previewZoomInBtn = document.getElementById("annolidPreviewZoomIn");
        const previewZoomResetBtn = document.getElementById("annolidPreviewZoomReset");
        const previewTitleEl = document.getElementById("annolidPreviewTitle");
        const previewTextEl = document.getElementById("annolidPreviewText");
        const previewBody = document.getElementById("annolidPreviewBody");
        const previewCanvas = document.getElementById("annolidPreviewCanvas");
        const previewWrap = document.getElementById("annolidPreviewCanvasWrap");
        const previewGrid = document.getElementById("annolidPreviewGrid");
        const previewHighlight = document.getElementById("annolidPreviewHighlight");
        const previewCtx = previewCanvas ? previewCanvas.getContext("2d") : null;
        const citePopover = document.getElementById("annolidCitePopover");
        const citeTitleEl = document.getElementById("annolidCiteTitle");
        const citeBodyEl = document.getElementById("annolidCiteBody");
        const previewState = {{
          open: false,
          scale: 2.0,
          pageNum: 1,
          info: null,
          title: "",
          mode: "",
          citation: null,
          autoclose: false,
        }};
        const citePopoverState = {{
          open: false,
          number: null,
          anchor: null,
          autoclose: true,
        }};
        let citeHoverTimer = null;
        let citeCloseTimer = null;
        let previewCloseTimer = null;
	
	        if (titleEl) titleEl.textContent = pdfTitle || "PDF";
	        if (totalPagesEl) totalPagesEl.textContent = String(total);
	        if (pageInput) pageInput.setAttribute("max", String(total));
	
	        function _annolidClampScale(value) {{
	          const v = parseFloat(value);
	          if (!isFinite(v)) return DEFAULT_SCALE;
	          return Math.max(MIN_SCALE, Math.min(MAX_SCALE, v));
	        }}

	        function _annolidNormalizeRotation(value) {{
	          const v = parseInt(value, 10);
	          const norm = isFinite(v) ? ((v % 360) + 360) % 360 : 0;
	          const steps = [0, 90, 180, 270];
	          let closest = 0;
	          let bestDiff = Infinity;
	          steps.forEach((step) => {{
	            const diff = Math.abs(step - norm);
	            if (diff < bestDiff) {{
	              bestDiff = diff;
	              closest = step;
	            }}
	          }});
	          return closest;
	        }}
	
	        function _annolidUpdateZoomLabel() {{
	          if (!zoomLabel) return;
	          const pct = Math.round(_annolidClampScale(scale) * 100);
	          zoomLabel.textContent = String(pct) + "%";
	        }}

        function _annolidCleanupObjectUrl() {{
          if (pdfObjectUrl) {{
            try {{ URL.revokeObjectURL(pdfObjectUrl); }} catch (e) {{}}
          }}
          pdfObjectUrl = null;
          pdfObjectUrlPromise = null;
        }}

        async function _annolidGetPdfObjectUrl() {{
          if (pdfObjectUrl) return pdfObjectUrl;
          if (!pdfObjectUrlPromise) {{
            if (pdf && typeof pdf.getData === "function") {{
              pdfObjectUrlPromise = pdf.getData().then((data) => {{
                const blob = new Blob([data], {{ type: "application/pdf" }});
                pdfObjectUrl = URL.createObjectURL(blob);
                return pdfObjectUrl;
              }}).catch(() => {{
                pdfObjectUrlPromise = null;
                return pdfUrl;
              }});
            }} else {{
              pdfObjectUrlPromise = Promise.resolve(pdfUrl);
            }}
          }}
          const url = await pdfObjectUrlPromise;
          if (url) pdfObjectUrl = url;
          return url || pdfUrl;
        }}

        function _annolidCloseMenu() {{
          if (menuPanel) menuPanel.classList.remove("annolid-open");
        }}
        function _annolidToggleMenu() {{
          if (!menuPanel) return;
          const open = menuPanel.classList.contains("annolid-open");
          if (open) menuPanel.classList.remove("annolid-open");
          else {{
            menuPanel.classList.add("annolid-open");
          }}
        }}
	
        if (menuBtn) menuBtn.addEventListener("click", (ev) => {{
          ev.stopPropagation();
          _annolidToggleMenu();
        }});
        if (menuPanel) menuPanel.addEventListener("click", (ev) => {{
          ev.stopPropagation();
          const target = ev.target;
          const action = target && target.dataset ? target.dataset.action : "";
          if (!action) return;
          _annolidCloseMenu();
          switch (action) {{
            case "first-page":
              _annolidGoToPage(1);
              break;
            case "fit":
              _annolidZoomFitWidth();
              break;
            case "reset":
              _annolidRerenderAll(1.0);
              break;
            case "print":
              _annolidPrintPdf().catch(() => {{}});
              break;
            default:
              break;
          }}
        }});
        document.addEventListener("keydown", (ev) => {{
          if (ev.key === "Escape") {{
            _annolidCloseMenu();
          }}
        }});
	        document.addEventListener("click", () => {{
            _annolidCloseMenu();
          }});
        window.addEventListener("beforeunload", _annolidCleanupObjectUrl);

        function _annolidOpenPreviewModal(title) {{
          if (!previewModal) return;
          if (previewCloseTimer) {{
            clearTimeout(previewCloseTimer);
            previewCloseTimer = null;
          }}
          previewState.open = true;
          previewState.title = String(title || "Preview");
          if (previewTitleEl) previewTitleEl.textContent = previewState.title;
          _annolidSetPreviewLayout(previewState.mode || "");
          previewModal.classList.add("annolid-open");
        }}

        function _annolidClosePreviewModal() {{
          if (!previewModal) return;
          previewState.open = false;
          previewModal.classList.remove("annolid-open");
          previewState.info = null;
          previewState.mode = "";
          previewState.citation = null;
          previewState.autoclose = false;
          if (previewHighlight) previewHighlight.style.display = "none";
          _annolidSetPreviewLayout("");
        }}

        function _annolidEscapeHtml(text) {{
          return String(text || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\"/g, "&quot;")
            .replace(/'/g, "&#39;");
        }}

        function _annolidSetPreviewText(text) {{
          if (!previewTextEl) return;
          const t = String(text || "").trim();
          if (!t) {{
            previewTextEl.innerHTML = '<span class="annolid-muted">No preview available.</span>';
            return;
          }}
          previewTextEl.textContent = t;
        }}

        function _annolidSetPreviewMessage(message) {{
          if (!previewTextEl) return;
          previewTextEl.innerHTML = '<span class="annolid-muted">' + _annolidEscapeHtml(message) + "</span>";
        }}

        function _annolidSetPreviewLayout(mode) {{
          const isCitation = mode === "citation";
          if (previewWrap) previewWrap.style.display = isCitation ? "none" : "";
          if (previewGrid) {{
            previewGrid.classList.toggle("annolid-preview-citation", isCitation);
          }}
          if (previewHighlight) previewHighlight.style.display = "none";
        }}

        function _annolidShowCitationPopover(number, text, anchorEl) {{
          if (!citePopover || !citeBodyEl) return;
          if (citeCloseTimer) {{
            clearTimeout(citeCloseTimer);
            citeCloseTimer = null;
          }}
          citePopoverState.open = true;
          citePopoverState.number = number;
          citePopoverState.anchor = anchorEl || null;
          if (citeTitleEl) {{
            citeTitleEl.textContent = `Reference [${{number}}]`;
          }}
          citeBodyEl.textContent = String(text || "").trim() || "No preview available.";
          citePopover.classList.add("annolid-open");
          _annolidPositionCitePopover(anchorEl);
        }}

        function _annolidHideCitationPopover() {{
          if (!citePopover) return;
          citePopoverState.open = false;
          citePopoverState.number = null;
          citePopoverState.anchor = null;
          citePopover.classList.remove("annolid-open");
        }}

        function _annolidPositionCitePopover(anchorEl) {{
          if (!citePopover || !anchorEl) return;
          if (!document.body.contains(anchorEl)) {{
            _annolidHideCitationPopover();
            return;
          }}
          const rect = anchorEl.getBoundingClientRect();
          citePopover.style.visibility = "hidden";
          citePopover.classList.add("annolid-open");
          const popRect = citePopover.getBoundingClientRect();
          const margin = 8;
          const viewportW = window.innerWidth || document.documentElement.clientWidth || 1;
          const viewportH = window.innerHeight || document.documentElement.clientHeight || 1;
          let left = rect.left + rect.width / 2 - popRect.width / 2;
          left = Math.max(margin, Math.min(left, viewportW - popRect.width - margin));
          let top = rect.bottom + margin;
          if (top + popRect.height > viewportH - margin) {{
            top = rect.top - popRect.height - margin;
          }}
          if (top < margin) {{
            top = Math.min(viewportH - popRect.height - margin, rect.bottom + margin);
          }}
          citePopover.style.left = Math.round(left) + "px";
          citePopover.style.top = Math.round(top) + "px";
          citePopover.style.visibility = "visible";
        }}

        function _annolidUpdateCitePopoverPosition() {{
          if (!citePopoverState.open) return;
          _annolidPositionCitePopover(citePopoverState.anchor);
        }}

	        const referenceIndex = {{
	          built: false,
	          building: false,
	          promise: null,
	          byNumber: {{}},
	          startPage: null,
	        }};

	        function _annolidSimplifyAlpha(text) {{
	          return String(text || "").toLowerCase().replace(/[^a-z]/g, "");
	        }}

	        function _annolidIsReferencesHeading(text) {{
	          const t = _annolidSimplifyAlpha(text);
	          return (
	            t.startsWith("references") ||
	            t.startsWith("bibliography") ||
	            t.startsWith("literaturecited")
	          );
	        }}

	        function _annolidParseReferenceStart(text) {{
	          const t = String(text || "").trim();
	          if (!t) return null;
	          let m = /^\\[\\s*(\\d{{1,4}})\\s*\\]/.exec(t);
	          if (m) return parseInt(m[1], 10);
	          m = /^\\(\\s*(\\d{{1,4}})\\s*\\)/.exec(t);
	          if (m) return parseInt(m[1], 10);
	          m = /^(\\d{{1,4}})\\s*[\\.)]\\s*/.exec(t);
	          if (m) {{
	            const n = parseInt(m[1], 10);
	            if (n >= 1 && n <= 999) return n;
	          }}
	          m = /^(\\d{{1,4}})\\s+(?=[A-Za-z])/.exec(t);
	          if (m) {{
	            const n = parseInt(m[1], 10);
	            if (n >= 1 && n <= 999) return n;
	          }}
	          return null;
	        }}

        function _annolidParseCitationNumber(text) {{
          const raw = String(text || "").replace(/\\s+/g, "");
          if (!raw) return null;
          let m = /^\\[(\\d{{1,4}})\\]$/.exec(raw);
          if (m) return parseInt(m[1], 10);
          m = /^\\[(\\d{{1,4}})[,;\\]]/.exec(raw);
          if (m) return parseInt(m[1], 10);
          m = /^\\((\\d{{1,4}})\\)$/.exec(raw);
          if (m) return parseInt(m[1], 10);
          return null;
        }}

        function _annolidGetSpanNeighborText(span, dir) {{
          try {{
            const sib = (dir < 0) ? span.previousElementSibling : span.nextElementSibling;
            if (!sib || sib.tagName !== "SPAN") return "";
            return String(sib.textContent || "").trim();
          }} catch (e) {{
            return "";
          }}
        }}

        function _annolidGetLineSpansForSpan(span) {{
          if (!span) return [];
          const layer = span.closest ? span.closest(".textLayer") : null;
          if (!layer) return [span];
          let targetRect = null;
          try {{ targetRect = span.getBoundingClientRect(); }} catch (e) {{ targetRect = null; }}
          if (!targetRect) return [span];
          const targetY = targetRect.top + targetRect.height * 0.5;
          const tol = Math.max(2, targetRect.height * 0.6);
          const spans = Array.from(layer.querySelectorAll("span"));
          return spans.filter((s) => {{
            try {{
              const r = s.getBoundingClientRect();
              const y = r.top + r.height * 0.5;
              return Math.abs(y - targetY) <= tol;
            }} catch (e) {{
              return false;
            }}
          }});
        }}

        function _annolidBuildLineText(spans) {{
          const items = [];
          spans.forEach((s) => {{
            try {{
              const text = String(s.textContent || "");
              if (!text) return;
              const r = s.getBoundingClientRect();
              items.push({{ span: s, rect: r, text }});
            }} catch (e) {{}}
          }});
          if (!items.length) return {{ text: "", ranges: new Map() }};
          items.sort((a, b) => a.rect.left - b.rect.left);
          let text = "";
          let cursor = 0;
          let prevRight = null;
          const ranges = new Map();
          items.forEach((item) => {{
            if (!item.text) return;
            if (text && prevRight != null) {{
              const gap = item.rect.left - prevRight;
              if (gap > 3) {{
                text += " ";
                cursor += 1;
              }}
            }}
            const start = cursor;
            text += item.text;
            cursor += item.text.length;
            ranges.set(item.span, {{ start, end: cursor }});
            prevRight = item.rect.right;
          }});
          return {{ text, ranges }};
        }}

        function _annolidExtractCitationMatches(text) {{
          const out = [];
          if (!text) return out;
          const re = /[\\[(]\\s*\\d{{1,4}}(?:\\s*[-–—]\\s*\\d{{1,4}})?(?:\\s*[,;]\\s*\\d{{1,4}}(?:\\s*[-–—]\\s*\\d{{1,4}})?)*\\s*[\\])]/g;
          let m;
          while ((m = re.exec(text)) !== null) {{
            out.push({{ text: m[0], start: m.index, end: m.index + m[0].length }});
          }}
          return out;
        }}

        function _annolidFindCitationNumberFromSpan(span) {{
          if (!span) return null;
          const spanDigits = (String(span.textContent || "").match(/\\d{{1,4}}/) || [])[0] || null;
          const lineSpans = _annolidGetLineSpansForSpan(span);
          const lineInfo = _annolidBuildLineText(lineSpans);
          const range = lineInfo.ranges.get(span);
          if (lineInfo.text && range) {{
            const matches = _annolidExtractCitationMatches(lineInfo.text);
            for (const match of matches) {{
              if (range.start < match.end && range.end > match.start) {{
                const nums = String(match.text || "").match(/\\d{{1,4}}/g) || [];
                if (!nums.length) return null;
                if (spanDigits && nums.includes(spanDigits)) return parseInt(spanDigits, 10);
                return parseInt(nums[0], 10);
              }}
            }}
          }}
          // Fallback to nearby spans when line reconstruction fails.
          const t0 = String(span.textContent || "").trim();
          const n0 = _annolidParseCitationNumber(t0);
          if (n0) return n0;
          const prev = _annolidGetSpanNeighborText(span, -1);
          const next = _annolidGetSpanNeighborText(span, 1);
          const combo = (prev || "") + (t0 || "") + (next || "");
          const n1 = _annolidParseCitationNumber(combo);
          if (n1) return n1;
          return spanDigits ? parseInt(spanDigits, 10) : null;
        }}

        function _annolidFindUnderlyingTextSpan(linkEl) {{
          try {{
            const r = linkEl.getBoundingClientRect();
            const x = r.left + Math.max(1, r.width) / 2;
            const y = r.top + Math.max(1, r.height) / 2;
            const els = document.elementsFromPoint ? document.elementsFromPoint(x, y) : [];
            for (const el of (els || [])) {{
              if (!el || !el.tagName) continue;
              if (el.tagName === "SPAN" && el.closest && el.closest(".textLayer")) {{
                return el;
              }}
            }}
          }} catch (e) {{}}
          return null;
        }}

        function _annolidExtractLinesFromTextContent(textContent) {{
          const items = (textContent && textContent.items) ? textContent.items : [];
          const rows = [];
          for (const it of items) {{
            try {{
              const str = _annolidNormalizeText(it.str || "");
              if (!str) continue;
              const tr = it.transform || [];
              const x = isFinite(tr[4]) ? tr[4] : 0;
              const y = isFinite(tr[5]) ? tr[5] : 0;
              const w = isFinite(it.width) ? it.width : 0;
              const h = isFinite(it.height) ? Math.abs(it.height) : 0;
              rows.push({{ str, x, y, w, h }});
            }} catch (e) {{}}
          }}
          if (!rows.length) return [];
          const typicalH = _annolidMedian(rows.map((r) => r.h).filter((h) => isFinite(h) && h > 0));
          const yTol = Math.max(1.5, (typicalH || 0) * 0.85);
          rows.forEach((r) => {{
            r.yCenter = r.y + (isFinite(r.h) ? r.h * 0.5 : 0);
          }});
          rows.sort((a, b) => {{
            const dy = b.yCenter - a.yCenter;
            if (Math.abs(dy) > 0.01) return dy;
            return a.x - b.x;
          }});

          const lines = [];
          for (const r of rows) {{
            const y = r.yCenter;
            let line = lines.length ? lines[lines.length - 1] : null;
            if (line && Math.abs(y - line.yCenter) > yTol) {{
              const prev = (lines.length >= 2) ? lines[lines.length - 2] : null;
              if (prev && Math.abs(y - prev.yCenter) <= yTol) {{
                line = prev;
              }} else {{
                line = null;
              }}
            }}
            if (!line || Math.abs(y - line.yCenter) > yTol) {{
              line = {{
                yCenter: y,
                y: r.y,
                items: [],
                xMin: r.x,
                xMax: r.x + r.w,
                yMin: r.y,
                yMax: r.y + r.h,
                text: "",
              }};
              lines.push(line);
            }}
            line.items.push(r);
            const n = line.items.length;
            line.yCenter = (line.yCenter * (n - 1) + y) / n;
            line.xMin = Math.min(line.xMin, r.x);
            line.xMax = Math.max(line.xMax, r.x + r.w);
            line.yMin = Math.min(line.yMin, r.y);
            line.yMax = Math.max(line.yMax, r.y + r.h);
          }}

          for (const line of lines) {{
            const items = Array.isArray(line.items) ? line.items : [];
            items.sort((a, b) => a.x - b.x);
            const parts = [];
            let lastX = null;
            for (const it of items) {{
              if (lastX != null && it.x - lastX > 8) parts.push(" ");
              parts.push(it.str);
              lastX = it.x + (it.w || 0);
            }}
            line.text = _annolidNormalizeText(parts.join(" "));
            line.y = line.yCenter;
            delete line.items;
          }}
          return lines.filter((l) => l.text && l.text.length);
        }}

	        function _annolidOrderLinesForReading(lines, pageWidth) {{
	          const pw = isFinite(pageWidth) ? Math.max(1, pageWidth) : 1;
	          const list = Array.isArray(lines)
	            ? lines.filter((l) => l && isFinite(l.xMin) && isFinite(l.xMax) && isFinite(l.y))
	            : [];
	          if (!list.length) return Array.isArray(lines) ? lines.slice() : [];

	          const xCenterOf = (l) => (l.xMin + l.xMax) * 0.5;
	          const xs = list.map((l) => xCenterOf(l)).filter((x) => isFinite(x));
	          if (xs.length < 8) {{
	            return list.slice().sort((a, b) => b.y - a.y);
	          }}
	          xs.sort((a, b) => a - b);
	          const q = (arr, t) => arr[Math.max(0, Math.min(arr.length - 1, Math.round(t * (arr.length - 1))))];
	          const mean = (arr) => {{
	            if (!arr.length) return 0;
	            let s = 0;
	            for (let i = 0; i < arr.length; i++) s += arr[i];
	            return s / arr.length;
	          }};
	          const std = (arr, m) => {{
	            if (!arr.length) return 0;
	            let s = 0;
	            for (let i = 0; i < arr.length; i++) {{
	              const d = arr[i] - m;
	              s += d * d;
	            }}
	            return Math.sqrt(s / arr.length);
	          }};

	          let m1 = q(xs, 0.25);
	          let m2 = q(xs, 0.75);
	          if (Math.abs(m2 - m1) < pw * 0.10) {{
	            return list.slice().sort((a, b) => b.y - a.y);
	          }}
	          let a = [];
	          let b = [];
	          for (let iter = 0; iter < 10; iter++) {{
	            a = [];
	            b = [];
	            for (let i = 0; i < xs.length; i++) {{
	              const x = xs[i];
	              if (Math.abs(x - m1) <= Math.abs(x - m2)) a.push(x);
	              else b.push(x);
	            }}
	            const nm1 = mean(a);
	            const nm2 = mean(b);
	            if (Math.abs(nm1 - m1) < 0.5 && Math.abs(nm2 - m2) < 0.5) break;
	            m1 = nm1;
	            m2 = nm2;
	          }}
	          if (!a.length || !b.length) {{
	            return list.slice().sort((a0, b0) => b0.y - a0.y);
	          }}
	          const leftMean = Math.min(m1, m2);
	          const rightMean = Math.max(m1, m2);
	          const separation = rightMean - leftMean;
	          const minSeparation = Math.max(36, pw * 0.16);
	          if (separation < minSeparation) {{
	            return list.slice().sort((a0, b0) => b0.y - a0.y);
	          }}
	          const score = separation / Math.max(1e-6, std(a, m1) + std(b, m2));
	          if (score < 1.15) {{
	            return list.slice().sort((a0, b0) => b0.y - a0.y);
	          }}

	          const minCount = Math.max(3, Math.floor(xs.length * 0.18));
	          const left = [];
	          const right = [];
	          for (const l of list) {{
	            const x = xCenterOf(l);
	            if (!isFinite(x)) continue;
	            if (Math.abs(x - leftMean) <= Math.abs(x - rightMean)) left.push(l);
	            else right.push(l);
	          }}
	          if (left.length < minCount || right.length < minCount) {{
	            return list.slice().sort((a0, b0) => b0.y - a0.y);
	          }}
	          left.sort((a0, b0) => b0.y - a0.y);
	          right.sort((a0, b0) => b0.y - a0.y);
	          return left.concat(right);
	        }}

	        async function _annolidFindReferencesStartPage() {{
	          const maxScan = Math.min(total, 120);
	          const minP = Math.max(1, total - maxScan + 1);
	          let inRefsBlock = false;
	          let earliestRefsPage = null;
	          for (let p = total; p >= minP; p--) {{
	            try {{
	              const page = await pdf.getPage(p);
	              const tc = await page.getTextContent();
	              const lines = _annolidExtractLinesFromTextContent(tc);
	              const anyHeading = lines.some((l) => _annolidIsReferencesHeading((l.text || "").trim()));
	              if (anyHeading) return p;
	              let starts = 0;
	              for (const line of lines) {{
	                if (_annolidParseReferenceStart((line.text || "").trim()) != null) {{
	                  starts += 1;
	                  if (starts >= 2) break;
	                }}
	              }}
	              if (starts >= 2) {{
	                inRefsBlock = true;
	                earliestRefsPage = p;
	                continue;
	              }}
	              if (inRefsBlock) {{
	                break;
	              }}
	            }} catch (e) {{}}
	          }}
	          if (earliestRefsPage != null) return earliestRefsPage;
	          return Math.max(1, total - 8);
	        }}

	        async function _annolidBuildReferenceIndex() {{
	          if (referenceIndex.built || referenceIndex.building) return referenceIndex.promise;
	          referenceIndex.building = true;
          referenceIndex.promise = (async () => {{
            const start = await _annolidFindReferencesStartPage();
            referenceIndex.startPage = start;
	            const byNum = {{}};
	            let started = false;
	            let current = null;
	            for (let p = start; p <= total; p++) {{
              let page = null;
              try {{
                page = await pdf.getPage(p);
              }} catch (e) {{
                continue;
              }}
              let tc = null;
              try {{
                tc = await page.getTextContent();
              }} catch (e) {{
                continue;
              }}
	              const viewport1 = page.getViewport({{ scale: 1, rotation: 0 }});
	              const linesRaw = _annolidExtractLinesFromTextContent(tc);
	              const lines = _annolidOrderLinesForReading(linesRaw, viewport1.width || 1);
	              for (const line of lines) {{
	                const text = (line.text || "").trim();
	                if (!text) continue;
	                if (!started) {{
	                  if (_annolidIsReferencesHeading(text)) {{
	                    started = true;
	                    continue;
	                  }}
	                  const firstN = _annolidParseReferenceStart(text);
	                  if (firstN == null) {{
	                    continue;
	                  }}
	                  started = true;
	                }}
	                const n = _annolidParseReferenceStart(text);
	                if (n != null && isFinite(n)) {{
	                  if (current && current.num != null) {{
	                    byNum[String(current.num)] = current;
                  }}
                  const rect = [line.xMin, line.yMin, line.xMax, line.yMax];
                  current = {{
                    num: n,
                    pageNum: p,
                    text: text,
                    rect,
                  }};
                  continue;
                }}
                if (current) {{
                  current.text = _annolidNormalizeText(current.text + " " + text);
                  const r = current.rect || [0, 0, 0, 0];
                  current.rect = [
                    Math.min(r[0], line.xMin),
                    Math.min(r[1], line.yMin),
                    Math.max(r[2], line.xMax),
                    Math.max(r[3], line.yMax),
                  ];
                }}
              }}
            }}
            if (current && current.num != null) {{
              byNum[String(current.num)] = current;
            }}
            referenceIndex.byNumber = byNum;
            referenceIndex.built = true;
            referenceIndex.building = false;
            return byNum;
          }})().catch(() => {{
            referenceIndex.byNumber = {{}};
            referenceIndex.built = true;
            referenceIndex.building = false;
            return referenceIndex.byNumber;
          }});
          return referenceIndex.promise;
        }}

        async function _annolidGetReference(number) {{
          const key = String(parseInt(number, 10) || "");
          if (!key) return null;
          if (!referenceIndex.built) {{
            await _annolidBuildReferenceIndex();
          }}
          return referenceIndex.byNumber[key] || null;
        }}

        async function _annolidOpenCitationPreview(number, fallbackDest, autoclose = true, anchorEl = null) {{
          const n = parseInt(number, 10);
          if (!n) return false;
          if (citePopoverState.open && citePopoverState.number === n) {{
            _annolidPositionCitePopover(anchorEl || citePopoverState.anchor);
            return true;
          }}
          citePopoverState.autoclose = !!autoclose;
          _annolidShowCitationPopover(n, `Loading reference [${{n}}]…`, anchorEl);
          const ref = await _annolidGetReference(n);
          if (!ref) {{
            _annolidShowCitationPopover(
              n,
              `Reference [${{n}}] not found in this PDF.`,
              anchorEl || citePopoverState.anchor
            );
            return false;
          }}
          _annolidShowCitationPopover(
            n,
            ref.text || "",
            anchorEl || citePopoverState.anchor
          );
          return true;
        }}

        async function _annolidResolveDestination(dest) {{
          try {{
            let resolved = dest;
            if (typeof resolved === "string") {{
              resolved = await pdf.getDestination(resolved);
            }}
            if (!resolved || !Array.isArray(resolved) || resolved.length < 2) return null;
            const ref = resolved[0];
            const kind = String(resolved[1] || "");
            let pageIndex = 0;
            try {{
              pageIndex = await pdf.getPageIndex(ref);
            }} catch (e) {{
              pageIndex = 0;
            }}
            const pageNum = (pageIndex || 0) + 1;
            const out = {{ pageNum, kind, left: null, top: null, right: null, bottom: null }};
            if (kind === "XYZ") {{
              out.left = resolved.length > 2 ? resolved[2] : null;
              out.top = resolved.length > 3 ? resolved[3] : null;
            }} else if (kind === "FitH" || kind === "FitBH") {{
              out.top = resolved.length > 2 ? resolved[2] : null;
            }} else if (kind === "FitV" || kind === "FitBV") {{
              out.left = resolved.length > 2 ? resolved[2] : null;
            }} else if (kind === "FitR") {{
              out.left = resolved.length > 2 ? resolved[2] : null;
              out.bottom = resolved.length > 3 ? resolved[3] : null;
              out.right = resolved.length > 4 ? resolved[4] : null;
              out.top = resolved.length > 5 ? resolved[5] : null;
            }}
            return out;
          }} catch (e) {{
            return null;
          }}
        }}

        async function _annolidRenderPreview() {{
          if (!previewState.open || !previewCanvas || !previewWrap || !previewBody || !previewCtx) return;
          if (previewState.mode === "citation") {{
            if (previewHighlight) previewHighlight.style.display = "none";
            return;
          }}
          const info = previewState.info;
          const pageNum = previewState.pageNum;
          try {{
            const page = await pdf.getPage(pageNum);
            const viewport = page.getViewport({{ scale: previewState.scale, rotation }});
            previewCanvas.width = Math.max(1, Math.round(viewport.width));
            previewCanvas.height = Math.max(1, Math.round(viewport.height));
            previewCanvas.style.width = viewport.width + "px";
            previewCanvas.style.height = viewport.height + "px";
            if (previewCtx) {{
              try {{ previewCtx.setTransform(1, 0, 0, 1, 0, 0); }} catch (e) {{}}
              try {{ previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height); }} catch (e) {{}}
            }}
            await page.render({{ canvasContext: previewCtx, viewport }}).promise;

            let highlightRect = null;
            if (info && info.kind === "FitR" && info.left != null && info.right != null && info.bottom != null && info.top != null) {{
              try {{
                const rect = [info.left, info.bottom, info.right, info.top];
                const vrect = viewport.convertToViewportRectangle(rect);
                const x1 = Math.min(vrect[0], vrect[2]);
                const x2 = Math.max(vrect[0], vrect[2]);
                const y1 = Math.min(vrect[1], vrect[3]);
                const y2 = Math.max(vrect[1], vrect[3]);
                highlightRect = {{ x: x1, y: y1, w: Math.max(2, x2 - x1), h: Math.max(2, y2 - y1) }};
              }} catch (e) {{
                highlightRect = null;
              }}
            }} else if (info && info.top != null) {{
              try {{
                const xPdf = (info.left != null) ? info.left : 0;
                const yPdf = info.top;
                const pt = viewport.convertToViewportPoint(xPdf, yPdf);
                const x = pt[0];
                const y = pt[1];
                highlightRect = {{ x: Math.max(0, x - 20), y: Math.max(0, y - 14), w: 220, h: 36 }};
              }} catch (e) {{
                highlightRect = null;
              }}
            }}

            if (previewHighlight && highlightRect) {{
              previewHighlight.style.display = "block";
              previewHighlight.style.left = Math.max(0, highlightRect.x) + "px";
              previewHighlight.style.top = Math.max(0, highlightRect.y) + "px";
              previewHighlight.style.width = Math.max(2, highlightRect.w) + "px";
              previewHighlight.style.height = Math.max(2, highlightRect.h) + "px";
              const targetY = Math.max(0, highlightRect.y - 90);
              try {{
                previewBody.scrollTo({{ top: targetY, behavior: "smooth" }});
              }} catch (e) {{
                previewBody.scrollTop = targetY;
              }}
            }} else if (previewHighlight) {{
              previewHighlight.style.display = "none";
            }}
          }} catch (e) {{
            if (previewHighlight) previewHighlight.style.display = "none";
          }}
        }}

        async function _annolidOpenDestinationPreview(dest, title) {{
          const info = await _annolidResolveDestination(dest);
          if (!info) return false;
          previewState.mode = "dest";
          previewState.citation = null;
          previewState.autoclose = false;
          previewState.pageNum = info.pageNum || 1;
          previewState.info = info;
          _annolidSetPreviewMessage("Jump target preview");
          _annolidSetPreviewLayout("dest");
          _annolidOpenPreviewModal(title || ("Page " + String(previewState.pageNum)));
          await _annolidRenderPreview();
          return true;
        }}

        if (previewCloseBtn) previewCloseBtn.addEventListener("click", () => _annolidClosePreviewModal());
        if (previewModal) previewModal.addEventListener("click", (ev) => {{
          if (ev.target === previewModal) _annolidClosePreviewModal();
        }});
        if (previewModal) previewModal.addEventListener("mouseenter", () => {{
          if (previewCloseTimer) {{
            clearTimeout(previewCloseTimer);
            previewCloseTimer = null;
          }}
        }});
        if (previewModal) previewModal.addEventListener("mouseleave", () => {{
          if (!previewState.open || !previewState.autoclose) return;
          if (previewCloseTimer) clearTimeout(previewCloseTimer);
          previewCloseTimer = setTimeout(() => _annolidClosePreviewModal(), 450);
        }});
        document.addEventListener("keydown", (ev) => {{
          if (ev.key === "Escape" && previewState.open) _annolidClosePreviewModal();
        }});
        if (previewZoomOutBtn) previewZoomOutBtn.addEventListener("click", async () => {{
          previewState.scale = Math.max(0.6, previewState.scale / 1.15);
          await _annolidRenderPreview();
        }});
        if (previewZoomInBtn) previewZoomInBtn.addEventListener("click", async () => {{
          previewState.scale = Math.min(6.0, previewState.scale * 1.15);
          await _annolidRenderPreview();
        }});
        if (previewZoomResetBtn) previewZoomResetBtn.addEventListener("click", async () => {{
          previewState.scale = 2.0;
          await _annolidRenderPreview();
        }});

        if (citePopover) {{
          citePopover.addEventListener("mouseenter", () => {{
            if (citeCloseTimer) {{
              clearTimeout(citeCloseTimer);
              citeCloseTimer = null;
            }}
          }});
          citePopover.addEventListener("mouseleave", () => {{
            _annolidScheduleCitationClose();
          }});
        }}
        if (container) {{
          container.addEventListener("scroll", () => _annolidUpdateCitePopoverPosition());
        }}
        window.addEventListener("resize", () => _annolidUpdateCitePopoverPosition());

        function _annolidCancelCitationHover() {{
          if (citeHoverTimer) {{
            clearTimeout(citeHoverTimer);
            citeHoverTimer = null;
          }}
        }}

        function _annolidScheduleCitationHover(span, fallbackDest, autoclose, delayMs) {{
          const citeNum = _annolidFindCitationNumberFromSpan(span);
          if (!citeNum) return false;
          if (citePopoverState.open && citePopoverState.number === citeNum) {{
            _annolidPositionCitePopover(span);
            return true;
          }}
          _annolidCancelCitationHover();
          citeHoverTimer = setTimeout(() => {{
            citeHoverTimer = null;
            _annolidOpenCitationPreview(
              citeNum,
              fallbackDest,
              autoclose,
              span
            ).catch(() => {{}});
          }}, Math.max(80, delayMs || 220));
          return true;
        }}

        function _annolidScheduleCitationClose() {{
          if (!citePopoverState.open || !citePopoverState.autoclose) return;
          if (citeCloseTimer) clearTimeout(citeCloseTimer);
          citeCloseTimer = setTimeout(() => _annolidHideCitationPopover(), 350);
        }}

        let _annolidHoverSpan = null;
        if (container) {{
          container.addEventListener("mousemove", (ev) => {{
            try {{
              if (ev.buttons && ev.buttons !== 0) return;
              const tool = (window.__annolidMarks && window.__annolidMarks.tool) ? window.__annolidMarks.tool : "select";
              if (tool !== "select") return;
              const link = ev.target && ev.target.closest ? ev.target.closest(".annotationLayer a") : null;
              if (link) return;
              const span = ev.target && ev.target.closest ? ev.target.closest(".textLayer span") : null;
              if (span === _annolidHoverSpan) return;
              _annolidHoverSpan = span;
              if (!span) {{
                _annolidCancelCitationHover();
                _annolidScheduleCitationClose();
                return;
              }}
              _annolidCancelCitationHover();
              const handled = _annolidScheduleCitationHover(span, null, true, 200);
              if (!handled) {{
                _annolidScheduleCitationClose();
              }}
            }} catch (e) {{}}
          }});
          container.addEventListener("mouseleave", () => {{
            _annolidHoverSpan = null;
            _annolidCancelCitationHover();
            _annolidScheduleCitationClose();
          }});
        }}

	        function _annolidQueueRender(fn) {{
	          const myEpoch = renderEpoch;
	          const task = renderChain.then(async () => {{
	            if (myEpoch !== renderEpoch) return;
	            return await fn(myEpoch);
	          }});
	          renderChain = task.catch((e) => {{
	            console.warn("Annolid render op failed", e);
	          }});
	          return task;
	        }}
	
	        function _annolidGetCurrentPageNum() {{
	          if (!container) return 1;
	          const pages = Array.from(container.querySelectorAll(".page"));
	          if (!pages.length) return 1;
	          const scrollTop = container.scrollTop;
	          let bestPage = 1;
	          let bestDist = Infinity;
	          pages.forEach((page) => {{
	            const top = page.offsetTop || 0;
	            const dist = Math.abs(top - scrollTop);
	            if (dist < bestDist) {{
	              bestDist = dist;
	              bestPage = parseInt(page.getAttribute("data-page-number") || "1", 10) || bestPage;
	            }}
	          }});
	          return bestPage;
	        }}
	
	        function _annolidSetDisabled(el, disabled) {{
	          if (!el) return;
	          el.classList.toggle("annolid-disabled", !!disabled);
	          try {{ el.disabled = !!disabled; }} catch (e) {{}}
	        }}
	
	        function _annolidUpdateNavState() {{
	          const current = _annolidGetCurrentPageNum();
	          if (pageInput && document.activeElement !== pageInput) {{
	            pageInput.value = String(current);
	          }}
	          _annolidSetDisabled(prevPageBtn, current <= 1);
	          _annolidSetDisabled(nextPageBtn, current >= total);
	          _annolidUpdateZoomLabel();
	        }}
	
	        function _annolidScrollToPage(pageNum, offsetFrac) {{
	          if (!container) return;
	          const el = container.querySelector(`.page[data-page-number="${{pageNum}}"]`);
	          if (!el) return;
	          const frac = isFinite(offsetFrac) ? Math.max(0, Math.min(1, offsetFrac)) : 0;
	          const inner = el.clientHeight || 1;
	          container.scrollTop = Math.max(0, (el.offsetTop || 0) + inner * frac - 8);
	        }}
	
	        function _annolidGetScrollAnchor() {{
	          if (!container) return {{ pageNum: 1, offsetFrac: 0 }};
	          const current = _annolidGetCurrentPageNum();
	          const el = container.querySelector(`.page[data-page-number="${{current}}"]`);
	          if (!el) return {{ pageNum: current, offsetFrac: 0 }};
	          const top = el.offsetTop || 0;
	          const h = el.clientHeight || 1;
	          const frac = (container.scrollTop - top) / h;
	          return {{ pageNum: current, offsetFrac: isFinite(frac) ? frac : 0 }};
	        }}
	
	        async function _annolidTransformMarksForView(oldScale, newScale, oldRotation, newRotation) {{
	          if (!pdf) return;
	          const pages = window.__annolidPages || {{}};
	          const keys = Object.keys(pages);
	          if (!keys.length) return;
	          const fromScale = _annolidClampScale(oldScale);
	          const toScale = _annolidClampScale(newScale);
	          const fromRotation = _annolidNormalizeRotation(oldRotation);
	          const toRotation = _annolidNormalizeRotation(newRotation);
	          await Promise.all(keys.map(async (key) => {{
	            const state = pages[key];
	            if (!state || !state.marks) return;
	            const pageNum = parseInt(key, 10) || state.pageNum || 0;
	            if (!pageNum) return;
	            let pageRef = null;
	            try {{
	              pageRef = await pdf.getPage(pageNum);
	            }} catch (e) {{
	              return;
	            }}
	            let fromVp = null;
	            let toVp = null;
	            try {{
	              fromVp = pageRef.getViewport({{ scale: fromScale, rotation: fromRotation }});
	              toVp = pageRef.getViewport({{ scale: toScale, rotation: toRotation }});
	            }} catch (e) {{
	              return;
	            }}
	            const convertPoint = (pt) => {{
	              if (!pt || !fromVp || !toVp || typeof fromVp.convertToPdfPoint !== "function") return pt;
	              try {{
	                const pdfPt = fromVp.convertToPdfPoint(pt.x, pt.y);
	                const nextPt = toVp.convertToViewportPoint(pdfPt[0], pdfPt[1]);
	                return {{ ...pt, x: nextPt[0], y: nextPt[1] }};
	              }} catch (e) {{
	                return pt;
	              }}
	            }};
	            const transformRect = (rect) => {{
	              if (!rect) return rect;
	              const corners = [
	                convertPoint({{ x: rect.x, y: rect.y }}),
	                convertPoint({{ x: rect.x + rect.w, y: rect.y }}),
	                convertPoint({{ x: rect.x + rect.w, y: rect.y + rect.h }}),
	                convertPoint({{ x: rect.x, y: rect.y + rect.h }}),
	              ].filter(Boolean);
	              if (!corners.length) return rect;
	              const xs = corners.map((c) => c.x);
	              const ys = corners.map((c) => c.y);
	              const minX = Math.min(...xs);
	              const maxX = Math.max(...xs);
	              const minY = Math.min(...ys);
	              const maxY = Math.max(...ys);
	              return {{ ...rect, x: minX, y: minY, w: maxX - minX, h: maxY - minY }};
	            }};
	            if (state.marks.strokes) {{
	              state.marks.strokes = state.marks.strokes.map((stroke) => ({{
	                ...stroke,
	                points: (stroke.points || []).map((p) => convertPoint(p)),
	              }}));
	            }}
	            if (state.marks.highlights) {{
	              state.marks.highlights = state.marks.highlights.map((hl) => ({{
	                ...hl,
	                rects: (hl.rects || []).map((r) => transformRect(r)),
	              }}));
	            }}
	          }}));
	        }}
	
	        async function _annolidEnsureRenderedThrough(pageNum) {{
	          const target = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
	          return _annolidQueueRender(async (epoch) => {{
	            while (nextPage <= target && nextPage <= total) {{
	              await renderPage(nextPage, epoch);
	              nextPage += 1;
	              await new Promise(r => setTimeout(r, 0));
	            }}
	          }});
	        }}
	
	        async function _annolidGoToPage(pageNum) {{
	          const target = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
	          await _annolidEnsureRenderedThrough(target);
	          _annolidScrollToPage(target, 0);
	          _annolidUpdateNavState();
	        }}

		        // Expose render helpers for the Qt bridge.
		        window.__annolidEnsureRenderedThrough = _annolidEnsureRenderedThrough;
		        window.__annolidGoToPage = _annolidGoToPage;
		        window.__annolidZoomFitWidth = _annolidZoomFitWidth;
	        window.__annolidRotate = function(delta) {{
	          const step = isFinite(delta) ? delta : 90;
	          _annolidRerenderAll(scale, rotation + step);
	        }};
        window.__annolidSetRotation = function(angle) {{
          if (!isFinite(angle)) return;
          _annolidRerenderAll(scale, angle);
        }};

	        window.__annolidHighlightParagraphByText = async function(pageNum, text) {{
	          try {{
	            const p = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
	            const wanted = _annolidNormalizeText(text || "").toLowerCase();
	            if (!wanted) return;
	            await _annolidEnsureRenderedThrough(p);
	            const paras = _annolidBuildParagraphsForPage(p) || [];
	            if (!paras.length) return;

	            function scoreCandidate(candidate) {{
	              const cand = _annolidNormalizeText(candidate || "").toLowerCase();
	              if (!cand) return 0;
	              if (cand === wanted) return 1.0;
	              if (cand.includes(wanted) || wanted.includes(cand)) {{
	                return Math.min(cand.length, wanted.length) / Math.max(1, Math.max(cand.length, wanted.length));
	              }}
	              const a = wanted.split(" ").filter(Boolean);
	              const b = cand.split(" ").filter(Boolean);
	              if (!a.length || !b.length) return 0;
	              const limit = 60;
	              const setA = new Set(a.slice(0, limit));
	              const setB = new Set(b.slice(0, limit));
	              let inter = 0;
	              setA.forEach((t) => {{ if (setB.has(t)) inter += 1; }});
	              const denom = Math.max(1, setA.size + setB.size - inter);
	              return inter / denom;
	            }}

	            let best = null;
	            let bestScore = 0;
	            for (const para of paras) {{
	              const s = scoreCandidate(para.text);
	              if (s > bestScore) {{
	                bestScore = s;
	                best = para;
	              }}
	            }}
	            if (!best || bestScore < 0.10) return;
	            const spans = best.spans || [];
	            if (!spans.length) return;
	            window.__annolidHighlightSentenceIndices(spans);
	          }} catch (e) {{
	            // ignore
	          }}
	        }};

          window.__annolidHighlightSentenceByText = async function(pageNum, text) {{
            try {{
              const p = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
              const wanted = _annolidNormalizeText(text || "").toLowerCase();
              const scrollPageFallback = () => {{
                try {{
                  const container = document.getElementById("viewerContainer");
                  const page = document.querySelector(`.page[data-page-number='${{p}}']`);
                  if (!container || !page) return;
                  const center = (page.offsetTop || 0) + (page.clientHeight || 0) / 2;
                  const target = Math.max(0, center - container.clientHeight / 2);
                  container.scrollTop = target;
                }} catch (e) {{}}
              }};
              const currentPage = (typeof _annolidGetCurrentPageNum === "function") ? _annolidGetCurrentPageNum() : 0;
              if (!wanted) {{
                await _annolidEnsureRenderedThrough(p);
                if (currentPage !== p) scrollPageFallback();
                return;
              }}

              await _annolidEnsureRenderedThrough(p);
              const paras = _annolidBuildParagraphsForPage(p) || [];
              if (!paras.length) {{
                if (currentPage !== p) scrollPageFallback();
                return;
              }}

              function scoreCandidate(candidate) {{
                const cand = _annolidNormalizeText(candidate || "").toLowerCase();
                if (!cand) return 0;
                if (cand === wanted) return 1.0;
                if (cand.includes(wanted) || wanted.includes(cand)) {{
                  return Math.min(cand.length, wanted.length) / Math.max(1, Math.max(cand.length, wanted.length));
                }}
                const a = wanted.split(" ").filter(Boolean);
                const b = cand.split(" ").filter(Boolean);
                if (!a.length || !b.length) return 0;
                const limit = 80;
                const setA = new Set(a.slice(0, limit));
                const setB = new Set(b.slice(0, limit));
                let inter = 0;
                setA.forEach((t) => {{ if (setB.has(t)) inter += 1; }});
                const denom = Math.max(1, setA.size + setB.size - inter);
                return inter / denom;
              }}

              let best = null;
              let bestScore = 0;
              for (const para of paras) {{
                const splits = (typeof window.__annolidSplitParagraphIntoSentences === "function")
                  ? (window.__annolidSplitParagraphIntoSentences(para) || [])
                  : [];
                if (splits.length) {{
                  for (const s of splits) {{
                    const sc = scoreCandidate(s.text || "");
                    if (sc > bestScore) {{
                      bestScore = sc;
                      best = s;
                    }}
                  }}
                }} else {{
                  const sc = scoreCandidate(para.text || "");
                  if (sc > bestScore) {{
                    bestScore = sc;
                    best = para;
                  }}
                }}
              }}

              if (!best || bestScore < 0.05) {{
                if (currentPage !== p) scrollPageFallback();
                return;
              }}
              const spans = best.spans || [];
              if (!spans.length) {{
                if (currentPage !== p) scrollPageFallback();
                return;
              }}
              window.__annolidHighlightSentenceIndices && window.__annolidHighlightSentenceIndices(spans);
              window.__annolidScrollToSentence && window.__annolidScrollToSentence(spans, p);
            }} catch (e) {{
              try {{
                const p = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
                if (typeof _annolidEnsureRenderedThrough === "function") {{
                  await _annolidEnsureRenderedThrough(p);
                }}
              }} catch (e2) {{}}
            }}
          }};
	
	        async function _annolidRerenderAll(newScale, newRotation = rotation) {{
	          if (!container) return;
	          const desiredScale = (newScale == null) ? scale : newScale;
	          const clamped = _annolidClampScale(desiredScale);
	          const targetRotation = _annolidNormalizeRotation(newRotation);
	          const oldScale = scale;
	          const oldRotation = rotation;
	          if (Math.abs(clamped - oldScale) < 0.001 && targetRotation === oldRotation) return;
	          if (zoomBusy) {{
	            pendingZoom = {{ scale: clamped, rotation: targetRotation }};
	            return;
	          }}
	          zoomBusy = true;
	          pendingZoom = null;
	
	          try {{
	            await _annolidTransformMarksForView(oldScale, clamped, oldRotation, targetRotation);
	            const anchor = _annolidGetScrollAnchor();
	            scale = clamped;
	            rotation = targetRotation;
	
	            renderEpoch += 1;
	            renderChain = Promise.resolve();
	            nextPage = 1;
	            container.innerHTML = "";
	            window.__annolidSpans = [];
	            window.__annolidSpanCounter = 0;
	            window.__annolidSpanMeta = {{}};
	            window.__annolidLinkTargets = {{}};
	            window.__annolidLinkTargetCounter = 0;
	            window.__annolidRenderedPages = 0;
	            const pages = window.__annolidPages || {{}};
	            Object.keys(pages).forEach((key) => {{
	              const state = pages[key];
	              if (!state) return;
	              state.pageDiv = null;
	              state.width = 0;
	              state.height = 0;
	              state.dpr = 1;
	              state.ttsCanvas = null;
	              state.ttsCtx = null;
	              state.markCanvas = null;
	              state.markCtx = null;
	            }});
	            window.__annolidParagraphsByPage = {{}};
	            window.__annolidParagraphOffsets = {{}};
	            window.__annolidParagraphTotal = 0;
	            window.__annolidParagraphs = [];
	
	            _annolidUpdateNavState();
	            await _annolidEnsureRenderedThrough(anchor.pageNum);
	            _annolidScrollToPage(anchor.pageNum, anchor.offsetFrac);
	            _annolidUpdateNavState();
	          }} finally {{
	            zoomBusy = false;
	            if (pendingZoom) {{
	              const next = pendingZoom;
	              pendingZoom = null;
	              _annolidRerenderAll(next.scale, next.rotation);
	            }}
	          }}
	        }}
	
	        async function _annolidZoomFitWidth() {{
	          if (!container) return;
	          try {{
	            const page = await pdf.getPage(1);
	            const baseViewport = page.getViewport({{ scale: 1, rotation }});
	            const gutter = 32;
	            const available = Math.max(100, container.clientWidth - gutter);
	            const target = available / Math.max(1, baseViewport.width);
	            await _annolidRerenderAll(target, rotation);
	          }} catch (e) {{
	            console.warn("Zoom fit failed", e);
	          }}
	        }}
	
	        function _annolidZoomBy(factor) {{
	          const next = _annolidClampScale(scale * factor);
	          _annolidRerenderAll(next, rotation);
	        }}
	
        async function _annolidPrintPdf() {{
          try {{
            const url = await _annolidGetPdfObjectUrl();
            const iframe = document.createElement("iframe");
            iframe.style.position = "fixed";
            iframe.style.right = "0";
            iframe.style.bottom = "0";
            iframe.style.width = "1px";
            iframe.style.height = "1px";
            iframe.style.border = "0";
            iframe.src = url || pdfUrl;
            iframe.onload = () => {{
              try {{
                iframe.contentWindow.focus();
                iframe.contentWindow.print();
              }} catch (e) {{
                window.print();
              }}
              setTimeout(() => {{
                try {{ iframe.remove(); }} catch (e) {{}}
              }}, 1500);
            }};
            document.body.appendChild(iframe);
          }} catch (e) {{
            window.print();
          }}
        }}
	
	        if (prevPageBtn) prevPageBtn.addEventListener("click", () => _annolidGoToPage(_annolidGetCurrentPageNum() - 1));
	        if (nextPageBtn) nextPageBtn.addEventListener("click", () => _annolidGoToPage(_annolidGetCurrentPageNum() + 1));
	        if (pageInput) {{
	          pageInput.addEventListener("keydown", (ev) => {{
	            if (ev.key === "Enter") {{
	              ev.preventDefault();
	              _annolidGoToPage(pageInput.value);
	            }}
	          }});
	          pageInput.addEventListener("change", () => _annolidGoToPage(pageInput.value));
	        }}
	        if (zoomOutBtn) zoomOutBtn.addEventListener("click", () => _annolidZoomBy(1 / 1.1));
	        if (zoomInBtn) zoomInBtn.addEventListener("click", () => _annolidZoomBy(1.1));
	        if (zoomResetBtn) zoomResetBtn.addEventListener("click", () => _annolidRerenderAll(1.0));
	        if (zoomFitBtn) zoomFitBtn.addEventListener("click", _annolidZoomFitWidth);
        if (rotateBtn) rotateBtn.addEventListener("click", () => _annolidRerenderAll(scale, rotation + 90));
        if (printBtn) printBtn.addEventListener("click", () => _annolidPrintPdf().catch(() => {{}}));
	        _annolidUpdateNavState();
		        if (container) {{
		          container.addEventListener("dblclick", async (ev) => {{
		            if (!window.__annolidReaderEnabled) return;
		            if (!window.__annolidBridge || typeof window.__annolidBridge.onParagraphClicked !== "function") return;
	            if (window.__annolidMarks && window.__annolidMarks.tool && window.__annolidMarks.tool !== "select") return;
	            const sel = window.getSelection ? window.getSelection() : null;
	            // Capture double-click selection to start reading from the exact word position.
	            let selectedText = "";
	            let selSpanEl = null;
	            let selSpanIdx = -1;
	            let selCharOffset = 0;
	            let selOffsetIsChar = false;
	            try {{
	              if (sel && !sel.isCollapsed && sel.rangeCount) {{
	                selectedText = String(sel.toString() || "").trim();
	                const range = sel.getRangeAt(0);
	                let node = range ? range.startContainer : null;
	                let el = null;
	                selOffsetIsChar = !!(node && node.nodeType === 3);
	                if (node && node.nodeType === 3) el = node.parentElement;
	                else if (node && node.nodeType === 1) el = node;
	                if (el && el.closest) {{
	                  const closestSpan = el.closest(".textLayer span");
	                  if (closestSpan) {{
	                    selSpanEl = closestSpan;
	                    if (closestSpan.dataset && closestSpan.dataset.annolidIndex) {{
	                      selSpanIdx = parseInt(closestSpan.dataset.annolidIndex, 10);
	                    }}
	                  }}
	                }}
	                selCharOffset = (
	                  selOffsetIsChar && range && typeof range.startOffset === "number"
	                ) ? range.startOffset : 0;
	              }}
	            }} catch (e) {{
	              selectedText = "";
	              selSpanEl = null;
	              selSpanIdx = -1;
	              selCharOffset = 0;
	              selOffsetIsChar = false;
	            }}
	            // Allow double-click to trigger reading even if the browser selects a word.
	            if (ev.type !== "dblclick" && sel && !sel.isCollapsed) return;
	            if (ev.type === "dblclick" && sel && !sel.isCollapsed) {{
	              try {{ sel.removeAllRanges(); }} catch (e) {{}}
	            }}
	            const pageDiv = ev.target && ev.target.closest ? ev.target.closest(".page") : null;
	            if (!pageDiv) return;
	            const pageNum = parseInt(pageDiv.getAttribute("data-page-number") || "0", 10);
	            if (!pageNum) return;

          let pageParas = window.__annolidParagraphsByPage[String(pageNum)];
          if (!pageParas || !pageParas.length) {{
            pageParas = _annolidBuildParagraphsForPage(pageNum);
            _annolidRebuildParagraphIndex();
          }}

	          let spanIdx = -1;
	          let spanEl = ev.target && ev.target.closest ? ev.target.closest(".textLayer span") : null;
	          if (selSpanIdx >= 0 && selSpanEl) {{
	            spanIdx = selSpanIdx;
	            spanEl = selSpanEl;
	          }} else if (spanEl && spanEl.dataset && spanEl.dataset.annolidIndex) {{
	            spanIdx = parseInt(spanEl.dataset.annolidIndex, 10);
	          }}
	          let paraIndex = -1;
	          if (spanIdx >= 0) {{
	            paraIndex = _annolidFindParagraphIndexBySpan(pageNum, spanIdx);
	          }}
          if (paraIndex < 0) {{
            const pageRect = pageDiv.getBoundingClientRect();
            const x = ev.clientX - pageRect.left;
            const y = ev.clientY - pageRect.top;
            paraIndex = _annolidFindParagraphIndexByPoint(pageNum, x, y);
            if (paraIndex < 0) {{
              // If textLayer spans are missing, fall back to PDF text extraction.
              await _annolidBuildTextParagraphsForPage(pageNum);
              await _annolidEnsureParagraphsFrom(pageNum);
              const rebuilt = window.__annolidParagraphsByPage[String(pageNum)] || [];
              if (rebuilt.length) {{
                const frac = pageRect.height > 0 ? Math.max(0, Math.min(1, y / pageRect.height)) : 0;
                paraIndex = Math.max(0, Math.min(rebuilt.length - 1, Math.floor(frac * rebuilt.length)));
              }}
            }}
          }}
          if (paraIndex < 0) return;
          await _annolidEnsureParagraphsFrom(pageNum);
            const offset = window.__annolidParagraphOffsets[pageNum] || 0;
            const startIndex = offset + paraIndex;
            const remaining = window.__annolidParagraphs.slice(startIndex).map((p) => ({{
              text: p.text || "",
              spans: p.spans || [],
              pageNum: (parseInt(p.pageNum || p.page || 0, 10) || pageNum),
            }}));
	            if (!remaining.length) return;
	            const sentences = [];
	            const targetSpanIdx = (spanIdx >= 0 && Number.isInteger(spanIdx)) ? spanIdx : null;
	            let sentenceCursor = 0;
            const splitFallback = (text) => {{
              const normalized = _annolidNormalizeText(text || "");
              if (!normalized) return [];
              const out = [];
              const ranges = (typeof window.__annolidSplitTextIntoSentenceRanges === "function")
                ? (window.__annolidSplitTextIntoSentenceRanges(normalized) || [])
                : [];
              for (const r of ranges) {{
                const seg = _annolidNormalizeText(normalized.slice(r[0], r[1]));
                if (seg) out.push(seg);
              }}
              return out.length ? out : [normalized];
            }};
	            let startSentenceIdx = 0;
	            remaining.forEach((p, idx) => {{
	              const isTargetParagraph = (idx === 0);
	              const paraSentences = [];
	              if (typeof window.__annolidSplitParagraphIntoSentences === "function") {{
	                const splits = window.__annolidSplitParagraphIntoSentences(p) || [];
	                if (splits.length) {{
	                  splits.forEach((s) => paraSentences.push(s));
	                }}
	              }}
	              const pageForPara = (parseInt(p.pageNum || p.page || 0, 10) || pageNum);
	              if (!paraSentences.length) {{
	                const segs = splitFallback(p.text || "");
	                if (segs.length) {{
	                  segs.forEach((seg) => {{
	                    paraSentences.push({{
	                      text: seg,
	                      spans: [],
	                      pageNum: pageForPara,
	                    }});
	                  }});
	                }}
	              }}
	              if (!paraSentences.length) {{
	                paraSentences.push({{
	                  text: p.text || "",
	                  spans: [],
	                  pageNum: pageForPara,
	                }});
	              }}

	              if (isTargetParagraph && paraSentences.length) {{
	                let matched = 0;
	                if (targetSpanIdx != null) {{
	                  for (let i = 0; i < paraSentences.length; i++) {{
	                    const spans = Array.isArray(paraSentences[i].spans) ? paraSentences[i].spans : [];
	                    if (spans.indexOf(targetSpanIdx) >= 0) {{
	                      matched = i;
	                      break;
	                    }}
	                  }}
	                }}
	                startSentenceIdx = sentenceCursor + matched;
	              }}

	              paraSentences.forEach((s) => sentences.push(s));
	              sentenceCursor += paraSentences.length;
	            }});

	            const totalSentences = sentences.length;
	            if (startSentenceIdx < 0 || startSentenceIdx >= totalSentences) {{
	              startSentenceIdx = 0;
	            }}
	            // Start from the clicked word within the starting sentence when possible.
	            if (totalSentences && targetSpanIdx != null) {{
	              try {{
	                const startSentence = sentences[startSentenceIdx];
	                const spans = Array.isArray(startSentence.spans) ? startSentence.spans : [];
	                const pos = spans.indexOf(targetSpanIdx);
	                if (pos >= 0) {{
	                  const nodes = window.__annolidSpans || [];
	                  const trimmedSpans = spans.slice(pos);
	                  const parts = [];
	                  for (let i = 0; i < trimmedSpans.length; i++) {{
	                    const spanIndex = trimmedSpans[i];
	                    const node = nodes[spanIndex];
	                    if (!node) continue;
	                    let raw = String(node.textContent || "");
	                    if (i === 0) {{
	                      // Prefer the exact char offset from DOM selection, otherwise best-effort locate the selected word.
	                      let cut = (selSpanIdx === targetSpanIdx && selOffsetIsChar)
	                        ? (parseInt(selCharOffset, 10) || 0)
	                        : 0;
	                      if ((!selOffsetIsChar || cut >= raw.length) && selectedText) {{
	                        const idx2 = raw.toLowerCase().indexOf(String(selectedText).toLowerCase());
	                        if (idx2 >= 0) cut = idx2;
	                      }}
	                      cut = Math.max(0, Math.min(raw.length, cut));
	                      raw = raw.slice(cut);
	                    }}
	                    const t = _annolidNormalizeText(raw);
	                    if (t) parts.push(t);
	                  }}
	                  const rebuilt = _annolidNormalizeText(parts.join(" "));
	                  if (rebuilt) {{
	                    startSentence.text = rebuilt;
	                    startSentence.spans = trimmedSpans;
	                  }}
	                }}
	              }} catch (e) {{}}
	            }}

	            window.__annolidBridge.onParagraphClicked({{
	              startIndex,
	              total: window.__annolidParagraphTotal || (startIndex + remaining.length),
	              paragraphs: remaining,
	              sentences,
	              sentenceStartIndex: 0,
	              sentenceLocalStartIndex: startSentenceIdx,
	              sentenceTotal: totalSentences,
	            }});
	          }});
	        }}

	        async function renderPage(pageNum, epoch) {{
	          if (epoch !== renderEpoch) return;
	          const page = await pdf.getPage(pageNum);
	          const viewport = page.getViewport({{ scale, rotation }});

          const pageDiv = document.createElement("div");
          pageDiv.className = "page";
          pageDiv.setAttribute("data-page-number", String(pageNum));
          pageDiv.style.width = viewport.width + "px";
          pageDiv.style.height = viewport.height + "px";

          const canvas = document.createElement("canvas");
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          pageDiv.appendChild(canvas);

          const textLayerDiv = document.createElement("div");
          textLayerDiv.className = "textLayer";
          pageDiv.appendChild(textLayerDiv);

          const annotationLayerDiv = document.createElement("div");
          annotationLayerDiv.className = "annotationLayer";
          annotationLayerDiv.style.pointerEvents = "none";
          pageDiv.appendChild(annotationLayerDiv);

          const ttsLayer = document.createElement("canvas");
          ttsLayer.className = "annolid-tts-layer";
          pageDiv.appendChild(ttsLayer);

          const markLayer = document.createElement("canvas");
          markLayer.className = "annolid-mark-layer";
          pageDiv.appendChild(markLayer);

	          if (epoch !== renderEpoch) return;
	          container.appendChild(pageDiv);

          const ctx = canvas.getContext("2d");
	          if (epoch !== renderEpoch) return;
	          await page.render({{ canvasContext: ctx, viewport }}).promise;
	          window.__annolidRenderedPages = (window.__annolidRenderedPages || 0) + 1;

          try {{
            const annots = await page.getAnnotations({{ intent: "display" }});
            annotationLayerDiv.innerHTML = "";
            (annots || []).forEach((a) => {{
              if (!a || a.subtype !== "Link" || !a.rect) return;
              let dest = a.dest || null;
              const url = a.url || a.unsafeUrl || null;
              const action = a.action || null;
              if (!dest && action && typeof action === "object") {{
                dest = action.dest || action.destination || null;
              }}
              if (!dest && !url && !action) return;
              let rect = null;
              try {{
                const vrect = viewport.convertToViewportRectangle(a.rect);
                const x1 = Math.min(vrect[0], vrect[2]);
                const x2 = Math.max(vrect[0], vrect[2]);
                const y1 = Math.min(vrect[1], vrect[3]);
                const y2 = Math.max(vrect[1], vrect[3]);
                rect = {{ left: x1, top: y1, width: Math.max(1, x2 - x1), height: Math.max(1, y2 - y1) }};
              }} catch (e) {{
                rect = null;
              }}
              if (!rect) return;
              const id = String(window.__annolidLinkTargetCounter || 0);
              window.__annolidLinkTargetCounter = (window.__annolidLinkTargetCounter || 0) + 1;
              window.__annolidLinkTargets[id] = {{ dest, url, action, pageNum }};
              const link = document.createElement("a");
              link.setAttribute("data-annolid-link-id", id);
              link.style.pointerEvents = "auto";
              link.style.left = rect.left + "px";
              link.style.top = rect.top + "px";
              link.style.width = rect.width + "px";
              link.style.height = rect.height + "px";
              link.href = url ? String(url) : "#";
              if (url) link.target = "_blank";
              link.addEventListener("click", async (ev) => {{
                try {{
                  const tool = (window.__annolidMarks && window.__annolidMarks.tool) ? window.__annolidMarks.tool : "select";
                  if (tool !== "select") return;
                  const payload = window.__annolidLinkTargets[id] || null;
                  if (!payload || !payload.dest) return; // allow external links
                  ev.preventDefault();
                  ev.stopPropagation();
                  const span = _annolidFindUnderlyingTextSpan(link);
                  const citeNum = _annolidFindCitationNumberFromSpan(span);
                  if (citeNum) {{
                    await _annolidOpenCitationPreview(
                      citeNum,
                      payload.dest,
                      true,
                      span || link
                    );
                    return;
                  }}
                  await _annolidOpenDestinationPreview(payload.dest, "Link preview");
                }} catch (e) {{}}
              }});
              link.addEventListener("mouseenter", () => {{
                try {{
                  const tool = (window.__annolidMarks && window.__annolidMarks.tool) ? window.__annolidMarks.tool : "select";
                  if (tool !== "select") return;
                  const payload = window.__annolidLinkTargets[id] || null;
                  if (!payload || !payload.dest) return;
                  const span = _annolidFindUnderlyingTextSpan(link);
                  _annolidScheduleCitationHover(span, payload.dest, true, 200);
                }} catch (e) {{}}
              }});
              link.addEventListener("mouseleave", () => {{
                try {{
                  _annolidCancelCitationHover();
                  _annolidScheduleCitationClose();
                }} catch (e) {{}}
              }});
              annotationLayerDiv.appendChild(link);
            }});
          }} catch (e) {{
            try {{ annotationLayerDiv.innerHTML = ""; }} catch (e2) {{}}
          }}

          try {{
            const textContent = await page.getTextContent();
            if (pdfjsLib.renderTextLayer) {{
	              const task = pdfjsLib.renderTextLayer({{
	                textContent,
	                container: textLayerDiv,
	                viewport,
	                textDivs: [],
	                enhanceTextSelection: false,
	              }});
              if (task && task.promise) {{
                await task.promise;
              }}
            }}
          }} catch (e) {{
            console.warn("PDF.js text layer failed", e);
          }}
          const newSpans = Array.from(textLayerDiv.querySelectorAll("span"));
          newSpans.forEach((span) => {{
            if (!span.dataset.annolidIndex) {{
              const nextIdx = window.__annolidSpanCounter || 0;
              span.dataset.annolidIndex = String(nextIdx);
              window.__annolidSpanCounter = nextIdx + 1;
            }}
            const idx = parseInt(span.dataset.annolidIndex, 10);
            if (!Number.isNaN(idx)) {{
              window.__annolidSpans[idx] = span;
            }}
          }});
	          _annolidBuildParagraphsForPage(pageNum);
	          _annolidRebuildParagraphIndex();
          try {{
            const state = _annolidGetPageState(pageNum);
            const ttsSetup = _annolidSetupHiDpiCanvas(ttsLayer, viewport.width, viewport.height);
            const markSetup = _annolidSetupHiDpiCanvas(markLayer, viewport.width, viewport.height);
            state.pageDiv = pageDiv;
            state.width = viewport.width;
            state.height = viewport.height;
            state.dpr = markSetup.dpr || 1;
            state.ttsCanvas = ttsLayer;
            state.ttsCtx = ttsSetup.ctx;
            state.markCanvas = markLayer;
            state.markCtx = markSetup.ctx;
            _annolidBindMarkCanvas(state);
            // Apply current tool mode to this newly created mark layer.
            const drawing = (window.__annolidMarks && window.__annolidMarks.tool) ? window.__annolidMarks.tool !== "select" : false;
            markLayer.style.pointerEvents = drawing ? "auto" : "none";
            markLayer.style.cursor = drawing ? "crosshair" : "default";
            _annolidRenderMarks(pageNum);
            if (window.__annolidRenderTts) window.__annolidRenderTts();
          }} catch (e) {{
            console.warn("Annolid page layer init failed", e);
          }}
        }}

	        function renderMore(maxCount) {{
	          return _annolidQueueRender(async (epoch) => {{
	            let count = 0;
	            while (nextPage <= total && count < maxCount) {{
	              await renderPage(nextPage, epoch);
	              nextPage += 1;
	              count += 1;
	              await new Promise(r => setTimeout(r, 0));
	            }}
	          }});
	        }}
	
	        await renderMore(2);
	        if (container) {{
	          let scrollScheduled = false;
	          container.addEventListener("scroll", () => {{
	            if (scrollScheduled) return;
	            scrollScheduled = true;
	            requestAnimationFrame(() => {{
	              scrollScheduled = false;
	              _annolidUpdateNavState();
	              const nearBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 600;
	              if (nearBottom) {{
	                renderMore(2);
	              }}
	            }});
	          }});
	        }}
      }} catch (err) {{
	        console.error("PDF.js render failed", err);
	        try {{
	          const msg = (err && err.message) ? err.message : String(err);
          document.body.setAttribute("data-pdfjs-error", msg);
        }} catch (e) {{
          document.body.setAttribute("data-pdfjs-error", "PDF.js render failed");
        }}
      }}
    }});
  </script>
</head>
<body>
  <div id="annolidToolbar">
    <div class="annolid-toolbar-left">
      <button id="annolidMenuBtn" title="Menu" class="annolid-icon-btn">☰</button>
      <div id="annolidMenuPanel">
        <button data-action="first-page">Go to first page</button>
        <button data-action="fit">Fit width</button>
        <button data-action="reset">Reset zoom</button>
        <button data-action="print">Print</button>
      </div>
      <div class="annolid-title" id="annolidTitle">PDF</div>
    </div>
    <div class="annolid-nav">
      <button id="annolidPrevPage" title="Previous page">◀</button>
      <input id="annolidPageInput" type="number" value="1" min="1" />
      <button id="annolidNextPage" title="Next page">▶</button>
      <span>/ <span id="annolidTotalPages">-</span></span>
      <span class="annolid-sep"></span>
      <button id="annolidZoomOut" title="Zoom out">-</button>
      <div id="annolidZoomLabel">125%</div>
      <button id="annolidZoomIn" title="Zoom in">+</button>
      <button id="annolidZoomReset" title="Reset zoom">100%</button>
      <button id="annolidZoomFit" title="Fit width">Fit</button>
      <button id="annolidRotate" title="Rotate 90°">⟳</button>
      <button id="annolidPrint" title="Print PDF">Print</button>
    </div>
    <div class="annolid-actions" id="annolidActionRow">
      <div class="annolid-group" data-overflow="auto" id="annolidToolsGroup">
        <button id="annolidToolSelect" class="annolid-active" title="Select">Select</button>
        <button id="annolidToolPen" title="Pen">Pen</button>
        <button id="annolidToolHighlighter" title="Marker">Mark</button>
      </div>
      <div class="annolid-group" data-overflow="auto" id="annolidEditGroup">
        <button id="annolidUndo" title="Undo">⟲</button>
        <button id="annolidClear" title="Clear all">✕</button>
      </div>
      <div class="annolid-group annolid-mark-options" data-overflow="auto" id="annolidMarkOptions">
        <button id="annolidHighlightSelection" title="Highlight selection">Highlight</button>
        <span class="annolid-option-label">Color</span>
        <input id="annolidColorInline" type="color" value="#ffb300" title="Stroke color" />
        <span class="annolid-option-label">Size</span>
        <input id="annolidSizeInline" type="range" min="2" max="24" value="10" title="Stroke size" />
      </div>
    </div>
  </div>
  <div id="viewerContainer">
  </div>
  <div class="annolid-modal" id="annolidPreviewModal" role="dialog" aria-modal="true">
    <div class="annolid-modal-content" role="document">
      <div class="annolid-modal-header">
        <div class="annolid-modal-title" id="annolidPreviewTitle">Preview</div>
        <div class="annolid-modal-actions">
          <button id="annolidPreviewZoomOut" title="Zoom out">-</button>
          <button id="annolidPreviewZoomIn" title="Zoom in">+</button>
          <button id="annolidPreviewZoomReset" title="Reset zoom">100%</button>
          <button id="annolidPreviewClose" title="Close">Close</button>
        </div>
      </div>
      <div class="annolid-modal-body" id="annolidPreviewBody">
        <div class="annolid-preview-grid" id="annolidPreviewGrid">
          <div class="annolid-preview-text" id="annolidPreviewText"><span class="annolid-muted">Hover a citation like [41] to preview the reference.</span></div>
          <div class="annolid-modal-canvas-wrap" id="annolidPreviewCanvasWrap">
            <canvas id="annolidPreviewCanvas"></canvas>
            <div id="annolidPreviewHighlight"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="annolid-cite-popover" id="annolidCitePopover" role="tooltip">
    <div class="annolid-cite-card">
      <div class="annolid-cite-title" id="annolidCiteTitle">Reference</div>
      <div class="annolid-cite-body" id="annolidCiteBody"></div>
    </div>
  </div>
</body>
</html>
        """.strip()
        try:
            self._web_view.setHtml(html, base_url)
            return True
        except Exception as exc:
            logger.info(f"Failed to load PDF.js viewer: {exc}")
            return False

    def _render_current_page(self) -> None:
        """Render the current page and update text/labels."""
        if self._doc is None:
            return

        # Only used in fallback mode.
        page = self._doc.load_page(self._current_page)
        try:
            import fitz  # type: ignore[import]

            matrix = fitz.Matrix(self._zoom, self._zoom)
            if self._rotation:
                matrix = matrix.prerotate(self._rotation)
            pix = page.get_pixmap(matrix=matrix)
        except Exception:
            pix = page.get_pixmap()

        fmt = (
            QtGui.QImage.Format_RGBA8888
            if pix.alpha
            else QtGui.QImage.Format_RGB888
        )
        image = QtGui.QImage(pix.samples, pix.width,
                             pix.height, pix.stride, fmt).copy()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(image))

        text = (page.get_text("text") or "").strip()
        self.text_view.setPlainText(text)
        self.text_view.moveCursor(QtGui.QTextCursor.Start)
        self.page_changed.emit(self._current_page, self._doc.page_count)

    def _change_page(self, delta: int) -> None:
        if self._doc is None:
            return
        target = self._current_page + delta
        if target < 0 or target >= self._doc.page_count:
            return
        self._current_page = target
        self._render_current_page()

    def _set_zoom_factor(self, factor: float) -> None:
        if self._use_web_engine and self._stack.currentWidget() is self._web_container:
            return
        self._zoom = max(0.5, min(3.0, factor))
        self._render_current_page()

    def _reset_zoom(self) -> None:
        self._set_zoom_factor(1.5)

    def _on_text_selection_changed(self) -> None:
        pass

    def _show_context_menu(self, position: QtCore.QPoint) -> None:
        menu = self.text_view.createStandardContextMenu()
        menu.insertSeparator(menu.actions()[0] if menu.actions() else None)
        speak_action = QtWidgets.QAction("Speak selection", self)
        speak_action.setEnabled(self._has_selection() and not self._speaking)
        speak_action.triggered.connect(self._request_speak_selection)
        menu.insertAction(
            menu.actions()[0] if menu.actions() else None, speak_action)
        menu.exec_(self.text_view.mapToGlobal(position))

    def _show_web_context_menu(self, position: QtCore.QPoint) -> None:
        if self._web_view is None:
            return
        page = self._web_view.page()
        global_pos = self._web_view.mapToGlobal(position)

        def show_menu(selection: object) -> None:
            selected_text = (
                str(selection) if selection is not None else "").strip()
            if not selected_text:
                selected_text = (self._web_view.selectedText() or "").strip()
            if selected_text:
                self._update_selection_cache(selected_text)
            else:
                self._clear_selection_cache()
            menu = page.createStandardContextMenu()
            menu.insertSeparator(menu.actions()[0] if menu.actions() else None)
            speak_action = QtWidgets.QAction("Speak selection", self)
            speak_action.setEnabled(not self._speaking)
            speak_action.triggered.connect(
                lambda: self._speak_web_selection(selected_text)
            )
            menu.insertAction(
                menu.actions()[0] if menu.actions() else None, speak_action
            )
            menu.exec_(global_pos)

        # Use DOM selection for PDF.js (QWebEngineView.selectedText can be empty).
        try:
            page.runJavaScript(
                """(() => {
  const sel = window.getSelection ? window.getSelection() : null;
  if (!sel || sel.isCollapsed) return '';
  return sel.toString();
})()""",
                show_menu,
            )
        except Exception:
            show_menu(None)

    def _speak_web_selection(self, selected_text: str = "") -> None:
        if self._speaking:
            return
        if self._web_view is None:
            return

        def start_with_chunks(payload: object) -> None:
            texts: list[str] = []
            indices: list[int] = []
            if isinstance(payload, dict):
                raw_texts = payload.get("texts") or []
                raw_indices = payload.get("indices") or []
                try:
                    texts = [str(t).strip()
                             for t in raw_texts if str(t).strip()]
                    indices = [int(i) for i in raw_indices]
                except Exception:
                    texts = []
                    indices = []
            if texts and indices and len(texts) == len(indices):
                self._web_selected_span_text = {
                    idx: text for text, idx in zip(texts, indices)
                }
                groups, sentences = self._group_web_spans_into_sentences(
                    texts, indices
                )
                if groups and sentences and len(groups) == len(sentences):
                    self._web_sentence_span_groups = groups
                    self._highlight_mode = "web-sentence"
                    self._speak_text(" ".join(sentences), chunks=sentences)
                    return
            # Fallback: speak selection as a whole.
            self._highlight_mode = "web"
            cleaned = (selected_text or "").strip()
            if cleaned:
                self._speak_text(cleaned)
                return
            if self._selection_cache and (time.monotonic() - self._selection_cache_time) < 2.0:
                self._speak_text(self._selection_cache)
                return
            self._speak_from_clipboard()

        try:
            self._web_view.page().runJavaScript(
                """(() => {
  const result = {texts: [], indices: []};
  const spans = window.__annolidSpans || [];
  const sel = window.getSelection ? window.getSelection() : null;
  if (!sel || sel.isCollapsed || !spans.length) return result;
  const ranges = [];
  for (let i = 0; i < sel.rangeCount; i++) ranges.push(sel.getRangeAt(i));
  spans.forEach((span, idx) => {
    for (const r of ranges) {
      if (r.intersectsNode(span)) {
        const text = span.textContent || "";
        if (text.trim().length) {
          result.texts.push(text);
          result.indices.push(idx);
        }
        break;
      }
    }
  });
  return result;
})()""",
                start_with_chunks,
            )
        except Exception:
            start_with_chunks({})

    def _speak_from_clipboard(self) -> None:
        if self._web_view is None:
            return
        try:
            clipboard = QtWidgets.QApplication.clipboard()
        except Exception:
            return
        original = clipboard.text()
        try:
            self._web_view.page().triggerAction(
                QtWebEngineWidgets.QWebEnginePage.Copy
            )
        except Exception:
            return

        def consume() -> None:
            text = (clipboard.text() or "").strip()
            if text and text != original:
                self._update_selection_cache(text)
                self._speak_text(text)
            else:
                QtWidgets.QToolTip.showText(
                    QtGui.QCursor.pos(),
                    "No selection detected.",
                )
            # Restore clipboard to reduce surprises.
            if original is not None:
                clipboard.setText(original)

        QtCore.QTimer.singleShot(120, consume)

    # ---- Reader controls -------------------------------------------------
    def reader_availability(self) -> tuple[bool, str]:
        if not (self._use_web_engine and self._web_view is not None):
            return False, "Reader requires the embedded web view."
        if not self._pdfjs_active:
            return False, "Reader requires PDF.js mode."
        if not (_WEBCHANNEL_AVAILABLE and self._web_channel is not None):
            return False, "Qt WebChannel is unavailable."
        return True, ""

    def reader_state(self) -> tuple[str, int, int]:
        return self._reader_state, self._reader_current_index, self._reader_total

    def reader_enabled(self) -> bool:
        return bool(self._reader_enabled)

    def _emit_reader_availability(self) -> None:
        available, reason = self.reader_availability()
        try:
            self.reader_availability_changed.emit(available, reason)
        except Exception:
            pass

    def _apply_reader_enabled_to_web(self) -> None:
        if self._web_view is None or not self._pdfjs_active:
            return
        enabled_value = "true" if self._reader_enabled else "false"
        try:
            self._web_view.page().runJavaScript(
                "window.__annolidSetReaderEnabled && "
                f"window.__annolidSetReaderEnabled({enabled_value});"
            )
        except Exception:
            pass

    def set_reader_enabled(self, enabled: bool) -> None:
        self._reader_enabled = bool(enabled)
        if not self._reader_enabled:
            self.stop_reader()
        self._apply_reader_enabled_to_web()

    def toggle_reader_pause_resume(self) -> None:
        if self._reader_state == "reading":
            self._pause_reader()
        elif self._reader_state == "paused":
            self._resume_reader()

    def stop_reader(self) -> None:
        if self._reader_state in {"reading", "paused"}:
            self._reader_stop_requested = True
            self._reader_pause_requested = False
            self._cancel_speaking()
            if not self._speaking:
                self._clear_highlight()
                self._reset_reader_state()
        else:
            self._reset_reader_state()

    def reader_next_sentence(self) -> None:
        """Advance to the next sentence in the reader queue."""
        if not self._reader_queue:
            return
        self._jump_reader_to_index(self._reader_current_index + 1)

    def reader_prev_sentence(self) -> None:
        """Go back to the previous sentence in the reader queue."""
        if not self._reader_queue:
            return
        self._jump_reader_to_index(self._reader_current_index - 1)

    def _jump_reader_to_index(self, global_index: int) -> None:
        """Seek to an absolute reader index and restart playback."""
        if not self._reader_queue:
            return
        max_global = self._reader_queue_offset + len(self._reader_queue) - 1
        clamped = max(self._reader_queue_offset,
                      min(max_global, int(global_index)))
        local_index = clamped - self._reader_queue_offset
        self._start_reader_from_local_index(local_index)

    def _pause_reader(self) -> None:
        if self._reader_state != "reading":
            return
        if not self._speaking:
            self._reader_state = "paused"
            self._reader_pause_requested = False
            try:
                self.reader_state_changed.emit(
                    self._reader_state,
                    self._reader_current_index,
                    self._reader_total,
                )
            except Exception:
                pass
            return
        self._reader_pause_requested = True
        self._reader_stop_requested = False
        self._cancel_speaking()

    def _resume_reader(self) -> None:
        if self._reader_state != "paused":
            return
        local_index = max(0, self._reader_current_index -
                          self._reader_queue_offset)
        self._start_reader_from_local_index(local_index)

    def _cancel_speaking(self) -> None:
        if self._active_speak_token is not None:
            self._active_speak_token.cancelled = True
        try:
            from annolid.utils.audio_playback import stop_audio_playback

            stop_audio_playback()
        except Exception:
            pass

    def _reset_reader_state(self) -> None:
        self._reader_state = "idle"
        self._reader_queue = []
        self._reader_spans = []
        self._reader_pages = []
        self._reader_queue_offset = 0
        self._reader_current_index = 0
        self._reader_total = 0
        self._reader_chunk_base = 0
        self._reader_pause_requested = False
        self._reader_stop_requested = False
        self._reader_pending_restart = None
        self._web_sentence_span_groups = []
        try:
            self.reader_state_changed.emit(
                self._reader_state, self._reader_current_index, self._reader_total
            )
        except Exception:
            pass

    def _handle_reader_click(self, payload: object) -> None:
        if not self._reader_enabled:
            return
        available, reason = self.reader_availability()
        if not available:
            QtWidgets.QToolTip.showText(
                QtGui.QCursor.pos(),
                reason or "Reader is unavailable.",
            )
            return
        if not isinstance(payload, dict):
            return
        raw_sentences = payload.get("sentences")
        use_sentences = isinstance(
            raw_sentences, list) and len(raw_sentences) > 0
        items = raw_sentences if use_sentences else payload.get("paragraphs")
        if not isinstance(items, list):
            return
        texts: list[str] = []
        spans_list: list[list[int]] = []
        pages: list[int] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            texts.append(text)
            raw_spans = entry.get("spans") or []
            spans: list[int] = []
            if isinstance(raw_spans, list):
                for item in raw_spans:
                    try:
                        spans.append(int(item))
                    except Exception:
                        continue
            spans_list.append(spans)
            try:
                pages.append(int(entry.get("pageNum", 0)))
            except Exception:
                pages.append(0)

        if not texts:
            return

        try:
            start_index = int(
                payload.get(
                    "sentenceStartIndex" if use_sentences else "startIndex", 0)
            )
        except Exception:
            start_index = 0
        local_start_index = 0
        if use_sentences:
            try:
                local_start_index = int(payload.get(
                    "sentenceLocalStartIndex", 0) or 0)
            except Exception:
                local_start_index = 0
        try:
            total = int(payload.get(
                "sentenceTotal" if use_sentences else "total", 0))
        except Exception:
            total = 0
        if total <= 0:
            total = start_index + len(texts)

        self._reader_queue = texts
        self._reader_spans = spans_list
        self._reader_pages = pages
        self._reader_queue_offset = max(
            0, start_index) if not use_sentences else 0
        self._reader_total = max(
            total, self._reader_queue_offset + len(texts))
        self._reader_current_index = self._reader_queue_offset + \
            max(0, local_start_index)
        self._reader_chunk_base = 0
        self._reader_pause_requested = False
        self._reader_stop_requested = False
        self._reader_pending_restart = None
        self._web_sentence_span_groups = spans_list if use_sentences else []
        self._reader_state = "reading"
        try:
            self.reader_state_changed.emit(
                self._reader_state, self._reader_current_index, self._reader_total
            )
        except Exception:
            pass
        self._highlight_mode = "web-sentence" if use_sentences else "web-paragraph"
        start_local = 0
        if use_sentences:
            start_local = max(
                0, min(local_start_index, len(self._reader_queue) - 1))
        self._start_reader_from_local_index(start_local)

    def _start_reader_from_local_index(self, local_index: int) -> None:
        if local_index < 0 or local_index >= len(self._reader_queue):
            self._reset_reader_state()
            return
        if self._speaking:
            self._reader_pending_restart = local_index
            self._cancel_speaking()
            return
        self._reader_chunk_base = local_index
        self._reader_current_index = self._reader_queue_offset + local_index
        self._reader_state = "reading"
        self._reader_pause_requested = False
        self._reader_stop_requested = False
        # Pre-scroll to the target sentence/page so manual navigation does not jump back.
        if self._web_view is not None and self._pdfjs_active:
            try:
                queue_idx = local_index
                spans: list[int] = []
                if 0 <= queue_idx < len(self._web_sentence_span_groups):
                    spans = self._web_sentence_span_groups[queue_idx] or []
                page_num = (
                    self._reader_pages[queue_idx]
                    if 0 <= queue_idx < len(self._reader_pages)
                    else 0
                )
                next_page = (
                    self._reader_pages[queue_idx + 1]
                    if 0 <= queue_idx + 1 < len(self._reader_pages)
                    else 0
                )
                sentence_text = (
                    self._reader_queue[queue_idx]
                    if 0 <= queue_idx < len(self._reader_queue)
                    else ""
                )
                if spans:
                    if page_num > 0:
                        self._web_view.page().runJavaScript(
                            "window.__annolidEnsureRenderedThrough && "
                            f"window.__annolidEnsureRenderedThrough({int(page_num)});"
                        )
                    self._web_view.page().runJavaScript(
                        "window.__annolidHighlightSentenceIndices && "
                        f"window.__annolidHighlightSentenceIndices({spans})"
                    )
                    self._web_view.page().runJavaScript(
                        "window.__annolidScrollToSentence && "
                        f"window.__annolidScrollToSentence({spans}, {int(page_num)})"
                    )
                elif page_num > 0:
                    self._web_view.page().runJavaScript(
                        "window.__annolidEnsureRenderedThrough && "
                        f"window.__annolidEnsureRenderedThrough({int(page_num)});"
                    )
                    self._web_view.page().runJavaScript(
                        "window.__annolidHighlightSentenceByText && "
                        f"window.__annolidHighlightSentenceByText({int(page_num)}, {json.dumps(sentence_text)})"
                    )
                if next_page and next_page != page_num:
                    try:
                        self._web_view.page().runJavaScript(
                            "window.__annolidEnsureRenderedThrough && "
                            f"window.__annolidEnsureRenderedThrough({int(next_page)});"
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            self.reader_state_changed.emit(
                self._reader_state, self._reader_current_index, self._reader_total
            )
        except Exception:
            pass
        chunks = self._reader_queue[local_index:]
        self._highlight_mode = (
            "web-sentence" if self._web_sentence_span_groups else "web-paragraph"
        )
        self._speak_text(" ".join(chunks), chunks=chunks)

    def _scroll_pdfjs_to_page(self, page_num: int) -> None:
        if self._web_view is None or not self._pdfjs_active:
            return
        try:
            self._web_view.page().runJavaScript(
                "window.__annolidScrollToPage && "
                f"window.__annolidScrollToPage({int(page_num)})"
            )
        except Exception:
            pass

    def _update_selection_cache(self, text: str) -> None:
        self._selection_cache = text
        self._selection_cache_time = time.monotonic()

    def _clear_selection_cache(self) -> None:
        self._selection_cache = ""
        self._selection_cache_time = 0.0

    def _extract_text_sentence_spans(
        self, cursor: QtGui.QTextCursor
    ) -> tuple[list[tuple[int, int]], list[str]]:
        import re

        selected_raw = cursor.selectedText()
        if not selected_raw:
            return [], []
        base = cursor.selectionStart()
        spans: list[tuple[int, int]] = []
        chunks: list[str] = []
        pattern = re.compile(r".+?(?:[.!?。！？]+|$)", re.DOTALL)
        for match in pattern.finditer(selected_raw):
            segment = match.group(0)
            if not segment or not segment.strip():
                continue
            leading = len(segment) - len(segment.lstrip())
            trailing = len(segment) - len(segment.rstrip())
            start = base + match.start() + leading
            end = base + match.end() - trailing
            if start >= end:
                continue
            cleaned = re.sub(
                r"\s+", " ", segment.replace("\u2029", " ")).strip()
            if not cleaned:
                continue
            spans.append((start, end))
            chunks.append(cleaned)
        return spans, chunks

    def _group_web_spans_into_sentences(
        self, texts: list[str], indices: list[int]
    ) -> tuple[list[list[int]], list[str]]:
        import re

        groups: list[list[int]] = []
        sentences: list[str] = []
        current_text: list[str] = []
        current_indices: list[int] = []
        end_punct = re.compile(r"[.!?。！？]")
        for text, idx in zip(texts, indices):
            cleaned = str(text).strip()
            if not cleaned:
                continue
            current_text.append(cleaned)
            current_indices.append(idx)
            if end_punct.search(cleaned):
                sentence = re.sub(r"\s+", " ", " ".join(current_text)).strip()
                if sentence:
                    groups.append(list(current_indices))
                    sentences.append(sentence)
                current_text = []
                current_indices = []
        if current_indices:
            sentence = re.sub(r"\s+", " ", " ".join(current_text)).strip()
            if sentence:
                groups.append(list(current_indices))
                sentences.append(sentence)
        return groups, sentences

    def _has_selection(self) -> bool:
        cursor = self.text_view.textCursor()
        return bool(cursor and cursor.hasSelection())

    def _selected_text(self) -> str:
        cursor = self.text_view.textCursor()
        if not cursor or not cursor.hasSelection():
            return ""
        return cursor.selectedText().replace("\u2029", "\n").strip()

    def _request_speak_selection(self) -> None:
        text = self._selected_text()
        if not text:
            return
        cursor = self.text_view.textCursor()
        if cursor and cursor.hasSelection():
            self._selection_anchor_start = cursor.selectionStart()
            self._selection_anchor_end = cursor.selectionEnd()
            self._highlight_mode = "text-sentence"
            self._text_sentence_spans, chunks = self._extract_text_sentence_spans(
                cursor
            )
        else:
            chunks = None
            self._highlight_mode = "text"
        self._speak_text(text, chunks=chunks)

    def _speak_text(self, text: str, *, chunks: Optional[list[str]] = None) -> None:
        """Kick off background speech for the provided text."""
        cleaned = (text or "").replace("\u2029", "\n").strip()
        # Fix common PDF hyphenation artifacts (e.g., "synergis- tic" -> "synergistic")
        # while preserving likely intentional hyphens for short prefixes (e.g., "non- linear" -> "non-linear").
        cleaned = cleaned.replace("\u00ad", "")
        try:
            import re

            # Merge line-wrapped words that left a hyphen plus whitespace between letters.
            cleaned = re.sub(
                r"([A-Za-z]{2,})[-\u2010\u2011\u2012\u2013]\s+([A-Za-z]{2,})",
                r"\1\2",
                cleaned,
            )
        except Exception:
            pass
        if not cleaned:
            return
        if self._speaking:
            return
        self._speaking = True
        token = _SpeakToken()
        self._active_speak_token = token
        if self._highlight_mode in {"text", "web"}:
            self._start_highlight()
        # Keep the public signal for downstream integrations if needed.
        self.selection_ready.emit(cleaned)
        settings = load_tts_settings()
        defaults = default_tts_settings()
        merged = {
            "voice": settings.get("voice", defaults["voice"]),
            "lang": settings.get("lang", defaults["lang"]),
            "speed": settings.get("speed", defaults["speed"]),
        }
        self._thread_pool.start(_SpeakTextTask(
            self, cleaned, merged, chunks=chunks, token=token))

    @QtCore.Slot()
    def _on_speak_finished(self) -> None:
        self._speaking = False
        self._active_speak_token = None
        self._clear_highlight()
        if self._reader_pending_restart is not None:
            pending = self._reader_pending_restart
            self._reader_pending_restart = None
            self._start_reader_from_local_index(pending)
            return
        if self._reader_state in {"reading", "paused"}:
            if self._reader_stop_requested:
                self._reset_reader_state()
                return
            if self._reader_pause_requested:
                self._reader_state = "paused"
                self._reader_pause_requested = False
                try:
                    self.reader_state_changed.emit(
                        self._reader_state,
                        self._reader_current_index,
                        self._reader_total,
                    )
                except Exception:
                    pass
                return
            self._reset_reader_state()

    @QtCore.Slot(int)
    def _on_speak_chunk(self, index: int) -> None:
        if self._highlight_mode == "text-sentence":
            self._stop_word_highlight()
            if 0 <= index < len(self._text_sentence_spans):
                start, end = self._text_sentence_spans[index]
                self._active_text_sentence_span = (start, end)
                self._set_text_tts_highlight((start, end), None)
                self._scroll_text_to_position(start)
            return
        if self._highlight_mode == "web-sentence":
            self._stop_word_highlight()
            global_idx = self._reader_chunk_base + index
            span_indices: list[int] = []
            if 0 <= global_idx < len(self._web_sentence_span_groups):
                span_indices = self._web_sentence_span_groups[global_idx] or []
            page_num = (
                self._reader_pages[global_idx]
                if 0 <= global_idx < len(self._reader_pages)
                else 0
            )
            next_page = (
                self._reader_pages[global_idx + 1]
                if 0 <= global_idx + 1 < len(self._reader_pages)
                else 0
            )
            sentence_text = (
                self._reader_queue[global_idx]
                if 0 <= global_idx < len(self._reader_queue)
                else ""
            )
            if self._web_view is not None:
                try:
                    if page_num > 0:
                        self._web_view.page().runJavaScript(
                            "window.__annolidEnsureRenderedThrough && "
                            f"window.__annolidEnsureRenderedThrough({int(page_num)});"
                        )
                    if span_indices:
                        self._web_view.page().runJavaScript(
                            "window.__annolidHighlightSentenceIndices && "
                            f"window.__annolidHighlightSentenceIndices({span_indices})"
                        )
                        self._web_view.page().runJavaScript(
                            "window.__annolidScrollToSentence && "
                            f"window.__annolidScrollToSentence({span_indices}, {int(page_num)})"
                        )
                    elif page_num > 0:
                        self._web_view.page().runJavaScript(
                            "window.__annolidHighlightSentenceByText && "
                            f"window.__annolidHighlightSentenceByText({int(page_num)}, {json.dumps(sentence_text)})"
                        )
                    if next_page and next_page != page_num:
                        self._web_view.page().runJavaScript(
                            "window.__annolidEnsureRenderedThrough && "
                            f"window.__annolidEnsureRenderedThrough({int(next_page)});"
                        )
                except Exception:
                    pass
            self._reader_current_index = self._reader_queue_offset + global_idx
            try:
                self.reader_state_changed.emit(
                    self._reader_state,
                    self._reader_current_index,
                    self._reader_total,
                )
            except Exception:
                pass
            return
        if self._highlight_mode == "web-paragraph":
            self._stop_word_highlight()
            local_index = self._reader_chunk_base + index
            if 0 <= local_index < len(self._reader_queue):
                self._reader_current_index = self._reader_queue_offset + local_index
                try:
                    self.reader_state_changed.emit(
                        self._reader_state,
                        self._reader_current_index,
                        self._reader_total,
                    )
                except Exception:
                    pass
                if local_index < len(self._reader_pages):
                    page_num = self._reader_pages[local_index]
                    if page_num > 0:
                        self._scroll_pdfjs_to_page(page_num)
                spans = (
                    self._reader_spans[local_index]
                    if local_index < len(self._reader_spans)
                    else []
                )
                if self._web_view is not None:
                    try:
                        if spans:
                            self._web_view.page().runJavaScript(
                                "window.__annolidHighlightParagraphIndices && "
                                f"window.__annolidHighlightParagraphIndices({spans})"
                            )
                        else:
                            page_num = (
                                self._reader_pages[local_index]
                                if local_index < len(self._reader_pages)
                                else 0
                            )
                            if page_num <= 0:
                                return
                            text = self._reader_queue[local_index]
                            self._web_view.page().runJavaScript(
                                "window.__annolidHighlightParagraphByText && "
                                f"window.__annolidHighlightParagraphByText({int(page_num)}, {json.dumps(text)})"
                            )
                    except Exception:
                        pass
            return

    @QtCore.Slot(int, int)
    def _on_speak_chunk_timing(self, index: int, duration_ms: int) -> None:
        if duration_ms <= 0:
            return
        if self._highlight_mode == "text-sentence":
            if not (0 <= index < len(self._text_sentence_spans)):
                return
            start, end = self._text_sentence_spans[index]
            self._active_text_sentence_span = (start, end)
            word_spans = self._split_text_range_into_words(start, end)
            if not word_spans:
                return
            weights = [max(1, word_end - word_start)
                       for word_start, word_end in word_spans]
            units, durations = self._build_weighted_timing(
                word_spans, weights, duration_ms
            )
            if not units or not durations:
                return
            self._word_highlight_units = list(units)
            self._word_highlight_durations_ms = list(durations)
            self._word_highlight_index = 0
            timer = QtCore.QTimer(self)
            timer.setSingleShot(False)
            timer.timeout.connect(self._advance_word_highlight)
            self._word_highlight_timer = timer
            self._apply_word_highlight_unit(self._word_highlight_units[0])
            timer.start(self._word_highlight_durations_ms[0])
            return
        if self._highlight_mode == "web-sentence":
            global_idx = self._reader_chunk_base + index
            if not (0 <= global_idx < len(self._web_sentence_span_groups)):
                return
            span_indices = self._web_sentence_span_groups[global_idx]
            if not span_indices:
                return
            weights = [
                max(1, len(self._web_selected_span_text.get(idx, "").strip()))
                for idx in span_indices
            ]
            units, durations = self._build_weighted_timing(
                span_indices, weights, duration_ms
            )
            if not units or not durations:
                return
            self._word_highlight_units = list(units)
            self._word_highlight_durations_ms = list(durations)
            self._word_highlight_index = 0
            timer = QtCore.QTimer(self)
            timer.setSingleShot(False)
            timer.timeout.connect(self._advance_word_highlight)
            self._word_highlight_timer = timer
            self._apply_word_highlight_unit(self._word_highlight_units[0])
            timer.start(self._word_highlight_durations_ms[0])
            return
        if self._highlight_mode == "web-paragraph":
            return

    def _start_highlight(self) -> None:
        if self._highlight_mode == "text":
            self._apply_text_highlight()
        elif self._highlight_mode == "web":
            self._apply_web_highlight()

    def _apply_text_highlight(self) -> None:
        if self._selection_anchor_start is None or self._selection_anchor_end is None:
            return
        cursor = self.text_view.textCursor()
        cursor.setPosition(self._selection_anchor_start)
        cursor.setPosition(self._selection_anchor_end,
                           QtGui.QTextCursor.KeepAnchor)
        self.text_view.setTextCursor(cursor)

    def _apply_web_highlight(self) -> None:
        if self._web_view is None:
            return
        try:
            self._web_view.page().runJavaScript(
                "window.__annolidHighlightSelection && window.__annolidHighlightSelection()"
            )
        except Exception:
            pass

    def _clear_highlight(self) -> None:
        self._stop_word_highlight()
        if self._web_view is not None and self._highlight_mode in {"web", "web-sentence", "web-paragraph"}:
            try:
                self._web_view.page().runJavaScript(
                    "window.__annolidClearHighlight && window.__annolidClearHighlight()"
                )
            except Exception:
                pass
        self._highlight_mode = None
        self._text_sentence_spans = []
        self._web_sentence_span_groups = []
        self._web_selected_span_text = {}
        self._active_text_sentence_span = None

    def _stop_word_highlight(self) -> None:
        timer = self._word_highlight_timer
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._word_highlight_timer = None
        self._word_highlight_units = []
        self._word_highlight_durations_ms = []
        self._word_highlight_index = 0
        try:
            self.text_view.setExtraSelections([])
        except Exception:
            pass
        if self._web_view is not None and self._highlight_mode in {"web", "web-sentence", "web-paragraph"}:
            try:
                self._web_view.page().runJavaScript(
                    "window.__annolidClearWordHighlight && window.__annolidClearWordHighlight()"
                )
            except Exception:
                pass

    def _advance_word_highlight(self) -> None:
        timer = self._word_highlight_timer
        if timer is None:
            return
        self._word_highlight_index += 1
        if self._word_highlight_index >= len(self._word_highlight_units):
            try:
                timer.stop()
            except Exception:
                pass
            self._word_highlight_timer = None
            return
        unit = self._word_highlight_units[self._word_highlight_index]
        self._apply_word_highlight_unit(unit)
        if self._word_highlight_index < len(self._word_highlight_durations_ms):
            timer.setInterval(
                max(1, int(
                    self._word_highlight_durations_ms[self._word_highlight_index]))
            )

    def _apply_word_highlight_unit(self, unit: object) -> None:
        if self._highlight_mode == "text-sentence":
            sentence = self._active_text_sentence_span
            if sentence is None:
                return
            if not (isinstance(unit, tuple) and len(unit) == 2):
                return
            word_start, word_end = unit
            self._set_text_tts_highlight(sentence, (word_start, word_end))
            self._scroll_text_to_position(word_start)
            return
        if self._highlight_mode == "web-sentence":
            if self._web_view is None:
                return
            if not isinstance(unit, int):
                return
            try:
                self._web_view.page().runJavaScript(
                    "window.__annolidHighlightWordIndex && "
                    f"window.__annolidHighlightWordIndex({unit})"
                )
            except Exception:
                pass

    def _scroll_text_to_position(self, position: int) -> None:
        try:
            cursor = self.text_view.textCursor()
            cursor.setPosition(position)
            self.text_view.setTextCursor(cursor)
            self.text_view.ensureCursorVisible()
        except Exception:
            pass

    def _set_text_tts_highlight(
        self,
        sentence_span: Optional[tuple[int, int]],
        word_span: Optional[tuple[int, int]],
    ) -> None:
        try:
            doc = self.text_view.document()
            selections: list[QtWidgets.QTextEdit.ExtraSelection] = []
            if sentence_span is not None:
                selections.append(self._make_text_extra_selection(
                    doc, sentence_span[0], sentence_span[1], QtGui.QColor(
                        255, 210, 80, 90)
                ))
            if word_span is not None:
                selections.append(self._make_text_extra_selection(
                    doc, word_span[0], word_span[1], QtGui.QColor(
                        255, 160, 0, 160)
                ))
            self.text_view.setExtraSelections(selections)
        except Exception:
            pass

    @staticmethod
    def _make_text_extra_selection(
        document: QtGui.QTextDocument, start: int, end: int, color: QtGui.QColor
    ) -> QtWidgets.QTextEdit.ExtraSelection:
        cursor = QtGui.QTextCursor(document)
        cursor.setPosition(start)
        cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
        selection = QtWidgets.QTextEdit.ExtraSelection()
        selection.cursor = cursor
        fmt = QtGui.QTextCharFormat()
        fmt.setBackground(QtGui.QBrush(color))
        selection.format = fmt
        return selection

    def _split_text_range_into_words(
        self, start: int, end: int
    ) -> list[tuple[int, int]]:
        import re

        if start >= end:
            return []
        doc = self.text_view.document()
        cursor = QtGui.QTextCursor(doc)
        cursor.setPosition(start)
        cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
        segment = cursor.selectedText()
        if not segment:
            return []
        word_spans: list[tuple[int, int]] = []
        punct = ".,!?;:\"'“”‘’()[]{}<>"
        for match in re.finditer(r"\S+", segment):
            token = match.group(0)
            if not token:
                continue
            left_trim = len(token) - len(token.lstrip(punct))
            right_trim = len(token) - len(token.rstrip(punct))
            w_start = start + match.start() + left_trim
            w_end = start + match.end() - right_trim
            if w_start >= w_end:
                continue
            word_spans.append((w_start, w_end))
        return word_spans

    def _build_weighted_timing(
        self, units: list[object], weights: list[int], total_ms: int
    ) -> tuple[list[object], list[int]]:
        if total_ms <= 0 or not units or len(units) != len(weights):
            return [], []
        weights = [max(1, int(w)) for w in weights]
        min_interval_ms = 60
        max_units = max(1, int(total_ms // min_interval_ms))
        if len(units) > max_units:
            units, weights = self._downsample_units(units, weights, max_units)
        durations = self._allocate_weighted_durations_ms(total_ms, weights)
        if len(durations) != len(units):
            return [], []
        return units, durations

    @staticmethod
    def _downsample_units(
        units: list[object], weights: list[int], target_len: int
    ) -> tuple[list[object], list[int]]:
        if target_len <= 0 or target_len >= len(units):
            return units, weights
        if target_len == 1:
            return [units[0]], [weights[0]]
        result_units: list[object] = []
        result_weights: list[int] = []
        last_index = len(units) - 1
        for i in range(target_len):
            pos = int(round(i * last_index / (target_len - 1)))
            result_units.append(units[pos])
            result_weights.append(weights[pos])
        return result_units, result_weights

    @staticmethod
    def _allocate_weighted_durations_ms(total_ms: int, weights: list[int]) -> list[int]:
        if total_ms <= 0 or not weights:
            return []
        weights = [max(1, int(w)) for w in weights]
        total_weight = sum(weights)
        if total_weight <= 0:
            return [max(1, int(total_ms // len(weights))) for _ in weights]
        raw = [total_ms * (w / total_weight) for w in weights]
        durations = [max(1, int(x)) for x in raw]
        remainder = int(total_ms) - sum(durations)
        if remainder == 0:
            return durations
        fractional = [x - int(x) for x in raw]
        order = sorted(
            range(len(weights)),
            key=lambda i: fractional[i],
            reverse=remainder > 0,
        )
        remaining = abs(remainder)
        idx = 0
        while remaining > 0 and order:
            i = order[idx % len(order)]
            if remainder > 0:
                durations[i] += 1
                remaining -= 1
            else:
                if durations[i] > 1:
                    durations[i] -= 1
                    remaining -= 1
            idx += 1
        return durations

    # ---- Public helpers for external controls ---------------------------------
    def next_page(self) -> None:
        self._change_page(1)

    def previous_page(self) -> None:
        self._change_page(-1)

    def page_count(self) -> int:
        if self._doc is None:
            return 0
        try:
            return int(self._doc.page_count)
        except Exception:
            return 0

    def current_page_index(self) -> int:
        return int(self._current_page)

    def set_zoom_percent(self, percent: float) -> None:
        self._set_zoom_factor(float(percent) / 100.0)

    def current_zoom_percent(self) -> int:
        return int(round(self._zoom * 100))

    def reset_zoom(self) -> None:
        self._reset_zoom()

    def rotate_clockwise(self) -> None:
        if self._use_web_engine and self._web_view is not None and self._pdfjs_active:
            try:
                self._web_view.page().runJavaScript(
                    "if (window.__annolidRotate) { window.__annolidRotate(90); }"
                )
            except Exception:
                pass
            return
        if self._doc is None:
            return
        self._rotation = (self._rotation + 90) % 360
        self._render_current_page()

    def pdfjs_active(self) -> bool:
        return bool(self._pdfjs_active)

    def web_pdf_capable(self) -> bool:
        return bool(self._web_pdf_capable)

    def force_pdfjs_enabled(self) -> bool:
        return bool(self._force_pdfjs)

    def set_force_pdfjs(self, enabled: bool) -> None:
        desired = bool(enabled)
        if desired == self._force_pdfjs:
            return
        self._force_pdfjs = desired
        if self._pdf_path is not None:
            try:
                self.load_pdf(str(self._pdf_path))
            except Exception:
                pass

    def controls_enabled(self) -> bool:
        return not self._web_mode_active

    def is_web_mode(self) -> bool:
        return bool(self._use_web_engine and self._web_mode_active)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI cleanup
        if self._doc is not None:
            self._doc.close()
        self._doc = None
        super().closeEvent(event)

    def _set_controls_for_web(self, web_mode: bool) -> None:
        """Inform listeners whether fallback controls should be enabled."""
        self._web_mode_active = bool(web_mode)
        try:
            self.controls_enabled_changed.emit(not self._web_mode_active)
        except Exception:
            pass

    # ---- Bookmarks -----------------------------------------------------------
    def _load_bookmarks_from_path(self, path: Path) -> None:
        """Load outlines using PyMuPDF; only used for the embed plugin path."""
        bookmarks: list[dict[str, object]] = []
        try:
            import fitz  # type: ignore[import]

            with fitz.open(str(path)) as doc:
                toc = doc.get_toc(simple=True) or []
                for entry in toc:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                        continue
                    level, title, page = entry[0], entry[1], entry[2]
                    try:
                        bookmarks.append(
                            {
                                "level": int(level),
                                "title": str(title),
                                "page": max(0, int(page) - 1),
                            }
                        )
                    except Exception:
                        continue
        except Exception:
            bookmarks = []
        self._bookmarks = bookmarks
        try:
            self.bookmarks_changed.emit(bookmarks)
        except Exception:
            pass

    def _clear_bookmarks(self) -> None:
        self._bookmarks = []
        try:
            self.bookmarks_changed.emit([])
        except Exception:
            pass

    def bookmarks(self) -> list[dict[str, object]]:
        return list(self._bookmarks)


class _SpeakTextTask(QtCore.QRunnable):
    """Background task to convert text to speech and play it."""

    def __init__(
        self,
        widget: PdfViewerWidget,
        text: str,
        tts_settings: Dict[str, object],
        chunks: Optional[list[str]] = None,
        token: Optional[_SpeakToken] = None,
    ) -> None:
        super().__init__()
        self.widget = widget
        self.text = text
        self.tts_settings = tts_settings
        self.chunks = chunks
        self.token = token

    def run(self) -> None:  # pragma: no cover - involves audio
        try:
            text = (self.text or "").strip()
            if not text:
                return
            chunks = self.chunks or self._chunk_text(text, max_chars=420)
            if not chunks:
                return

            def cancelled() -> bool:
                return bool(self.token is not None and self.token.cancelled)

            def notify_chunk(index: int, duration_ms: int) -> None:
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "_on_speak_chunk",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(int, int(index)),
                )
                if duration_ms > 0:
                    QtCore.QMetaObject.invokeMethod(
                        self.widget,
                        "_on_speak_chunk_timing",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(int, int(index)),
                        QtCore.Q_ARG(int, int(duration_ms)),
                    )

            try:
                from annolid.agents.kokoro_tts import text_to_speech, play_audio

                from concurrent.futures import ThreadPoolExecutor

                voice = str(self.tts_settings.get("voice", "af_sarah"))
                speed = float(self.tts_settings.get("speed", 1.0))
                lang = str(self.tts_settings.get("lang", "en-us"))

                def synthesize(chunk: str) -> tuple[object, int, int]:
                    audio_data = text_to_speech(
                        chunk,
                        voice=voice,
                        speed=speed,
                        lang=lang,
                    )
                    if not audio_data:
                        raise RuntimeError("Kokoro returned no audio")
                    samples, sample_rate = audio_data
                    duration_ms = 0
                    try:
                        duration_ms = int(
                            round((len(samples) / float(sample_rate)) * 1000)
                        )
                    except Exception:
                        duration_ms = 0
                    return samples, int(sample_rate), int(duration_ms)

                # Pipeline synthesis (next chunk) while playback runs (current chunk).
                with ThreadPoolExecutor(max_workers=1) as executor:
                    current_future = executor.submit(synthesize, chunks[0])
                    for idx, chunk in enumerate(chunks):
                        if cancelled():
                            return
                        samples, sample_rate, duration_ms = current_future.result()
                        next_future = None
                        if idx + 1 < len(chunks):
                            next_future = executor.submit(
                                synthesize, chunks[idx + 1]
                            )
                        notify_chunk(idx, duration_ms)
                        if cancelled():
                            return
                        play_audio(samples, sample_rate)
                        if next_future is None:
                            break
                        current_future = next_future
                return
            except Exception:
                pass

            # Fallback to gTTS + in-memory playback
            try:
                from gtts import gTTS
                from pydub import AudioSegment
                import numpy as np
                from io import BytesIO

                lang = str(self.tts_settings.get("lang", "en-us")).lower()
                gtts_lang = lang.split("-")[0] if lang else "en"
                from concurrent.futures import ThreadPoolExecutor

                def synthesize_gtts(chunk: str) -> tuple[object, int, int]:
                    tts = gTTS(text=chunk, lang=gtts_lang)
                    buf = BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    audio = AudioSegment.from_file(buf, format="mp3")
                    samples = np.array(audio.get_array_of_samples())
                    if samples.size == 0:
                        raise RuntimeError("gTTS produced empty audio")
                    if audio.channels == 2:
                        samples = samples.reshape((-1, 2))
                    duration_ms = 0
                    try:
                        duration_ms = int(
                            round((len(samples) / float(audio.frame_rate)) * 1000)
                        )
                    except Exception:
                        duration_ms = 0
                    return samples, int(audio.frame_rate), int(duration_ms)

                with ThreadPoolExecutor(max_workers=1) as executor:
                    current_future = executor.submit(
                        synthesize_gtts, chunks[0])
                    for idx, chunk in enumerate(chunks):
                        if cancelled():
                            return
                        samples, sample_rate, duration_ms = current_future.result()
                        next_future = None
                        if idx + 1 < len(chunks):
                            next_future = executor.submit(
                                synthesize_gtts, chunks[idx + 1]
                            )
                        notify_chunk(idx, duration_ms)
                        if cancelled():
                            return
                        play_audio_buffer(samples, sample_rate, blocking=True)
                        if next_future is None:
                            break
                        current_future = next_future
            except Exception:
                return
        finally:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "_on_speak_finished",
                QtCore.Qt.QueuedConnection,
            )

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 420) -> list[str]:
        import re

        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return []
        # Merge line-wrapped words that left a hyphen plus whitespace between letters.
        cleaned = re.sub(
            r"([A-Za-z]{2,})[-\u2010\u2011\u2012\u2013]\s+([A-Za-z]{2,})",
            r"\1\2",
            cleaned,
        )
        # PDFs sometimes insert whitespace between end punctuation and closing quotes/brackets.
        # Merge that whitespace so sentences don't start with a dangling closer like "”小鹊...".
        cleaned = re.sub(
            r"([.!?。！？])\s+([”’\"'\)\]\}）】》」』〉])",
            r"\1\2",
            cleaned,
        )
        # Split on sentence-ending punctuation, including common trailing closers like Chinese quotes.
        end = r"[.!?。！？]"
        closers = r"[”’\"'\)\]\}）】》」』〉]*"
        pattern = re.compile(rf".+?{end}+{closers}")
        sentences = []
        last_end = 0
        for m in pattern.finditer(cleaned):
            seg = m.group(0)
            if seg:
                sentences.append(seg)
            last_end = m.end()
        tail = cleaned[last_end:].strip()
        if tail:
            sentences.append(tail)
        if not sentences:
            sentences = [cleaned]
        # Fix common skipped patterns around Chinese quote endings like:
        #   "。”说完..." / "。”小鹊..."
        # Merge the quote-terminated sentence with the immediately-following narration so
        # chunk boundaries don't land on unstable PDF text runs.
        try:
            end_quote = re.compile(r"[.!?。！？]+[”’\"']$")
            cjk_lead = re.compile(
                r"^[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
            )
            merged: list[str] = []
            i = 0
            while i < len(sentences):
                current = str(sentences[i]).strip()
                if not current:
                    i += 1
                    continue
                if i + 1 < len(sentences):
                    nxt = str(sentences[i + 1]).strip()
                    if nxt and end_quote.search(current):
                        lead = nxt[0]
                        if cjk_lead.match(lead) or lead in {"（", "【", "“", "‘"}:
                            current = current + nxt
                            i += 2
                            merged.append(current)
                            continue
                merged.append(current)
                i += 1
            if merged:
                sentences = merged
        except Exception:
            pass
        chunks: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                chunks.append(sentence)
            else:
                for i in range(0, len(sentence), max_chars):
                    chunk = sentence[i:i + max_chars].strip()
                    if chunk:
                        chunks.append(chunk)
        return chunks
