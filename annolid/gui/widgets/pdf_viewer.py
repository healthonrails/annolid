from __future__ import annotations

import base64
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
        self._render_current_page()

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
        window.__annolidSplitParagraphIntoSentences = function(para) {{
          try {{
            const spans = (para && Array.isArray(para.spans)) ? para.spans.filter((n) => Number.isInteger(n)) : [];
            if (!spans.length) return [];
            const nodes = window.__annolidSpans || [];
            const parts = [];
            let combined = "";
            spans.forEach((idx) => {{
              const node = nodes[idx];
              if (!node) return;
              const raw = node.textContent || "";
              const text = _annolidNormalizeText(raw);
              if (!text) return;
              const start = combined ? combined.length + 1 : 0;
              if (combined) combined += " ";
              combined += text;
              const end = combined.length;
              parts.push({{ idx, start, end }});
            }});
            const paraText = _annolidNormalizeText(para.text || combined);
            const resolvedPageNum = (parseInt(para.pageNum || para.page || 0, 10) || 0);
            const sentences = [];
            const regex = /[^.!?。！？]+[.!?。！？]*/g;
            let cursor = 0;
            let match;
            while ((match = regex.exec(paraText)) !== null) {{
              const raw = _annolidNormalizeText(match[0]);
              if (!raw) continue;
              const start = paraText.indexOf(raw, cursor);
              if (start < 0) continue;
              const end = start + raw.length;
              cursor = end;
              const group = parts.filter((p) => p.end > start && p.start < end).map((p) => p.idx);
              sentences.push({{
                text: raw,
                spans: group.length ? group : spans.slice(),
                pageNum: resolvedPageNum,
              }});
            }}
            if (!sentences.length && paraText) {{
              sentences.push({{
                text: paraText,
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
	          const spanRect = span.getBoundingClientRect();
	          if (!isFinite(spanRect.width) || !isFinite(spanRect.height) || spanRect.width <= 0 || spanRect.height <= 0) return null;
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
          return String(text || "").replace(/\\s+/g, " ").trim();
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
          entries.sort((a, b) => {{
            if (Math.abs(a.y - b.y) < 1.0) return a.x - b.x;
            return a.y - b.y;
          }});
          const lines = [];
          entries.forEach((entry) => {{
            let line = lines.length ? lines[lines.length - 1] : null;
            const tol = Math.max(2, entry.h * 0.6);
            if (!line || Math.abs(entry.y - line.y) > tol) {{
              line = {{
                y: entry.y,
                h: entry.h,
                spans: [],
                texts: [],
                yMin: entry.y,
                yMax: entry.y + entry.h,
              }};
              lines.push(line);
            }}
            line.spans.push(entry.idx);
            line.texts.push(entry.text);
            line.h = Math.max(line.h, entry.h);
            line.yMin = Math.min(line.yMin, entry.y);
            line.yMax = Math.max(line.yMax, entry.y + entry.h);
          }});

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
              }};
            }} else {{
              current.text = _annolidNormalizeText(current.text + " " + lineText);
              current.spans = current.spans.concat(line.spans);
              current.yMax = Math.max(current.yMax, line.yMax);
            }}
          }});
          if (current) paragraphs.push(current);
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

        function _annolidFindParagraphIndexByPoint(pageNum, y) {{
          const list = window.__annolidParagraphsByPage[String(pageNum)] || [];
          let best = -1;
          let bestDist = Infinity;
          for (let i = 0; i < list.length; i++) {{
            const para = list[i];
            if (para.yMin == null || para.yMax == null) continue;
            if (y >= para.yMin && y <= para.yMax) return i;
            const dist = Math.min(Math.abs(y - para.yMin), Math.abs(y - para.yMax));
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
          const lines = [];
          let currentLine = "";
          for (const item of (textContent.items || [])) {{
            const text = _annolidNormalizeText(item.str || "");
            if (!text) {{
              if (item.hasEOL && currentLine) {{
                lines.push(currentLine);
                currentLine = "";
              }}
              continue;
            }}
            currentLine += (currentLine ? " " : "") + text;
            if (item.hasEOL) {{
              lines.push(currentLine);
              currentLine = "";
            }}
          }}
          if (currentLine) lines.push(currentLine);
          const paragraphs = [];
          let current = "";
          const sentenceEnd = /[.!?。！？]\\s*$/;
          for (const line of lines) {{
            const cleaned = _annolidNormalizeText(line);
            if (!cleaned) {{
              if (current) {{
                paragraphs.push(current);
                current = "";
              }}
              continue;
            }}
            current = current ? _annolidNormalizeText(current + " " + cleaned) : cleaned;
            if (sentenceEnd.test(cleaned) && current.length > 180) {{
              paragraphs.push(current);
              current = "";
            }}
          }}
          if (current) paragraphs.push(current);
          window.__annolidParagraphsByPage[String(pageNum)] = paragraphs.map((text) => ({{
            pageNum,
            text,
            spans: [],
            yMin: null,
            yMax: null,
          }}));
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
	        const zoomLabel = document.getElementById("annolidZoomLabel");
	        const printBtn = document.getElementById("annolidPrint");
        const menuBtn = document.getElementById("annolidMenuBtn");
        const menuPanel = document.getElementById("annolidMenuPanel");
	
	        if (titleEl) titleEl.textContent = pdfTitle || "PDF";
	        if (totalPagesEl) totalPagesEl.textContent = String(total);
	        if (pageInput) pageInput.setAttribute("max", String(total));
	
	        function _annolidClampScale(value) {{
	          const v = parseFloat(value);
	          if (!isFinite(v)) return DEFAULT_SCALE;
	          return Math.max(MIN_SCALE, Math.min(MAX_SCALE, v));
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
	
	        function _annolidScaleMarks(ratio) {{
	          if (!isFinite(ratio) || ratio === 1) return;
	          const pages = window.__annolidPages || {{}};
	          Object.keys(pages).forEach((key) => {{
	            const state = pages[key];
	            if (!state || !state.marks) return;
	            for (const hl of (state.marks.highlights || [])) {{
	              const rects = hl.rects || [];
	              rects.forEach((r) => {{
	                if (!r) return;
	                r.x *= ratio; r.y *= ratio; r.w *= ratio; r.h *= ratio;
	              }});
	            }}
	            for (const stroke of (state.marks.strokes || [])) {{
	              const pts = stroke.points || [];
	              pts.forEach((p) => {{
	                if (!p) return;
	                p.x *= ratio; p.y *= ratio;
	              }});
	              if (stroke.size != null) {{
	                stroke.size = Math.max(1, stroke.size * ratio);
	              }}
	            }}
	          }});
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
	
	        async function _annolidRerenderAll(newScale) {{
	          if (!container) return;
	          const clamped = _annolidClampScale(newScale);
	          const oldScale = scale;
	          if (Math.abs(clamped - oldScale) < 0.001) return;
	          if (zoomBusy) {{
	            pendingZoom = clamped;
	            return;
	          }}
	          zoomBusy = true;
	          pendingZoom = null;
	
	          try {{
	            const anchor = _annolidGetScrollAnchor();
	            scale = clamped;
	            _annolidScaleMarks(scale / oldScale);
	
	            renderEpoch += 1;
	            renderChain = Promise.resolve();
	            nextPage = 1;
	            container.innerHTML = "";
	            window.__annolidSpans = [];
	            window.__annolidSpanCounter = 0;
	            window.__annolidSpanMeta = {{}};
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
	            window.__annolidParagraphOffsets = {{}};
	            window.__annolidParagraphTotal = 0;
	            window.__annolidParagraphs = [];
	
	            _annolidUpdateNavState();
	            await _annolidEnsureRenderedThrough(anchor.pageNum);
	            _annolidScrollToPage(anchor.pageNum, anchor.offsetFrac);
	            _annolidUpdateNavState();
	          }} finally {{
	            zoomBusy = false;
	            if (pendingZoom != null) {{
	              const next = pendingZoom;
	              pendingZoom = null;
	              _annolidRerenderAll(next);
	            }}
	          }}
	        }}
	
	        async function _annolidZoomFitWidth() {{
	          if (!container) return;
	          try {{
	            const page = await pdf.getPage(1);
	            const baseViewport = page.getViewport({{ scale: 1 }});
	            const gutter = 32;
	            const available = Math.max(100, container.clientWidth - gutter);
	            const target = available / Math.max(1, baseViewport.width);
	            await _annolidRerenderAll(target);
	          }} catch (e) {{
	            console.warn("Zoom fit failed", e);
	          }}
	        }}
	
	        function _annolidZoomBy(factor) {{
	          const next = _annolidClampScale(scale * factor);
	          _annolidRerenderAll(next);
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
        if (printBtn) printBtn.addEventListener("click", () => _annolidPrintPdf().catch(() => {{}}));
	        _annolidUpdateNavState();
	        if (container) {{
	          container.addEventListener("click", async (ev) => {{
	            if (!window.__annolidReaderEnabled) return;
	            if (!window.__annolidBridge || typeof window.__annolidBridge.onParagraphClicked !== "function") return;
            if (window.__annolidMarks && window.__annolidMarks.tool && window.__annolidMarks.tool !== "select") return;
            const sel = window.getSelection ? window.getSelection() : null;
            if (sel && !sel.isCollapsed) return;
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
          const spanEl = ev.target && ev.target.closest ? ev.target.closest(".textLayer span") : null;
          if (spanEl && spanEl.dataset && spanEl.dataset.annolidIndex) {{
            spanIdx = parseInt(spanEl.dataset.annolidIndex, 10);
          }}
          let paraIndex = -1;
          if (spanIdx >= 0) {{
            paraIndex = _annolidFindParagraphIndexBySpan(pageNum, spanIdx);
          }}
          if (paraIndex < 0) {{
            const pageRect = pageDiv.getBoundingClientRect();
            const y = ev.clientY - pageRect.top;
            paraIndex = _annolidFindParagraphIndexByPoint(pageNum, y);
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
            const splitFallback = (text) => {{
              const normalized = _annolidNormalizeText(text || "");
              if (!normalized) return [];
              const out = [];
              const pattern = /.+?(?:[.!?。！？]+|$)/g;
              let match;
              while ((match = pattern.exec(normalized)) !== null) {{
                const seg = _annolidNormalizeText(match[0] || "");
                if (seg) out.push(seg);
              }}
              return out.length ? out : [normalized];
            }};
            remaining.forEach((p) => {{
              if (typeof window.__annolidSplitParagraphIntoSentences === "function") {{
                const splits = window.__annolidSplitParagraphIntoSentences(p) || [];
                if (splits.length) {{
                  splits.forEach((s) => sentences.push(s));
                  return;
                }}
              }}
              const pageForPara = (parseInt(p.pageNum || p.page || 0, 10) || pageNum);
              const segs = splitFallback(p.text || "");
              if (segs.length) {{
                segs.forEach((seg) => {{
                  sentences.push({{
                    text: seg,
                    spans: [],
                    pageNum: pageForPara,
                  }});
                }});
                return;
              }}
              sentences.push({{
                text: p.text || "",
                spans: [],
                pageNum: pageForPara,
              }});
            }});
            window.__annolidBridge.onParagraphClicked({{
              startIndex,
              total: window.__annolidParagraphTotal || (startIndex + remaining.length),
              paragraphs: remaining,
              sentences,
              sentenceStartIndex: 0,
              sentenceTotal: sentences.length,
            }});
          }});
        }}

	        async function renderPage(pageNum, epoch) {{
	          if (epoch !== renderEpoch) return;
	          const page = await pdf.getPage(pageNum);
	          const viewport = page.getViewport({{ scale }});

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

            pix = page.get_pixmap(matrix=fitz.Matrix(self._zoom, self._zoom))
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
        self._reader_queue_offset = max(0, start_index)
        self._reader_total = max(
            total, self._reader_queue_offset + len(texts))
        self._reader_current_index = self._reader_queue_offset
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
        self._start_reader_from_local_index(0)

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
            try:
                from annolid.agents.kokoro_tts import text_to_speech, play_audio

                for idx, chunk in enumerate(chunks):
                    if self.token is not None and self.token.cancelled:
                        return
                    QtCore.QMetaObject.invokeMethod(
                        self.widget,
                        "_on_speak_chunk",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(int, idx),
                    )
                    audio_data = text_to_speech(
                        chunk,
                        voice=str(self.tts_settings.get("voice", "af_sarah")),
                        speed=float(self.tts_settings.get("speed", 1.0)),
                        lang=str(self.tts_settings.get("lang", "en-us")),
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
                    if duration_ms > 0:
                        QtCore.QMetaObject.invokeMethod(
                            self.widget,
                            "_on_speak_chunk_timing",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(int, idx),
                            QtCore.Q_ARG(int, duration_ms),
                        )
                    if self.token is not None and self.token.cancelled:
                        return
                    play_audio(samples, sample_rate)
                return
            except Exception:
                pass

            # Fallback to gTTS + in-memory playback
            try:
                from gtts import gTTS
                from pydub import AudioSegment
                import numpy as np
                import tempfile
                import os

                lang = str(self.tts_settings.get("lang", "en-us")).lower()
                gtts_lang = lang.split("-")[0] if lang else "en"
                for idx, chunk in enumerate(chunks):
                    if self.token is not None and self.token.cancelled:
                        return
                    QtCore.QMetaObject.invokeMethod(
                        self.widget,
                        "_on_speak_chunk",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(int, idx),
                    )
                    tts = gTTS(text=chunk, lang=gtts_lang)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        temp_file_path = os.path.join(
                            tmpdir, "temp_caption.mp3")
                        tts.save(temp_file_path)

                        audio = AudioSegment.from_file(
                            temp_file_path, format="mp3")
                        samples = np.array(audio.get_array_of_samples())
                        if samples.size == 0:
                            continue
                        if audio.channels == 2:
                            samples = samples.reshape((-1, 2))
                        duration_ms = 0
                        try:
                            duration_ms = int(
                                round(
                                    (len(samples) / float(audio.frame_rate)) * 1000)
                            )
                        except Exception:
                            duration_ms = 0
                        if duration_ms > 0:
                            QtCore.QMetaObject.invokeMethod(
                                self.widget,
                                "_on_speak_chunk_timing",
                                QtCore.Qt.QueuedConnection,
                                QtCore.Q_ARG(int, idx),
                                QtCore.Q_ARG(int, duration_ms),
                            )
                        if self.token is not None and self.token.cancelled:
                            return
                        play_audio_buffer(
                            samples, audio.frame_rate, blocking=True)
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
        sentences = re.split(r"(?<=[.!?。！？])\s+", cleaned)
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
