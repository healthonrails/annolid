"""PDF viewer implementation.

The public entrypoint is `annolid.gui.widgets.pdf_viewer.PdfViewerWidget`.
This module contains the full implementation and may be split further over time.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

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
from annolid.gui.widgets.pdf_viewer_bridge import _PdfReaderBridge, _SpeakToken
from annolid.gui.widgets.pdf_viewer_server import (
    _ensure_pdfjs_http_server,
    _register_pdfjs_http_pdf,
)
from annolid.gui.widgets.pdf_user_state import (
    delete_pdf_state,
    load_pdf_state,
    pdf_state_key,
    save_pdf_state,
)


if _WEBENGINE_AVAILABLE:
    # type: ignore[misc]
    class _AnnolidWebEnginePage(QtWebEngineWidgets.QWebEnginePage):
        def _should_open_url_externally(self, url: QtCore.QUrl) -> bool:
            try:
                if not url.isValid():
                    return False
                scheme = (url.scheme() or "").lower()
                if not scheme:
                    return False
                if scheme in {"http", "https"}:
                    host = (url.host() or "").lower()
                    if host in {"127.0.0.1", "localhost"}:
                        return False
                    return True
                if scheme in {"mailto", "tel"}:
                    return True
            except Exception:
                return False
            return False

        def acceptNavigationRequest(  # noqa: N802 - Qt override
            self,
            url: QtCore.QUrl,
            # type: ignore[name-defined]
            navType: "QtWebEngineWidgets.QWebEnginePage.NavigationType",
            isMainFrame: bool,
        ) -> bool:
            try:
                if isMainFrame and self._should_open_url_externally(url):
                    QtGui.QDesktopServices.openUrl(url)
                    return False
            except Exception:
                return True
            return super().acceptNavigationRequest(url, navType, isMainFrame)

        def createWindow(  # noqa: N802 - Qt override
            self,
            # type: ignore[name-defined]
            windowType: "QtWebEngineWidgets.QWebEnginePage.WebWindowType",
        ) -> "QtWebEngineWidgets.QWebEnginePage":
            parent_page = self

            class _ExternalBrowserPage(QtWebEngineWidgets.QWebEnginePage):
                def acceptNavigationRequest(  # noqa: N802 - Qt override
                    self,
                    url: QtCore.QUrl,
                    # type: ignore[name-defined]
                    navType: "QtWebEngineWidgets.QWebEnginePage.NavigationType",
                    isMainFrame: bool,
                ) -> bool:
                    try:
                        if isMainFrame and url.isValid():
                            if parent_page._should_open_url_externally(url):
                                QtGui.QDesktopServices.openUrl(url)
                                QtCore.QTimer.singleShot(0, self.deleteLater)
                                return False
                    except Exception:
                        QtCore.QTimer.singleShot(0, self.deleteLater)
                        return False
                    # No in-app tabs/windows for the PDF viewer; drop the request.
                    QtCore.QTimer.singleShot(0, self.deleteLater)
                    return False

            return _ExternalBrowserPage(self)

        def javaScriptConsoleMessage(  # noqa: N802 - Qt override
            self,
            # type: ignore[name-defined]
            level: "QtWebEngineWidgets.QWebEnginePage.JavaScriptConsoleMessageLevel",
            message: str,
            lineNumber: int,
            sourceID: str,
        ) -> None:
            try:
                logger.info(f"QtWebEngine js: {message} ({sourceID}:{lineNumber})")
            except Exception:
                pass


# NOTE: The PDF.js HTTP server and the Qt WebChannel bridge are implemented in
# `annolid.gui.widgets.pdf_viewer_server` and `annolid.gui.widgets.pdf_viewer_bridge`.
class PdfViewerWidget(QtWidgets.QWidget):
    """PDF viewer that prefers an embedded browser (if available) with fallback rendering."""

    selection_ready = QtCore.Signal(str)
    page_changed = QtCore.Signal(int, int)
    controls_enabled_changed = QtCore.Signal(bool)
    bookmarks_changed = QtCore.Signal(list)
    reader_state_changed = QtCore.Signal(str, int, int)
    reader_availability_changed = QtCore.Signal(bool, str)
    reading_log_event = QtCore.Signal(dict)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._doc = None
        self._pdf_path: Optional[Path] = None
        self._pdf_key: str = ""
        self._pdf_user_state: dict[str, object] = {}
        self._pdf_user_state_pending: Optional[dict[str, object]] = None
        self._pdf_user_state_timer = QtCore.QTimer(self)
        self._pdf_user_state_timer.setSingleShot(True)
        self._pdf_user_state_timer.timeout.connect(self._flush_pdf_user_state_to_disk)
        self._reading_log: list[dict[str, object]] = []
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
        self._dictionary_lookup_id = ""
        self._dictionary_popup_pos: Optional[QtCore.QPoint] = None
        self._dictionary_save_to_notes = False
        self._dictionary_note_anchor: Optional[dict[str, float]] = None
        self._active_dictionary_dialog: Optional[QtWidgets.QDialog] = None
        self._scholar_dialog: Optional[QtWidgets.QDialog] = None
        self._scholar_web_view = None
        self._scholar_items: list[dict[str, str]] = []
        self._scholar_index = 0
        self._scholar_group_label = ""
        self._scholar_label = None
        self._scholar_counter = None
        self._scholar_prev_btn = None
        self._scholar_next_btn = None
        self._scholar_open_btn = None
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
        self._last_reader_stop_log_ts = 0.0
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
            "Select text on this page, then right-click to speak it."
        )
        self.text_view.selectionChanged.connect(self._on_text_selection_changed)
        self.text_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.text_view.customContextMenuRequested.connect(self._show_context_menu)
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
                    self._web_channel = QtWebChannel.QWebChannel(self._web_view.page())
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
                    QtWebEngineWidgets.QWebEngineSettings, "PdfViewerEnabled", None
                )
                plugins_attr = getattr(
                    QtWebEngineWidgets.QWebEngineSettings, "PluginsEnabled", None
                )
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
                    lambda *_: logger.warning("QtWebEngine render process terminated")
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
        self._use_web_engine = bool(_WEBENGINE_AVAILABLE and self._web_view is not None)
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
        self._pdf_path = path
        try:
            self._pdf_key = pdf_state_key(path)
        except Exception:
            self._pdf_key = ""
        try:
            self._pdf_user_state = load_pdf_state(path)
        except Exception:
            self._pdf_user_state = {}
        self._reading_log = []
        try:
            existing_log = (
                self._pdf_user_state.get("log")
                if isinstance(self._pdf_user_state, dict)
                else None
            )
            if isinstance(existing_log, list):
                self._reading_log = [e for e in existing_log if isinstance(e, dict)][
                    :500
                ]
        except Exception:
            self._reading_log = []

        self._append_reading_log_event(
            {
                "type": "open",
                "label": f"Opened {path.name}",
            }
        )

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
        if not self._pdf_key:
            try:
                self._pdf_key = pdf_state_key(path)
            except Exception:
                self._pdf_key = ""
        if not self._pdf_user_state:
            try:
                self._pdf_user_state = load_pdf_state(path)
            except Exception:
                self._pdf_user_state = {}

        self._current_page = 0
        self._stack.setCurrentIndex(0)
        self._set_controls_for_web(False)
        self._emit_reader_availability()
        restored_zoom = self._restore_pymupdf_reading_state()
        if not restored_zoom:
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
                            rendered_pages = int(result.get("renderedPages", 0) or 0)
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
                self._fallback_from_web(path, "PDF mimeType not available in WebEngine")
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
        # Serve the PDF over a local HTTP endpoint so PDF.js can fetch it
        # reliably (fetch/XHR against file:// can hang in QtWebEngine).
        base = _ensure_pdfjs_http_server()
        base_url = QtCore.QUrl(base + "/")
        pdf_url = _register_pdfjs_http_pdf(path)
        pdf_b64 = ""
        pdf_key = self._pdf_key or ""
        initial_state = (
            self._pdf_user_state if isinstance(self._pdf_user_state, dict) else {}
        )
        try:
            initial_state_js = json.dumps(initial_state, ensure_ascii=False)
            initial_state_js = initial_state_js.replace("</", "<\\/")
        except Exception:
            initial_state_js = "{}"
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="pdfjs/annolid_viewer.css" />
  <script src="pdfjs/annolid_viewer_polyfills.js"></script>
  <script>
    window.__annolidPdfKey = "{pdf_key}";
    window.__annolidInitialUserState = {initial_state_js};
    window.__annolidPdfUrl = "{pdf_url}";
    window.__annolidPdfBase64 = "{pdf_b64}";
    window.__annolidPdfTitle = "{path.name}";
  </script>
  <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
  <script src="pdfjs/pdf.min.js"></script>
  <script src="pdfjs/annolid_viewer.js"></script>
</head>
<body>
  <div id="annolidToolbar">
    <div class="annolid-toolbar-left">
      <button id="annolidMenuBtn" title="Menu" class="annolid-icon-btn">☰</button>
      <div id="annolidMenuPanel">
        <button data-action="resume" id="annolidResumeBtn" class="annolid-hidden">Resume reading</button>
        <button data-action="first-page">Go to first page</button>
        <button data-action="fit">Fit width</button>
        <button data-action="reset">Reset zoom</button>
        <button data-action="print">Print</button>
        <span class="annolid-sep"></span>
        <button data-action="clear-state" id="annolidClearStateBtn" class="annolid-hidden">Clear saved data</button>
      </div>
      <div class="annolid-title" id="annolidTitle">PDF</div>
    </div>
    <div class="annolid-nav">
      <button id="annolidPrevPage" title="Previous page">◀</button>
      <input id="annolidPageInput" type="number" value="1" min="1" />
      <button id="annolidNextPage" title="Next page">▶</button>
      <button id="annolidBookmarkBtn" title="Bookmark this page">☆</button>
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
      <div class="annolid-group" data-overflow="auto" id="annolidMetaGroup">
        <button id="annolidNotesBtn" title="Notes and bookmarks">Notes</button>
      </div>
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
  <div id="viewerContainer"></div>
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
  <div class="annolid-popup" id="annolidNotesModal" role="dialog" aria-label="Notes &amp; Bookmarks">
    <div class="annolid-modal-content" role="document">
      <div class="annolid-modal-header">
        <div class="annolid-modal-title" id="annolidNotesTitle">Notes &amp; Bookmarks</div>
        <div class="annolid-modal-actions">
          <button id="annolidNoteAdd" title="Add note from selection">Add</button>
          <button id="annolidNoteDelete" title="Delete selected note">Delete</button>
          <button id="annolidNoteSave" title="Save note">Save</button>
          <button id="annolidNoteClose" title="Close">Close</button>
        </div>
      </div>
      <div class="annolid-modal-body">
        <div class="annolid-notes-grid">
          <div class="annolid-notes-left">
            <input id="annolidNotesSearch" class="annolid-notes-search" type="search" placeholder="Search notes…" />
            <div class="annolid-notes-section">
              <div class="annolid-notes-section-title">Bookmarks</div>
              <div id="annolidBookmarksList" class="annolid-notes-list"></div>
            </div>
            <div class="annolid-notes-section">
              <div class="annolid-notes-section-title">Notes</div>
              <div id="annolidNotesList" class="annolid-notes-list"></div>
            </div>
          </div>
          <div class="annolid-notes-right">
            <div id="annolidNoteMeta" class="annolid-notes-meta annolid-muted">Select a note to view/edit.</div>
            <textarea id="annolidNoteEditor" class="annolid-notes-editor" placeholder="Write a comment…"></textarea>
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

        fmt = QtGui.QImage.Format_RGBA8888 if pix.alpha else QtGui.QImage.Format_RGB888
        image = QtGui.QImage(pix.samples, pix.width, pix.height, pix.stride, fmt).copy()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(image))

        text = (page.get_text("text") or "").strip()
        self.text_view.setPlainText(text)
        self.text_view.moveCursor(QtGui.QTextCursor.Start)
        self.page_changed.emit(self._current_page, self._doc.page_count)
        self._schedule_pdf_user_state_save(
            {
                "reading": {
                    "page": int(self._current_page),
                    "zoom": float(self._zoom),
                    "rotation": int(self._rotation),
                }
            }
        )

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
        selected_text = self._selected_text()
        lookup_action = QtWidgets.QAction("Look up in dictionary…", self)
        lookup_action.setEnabled(bool(self._extract_single_word(selected_text)))
        lookup_action.triggered.connect(
            lambda: self._request_dictionary_lookup(
                selected_text, global_pos=self.text_view.mapToGlobal(position)
            )
        )
        menu.insertAction(menu.actions()[0] if menu.actions() else None, lookup_action)
        lookup_save_action = QtWidgets.QAction("Look up and save to notes", self)
        lookup_save_action.setEnabled(bool(self._extract_single_word(selected_text)))
        lookup_save_action.triggered.connect(
            lambda: self._request_dictionary_lookup(
                selected_text,
                global_pos=self.text_view.mapToGlobal(position),
                save_to_notes=True,
            )
        )
        menu.insertAction(
            menu.actions()[0] if menu.actions() else None, lookup_save_action
        )
        speak_action = QtWidgets.QAction("Speak selection", self)
        speak_action.setEnabled(self._has_selection() and not self._speaking)
        speak_action.triggered.connect(self._request_speak_selection)
        menu.insertAction(menu.actions()[0] if menu.actions() else None, speak_action)
        bookmark_action = QtWidgets.QAction("Bookmark this page", self)
        bookmark_action.triggered.connect(self._toggle_fallback_bookmark)
        menu.insertAction(
            menu.actions()[0] if menu.actions() else None, bookmark_action
        )
        note_action = QtWidgets.QAction("Add note…", self)
        note_action.setEnabled(True)
        note_action.triggered.connect(self._add_fallback_note)
        menu.insertAction(menu.actions()[0] if menu.actions() else None, note_action)
        menu.exec_(self.text_view.mapToGlobal(position))

    def _show_web_context_menu(self, position: QtCore.QPoint) -> None:
        if self._web_view is None:
            return
        page = self._web_view.page()
        global_pos = self._web_view.mapToGlobal(position)

        def show_menu(selection: object) -> None:
            selected_text = (str(selection) if selection is not None else "").strip()
            if not selected_text:
                selected_text = (self._web_view.selectedText() or "").strip()
            if selected_text:
                self._update_selection_cache(selected_text)
            else:
                self._clear_selection_cache()
            menu = page.createStandardContextMenu()
            menu.insertSeparator(menu.actions()[0] if menu.actions() else None)
            lookup_action = QtWidgets.QAction("Look up in dictionary…", self)
            lookup_action.setEnabled(bool(self._extract_single_word(selected_text)))
            lookup_action.triggered.connect(
                lambda: self._request_dictionary_lookup(
                    selected_text, global_pos=global_pos
                )
            )
            menu.insertAction(
                menu.actions()[0] if menu.actions() else None, lookup_action
            )
            lookup_save_action = QtWidgets.QAction("Look up and save to notes", self)
            lookup_save_action.setEnabled(
                bool(self._extract_single_word(selected_text))
            )
            lookup_save_action.triggered.connect(
                lambda: self._request_dictionary_lookup(
                    selected_text, global_pos=global_pos, save_to_notes=True
                )
            )
            menu.insertAction(
                menu.actions()[0] if menu.actions() else None, lookup_save_action
            )
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
                    texts = [str(t).strip() for t in raw_texts if str(t).strip()]
                    indices = [int(i) for i in raw_indices]
                except Exception:
                    texts = []
                    indices = []
            if texts and indices and len(texts) == len(indices):
                self._web_selected_span_text = {
                    idx: text for text, idx in zip(texts, indices)
                }
                groups, sentences = self._group_web_spans_into_sentences(texts, indices)
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
            if (
                self._selection_cache
                and (time.monotonic() - self._selection_cache_time) < 2.0
            ):
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
            self._web_view.page().triggerAction(QtWebEngineWidgets.QWebEnginePage.Copy)
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
            self._log_reader_stop_event()
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
        clamped = max(self._reader_queue_offset, min(max_global, int(global_index)))
        local_index = clamped - self._reader_queue_offset
        self._start_reader_from_local_index(local_index)

    def _pause_reader(self) -> None:
        if self._reader_state != "reading":
            return
        if not self._speaking:
            self._reader_state = "paused"
            self._reader_pause_requested = False
            self._log_reader_stop_event(kind="reader_pause", label="Reader paused")
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
        self._append_reading_log_event(
            {"type": "reader_resume", "label": "Reader resumed"}
        )
        local_index = max(0, self._reader_current_index - self._reader_queue_offset)
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
        use_sentences = isinstance(raw_sentences, list) and len(raw_sentences) > 0
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
                payload.get("sentenceStartIndex" if use_sentences else "startIndex", 0)
            )
        except Exception:
            start_index = 0
        local_start_index = 0
        if use_sentences:
            try:
                local_start_index = int(payload.get("sentenceLocalStartIndex", 0) or 0)
            except Exception:
                local_start_index = 0
        try:
            total = int(payload.get("sentenceTotal" if use_sentences else "total", 0))
        except Exception:
            total = 0
        if total <= 0:
            total = start_index + len(texts)

        self._reader_queue = texts
        self._reader_spans = spans_list
        self._reader_pages = pages
        self._reader_queue_offset = max(0, start_index) if not use_sentences else 0
        self._reader_total = max(total, self._reader_queue_offset + len(texts))
        self._reader_current_index = self._reader_queue_offset + max(
            0, local_start_index
        )
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
            start_local = max(0, min(local_start_index, len(self._reader_queue) - 1))
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
            cleaned = re.sub(r"\s+", " ", segment.replace("\u2029", " ")).strip()
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

    def _extract_single_word(self, text: str) -> str:
        import re

        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        tokens = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", cleaned)
        if len(tokens) != 1:
            return ""
        return tokens[0]

    def _request_dictionary_lookup(
        self,
        selected_text: str,
        *,
        global_pos: Optional[QtCore.QPoint] = None,
        save_to_notes: bool = False,
    ) -> None:
        word = self._extract_single_word(selected_text)
        if not word:
            QtWidgets.QToolTip.showText(
                QtGui.QCursor.pos(),
                "Select a single word to look it up.",
                self,
            )
            return
        self._capture_dictionary_note_anchor(
            global_pos=global_pos,
            done=lambda anchor: self._start_dictionary_lookup(
                word,
                global_pos=global_pos,
                save_to_notes=save_to_notes,
                note_anchor=anchor,
            ),
        )

    def _capture_dictionary_note_anchor(
        self,
        *,
        global_pos: Optional[QtCore.QPoint] = None,
        done: Optional[object] = None,
    ) -> None:
        callback = done if callable(done) else None
        if callback is None:
            return
        page_num = int(self._current_page) + 1
        offset_frac = 0.0

        if (
            self._use_web_engine
            and self._web_view is not None
            and self._web_container is not None
            and self._stack.currentWidget() is self._web_container
        ):
            try:
                self._dictionary_popup_pos = global_pos

                def handle(payload: object) -> None:
                    anchor = {
                        "pageNum": float(page_num),
                        "offsetFrac": float(offset_frac),
                    }
                    if isinstance(payload, dict):
                        try:
                            pn = int(payload.get("pageNum") or page_num)
                            frac = float(payload.get("offsetFrac") or 0.0)
                            anchor = {"pageNum": float(pn), "offsetFrac": float(frac)}
                        except Exception:
                            pass
                    callback(anchor)

                self._web_view.page().runJavaScript(
                    """(() => {
  try {
    const st = (window.__annolidExportUserState && window.__annolidExportUserState()) || (window.__annolidUserState || {});
    const reading = (st && st.reading) ? st.reading : {};
    return { pageNum: reading.pageNum || 1, offsetFrac: reading.offsetFrac || 0 };
  } catch (e) { return null; }
})()""",
                    handle,
                )
                return
            except Exception:
                pass

        try:
            if self.text_view is not None:
                sb = self.text_view.verticalScrollBar()
                maximum = float(sb.maximum() or 0)
                if maximum > 0:
                    offset_frac = float(sb.value()) / maximum
        except Exception:
            offset_frac = 0.0
        callback({"pageNum": float(page_num), "offsetFrac": float(offset_frac)})

    def _start_dictionary_lookup(
        self,
        word: str,
        *,
        global_pos: Optional[QtCore.QPoint] = None,
        save_to_notes: bool = False,
        note_anchor: Optional[dict[str, float]] = None,
    ) -> None:
        self._dictionary_lookup_id = uuid.uuid4().hex
        self._dictionary_popup_pos = global_pos
        self._dictionary_save_to_notes = bool(save_to_notes)
        self._dictionary_note_anchor = (
            dict(note_anchor) if isinstance(note_anchor, dict) else None
        )
        QtWidgets.QToolTip.showText(
            QtGui.QCursor.pos(),
            f"Looking up “{word}”…",
            self,
        )
        self._thread_pool.start(
            _DictionaryLookupTask(
                widget=self,
                request_id=self._dictionary_lookup_id,
                word=word,
            )
        )

    @QtCore.Slot(str, str, str, str)
    def _on_dictionary_lookup_finished(  # pragma: no cover - UI glue
        self,
        request_id: str,
        word: str,
        html: str,
        error: str,
    ) -> None:
        if request_id != self._dictionary_lookup_id:
            return
        note_id: str = ""
        if self._dictionary_save_to_notes and not error:
            note_id = self._save_dictionary_lookup_to_notes(
                word, html=html, anchor=self._dictionary_note_anchor
            )
        self._show_dictionary_popup(
            word,
            html=html,
            error=error,
            global_pos=self._dictionary_popup_pos,
            saved_note_id=note_id,
            note_anchor=self._dictionary_note_anchor,
        )

    def _html_to_plain_text(self, html: str) -> str:
        doc = QtGui.QTextDocument()
        try:
            doc.setHtml(html or "")
        except Exception:
            return (html or "").strip()
        return doc.toPlainText().strip()

    def _push_user_state_to_web(self) -> None:
        if self._web_view is None:
            return
        if not self._use_web_engine:
            return
        state = self._pdf_user_state if isinstance(self._pdf_user_state, dict) else {}
        try:
            state_js = json.dumps(state, ensure_ascii=False).replace("</", "<\\/")
        except Exception:
            return
        try:
            self._web_view.page().runJavaScript(
                f"(() => {{ window.__annolidUserState = {state_js}; }})()"
            )
        except Exception:
            pass

    def _open_notes_and_select_web(self, note_id: str) -> None:
        if not note_id:
            return
        if self._web_view is None:
            return
        if not self._use_web_engine:
            return
        try:
            note_id_js = json.dumps(str(note_id), ensure_ascii=False)
        except Exception:
            return
        try:
            self._web_view.page().runJavaScript(
                f"window.__annolidOpenNotesAndSelect && window.__annolidOpenNotesAndSelect({note_id_js});"
            )
        except Exception:
            pass

    def _save_dictionary_lookup_to_notes(
        self, word: str, *, html: str, anchor: Optional[dict[str, float]] = None
    ) -> str:
        if self._pdf_path is None:
            QtWidgets.QToolTip.showText(
                QtGui.QCursor.pos(),
                "Open a PDF first to save notes.",
                self,
            )
            return ""
        definition_text = self._html_to_plain_text(html)
        if not definition_text:
            return ""
        page_num = int(self._current_page) + 1
        offset_frac = 0.0
        if isinstance(anchor, dict):
            try:
                page_num = int(anchor.get("pageNum") or page_num)
            except Exception:
                pass
            try:
                offset_frac = float(anchor.get("offsetFrac") or 0.0)
            except Exception:
                offset_frac = 0.0
        now = float(time.time())
        note_id = "note:" + uuid.uuid4().hex
        note_text = f"{word}\n\n{definition_text}".strip()
        if len(note_text) > 6000:
            note_text = note_text[:5997] + "…"

        state = self._pdf_user_state if isinstance(self._pdf_user_state, dict) else {}
        notes = state.get("notes") if isinstance(state.get("notes"), list) else []
        notes.insert(
            0,
            {
                "id": note_id,
                "pageNum": page_num,
                "page": max(0, page_num - 1),
                "offsetFrac": max(0.0, min(1.0, float(offset_frac))),
                "snippet": str(word)[:400],
                "text": note_text,
                "createdAt": now,
                "updatedAt": now,
            },
        )
        self._schedule_pdf_user_state_save({"notes": notes})
        self._append_reading_log_event(
            {
                "type": "note_add",
                "label": "Dictionary note added",
                "noteId": note_id,
                "pageNum": page_num,
                "snippet": str(word)[:120],
            }
        )
        self._push_user_state_to_web()
        self._open_notes_and_select_web(note_id)
        QtWidgets.QToolTip.showText(
            QtGui.QCursor.pos(),
            "Saved to notes.",
            self,
        )
        return note_id

    def _show_dictionary_popup(
        self,
        word: str,
        *,
        html: str = "",
        error: str = "",
        global_pos: Optional[QtCore.QPoint] = None,
        saved_note_id: str = "",
        note_anchor: Optional[dict[str, float]] = None,
    ) -> None:
        if self._active_dictionary_dialog is not None:
            try:
                self._active_dictionary_dialog.close()
            except Exception:
                pass
            self._active_dictionary_dialog = None

        dialog = QtWidgets.QDialog(self)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.setWindowTitle(f"Dictionary: {word}")
        dialog.setModal(False)
        dialog.resize(520, 420)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        view = QtWidgets.QTextBrowser(dialog)
        view.setOpenExternalLinks(True)
        if error:
            view.setPlainText(str(error))
        else:
            view.setHtml(html or "")
        layout.addWidget(view, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close, parent=dialog
        )
        save_btn = buttons.addButton(
            "Save to Notes", QtWidgets.QDialogButtonBox.ActionRole
        )
        save_btn.setEnabled(
            bool(html)
            and not bool(error)
            and not bool(saved_note_id)
            and self._pdf_path is not None
        )

        def handle_save() -> None:
            note_id = self._save_dictionary_lookup_to_notes(
                word, html=html, anchor=note_anchor
            )
            if note_id:
                try:
                    save_btn.setEnabled(False)
                except Exception:
                    pass

        save_btn.clicked.connect(handle_save)
        buttons.rejected.connect(dialog.close)
        buttons.accepted.connect(dialog.close)
        layout.addWidget(buttons, 0)

        anchor = global_pos if global_pos is not None else QtGui.QCursor.pos()
        try:
            dialog.move(anchor + QtCore.QPoint(12, 12))
        except Exception:
            pass
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._active_dictionary_dialog = dialog

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

    def _open_scholar_citations(self, payload: object) -> None:
        if not _WEBENGINE_AVAILABLE:
            return
        if QtWebEngineWidgets is None:
            return

        data = payload if isinstance(payload, dict) else {}
        raw_items = data.get("items") if isinstance(data, dict) else None
        items: list[dict[str, str]] = []
        if isinstance(raw_items, list):
            for entry in raw_items:
                if not isinstance(entry, dict):
                    continue
                url = str(entry.get("url") or "").strip()
                if not url:
                    continue
                title = str(entry.get("title") or "").strip()
                number = str(entry.get("number") or "").strip()
                query = str(entry.get("query") or "").strip()
                items.append(
                    {
                        "url": url,
                        "title": title,
                        "number": number,
                        "query": query,
                    }
                )
        if not items:
            return

        group_label = str(data.get("groupLabel") or "").strip()
        active_index = 0
        try:
            active_index = int(data.get("activeIndex") or 0)
        except Exception:
            active_index = 0
        active_index = max(0, min(active_index, len(items) - 1))

        self._scholar_items = items
        self._scholar_index = active_index
        self._scholar_group_label = group_label

        dialog = self._scholar_dialog
        if dialog is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            dialog.setModal(False)
            dialog.resize(980, 720)
            dialog.setWindowTitle("Google Scholar")

            layout = QtWidgets.QVBoxLayout(dialog)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

            header = QtWidgets.QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(8)

            label = QtWidgets.QLabel(dialog)
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            label.setStyleSheet("font-weight: 600;")
            counter = QtWidgets.QLabel(dialog)
            counter.setStyleSheet("color: #666;")

            prev_btn = QtWidgets.QToolButton(dialog)
            prev_btn.setText("‹")
            prev_btn.setAutoRaise(True)
            next_btn = QtWidgets.QToolButton(dialog)
            next_btn.setText("›")
            next_btn.setAutoRaise(True)

            open_btn = QtWidgets.QToolButton(dialog)
            open_btn.setText("Open in browser")
            open_btn.setAutoRaise(True)

            close_btn = QtWidgets.QToolButton(dialog)
            close_btn.setText("✕")
            close_btn.setAutoRaise(True)

            header.addWidget(label, 1)
            header.addWidget(counter, 0)
            header.addWidget(prev_btn, 0)
            header.addWidget(next_btn, 0)
            header.addWidget(open_btn, 0)
            header.addWidget(close_btn, 0)
            layout.addLayout(header, 0)

            web_view = QtWebEngineWidgets.QWebEngineView(dialog)
            layout.addWidget(web_view, 1)

            def cleanup() -> None:
                self._scholar_dialog = None
                self._scholar_web_view = None
                self._scholar_label = None
                self._scholar_counter = None
                self._scholar_prev_btn = None
                self._scholar_next_btn = None
                self._scholar_open_btn = None

            dialog.destroyed.connect(lambda *_: cleanup())

            def nav(delta: int) -> None:
                if not self._scholar_items:
                    return
                self._scholar_index = max(
                    0,
                    min(self._scholar_index + int(delta), len(self._scholar_items) - 1),
                )
                self._update_scholar_dialog()

            def open_in_browser() -> None:
                url = self._current_scholar_url()
                if not url:
                    return
                try:
                    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
                except Exception:
                    return

            prev_btn.clicked.connect(lambda: nav(-1))
            next_btn.clicked.connect(lambda: nav(1))
            open_btn.clicked.connect(open_in_browser)
            close_btn.clicked.connect(dialog.close)

            self._scholar_dialog = dialog
            self._scholar_web_view = web_view
            self._scholar_label = label
            self._scholar_counter = counter
            self._scholar_prev_btn = prev_btn
            self._scholar_next_btn = next_btn
            self._scholar_open_btn = open_btn

        self._update_scholar_dialog()
        try:
            self._scholar_dialog.show()
            self._scholar_dialog.raise_()
            self._scholar_dialog.activateWindow()
        except Exception:
            pass

    def _current_scholar_url(self) -> str:
        try:
            if not self._scholar_items:
                return ""
            idx = max(0, min(int(self._scholar_index), len(self._scholar_items) - 1))
            return str(self._scholar_items[idx].get("url") or "")
        except Exception:
            return ""

    def _update_scholar_dialog(self) -> None:
        if self._scholar_dialog is None:
            return
        items = self._scholar_items
        if not items:
            try:
                self._scholar_dialog.close()
            except Exception:
                pass
            return
        idx = max(0, min(int(self._scholar_index), len(items) - 1))
        item = items[idx]
        label_text = self._scholar_group_label or ""
        if not label_text:
            nums = [str(it.get("number") or "").strip() for it in items]
            nums = [n for n in nums if n]
            label_text = "[" + ", ".join(nums) + "]" if nums else "Citation"
        title = str(item.get("title") or "").strip()
        if title:
            label_text = f"{label_text}  —  {title}"
        if self._scholar_label is not None:
            try:
                self._scholar_label.setText(label_text)
            except Exception:
                pass
        if self._scholar_counter is not None:
            try:
                self._scholar_counter.setText(f"{idx + 1} / {len(items)}")
            except Exception:
                pass
        if self._scholar_prev_btn is not None:
            try:
                self._scholar_prev_btn.setEnabled(idx > 0)
            except Exception:
                pass
        if self._scholar_next_btn is not None:
            try:
                self._scholar_next_btn.setEnabled(idx + 1 < len(items))
            except Exception:
                pass
        url = str(item.get("url") or "").strip()
        if url and self._scholar_web_view is not None:
            try:
                self._scholar_web_view.load(QtCore.QUrl(url))
            except Exception:
                pass

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
            "engine": settings.get("engine", defaults.get("engine", "auto")),
            "voice": settings.get("voice", defaults["voice"]),
            "pocket_voice": settings.get(
                "pocket_voice", defaults.get("pocket_voice", "alba")
            ),
            "pocket_prompt_path": settings.get(
                "pocket_prompt_path", defaults.get("pocket_prompt_path", "")
            ),
            "pocket_speed": settings.get(
                "pocket_speed", defaults.get("pocket_speed", 1.0)
            ),
            "lang": settings.get("lang", defaults["lang"]),
            "speed": settings.get("speed", defaults["speed"]),
            "chatterbox_voice_path": settings.get(
                "chatterbox_voice_path", defaults.get("chatterbox_voice_path", "")
            ),
            "chatterbox_dtype": settings.get(
                "chatterbox_dtype", defaults.get("chatterbox_dtype", "fp32")
            ),
            "chatterbox_max_new_tokens": settings.get(
                "chatterbox_max_new_tokens",
                defaults.get("chatterbox_max_new_tokens", 1024),
            ),
            "chatterbox_repetition_penalty": settings.get(
                "chatterbox_repetition_penalty",
                defaults.get("chatterbox_repetition_penalty", 1.2),
            ),
            "chatterbox_apply_watermark": settings.get(
                "chatterbox_apply_watermark",
                defaults.get("chatterbox_apply_watermark", False),
            ),
        }
        self._thread_pool.start(
            _SpeakTextTask(self, cleaned, merged, chunks=chunks, token=token)
        )

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
            weights = [
                max(1, word_end - word_start) for word_start, word_end in word_spans
            ]
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
        cursor.setPosition(self._selection_anchor_end, QtGui.QTextCursor.KeepAnchor)
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
        if self._web_view is not None and self._highlight_mode in {
            "web",
            "web-sentence",
            "web-paragraph",
        }:
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
        if self._web_view is not None and self._highlight_mode in {
            "web",
            "web-sentence",
            "web-paragraph",
        }:
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
                max(
                    1,
                    int(self._word_highlight_durations_ms[self._word_highlight_index]),
                )
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
                selections.append(
                    self._make_text_extra_selection(
                        doc,
                        sentence_span[0],
                        sentence_span[1],
                        QtGui.QColor(255, 210, 80, 90),
                    )
                )
            if word_span is not None:
                selections.append(
                    self._make_text_extra_selection(
                        doc, word_span[0], word_span[1], QtGui.QColor(255, 160, 0, 160)
                    )
                )
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

    def closeEvent(
        self, event: QtGui.QCloseEvent
    ) -> None:  # pragma: no cover - GUI cleanup
        try:
            self._record_stop_event()
        except Exception:
            pass
        try:
            self._flush_pdf_user_state_to_disk()
        except Exception:
            pass
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

    # ---- Persistent reader state (progress, marks, notes) -------------------
    def _restore_pymupdf_reading_state(self) -> bool:
        """Apply persisted reading state in PyMuPDF fallback mode.

        Returns True if a specific zoom level was restored (so fit-to-width should be skipped).
        """
        state = self._pdf_user_state if isinstance(self._pdf_user_state, dict) else {}
        reading = state.get("reading") if isinstance(state.get("reading"), dict) else {}
        restored_zoom = False

        try:
            page = reading.get("page", 0)
            if page is None:
                page = 0
            page_index = int(page)
        except Exception:
            page_index = 0

        try:
            rotation = int(reading.get("rotation", 0) or 0) % 360
        except Exception:
            rotation = 0

        zoom = reading.get("zoom", None)
        zoom_factor: Optional[float] = None
        try:
            if zoom is not None:
                zoom_factor = float(zoom)
        except Exception:
            zoom_factor = None

        if self._doc is not None:
            try:
                page_index = max(0, min(page_index, int(self._doc.page_count) - 1))
            except Exception:
                page_index = max(0, page_index)
        self._current_page = page_index
        self._rotation = rotation
        if zoom_factor is not None:
            self._zoom = max(0.5, min(3.0, float(zoom_factor)))
            restored_zoom = True
        return restored_zoom

    @staticmethod
    def _deep_merge_state(
        dst: dict[str, object], src: dict[str, object]
    ) -> dict[str, object]:
        for key, value in (src or {}).items():
            if (
                isinstance(value, dict)
                and isinstance(dst.get(key), dict)
                and dst.get(key) is not None
            ):
                dst[key] = PdfViewerWidget._deep_merge_state(  # type: ignore[arg-type]
                    dict(dst.get(key) or {}),  # type: ignore[arg-type]
                    dict(value),
                )
            else:
                dst[key] = value  # type: ignore[assignment]
        return dst

    def _schedule_pdf_user_state_save(
        self, update: dict[str, object], *, replace: bool = False
    ) -> None:
        if self._pdf_path is None:
            return
        if replace:
            self._pdf_user_state = dict(update or {})
        else:
            base = (
                self._pdf_user_state if isinstance(self._pdf_user_state, dict) else {}
            )
            self._pdf_user_state = self._deep_merge_state(
                dict(base), dict(update or {})
            )
        # Debounce writes to disk.
        self._pdf_user_state_pending = (
            self._pdf_user_state if isinstance(self._pdf_user_state, dict) else {}
        )
        try:
            self._pdf_user_state_timer.start(800)
        except Exception:
            self._flush_pdf_user_state_to_disk()

    def _flush_pdf_user_state_to_disk(self) -> None:
        if self._pdf_path is None:
            return
        pending = (
            self._pdf_user_state_pending
            if isinstance(self._pdf_user_state_pending, dict)
            else None
        )
        if not pending:
            return
        try:
            save_pdf_state(self._pdf_path, pending)
        except Exception:
            pass
        self._pdf_user_state_pending = None

    def _append_reading_log_event(self, event: dict[str, object]) -> None:
        if self._pdf_path is None:
            return
        payload = dict(event or {})
        payload.setdefault("id", "evt:" + uuid.uuid4().hex)
        payload.setdefault("ts", float(time.time()))
        if "pageNum" not in payload:
            try:
                payload["pageNum"] = int(self._current_page) + 1
            except Exception:
                payload["pageNum"] = 0
        # Optional offset for PDF.js view; set later when available.
        payload.setdefault("offsetFrac", 0.0)

        self._reading_log.insert(0, payload)
        self._reading_log = self._reading_log[:600]
        self._schedule_pdf_user_state_save({"log": list(self._reading_log)})
        try:
            self.reading_log_event.emit(payload)
        except Exception:
            pass

    def _log_reader_stop_event(
        self,
        *,
        kind: str = "reader_stop",
        label: str = "Reader stopped",
    ) -> None:
        """Log a reader stop/pause event and persist a resume anchor."""
        # Debounce: stop() + on_speak_finished can happen close together.
        now = time.time()
        if (now - float(self._last_reader_stop_log_ts or 0.0)) < 0.4:
            return
        self._last_reader_stop_log_ts = now

        page_num = int(self._current_page) + 1
        snippet = ""
        try:
            local_index = int(self._reader_current_index - self._reader_queue_offset)
            if 0 <= local_index < len(self._reader_queue):
                snippet = str(self._reader_queue[local_index] or "").strip()[:160]
            if 0 <= local_index < len(self._reader_pages):
                maybe_page = int(self._reader_pages[local_index] or 0)
                if maybe_page > 0:
                    page_num = maybe_page
        except Exception:
            pass

        def finalize(offset_frac: float) -> None:
            offset_frac_clamped = max(0.0, min(1.0, float(offset_frac or 0.0)))
            self._schedule_pdf_user_state_save(
                {
                    "readingStopped": {
                        "pageNum": int(page_num),
                        "offsetFrac": offset_frac_clamped,
                        "ts": float(now),
                        "snippet": snippet,
                    }
                }
            )
            self._append_reading_log_event(
                {
                    "type": kind,
                    "label": label,
                    "pageNum": int(page_num),
                    "offsetFrac": offset_frac_clamped,
                    "snippet": snippet,
                }
            )

        if self._web_view is not None and self._pdfjs_active:
            try:
                self._web_view.page().runJavaScript(
                    "window.__annolidExportUserState && window.__annolidExportUserState();",
                    lambda payload: finalize(
                        float(
                            (
                                (payload or {})
                                .get("reading", {})
                                .get("offsetFrac", 0.0)
                                if isinstance(payload, dict)
                                else 0.0
                            )
                        )
                    ),
                )
                return
            except Exception:
                pass
        finalize(0.0)

    def reading_log(self) -> list[dict[str, object]]:
        return list(self._reading_log)

    def clear_reading_log(self) -> None:
        self._reading_log = []
        self._schedule_pdf_user_state_save({"log": []})

    def _record_stop_event(self) -> None:
        # Best-effort: ask PDF.js for current anchor; fallback to current page.
        if self._web_view is not None and self._pdfjs_active:

            def _after(payload: object) -> None:
                event: dict[str, object] = {
                    "type": "stop",
                    "label": "Last stop",
                }
                if isinstance(payload, dict):
                    reading = payload.get("reading")
                    if isinstance(reading, dict):
                        try:
                            event["pageNum"] = int(reading.get("pageNum", 0) or 0)
                        except Exception:
                            pass
                        try:
                            event["offsetFrac"] = float(
                                reading.get("offsetFrac", 0.0) or 0.0
                            )
                        except Exception:
                            pass
                self._append_reading_log_event(event)

            try:
                self._web_view.page().runJavaScript(
                    "window.__annolidExportUserState && window.__annolidExportUserState();",
                    _after,
                )
                return
            except Exception:
                pass
        self._append_reading_log_event({"type": "stop", "label": "Last stop"})

    def record_stop_event(self) -> None:
        self._record_stop_event()

    def go_to_anchor(self, page_num: int, offset_frac: float = 0.0) -> None:
        """Jump to a page/position for both PDF.js and PyMuPDF fallback."""
        page_num = int(page_num)
        offset_frac = float(offset_frac or 0.0)
        offset_frac = max(0.0, min(1.0, offset_frac))
        if self._web_view is not None and self._pdfjs_active:
            try:
                self._web_view.page().runJavaScript(
                    f"window.__annolidScrollToAnchor && window.__annolidScrollToAnchor({page_num}, {offset_frac});"
                )
            except Exception:
                pass
            return
        if self._doc is None:
            return
        target_index = max(0, page_num - 1)
        try:
            target_index = min(target_index, int(self._doc.page_count) - 1)
        except Exception:
            pass
        if target_index != self._current_page:
            self._current_page = target_index
            self._render_current_page()

    def activate_log_entry(self, entry: dict[str, object]) -> None:
        """Activate a reading log entry (jump + optional UI focus)."""
        try:
            page_num = int(entry.get("pageNum") or 0)
        except Exception:
            page_num = 0
        try:
            offset = float(entry.get("offsetFrac") or 0.0)
        except Exception:
            offset = 0.0
        if page_num <= 0:
            page_num = int(self._current_page) + 1
        self.go_to_anchor(page_num, offset)

        if self._web_view is None or not self._pdfjs_active:
            return
        note_id = entry.get("noteId") or entry.get("id")
        event_type = str(entry.get("type") or "")
        if event_type.startswith("note_") and note_id:
            try:
                self._web_view.page().runJavaScript(
                    f"window.__annolidOpenNotesAndSelect && window.__annolidOpenNotesAndSelect({json.dumps(str(note_id))});"
                )
            except Exception:
                pass
        if event_type in {"resume", "dblclick_read"}:
            # Ensure reader is enabled, then start reading from the stored anchor.
            try:
                if not self._reader_enabled:
                    self.set_reader_enabled(True)
            except Exception:
                pass
            try:
                self._web_view.page().runJavaScript(
                    f"window.__annolidStartReaderAtAnchor && window.__annolidStartReaderAtAnchor({page_num}, {offset});"
                )
            except Exception:
                pass
        if event_type in {"reader_stop", "reader_pause"}:
            # Resume from the last reader-stopped anchor if available.
            page_num2 = page_num
            offset2 = offset
            try:
                state = (
                    self._pdf_user_state
                    if isinstance(self._pdf_user_state, dict)
                    else {}
                )
                stopped = state.get("readingStopped")
                if isinstance(stopped, dict):
                    page_num2 = int(stopped.get("pageNum") or page_num2)
                    offset2 = float(stopped.get("offsetFrac") or offset2)
            except Exception:
                pass
            try:
                if not self._reader_enabled:
                    self.set_reader_enabled(True)
            except Exception:
                pass
            try:
                self._web_view.page().runJavaScript(
                    f"window.__annolidStartReaderAtAnchor && window.__annolidStartReaderAtAnchor({page_num2}, {offset2});"
                )
            except Exception:
                pass

    def _get_pdf_user_state(self, pdf_key: str) -> object:
        if not pdf_key or pdf_key != (self._pdf_key or ""):
            return {}
        return self._pdf_user_state if isinstance(self._pdf_user_state, dict) else {}

    def _handle_pdf_user_state_save(self, payload: object) -> None:
        if self._pdf_path is None:
            return
        if not isinstance(payload, dict):
            return
        pdf_key = payload.get("pdfKey")
        if pdf_key and str(pdf_key) != (self._pdf_key or ""):
            return
        state = payload.get("state", payload)
        if not isinstance(state, dict):
            return
        merged = dict(state)
        # Preserve any Python-side log entries if JS did not include them.
        if "log" not in merged and self._reading_log:
            merged["log"] = list(self._reading_log)
        # Preserve reading-stopped anchor if JS did not include it.
        if (
            "readingStopped" not in merged
            and isinstance(self._pdf_user_state, dict)
            and "readingStopped" in self._pdf_user_state
        ):
            merged["readingStopped"] = self._pdf_user_state.get("readingStopped")
        try:
            incoming_log = merged.get("log")
            if isinstance(incoming_log, list):
                self._reading_log = [e for e in incoming_log if isinstance(e, dict)][
                    :600
                ]
        except Exception:
            pass
        self._schedule_pdf_user_state_save(dict(merged), replace=True)

    def _handle_pdf_log_event(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        # payload may be {type,label,pageNum,offsetFrac,...}
        event = dict(payload)
        # Normalize some common fields.
        try:
            if "pageNum" in event:
                event["pageNum"] = int(event.get("pageNum") or 0)
        except Exception:
            pass
        try:
            if "offsetFrac" in event:
                event["offsetFrac"] = float(event.get("offsetFrac") or 0.0)
        except Exception:
            pass
        self._append_reading_log_event(event)

    def _clear_pdf_user_state(self, pdf_key: str) -> None:
        if not pdf_key or str(pdf_key) != (self._pdf_key or ""):
            return
        if self._pdf_path is None:
            return
        try:
            delete_pdf_state(self._pdf_path)
        except Exception:
            pass
        self._pdf_user_state = {}
        self._pdf_user_state_pending = None

    def _toggle_fallback_bookmark(self) -> None:
        if self._pdf_path is None:
            return
        page_num = int(self._current_page) + 1
        state = self._pdf_user_state if isinstance(self._pdf_user_state, dict) else {}
        bookmarks = (
            state.get("bookmarks") if isinstance(state.get("bookmarks"), list) else []
        )
        try:
            existing = next(
                (
                    i
                    for i, b in enumerate(bookmarks)
                    if isinstance(b, dict)
                    and int(b.get("pageNum", b.get("page", -1) + 1) or 0) == page_num
                ),
                -1,
            )
        except Exception:
            existing = -1
        if existing >= 0:
            try:
                bookmarks.pop(existing)
            except Exception:
                bookmarks = [
                    b
                    for b in bookmarks
                    if not (
                        isinstance(b, dict)
                        and int(b.get("pageNum", b.get("page", -1) + 1) or 0)
                        == page_num
                    )
                ]
        else:
            bookmarks.append(
                {
                    "pageNum": page_num,
                    "page": max(0, page_num - 1),
                    "title": f"Page {page_num}",
                    "createdAt": float(time.time()),
                }
            )
            self._append_reading_log_event(
                {"type": "bookmark_add", "label": "Bookmark added", "pageNum": page_num}
            )
        if existing >= 0:
            self._append_reading_log_event(
                {
                    "type": "bookmark_remove",
                    "label": "Bookmark removed",
                    "pageNum": page_num,
                }
            )
        try:
            bookmarks.sort(
                key=lambda b: int(b.get("pageNum", 0)) if isinstance(b, dict) else 0
            )
        except Exception:
            pass
        self._schedule_pdf_user_state_save({"bookmarks": bookmarks})

    def _add_fallback_note(self) -> None:
        if self._pdf_path is None:
            return
        page_num = int(self._current_page) + 1
        try:
            selected = self.text_view.textCursor().selectedText()
            snippet = (selected or "").replace("\u2029", "\n").strip()
        except Exception:
            snippet = ""
        comment, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            self.tr("Add Note"),
            self.tr("Comment"),
            "",
        )
        if not ok:
            return
        comment = (comment or "").strip()
        if not comment and not snippet:
            return

        state = self._pdf_user_state if isinstance(self._pdf_user_state, dict) else {}
        notes = state.get("notes") if isinstance(state.get("notes"), list) else []
        now = float(time.time())
        note_id = "note:" + uuid.uuid4().hex
        notes.insert(
            0,
            {
                "id": note_id,
                "pageNum": page_num,
                "page": max(0, page_num - 1),
                "offsetFrac": 0.0,
                "snippet": snippet[:400],
                "text": comment,
                "createdAt": now,
                "updatedAt": now,
            },
        )
        self._schedule_pdf_user_state_save({"notes": notes})
        self._append_reading_log_event(
            {
                "type": "note_add",
                "label": "Note added",
                "noteId": note_id,
                "pageNum": page_num,
                "snippet": snippet[:120],
            }
        )


class _DictionaryLookupTask(QtCore.QRunnable):
    """Background task to fetch dictionary definitions for a single word."""

    def __init__(self, widget: PdfViewerWidget, request_id: str, word: str) -> None:
        super().__init__()
        self.widget = widget
        self.request_id = request_id
        self.word = word

    @staticmethod
    def _format_html(word: str, payload: object) -> str:
        from html import escape

        safe_word = escape(word)
        if not isinstance(payload, list):
            return f"<h2>{safe_word}</h2><pre>{escape(str(payload))}</pre>"

        parts: list[str] = [f"<h2>{safe_word}</h2>"]

        phonetic = ""
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            phonetic = str(entry.get("phonetic") or "").strip()
            if phonetic:
                break
            for ph in entry.get("phonetics") or []:
                if isinstance(ph, dict):
                    phonetic = str(ph.get("text") or "").strip()
                    if phonetic:
                        break
            if phonetic:
                break
        if phonetic:
            parts.append(f"<p><i>{escape(phonetic)}</i></p>")

        for entry in payload[:2]:
            if not isinstance(entry, dict):
                continue
            meanings = entry.get("meanings") or []
            if not isinstance(meanings, list):
                continue
            for meaning in meanings[:6]:
                if not isinstance(meaning, dict):
                    continue
                pos = str(meaning.get("partOfSpeech") or "").strip()
                if pos:
                    parts.append(f"<h3>{escape(pos)}</h3>")
                defs = meaning.get("definitions") or []
                if not isinstance(defs, list):
                    continue
                items: list[str] = []
                for d in defs[:6]:
                    if not isinstance(d, dict):
                        continue
                    definition = str(d.get("definition") or "").strip()
                    if not definition:
                        continue
                    example = str(d.get("example") or "").strip()
                    block = f"<li>{escape(definition)}"
                    if example:
                        block += f"<br/><span style='color:#555'><i>Example: {escape(example)}</i></span>"
                    block += "</li>"
                    items.append(block)
                if items:
                    parts.append("<ol>" + "".join(items) + "</ol>")
        parts.append(
            "<p style='color:#777;font-size:11px'>Source: dictionaryapi.dev</p>"
        )
        return "".join(parts)

    @staticmethod
    def _lookup_macos_dictionary(word: str) -> Optional[str]:
        import ctypes
        import ctypes.util

        cf_path = ctypes.util.find_library("CoreFoundation")
        if not cf_path:
            return None
        try:
            core_foundation = ctypes.cdll.LoadLibrary(cf_path)
        except Exception:
            return None

        dictionary_services = None
        for path in (
            "/System/Library/Frameworks/CoreServices.framework/Frameworks/DictionaryServices.framework/DictionaryServices",
            "/System/Library/Frameworks/DictionaryServices.framework/DictionaryServices",
        ):
            try:
                dictionary_services = ctypes.cdll.LoadLibrary(path)
                break
            except Exception:
                continue
        if dictionary_services is None:
            return None

        kCFStringEncodingUTF8 = 0x08000100

        class CFRange(ctypes.Structure):
            _fields_ = [("location", ctypes.c_long), ("length", ctypes.c_long)]

        core_foundation.CFStringCreateWithCString.restype = ctypes.c_void_p
        core_foundation.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]

        core_foundation.CFStringGetLength.restype = ctypes.c_long
        core_foundation.CFStringGetLength.argtypes = [ctypes.c_void_p]

        core_foundation.CFStringGetMaximumSizeForEncoding.restype = ctypes.c_long
        core_foundation.CFStringGetMaximumSizeForEncoding.argtypes = [
            ctypes.c_long,
            ctypes.c_int32,
        ]

        core_foundation.CFStringGetCString.restype = ctypes.c_bool
        core_foundation.CFStringGetCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_int32,
        ]

        core_foundation.CFRelease.restype = None
        core_foundation.CFRelease.argtypes = [ctypes.c_void_p]

        dictionary_services.DCSCopyTextDefinition.restype = ctypes.c_void_p
        dictionary_services.DCSCopyTextDefinition.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            CFRange,
        ]

        cf_word = None
        cf_definition = None
        try:
            cf_word = core_foundation.CFStringCreateWithCString(
                None, word.encode("utf-8"), kCFStringEncodingUTF8
            )
            if not cf_word:
                return None
            # NOTE: CFRange uses character indices; for ASCII words len(word) is OK.
            cf_definition = dictionary_services.DCSCopyTextDefinition(
                None, cf_word, CFRange(0, len(word))
            )
            if not cf_definition:
                return None
            length = core_foundation.CFStringGetLength(cf_definition)
            max_size = (
                core_foundation.CFStringGetMaximumSizeForEncoding(
                    length, kCFStringEncodingUTF8
                )
                + 1
            )
            buffer = ctypes.create_string_buffer(max_size)
            ok = core_foundation.CFStringGetCString(
                cf_definition, buffer, max_size, kCFStringEncodingUTF8
            )
            if not ok:
                return None
            return buffer.value.decode("utf-8", errors="replace").strip()
        finally:
            try:
                if cf_definition:
                    core_foundation.CFRelease(cf_definition)
            except Exception:
                pass
            try:
                if cf_word:
                    core_foundation.CFRelease(cf_word)
            except Exception:
                pass

    def run(self) -> None:  # pragma: no cover - network + UI
        import sys

        word = (self.word or "").strip()
        html = ""
        error = ""
        try:
            if sys.platform == "darwin":
                definition = self._lookup_macos_dictionary(word)
                if definition:
                    from html import escape

                    html = (
                        f"<h2>{escape(word)}</h2>"
                        "<pre style='white-space:pre-wrap'>"
                        f"{escape(definition)}"
                        "</pre>"
                        "<p style='color:#777;font-size:11px'>Source: macOS Dictionary</p>"
                    )
            if not html:
                import requests
                from urllib.parse import quote

                url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote(word)}"
                response = requests.get(url, timeout=8)
                if response.status_code == 404:
                    error = f"No definition found for “{word}”."
                else:
                    response.raise_for_status()
                    html = self._format_html(word, response.json())
        except Exception as exc:
            error = f"Dictionary lookup failed: {exc}"

        try:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "_on_dictionary_lookup_finished",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, str(self.request_id)),
                QtCore.Q_ARG(str, str(word)),
                QtCore.Q_ARG(str, str(html)),
                QtCore.Q_ARG(str, str(error)),
            )
        except Exception:
            pass


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
            engine = (
                str(self.tts_settings.get("engine", "kokoro") or "kokoro")
                .strip()
                .lower()
            )
            max_chars = 800 if engine == "chatterbox" else 420
            chunks = self.chunks or self._chunk_text(text, max_chars=max_chars)
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
                from concurrent.futures import ThreadPoolExecutor
                from annolid.agents.tts_router import synthesize_tts

                def synthesize(chunk: str) -> tuple[object, int, int]:
                    audio_data = synthesize_tts(chunk, self.tts_settings)
                    if not audio_data:
                        raise RuntimeError("No audio returned by TTS engine")
                    samples, sample_rate = audio_data
                    duration_ms = 0
                    try:
                        duration_ms = int(
                            round((len(samples) / float(sample_rate)) * 1000)
                        )
                    except Exception:
                        duration_ms = 0
                    return samples, int(sample_rate), int(duration_ms)

                with ThreadPoolExecutor(max_workers=1) as executor:
                    current_future = executor.submit(synthesize, chunks[0])
                    for idx, chunk in enumerate(chunks):
                        if cancelled():
                            return
                        samples, sample_rate, duration_ms = current_future.result()
                        next_future = None
                        if idx + 1 < len(chunks):
                            next_future = executor.submit(synthesize, chunks[idx + 1])
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
                    chunk = sentence[i : i + max_chars].strip()
                    if chunk:
                        chunks.append(chunk)
        return chunks
