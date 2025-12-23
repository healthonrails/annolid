from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Dict, Optional

from qtpy import QtCore, QtGui, QtWidgets

try:
    from qtpy import QtWebEngineWidgets  # type: ignore

    _WEBENGINE_AVAILABLE = True
except Exception:
    QtWebEngineWidgets = None  # type: ignore
    _WEBENGINE_AVAILABLE = False

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


class PdfViewerWidget(QtWidgets.QWidget):
    """PDF viewer that prefers an embedded browser (if available) with fallback rendering."""

    selection_ready = QtCore.Signal(str)
    page_changed = QtCore.Signal(int, int)
    controls_enabled_changed = QtCore.Signal(bool)
    bookmarks_changed = QtCore.Signal(list)

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
        self._force_pdfjs = False
        self._bookmarks: list[dict[str, object]] = []
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
        self._use_web_engine = bool(_WEBENGINE_AVAILABLE and self._web_view is not None)
        if _WEBENGINE_AVAILABLE and not self._web_pdf_capable:
            logger.info(
                "QtWebEngine PDF plugin support appears disabled; PDF.js will be used instead."
            )

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
            def probe_pdfjs(attempts_left: int = 12) -> None:
                def _after_pdfjs_probe(result: object) -> None:
                    # Abort if another PDF load started meanwhile.
                    if self._web_loading_path is not None and self._web_loading_path != path:
                        return

                    err = ""
                    spans = 0
                    try:
                        if isinstance(result, dict):
                            err = str(result.get("err", "") or "")
                            spans = int(result.get("spans", 0))
                    except Exception:
                        err = ""
                        spans = 0
                    if err:
                        self._fallback_from_web(path, f"PDF.js error: {err}")
                        return
                    if spans <= 0:
                        if attempts_left > 0:
                            QtCore.QTimer.singleShot(
                                250, lambda: probe_pdfjs(attempts_left - 1)
                            )
                            return
                        self._fallback_from_web(path, "PDF.js rendered no text spans")
                        return
                    self._pdf_path = path
                    self._web_loading_path = None
                    logger.info(f"QtWebEngine PDF.js viewer active for {path}")

                try:
                    self._web_view.page().runJavaScript(
                        """(() => {
  const err = document.body ? (document.body.getAttribute("data-pdfjs-error") || "") : "";
  const spans = (window.__annolidSpans || []).length;
  return {err, spans};
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
        # Use CDN for PDF.js; if it fails, we fall back to PyMuPDF.
        pdfjs_version = "2.16.105"  # Compatible with older Chromium in Qt 5.15
        base_url = QtCore.QUrl.fromLocalFile(str(path.parent) + "/")
        pdf_url = QtCore.QUrl.fromLocalFile(str(path)).toString()
        pdf_b64 = ""
        try:
            if path.stat().st_size <= 12 * 1024 * 1024:
                pdf_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        except Exception as exc:
            logger.info(f"Failed to base64-encode PDF for PDF.js: {exc}")
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
    }}
    #viewerContainer {{
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      overflow: auto;
      background: #1e1e1e;
      padding-top: 52px;
    }}
    #annolidToolbar {{
      position: fixed;
      top: 8px;
      left: 8px;
      z-index: 9999;
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 8px;
      background: rgba(32, 33, 36, 0.92);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 10px;
      font: 13px system-ui, -apple-system, Segoe UI, sans-serif;
      user-select: none;
      -webkit-user-select: none;
    }}
    #annolidToolbar button {{
      background: #303134;
      color: #e8eaed;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 8px;
      padding: 5px 10px;
      cursor: pointer;
    }}
    #annolidToolbar button.annolid-active {{
      background: #1a73e8;
      border-color: #1a73e8;
      color: white;
    }}
    #annolidToolbar .annolid-sep {{
      width: 1px;
      height: 22px;
      background: rgba(255, 255, 255, 0.14);
      margin: 0 2px;
    }}
    #annolidToolbar label {{
      opacity: 0.9;
      font-size: 12px;
    }}
    #annolidToolbar input[type="color"] {{
      width: 32px;
      height: 28px;
      padding: 0;
      border: 0;
      background: transparent;
      cursor: pointer;
    }}
    #annolidToolbar input[type="range"] {{
      width: 92px;
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
  </style>
  <script>
    // Polyfill `.at()` for older Chromium (QtWebEngine 5.15).
    function _atPolyfill(n) {{
      n = Math.trunc(n) || 0;
      if (n < 0) n += this.length;
      if (n < 0 || n >= this.length) return undefined;
      return this[n];
    }}
    if (!Array.prototype.at) {{
      Array.prototype.at = _atPolyfill;
    }}
    if (!String.prototype.at) {{
      String.prototype.at = function(n) {{
        n = Math.trunc(n) || 0;
        if (n < 0) n += this.length;
        if (n < 0 || n >= this.length) return undefined;
        return this.charAt(n);
      }};
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
      if (T && T.prototype && !T.prototype.at) {{
        T.prototype.at = _atPolyfill;
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
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/{pdfjs_version}/pdf.min.js"></script>
  <script>
    const pdfUrl = "{pdf_url}";
    const pdfBase64 = "{pdf_b64}";
    // Avoid WebWorker requirements in older Chromium (no structuredClone).
    pdfjsLib.disableWorker = true;
    document.addEventListener("DOMContentLoaded", async () => {{
      try {{
        window.__annolidSpans = [];
        window.__annolidSelectionSpans = [];
        window.__annolidSpanMeta = {{}};
        window.__annolidPages = {{}};
        window.__annolidTts = {{ sentenceIndices: [], wordIndex: null, lastPages: [] }};
        window.__annolidMarks = {{ tool: "select", color: "#ffb300", size: 10, undo: [], drawing: null }};

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
            return window.__annolidSpanMeta[idx];
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
          const meta = {{
            pageNum,
            x: spanRect.left - pageRect.left,
            y: spanRect.top - pageRect.top,
            w: spanRect.width,
            h: spanRect.height,
          }};
          if (window.__annolidSpanMeta) window.__annolidSpanMeta[idx] = meta;
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

        if (selectBtn) selectBtn.addEventListener("click", () => _annolidSetTool("select"));
        if (penBtn) penBtn.addEventListener("click", () => _annolidSetTool("pen"));
        if (hiBtn) hiBtn.addEventListener("click", () => _annolidSetTool("highlighter"));
        if (highlightBtn) highlightBtn.addEventListener("click", _annolidManualHighlightSelection);
        if (undoBtn) undoBtn.addEventListener("click", _annolidUndo);
        if (clearBtn) clearBtn.addEventListener("click", _annolidClearMarks);

        if (colorInput) {{
          colorInput.addEventListener("input", (ev) => {{
            window.__annolidMarks.color = ev.target.value || "#ffb300";
          }});
        }}
        if (sizeInput) {{
          sizeInput.addEventListener("input", (ev) => {{
            const v = parseFloat(ev.target.value || "10");
            window.__annolidMarks.size = isFinite(v) ? v : 10;
          }});
        }}

        _annolidSetTool("select");
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
            disableWorker: true,
          }});
        }} else {{
          loadingTask = pdfjsLib.getDocument({{
            url: pdfUrl,
            disableWorker: true,
          }});
        }}
        const pdf = await loadingTask.promise;
        const container = document.getElementById("viewerContainer");
        const scale = 1.25;
        let nextPage = 1;
        const total = pdf.numPages || 1;

        async function renderPage(pageNum) {{
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

          container.appendChild(pageDiv);

          const ctx = canvas.getContext("2d");
          await page.render({{ canvasContext: ctx, viewport }}).promise;

          try {{
            const textContent = await page.getTextContent();
            if (pdfjsLib.renderTextLayer) {{
              const task = pdfjsLib.renderTextLayer({{
                textContent,
                container: textLayerDiv,
                viewport,
                textDivs: [],
                enhanceTextSelection: true,
              }});
              if (task && task.promise) {{
                await task.promise;
              }}
            }}
          }} catch (e) {{
            console.warn("PDF.js text layer failed", e);
          }}
          window.__annolidSpans = Array.from(document.querySelectorAll(".textLayer span"));
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

        async function renderMore(maxCount) {{
          let count = 0;
          while (nextPage <= total && count < maxCount) {{
            await renderPage(nextPage);
            nextPage += 1;
            count += 1;
            await new Promise(r => setTimeout(r, 0));
          }}
        }}

        await renderMore(2);
        container.addEventListener("scroll", async () => {{
          const nearBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 600;
          if (nearBottom) {{
            await renderMore(2);
          }}
        }});
      }} catch (err) {{
        console.error("PDF.js render failed", err);
        document.body.setAttribute("data-pdfjs-error", err.toString());
      }}
    }});
  </script>
</head>
<body>
  <div id="annolidToolbar">
    <button id="annolidToolSelect" class="annolid-active">Select</button>
    <button id="annolidToolPen">Pen</button>
    <button id="annolidToolHighlighter">Highlighter</button>
    <span class="annolid-sep"></span>
    <button id="annolidHighlightSelection">Highlight selection</button>
    <button id="annolidUndo">Undo</button>
    <button id="annolidClear">Clear</button>
    <span class="annolid-sep"></span>
    <label for="annolidColor">Color</label>
    <input id="annolidColor" type="color" value="#ffb300" />
    <label for="annolidSize">Size</label>
    <input id="annolidSize" type="range" min="2" max="24" value="10" />
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
            self, cleaned, merged, chunks=chunks))

    @QtCore.Slot()
    def _on_speak_finished(self) -> None:
        self._speaking = False
        self._clear_highlight()

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
            if 0 <= index < len(self._web_sentence_span_groups):
                span_indices = self._web_sentence_span_groups[index]
                if self._web_view is not None:
                    try:
                        self._web_view.page().runJavaScript(
                            "window.__annolidHighlightSentenceIndices && "
                            f"window.__annolidHighlightSentenceIndices({span_indices})"
                        )
                    except Exception:
                        pass

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
            if not (0 <= index < len(self._web_sentence_span_groups)):
                return
            span_indices = self._web_sentence_span_groups[index]
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
        if self._web_view is not None and self._highlight_mode in {"web", "web-sentence"}:
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
        if self._web_view is not None and self._highlight_mode in {"web", "web-sentence"}:
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
                max(1, int(self._word_highlight_durations_ms[self._word_highlight_index]))
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
                    doc, sentence_span[0], sentence_span[1], QtGui.QColor(255, 210, 80, 90)
                ))
            if word_span is not None:
                selections.append(self._make_text_extra_selection(
                    doc, word_span[0], word_span[1], QtGui.QColor(255, 160, 0, 160)
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
    ) -> None:
        super().__init__()
        self.widget = widget
        self.text = text
        self.tts_settings = tts_settings
        self.chunks = chunks

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
                                round((len(samples) / float(audio.frame_rate)) * 1000)
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
