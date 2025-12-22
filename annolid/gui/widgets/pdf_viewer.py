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
                if plugins_attr is not None:
                    settings.setAttribute(plugins_attr, True)
                if pdf_attr is not None:
                    settings.setAttribute(pdf_attr, True)
                    self._web_pdf_capable = settings.testAttribute(pdf_attr)
                logger.info(
                    "QtWebEngine settings: "
                    f"PluginsEnabled={settings.testAttribute(plugins_attr) if plugins_attr else 'n/a'} "
                    f"PdfViewerEnabled={settings.testAttribute(pdf_attr) if pdf_attr else 'n/a'}"
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
        self._use_web_engine = bool(
            _WEBENGINE_AVAILABLE and self._web_pdf_capable)
        if _WEBENGINE_AVAILABLE and not self._web_pdf_capable:
            logger.warning(
                "QtWebEngine is available but PDF viewer support appears disabled; falling back to PyMuPDF."
            )

    def load_pdf(self, pdf_path: str) -> None:
        """Load a PDF file and render the first page."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Preferred: QtWebEngine (Chromium PDF viewer).
        if self._use_web_engine and self._web_view is not None:
            from qtpy import QtCore

            logger.info(f"Loading PDF with QtWebEngine: {path}")
            self._pdfjs_active = False
            self._stack.setCurrentWidget(self._web_container)
            self._set_controls_for_web(True)
            self._web_loading_path = path
            self._load_web_embed_pdf(path)
            if self._doc is not None:
                self._doc.close()
                self._doc = None
            return

        # Fallback: PyMuPDF rendering.
        logger.info(f"Loading PDF with PyMuPDF fallback: {path}")
        self._pdfjs_active = False
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
        if not ok or path is None:
            if path:
                self._fallback_from_web(path, "loadFinished returned False")
            self._web_loading_path = None
            return

        if self._pdfjs_active:
            # Verify PDF.js actually rendered; otherwise fall back.
            def _after_pdfjs_probe(result: object) -> None:
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
                    self._fallback_from_web(
                        path, "PDF.js rendered no text spans")
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
    }}
    .textLayer span {{
      position: absolute;
      white-space: pre;
      transform-origin: 0% 0%;
      color: transparent;
    }}
    .textLayer ::selection {{
      background: rgba(0, 120, 215, 0.35);
    }}
    .textLayer .annolid-speaking {{
      background: rgba(255, 210, 80, 0.45);
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
        window.__annolidLastHighlight = null;
        window.__annolidHighlightSelection = function() {{
          if (!window.__annolidSpans || !window.__annolidSelectionSpans) return;
          window.__annolidSpans.forEach(span => span.classList.remove("annolid-speaking"));
          window.__annolidSelectionSpans.forEach(idx => {{
            const span = window.__annolidSpans[idx];
            if (span) span.classList.add("annolid-speaking");
          }});
        }};
        window.__annolidHighlightSpanIndex = function(idx) {{
          if (!window.__annolidSpans) return;
          if (window.__annolidLastHighlight !== null) {{
            const prev = window.__annolidSpans[window.__annolidLastHighlight];
            if (prev) prev.classList.remove("annolid-speaking");
          }}
          const span = window.__annolidSpans[idx];
          if (span) {{
            span.classList.add("annolid-speaking");
            window.__annolidLastHighlight = idx;
          }}
        }};
        window.__annolidHighlightSpanIndices = function(indices) {{
          if (!window.__annolidSpans) return;
          window.__annolidSpans.forEach(span => span.classList.remove("annolid-speaking"));
          if (!indices || !indices.length) return;
          indices.forEach(idx => {{
            const span = window.__annolidSpans[idx];
            if (span) span.classList.add("annolid-speaking");
          }});
          window.__annolidLastHighlight = indices[indices.length - 1];
        }};
        window.__annolidClearHighlight = function() {{
          if (!window.__annolidSpans) return;
          window.__annolidSpans.forEach(span => span.classList.remove("annolid-speaking"));
          window.__annolidLastHighlight = null;
        }};
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
          pageDiv.style.width = viewport.width + "px";
          pageDiv.style.height = viewport.height + "px";

          const canvas = document.createElement("canvas");
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          pageDiv.appendChild(canvas);

          const textLayerDiv = document.createElement("div");
          textLayerDiv.className = "textLayer";
          pageDiv.appendChild(textLayerDiv);

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
            if 0 <= index < len(self._text_sentence_spans):
                start, end = self._text_sentence_spans[index]
                cursor = self.text_view.textCursor()
                cursor.setPosition(start)
                cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
                self.text_view.setTextCursor(cursor)
            return
        if self._highlight_mode == "web-sentence":
            if 0 <= index < len(self._web_sentence_span_groups):
                span_indices = self._web_sentence_span_groups[index]
                if self._web_view is not None:
                    try:
                        self._web_view.page().runJavaScript(
                            "window.__annolidHighlightSpanIndices && "
                            f"window.__annolidHighlightSpanIndices({span_indices})"
                        )
                    except Exception:
                        pass

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
