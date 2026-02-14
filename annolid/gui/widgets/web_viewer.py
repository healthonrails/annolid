from __future__ import annotations

import json
import time
from typing import Optional

from qtpy import QtCore, QtGui, QtWidgets
from annolid.utils.logger import logger

try:
    from qtpy import QtWebEngineWidgets  # type: ignore

    _WEBENGINE_AVAILABLE = True
except Exception:
    QtWebEngineWidgets = None  # type: ignore
    _WEBENGINE_AVAILABLE = False


def _is_ignorable_js_console_message(message: str) -> bool:
    value = str(message or "").strip().lower()
    if not value:
        return True
    noisy_markers = (
        "unrecognized feature: 'attribution-reporting'",
        "unrecognized feature: 'browsing-topics'",
        "deprecated api for given entry type",
        "window.webkitstorageinfo is deprecated",
        "three.webglprogram: gl.getprograminfolog() warning",
        "crossmark script out of date",
    )
    return any(marker in value for marker in noisy_markers)


if _WEBENGINE_AVAILABLE:

    class _AnnolidWebEnginePage(QtWebEngineWidgets.QWebEnginePage):
        """Embedded browser page with compatibility helpers and noise filtering."""

        def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
            super().__init__(parent)
            self._console_seen: dict[str, int] = {}
            self._console_last_cleanup = time.monotonic()
            self._install_compat_scripts()

        def _install_compat_scripts(self) -> None:
            # Add minimal polyfills for older QtWebEngine Chromium builds.
            compat_js = r"""
(() => {
  // Polyfill crypto.randomUUID() for older Chromium builds used by QtWebEngine.
  try {
    const c = (typeof globalThis !== "undefined" && globalThis.crypto) ? globalThis.crypto : null;
    if (c && typeof c.getRandomValues === "function" && typeof c.randomUUID !== "function") {
      c.randomUUID = function randomUUID() {
        const bytes = new Uint8Array(16);
        c.getRandomValues(bytes);
        bytes[6] = (bytes[6] & 0x0f) | 0x40; // RFC 4122 version 4
        bytes[8] = (bytes[8] & 0x3f) | 0x80; // RFC 4122 variant
        const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
        return (
          hex.slice(0, 8) + "-" +
          hex.slice(8, 12) + "-" +
          hex.slice(12, 16) + "-" +
          hex.slice(16, 20) + "-" +
          hex.slice(20)
        );
      };
    }
  } catch (e) {}

  // Polyfill .at() used by modern bundles.
  const defineAt = (proto) => {
    if (!proto || Object.prototype.hasOwnProperty.call(proto, "at")) return;
    Object.defineProperty(proto, "at", {
      value: function at(index) {
        const len = this == null ? 0 : this.length >>> 0;
        if (!len) return undefined;
        let i = Number(index) || 0;
        if (Number.isNaN(i)) i = 0;
        if (i < 0) i += len;
        if (i < 0 || i >= len) return undefined;
        return this[i];
      },
      writable: true,
      enumerable: false,
      configurable: true
    });
  };
  try {
    defineAt(Array.prototype);
    defineAt(String.prototype);
    if (typeof Int8Array !== "undefined") defineAt(Int8Array.prototype);
    if (typeof Uint8Array !== "undefined") defineAt(Uint8Array.prototype);
    if (typeof Uint8ClampedArray !== "undefined") defineAt(Uint8ClampedArray.prototype);
    if (typeof Int16Array !== "undefined") defineAt(Int16Array.prototype);
    if (typeof Uint16Array !== "undefined") defineAt(Uint16Array.prototype);
    if (typeof Int32Array !== "undefined") defineAt(Int32Array.prototype);
    if (typeof Uint32Array !== "undefined") defineAt(Uint32Array.prototype);
    if (typeof Float32Array !== "undefined") defineAt(Float32Array.prototype);
    if (typeof Float64Array !== "undefined") defineAt(Float64Array.prototype);
    if (typeof BigInt64Array !== "undefined") defineAt(BigInt64Array.prototype);
    if (typeof BigUint64Array !== "undefined") defineAt(BigUint64Array.prototype);
  } catch (e) {}

  // Fallback for unsupported :modal selector in older selector engines.
  try {
    const proto = (typeof Element !== "undefined" && Element.prototype) ? Element.prototype : null;
    const nativeMatches = proto && (proto.matches || proto.msMatchesSelector || proto.webkitMatchesSelector);
    if (proto && typeof nativeMatches === "function" && !proto.__annolidModalCompatPatched) {
      Object.defineProperty(proto, "__annolidModalCompatPatched", {
        value: true, writable: false, enumerable: false, configurable: true
      });
      proto.matches = function patchedMatches(selector) {
        try {
          return nativeMatches.call(this, selector);
        } catch (err) {
          const raw = String(selector || "");
          if (!raw || raw.indexOf(":modal") < 0) throw err;
          const hasNotModal = /:not\(\s*:modal\s*\)/.test(raw);
          const sanitized = raw
            .replace(/:not\(\s*:modal\s*\)/g, "")
            .replace(/:modal\b/g, "")
            .trim();
          if (!sanitized) {
            return hasNotModal;
          }
          try {
            const base = !!nativeMatches.call(this, sanitized);
            if (!base) return false;
            // We cannot reliably detect top-layer modal state on older engines.
            // Treat :modal as false and :not(:modal) as true after base selector match.
            return hasNotModal;
          } catch (innerErr) {
            throw err;
          }
        }
      };
    }
  } catch (e) {}
})();
            """.strip()
            try:
                script = QtWebEngineWidgets.QWebEngineScript()
                script.setName("annolid_polyfills")
                script.setSourceCode(compat_js)
                script.setInjectionPoint(
                    QtWebEngineWidgets.QWebEngineScript.DocumentCreation
                )
                script.setWorldId(QtWebEngineWidgets.QWebEngineScript.MainWorld)
                script.setRunsOnSubFrames(True)
                self.scripts().insert(script)
            except Exception:
                pass

        def _cleanup_console_seen(self) -> None:
            now = time.monotonic()
            if now - self._console_last_cleanup < 120:
                return
            if len(self._console_seen) > 500:
                self._console_seen.clear()
            self._console_last_cleanup = now

        def javaScriptConsoleMessage(  # noqa: N802 - Qt override
            self,
            level: "QtWebEngineWidgets.QWebEnginePage.JavaScriptConsoleMessageLevel",
            message: str,
            lineNumber: int,
            sourceID: str,
        ) -> None:
            msg = str(message or "").strip()
            if _is_ignorable_js_console_message(msg):
                return

            self._cleanup_console_seen()
            count = self._console_seen.get(msg, 0) + 1
            self._console_seen[msg] = count
            if count > 3:
                if count == 4:
                    logger.info("QtWebEngine js: suppressing repeated message: %s", msg)
                return

            try:
                info_level = getattr(
                    QtWebEngineWidgets.QWebEnginePage, "InfoMessageLevel", None
                )
                warning_level = getattr(
                    QtWebEngineWidgets.QWebEnginePage, "WarningMessageLevel", None
                )
                if warning_level is not None and level == warning_level:
                    logger.warning(
                        "QtWebEngine js: %s (%s:%s)", msg, sourceID, lineNumber
                    )
                elif info_level is not None and level == info_level:
                    logger.info("QtWebEngine js: %s (%s:%s)", msg, sourceID, lineNumber)
                else:
                    logger.error(
                        "QtWebEngine js: %s (%s:%s)", msg, sourceID, lineNumber
                    )
            except Exception:
                pass


class WebViewerWidget(QtWidgets.QWidget):
    """Simple embedded browser for opening web pages inside the shared canvas stack."""

    status_changed = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._web_view = None
        self._current_url = ""
        self._build_ui()

    @property
    def webengine_available(self) -> bool:
        return bool(_WEBENGINE_AVAILABLE)

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not _WEBENGINE_AVAILABLE:
            placeholder = QtWidgets.QLabel(
                "Qt WebEngine is unavailable. Embedded browser is disabled.", self
            )
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            placeholder.setWordWrap(True)
            root.addWidget(placeholder, 1)
            return

        toolbar = QtWidgets.QWidget(self)
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(6)

        self.back_button = QtWidgets.QToolButton(toolbar)
        self.back_button.setText("<")
        self.back_button.setToolTip("Back")
        toolbar_layout.addWidget(self.back_button)

        self.forward_button = QtWidgets.QToolButton(toolbar)
        self.forward_button.setText(">")
        self.forward_button.setToolTip("Forward")
        toolbar_layout.addWidget(self.forward_button)

        self.reload_button = QtWidgets.QToolButton(toolbar)
        self.reload_button.setText("Reload")
        toolbar_layout.addWidget(self.reload_button)

        self.url_edit = QtWidgets.QLineEdit(toolbar)
        self.url_edit.setPlaceholderText("https://")
        toolbar_layout.addWidget(self.url_edit, 1)

        self.open_in_browser_button = QtWidgets.QPushButton("Open in Browser", toolbar)
        toolbar_layout.addWidget(self.open_in_browser_button)
        root.addWidget(toolbar, 0)

        self._web_view = QtWebEngineWidgets.QWebEngineView(self)
        try:
            self._web_view.setPage(_AnnolidWebEnginePage(self._web_view))
        except Exception:
            pass
        root.addWidget(self._web_view, 1)

        self.back_button.clicked.connect(self._web_view.back)
        self.forward_button.clicked.connect(self._web_view.forward)
        self.reload_button.clicked.connect(self._web_view.reload)
        self.url_edit.returnPressed.connect(self._on_url_entered)
        self.open_in_browser_button.clicked.connect(self.open_current_in_browser)
        self._web_view.urlChanged.connect(self._on_url_changed)
        self._web_view.loadFinished.connect(self._on_load_finished)

    def _normalize_url(self, url: str) -> QtCore.QUrl:
        value = str(url or "").strip()
        if not value:
            return QtCore.QUrl()
        if "://" not in value:
            value = f"https://{value}"
        parsed = QtCore.QUrl(value)
        if parsed.scheme().lower() not in {"http", "https"}:
            return QtCore.QUrl()
        return parsed

    def load_url(self, url: str) -> bool:
        if self._web_view is None:
            return False
        parsed = self._normalize_url(url)
        if not parsed.isValid() or parsed.isEmpty():
            return False
        self._current_url = parsed.toString()
        self.url_edit.setText(self._current_url)
        self._web_view.setUrl(parsed)
        return True

    def open_current_in_browser(self) -> None:
        target = str(self.url_edit.text() or "").strip() or self._current_url
        parsed = self._normalize_url(target)
        if not parsed.isValid() or parsed.isEmpty():
            self.status_changed.emit("Invalid URL.")
            return
        QtGui.QDesktopServices.openUrl(parsed)
        self.status_changed.emit(f"Opened in system browser: {parsed.toString()}")

    def clear(self) -> None:
        if self._web_view is None:
            return
        self._current_url = ""
        self.url_edit.clear()
        self._web_view.setUrl(QtCore.QUrl("about:blank"))

    def get_state(self) -> dict:
        if self._web_view is None:
            return {
                "ok": False,
                "webengine_available": bool(_WEBENGINE_AVAILABLE),
                "has_page": False,
                "url": "",
                "title": "",
            }
        page = self._web_view.page()
        title = ""
        if page is not None:
            try:
                title = str(page.title() or "").strip()
            except Exception:
                title = ""
        current_url = ""
        try:
            current_url = str(self._web_view.url().toString() or "").strip()
        except Exception:
            current_url = str(self._current_url or "").strip()
        if not current_url:
            current_url = str(self._current_url or "").strip()
        has_page = bool(current_url) and current_url.lower() != "about:blank"
        return {
            "ok": True,
            "webengine_available": bool(_WEBENGINE_AVAILABLE),
            "has_page": bool(has_page),
            "url": current_url,
            "title": title,
        }

    def _run_js_sync(self, script: str, *, timeout_ms: int = 5000) -> object:
        if self._web_view is None:
            return {"error": "Embedded web view is unavailable."}
        page = self._web_view.page()
        if page is None:
            return {"error": "Web page object is unavailable."}

        loop = QtCore.QEventLoop(self)
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        result: dict[str, object] = {"done": False, "value": None}

        def _finish(value: object) -> None:
            if bool(result.get("done")):
                return
            result["done"] = True
            result["value"] = value
            loop.quit()

        timer.timeout.connect(lambda: _finish({"error": "JavaScript timed out."}))
        page.runJavaScript(script, _finish)
        timer.start(max(100, int(timeout_ms)))
        loop.exec_()
        timer.stop()
        return result.get("value")

    def get_page_text(self, max_chars: int = 8000) -> dict:
        if self._web_view is None:
            return {"ok": False, "error": "Embedded web view is unavailable."}
        limit = max(200, min(int(max_chars or 8000), 200000))
        script = """
(() => {
  const text = String((document && document.body && document.body.innerText) || "");
  const title = String((document && document.title) || "");
  const href = String((window && window.location && window.location.href) || "");
  return { ok: true, text, title, url: href, length: text.length };
})()
        """.strip()
        payload = self._run_js_sync(script)
        if not isinstance(payload, dict):
            return {"ok": False, "error": "Failed to read page text."}
        if payload.get("error"):
            return {"ok": False, "error": str(payload.get("error") or "")}
        text = str(payload.get("text") or "")
        truncated = len(text) > limit
        if truncated:
            text = text[:limit]
        return {
            "ok": True,
            "url": str(payload.get("url") or self._current_url),
            "title": str(payload.get("title") or ""),
            "text": text,
            "length": int(payload.get("length") or len(text)),
            "truncated": truncated,
        }

    def click_selector(self, selector: str) -> dict:
        value = str(selector or "").strip()
        if not value:
            return {"ok": False, "error": "selector is required"}
        selector_js = json.dumps(value)
        script = f"""
(() => {{
  const selector = {selector_js};
  const el = document.querySelector(selector);
  if (!el) return {{ ok: false, error: "Element not found", selector }};
  try {{ el.scrollIntoView({{ behavior: "instant", block: "center" }}); }} catch (e) {{}}
  try {{
    el.dispatchEvent(new MouseEvent("click", {{ bubbles: true, cancelable: true, view: window }}));
    if (typeof el.click === "function") el.click();
  }} catch (err) {{
    return {{ ok: false, error: String(err), selector }};
  }}
  const tag = String(el.tagName || "").toLowerCase();
  const text = String(el.innerText || el.textContent || "").trim();
  return {{ ok: true, selector, tag, text: text.slice(0, 200) }};
}})()
        """.strip()
        payload = self._run_js_sync(script)
        if isinstance(payload, dict):
            return dict(payload)
        return {"ok": False, "error": "Failed to click selector."}

    def type_selector(self, selector: str, text: str, submit: bool = False) -> dict:
        selector_value = str(selector or "").strip()
        if not selector_value:
            return {"ok": False, "error": "selector is required"}
        selector_js = json.dumps(selector_value)
        text_js = json.dumps(str(text or ""))
        submit_js = "true" if bool(submit) else "false"
        script = f"""
(() => {{
  const selector = {selector_js};
  const value = {text_js};
  const submit = {submit_js};
  const el = document.querySelector(selector);
  if (!el) return {{ ok: false, error: "Element not found", selector }};
  try {{ el.focus(); }} catch (e) {{}}

  const isInputLike = (
    el instanceof HTMLInputElement ||
    el instanceof HTMLTextAreaElement ||
    el.isContentEditable
  );
  if (!isInputLike) {{
    return {{ ok: false, error: "Element is not input-like", selector }};
  }}

  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {{
    el.value = value;
  }} else if (el.isContentEditable) {{
    el.textContent = value;
  }}
  el.dispatchEvent(new Event("input", {{ bubbles: true }}));
  el.dispatchEvent(new Event("change", {{ bubbles: true }}));

  let submitted = false;
  if (submit) {{
    const form = (el.form || el.closest("form"));
    if (form) {{
      try {{ form.requestSubmit ? form.requestSubmit() : form.submit(); submitted = true; }} catch (e) {{}}
    }} else {{
      try {{
        el.dispatchEvent(new KeyboardEvent("keydown", {{ key: "Enter", bubbles: true }}));
        el.dispatchEvent(new KeyboardEvent("keyup", {{ key: "Enter", bubbles: true }}));
      }} catch (e) {{}}
    }}
  }}
  return {{ ok: true, selector, typedChars: value.length, submitted }};
}})()
        """.strip()
        payload = self._run_js_sync(script)
        if isinstance(payload, dict):
            return dict(payload)
        return {"ok": False, "error": "Failed to type into selector."}

    def scroll_by(self, delta_y: int = 800) -> dict:
        amount = int(delta_y or 0)
        script = f"""
(() => {{
  const deltaY = {amount};
  window.scrollBy(0, deltaY);
  const y = Number(window.scrollY || window.pageYOffset || 0);
  const total = Number(
    (document && document.documentElement && document.documentElement.scrollHeight) ||
    (document && document.body && document.body.scrollHeight) || 0
  );
  return {{ ok: true, deltaY, scrollY: y, scrollHeight: total }};
}})()
        """.strip()
        payload = self._run_js_sync(script)
        if isinstance(payload, dict):
            return dict(payload)
        return {"ok": False, "error": "Failed to scroll page."}

    def find_forms(self) -> dict:
        script = """
(() => {
  const forms = Array.from(document.forms || []).slice(0, 50).map((form, i) => {
    const fields = Array.from(form.elements || []).slice(0, 200).map((el) => ({
      name: String(el.name || ""),
      id: String(el.id || ""),
      type: String(el.type || el.tagName || "").toLowerCase(),
      placeholder: String(el.placeholder || ""),
      required: !!el.required
    }));
    return {
      index: i,
      id: String(form.id || ""),
      name: String(form.name || ""),
      method: String(form.method || "get").toLowerCase(),
      action: String(form.action || ""),
      fieldCount: fields.length,
      fields
    };
  });
  return { ok: true, count: forms.length, forms };
})()
        """.strip()
        payload = self._run_js_sync(script)
        if isinstance(payload, dict):
            return dict(payload)
        return {"ok": False, "error": "Failed to inspect forms."}

    def _on_url_entered(self) -> None:
        text = str(self.url_edit.text() or "").strip()
        if not self.load_url(text):
            self.status_changed.emit("Invalid URL.")

    def _on_url_changed(self, url: QtCore.QUrl) -> None:
        text = url.toString()
        self._current_url = text
        self.url_edit.setText(text)

    def _on_load_finished(self, ok: bool) -> None:
        if ok:
            self.status_changed.emit(f"Loaded: {self._current_url}")
            return
        self.status_changed.emit("Failed to load page.")
