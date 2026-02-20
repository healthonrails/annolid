from __future__ import annotations

import base64
import binascii
import contextlib
import json
import mimetypes
import re
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, Optional

from qtpy import QtCore, QtGui, QtWidgets
from annolid.core.agent.utils import get_agent_workspace_path
from annolid.gui.widgets.bot_explain import (
    explain_image_with_annolid_bot,
    explain_selection_with_annolid_bot,
)
from annolid.gui.widgets.dictionary_lookup import DictionaryLookupTask
from annolid.utils.logger import logger


import os

# The macOS QtWebEngine sandbox patches (fixing dyld and ICU mmap errors)
# have been moved to `annolid.utils.macos_fixes.apply_macos_webengine_sandbox_patch`
# and are executed early in `annolid/gui/app.py` before QApplication initialization.

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
    noisy_exact = {
        "error",
        "[object object]",
    }
    if value in noisy_exact:
        return True
    noisy_markers = (
        "unrecognized feature: 'attribution-reporting'",
        "unrecognized feature: 'browsing-topics'",
        "deprecated api for given entry type",
        "window.webkitstorageinfo is deprecated",
        "three.webglprogram: gl.getprograminfolog() warning",
        "crossmark script out of date",
        "failed to find a valid digest in the 'integrity' attribute for resource",
        "uncaught referenceerror: solvesimplechallenge is not defined",
        "uncaught typeerror: cannot read property 'style' of undefined",
        "atom change detected, updating - store value:",
        "rangeerror: value longoffset out of range for intl.datetimeformat options property timezonename",
        "was preloaded using link preload but not used within a few seconds from the window's load event",
        # QtWebEngine CSP warnings about external site CSP headers (not our code)
        "the source list for content security policy directive",
        "contains an invalid source",
        # Common external page JavaScript errors that are not actionable
        "uncaught referenceerror: _d is not defined",
        # CORS errors for external site fonts - these are external site issues, not actionable
        "access to font at",
        "has been blocked by cors policy",
        "no 'access-control-allow-origin' header is present on the requested resource",
    )
    return any(marker in value for marker in noisy_markers)


class _SavePdfTask(QtCore.QRunnable):
    """Background task that downloads a PDF URL and reports the saved path."""

    def __init__(
        self,
        widget: QtCore.QObject,
        *,
        source_url: str,
        download_url: str,
        output_dir: Path,
        max_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        super().__init__()
        self.widget = widget
        self.source_url = str(source_url or "").strip()
        self.download_url = str(download_url or "").strip()
        self.output_dir = Path(output_dir).expanduser()
        self.max_bytes = max(1, int(max_bytes or 0))

    @staticmethod
    def _safe_name(name: str) -> str:
        cleaned = "".join(
            ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_"
            for ch in str(name or "")
        ).strip("._")
        return cleaned or "download.pdf"

    @staticmethod
    def _filename_from_content_disposition(header: str) -> str:
        text = str(header or "").strip()
        if not text:
            return ""
        match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', text, re.I)
        if not match:
            return ""
        raw = str(match.group(1) or "").strip()
        if not raw:
            return ""
        return urllib.parse.unquote(raw)

    @staticmethod
    def _filename_from_url(url: str) -> str:
        parsed = urllib.parse.urlparse(str(url or "").strip())
        name = Path(urllib.parse.unquote(parsed.path or "")).name
        return name or ""

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        for idx in range(2, 1000):
            candidate = path.with_name(f"{stem}_{idx}{suffix}")
            if not candidate.exists():
                return candidate
        return path

    @staticmethod
    def _candidate_download_urls(url: str) -> list[str]:
        primary = str(url or "").strip()
        if not primary:
            return []
        candidates = [primary]
        parsed = urllib.parse.urlparse(primary)
        host = str(parsed.netloc or "").lower()
        if "pmc.ncbi.nlm.nih.gov" in host and "/pdf/" in str(parsed.path or "").lower():
            match = re.search(r"\b(PMC\d+)\b", primary, flags=re.IGNORECASE)
            if match:
                pmcid = match.group(1).upper()
                candidates.insert(
                    0,
                    f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf",
                )
            has_download_query = "download=" in str(parsed.query or "").lower()
            if not has_download_query:
                query = str(parsed.query or "").strip()
                query = f"{query}&download=1" if query else "download=1"
                fallback = urllib.parse.urlunparse(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        parsed.path,
                        parsed.params,
                        query,
                        parsed.fragment,
                    )
                )
                candidates.append(fallback)
        return candidates

    def run(self) -> None:  # pragma: no cover - network + filesystem
        saved_path = ""
        message = ""
        error = ""
        output_path: Optional[Path] = None
        last_exc: Optional[Exception] = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            urls = self._candidate_download_urls(self.download_url)
            content_type = ""
            for candidate_url in urls:
                request = urllib.request.Request(
                    candidate_url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/122.0.0.0 Safari/537.36"
                        ),
                        "Accept": "application/pdf,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": self.source_url,
                    },
                    method="GET",
                )
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        content_type = str(
                            response.headers.get("Content-Type") or ""
                        ).lower()
                        disposition = str(
                            response.headers.get("Content-Disposition") or ""
                        ).strip()
                        resolved_url = (
                            str(response.geturl() or "").strip() or candidate_url
                        )
                        file_name = (
                            self._filename_from_content_disposition(disposition)
                            or self._filename_from_url(resolved_url)
                            or self._filename_from_url(self.source_url)
                            or "download.pdf"
                        )
                        file_name = self._safe_name(file_name)
                        if not file_name.lower().endswith(".pdf"):
                            file_name = f"{Path(file_name).stem}.pdf"
                        output_path = self._unique_path(self.output_dir / file_name)
                        size = 0
                        with output_path.open("wb") as handle:
                            while True:
                                chunk = response.read(8192)
                                if not chunk:
                                    break
                                size += len(chunk)
                                if size > self.max_bytes:
                                    raise ValueError(
                                        f"PDF is too large ({size} bytes). Max is {self.max_bytes} bytes."
                                    )
                                handle.write(chunk)
                    break
                except Exception as exc:
                    last_exc = exc
                    with contextlib.suppress(OSError):
                        if output_path is not None and output_path.exists():
                            output_path.unlink()
                    output_path = None
                    continue

            if output_path is None:
                if last_exc is not None:
                    raise last_exc
                raise ValueError("No valid download URL resolved.")

            with output_path.open("rb") as handle:
                header_bytes = handle.read(5)
            if header_bytes != b"%PDF-":
                with output_path.open("rb") as handle:
                    preview = handle.read(256)
                if "application/pdf" not in content_type and b"%PDF-" not in preview:
                    output_path.unlink(missing_ok=True)
                    raise ValueError("Downloaded file is not a valid PDF.")

            saved_path = str(output_path)
            message = f"Saved PDF: {output_path.name}"
        except Exception as exc:
            with contextlib.suppress(OSError):
                if output_path is not None and output_path.exists():
                    output_path.unlink()
            error = f"Failed to save PDF: {exc}"

        QtCore.QMetaObject.invokeMethod(
            self.widget,
            "_on_pdf_saved_from_web",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, saved_path),
            QtCore.Q_ARG(str, message),
            QtCore.Q_ARG(str, error),
        )


if _WEBENGINE_AVAILABLE:

    class _AnnolidWebEnginePage(QtWebEngineWidgets.QWebEnginePage):
        """Embedded browser page with compatibility helpers and noise filtering."""

        def __init__(
            self,
            parent: Optional[QtCore.QObject] = None,
            profile: Optional[QtCore.QObject] = None,
            open_url_callback=None,
        ) -> None:
            if profile is not None:
                try:
                    super().__init__(profile, parent)
                except Exception:
                    super().__init__(parent)
            else:
                super().__init__(parent)
            self._open_url_callback = open_url_callback
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

        def createWindow(  # noqa: N802 - Qt override
            self,
            windowType: "QtWebEngineWidgets.QWebEnginePage.WebWindowType",
        ) -> "QtWebEngineWidgets.QWebEnginePage":
            page = None
            try:
                page = _AnnolidWebEnginePage(
                    parent=self,
                    profile=self.profile(),
                    open_url_callback=self._open_url_callback,
                )
            except Exception:
                page = _AnnolidWebEnginePage(
                    parent=self,
                    open_url_callback=self._open_url_callback,
                )

            def _forward_url(url: QtCore.QUrl) -> None:
                target = str(url.toString() if url is not None else "").strip()
                if not target:
                    return
                callback = getattr(self, "_open_url_callback", None)
                if callable(callback):
                    try:
                        callback(target)
                    except Exception:
                        pass
                QtCore.QTimer.singleShot(0, page.deleteLater)

            try:
                page.urlChanged.connect(_forward_url)
            except Exception:
                pass
            return page

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


def _create_ephemeral_web_profile(
    parent: Optional[QtCore.QObject],
) -> Optional[QtCore.QObject]:
    if not _WEBENGINE_AVAILABLE:
        return None
    try:
        profile = QtWebEngineWidgets.QWebEngineProfile(parent)
        cookie_policy = getattr(
            QtWebEngineWidgets.QWebEngineProfile, "NoPersistentCookies", None
        )
        if cookie_policy is not None:
            profile.setPersistentCookiesPolicy(cookie_policy)
        cache_kind = getattr(
            QtWebEngineWidgets.QWebEngineProfile, "MemoryHttpCache", None
        )
        if cache_kind is not None:
            profile.setHttpCacheType(cache_kind)
        return profile
    except Exception:
        return None


class WebViewerWidget(QtWidgets.QWidget):
    """Simple embedded browser for opening web pages inside the shared canvas stack."""

    status_changed = QtCore.Signal(str)
    close_requested = QtCore.Signal()
    _ZOOM_MIN = 0.25
    _ZOOM_MAX = 5.0
    _ZOOM_STEP = 0.1

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._web_view = None
        self._current_url = ""
        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._pdf_save_in_progress = False
        self._speaking = False
        self._active_speak_token: Optional[_SpeakToken] = None
        self._dictionary_lookup_id = ""
        self._dictionary_popup_pos: Optional[QtCore.QPoint] = None
        self._active_dictionary_dialog: Optional[QtWidgets.QDialog] = None
        self._js_running = False
        self._last_scrape_time = 0.0
        self._shortcuts: list[QtWidgets.QShortcut] = []
        self._web_profile = None
        self._build_ui()

    @property
    def webengine_available(self) -> bool:
        return bool(_WEBENGINE_AVAILABLE)

    @staticmethod
    def _apply_nav_icon(
        button: QtWidgets.QToolButton, theme_icon: str, fallback: str
    ) -> None:
        """Apply navigation icon to button using theme or fallback."""
        icon = QtGui.QIcon.fromTheme(theme_icon)
        if icon.isNull():
            # Use text fallback if theme icon is not available
            button.setText(fallback)
            font = button.font()
            font.setPointSize(14)
            button.setFont(font)
        else:
            button.setIcon(icon)
            button.setIconSize(QtCore.QSize(18, 18))

    def _update_nav_buttons(self) -> None:
        """Update navigation button states based on history."""
        if self._web_view is not None:
            self.back_button.setEnabled(self._web_view.history().canGoBack())
            self.forward_button.setEnabled(self._web_view.history().canGoForward())

        url = self._current_url
        is_pdf = self._is_saveable_pdf_url(url) if url else False
        if hasattr(self, "save_pdf_button"):
            self.save_pdf_button.setVisible(is_pdf)

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

        # Create Chrome-style toolbar
        toolbar = QtWidgets.QWidget(self)
        toolbar.setObjectName("webViewerToolbar")
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(4)

        # Navigation button style
        nav_button_style = """
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 4px;
                color: #5f6368;
            }
            QToolButton:hover {
                background-color: #e8eaed;
            }
            QToolButton:pressed {
                background-color: #dfe0e0;
            }
            QToolButton:disabled {
                color: #9aa0a6;
            }
        """

        # Back button
        self.back_button = QtWidgets.QToolButton(toolbar)
        self.back_button.setToolTip("Go back")
        self.back_button.setEnabled(False)
        self._apply_nav_icon(self.back_button, "go-previous", "←")
        self.back_button.setStyleSheet(nav_button_style)
        toolbar_layout.addWidget(self.back_button)

        # Forward button
        self.forward_button = QtWidgets.QToolButton(toolbar)
        self.forward_button.setToolTip("Go forward")
        self.forward_button.setEnabled(False)
        self._apply_nav_icon(self.forward_button, "go-next", "→")
        self.forward_button.setStyleSheet(nav_button_style)
        toolbar_layout.addWidget(self.forward_button)

        # Reload button
        self.reload_button = QtWidgets.QToolButton(toolbar)
        self.reload_button.setToolTip("Reload this page")
        self._apply_nav_icon(self.reload_button, "view-refresh", "↻")
        self.reload_button.setStyleSheet(nav_button_style)
        toolbar_layout.addWidget(self.reload_button)

        # Address bar (Chrome-style Omnibox)
        self.url_edit = QtWidgets.QLineEdit(toolbar)
        self.url_edit.setPlaceholderText("Search or enter URL")
        self.url_edit.setObjectName("webViewerUrlEdit")

        # Chrome-style address bar styling
        address_bar_style = """
            QLineEdit {
                background-color: #f1f3f4;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
                color: #202124;
            }
            QLineEdit:focus {
                background-color: #ffffff;
                border: 2px solid #4285f4;
                padding: 6px 10px;
            }
            QLineEdit:hover {
                background-color: #e8eaed;
            }
        """

        self.url_edit.setStyleSheet(address_bar_style)
        toolbar_layout.addWidget(self.url_edit, 1)

        # Lock/security icon placeholder
        self.security_icon = QtWidgets.QLabel(toolbar)
        self.security_icon.setText("🔒")
        self.security_icon.setToolTip("Connection is secure")
        self.security_icon.setStyleSheet("padding: 2px;")
        toolbar_layout.addWidget(self.security_icon)

        # Open in browser button (Chrome-style)
        self.open_in_browser_button = QtWidgets.QToolButton(toolbar)
        self.open_in_browser_button.setToolTip("Open in default browser")
        self._apply_nav_icon(self.open_in_browser_button, "external-browser", "↗")
        self.open_in_browser_button.setStyleSheet(nav_button_style)
        toolbar_layout.addWidget(self.open_in_browser_button)

        # Save PDF button
        self.save_pdf_button = QtWidgets.QToolButton(toolbar)
        self.save_pdf_button.setToolTip(
            "Save this PDF and open it in Annolid PDF Viewer"
        )
        self._apply_nav_icon(self.save_pdf_button, "document-save", "📥")
        self.save_pdf_button.setStyleSheet(nav_button_style)
        self.save_pdf_button.setVisible(False)
        toolbar_layout.addWidget(self.save_pdf_button)

        # Close button to close the web viewer tab
        self.close_button = QtWidgets.QToolButton(toolbar)
        self.close_button.setToolTip("Close this tab")
        self._apply_nav_icon(self.close_button, "window-close", "✕")
        self.close_button.setStyleSheet(nav_button_style)
        toolbar_layout.addWidget(self.close_button)

        root.addWidget(toolbar, 0)

        self._web_view = QtWebEngineWidgets.QWebEngineView(self)
        try:
            self._web_profile = _create_ephemeral_web_profile(self._web_view)
            if self._web_profile is not None:
                self._web_view.setPage(
                    _AnnolidWebEnginePage(
                        self._web_view,
                        profile=self._web_profile,
                        open_url_callback=self._open_url_from_page,
                    )
                )
            else:
                self._web_view.setPage(
                    _AnnolidWebEnginePage(
                        self._web_view,
                        open_url_callback=self._open_url_from_page,
                    )
                )
        except Exception:
            pass
        # Enable Chromium built-in PDF viewer so PDFs render inline.
        try:
            settings = self._web_view.settings()
            plugins_attr = getattr(
                QtWebEngineWidgets.QWebEngineSettings, "PluginsEnabled", None
            )
            pdf_attr = getattr(
                QtWebEngineWidgets.QWebEngineSettings, "PdfViewerEnabled", None
            )
            if plugins_attr is not None:
                settings.setAttribute(plugins_attr, True)
            if pdf_attr is not None:
                settings.setAttribute(pdf_attr, True)
        except Exception:
            pass
        root.addWidget(self._web_view, 1)

        self.back_button.clicked.connect(self._go_back)
        self.forward_button.clicked.connect(self._go_forward)
        self.reload_button.clicked.connect(self._reload_page)
        self.url_edit.returnPressed.connect(self._on_url_entered)
        self.open_in_browser_button.clicked.connect(self.open_current_in_browser)
        self.save_pdf_button.clicked.connect(self._on_save_pdf_clicked)
        self.close_button.clicked.connect(self.close_requested.emit)
        self._web_view.urlChanged.connect(self._on_url_changed)
        self._web_view.loadFinished.connect(self._on_load_finished)
        self._web_view.loadStarted.connect(
            lambda: QtCore.QTimer.singleShot(0, self._update_nav_buttons)
        )
        self._web_view.loadFinished.connect(
            lambda _ok=False: QtCore.QTimer.singleShot(0, self._update_nav_buttons)
        )
        self._web_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._web_view.customContextMenuRequested.connect(self._show_context_menu)

        # Update navigation button states when URL changes (QWebEngineHistory doesn't have changed signal)
        self._web_view.urlChanged.connect(self._update_nav_buttons)
        self._setup_shortcuts()

    def _open_url_from_page(self, url: str) -> None:
        target = str(url or "").strip()
        if not target:
            return
        self.load_url(target)

    def _go_back(self) -> None:
        if self._web_view is None:
            return
        history = self._web_view.history()
        if history.canGoBack():
            self._web_view.back()
            QtCore.QTimer.singleShot(0, self._update_nav_buttons)
            return
        self.status_changed.emit("No previous page.")
        self._update_nav_buttons()

    def _go_forward(self) -> None:
        if self._web_view is None:
            return
        history = self._web_view.history()
        if history.canGoForward():
            self._web_view.forward()
            QtCore.QTimer.singleShot(0, self._update_nav_buttons)
            return
        self.status_changed.emit("No next page.")
        self._update_nav_buttons()

    def _reload_page(self) -> None:
        if self._web_view is None:
            return
        self._web_view.reload()
        QtCore.QTimer.singleShot(0, self._update_nav_buttons)

    def _setup_shortcuts(self) -> None:
        if self._web_view is None:
            return

        def _add_shortcut(sequence: QtGui.QKeySequence, callback) -> None:
            shortcut = QtWidgets.QShortcut(sequence, self)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

        # Common browser zoom shortcuts. QKeySequence handles Cmd on macOS.
        _add_shortcut(QtGui.QKeySequence.ZoomIn, self._zoom_in)
        _add_shortcut(QtGui.QKeySequence.ZoomOut, self._zoom_out)
        _add_shortcut(QtGui.QKeySequence("Ctrl+="), self._zoom_in)
        _add_shortcut(QtGui.QKeySequence("Meta+="), self._zoom_in)
        _add_shortcut(QtGui.QKeySequence("Ctrl+0"), self._reset_zoom)
        _add_shortcut(QtGui.QKeySequence("Meta+0"), self._reset_zoom)

    def _adjust_zoom(self, delta: float) -> None:
        if self._web_view is None:
            return
        current = float(self._web_view.zoomFactor())
        next_zoom = max(self._ZOOM_MIN, min(self._ZOOM_MAX, current + delta))
        if abs(next_zoom - current) < 1e-9:
            return
        self._web_view.setZoomFactor(next_zoom)
        self.status_changed.emit(f"Zoom: {int(round(next_zoom * 100))}%")

    def _zoom_in(self) -> None:
        self._adjust_zoom(self._ZOOM_STEP)

    def _zoom_out(self) -> None:
        self._adjust_zoom(-self._ZOOM_STEP)

    def _reset_zoom(self) -> None:
        if self._web_view is None:
            return
        self._web_view.setZoomFactor(1.0)
        self.status_changed.emit("Zoom: 100%")

    def _normalize_url(self, url: str) -> QtCore.QUrl:
        value = str(url or "").strip()
        if not value:
            return QtCore.QUrl()
        # Existing file path support (absolute/relative) for local HTML and docs.
        local_path = Path(value).expanduser()
        if local_path.exists() and local_path.is_file():
            return QtCore.QUrl.fromLocalFile(str(local_path.resolve()))
        if "://" not in value:
            value = f"https://{value}"
        parsed = QtCore.QUrl(value)
        if parsed.scheme().lower() not in {"http", "https", "file"}:
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

    def _on_save_pdf_clicked(self) -> None:
        target = str(self.url_edit.text() or "").strip() or self._current_url
        if self._pdf_save_in_progress:
            self.status_changed.emit("PDF download already in progress.")
            return
        parsed = self._normalize_url(target)
        if not parsed.isValid() or parsed.isEmpty():
            self.status_changed.emit("Invalid URL.")
            return

        source_url = parsed.toString()
        download_url = self._resolve_pdf_download_url(source_url)
        if not self._is_saveable_pdf_url(download_url):
            self.status_changed.emit("Current page is not a PDF URL.")
            return

        output_dir = get_agent_workspace_path() / "downloads"
        self._pdf_save_in_progress = True
        self.save_pdf_button.setEnabled(False)
        self.status_changed.emit("Downloading PDF…")
        self._thread_pool.start(
            _SavePdfTask(
                self,
                source_url=source_url,
                download_url=download_url,
                output_dir=output_dir,
            )
        )

    @QtCore.Slot(str, str, str)
    def _on_pdf_saved_from_web(self, saved_path: str, message: str, error: str) -> None:
        self._pdf_save_in_progress = False
        if hasattr(self, "save_pdf_button"):
            self.save_pdf_button.setEnabled(True)
        if error:
            self.status_changed.emit(str(error))
            return
        if not saved_path:
            self.status_changed.emit("Failed to save PDF.")
            return

        host = self.window()
        open_pdf = getattr(host, "show_pdf_in_viewer", None)
        if callable(open_pdf):
            try:
                open_pdf(str(saved_path))
                self.status_changed.emit(message or "Saved PDF and opened in viewer.")
                return
            except Exception as exc:
                self.status_changed.emit(f"Saved PDF, but failed to open viewer: {exc}")
                return
        self.status_changed.emit(message or "Saved PDF.")

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
        if self._js_running:
            return {"error": "Another JavaScript task is already running."}
        page = self._web_view.page()
        if page is None:
            return {"error": "Web page object is unavailable."}

        self._js_running = True
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
        try:
            page.runJavaScript(script, _finish)
            timer.start(max(100, int(timeout_ms)))
            loop.exec_()
        finally:
            timer.stop()
            self._js_running = False
        return result.get("value")

    def get_page_text(self, max_chars: int = 8000) -> dict:
        if self._web_view is None:
            return {"ok": False, "error": "Embedded web view is unavailable."}

        # Throttle scrapes to avoid thrashing the V8 engine
        now = time.monotonic()
        if now - self._last_scrape_time < 0.5:
            time.sleep(0.5)  # Minimal safety delay
        self._last_scrape_time = time.monotonic()

        limit = max(200, min(int(max_chars or 8000), 200000))
        # Optimize script to truncate inside JS context to save IPC bandwidth
        script = f"""
(() => {{
  try {{
    const interactiveTags = new Set(["A", "BUTTON", "INPUT", "TEXTAREA", "SELECT"]);
    const isInteractive = (el) => {{
      const tag = el.tagName;
      if (interactiveTags.has(tag)) return true;
      const role = el.getAttribute("role");
      if (role === "button" || role === "link" || role === "menuitem") return true;
      if (el.onclick != null || el.getAttribute("tabindex") != null) return true;
      return false;
    }};

    let nextId = 1;
    let parts = [];

    const isVisible = (el) => {{
      if (el.offsetWidth === 0 || el.offsetHeight === 0) return false;
      const style = window.getComputedStyle(el);
      if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;
      return true;
    }};

    const traverse = (node) => {{
      if (node.nodeType === Node.TEXT_NODE) {{
        let val = node.textContent.replace(/\\s+/g, " ").trim();
        if (val) parts.push(val + " ");
        return;
      }}
      if (node.nodeType !== Node.ELEMENT_NODE) return;
      if (!isVisible(node)) return;

      const tag = node.tagName;
      if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT" || tag === "SVG") return;

      let interactive = isInteractive(node);
      let idx = 0;
      if (interactive) {{
          idx = nextId++;
          node.setAttribute("annolid-idx", idx);

          let roleDesc = tag.toLowerCase();
          if (tag === "INPUT" || tag === "BUTTON") {{
              const type = node.getAttribute("type");
              if (type) roleDesc += ` type=${{type}}`;
          }}
          parts.push(`\\n[${{idx}}: ${{roleDesc}}] `);
      }}

      const blockTags = new Set(["DIV", "P", "H1", "H2", "H3", "H4", "H5", "H6", "LI", "TR", "ARTICLE", "SECTION", "NAV", "HEADER", "FOOTER"]);
      if (blockTags.has(tag)) {{
          parts.push("\\n");
      }}

      for (let child of node.childNodes) {{
        traverse(child);
      }}

      if (interactive) {{
          if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {{
              const val = node.value || "";
              const ph = node.getAttribute("placeholder") || "";
              if (val) parts.push(` (value: "${{val}}")`);
              else if (ph) parts.push(` (placeholder: "${{ph}}")`);
          }}
          parts.push("\\n");
      }} else if (blockTags.has(tag)) {{
          parts.push("\\n");
      }}
    }};

    if (document.body) traverse(document.body);

    let rawText = parts.join("").replace(/\\n{{3,}}/g, "\\n\\n").trim();

    const title = String((document && document.title) || "");
    const href = String((window && window.location && window.location.href) || "");
    const truncated = rawText.length > {limit + 1000};
    const part = truncated ? rawText.slice(0, {limit}) : rawText;

    return {{ ok: true, text: part, title, url: href, length: rawText.length, truncated }};
  }} catch (e) {{
    return {{ ok: false, error: String(e) }};
  }}
}})()
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
  let selector = {selector_js};
  if (/^\\d+$/.test(selector)) {{
    selector = `[annolid-idx="${{selector}}"]`;
  }}
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
  let selector = {selector_js};
  if (/^\\d+$/.test(selector)) {{
    selector = `[annolid-idx="${{selector}}"]`;
  }}
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

    def _is_pdf_url(self, url: str) -> bool:
        """Check if the URL points to a PDF file.

        ArXiv ``/pdf/`` URLs render via their own viewer and are handled
        natively by the Chromium PDF plugin, so they are *not* flagged.
        """
        url_lower = url.lower()
        # ArXiv /pdf/ URLs render inline; treat as normal pages.
        if "arxiv.org/pdf/" in url_lower:
            return False
        return (
            url_lower.endswith(".pdf") or ".pdf?" in url_lower or ".pdf#" in url_lower
        )

    def _is_saveable_pdf_url(self, url: str) -> bool:
        """Check if a URL should expose the Save PDF button."""
        url_lower = str(url or "").strip().lower()
        if not url_lower:
            return False
        return (
            WebViewerWidget._is_pdf_url(None, url_lower)
            or "arxiv.org/pdf/" in url_lower
        )

    @staticmethod
    def _resolve_pdf_download_url(url: str) -> str:
        text = str(url or "").strip()
        if not text:
            return text
        parsed = urllib.parse.urlparse(text)
        host = str(parsed.netloc or "").lower()
        if "arxiv.org" not in host:
            return text

        path = str(parsed.path or "").strip()
        abs_match = re.search(
            r"^/abs/(\d{4}\.\d{4,5}(?:v\d+)?)$",
            path,
            re.IGNORECASE,
        )
        if abs_match:
            arxiv_id = str(abs_match.group(1) or "").strip()
            return urllib.parse.urlunparse(
                (
                    parsed.scheme or "https",
                    parsed.netloc,
                    f"/pdf/{arxiv_id}.pdf",
                    "",
                    "",
                    "",
                )
            )

        pdf_match = re.search(
            r"^/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?$",
            path,
            re.IGNORECASE,
        )
        if pdf_match:
            arxiv_id = str(pdf_match.group(1) or "").strip()
            return urllib.parse.urlunparse(
                (
                    parsed.scheme or "https",
                    parsed.netloc,
                    f"/pdf/{arxiv_id}.pdf",
                    "",
                    "",
                    "",
                )
            )
        return text

    def _on_load_finished(self, ok: bool) -> None:
        url = self._current_url

        if self._is_saveable_pdf_url(url):
            self.status_changed.emit(
                "PDF loaded. Use Save to download and open in Annolid PDF Viewer."
            )
            return

        if not ok:
            # If load failed and it's not a PDF URL we already handled
            self.status_changed.emit("Failed to load page.")
            return

        self.status_changed.emit(f"Loaded: {url}")

    def _show_context_menu(self, position: QtCore.QPoint) -> None:
        if self._web_view is None:
            return
        page = self._web_view.page()
        if page is None:
            return
        global_pos = self._web_view.mapToGlobal(position)

        def show_menu(selection: object) -> None:
            payload = selection if isinstance(selection, dict) else {}
            selected_text = (
                str(payload.get("selectedText") or "") if payload else ""
            ).strip()
            if not selected_text:
                selected_text = str(self._web_view.selectedText() or "").strip()
            image_src = str(payload.get("imageSrc") or "").strip() if payload else ""
            image_data_url = (
                str(payload.get("imageDataUrl") or "").strip() if payload else ""
            )
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

            explain_action = QtWidgets.QAction("Explain with Annolid Bot", self)
            explain_action.setEnabled(bool(selected_text))
            explain_action.triggered.connect(
                lambda: self._request_bot_explanation(selected_text)
            )
            menu.insertAction(
                menu.actions()[0] if menu.actions() else None, explain_action
            )

            describe_image_action = QtWidgets.QAction(
                "Describe image with Annolid Bot", self
            )
            describe_image_action.setEnabled(bool(image_src or image_data_url))
            describe_image_action.triggered.connect(
                lambda: self._request_bot_image_description(
                    image_src=image_src,
                    image_data_url=image_data_url,
                )
            )
            menu.insertAction(
                menu.actions()[0] if menu.actions() else None, describe_image_action
            )

            speak_action = QtWidgets.QAction("Speak selection", self)
            speak_action.setEnabled(bool(selected_text) and not self._speaking)
            speak_action.triggered.connect(
                lambda: self._speak_selected_text(selected_text)
            )
            menu.insertAction(
                menu.actions()[0] if menu.actions() else None, speak_action
            )

            if self._speaking:
                stop_action = QtWidgets.QAction("Stop speaking", self)
                stop_action.triggered.connect(self._cancel_speaking)
                menu.insertAction(
                    menu.actions()[0] if menu.actions() else None, stop_action
                )

            menu.exec_(global_pos)

        try:
            page.runJavaScript(
                f"""(() => {{
  const x = {int(position.x())};
  const y = {int(position.y())};
  const sel = window.getSelection ? window.getSelection() : null;
  const selectedText = (!sel || sel.isCollapsed) ? '' : String(sel.toString() || '');
  let imageSrc = '';
  let imageDataUrl = '';
  try {{
    let el = document.elementFromPoint(x, y);
    if (el) {{
      if (el.tagName !== 'IMG' && typeof el.closest === 'function') {{
        el = el.closest('img');
      }}
      if (el && el.tagName === 'IMG') {{
        const img = el;
        imageSrc = String(img.currentSrc || img.src || '').trim();
        try {{
          const w = Number(img.naturalWidth || img.width || 0);
          const h = Number(img.naturalHeight || img.height || 0);
          if (w > 0 && h > 0) {{
            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d');
            if (ctx) {{
              ctx.drawImage(img, 0, 0, w, h);
              imageDataUrl = String(canvas.toDataURL('image/png') || '');
            }}
          }}
        }} catch (e) {{}}
      }}
    }}
  }} catch (e) {{}}
  return {{ selectedText, imageSrc, imageDataUrl }};
}})()""",
                show_menu,
            )
        except Exception:
            show_menu(None)

    @staticmethod
    def _extract_single_word(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        word_chars = (
            r"A-Za-z"
            r"À-ÖØ-öø-ÿ"
            r"Ā-ſ"
            r"Ḁ-ỿ"
            r"\u4e00-\u9fff"
            r"\u3400-\u4dbf"
        )
        cleaned = re.sub(rf"[^{word_chars}'\u2019-]+", " ", raw)
        tokens = [tok for tok in cleaned.strip().split() if tok]
        if len(tokens) != 1:
            return ""
        return tokens[0].strip("'\u2019-").lower()

    def _request_dictionary_lookup(
        self, selected_text: str, *, global_pos: Optional[QtCore.QPoint] = None
    ) -> None:
        word = self._extract_single_word(selected_text)
        if not word:
            QtWidgets.QToolTip.showText(
                QtGui.QCursor.pos(),
                "Select exactly one word to look up.",
                self,
            )
            return
        self._dictionary_lookup_id = uuid.uuid4().hex
        self._dictionary_popup_pos = global_pos
        QtWidgets.QToolTip.showText(
            QtGui.QCursor.pos(),
            f'Looking up "{word}"…',
            self,
        )
        self._thread_pool.start(
            DictionaryLookupTask(
                widget=self,
                request_id=self._dictionary_lookup_id,
                word=word,
            )
        )

    @QtCore.Slot(str, str, str, str)
    def _on_dictionary_lookup_finished(
        self,
        request_id: str,
        word: str,
        html: str,
        error: str,
    ) -> None:
        if request_id != self._dictionary_lookup_id:
            return
        self._show_dictionary_popup(
            word,
            html=html,
            error=error,
            global_pos=self._dictionary_popup_pos,
        )

    def _show_dictionary_popup(
        self,
        word: str,
        *,
        html: str = "",
        error: str = "",
        global_pos: Optional[QtCore.QPoint] = None,
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

    def _request_bot_explanation(self, selected_text: str) -> None:
        ok, message = explain_selection_with_annolid_bot(
            self,
            selected_text,
            source_hint=str(self._current_url or "").strip(),
        )
        if message:
            self.status_changed.emit(message if ok else f"Explain failed: {message}")

    @staticmethod
    def _decode_data_url(data_url: str) -> tuple[bytes, str]:
        value = str(data_url or "").strip()
        if not value.startswith("data:"):
            return b"", ""
        match = re.match(r"^data:([^;,]+)?(?:;charset=[^;,]+)?(;base64)?,", value)
        if not match:
            return b"", ""
        mime = str(match.group(1) or "application/octet-stream").strip().lower()
        is_base64 = bool(match.group(2))
        body = value[match.end() :]
        try:
            if is_base64:
                return base64.b64decode(body, validate=False), mime
            return urllib.parse.unquote_to_bytes(body), mime
        except (ValueError, binascii.Error):
            return b"", ""

    @staticmethod
    def _mime_to_suffix(mime: str) -> str:
        value = str(mime or "").strip().lower()
        if value in {"image/jpg", "image/jpeg"}:
            return ".jpg"
        if value == "image/png":
            return ".png"
        if value == "image/webp":
            return ".webp"
        if value == "image/gif":
            return ".gif"
        guessed = mimetypes.guess_extension(value or "")
        return guessed if guessed else ".png"

    def _save_image_from_context(
        self, *, image_src: str, image_data_url: str
    ) -> tuple[str, str]:
        if image_data_url:
            raw, mime = self._decode_data_url(image_data_url)
            if raw:
                fd, path = tempfile.mkstemp(
                    prefix="annolid_web_image_",
                    suffix=self._mime_to_suffix(mime),
                )
                try:
                    os.close(fd)
                    with open(path, "wb") as f:
                        f.write(raw)
                    return path, ""
                except Exception as exc:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    return "", f"Failed to save selected image: {exc}"

        src = str(image_src or "").strip()
        if not src:
            return "", "No image found at the clicked location."
        if src.startswith("data:"):
            raw, mime = self._decode_data_url(src)
            if not raw:
                return "", "Failed to decode selected image data."
            fd, path = tempfile.mkstemp(
                prefix="annolid_web_image_",
                suffix=self._mime_to_suffix(mime),
            )
            try:
                os.close(fd)
                with open(path, "wb") as f:
                    f.write(raw)
                return path, ""
            except Exception as exc:
                try:
                    os.remove(path)
                except OSError:
                    pass
                return "", f"Failed to save selected image: {exc}"

        try:
            req = urllib.request.Request(
                src,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                },
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                content_type = str(response.headers.get("content-type") or "").split(
                    ";", 1
                )[0]
                raw = response.read()
            if not raw:
                return "", "Selected image URL returned no data."
            fd, path = tempfile.mkstemp(
                prefix="annolid_web_image_",
                suffix=self._mime_to_suffix(content_type),
            )
            os.close(fd)
            with open(path, "wb") as f:
                f.write(raw)
            return path, ""
        except Exception as exc:
            return "", f"Failed to fetch selected image: {exc}"

    def _request_bot_image_description(
        self, *, image_src: str = "", image_data_url: str = ""
    ) -> None:
        image_path, error = self._save_image_from_context(
            image_src=image_src,
            image_data_url=image_data_url,
        )
        if error:
            self.status_changed.emit(f"Describe image failed: {error}")
            return
        ok, message = explain_image_with_annolid_bot(
            self,
            image_path,
            source_hint=str(self._current_url or "").strip(),
            image_url=str(image_src or "").strip(),
        )
        if message:
            self.status_changed.emit(
                message if ok else f"Describe image failed: {message}"
            )

    def _speak_selected_text(self, text: str) -> None:
        cleaned = " ".join(str(text or "").strip().split())
        if not cleaned:
            self.status_changed.emit("No selected text to speak.")
            return
        if self._speaking:
            self.status_changed.emit("Already speaking. Use 'Stop speaking' first.")
            return

        settings = self._tts_settings_snapshot()
        token = _SpeakToken()
        self._active_speak_token = token
        self._speaking = True
        self.status_changed.emit("Speaking selected text…")
        self._thread_pool.start(_WebSpeakTask(self, cleaned, settings, token))

    def _cancel_speaking(self) -> None:
        if self._active_speak_token is not None:
            self._active_speak_token.cancelled = True
            self.status_changed.emit("Stopping speech…")

    @staticmethod
    def _tts_settings_snapshot() -> Dict[str, object]:
        from annolid.utils.tts_settings import default_tts_settings, load_tts_settings

        settings = load_tts_settings()
        defaults = default_tts_settings()
        return {
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

    @QtCore.Slot(str)
    def _on_speak_status(self, message: str) -> None:
        text = str(message or "").strip()
        if text:
            self.status_changed.emit(text)

    @QtCore.Slot()
    def _on_speak_finished(self) -> None:
        self._speaking = False
        self._active_speak_token = None


class _SpeakToken:
    def __init__(self) -> None:
        self.cancelled = False


class _WebSpeakTask(QtCore.QRunnable):
    """Background task to stream TTS chunks for selected web text."""

    def __init__(
        self,
        widget: WebViewerWidget,
        text: str,
        tts_settings: Dict[str, object],
        token: Optional[_SpeakToken] = None,
    ) -> None:
        super().__init__()
        self.widget = widget
        self.text = text
        self.tts_settings = tts_settings
        self.token = token

    def run(self) -> None:  # pragma: no cover - involves audio device/TTS backends
        try:
            text = str(self.text or "").strip()
            if not text:
                return
            chunks = self._chunk_text(text)
            if not chunks:
                return

            def cancelled() -> bool:
                return bool(self.token is not None and self.token.cancelled)

            from concurrent.futures import ThreadPoolExecutor

            from annolid.agents.tts_router import synthesize_tts
            from annolid.utils.audio_playback import play_audio_buffer

            def synthesize(chunk: str):
                audio_data = synthesize_tts(chunk, self.tts_settings)
                if not audio_data:
                    raise RuntimeError("No audio returned by TTS engine.")
                return audio_data

            with ThreadPoolExecutor(max_workers=1) as executor:
                current_future = executor.submit(synthesize, chunks[0])
                for idx in range(len(chunks)):
                    if cancelled():
                        return
                    samples, sample_rate = current_future.result()
                    next_future = None
                    if idx + 1 < len(chunks):
                        next_future = executor.submit(synthesize, chunks[idx + 1])
                    QtCore.QMetaObject.invokeMethod(
                        self.widget,
                        "_on_speak_status",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(
                            str, f"Speaking selected text ({idx + 1}/{len(chunks)})…"
                        ),
                    )
                    if cancelled():
                        return
                    played = play_audio_buffer(samples, int(sample_rate), blocking=True)
                    if not played:
                        raise RuntimeError("No usable audio device found.")
                    if next_future is None:
                        break
                    current_future = next_future

            if not cancelled():
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "_on_speak_status",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, "Finished speaking selected text."),
                )
        except Exception as exc:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "_on_speak_status",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Speak selection failed: {exc}"),
            )
        finally:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "_on_speak_finished",
                QtCore.Qt.QueuedConnection,
            )

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 420) -> list[str]:
        import re

        cleaned = re.sub(r"\s+", " ", str(text or "").strip())
        if not cleaned:
            return []
        sentences = re.split(r"(?<=[.!?。！？])\s+", cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [cleaned]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            add_len = sentence_len + (1 if current else 0)
            if current and current_len + add_len > max_chars:
                chunks.append(" ".join(current))
                current = [sentence]
                current_len = sentence_len
            else:
                current.append(sentence)
                current_len += add_len
            if sentence_len > max_chars:
                if current:
                    chunks.append(" ".join(current[:-1]).strip())
                current = []
                current_len = 0
                words = sentence.split()
                piece: list[str] = []
                piece_len = 0
                for word in words:
                    extra = len(word) + (1 if piece else 0)
                    if piece and piece_len + extra > max_chars:
                        chunks.append(" ".join(piece))
                        piece = [word]
                        piece_len = len(word)
                    else:
                        piece.append(word)
                        piece_len += extra
                if piece:
                    current = [" ".join(piece)]
                    current_len = len(current[0])

        if current:
            chunks.append(" ".join(current))
        return [c for c in chunks if c.strip()]
