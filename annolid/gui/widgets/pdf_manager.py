from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, TYPE_CHECKING

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt

from annolid.gui.widgets.pdf_viewer import PdfViewerWidget
from annolid.gui.widgets.pdf_controls import PdfControlsWidget
from annolid.gui.widgets.pdf_reader_controls import PdfReaderControlsWidget
from annolid.gui.widgets.pdf_reading_log import PdfReadingLogWidget
from annolid.gui.widgets.tts_controls import TtsControlsWidget
from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class PdfManager(QtCore.QObject):
    """Encapsulates PDF viewer, controls, and docks wiring for the main window."""

    _RECENT_PDFS_KEY = "pdf/recent_files"
    _RECENT_PDFS_LIMIT = 40

    def __init__(
        self, window: "AnnolidWindow", viewer_stack: QtWidgets.QStackedWidget
    ) -> None:
        super().__init__(window)
        self.window = window
        self.viewer_stack = viewer_stack
        self.pdf_viewer: Optional[PdfViewerWidget] = None
        self.pdf_tts_dock: Optional[QtWidgets.QDockWidget] = None
        self.pdf_tts_controls: Optional[TtsControlsWidget] = None
        self.pdf_controls_dock: Optional[QtWidgets.QDockWidget] = None
        self.pdf_controls_widget: Optional[PdfControlsWidget] = None
        self.pdf_reader_dock: Optional[QtWidgets.QDockWidget] = None
        self.pdf_reader_controls: Optional[PdfReaderControlsWidget] = None
        self.pdf_log_dock: Optional[QtWidgets.QDockWidget] = None
        self.pdf_log_widget: Optional[PdfReadingLogWidget] = None
        self._pdf_files: list[str] = []
        self._pdf_file_signals_connected = False
        self._hidden_docks: list[QtWidgets.QDockWidget] = []
        self._labelme_file_selection_disabled = False
        self._load_recent_pdfs()

    def _settings(self) -> Optional[QtCore.QSettings]:
        return getattr(self.window, "settings", None)

    def _load_recent_pdfs(self) -> None:
        settings = self._settings()
        if settings is None:
            return
        try:
            raw = settings.value(self._RECENT_PDFS_KEY, [])
        except Exception:
            raw = []
        paths: list[str] = []
        if isinstance(raw, (list, tuple)):
            paths = [str(p) for p in raw if p]
        elif isinstance(raw, str) and raw.strip():
            # Backwards/portable: allow comma-separated or JSON list payloads.
            text = raw.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    import json

                    loaded = json.loads(text)
                    if isinstance(loaded, list):
                        paths = [str(p) for p in loaded if p]
                except Exception:
                    paths = []
            if not paths:
                paths = [p.strip() for p in text.split(",") if p.strip()]

        cleaned: list[str] = []
        seen: set[str] = set()
        for p in paths:
            try:
                resolved = str(Path(p).expanduser().resolve())
            except Exception:
                resolved = str(p)
            if not resolved or resolved in seen:
                continue
            try:
                if not Path(resolved).exists():
                    continue
            except Exception:
                continue
            seen.add(resolved)
            cleaned.append(resolved)
        self._pdf_files = cleaned[-self._RECENT_PDFS_LIMIT :]

    def _save_recent_pdfs(self) -> None:
        settings = self._settings()
        if settings is None:
            return
        try:
            settings.setValue(self._RECENT_PDFS_KEY, list(self._pdf_files))
        except Exception:
            pass

    # ------------------------------------------------------------------ setup
    def ensure_pdf_viewer(self) -> PdfViewerWidget:
        if self.pdf_viewer is None:
            viewer = PdfViewerWidget(self.window)
            viewer.page_changed.connect(self._on_page_changed)
            viewer.page_changed.connect(self._update_pdf_controls_page)
            viewer.controls_enabled_changed.connect(
                self._on_pdf_controls_enabled_changed
            )
            viewer.reader_state_changed.connect(self._on_reader_state_changed)
            viewer.reader_availability_changed.connect(
                self._on_reader_availability_changed
            )
            viewer.reading_log_event.connect(self._on_reading_log_event)
            self.viewer_stack.addWidget(viewer)
            self.pdf_viewer = viewer
        return self.pdf_viewer

    def ensure_pdf_tts_dock(self) -> None:
        if self.pdf_tts_dock is None:
            dock = QtWidgets.QDockWidget(self.window.tr("PDF Speech"), self.window)
            dock.setObjectName("PdfTtsDock")
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            controls = TtsControlsWidget(dock)
            container = QtWidgets.QWidget(dock)
            lay = QtWidgets.QVBoxLayout(container)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(6)
            lay.addWidget(controls, alignment=Qt.AlignTop)
            lay.addStretch(1)
            container.setLayout(lay)
            container.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
            )
            dock.setWidget(container)
            self.window.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.pdf_tts_dock = dock
            self.pdf_tts_controls = controls
        if self.pdf_tts_dock is not None:
            self.pdf_tts_dock.show()
            self.pdf_tts_dock.raise_()
        self._connect_file_list_signals()

    def ensure_pdf_controls_dock(self) -> None:
        if self.pdf_controls_dock is None:
            dock = QtWidgets.QDockWidget(self.window.tr("PDF Controls"), self.window)
            dock.setObjectName("PdfControlsDock")
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            controls = PdfControlsWidget(dock)
            container = QtWidgets.QWidget(dock)
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(4)
            layout.addWidget(controls, alignment=Qt.AlignTop)
            container.setLayout(layout)
            container.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
            )
            dock.setWidget(container)
            self.window.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.pdf_controls_dock = dock
            self.pdf_controls_widget = controls
            controls.previous_requested.connect(self._pdf_prev_page)
            controls.next_requested.connect(self._pdf_next_page)
            controls.rotation_requested.connect(self._pdf_rotate_clockwise)
            controls.reset_zoom_requested.connect(self._pdf_reset_zoom)
            controls.zoom_changed.connect(self._pdf_set_zoom)

        if self.pdf_controls_dock is not None:
            self.pdf_controls_dock.show()
            self.pdf_controls_dock.raise_()
        self._sync_pdf_controls_state()

    def ensure_pdf_reader_dock(self) -> None:
        if self.pdf_reader_dock is None:
            dock = QtWidgets.QDockWidget(self.window.tr("PDF Reader"), self.window)
            dock.setObjectName("PdfReaderDock")
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            controls = PdfReaderControlsWidget(dock)
            container = QtWidgets.QWidget(dock)
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(4)
            layout.addWidget(controls, alignment=Qt.AlignTop)
            container.setLayout(layout)
            container.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
            )
            dock.setWidget(container)
            self.window.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.pdf_reader_dock = dock
            self.pdf_reader_controls = controls

            controls.reader_enabled_changed.connect(self._set_reader_enabled)
            controls.pdfjs_mode_requested.connect(self._set_reader_pdfjs_mode)
            controls.pause_resume_requested.connect(self._toggle_reader_pause_resume)
            controls.stop_requested.connect(self._stop_reader)
            controls.previous_requested.connect(self._reader_prev_sentence)
            controls.next_requested.connect(self._reader_next_sentence)

        if self.pdf_reader_dock is not None:
            self.pdf_reader_dock.show()
            self.pdf_reader_dock.raise_()
        self._sync_pdf_reader_state()

    def ensure_pdf_log_dock(self) -> None:
        if self.pdf_log_dock is None:
            dock = QtWidgets.QDockWidget(self.window.tr("PDF Reading Log"), self.window)
            dock.setObjectName("PdfReadingLogDock")
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            widget = PdfReadingLogWidget(dock)
            dock.setWidget(widget)
            self.window.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.pdf_log_dock = dock
            self.pdf_log_widget = widget
            widget.entry_activated.connect(self._on_log_entry_activated)
            widget.clear_requested.connect(self._clear_pdf_reading_log)

        if self.pdf_log_dock is not None:
            self.pdf_log_dock.show()
            self.pdf_log_dock.raise_()
        self._sync_pdf_log_entries()

    def _connect_file_list_signals(self) -> None:
        if self._pdf_file_signals_connected:
            return
        widget = getattr(self.window, "fileListWidget", None)
        if widget is None:
            return
        try:
            widget.itemActivated.connect(self._handle_file_list_activation)
            self._pdf_file_signals_connected = True
        except Exception:
            self._pdf_file_signals_connected = False

    # ------------------------------------------------------------------ actions
    def show_pdf_in_viewer(self, pdf_path: str) -> None:
        """Load a PDF into the viewer and display it in place of the canvas."""
        viewer = self.ensure_pdf_viewer()
        try:
            viewer.load_pdf(pdf_path)
        except Exception as exc:  # pragma: no cover - user-facing dialog
            logger.error("Failed to open PDF %s: %s", pdf_path, exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self.window,
                self.window.tr("Failed to Open PDF"),
                self.window.tr("Could not open the selected PDF:\n%1").replace(
                    "%1", str(exc)
                ),
            )
            self.window._set_active_view("canvas")
            return

        self.window.video_loader = None
        self.window.filename = None
        self.window._set_active_view("pdf")
        self._disable_labelme_file_selection()
        self._close_unrelated_docks_for_pdf()
        self.ensure_pdf_tts_dock()
        # Pick a reasonable default TTS language/voice early so the dock reflects it immediately.
        self._auto_select_tts_from_pdf_language(str(pdf_path))
        self.ensure_pdf_controls_dock()
        self.ensure_pdf_reader_dock()
        self.ensure_pdf_log_dock()
        self._tighten_right_docks()
        try:
            self.pdf_viewer.fit_to_width()
        except Exception:
            pass
        self._record_pdf_entry(str(pdf_path))
        self.window.lastOpenDir = str(Path(pdf_path).parent)
        self.window.statusBar().showMessage(
            self.window.tr("Loaded PDF %1").replace("%1", Path(pdf_path).name), 3000
        )
        try:
            close_action = getattr(getattr(self.window, "actions", None), "close", None)
            if close_action is not None:
                close_action.setEnabled(True)
        except Exception:
            pass

    def close_pdf(self) -> None:
        """Close PDF view, restore docks, and return to canvas."""
        try:
            if self.pdf_viewer is not None:
                self.pdf_viewer.record_stop_event()
        except Exception:
            pass
        # Restore docks hidden for PDF.
        self._restore_hidden_docks()
        self._restore_labelme_file_selection()
        # Hide PDF docks.
        for dock in (
            self.pdf_tts_dock,
            self.pdf_controls_dock,
            self.pdf_reader_dock,
            self.pdf_log_dock,
        ):
            try:
                if dock is not None:
                    dock.hide()
            except Exception:
                pass
        # Switch back to canvas.
        try:
            self.window._set_active_view("canvas")
        except Exception:
            pass

    # ------------------------------------------------------------------ slots/helpers
    def _on_page_changed(self, current: int, total: int) -> None:
        try:
            self.window.statusBar().showMessage(
                self.window.tr("PDF page %1 of %2")
                .replace("%1", str(current + 1))
                .replace("%2", str(total)),
                3000,
            )
        except Exception:
            pass

    def _close_unrelated_docks_for_pdf(self) -> None:
        """Hide docks not useful when viewing PDFs to reduce clutter."""
        self._hidden_docks.clear()
        docks = [
            getattr(self.window, "behavior_log_dock", None),
            getattr(self.window, "behavior_controls_dock", None),
            getattr(self.window, "flag_dock", None),
            getattr(self.window, "audio_dock", None),
            getattr(self.window, "florence_dock", None),
            getattr(self.window, "video_dock", None),
            getattr(self.window, "label_dock", None),
            getattr(self.window, "shape_dock", None),
        ]
        for dock in docks:
            try:
                if dock is not None:
                    dock.hide()
                    self._hidden_docks.append(dock)
            except Exception:
                continue

        # Keep the file list visible for PDF navigation.
        file_dock = getattr(self.window, "file_dock", None)
        if file_dock is not None:
            file_dock.show()
            file_dock.raise_()

    def _tighten_right_docks(self) -> None:
        """Shrink the right dock area to fit PDF tools by default."""
        docks: list[QtWidgets.QDockWidget] = []
        for attr in ("file_dock",):
            dock = getattr(self.window, attr, None)
            if isinstance(dock, QtWidgets.QDockWidget) and dock.isVisible():
                docks.append(dock)
        for dock in (self.pdf_tts_dock, self.pdf_controls_dock, self.pdf_reader_dock):
            if isinstance(dock, QtWidgets.QDockWidget) and dock.isVisible():
                docks.append(dock)
        if not docks:
            return
        try:
            for dock in docks:
                w = dock.widget()
                if w is not None:
                    w.setSizePolicy(
                        QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
                    )
        except Exception:
            pass
        widths: list[int] = []
        for dock in docks:
            try:
                hint = dock.sizeHint().width()
            except Exception:
                hint = 0
            try:
                if dock.widget() is not None:
                    hint = max(hint, dock.widget().sizeHint().width())
            except Exception:
                pass
            widths.append(int(hint) if hint else 0)
        target = max(widths) if widths else 0
        # Clamp to a sane narrow default; users can still resize manually.
        target = max(240, min(360, target + 16))
        try:
            self.window.resizeDocks(docks, [target] * len(docks), Qt.Horizontal)
        except Exception:
            # Fallback: set a minimum width on the PDF docks only.
            for dock in (
                self.pdf_tts_dock,
                self.pdf_controls_dock,
                self.pdf_reader_dock,
            ):
                try:
                    if dock is not None:
                        dock.setMinimumWidth(target)
                except Exception:
                    pass

    def _restore_hidden_docks(self) -> None:
        """Show docks that were hidden for PDF viewing."""
        for dock in self._hidden_docks:
            try:
                dock.show()
            except Exception:
                continue
        self._hidden_docks.clear()

    def _disable_labelme_file_selection(self) -> None:
        """Prevent LabelMe's file list selection handler from firing on PDF items."""
        if self._labelme_file_selection_disabled:
            return
        widget = getattr(self.window, "fileListWidget", None)
        handler = getattr(self.window, "fileSelectionChanged", None)
        if widget is None or handler is None:
            return
        try:
            widget.itemSelectionChanged.disconnect(handler)
            self._labelme_file_selection_disabled = True
        except Exception:
            self._labelme_file_selection_disabled = False

    def _restore_labelme_file_selection(self) -> None:
        """Restore LabelMe's file list selection handler after closing PDFs."""
        if not self._labelme_file_selection_disabled:
            return
        widget = getattr(self.window, "fileListWidget", None)
        handler = getattr(self.window, "fileSelectionChanged", None)
        if widget is None or handler is None:
            self._labelme_file_selection_disabled = False
            return
        try:
            widget.itemSelectionChanged.connect(handler)
        except Exception:
            pass
        self._labelme_file_selection_disabled = False

    def _record_pdf_entry(self, pdf_path: str) -> None:
        try:
            resolved = str(Path(pdf_path).resolve())
        except Exception:
            resolved = pdf_path
        try:
            if not Path(resolved).exists():
                return
        except Exception:
            pass
        if resolved in self._pdf_files:
            try:
                self._pdf_files.remove(resolved)
            except ValueError:
                pass
        self._pdf_files.append(resolved)
        self._pdf_files = self._pdf_files[-self._RECENT_PDFS_LIMIT :]
        self._save_recent_pdfs()
        self._populate_pdf_file_list()

    def _auto_select_tts_from_pdf_language(self, pdf_path: str) -> None:
        controls = self.pdf_tts_controls
        if controls is None:
            return
        sample = self._extract_pdf_text_sample(pdf_path)
        lang = self._detect_kokoro_language(sample)
        if not lang:
            return
        voice = self._suggest_default_voice(lang)
        try:
            controls.set_language_and_voice(lang=lang, voice=voice, persist=True)
        except Exception:
            return

    @staticmethod
    def _extract_pdf_text_sample(pdf_path: str) -> str:
        """Extract a small text sample to infer document language."""
        parts: list[str] = []
        try:
            path = Path(pdf_path)
            parts.append(path.stem)
        except Exception:
            pass
        try:
            import fitz  # type: ignore[import]

            path = Path(pdf_path)
            with fitz.open(str(path)) as doc:
                meta = getattr(doc, "metadata", None) or {}
                if isinstance(meta, dict):
                    for key in ("title", "subject", "keywords"):
                        try:
                            v = str(meta.get(key) or "").strip()
                            if v:
                                parts.append(v)
                        except Exception:
                            continue
                pages = min(3, int(getattr(doc, "page_count", 0) or 0))
                for i in range(pages):
                    try:
                        page = doc.load_page(i)
                    except Exception:
                        continue
                    chunk = ""
                    try:
                        chunk = str(page.get_text("text") or "")
                    except Exception:
                        chunk = ""
                    chunk = chunk.replace("\u2029", "\n").strip()

                    # Some PDFs return empty/fragmented results for "text" but have usable
                    # strings in blocks/words.
                    if len(chunk) < 60:
                        try:
                            blocks = page.get_text("blocks") or []
                            block_texts: list[str] = []
                            for b in blocks:
                                if not isinstance(b, (list, tuple)) or len(b) < 5:
                                    continue
                                t = str(b[4] or "").strip()
                                if t:
                                    block_texts.append(t)
                            if block_texts:
                                chunk = (chunk + "\n" + "\n".join(block_texts)).strip()
                        except Exception:
                            pass
                    if len(chunk) < 60:
                        try:
                            words = page.get_text("words") or []
                            ws: list[str] = []
                            for w in words:
                                if not isinstance(w, (list, tuple)) or len(w) < 5:
                                    continue
                                t = str(w[4] or "").strip()
                                if t:
                                    ws.append(t)
                            if ws:
                                chunk = (chunk + "\n" + " ".join(ws)).strip()
                        except Exception:
                            pass
                    if chunk:
                        parts.append(chunk)
                    if sum(len(p) for p in parts) >= 12000:
                        break
        except Exception:
            pass
        # Keep sample bounded to stay fast.
        sample = "\n".join([p for p in parts if p]).strip()
        return sample[:12000]

    @staticmethod
    def _detect_kokoro_language(text: str) -> str:
        """Return a Kokoro language code best-effort (e.g. zh, ja, en-us)."""
        s = (text or "").strip()
        if not s:
            return ""
        # Consider only a prefix to stay fast.
        s = s[:8000]
        counts = {
            "hiragana": 0,
            "katakana": 0,
            "han": 0,
            "hangul": 0,
            "cyrillic": 0,
            "devanagari": 0,
            "latin": 0,
        }
        for ch in s:
            o = ord(ch)
            if 0x3040 <= o <= 0x309F:
                counts["hiragana"] += 1
            elif 0x30A0 <= o <= 0x30FF:
                counts["katakana"] += 1
            elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
                counts["han"] += 1
            elif 0xAC00 <= o <= 0xD7AF:
                counts["hangul"] += 1
            elif 0x0400 <= o <= 0x04FF:
                counts["cyrillic"] += 1
            elif 0x0900 <= o <= 0x097F:
                counts["devanagari"] += 1
            elif (0x0041 <= o <= 0x007A) or (0x00C0 <= o <= 0x024F):
                counts["latin"] += 1

        kana = counts["hiragana"] + counts["katakana"]
        han = counts["han"]
        total = (
            kana
            + han
            + counts["hangul"]
            + counts["cyrillic"]
            + counts["devanagari"]
            + counts["latin"]
        )
        total = max(1, int(total))

        if kana >= 8 and kana / total >= 0.06:
            return "ja"
        if counts["hangul"] >= 8 and counts["hangul"] / total >= 0.06:
            return "ko"
        # Chinese: require enough Han chars and meaningful dominance.
        if han >= 5 and kana < 6 and (han / total >= 0.12 or han >= 30):
            return "zh"
        if counts["cyrillic"] >= 10 and counts["cyrillic"] / total >= 0.08:
            return "ru"
        if counts["devanagari"] >= 10 and counts["devanagari"] / total >= 0.08:
            return "hi"
        if counts["latin"] >= 24 and counts["latin"] / total >= 0.20:
            # Conservative: most papers are English; keep a stable default.
            return "en-us"
        return ""

    @staticmethod
    def _suggest_default_voice(lang: str) -> str:
        """Pick a reasonable default voice for the given Kokoro language code."""
        lang = (lang or "").strip().lower()
        try:
            from annolid.agents.kokoro_tts import (
                DEFAULT_VOICE,
                get_available_voices,
            )
        except Exception:
            return "af_sarah"

        voices = get_available_voices(lang=lang) or []
        if not voices:
            return DEFAULT_VOICE

        def first_with_prefix(prefixes: tuple[str, ...]) -> str:
            for v in voices:
                lv = str(v).strip().lower()
                if any(lv.startswith(p) for p in prefixes):
                    return str(v)
            return ""

        if lang.startswith("ja"):
            v = first_with_prefix(("j",))
            return v or DEFAULT_VOICE
        if lang.startswith("zh") or lang in {"cmn"}:
            v = first_with_prefix(("z",))
            return v or DEFAULT_VOICE
        if lang.startswith("en-gb") or lang == "en-gb":
            # Prefer "bf_*" (British female) if available.
            v = first_with_prefix(("bf_", "bm_", "b"))
            return v or DEFAULT_VOICE
        if lang.startswith("en"):
            # Prefer an American female voice when possible.
            v = first_with_prefix(("af_", "am_", "a"))
            return v or DEFAULT_VOICE
        # For languages without dedicated voices, keep a stable default.
        return DEFAULT_VOICE

    def _populate_pdf_file_list(self) -> None:
        widget = getattr(self.window, "fileListWidget", None)
        if widget is None:
            return
        widget.clear()
        for path in self._pdf_files:
            name = Path(path).name
            item = QtWidgets.QListWidgetItem(name)
            item.setData(Qt.UserRole, path)
            item.setData(Qt.UserRole + 1, "pdf")
            item.setToolTip(path)
            widget.addItem(item)
        if widget.count() > 0:
            widget.setCurrentRow(widget.count() - 1)

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def _handle_file_list_activation(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is None:
            return
        kind = item.data(Qt.UserRole + 1)
        if kind != "pdf":
            return
        path = item.data(Qt.UserRole)
        if not path:
            return
        self.show_pdf_in_viewer(str(path))

    def _pdf_prev_page(self) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.previous_page()

    def _pdf_next_page(self) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.next_page()

    def _pdf_reset_zoom(self) -> None:
        if self.pdf_viewer is None:
            return
        self.pdf_viewer.reset_zoom()
        self._sync_pdf_controls_state()

    def _pdf_rotate_clockwise(self) -> None:
        if self.pdf_viewer is None:
            return
        self.pdf_viewer.rotate_clockwise()
        self._sync_pdf_controls_state()

    def _pdf_set_zoom(self, percent: float) -> None:
        if self.pdf_viewer is None:
            return
        self.pdf_viewer.set_zoom_percent(percent)
        self._sync_pdf_controls_state()

    def _update_pdf_controls_page(self, current: int, total: int) -> None:
        if self.pdf_controls_widget is not None:
            self.pdf_controls_widget.set_page_info(current, total)

    def _on_pdf_controls_enabled_changed(self, enabled: bool) -> None:
        if self.pdf_controls_widget is not None:
            reason = (
                self.window.tr("Navigation disabled in embedded browser mode.")
                if not enabled
                else ""
            )
            self.pdf_controls_widget.set_controls_enabled(enabled, reason=reason)

    def _sync_pdf_controls_state(self) -> None:
        viewer = self.pdf_viewer
        controls = self.pdf_controls_widget
        if viewer is None or controls is None:
            return
        controls.set_controls_enabled(viewer.controls_enabled())
        controls.set_zoom_percent(viewer.current_zoom_percent())
        total = viewer.page_count()
        controls.set_page_info(viewer.current_page_index(), total)

    def _sync_pdf_reader_state(self) -> None:
        viewer = self.pdf_viewer
        controls = self.pdf_reader_controls
        if viewer is None or controls is None:
            return
        available, reason = viewer.reader_availability()
        controls.set_reader_available(available, reason=reason)
        controls.set_reader_enabled(viewer.reader_enabled())
        controls.set_pdfjs_checked(viewer.force_pdfjs_enabled())
        state, current, total = viewer.reader_state()
        controls.set_reader_state(state, current, total)

    def _sync_pdf_log_entries(self) -> None:
        widget = self.pdf_log_widget
        if widget is None:
            return
        if self.pdf_viewer is None:
            widget.clear()
            return
        try:
            widget.set_entries(self.pdf_viewer.reading_log())
        except Exception:
            widget.clear()

    def _clear_pdf_reading_log(self) -> None:
        if self.pdf_viewer is not None:
            try:
                self.pdf_viewer.clear_reading_log()
            except Exception:
                pass
        if self.pdf_log_widget is not None:
            self.pdf_log_widget.clear()

    def _on_log_entry_activated(self, entry: dict) -> None:
        if self.pdf_viewer is None:
            return
        try:
            self.pdf_viewer.activate_log_entry(entry)
        except Exception:
            pass

    def _on_reading_log_event(self, entry: dict) -> None:
        if self.pdf_log_widget is None:
            return
        try:
            self.pdf_log_widget.add_entry(entry)
        except Exception:
            pass

    def _on_reader_state_changed(self, state: str, current: int, total: int) -> None:
        if self.pdf_reader_controls is not None:
            self.pdf_reader_controls.set_reader_state(state, current, total)

    def _on_reader_availability_changed(self, available: bool, reason: str) -> None:
        if self.pdf_reader_controls is not None:
            self.pdf_reader_controls.set_reader_available(available, reason=reason)
        self._sync_pdf_reader_state()

    def _set_reader_enabled(self, enabled: bool) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.set_reader_enabled(enabled)

    def _set_reader_pdfjs_mode(self, enabled: bool) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.set_force_pdfjs(enabled)
        self._sync_pdf_reader_state()

    def _toggle_reader_pause_resume(self) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.toggle_reader_pause_resume()

    def _stop_reader(self) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.stop_reader()

    def _reader_prev_sentence(self) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.reader_prev_sentence()

    def _reader_next_sentence(self) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.reader_next_sentence()

    # ------------------------------------------------------------------ helpers
    def pdf_widget(self) -> Optional[PdfViewerWidget]:
        return self.pdf_viewer

    def get_pdf_state(self) -> dict:
        viewer = self.pdf_viewer
        if viewer is None:
            return {
                "ok": True,
                "has_pdf": False,
                "path": "",
                "title": "",
                "current_page": 0,
                "total_pages": 0,
            }
        path = ""
        try:
            path = str(viewer.current_pdf_path() or "").strip()
        except Exception:
            path = ""
        total_pages = 0
        current_page = 0
        try:
            total_pages = max(0, int(viewer.page_count() or 0))
        except Exception:
            total_pages = 0
        try:
            current_page = max(0, int(viewer.current_page_index() or 0))
        except Exception:
            current_page = 0
        has_pdf = bool(path)
        return {
            "ok": True,
            "has_pdf": has_pdf,
            "path": path,
            "title": Path(path).name if has_pdf else "",
            "current_page": (current_page + 1) if has_pdf else 0,
            "total_pages": total_pages if has_pdf else 0,
            "pdfjs_active": bool(getattr(viewer, "pdfjs_active", lambda: False)()),
        }

    def get_pdf_text(self, max_chars: int = 8000, pages: int = 2) -> dict:
        state = self.get_pdf_state()
        if not bool(state.get("has_pdf")):
            return {"ok": False, "error": "No PDF is currently open in Annolid."}
        path_text = str(state.get("path") or "").strip()
        if not path_text:
            return {"ok": False, "error": "Active PDF path is unavailable."}
        try:
            import fitz  # type: ignore
        except Exception:
            return {"ok": False, "error": "PyMuPDF (pymupdf) is required."}
        path = Path(path_text)
        if not path.exists():
            return {"ok": False, "error": f"Active PDF is missing on disk: {path_text}"}

        limit = max(200, min(int(max_chars or 8000), 200000))
        pages_limit = max(1, min(int(pages or 2), 5))
        total_pages = int(state.get("total_pages") or 0)
        start_index = max(0, int(state.get("current_page") or 1) - 1)
        if total_pages > 0:
            start_index = min(start_index, total_pages - 1)

        text_parts: list[str] = []
        pages_read = 0
        try:
            with fitz.open(str(path)) as doc:
                total_pages = int(getattr(doc, "page_count", 0) or 0)
                if total_pages <= 0:
                    return {"ok": False, "error": "PDF has no pages."}
                start_index = min(start_index, total_pages - 1)
                end_index = min(start_index + pages_limit, total_pages)
                for idx in range(start_index, end_index):
                    page = doc.load_page(idx)
                    chunk = str(page.get_text("text") or "").strip()
                    if chunk:
                        text_parts.append(chunk)
                    pages_read += 1
        except Exception as exc:
            return {"ok": False, "error": f"Failed to read active PDF text: {exc}"}

        text = "\n\n".join(text_parts).strip()
        truncated = len(text) > limit
        if truncated:
            text = text[:limit]
        return {
            "ok": True,
            "path": path_text,
            "title": path.name,
            "current_page": start_index + 1,
            "total_pages": total_pages,
            "pages_read": pages_read,
            "text": text,
            "length": len(text),
            "truncated": truncated,
        }

    def get_pdf_sections(self, max_sections: int = 20, max_pages: int = 12) -> dict:
        state = self.get_pdf_state()
        if not bool(state.get("has_pdf")):
            return {"ok": False, "error": "No PDF is currently open in Annolid."}
        path_text = str(state.get("path") or "").strip()
        if not path_text:
            return {"ok": False, "error": "Active PDF path is unavailable."}
        try:
            import fitz  # type: ignore
        except Exception:
            return {"ok": False, "error": "PyMuPDF (pymupdf) is required."}
        path = Path(path_text)
        if not path.exists():
            return {"ok": False, "error": f"Active PDF is missing on disk: {path_text}"}

        section_limit = max(1, min(int(max_sections or 20), 200))
        page_limit = max(1, min(int(max_pages or 12), 100))
        found: list[dict[str, object]] = []
        seen: set[str] = set()
        total_pages = 0
        scanned_pages = 0
        common_titles = {
            "abstract",
            "introduction",
            "background",
            "related work",
            "methods",
            "materials and methods",
            "results",
            "discussion",
            "conclusion",
            "conclusions",
            "references",
            "acknowledgments",
            "acknowledgements",
            "supplementary materials",
        }

        def _is_section_candidate(line: str) -> bool:
            text = " ".join(str(line or "").split()).strip()
            if len(text) < 3 or len(text) > 140:
                return False
            if text.endswith(".") and len(text.split()) > 5:
                return False
            lowered = text.lower()
            if lowered in common_titles:
                return True
            if re.match(
                r"^(?:\d+(?:\.\d+){0,4}|[ivxlcdm]+)\s+[a-z0-9].*$",
                lowered,
                flags=re.IGNORECASE,
            ):
                return True
            words = text.split()
            if len(words) > 14:
                return False
            alpha = [c for c in text if c.isalpha()]
            if not alpha:
                return False
            upper_ratio = sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)
            title_ratio = sum(1 for w in words if w[:1].isupper()) / max(len(words), 1)
            starts_cap = bool(text[:1].isupper())
            return starts_cap and (upper_ratio >= 0.65 or title_ratio >= 0.75)

        try:
            with fitz.open(str(path)) as doc:
                total_pages = int(getattr(doc, "page_count", 0) or 0)
                if total_pages <= 0:
                    return {"ok": False, "error": "PDF has no pages."}
                end_page = min(page_limit, total_pages)
                for page_index in range(end_page):
                    scanned_pages += 1
                    page = doc.load_page(page_index)
                    lines = str(page.get_text("text") or "").splitlines()
                    for line_idx, line in enumerate(lines):
                        title = " ".join(str(line or "").split()).strip()
                        if not _is_section_candidate(title):
                            continue
                        key = re.sub(r"\s+", " ", title.lower()).strip()
                        if not key or key in seen:
                            continue
                        seen.add(key)
                        found.append(
                            {
                                "title": title,
                                "page": page_index + 1,
                                "line": line_idx + 1,
                            }
                        )
                        if len(found) >= section_limit:
                            break
                    if len(found) >= section_limit:
                        break
        except Exception as exc:
            return {"ok": False, "error": f"Failed to detect PDF sections: {exc}"}

        return {
            "ok": True,
            "path": path_text,
            "title": path.name,
            "sections": found,
            "count": len(found),
            "max_sections": section_limit,
            "scanned_pages": scanned_pages,
            "total_pages": total_pages,
        }
