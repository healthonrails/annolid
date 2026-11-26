from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt

from annolid.gui.widgets.pdf_viewer import PdfViewerWidget
from annolid.gui.widgets.pdf_controls import PdfControlsWidget
from annolid.gui.widgets.pdf_reader_controls import PdfReaderControlsWidget
from annolid.gui.widgets.pdf_reading_log import PdfReadingLogWidget
from annolid.gui.widgets.tts_controls import TtsControlsWidget
from annolid.services.pdf_parsing import (
    detect_document_language,
    extract_pdf_sections,
    extract_pdf_text,
    extract_pdf_text_sample,
)
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

    def _prune_missing_recent_pdfs(self) -> None:
        before = list(self._pdf_files)
        kept: list[str] = []
        for raw in before:
            text = str(raw or "").strip()
            if not text:
                continue
            try:
                candidate = Path(text)
                if not candidate.exists() or not candidate.is_file():
                    continue
            except Exception:
                continue
            kept.append(text)
        if kept != before:
            self._pdf_files = kept[-self._RECENT_PDFS_LIMIT :]
            self._save_recent_pdfs()

    def _remove_recent_pdf(self, pdf_path: str) -> None:
        value = str(pdf_path or "").strip()
        if not value:
            return
        try:
            resolved = str(Path(value).expanduser().resolve())
        except Exception:
            resolved = value
        updated = [p for p in self._pdf_files if str(p).strip() != resolved]
        if updated == self._pdf_files:
            return
        self._pdf_files = updated[-self._RECENT_PDFS_LIMIT :]
        self._save_recent_pdfs()
        self._populate_pdf_file_list()

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
            viewer.keyword_config_changed.connect(self._on_keyword_config_changed)
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
            controls.keywords_changed.connect(self._set_highlight_keywords)
            controls.keyword_highlighting_changed.connect(
                self._set_keyword_highlighting_enabled
            )
            controls.auto_keywords_changed.connect(
                self._set_auto_keyword_highlighting_enabled
            )

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
        requested_path = str(pdf_path or "").strip()
        if not requested_path:
            return
        candidate = Path(requested_path).expanduser()
        if not candidate.exists() or not candidate.is_file():
            self._remove_recent_pdf(str(candidate))
            logger.warning("Skipping missing PDF path: %s", candidate)
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("PDF Missing"),
                self.window.tr("PDF file not found:\n%1").replace("%1", str(candidate)),
            )
            self.window._set_active_view("canvas")
            return

        viewer = self.ensure_pdf_viewer()
        try:
            viewer.load_pdf(str(candidate))
        except Exception as exc:  # pragma: no cover - user-facing dialog
            if isinstance(exc, FileNotFoundError):
                self._remove_recent_pdf(str(candidate))
                logger.warning("PDF no longer exists: %s", candidate)
                QtWidgets.QMessageBox.warning(
                    self.window,
                    self.window.tr("PDF Missing"),
                    self.window.tr("PDF file not found:\n%1").replace(
                        "%1", str(candidate)
                    ),
                )
                self.window._set_active_view("canvas")
                return
            logger.error("Failed to open PDF %s: %s", candidate, exc, exc_info=True)
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

        def finalize_pdf_view():
            try:
                self.window._set_active_view("pdf")
            except Exception as e:
                logger.error("Failed to switch to PDF view: %s", e)
                return

            self._disable_labelme_file_selection()
            self._close_unrelated_docks_for_pdf()
            self.ensure_pdf_tts_dock()
            # Pick a reasonable default TTS language/voice early so the dock reflects it immediately.
            self._auto_select_tts_from_pdf_language(str(candidate))
            self.ensure_pdf_controls_dock()
            self.ensure_pdf_reader_dock()
            self.ensure_pdf_log_dock()
            self._tighten_right_docks()
            try:
                if self.pdf_viewer is not None:
                    self.pdf_viewer.fit_to_width()
            except Exception:
                pass

        # Use a singleShot timer to ensure this happens after any pending state resets
        # (e.g. from closeFile or resetState triggered by side effects).
        QtCore.QTimer.singleShot(0, finalize_pdf_view)

        self._record_pdf_entry(str(candidate))
        self.window.lastOpenDir = str(candidate.parent)
        self.window.statusBar().showMessage(
            self.window.tr("Loaded PDF %1").replace("%1", candidate.name), 3000
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
                self.pdf_viewer.stop_reader()
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
        # Preserve the user's chosen voice when already in the same language family.
        try:
            current = controls.current_settings()
            cur_lang = str(current.get("lang") or "").strip().lower()
            if cur_lang and cur_lang.split("-")[0] == lang.split("-")[0]:
                return
        except Exception:
            pass
        voice = self._suggest_default_voice(lang)
        try:
            controls.set_language_and_voice(lang=lang, voice=voice, persist=True)
        except Exception:
            return

    @staticmethod
    def _extract_pdf_text_sample(pdf_path: str) -> str:
        """Extract a small text sample to infer document language."""
        return extract_pdf_text_sample(pdf_path, max_pages=3, max_chars=12000)

    @staticmethod
    def _detect_kokoro_language(text: str) -> str:
        """Return a Kokoro language code best-effort (e.g. zh, ja, en-us)."""
        return detect_document_language(text)

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
        self._prune_missing_recent_pdfs()
        widget.clear()
        for path in self._pdf_files:
            name = Path(path).name
            item = QtWidgets.QListWidgetItem(name)
            item.setData(Qt.UserRole, path)
            item.setData(Qt.UserRole + 1, "pdf")
            item.setToolTip(path)
            widget.addItem(item)
        try:
            if hasattr(self.window, "_apply_file_dock_sort"):
                self.window._apply_file_dock_sort()
        except Exception:
            pass
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
        path_text = str(path).strip()
        if not path_text:
            return
        if not Path(path_text).exists():
            self._remove_recent_pdf(path_text)
            self.window.statusBar().showMessage(
                self.window.tr("PDF missing on disk: %1").replace(
                    "%1", Path(path_text).name
                ),
                4000,
            )
            return
        self.show_pdf_in_viewer(path_text)

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
        controls.set_keywords(viewer.highlight_keywords())
        controls.set_keyword_highlighting_enabled(viewer.keyword_highlighting_enabled())
        controls.set_auto_keywords_enabled(viewer.auto_keyword_highlighting_enabled())

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

    def _set_highlight_keywords(self, keywords: list[str]) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.set_highlight_keywords(keywords)

    def _set_keyword_highlighting_enabled(self, enabled: bool) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.set_keyword_highlighting_enabled(enabled)

    def _set_auto_keyword_highlighting_enabled(self, enabled: bool) -> None:
        if self.pdf_viewer is not None:
            self.pdf_viewer.set_auto_keyword_highlighting_enabled(enabled)

    def _on_keyword_config_changed(
        self, keywords: list[str], enabled: bool, auto_enabled: bool
    ) -> None:
        controls = self.pdf_reader_controls
        if controls is None:
            return
        controls.set_keywords(keywords)
        controls.set_keyword_highlighting_enabled(enabled)
        controls.set_auto_keywords_enabled(auto_enabled)

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

    def get_pdf_text(
        self, max_chars: int = 8000, pages: int = 2, start_page: int = 0
    ) -> dict:
        state = self.get_pdf_state()
        if not bool(state.get("has_pdf")):
            return {"ok": False, "error": "No PDF is currently open in Annolid."}
        path_text = str(state.get("path") or "").strip()
        if not path_text:
            return {"ok": False, "error": "Active PDF path is unavailable."}
        path = Path(path_text)
        if not path.exists():
            return {"ok": False, "error": f"Active PDF is missing on disk: {path_text}"}

        limit = max(200, min(int(max_chars or 8000), 200000))
        pages_limit = max(1, min(int(pages or 2), 5))
        total_pages = int(state.get("total_pages") or 0)
        requested_page = max(0, int(start_page or 0))
        if requested_page > 0:
            start_index = requested_page - 1
        else:
            start_index = max(0, int(state.get("current_page") or 1) - 1)
        if total_pages > 0:
            start_index = min(start_index, total_pages - 1)

        try:
            raw_text, pages_read, total_pages, _backend = extract_pdf_text(
                path,
                start_index=start_index,
                pages_limit=pages_limit,
                dehyphenate=True,
                fallback_blocks_words=True,
            )
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": f"Failed to read active PDF text: {exc}"}

        if total_pages <= 0:
            return {"ok": False, "error": "PDF has no pages."}

        start_index = min(start_index, total_pages - 1)
        text = raw_text.strip()
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
        path = Path(path_text)
        if not path.exists():
            return {"ok": False, "error": f"Active PDF is missing on disk: {path_text}"}
        try:
            result = extract_pdf_sections(
                path, max_sections=max_sections, max_pages=max_pages
            )
            if result.get("ok"):
                result["path"] = path_text
            return result
        except Exception as exc:
            return {"ok": False, "error": f"Failed to detect PDF sections: {exc}"}
