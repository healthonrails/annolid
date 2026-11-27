"""Embedding Search dock widget supporting ONNX EmbeddingGemma 2 text & frame search."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.qt_compat import configure_compact_combo
from annolid.infrastructure import AnnotationStore


class EmbeddingSearchWidget(QtWidgets.QWidget):
    """Multimodal embedding search UI for finding similar frames and behaviors."""

    jumpToFrame = QtCore.Signal(int)
    statusMessage = QtCore.Signal(str)
    labelFramesRequested = QtCore.Signal(list)
    queueFramesRequested = QtCore.Signal(list)
    markFramesRequested = QtCore.Signal(list)
    clearMarkedFramesRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        # Capture a stable reference before Qt reparents this widget (e.g. when
        # inserted into a QDockWidget), so settings access remains reliable.
        self._settings = getattr(parent, "settings", None)
        self._current_image_path: Optional[Path] = None
        self._results: List[dict] = []
        self._last_bouts: List[dict] = []
        self._last_score_summary: dict[str, Any] = {}
        self._last_query_text: str = ""
        self._threshold_user_edited: bool = False
        self._updating_threshold_programmatically: bool = False
        self._is_running: bool = False
        self._search_started_monotonic: Optional[float] = None
        self._all_files: List[Path] = []
        self._video_path: Optional[Path] = None
        self._annotation_dir: Optional[Path] = None
        self._query_frame_index: Optional[int] = None
        self._service = None
        self._req_cls = None
        self._pending_marks: set[int] = set()
        self._flush_timer: Optional[QtCore.QTimer] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        query_layout = QtWidgets.QHBoxLayout()
        query_layout.setContentsMargins(0, 0, 0, 0)
        query_layout.setSpacing(4)
        self._query_input = QtWidgets.QLineEdit()
        self._query_input.setMinimumWidth(60)
        self._query_input.setPlaceholderText(
            "Search behavior (e.g. grooming, rearing)…"
        )
        self._query_input.returnPressed.connect(self._run_text_search)
        query_layout.addWidget(self._query_input, 1)

        self._text_button = QtWidgets.QPushButton("Text")
        self._text_button.setToolTip(
            "Run multimodal text-to-frame behavior search (ONNX EmbeddingGemma 2 / Qwen3-VL)"
        )
        self._text_button.clicked.connect(self._run_text_search)
        query_layout.addWidget(self._text_button)
        layout.addLayout(query_layout)

        image_layout = QtWidgets.QHBoxLayout()
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(4)
        self._image_label = QtWidgets.QLabel("No frame selected")
        self._image_label.setMinimumWidth(0)
        self._image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        self._image_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self._backend_combo = QtWidgets.QComboBox()
        self._backend_combo.addItem("EmbeddingGemma 2 (ONNX)", "embeddinggemma2_onnx")
        self._backend_combo.addItem("DINOv3 (visual)", "dinov3")
        self._backend_combo.addItem("Qwen3-VL Embedding", "qwen3vl")
        configure_compact_combo(self._backend_combo, min_chars=8, popup_min_width=220)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        image_layout.addWidget(self._backend_combo, 1)

        self._image_button = QtWidgets.QPushButton("Search")
        self._image_button.setToolTip(
            "Search video for frames similar to current frame"
        )
        self._image_button.clicked.connect(self._run_video_search)
        image_layout.addWidget(self._image_button)
        self._stop_button = QtWidgets.QPushButton("Stop")
        self._stop_button.clicked.connect(self._stop_search)
        self._stop_button.setEnabled(False)
        image_layout.addWidget(self._stop_button)
        layout.addWidget(self._image_label)
        layout.addLayout(image_layout)

        # Compact ONNX EmbeddingGemma 2 precision / soft-token / Matryoshka & clip-window controls.
        onnx_layout = QtWidgets.QHBoxLayout()
        onnx_layout.setContentsMargins(0, 0, 0, 0)
        onnx_layout.setSpacing(4)
        self._onnx_preset_combo = QtWidgets.QComboBox()
        self._onnx_preset_combo.addItem("Q4 · 140t · 768d", ("q4", 768, 140))
        self._onnx_preset_combo.addItem("Q4 · 70t · 256d (fast)", ("q4", 256, 70))
        self._onnx_preset_combo.addItem("Q4 · 280t · 768d (high)", ("q4", 768, 280))
        self._onnx_preset_combo.addItem("INT8 · 140t · 768d", ("quantized", 768, 140))
        self._onnx_preset_combo.addItem("FP32 · 140t · 768d", ("fp32", 768, 140))
        self._onnx_preset_combo.setToolTip(
            "ONNX EmbeddingGemma 2 quantization, visual soft-token budget (70=fastest, 140=balanced, 280=high detail), and Matryoshka dimension"
        )
        configure_compact_combo(
            self._onnx_preset_combo, min_chars=7, popup_min_width=190
        )
        onnx_layout.addWidget(self._onnx_preset_combo, 1)

        self._clip_combo = QtWidgets.QComboBox()
        self._clip_combo.addItem("1f (frame)", 1)
        self._clip_combo.addItem("4f (clip)", 4)
        self._clip_combo.addItem("8f (clip)", 8)
        self._clip_combo.setToolTip(
            "Temporal window per search sample (1f = single frame; 4f/8f = multi-frame behavior clip)"
        )
        configure_compact_combo(self._clip_combo, min_chars=6, popup_min_width=130)
        onnx_layout.addWidget(self._clip_combo, 1)
        layout.addLayout(onnx_layout)

        params_layout = QtWidgets.QHBoxLayout()
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(4)
        self._stride_spin = QtWidgets.QSpinBox()
        self._stride_spin.setRange(1, 10_000)
        self._stride_spin.setValue(5)
        self._stride_spin.setPrefix("Step ")
        self._stride_spin.setToolTip("Check every Nth frame for faster search")
        params_layout.addWidget(self._stride_spin, 1)

        self._auto_threshold_chk = QtWidgets.QCheckBox("Auto")
        self._auto_threshold_chk.setChecked(True)
        self._auto_threshold_chk.setToolTip(
            "Automatically adapt similarity threshold to each video's score distribution (uncheck to set a fixed ≥ threshold)"
        )
        self._auto_threshold_chk.toggled.connect(self._on_auto_threshold_toggled)
        params_layout.addWidget(self._auto_threshold_chk)

        self._threshold_spin = QtWidgets.QDoubleSpinBox()
        self._threshold_spin.setRange(-1.0, 1.0)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setValue(0.66)
        self._threshold_spin.setPrefix("≥")
        self._threshold_spin.setToolTip(
            "Fixed minimum cosine similarity when Auto is unchecked (EmbeddingGemma 2: ~0.66 text, ~0.82 frame)"
        )
        self._threshold_spin.valueChanged.connect(self._on_threshold_edited)
        params_layout.addWidget(self._threshold_spin, 1)

        self._topk_spin = QtWidgets.QSpinBox()
        self._topk_spin.setRange(1, 10_000)
        self._topk_spin.setValue(50)
        self._topk_spin.setPrefix("Top ")
        self._topk_spin.setToolTip("Maximum number of matches to keep")
        params_layout.addWidget(self._topk_spin, 1)
        for spin in (self._stride_spin, self._threshold_spin, self._topk_spin):
            spin.setMinimumWidth(48)
        layout.addLayout(params_layout)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(3)
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        self._file_filter = QtWidgets.QLineEdit()
        self._file_filter.setMinimumWidth(60)
        self._file_filter.setPlaceholderText("Filter frames…")
        self._file_filter.textChanged.connect(self._apply_file_filter)
        filter_layout.addWidget(self._file_filter)
        left_layout.addLayout(filter_layout)

        self._file_list = QtWidgets.QListWidget()
        self._file_list.itemSelectionChanged.connect(self._on_file_selected)
        self._file_list.itemDoubleClicked.connect(self._handle_file_double_click)
        left_layout.addWidget(self._file_list, 1)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(3)
        self._results_list = QtWidgets.QListWidget()
        self._results_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self._results_list.itemDoubleClicked.connect(self._handle_result_double_click)
        right_layout.addWidget(self._results_list, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        toggles_layout = QtWidgets.QHBoxLayout()
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.setSpacing(6)
        self._mark_chk = QtWidgets.QCheckBox("Mark timeline")
        self._mark_chk.setToolTip("Mark matching frames on the video timeline")
        self._mark_chk.setChecked(True)
        toggles_layout.addWidget(self._mark_chk)
        self._overlay_chk = QtWidgets.QCheckBox("Overlay shapes")
        self._overlay_chk.setChecked(False)
        self._overlay_chk.setToolTip(
            "If per-frame shapes exist, draw them on the frame before embedding."
        )
        toggles_layout.addWidget(self._overlay_chk)
        toggles_layout.addStretch(1)
        layout.addLayout(toggles_layout)

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        self._clear_marks_btn = QtWidgets.QPushButton("Clear marks")
        self._clear_marks_btn.clicked.connect(self.clearMarkedFramesRequested.emit)
        action_layout.addWidget(self._clear_marks_btn, 1)
        self._label_button = QtWidgets.QPushButton("Label Selected")
        self._label_button.setToolTip("Auto-label selected match frames")
        self._label_button.clicked.connect(self._label_selected_frames)
        action_layout.addWidget(self._label_button, 1)
        self._queue_button = QtWidgets.QPushButton("To Queue")
        self._queue_button.setToolTip(
            "Send selected match frames (or all matches if none selected) to the Label Queue without extracting PNGs"
        )
        self._queue_button.clicked.connect(self._queue_selected_frames)
        action_layout.addWidget(self._queue_button, 1)
        layout.addLayout(action_layout)

        self._on_backend_changed()

    def last_query_text(self) -> str:
        """Return the most recent non-empty behavior text query."""
        return self._last_query_text

    def last_bouts(self) -> List[dict]:
        """Return grouped behavior bouts from the most recent search."""
        return list(self._last_bouts)

    def last_score_summary(self) -> dict[str, Any]:
        """Return similarity score distribution summary from the most recent search."""
        return dict(self._last_score_summary)

    def set_files(self, files: List[Path]) -> None:
        """Populate the file list with available frame image paths."""
        self._all_files = sorted(Path(f).resolve() for f in files)
        self._refresh_file_list()

    def clear_files(self) -> None:
        self._all_files = []
        self._refresh_file_list()

    def set_video_path(self, path: Optional[Path]) -> None:
        raw_key = str(path) if path else None
        if raw_key == getattr(self, "_video_path_raw_key", None):
            return
        self._video_path_raw_key = raw_key
        self._video_path = Path(path).expanduser().resolve() if path else None

    def set_annotation_dir(self, path: Optional[Path]) -> None:
        raw_key = str(path) if path else None
        if raw_key == getattr(self, "_annotation_dir_raw_key", None):
            return
        self._annotation_dir_raw_key = raw_key
        self._annotation_dir = Path(path).expanduser().resolve() if path else None

    def set_query_frame_index(self, frame_index: Optional[int]) -> None:
        self._query_frame_index = int(frame_index) if frame_index is not None else None
        self._update_image_label()

    def set_query_image(self, path: Optional[Path]) -> None:
        self._current_image_path = path
        if path:
            idx = AnnotationStore.frame_number_from_path(path)
            if idx is not None:
                self._query_frame_index = int(idx)
        self._update_image_label()

    def _on_backend_changed(self) -> None:
        backend = str(self._backend_combo.currentData() or "embeddinggemma2_onnx")
        is_onnx = backend == "embeddinggemma2_onnx"
        supports_text = backend in {"embeddinggemma2_onnx", "qwen3vl"}
        self._onnx_preset_combo.setEnabled(is_onnx and not self._is_running)
        self._clip_combo.setEnabled(supports_text and not self._is_running)
        if not supports_text:
            self._clip_combo.setCurrentIndex(0)
        if not self._threshold_user_edited:
            self._updating_threshold_programmatically = True
            try:
                self._threshold_spin.setValue(0.66 if is_onnx else 0.35)
            finally:
                self._updating_threshold_programmatically = False
        self._sync_threshold_spin_enabled()

    def _on_auto_threshold_toggled(self, _checked: bool) -> None:
        self._sync_threshold_spin_enabled()

    def _on_threshold_edited(self, _value: float) -> None:
        if not self._updating_threshold_programmatically:
            self._threshold_user_edited = True

    def _sync_threshold_spin_enabled(self) -> None:
        self._threshold_spin.setEnabled(
            (not self._is_running) and (not self._auto_threshold_chk.isChecked())
        )

    def _ensure_service(self) -> bool:
        if self._service is not None and self._req_cls is not None:
            return True
        try:
            from annolid.gui.frame_similarity_service import (
                FrameSimilaritySearchRequest,
                FrameSimilarityService,
            )
        except Exception as exc:
            self.statusMessage.emit(f"Frame search unavailable: {exc}")
            return False
        self._req_cls = FrameSimilaritySearchRequest
        self._service = FrameSimilarityService(self)
        self._service.started.connect(self._on_search_started)
        self._service.progress.connect(self._on_search_progress)
        self._service.matchFound.connect(self._on_match_found)
        self._service.scoreSummary.connect(self._on_score_summary)
        self._service.boutsFound.connect(self._on_bouts_found)
        self._service.finished.connect(self._on_search_finished)
        self._service.error.connect(self._on_search_error)
        return True

    def _collect_backend_params(self, backend: str) -> dict[str, Any]:
        if backend == "embeddinggemma2_onnx":
            preset = self._onnx_preset_combo.currentData()
            if isinstance(preset, tuple) and len(preset) >= 3:
                quant, dim, soft_tokens = preset[0], preset[1], preset[2]
            elif isinstance(preset, tuple) and len(preset) == 2:
                quant, dim, soft_tokens = preset[0], preset[1], 140
            else:
                quant, dim, soft_tokens = "q4", 768, 140
            params: dict[str, Any] = {
                "quantization": str(quant),
                "matryoshka_dim": int(dim),
                "image_max_soft_tokens": int(soft_tokens),
            }
            if self._settings is not None:
                try:
                    custom_repo = self._settings.value(
                        "embedding_search/onnx_model_id", ""
                    )
                    if str(custom_repo or "").strip():
                        params["model_id"] = str(custom_repo).strip()
                except Exception:
                    pass
            return params

        if backend == "dinov3":
            settings = self._settings
            if settings is not None:
                try:
                    model_name = settings.value(
                        "patch_similarity/model",
                        "facebook/dinov3-vits16-pretrain-lvd1689m",
                    )
                    return {
                        "model_name": str(model_name),
                        # Match patch-similarity defaults (higher resolution embedding).
                        "short_side": 768,
                    }
                except Exception:
                    return {}
        return {}

    def _cancel_flush_timer(self) -> None:
        if self._flush_timer is not None and self._flush_timer.isActive():
            self._flush_timer.stop()
        self._pending_marks.clear()

    def _submit_video_request(
        self,
        *,
        query_mode: str,
        query_text: Optional[str] = None,
        query_frame_index: Optional[int] = None,
    ) -> None:
        if self._video_path is None:
            self.statusMessage.emit("Open a video before searching.")
            return
        if not self._ensure_service():
            return

        backend = str(self._backend_combo.currentData() or "embeddinggemma2_onnx")
        if query_mode in {"text", "hybrid"} and backend == "dinov3":
            self.statusMessage.emit(
                "DINOv3 is visual-only. Switch backend to EmbeddingGemma 2 (ONNX) or Qwen3-VL for text behavior search."
            )
            return

        backend_params = self._collect_backend_params(backend)
        clip_frames = int(self._clip_combo.currentData() or 1)
        if backend == "dinov3":
            clip_frames = 1

        self._last_bouts = []
        self._last_score_summary = {}
        self._last_query_text = str(query_text or "").strip()
        req_threshold = (
            None
            if self._auto_threshold_chk.isChecked()
            else float(self._threshold_spin.value())
        )

        req = self._req_cls(
            video_path=Path(self._video_path),
            query_frame_index=query_frame_index,
            annotation_dir=Path(self._annotation_dir) if self._annotation_dir else None,
            overlay_shapes=bool(self._overlay_chk.isChecked()),
            stride=int(self._stride_spin.value()),
            top_k=int(self._topk_spin.value()),
            threshold=req_threshold,
            backend=backend,
            backend_params=backend_params,
            query_text=query_text,
            query_mode=query_mode,
            clip_frames=clip_frames,
            group_bouts=True,
        )
        self._cancel_flush_timer()
        if self._mark_chk.isChecked():
            self.clearMarkedFramesRequested.emit()
        if not self._service.request(req):
            self.statusMessage.emit("Search is already running…")
            self._set_running_state(False)
            return
        self._set_running_state(True)

    def _run_image_search(self) -> None:
        """Backward-compatible helper for querying a pre-built frame folder index."""
        if self._current_image_path is None:
            selected = self._file_list.selectedItems()
            if selected:
                candidate = selected[0].data(QtCore.Qt.UserRole)
                if candidate:
                    self._current_image_path = Path(candidate)
                    self._image_label.setText(str(candidate))
        if self._current_image_path is None:
            self.statusMessage.emit("Select a frame before searching.")
            return
        if not self._current_image_path.exists():
            self.statusMessage.emit("Frame image not found for search.")
            return
        try:
            from annolid.agents.frame_search import search_frames
        except Exception as exc:
            self.statusMessage.emit(f"Embedding search unavailable: {exc}")
            return

        try:
            results = search_frames(str(self._current_image_path), limit=10)
        except Exception as exc:
            self.statusMessage.emit(f"Search failed: {exc}")
            return
        self._populate_results(results)

    def _run_text_search(self) -> None:
        query = self._query_input.text().strip()
        if not query:
            self.statusMessage.emit("Enter behavior text to search.")
            return
        self._submit_video_request(
            query_mode="text",
            query_text=query,
            query_frame_index=None,
        )

    def _run_video_search(self) -> None:
        if self._query_frame_index is None:
            # Convenience: if no frame is selected yet but user typed text, run text search.
            if self._query_input.text().strip():
                self._run_text_search()
                return
            self.statusMessage.emit("Select a frame or enter behavior text to search.")
            return
        self._submit_video_request(
            query_mode="frame",
            query_text=None,
            query_frame_index=int(self._query_frame_index),
        )

    def _stop_search(self) -> None:
        if self._service is None:
            return
        try:
            self._service.stop()
        except Exception:
            pass
        self._cancel_flush_timer()
        self.statusMessage.emit("Stopping search…")
        self._stop_button.setEnabled(False)

    def _set_running_state(self, running: bool) -> None:
        self._is_running = bool(running)
        self._image_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._backend_combo.setEnabled(not running)
        if running:
            self._onnx_preset_combo.setEnabled(False)
            self._clip_combo.setEnabled(False)
        else:
            self._on_backend_changed()
        self._stride_spin.setEnabled(not running)
        self._auto_threshold_chk.setEnabled(not running)
        self._sync_threshold_spin_enabled()
        self._topk_spin.setEnabled(not running)
        self._file_list.setEnabled(not running)
        self._file_filter.setEnabled(not running)
        self._query_input.setEnabled(not running)
        self._text_button.setEnabled(not running)
        self._label_button.setEnabled(not running)
        self._queue_button.setEnabled(not running)
        self._clear_marks_btn.setEnabled(not running)
        self._mark_chk.setEnabled(not running)
        self._overlay_chk.setEnabled(not running)

    def _on_search_started(self) -> None:
        self._search_started_monotonic = time.monotonic()
        if self._last_query_text:
            self.statusMessage.emit(f"Searching behavior '{self._last_query_text}'…")
        else:
            self.statusMessage.emit("Embedding search started…")

    def _on_search_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.statusMessage.emit(f"Searching… {done}")
            return
        eta_note = ""
        if self._search_started_monotonic is not None and 0 < done < total:
            elapsed = time.monotonic() - self._search_started_monotonic
            if elapsed >= 0.4:
                rem = (elapsed / float(done)) * float(total - done)
                eta_note = f" (~{_format_eta_seconds(rem)} left)"
        self.statusMessage.emit(f"Searching… {done}/{total}{eta_note}")

    def _on_match_found(self, frame_index: int, similarity: float) -> None:
        _ = similarity
        if not self._mark_chk.isChecked():
            return
        self._pending_marks.add(int(frame_index))
        if self._flush_timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_marks)
            self._flush_timer = timer
        if not self._flush_timer.isActive():
            self._flush_timer.start(250)

    def _on_score_summary(self, summary: dict) -> None:
        self._last_score_summary = dict(summary or {})

    def _on_bouts_found(self, bouts: list) -> None:
        self._last_bouts = list(bouts or [])

    def _flush_marks(self) -> None:
        if not self._pending_marks:
            return
        frames = sorted(self._pending_marks)
        self._pending_marks.clear()
        self.markFramesRequested.emit(frames)

    def _on_search_finished(self, results: list) -> None:
        self._cancel_flush_timer()
        self._set_running_state(False)
        self._populate_results(list(results or []))
        if self._mark_chk.isChecked():
            frames = []
            for item in results or []:
                idx = item.get("frame_index")
                if idx is not None:
                    frames.append(int(idx))
            if frames:
                self.markFramesRequested.emit(sorted(set(frames)))

    def _on_search_error(self, message: str) -> None:
        self._cancel_flush_timer()
        self._set_running_state(False)
        self.statusMessage.emit(f"Search failed: {message}")

    def _populate_results(self, results: List[dict]) -> None:
        self._results = list(results or [])
        self._results_list.clear()
        summary = self._last_score_summary or {}
        if not self._results:
            if summary and int(summary.get("count", 0)) > 0:
                eff_thr = float(summary.get("effective_threshold", 0.0))
                max_sim = float(summary.get("max_similarity", 0.0))
                min_sim = float(summary.get("min_similarity", 0.0))
                best_idx = summary.get("best_frame_index")
                best_str = (
                    f"Frame {int(best_idx):09d}" if best_idx is not None else "n/a"
                )
                self.statusMessage.emit(
                    f"No matches ≥ {eff_thr:.2f} (best {best_str} • sim={max_sim:.3f}; range {min_sim:.2f}–{max_sim:.2f})."
                )
            else:
                self.statusMessage.emit("No results found.")
            return
        for item in self._results:
            uri = str(item.get("image_uri") or "")
            caption = str(item.get("caption") or "")
            similarity = item.get("similarity")
            if similarity is not None:
                try:
                    sim_str = f"{float(similarity):.3f}"
                except Exception:
                    sim_str = None
            else:
                sim_str = None
            frame_idx = item.get("frame_index")
            if frame_idx is None:
                frame_idx = AnnotationStore.frame_number_from_path(uri)
            start_f = item.get("start_frame")
            end_f = item.get("end_frame")
            span_str = ""
            if start_f is not None and end_f is not None and int(end_f) > int(start_f):
                span_str = f" [{int(start_f)}-{int(end_f)}]"
            if frame_idx is not None and sim_str is not None:
                label = f"Frame {int(frame_idx):09}{span_str} • sim={sim_str}"
            elif frame_idx is not None:
                label = f"Frame {int(frame_idx):09}{span_str}"
            else:
                label = caption if caption else uri
            widget_item = QtWidgets.QListWidgetItem(label)
            widget_item.setData(QtCore.Qt.UserRole, uri)
            if frame_idx is not None:
                widget_item.setData(QtCore.Qt.UserRole + 1, int(frame_idx))
            self._results_list.addItem(widget_item)

        thr_suffix = ""
        if summary and str(summary.get("threshold_mode") or "") == "auto":
            thr_suffix = (
                f" (auto ≥ {float(summary.get('effective_threshold', 0.0)):.2f})"
            )

        if self._last_bouts:
            self.statusMessage.emit(
                f"Found {len(self._results)} matches across {len(self._last_bouts)} bouts{thr_suffix}."
            )
        else:
            self.statusMessage.emit(f"Found {len(self._results)} results{thr_suffix}.")

    def _handle_result_double_click(self, item: QtWidgets.QListWidgetItem) -> None:
        idx = item.data(QtCore.Qt.UserRole + 1)
        if idx is not None:
            try:
                self.jumpToFrame.emit(int(idx))
                return
            except Exception:
                pass
        uri = item.data(QtCore.Qt.UserRole)
        if not uri:
            return
        path = _resolve_uri_to_path(str(uri))
        frame_idx = AnnotationStore.frame_number_from_path(path) if path else None
        if frame_idx is not None:
            self.jumpToFrame.emit(int(frame_idx))
        else:
            self.statusMessage.emit("Unable to map result to a frame index.")

    def _label_selected_frames(self) -> None:
        items = self._results_list.selectedItems()
        if not items:
            self.statusMessage.emit("Select one or more results to label.")
            return
        frames: list[int] = []
        for item in items:
            idx = item.data(QtCore.Qt.UserRole + 1)
            if idx is None:
                uri = item.data(QtCore.Qt.UserRole)
                frame_idx = AnnotationStore.frame_number_from_path(
                    _resolve_uri_to_path(str(uri)) if uri else None
                )
                idx = frame_idx
            if idx is not None:
                frames.append(int(idx))
        if not frames:
            self.statusMessage.emit("No valid frame indices in selection.")
            return
        self.labelFramesRequested.emit(sorted(set(frames)))

    def _queue_selected_frames(self) -> None:
        items = self._results_list.selectedItems()
        if not items:
            items = [
                self._results_list.item(i)
                for i in range(self._results_list.count())
                if self._results_list.item(i) is not None
            ]
        if not items:
            self.statusMessage.emit("Run a search first to queue matching frames.")
            return
        frames: list[int] = []
        for item in items:
            idx = item.data(QtCore.Qt.UserRole + 1)
            if idx is None:
                uri = item.data(QtCore.Qt.UserRole)
                idx = AnnotationStore.frame_number_from_path(
                    _resolve_uri_to_path(str(uri)) if uri else None
                )
            if idx is not None:
                frames.append(int(idx))
        if not frames:
            self.statusMessage.emit("No valid frame indices to queue.")
            return
        self.queueFramesRequested.emit(sorted(set(frames)))

    # type: ignore[override]
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._cancel_flush_timer()
        try:
            if self._service is not None:
                self._service.close()
        except Exception:
            pass
        super().closeEvent(event)

    # type: ignore[override]
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_image_label()

    def _update_image_label(self) -> None:
        if self._query_frame_index is not None:
            display = f"Frame {int(self._query_frame_index):09}"
            self._image_label.setText(display)
            tooltip = ""
            if self._video_path is not None:
                tooltip = (
                    f"{self._video_path.name} • frame {int(self._query_frame_index):09}"
                )
            self._image_label.setToolTip(tooltip)
        else:
            self._image_label.setText("No frame selected")
            self._image_label.setToolTip("")

    def _refresh_file_list(self) -> None:
        self._file_list.clear()
        if not self._all_files:
            return
        filter_text = self._file_filter.text().strip().lower()
        for path in self._all_files:
            name = path.name
            if filter_text and filter_text not in name.lower():
                continue
            item = QtWidgets.QListWidgetItem(name)
            item.setToolTip(str(path))
            item.setData(QtCore.Qt.UserRole, str(path))
            self._file_list.addItem(item)

    def _apply_file_filter(self) -> None:
        self._refresh_file_list()

    def _on_file_selected(self) -> None:
        items = self._file_list.selectedItems()
        if not items:
            return
        path = items[0].data(QtCore.Qt.UserRole)
        if path:
            resolved = Path(str(path))
            self.set_query_image(resolved)

    def _handle_file_double_click(self, item: QtWidgets.QListWidgetItem) -> None:
        path_str = item.data(QtCore.Qt.UserRole)
        if not path_str:
            return
        resolved = Path(str(path_str))
        frame_idx = AnnotationStore.frame_number_from_path(resolved)
        if frame_idx is not None:
            self.jumpToFrame.emit(int(frame_idx))
        else:
            self.set_query_image(resolved)


def _format_eta_seconds(seconds: float) -> str:
    sec = max(1, int(round(float(seconds))))
    mins, rem = divmod(sec, 60)
    if mins >= 60:
        hours, mins = divmod(mins, 60)
        return f"{hours}h {mins}m"
    if mins > 0:
        return f"{mins}m {rem:02d}s"
    return f"{rem}s"


def _resolve_uri_to_path(uri: str) -> Optional[Path]:
    if uri.startswith("file://"):
        uri = uri.replace("file://", "", 1)
    try:
        path = Path(uri)
    except Exception:
        return None
    return path if path.exists() else None
