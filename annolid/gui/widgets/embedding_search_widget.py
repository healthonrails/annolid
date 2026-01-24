from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.utils.annotation_store import AnnotationStore


class EmbeddingSearchWidget(QtWidgets.QWidget):
    """Embedding search UI for finding similar frames."""

    jumpToFrame = QtCore.Signal(int)
    statusMessage = QtCore.Signal(str)
    labelFramesRequested = QtCore.Signal(list)
    markFramesRequested = QtCore.Signal(list)
    clearMarkedFramesRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        # Capture a stable reference before Qt reparents this widget (e.g. when
        # inserted into a QDockWidget), so settings access remains reliable.
        self._settings = getattr(parent, "settings", None)
        self._current_image_path: Optional[Path] = None
        self._results: List[dict] = []
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

        query_layout = QtWidgets.QHBoxLayout()
        self._query_input = QtWidgets.QLineEdit()
        self._query_input.setPlaceholderText("Search text (optional)")
        query_layout.addWidget(self._query_input, 1)

        self._text_button = QtWidgets.QPushButton("Search Text")
        self._text_button.clicked.connect(self._run_text_search)
        query_layout.addWidget(self._text_button)
        layout.addLayout(query_layout)

        image_layout = QtWidgets.QHBoxLayout()
        self._image_label = QtWidgets.QLabel("No frame selected")
        self._image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self._image_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        image_layout.addWidget(self._image_label, 1)
        self._stride_spin = QtWidgets.QSpinBox()
        self._stride_spin.setRange(1, 10_000)
        self._stride_spin.setValue(5)
        self._stride_spin.setToolTip("Check every Nth frame for faster search")
        image_layout.addWidget(self._stride_spin)

        self._threshold_spin = QtWidgets.QDoubleSpinBox()
        self._threshold_spin.setRange(-1.0, 1.0)
        self._threshold_spin.setDecimals(3)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setValue(0.35)
        self._threshold_spin.setToolTip("Only mark frames with similarity >= threshold")
        image_layout.addWidget(self._threshold_spin)

        self._topk_spin = QtWidgets.QSpinBox()
        self._topk_spin.setRange(1, 10_000)
        self._topk_spin.setValue(50)
        self._topk_spin.setToolTip("Maximum number of matches to keep")
        image_layout.addWidget(self._topk_spin)

        self._backend_combo = QtWidgets.QComboBox()
        self._backend_combo.addItem("DINOv3 (default)", "dinov3")
        self._backend_combo.addItem("Qwen3-VL Embedding", "qwen3vl")
        image_layout.addWidget(self._backend_combo)

        self._image_button = QtWidgets.QPushButton("Search Video")
        self._image_button.clicked.connect(self._run_video_search)
        image_layout.addWidget(self._image_button)
        self._stop_button = QtWidgets.QPushButton("Stop")
        self._stop_button.clicked.connect(self._stop_search)
        self._stop_button.setEnabled(False)
        image_layout.addWidget(self._stop_button)
        layout.addLayout(image_layout)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Horizontal)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout = QtWidgets.QHBoxLayout()
        self._file_filter = QtWidgets.QLineEdit()
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
        self._results_list = QtWidgets.QListWidget()
        self._results_list.itemDoubleClicked.connect(self._handle_result_double_click)
        right_layout.addWidget(self._results_list, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        action_layout = QtWidgets.QHBoxLayout()
        self._mark_chk = QtWidgets.QCheckBox("Mark matches on timeline")
        self._mark_chk.setChecked(True)
        action_layout.addWidget(self._mark_chk)
        self._overlay_chk = QtWidgets.QCheckBox("Overlay annotations")
        self._overlay_chk.setChecked(True)
        self._overlay_chk.setToolTip(
            "If per-frame shapes exist, draw them on the frame before embedding."
        )
        action_layout.addWidget(self._overlay_chk)
        self._clear_marks_btn = QtWidgets.QPushButton("Clear marks")
        self._clear_marks_btn.clicked.connect(self.clearMarkedFramesRequested.emit)
        action_layout.addWidget(self._clear_marks_btn)
        self._label_button = QtWidgets.QPushButton("Label Selected Frames")
        self._label_button.clicked.connect(self._label_selected_frames)
        action_layout.addWidget(self._label_button)
        action_layout.addStretch(1)
        layout.addLayout(action_layout)

    def set_files(self, files: List[Path]) -> None:
        """Populate the file list with available frame image paths."""
        self._all_files = sorted(Path(f).resolve() for f in files)
        self._refresh_file_list()

    def clear_files(self) -> None:
        self._all_files = []
        self._refresh_file_list()

    def set_video_path(self, path: Optional[Path]) -> None:
        self._video_path = Path(path).expanduser().resolve() if path else None

    def set_annotation_dir(self, path: Optional[Path]) -> None:
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

    def _run_image_search(self) -> None:
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
            self.statusMessage.emit("Enter text to search.")
            return
        self.statusMessage.emit("Text search is not available in this build.")

    def _run_video_search(self) -> None:
        if self._video_path is None:
            self.statusMessage.emit("Open a video before searching.")
            return
        if self._query_frame_index is None:
            self.statusMessage.emit("Select a frame before searching.")
            return

        if self._service is None:
            try:
                from annolid.gui.frame_similarity_service import (
                    FrameSimilaritySearchRequest,
                    FrameSimilarityService,
                )
            except Exception as exc:
                self.statusMessage.emit(f"Frame search unavailable: {exc}")
                return
            self._req_cls = FrameSimilaritySearchRequest
            self._service = FrameSimilarityService(self)
            self._service.started.connect(self._on_search_started)
            self._service.progress.connect(
                lambda done, total: self.statusMessage.emit(
                    f"Searching… {done}/{total}"
                )
            )
            self._service.matchFound.connect(self._on_match_found)
            self._service.finished.connect(self._on_search_finished)
            self._service.error.connect(self._on_search_error)

        backend = self._backend_combo.currentData()
        backend_params: dict = {}
        if str(backend) == "dinov3":
            settings = self._settings
            if settings is not None:
                try:
                    model_name = settings.value(
                        "patch_similarity/model",
                        "facebook/dinov3-vits16-pretrain-lvd1689m",
                    )
                    backend_params = {
                        "model_name": str(model_name),
                        # Match patch-similarity defaults (higher resolution embedding).
                        "short_side": 768,
                    }
                except Exception:
                    backend_params = {}

        req = self._req_cls(
            video_path=Path(self._video_path),
            query_frame_index=int(self._query_frame_index),
            annotation_dir=Path(self._annotation_dir) if self._annotation_dir else None,
            overlay_shapes=bool(self._overlay_chk.isChecked()),
            stride=int(self._stride_spin.value()),
            top_k=int(self._topk_spin.value()),
            threshold=float(self._threshold_spin.value()),
            backend=str(backend),
            backend_params=backend_params,
        )
        if self._mark_chk.isChecked():
            self._pending_marks = set()
            self.clearMarkedFramesRequested.emit()
        if not self._service.request(req):
            self.statusMessage.emit("Search is already running…")
            self._set_running_state(False)
            return
        self._set_running_state(True)

    def _stop_search(self) -> None:
        if self._service is None:
            return
        try:
            self._service.stop()
        except Exception:
            pass
        self.statusMessage.emit("Stopping search…")
        self._stop_button.setEnabled(False)

    def _set_running_state(self, running: bool) -> None:
        self._image_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._backend_combo.setEnabled(not running)
        self._stride_spin.setEnabled(not running)
        self._threshold_spin.setEnabled(not running)
        self._topk_spin.setEnabled(not running)
        self._file_list.setEnabled(not running)
        self._file_filter.setEnabled(not running)
        self._query_input.setEnabled(not running)
        self._text_button.setEnabled(not running)
        self._label_button.setEnabled(not running)
        self._clear_marks_btn.setEnabled(not running)
        self._mark_chk.setEnabled(not running)
        self._overlay_chk.setEnabled(not running)

    def _on_search_started(self) -> None:
        self.statusMessage.emit("Embedding search started…")

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

    def _flush_marks(self) -> None:
        if not self._pending_marks:
            return
        frames = sorted(self._pending_marks)
        self._pending_marks.clear()
        self.markFramesRequested.emit(frames)

    def _on_search_finished(self, results: list) -> None:
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
        self._set_running_state(False)
        self.statusMessage.emit(f"Search failed: {message}")

    def _populate_results(self, results: List[dict]) -> None:
        self._results = list(results or [])
        self._results_list.clear()
        if not self._results:
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
            if frame_idx is not None and sim_str is not None:
                label = f"Frame {int(frame_idx):09} • sim={sim_str}"
            elif frame_idx is not None:
                label = f"Frame {int(frame_idx):09}"
            else:
                label = caption if caption else uri
            widget_item = QtWidgets.QListWidgetItem(label)
            widget_item.setData(QtCore.Qt.UserRole, uri)
            if frame_idx is not None:
                widget_item.setData(QtCore.Qt.UserRole + 1, int(frame_idx))
            self._results_list.addItem(widget_item)
        self.statusMessage.emit(f"Found {len(self._results)} results.")

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

    # type: ignore[override]
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
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


def _resolve_uri_to_path(uri: str) -> Optional[Path]:
    if uri.startswith("file://"):
        uri = uri.replace("file://", "", 1)
    try:
        path = Path(uri)
    except Exception:
        return None
    return path if path.exists() else None
