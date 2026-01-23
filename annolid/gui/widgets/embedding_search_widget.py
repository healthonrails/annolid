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

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._current_image_path: Optional[Path] = None
        self._results: List[dict] = []
        self._all_files: List[Path] = []

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
        self._image_button = QtWidgets.QPushButton("Find Similar to Frame")
        self._image_button.clicked.connect(self._run_image_search)
        image_layout.addWidget(self._image_button)
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

    def set_query_image(self, path: Optional[Path]) -> None:
        self._current_image_path = path
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

    def _populate_results(self, results: List[dict]) -> None:
        self._results = list(results or [])
        self._results_list.clear()
        if not self._results:
            self.statusMessage.emit("No results found.")
            return
        for item in self._results:
            uri = str(item.get("image_uri") or "")
            caption = str(item.get("caption") or "")
            label = caption if caption else uri
            widget_item = QtWidgets.QListWidgetItem(label)
            widget_item.setData(QtCore.Qt.UserRole, uri)
            frame_idx = AnnotationStore.frame_number_from_path(uri)
            if frame_idx is not None:
                widget_item.setData(QtCore.Qt.UserRole + 1, int(frame_idx))
            self._results_list.addItem(widget_item)
        self.statusMessage.emit(f"Found {len(self._results)} results.")

    def _handle_result_double_click(self, item: QtWidgets.QListWidgetItem) -> None:
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
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_image_label()

    def _update_image_label(self) -> None:
        if self._current_image_path:
            frame_idx = AnnotationStore.frame_number_from_path(self._current_image_path)
            if frame_idx is not None:
                display = f"Frame {frame_idx:09}"
            else:
                display = self._current_image_path.name
            self._image_label.setText(display)
            self._image_label.setToolTip(str(self._current_image_path))
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
