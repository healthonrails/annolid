"""Interactive Label Queue dock widget for zero-extraction in-video frame labeling.

Allows users to step through selected diverse frames directly inside an open
video (for pose or segmentation), track live labeled progress across both
LabelMe disk JSONs and AnnotationStore, curate the queue, and optionally
export/extract the curated frames to disk at any time.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from annolid.core.media.video import CV2Video, read_capture_frame_bgr
from annolid.data.frame_sampling import discover_labeled_frame_indices
from annolid.gui.qt_compat import configure_compact_combo
from annolid.services.frame_selection import (
    FrameSelectionResult,
    SelectedFrameItem,
    default_queue_manifest_path,
    load_frame_selection_queue,
    materialize_selected_frames,
    save_frame_selection_queue,
)


class _ThumbnailLoaderWorker(QtCore.QThread):
    """Decodes compact RGB thumbnails off the GUI thread for queued frames."""

    thumbnailReady = QtCore.Signal(int, object)  # frame_index, QImage

    def __init__(
        self,
        video_path: Path,
        frame_indices: Sequence[int],
        *,
        thumb_width: int = 64,
        thumb_height: int = 40,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = Path(video_path)
        self._frame_indices = sorted({int(i) for i in frame_indices if int(i) >= 0})
        self._thumb_width = max(32, int(thumb_width))
        self._thumb_height = max(24, int(thumb_height))
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:  # noqa: N802
        if not self._frame_indices or not self._video_path.is_file():
            return
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            return
        try:
            next_expected: Optional[int] = 0
            grab_skip_disabled = False
            for fidx in self._frame_indices:
                if self._stop_requested or self.isInterruptionRequested():
                    break
                ret, frame_bgr, next_expected, grab_skip_disabled = (
                    read_capture_frame_bgr(
                        cap,
                        fidx,
                        expected_next=next_expected,
                        max_forward_grab_skip=CV2Video._MAX_FORWARD_GRAB_SKIP,
                        grab_skip_disabled=grab_skip_disabled,
                    )
                )
                if not ret or frame_bgr is None:
                    continue
                resized = cv2.resize(
                    frame_bgr,
                    (self._thumb_width, self._thumb_height),
                    interpolation=cv2.INTER_AREA,
                )
                rgb = np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                h, w = rgb.shape[:2]
                qimg = QtGui.QImage(
                    rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888
                ).copy()
                self.thumbnailReady.emit(int(fidx), qimg)
        finally:
            cap.release()


class _MaterializeQueueWorker(QtCore.QThread):
    """Exports queued frames to disk off the GUI thread."""

    progress = QtCore.Signal(int, str)
    finishedResult = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        video_path: Path,
        items: Sequence[SelectedFrameItem],
        output_dir: Path,
        *,
        image_format: str = "png",
        copy_existing_labels: bool = True,
        annotation_dir: Optional[Path] = None,
        base_result: Optional[FrameSelectionResult] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = Path(video_path)
        self._items = list(items)
        self._output_dir = Path(output_dir)
        self._image_format = str(image_format or "png")
        self._copy_existing_labels = bool(copy_existing_labels)
        self._annotation_dir = annotation_dir
        self._base_result = base_result

    def run(self) -> None:  # noqa: N802
        try:
            res = materialize_selected_frames(
                self._video_path,
                self._items,
                self._output_dir,
                image_format=self._image_format,
                copy_existing_labels=self._copy_existing_labels,
                annotation_dir=self._annotation_dir,
                base_result=self._base_result,
                progress_callback=lambda pct, msg: self.progress.emit(
                    max(0, min(100, int(round(float(pct))))), str(msg)
                ),
            )
            self.finishedResult.emit(res)
        except Exception as exc:
            self.failed.emit(str(exc))


class FrameSelectionQueueWidget(QtWidgets.QWidget):
    """Interactive queue for labeling diverse video frames without forced extraction."""

    jumpToFrameRequested = QtCore.Signal(int)
    queueMarksChanged = QtCore.Signal(list)
    selectFramesRequested = QtCore.Signal()
    extractedDirectoryReady = QtCore.Signal(str)
    statusMessage = QtCore.Signal(str)

    _MAX_THUMB_CACHE = 256

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._video_path: Optional[Path] = None
        self._annotation_dir: Optional[Path] = None
        self._fps: float = 0.0
        self._current_frame: Optional[int] = None
        self._result: Optional[FrameSelectionResult] = None
        self._items_by_frame: Dict[int, SelectedFrameItem] = {}
        self._labeled_frames: set[int] = set()
        self._thumbnails: "OrderedDict[int, QtGui.QIcon]" = OrderedDict()
        self._thumb_worker: Optional[_ThumbnailLoaderWorker] = None
        self._export_worker: Optional[_MaterializeQueueWorker] = None

        self._build_ui()
        self._setup_shortcuts()
        self._update_header_and_buttons()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        # Summary + progress header
        self.summary_label = QtWidgets.QLabel(
            "No frames queued. Click Select… to pick diverse frames for pose or segmentation."
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-size: 11px; color: #6b7280;")
        layout.addWidget(self.summary_label)

        prog_row = QtWidgets.QHBoxLayout()
        prog_row.setContentsMargins(0, 0, 0, 0)
        prog_row.setSpacing(6)
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("0 / 0 labeled")
        self.progress_bar.setFixedHeight(18)
        prog_row.addWidget(self.progress_bar, 1)
        layout.addLayout(prog_row)

        # Primary navigation bar
        nav_row = QtWidgets.QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(4)
        self.prev_btn = QtWidgets.QPushButton("◀ Prev", self)
        self.prev_btn.setToolTip("Jump to previous queued frame (Alt+Left)")
        self.prev_btn.clicked.connect(self.jump_to_prev)

        self.next_unlabeled_btn = QtWidgets.QPushButton("Next Unlabeled ▶", self)
        self.next_unlabeled_btn.setToolTip(
            "Jump to the next unlabeled queued frame (Alt+Right)"
        )
        self.next_unlabeled_btn.clicked.connect(self.jump_to_next_unlabeled)

        self.next_btn = QtWidgets.QPushButton("Next ▶", self)
        self.next_btn.setToolTip("Jump to next queued frame")
        self.next_btn.clicked.connect(self.jump_to_next)

        for btn in (self.prev_btn, self.next_unlabeled_btn, self.next_btn):
            btn.setMinimumWidth(48)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_unlabeled_btn, 1)
        nav_row.addWidget(self.next_btn)
        layout.addLayout(nav_row)

        # Filter, Sort, Auto-advance bar
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        ctrl_row.setSpacing(4)
        self.filter_combo = QtWidgets.QComboBox(self)
        self.filter_combo.addItem("All", "all")
        self.filter_combo.addItem("Unlabeled", "unlabeled")
        self.filter_combo.addItem("Labeled", "labeled")
        configure_compact_combo(self.filter_combo, min_chars=6, popup_min_width=110)
        self.filter_combo.currentIndexChanged.connect(self._rebuild_list_view)
        ctrl_row.addWidget(self.filter_combo)

        self.sort_combo = QtWidgets.QComboBox(self)
        self.sort_combo.addItem("Timeline", "frame")
        self.sort_combo.addItem("Diversity", "diversity")
        self.sort_combo.addItem("Cluster", "cluster")
        self.sort_combo.addItem("Motion", "motion")
        configure_compact_combo(self.sort_combo, min_chars=6, popup_min_width=115)
        self.sort_combo.currentIndexChanged.connect(self._rebuild_list_view)
        ctrl_row.addWidget(self.sort_combo)

        self.auto_advance_cb = QtWidgets.QCheckBox("Auto-next", self)
        self.auto_advance_cb.setChecked(False)
        self.auto_advance_cb.setToolTip(
            "After saving annotations on a queued frame, automatically jump to the next unlabeled frame."
        )
        ctrl_row.addWidget(self.auto_advance_cb)
        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        # Frame list with thumbnails
        self.frame_list = QtWidgets.QListWidget(self)
        self.frame_list.setIconSize(QtCore.QSize(64, 40))
        self.frame_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.frame_list.setUniformItemSizes(True)
        self.frame_list.itemClicked.connect(self._on_item_activated)
        self.frame_list.itemDoubleClicked.connect(self._on_item_activated)
        layout.addWidget(self.frame_list, 1)

        # Curation & Export toolbar
        action_grid = QtWidgets.QGridLayout()
        action_grid.setContentsMargins(0, 0, 0, 0)
        action_grid.setSpacing(4)

        self.select_btn = QtWidgets.QPushButton("Select…", self)
        self.select_btn.setToolTip(
            "Select diverse frames using visual or embedding features"
        )
        self.select_btn.clicked.connect(self.selectFramesRequested.emit)

        self.add_current_btn = QtWidgets.QPushButton("+ Current", self)
        self.add_current_btn.setToolTip(
            "Add the current video frame to this label queue"
        )
        self.add_current_btn.clicked.connect(self._add_current_frame)

        self.remove_btn = QtWidgets.QPushButton("− Remove", self)
        self.remove_btn.setToolTip("Remove selected frame(s) from the queue")
        self.remove_btn.clicked.connect(self.remove_selected_frames)

        self.extract_btn = QtWidgets.QPushButton("Extract…", self)
        self.extract_btn.setToolTip(
            "Export queued frames as PNG/JPG images (+ existing annotations) to a folder"
        )
        self.extract_btn.clicked.connect(self._extract_queue_to_folder)

        self.clear_btn = QtWidgets.QPushButton("Clear", self)
        self.clear_btn.setToolTip("Clear all frames from this queue")
        self.clear_btn.clicked.connect(lambda: self.clear_queue(remove_manifest=True))

        for btn in (
            self.select_btn,
            self.add_current_btn,
            self.remove_btn,
            self.extract_btn,
            self.clear_btn,
        ):
            btn.setMinimumWidth(48)

        action_grid.addWidget(self.select_btn, 0, 0)
        action_grid.addWidget(self.add_current_btn, 0, 1)
        action_grid.addWidget(self.remove_btn, 0, 2)
        action_grid.addWidget(self.extract_btn, 1, 0, 1, 2)
        action_grid.addWidget(self.clear_btn, 1, 2)
        layout.addLayout(action_grid)

    def _setup_shortcuts(self) -> None:
        # Scoped to this dock widget so they never hijack main-window shortcuts.
        next_sc = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.ALT | QtCore.Qt.Key_Right), self
        )
        next_sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        next_sc.activated.connect(self.jump_to_next_unlabeled)

        prev_sc = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.ALT | QtCore.Qt.Key_Left), self
        )
        prev_sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        prev_sc.activated.connect(self.jump_to_prev)

        self._shortcuts = [next_sc, prev_sc]

    # ------------------------------------------------------------------
    # Public API & Synchronization
    # ------------------------------------------------------------------
    def queued_frame_indices(self) -> List[int]:
        return sorted(self._items_by_frame.keys())

    def unlabeled_frame_indices(self) -> List[int]:
        return [
            idx
            for idx in sorted(self._items_by_frame.keys())
            if idx not in self._labeled_frames
        ]

    def labeled_count(self) -> int:
        return sum(1 for idx in self._items_by_frame if idx in self._labeled_frames)

    def total_count(self) -> int:
        return len(self._items_by_frame)

    def has_queue(self) -> bool:
        return bool(self._items_by_frame)

    def should_auto_advance_on_save(self) -> bool:
        return bool(self.auto_advance_cb.isChecked())

    def tooltip_for_frame(self, frame_index: int) -> str:
        item = self._items_by_frame.get(int(frame_index))
        if item is None:
            return ""
        status = "Labeled" if int(frame_index) in self._labeled_frames else "Queued"
        task_tag = self._result.task if self._result is not None else "labeling"
        reason = f" · {item.reason}" if item.reason else ""
        return f"Frame {int(frame_index)} ({status} for {task_tag}{reason})"

    def set_video_context(
        self,
        video_path: Optional[str | Path],
        *,
        annotation_dir: Optional[str | Path] = None,
        fps: Optional[float] = None,
        autoload_queue: bool = True,
    ) -> bool:
        """Bind widget to an active video and optionally restore its saved queue."""
        self._stop_thumbnail_worker()
        self._thumbnails.clear()
        self._video_path = (
            Path(video_path).expanduser().resolve() if video_path else None
        )
        self._annotation_dir = (
            Path(annotation_dir).expanduser().resolve()
            if annotation_dir
            else (self._video_path.with_suffix("") if self._video_path else None)
        )
        self._fps = float(fps) if fps and float(fps) > 0 else 0.0
        self._current_frame = None
        self._result = None
        self._items_by_frame.clear()
        self._labeled_frames.clear()

        restored = False
        if autoload_queue and self._video_path is not None:
            qpath = default_queue_manifest_path(self._video_path, self._annotation_dir)
            if qpath.is_file():
                try:
                    loaded = load_frame_selection_queue(qpath)
                    if loaded.selected_items:
                        self.set_selection_result(loaded, persist=False)
                        restored = True
                except Exception:
                    restored = False

        if not restored:
            self._rebuild_list_view()
            self._update_header_and_buttons()
            self.queueMarksChanged.emit([])
        return restored

    def set_selection_result(
        self,
        result: FrameSelectionResult,
        *,
        persist: bool = True,
        load_thumbnails: bool = True,
    ) -> None:
        self._stop_thumbnail_worker()
        if result.video_path:
            new_vpath = Path(result.video_path).expanduser().resolve()
            # Keep the currently bound live video path if the persisted queue manifest
            # references an old/relocated path that no longer exists.
            if self._video_path is None or (
                new_vpath.is_file() and self._video_path != new_vpath
            ):
                if self._video_path != new_vpath:
                    self._thumbnails.clear()
                self._video_path = new_vpath
        if result.fps and result.fps > 0:
            self._fps = float(result.fps)
        if self._annotation_dir is None and self._video_path is not None:
            self._annotation_dir = self._video_path.with_suffix("")

        self._result = result
        self._items_by_frame = {
            int(item.frame_index): item for item in result.selected_items
        }
        self.refresh_labeled_status()
        if persist:
            self._persist_queue()
        self._rebuild_list_view()
        self._update_header_and_buttons()
        self.queueMarksChanged.emit(self.queued_frame_indices())
        if load_thumbnails:
            self._start_thumbnail_worker()

    def refresh_labeled_status(self) -> None:
        """Re-scan dual annotation storage to refresh labeled/pending badges."""
        if self._annotation_dir is None:
            self._labeled_frames = {
                idx
                for idx, item in self._items_by_frame.items()
                if item.already_labeled
            }
            return
        discovered = discover_labeled_frame_indices(
            self._annotation_dir, require_shapes=True
        )
        self._labeled_frames = {
            idx for idx in self._items_by_frame if idx in discovered
        }

    def mark_frame_labeled(self, frame_index: int, *, labeled: bool = True) -> bool:
        """Update queue state when a frame is saved. Returns True if auto-advance should run."""
        idx = int(frame_index)
        if idx not in self._items_by_frame:
            return False
        if labeled:
            self._labeled_frames.add(idx)
        else:
            self._labeled_frames.discard(idx)
        self._rebuild_list_view()
        self._update_header_and_buttons()
        if labeled and not self.unlabeled_frame_indices():
            self.statusMessage.emit("All queued frames are labeled!")
            return False
        return bool(labeled and self.auto_advance_cb.isChecked())

    def set_current_frame(self, frame_index: Optional[int]) -> None:
        self._current_frame = int(frame_index) if frame_index is not None else None
        if self._current_frame is None:
            return
        for row in range(self.frame_list.count()):
            item = self.frame_list.item(row)
            if (
                item is not None
                and item.data(QtCore.Qt.UserRole) == self._current_frame
            ):
                self.frame_list.blockSignals(True)
                self.frame_list.setCurrentRow(row)
                self.frame_list.blockSignals(False)
                self.frame_list.scrollToItem(item)
                break

    def add_frame(self, frame_index: int, *, reason: str = "Added manually") -> bool:
        idx = int(frame_index)
        if idx < 0 or idx in self._items_by_frame:
            return False
        ts = (float(idx) / self._fps) if self._fps > 0.0 else None
        self._items_by_frame[idx] = SelectedFrameItem(
            frame_index=idx,
            timestamp_sec=ts,
            selection_role="manual",
            already_labeled=(idx in self._labeled_frames),
            reason=reason,
        )
        self.refresh_labeled_status()
        self._persist_queue()
        self._rebuild_list_view()
        self._update_header_and_buttons()
        self.queueMarksChanged.emit(self.queued_frame_indices())
        self._start_thumbnail_worker(only_indices=[idx])
        return True

    def remove_selected_frames(self) -> int:
        selected = self.frame_list.selectedItems()
        if not selected:
            return 0
        removed = 0
        for widget_item in selected:
            fidx = widget_item.data(QtCore.Qt.UserRole)
            if fidx is not None and int(fidx) in self._items_by_frame:
                self._items_by_frame.pop(int(fidx), None)
                self._labeled_frames.discard(int(fidx))
                removed += 1
        if removed:
            self._persist_queue()
            self._rebuild_list_view()
            self._update_header_and_buttons()
            self.queueMarksChanged.emit(self.queued_frame_indices())
            self.statusMessage.emit(f"Removed {removed} frame(s) from Label Queue.")
        return removed

    def clear_queue(self, *, remove_manifest: bool = False) -> None:
        self._stop_thumbnail_worker()
        self._result = None
        self._items_by_frame.clear()
        self._labeled_frames.clear()
        if remove_manifest and self._video_path is not None:
            qpath = default_queue_manifest_path(self._video_path, self._annotation_dir)
            try:
                if qpath.is_file():
                    qpath.unlink()
            except OSError:
                pass
        self._rebuild_list_view()
        self._update_header_and_buttons()
        self.queueMarksChanged.emit([])

    def jump_to_next_unlabeled(self) -> Optional[int]:
        unlabeled = self.unlabeled_frame_indices()
        if not unlabeled:
            if self._items_by_frame:
                self.statusMessage.emit("All queued frames are labeled!")
            return None
        cur = self._current_frame if self._current_frame is not None else -1
        target = next((idx for idx in unlabeled if idx > cur), unlabeled[0])
        self._emit_jump(target)
        return target

    def jump_to_next(self) -> Optional[int]:
        ordered = self._visible_frame_order()
        if not ordered:
            return None
        if self._current_frame in ordered:
            pos = ordered.index(self._current_frame)
            target = ordered[(pos + 1) % len(ordered)]
        else:
            cur = self._current_frame if self._current_frame is not None else -1
            target = next((idx for idx in ordered if idx > cur), ordered[0])
        self._emit_jump(target)
        return target

    def jump_to_prev(self) -> Optional[int]:
        ordered = self._visible_frame_order()
        if not ordered:
            return None
        if self._current_frame in ordered:
            pos = ordered.index(self._current_frame)
            target = ordered[(pos - 1) % len(ordered)]
        else:
            cur = self._current_frame if self._current_frame is not None else 10**12
            prevs = [idx for idx in ordered if idx < cur]
            target = prevs[-1] if prevs else ordered[-1]
        self._emit_jump(target)
        return target

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _emit_jump(self, frame_index: int) -> None:
        self.set_current_frame(frame_index)
        self.jumpToFrameRequested.emit(int(frame_index))

    def _add_current_frame(self) -> None:
        if self._current_frame is None:
            self.statusMessage.emit("Open a video to add the current frame.")
            return
        if self.add_frame(self._current_frame):
            self.statusMessage.emit(
                f"Added frame {self._current_frame} to Label Queue."
            )
        else:
            self.statusMessage.emit(
                f"Frame {self._current_frame} is already in Label Queue."
            )

    def _visible_frame_order(self) -> List[int]:
        filt = str(self.filter_combo.currentData() or "all")
        sort_mode = str(self.sort_combo.currentData() or "frame")
        items = list(self._items_by_frame.values())
        if filt == "unlabeled":
            items = [it for it in items if it.frame_index not in self._labeled_frames]
        elif filt == "labeled":
            items = [it for it in items if it.frame_index in self._labeled_frames]

        if sort_mode == "diversity":
            items.sort(
                key=lambda it: (
                    -max(it.diversity_score, it.novelty_score),
                    it.frame_index,
                )
            )
        elif sort_mode == "cluster":
            items.sort(
                key=lambda it: (
                    it.cluster_id if it.cluster_id is not None else 10**6,
                    it.frame_index,
                )
            )
        elif sort_mode == "motion":
            items.sort(key=lambda it: (-it.motion_score, it.frame_index))
        else:
            items.sort(key=lambda it: it.frame_index)
        return [it.frame_index for it in items]

    def _rebuild_list_view(self) -> None:
        self.frame_list.clear()
        ordered = self._visible_frame_order()
        for fidx in ordered:
            item = self._items_by_frame[fidx]
            is_labeled = fidx in self._labeled_frames
            badge = "✓" if is_labeled else "○"
            ts_str = (
                f" · {_format_seconds_short(item.timestamp_sec)}"
                if item.timestamp_sec is not None
                else ""
            )
            subtitle = item.reason or item.selection_role
            text = f"{badge} Frame {fidx:06d}{ts_str}\n   {subtitle}"
            witem = QtWidgets.QListWidgetItem(text)
            witem.setData(QtCore.Qt.UserRole, int(fidx))
            witem.setToolTip(self.tooltip_for_frame(fidx))
            if is_labeled:
                witem.setForeground(QtGui.QBrush(QtGui.QColor(34, 139, 34)))
            icon = self._thumbnails.get(fidx)
            if icon is not None:
                witem.setIcon(icon)
            self.frame_list.addItem(witem)
        if self._current_frame is not None:
            self.set_current_frame(self._current_frame)

    def _update_header_and_buttons(self) -> None:
        total = self.total_count()
        labeled = self.labeled_count()
        has_items = total > 0
        has_video = self._video_path is not None

        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(labeled if has_items else 0)
        pct = int(round((100.0 * labeled) / float(total))) if total > 0 else 0
        self.progress_bar.setFormat(f"{labeled} / {total} labeled ({pct}%)")

        if has_items and self._result is not None:
            cache_tag = " · cached" if self._result.embedding_cache_hit else ""
            prompt_tag = (
                f" · '{self._result.query_text}'" if self._result.query_text else ""
            )
            self.summary_label.setText(
                f"{self._result.task.title()} · {self._result.algo} · "
                f"{self._result.feature_backend}{cache_tag}{prompt_tag}"
            )
        elif has_items:
            self.summary_label.setText(f"Custom queue ({total} frames)")
        else:
            self.summary_label.setText(
                "No frames queued. Click Select… to pick diverse frames for pose or segmentation."
            )

        self.prev_btn.setEnabled(has_items)
        self.next_btn.setEnabled(has_items)
        self.next_unlabeled_btn.setEnabled(has_items and labeled < total)
        self.add_current_btn.setEnabled(has_video)
        self.remove_btn.setEnabled(has_items)
        self.extract_btn.setEnabled(has_items and has_video)
        self.clear_btn.setEnabled(has_items)

    def _on_item_activated(self, widget_item: QtWidgets.QListWidgetItem) -> None:
        fidx = widget_item.data(QtCore.Qt.UserRole)
        if fidx is not None:
            self._emit_jump(int(fidx))

    def _persist_queue(self) -> None:
        if self._video_path is None:
            return
        qpath = default_queue_manifest_path(self._video_path, self._annotation_dir)
        ordered_items = tuple(
            self._items_by_frame[idx] for idx in sorted(self._items_by_frame.keys())
        )
        base = self._result
        updated = FrameSelectionResult(
            video_path=str(self._video_path),
            task=base.task if base is not None else "general",
            algo=base.algo if base is not None else "custom",
            feature_backend=base.feature_backend if base is not None else "none",
            extract_images=False,
            output_dir=None,
            total_video_frames=base.total_video_frames if base is not None else 0,
            fps=self._fps or (base.fps if base is not None else 0.0),
            start_frame=ordered_items[0].frame_index if ordered_items else 0,
            end_frame=ordered_items[-1].frame_index if ordered_items else 0,
            candidate_count=(
                base.candidate_count if base is not None else len(ordered_items)
            ),
            excluded_labeled_count=(
                base.excluded_labeled_count if base is not None else 0
            ),
            filtered_blur_count=(base.filtered_blur_count if base is not None else 0),
            selected_items=ordered_items,
            manifest_path=str(qpath),
            embedding_cache_path=(
                base.embedding_cache_path if base is not None else None
            ),
            embedding_cache_hit=(
                base.embedding_cache_hit if base is not None else False
            ),
            query_text=base.query_text if base is not None else None,
        )
        self._result = updated
        try:
            save_frame_selection_queue(updated, qpath)
        except OSError:
            pass

    def _start_thumbnail_worker(
        self, *, only_indices: Optional[Sequence[int]] = None
    ) -> None:
        if self._video_path is None or not self._video_path.is_file():
            return
        targets = (
            [int(i) for i in only_indices if int(i) not in self._thumbnails]
            if only_indices is not None
            else [
                idx
                for idx in sorted(self._items_by_frame.keys())
                if idx not in self._thumbnails
            ]
        )
        if not targets:
            return
        self._stop_thumbnail_worker()
        worker = _ThumbnailLoaderWorker(self._video_path, targets, parent=self)
        worker.thumbnailReady.connect(self._on_thumbnail_ready)
        worker.finished.connect(self._on_thumbnail_worker_finished)
        self._thumb_worker = worker
        worker.start()

    def _stop_thumbnail_worker(self) -> None:
        worker = self._thumb_worker
        self._thumb_worker = None
        if worker is not None:
            try:
                worker.stop()
                worker.requestInterruption()
                if worker.isRunning():
                    worker.wait(1500)
            except Exception:
                pass

    def _on_thumbnail_ready(self, frame_index: int, qimage: object) -> None:
        if not isinstance(qimage, QtGui.QImage) or qimage.isNull():
            return
        icon = QtGui.QIcon(QtGui.QPixmap.fromImage(qimage))
        self._thumbnails[int(frame_index)] = icon
        self._thumbnails.move_to_end(int(frame_index))
        while len(self._thumbnails) > self._MAX_THUMB_CACHE:
            self._thumbnails.popitem(last=False)
        for row in range(self.frame_list.count()):
            item = self.frame_list.item(row)
            if item is not None and item.data(QtCore.Qt.UserRole) == int(frame_index):
                item.setIcon(icon)
                break

    def _on_thumbnail_worker_finished(self) -> None:
        worker = self.sender()
        if worker is self._thumb_worker:
            self._thumb_worker = None
        if worker is not None:
            worker.deleteLater()

    def _extract_queue_to_folder(self) -> None:
        if not self._items_by_frame or self._video_path is None:
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            self.statusMessage.emit("Queue extraction is already running…")
            return
        default_dir = str(
            self._video_path.parent
            / f"{self._video_path.stem.replace(' ', '_')}_selected"
        )
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Queued Frames", str(self._video_path.parent)
        )
        if not chosen:
            return
        target_dir = Path(chosen)
        if target_dir.is_dir() and any(target_dir.iterdir()):
            target_dir = target_dir / Path(default_dir).name

        ordered_items = [
            self._items_by_frame[idx] for idx in sorted(self._items_by_frame.keys())
        ]
        self.extract_btn.setEnabled(False)
        self.statusMessage.emit(f"Extracting {len(ordered_items)} queued frames…")
        worker = _MaterializeQueueWorker(
            self._video_path,
            ordered_items,
            target_dir,
            image_format="png",
            copy_existing_labels=True,
            annotation_dir=self._annotation_dir,
            base_result=self._result,
            parent=self,
        )
        worker.progress.connect(
            lambda pct, msg: self.statusMessage.emit(f"{pct}% · {msg}")
        )
        worker.finishedResult.connect(self._on_extract_finished)
        worker.failed.connect(self._on_extract_failed)
        worker.finished.connect(self._on_extract_cleanup)
        self._export_worker = worker
        worker.start()

    def _on_extract_finished(self, result: FrameSelectionResult) -> None:
        out_dir = str(result.output_dir or "")
        self.statusMessage.emit(
            f"Extracted {len(result.saved_paths)} queued frames to {out_dir}"
        )
        reply = QtWidgets.QMessageBox.question(
            self,
            "Extraction Complete",
            f"Saved {len(result.saved_paths)} frame(s) to:\n{out_dir}\n\n"
            "Open the extracted folder in the Files panel?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes and out_dir:
            self.extractedDirectoryReady.emit(out_dir)

    def _on_extract_failed(self, error_msg: str) -> None:
        self.statusMessage.emit(f"Queue extraction failed: {error_msg}")
        QtWidgets.QMessageBox.warning(self, "Extraction Failed", error_msg)

    def _on_extract_cleanup(self) -> None:
        self._update_header_and_buttons()
        if self._export_worker is not None:
            self._export_worker.deleteLater()
            self._export_worker = None

    def shutdown_workers(self) -> None:
        self._stop_thumbnail_worker()
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.wait(2000)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.shutdown_workers()
        super().closeEvent(event)


def _format_seconds_short(seconds: float) -> str:
    total_sec = max(0.0, float(seconds))
    mins = int(total_sec // 60)
    rem = total_sec - mins * 60
    if mins >= 60:
        hours = mins // 60
        mins = mins % 60
        return f"{hours}:{mins:02d}:{rem:04.1f}"
    return f"{mins:02d}:{rem:04.1f}"


__all__ = ["FrameSelectionQueueWidget"]
