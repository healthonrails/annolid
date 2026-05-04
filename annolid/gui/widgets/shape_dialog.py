from pathlib import Path
import json
import math
import os.path as osp
from qtpy import QtWidgets, QtCore
from annolid.utils.logger import logger
from annolid.gui.label_file import LabelFile
from annolid.annotation.polygons import are_polygons_close_or_overlap
from annolid.utils.shapes import shape_to_dict
from annolid.utils.annotation_store import AnnotationStore, AnnotationStoreError
from annolid.gui.workers import FlexibleWorker


class ShapePropagationDialog(QtWidgets.QDialog):
    """
    A dialog that allows the user to select a shape from the current canvas and specify
    a target frame. Depending on the selected action, the dialog will either propagate
    (copy) the shape into the JSON annotation files for all frames from the current one
    to the target, delete the same shape from those frames, or define a proximity-based
    event rule that automatically flags frames where a spatial relationship between annotated
    shapes is met.
    """

    def __init__(self, canvas, main_window, current_frame, max_frame, parent=None):
        super().__init__(parent)
        self.canvas = canvas  # Reference to the Canvas widget
        self.main_window = main_window  # Reference to the main window
        self.current_frame = current_frame
        self.max_frame = max_frame
        self._annotation_store_batch = None
        self._annotation_json_batch = None
        self._shape_action_thread = None
        self._shape_action_worker = None
        self._shape_action_progress = None
        self._shape_action_cancel_button = None
        self._pending_shape_action = None
        self._annotation_target_cache = None
        self._shape_action_cancel_requested = False
        self.background_action_started = False

        self.setWindowTitle("Shape Action on Future Frames")

        # Create a list widget to display available shapes.
        self.shape_list = QtWidgets.QListWidget(self)
        for shape in self.canvas.shapes:
            if str(getattr(shape, "shape_type", "") or "").strip().lower() != "point":
                item = QtWidgets.QListWidgetItem(shape.label)
                item.setData(QtCore.Qt.UserRole, shape)
                self.shape_list.addItem(item)

        # Drop-down for selecting the action.
        # Now includes "Define Proximity Event" as an additional action.
        self.action_combo = QtWidgets.QComboBox(self)
        self.action_combo.addItems(
            [
                "Propagate",
                "Rename & Propagate",
                "Switch Labels",
                "Delete",
                "Define Proximity Event",
            ]
        )
        self.action_combo.currentIndexChanged.connect(self.update_action_fields)

        # Spin box for selecting the target frame (used in propagate and delete actions).
        self.frame_spin_label = QtWidgets.QLabel("Apply action until frame:")
        self.frame_spin = QtWidgets.QSpinBox(self)
        self.frame_spin.setMinimum(current_frame + 1)

        # Determine the default stop frame from the next manual seed frame.
        default_future_frame = self._resolve_default_action_end_frame(max_frame)

        if default_future_frame > max_frame:
            default_future_frame = max_frame

        self.frame_spin.setMaximum(max_frame)
        self.frame_spin.setValue(default_future_frame)
        self.frame_spin.setToolTip(f"Maximum frame: {max_frame}")
        self.frame_spin.lineEdit().setPlaceholderText(str(default_future_frame))

        self.rename_widget = QtWidgets.QWidget(self)
        rename_layout = QtWidgets.QGridLayout(self.rename_widget)
        rename_layout.setContentsMargins(0, 0, 0, 0)
        self.rename_label = QtWidgets.QLabel("Rename to:", self.rename_widget)
        self.rename_line = QtWidgets.QLineEdit(self.rename_widget)
        self.rename_line.setPlaceholderText("New label")
        self.label_switch_combo = QtWidgets.QComboBox(self.rename_widget)
        self.label_switch_combo.setEditable(False)
        self.label_switch_combo.addItem("Select shape label", "")
        for label in self._available_shape_labels():
            self.label_switch_combo.addItem(label, label)
        self.label_switch_combo.currentIndexChanged.connect(
            self._switch_rename_label_from_combo
        )
        rename_layout.addWidget(self.rename_label, 0, 0)
        rename_layout.addWidget(self.rename_line, 0, 1)
        rename_layout.addWidget(
            QtWidgets.QLabel("Switch to shape label:", self.rename_widget), 1, 0
        )
        rename_layout.addWidget(self.label_switch_combo, 1, 1)
        self.rename_widget.hide()

        # --- New UI Elements for "Define Proximity Event" ---
        self.event_widget = QtWidgets.QWidget(self)
        event_layout = QtWidgets.QVBoxLayout(self.event_widget)

        self.target_group_combo = QtWidgets.QComboBox(self)
        shape_labels = sorted(
            {
                shape.label
                for shape in self.canvas.shapes
                if str(getattr(shape, "shape_type", "") or "").strip().lower()
                != "point"
            }
        )
        for label in shape_labels:
            self.target_group_combo.addItem(label)
        self.target_group_combo.addItem("All Others")
        event_layout.addWidget(QtWidgets.QLabel("Select Target Group:"))
        event_layout.addWidget(self.target_group_combo)

        self.event_name_line = QtWidgets.QLineEdit(self)
        event_layout.addWidget(QtWidgets.QLabel("Event Name:"))
        event_layout.addWidget(self.event_name_line)

        self.proximity_threshold_spin = QtWidgets.QSpinBox(self)
        self.proximity_threshold_spin.setMinimum(1)
        self.proximity_threshold_spin.setMaximum(10000)
        self.proximity_threshold_spin.setValue(50)
        event_layout.addWidget(QtWidgets.QLabel("Proximity Threshold (pixels):"))
        event_layout.addWidget(self.proximity_threshold_spin)

        self.rule_type_combo = QtWidgets.QComboBox(self)
        self.rule_type_combo.addItems(["any", "all"])
        event_layout.addWidget(QtWidgets.QLabel("Rule Type:"))
        event_layout.addWidget(self.rule_type_combo)

        self.event_start_frame_spin = QtWidgets.QSpinBox(self)
        self.event_start_frame_spin.setMinimum(current_frame + 1)
        self.event_start_frame_spin.setMaximum(max_frame)
        self.event_start_frame_spin.setValue(current_frame + 1)
        self.event_end_frame_spin = QtWidgets.QSpinBox(self)
        self.event_end_frame_spin.setMinimum(current_frame + 1)
        self.event_end_frame_spin.setMaximum(max_frame)
        self.event_end_frame_spin.setValue(default_future_frame)
        # add tooltips and placeholder text
        self.event_end_frame_spin.setToolTip(f"Maximum frame: {max_frame}")
        self.event_end_frame_spin.lineEdit().setPlaceholderText(
            str(default_future_frame)
        )

        event_layout.addWidget(QtWidgets.QLabel("Event Start Frame:"))
        event_layout.addWidget(self.event_start_frame_spin)
        event_layout.addWidget(QtWidgets.QLabel("Event End Frame:"))
        event_layout.addWidget(self.event_end_frame_spin)

        self.event_widget.hide()

        self.apply_btn = QtWidgets.QPushButton("Apply", self)
        self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self.apply_btn.clicked.connect(self.do_action)
        self.cancel_btn.clicked.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Select a shape:"))
        layout.addWidget(self.shape_list)
        layout.addWidget(QtWidgets.QLabel("Select action:"))
        layout.addWidget(self.action_combo)
        layout.addWidget(self.rename_widget)
        layout.addWidget(self.frame_spin_label)
        layout.addWidget(self.frame_spin)
        layout.addWidget(self.event_widget)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def update_action_fields(self):
        """Toggle the visibility of input fields based on the selected action."""
        current_action = self.action_combo.currentText().lower()
        rename_action = current_action in {"rename & propagate", "switch labels"}
        if current_action == "define proximity event":
            self.frame_spin_label.hide()
            self.frame_spin.hide()
            self.rename_widget.hide()
            self.event_widget.show()
        else:
            self.frame_spin_label.show()
            self.frame_spin.show()
            self.rename_widget.setVisible(rename_action)
            self.event_widget.hide()

    @staticmethod
    def _shape_label(shape):
        if isinstance(shape, dict):
            return str(shape.get("label", "") or "").strip()
        return str(getattr(shape, "label", "") or "").strip()

    @staticmethod
    def _append_unique_label(labels, value) -> None:
        label = str(value or "").strip()
        if label and label not in labels:
            labels.append(label)

    def _available_shape_labels(self) -> list[str]:
        labels: list[str] = []
        selected_shape = None
        try:
            selected_shape = (
                self.canvas.selectedShapes[0]
                if getattr(self.canvas, "selectedShapes", None)
                else None
            )
        except Exception:
            selected_shape = None

        for shape in getattr(self.canvas, "shapes", []) or []:
            if shape is selected_shape:
                continue
            self._append_unique_label(labels, self._shape_label(shape))

        uniq_label_list = getattr(self.main_window, "uniqLabelList", None)
        if uniq_label_list is not None:
            try:
                for idx in range(uniq_label_list.count()):
                    item = uniq_label_list.item(idx)
                    if item is None:
                        continue
                    label = item.data(QtCore.Qt.UserRole)
                    if label is None:
                        label = item.text()
                    self._append_unique_label(labels, label)
            except Exception:
                logger.debug(
                    "Failed to collect labels from the unique label list.",
                    exc_info=True,
                )

        label_dialog = getattr(self.main_window, "labelDialog", None)
        for value in getattr(label_dialog, "_history", []) or []:
            self._append_unique_label(labels, value)
        config = getattr(label_dialog, "_config", {}) or {}
        for value in config.get("labels") or []:
            self._append_unique_label(labels, value)

        return labels

    def _switch_rename_label_from_combo(self, _index) -> None:
        label = self.label_switch_combo.currentData()
        if label:
            self.rename_line.setText(str(label))

    def _find_selected_label_switch_target(self, selected_shape, new_label: str):
        selected_shapes = list(getattr(self.canvas, "selectedShapes", []) or [])
        if selected_shape not in selected_shapes or len(selected_shapes) < 2:
            return None
        selected_label = self._shape_label(selected_shape)
        if not new_label or new_label == selected_label:
            return None
        for shape in selected_shapes:
            if shape is selected_shape:
                continue
            if self._shape_label(shape) == new_label:
                return shape
        return None

    @staticmethod
    def _shape_type(shape):
        if isinstance(shape, dict):
            return str(shape.get("shape_type", "") or "").strip().lower()
        return str(getattr(shape, "shape_type", "") or "").strip().lower()

    @staticmethod
    def _shape_group_id(shape):
        if isinstance(shape, dict):
            group_id = shape.get("group_id", None)
        else:
            group_id = getattr(shape, "group_id", None)
        if group_id in (None, ""):
            return None
        return group_id

    def _shape_centroid(self, shape):
        try:
            return self.compute_centroid(shape)
        except Exception:
            return (0.0, 0.0)

    @staticmethod
    def _distance(a, b):
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _resolve_default_action_end_frame(self, max_frame: int) -> int:
        folder = None
        main_window = self.main_window
        if getattr(main_window, "video_results_folder", None):
            folder = Path(main_window.video_results_folder)
        elif getattr(main_window, "annotation_dir", None):
            folder = Path(main_window.annotation_dir)
        if folder is None:
            return int(max_frame)

        seed_frames: set[int] = set()
        discover_fn = getattr(main_window, "_discover_manual_seed_frames", None)
        try:
            if callable(discover_fn):
                seed_frames = {int(v) for v in discover_fn(folder) or set()}
            else:
                stem_prefix = f"{folder.name}_".lower()
                for png_path in folder.glob("*.png"):
                    stem = png_path.stem.lower()
                    if not stem.startswith(stem_prefix):
                        continue
                    suffix = stem[len(stem_prefix) :]
                    if len(suffix) != 9 or not suffix.isdigit():
                        continue
                    if png_path.with_suffix(".json").exists():
                        seed_frames.add(int(suffix))
        except Exception:
            logger.debug(
                "Failed to resolve default action end frame from %s",
                folder,
                exc_info=True,
            )
            return int(max_frame)

        future_seeds = [frame for frame in seed_frames if frame > self.current_frame]
        if not future_seeds:
            return int(max_frame)
        return max(self.current_frame + 1, min(future_seeds) - 1)

    def _allocate_group_id(self):
        allocated = self._allocate_group_ids(1)
        return allocated[0] if allocated else 0

    def _allocate_group_ids(self, count: int) -> list[int]:
        candidates = []
        for shape in getattr(self.canvas, "shapes", []) or []:
            group_id = self._shape_group_id(shape)
            if group_id is None:
                continue
            try:
                candidates.append(int(group_id))
            except Exception:
                continue
        if not candidates:
            start = 0
        else:
            start = max(candidates) + 1
        return list(range(start, start + max(0, int(count))))

    def _find_label_list_item_for_shape(self, shape):
        label_list = getattr(self.main_window, "labelList", None)
        if label_list is None:
            return None
        for idx in range(label_list.count()):
            item = label_list.item(idx)
            if item is None:
                continue
            try:
                if item.shape() is shape:
                    return item
            except Exception:
                continue
        return None

    def _refresh_current_shape_views(self, shape):
        main_window = self.main_window
        try:
            if hasattr(main_window, "_refresh_label_list_items_for_shapes"):
                main_window._refresh_label_list_items_for_shapes([shape])
            elif hasattr(main_window, "_set_label_list_item_text"):
                item = self._find_label_list_item_for_shape(shape)
                if item is not None:
                    rgb = None
                    if hasattr(main_window, "_update_shape_color"):
                        rgb = main_window._update_shape_color(shape)
                    base_text = (
                        self._shape_label(shape)
                        if self._shape_group_id(shape) is None
                        else f"{self._shape_label(shape)} ({self._shape_group_id(shape)})"
                    )
                    main_window._set_label_list_item_text(
                        item,
                        base_text=base_text,
                        marker="●",
                        rgb=rgb,
                    )
        except Exception:
            logger.debug(
                "Failed to refresh the current shape list item.", exc_info=True
            )
        try:
            if hasattr(main_window, "_rebuild_unique_label_list"):
                main_window._rebuild_unique_label_list()
        except Exception:
            logger.debug("Failed to rebuild the unique label list.", exc_info=True)
        try:
            if hasattr(main_window, "canvas") and main_window.canvas is not None:
                main_window.canvas.update()
        except Exception:
            logger.debug("Failed to refresh the canvas after rename.", exc_info=True)

    def _shape_matches_reference(self, candidate, reference):
        if self._shape_type(candidate) != self._shape_type(reference):
            return False

        reference_group_id = self._shape_group_id(reference)
        candidate_group_id = self._shape_group_id(candidate)
        if reference_group_id is not None:
            return candidate_group_id == reference_group_id

        return self._shape_label(candidate) == self._shape_label(reference)

    def _match_shapes(self, shapes, reference):
        matches = [
            shape for shape in shapes if self._shape_matches_reference(shape, reference)
        ]
        if not matches:
            return []
        if self._shape_group_id(reference) is not None or len(matches) == 1:
            return matches
        reference_centroid = self._shape_centroid(reference)
        return [
            min(
                matches,
                key=lambda shape: self._distance(
                    self._shape_centroid(shape), reference_centroid
                ),
            )
        ]

    def _update_shape_record(self, record, new_label, new_group_id):
        if isinstance(record, dict):
            updated = dict(record)
            updated["label"] = new_label
            updated["group_id"] = new_group_id
            return updated

        record.label = new_label
        record.group_id = new_group_id
        return shape_to_dict(record)

    def _save_shape_file(self, label_file, lf):
        self._save_annotation_target(label_file, lf)

    def _resolve_annotation_target(self, label_file):
        path = osp.abspath(label_file)
        if isinstance(self._annotation_target_cache, dict):
            cached = self._annotation_target_cache.get(path)
            if cached is not None:
                return cached
        if path.lower().endswith(".ndjson"):
            result = ("store", AnnotationStore(Path(path)), None)
            if isinstance(self._annotation_target_cache, dict):
                self._annotation_target_cache[path] = result
            return result
        if not path.lower().endswith(".json"):
            result = ("json", None, None)
            if isinstance(self._annotation_target_cache, dict):
                self._annotation_target_cache[path] = result
            return result

        frame = AnnotationStore.frame_number_from_path(path)
        fallback_store = AnnotationStore.for_frame_path(path)
        try:
            raw_text = Path(path).read_text(encoding="utf-8")
            raw = json.loads(raw_text)
        except Exception:
            raw = None

        if isinstance(raw, dict):
            if "annotation_store" in raw and not isinstance(raw.get("shapes"), list):
                store_name = str(raw.get("annotation_store") or "").strip()
                store = AnnotationStore.for_frame_path(path, store_name or None)
                try:
                    frame_value = raw.get("frame")
                    frame_value = int(frame_value)
                except (TypeError, ValueError):
                    frame_value = AnnotationStore.frame_number_from_path(path)
                result = ("store", store, frame_value)
                if isinstance(self._annotation_target_cache, dict):
                    self._annotation_target_cache[path] = result
                return result

            if isinstance(raw.get("shapes"), list) and len(raw.get("shapes")) > 0:
                result = ("json", None, None)
                if isinstance(self._annotation_target_cache, dict):
                    self._annotation_target_cache[path] = result
                return result

        if (
            fallback_store.store_path.exists()
            and frame is not None
            and not Path(path).exists()
        ):
            result = ("store", fallback_store, int(frame))
            if isinstance(self._annotation_target_cache, dict):
                self._annotation_target_cache[path] = result
            return result

        if fallback_store.store_path.exists() and frame is not None:
            try:
                record = fallback_store.get_frame_fast(int(frame))
                if record is None:
                    record = fallback_store.get_frame(int(frame))
                if record is not None:
                    result = ("store", fallback_store, int(frame))
                    if isinstance(self._annotation_target_cache, dict):
                        self._annotation_target_cache[path] = result
                    return result
            except Exception:
                logger.debug(
                    "Failed to inspect fallback annotation store for %s",
                    path,
                    exc_info=True,
                )

        result = ("json", None, None)
        if isinstance(self._annotation_target_cache, dict):
            self._annotation_target_cache[path] = result
        return result

    def _build_labelme_record(self, lf):
        return {
            "version": getattr(lf, "version", None) or "annolid",
            "flags": lf.flags or {},
            "shapes": lf.shapes,
            "imagePath": lf.imagePath,
            "imageData": lf.imageData,
            "imageHeight": getattr(lf, "imageHeight", None),
            "imageWidth": getattr(lf, "imageWidth", None),
            "caption": lf.caption,
            "otherData": dict(getattr(lf, "otherData", {}) or {}),
        }

    def _save_annotation_target(self, label_file, lf):
        target_kind, store, frame = self._resolve_annotation_target(label_file)
        if target_kind == "store" and store is not None:
            if frame is None:
                raise AnnotationStoreError(
                    f"Frame number required to update annotation store-backed file: {label_file}"
                )
            log = (
                logger.debug
                if isinstance(self._annotation_store_batch, dict)
                else logger.info
            )
            log(
                "Saving frame %s via annotation store %s (%s)",
                frame,
                store.store_path,
                label_file,
            )
            record = self._build_labelme_record(lf)
            if isinstance(self._annotation_store_batch, dict):
                batch = self._annotation_store_batch.setdefault(store.store_path, {})
                batch[int(frame)] = record
            else:
                store.update_frame(int(frame), record)
            return

        if isinstance(self._annotation_json_batch, dict):
            logger.debug("Saving frame via JSON file %s", label_file)
            self._annotation_json_batch[str(label_file)] = {
                "shapes": lf.shapes,
                "imagePath": lf.imagePath,
                "imageHeight": getattr(lf, "imageHeight", None),
                "imageWidth": getattr(lf, "imageWidth", None),
                "imageData": lf.imageData,
                "otherData": lf.otherData,
                "flags": lf.flags,
                "caption": lf.caption,
            }
            return

        logger.info("Saving frame via JSON file %s", label_file)
        lf.save(
            label_file,
            lf.shapes,
            lf.imagePath,
            getattr(lf, "imageHeight", None),
            getattr(lf, "imageWidth", None),
            lf.imageData,
            lf.otherData,
            lf.flags,
            lf.caption,
        )

    def _flush_annotation_batches(self) -> tuple[int, int]:
        store_count = 0
        json_count = 0
        if isinstance(self._annotation_store_batch, dict):
            for store_path, updates in self._annotation_store_batch.items():
                if not updates:
                    continue
                store_count += len(updates)
                logger.info(
                    "Saving %s frame(s) via annotation store %s",
                    len(updates),
                    store_path,
                )
                AnnotationStore(Path(store_path)).update_frames(updates)
            self._annotation_store_batch.clear()

        if isinstance(self._annotation_json_batch, dict):
            for label_file, payload in self._annotation_json_batch.items():
                json_count += 1
                lf = LabelFile()
                lf.save(
                    label_file,
                    payload["shapes"],
                    payload["imagePath"],
                    payload["imageHeight"],
                    payload["imageWidth"],
                    payload["imageData"],
                    payload["otherData"],
                    payload["flags"],
                    payload["caption"],
                )
            if json_count:
                logger.info("Saving %s frame(s) via JSON files", json_count)
            self._annotation_json_batch.clear()

        return store_count, json_count

    def _discard_annotation_batches(self) -> tuple[int, int]:
        store_count = 0
        json_count = 0
        if isinstance(self._annotation_store_batch, dict):
            store_count = sum(
                len(updates) for updates in self._annotation_store_batch.values()
            )
            self._annotation_store_batch.clear()
        if isinstance(self._annotation_json_batch, dict):
            json_count = len(self._annotation_json_batch)
            self._annotation_json_batch.clear()
        return store_count, json_count

    def _should_run_shape_action_in_background(self, frame_count: int) -> bool:
        _ = frame_count
        main_window = self.main_window
        return bool(getattr(main_window, "shape_actions_run_in_background", True))

    def _start_shape_action_worker(
        self, task, *, action: str, original_frame: int, restore_shape_state=None
    ):
        owner = self.main_window if self.main_window is not None else self
        parent_widget = owner if isinstance(owner, QtWidgets.QWidget) else self
        thread_parent = owner if isinstance(owner, QtCore.QObject) else self
        progress = QtWidgets.QProgressDialog(
            f"{action.capitalize()} shapes in background...",
            "Cancel",
            0,
            0,
            parent_widget,
        )
        progress.setWindowTitle("Shape Action")
        cancel_button = QtWidgets.QPushButton("Cancel", progress)
        progress.setCancelButton(cancel_button)
        progress.setMinimumDuration(0)
        progress.setWindowModality(QtCore.Qt.NonModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)

        thread = QtCore.QThread(thread_parent)
        worker = FlexibleWorker(task)
        worker.moveToThread(thread)

        self._shape_action_thread = thread
        self._shape_action_worker = worker
        self._shape_action_progress = progress
        self._shape_action_cancel_button = cancel_button
        self._pending_shape_action = {
            "action": action,
            "original_frame": int(original_frame),
            "restore_shape_state": restore_shape_state,
        }
        self._shape_action_cancel_requested = False
        self.background_action_started = True
        active_jobs = getattr(owner, "_shape_action_dialog_jobs", None)
        if not isinstance(active_jobs, list):
            active_jobs = []
            setattr(owner, "_shape_action_dialog_jobs", active_jobs)
        active_jobs.append(self)

        thread.started.connect(worker.run)
        progress.canceled.connect(self._cancel_shape_action)
        worker.finished_signal.connect(
            self._on_shape_action_finished, QtCore.Qt.QueuedConnection
        )
        worker.finished_signal.connect(thread.quit, QtCore.Qt.QueuedConnection)
        thread.finished.connect(worker.deleteLater, QtCore.Qt.QueuedConnection)
        thread.finished.connect(thread.deleteLater, QtCore.Qt.QueuedConnection)
        thread.finished.connect(
            self._clear_shape_action_worker, QtCore.Qt.QueuedConnection
        )

        progress.show()
        if hasattr(self.main_window, "statusBar"):
            self.main_window.statusBar().showMessage(
                f"{action.capitalize()} is running in the background..."
            )
        thread.start()

    def _cancel_shape_action(self):
        worker = self._shape_action_worker
        progress = self._shape_action_progress
        self._shape_action_cancel_requested = True
        if progress is not None:
            progress.setLabelText("Canceling shape action...")
        cancel_button = self._shape_action_cancel_button
        if cancel_button is not None:
            cancel_button.setEnabled(False)
        if worker is not None and hasattr(worker, "request_stop"):
            worker.request_stop()
        if hasattr(self.main_window, "statusBar"):
            self.main_window.statusBar().showMessage("Canceling shape action...")

    def _clear_shape_action_worker(self):
        self._shape_action_thread = None
        self._shape_action_worker = None
        self._shape_action_cancel_button = None
        active_jobs = getattr(self.main_window, "_shape_action_dialog_jobs", None)
        if isinstance(active_jobs, list) and self in active_jobs:
            active_jobs.remove(self)

    def _on_shape_action_finished(self, result):
        progress = self._shape_action_progress
        self._shape_action_progress = None
        if progress is not None:
            progress.close()

        pending = self._pending_shape_action or {}
        self._pending_shape_action = None
        action = pending.get("action", "action")
        original_frame = pending.get("original_frame", self.current_frame)
        restore_shape_state = pending.get("restore_shape_state")

        if isinstance(result, Exception):
            logger.error(
                "Background shape action failed.",
                exc_info=(type(result), result, result.__traceback__),
            )
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "Shape Action Failed",
                f"Failed to {action}: {result}",
            )
            self._restore_shape_state(restore_shape_state)
            self.apply_btn.setEnabled(True)
            return

        final_updated_frame = None
        canceled = False
        updated_frames = None
        skipped_frames = None
        if isinstance(result, dict):
            final_updated_frame = result.get("final_updated_frame")
            canceled = bool(result.get("canceled", False))
            updated_frames = result.get("updated_frames")
            skipped_frames = result.get("skipped_frames")
        if canceled:
            self._finish_shape_action_canceled(
                action,
                original_frame,
                restore_shape_state=restore_shape_state,
                updated_frames=updated_frames,
                skipped_frames=skipped_frames,
            )
            return
        self._finish_shape_action_success(action, original_frame, final_updated_frame)

    def _finish_shape_action_canceled(
        self,
        action: str,
        original_frame: int,
        *,
        restore_shape_state=None,
        updated_frames=None,
        skipped_frames=None,
    ) -> None:
        self._restore_shape_state(restore_shape_state)
        try:
            self.main_window.set_frame_number(original_frame)
        except Exception:
            logger.debug(
                "Failed to restore frame after canceled shape action.", exc_info=True
            )

        if hasattr(self.main_window, "statusBar"):
            self.main_window.statusBar().showMessage(f"{action.capitalize()} canceled.")
        detail = ""
        if updated_frames is not None and skipped_frames is not None:
            detail = f" Updated {updated_frames} frame(s), skipped {skipped_frames}."
        QtWidgets.QMessageBox.information(
            self.main_window,
            f"{action.capitalize()} Canceled",
            f"The shape action was canceled.{detail}",
        )

    def _restore_shape_state(self, restore_shape_state) -> None:
        if isinstance(restore_shape_state, dict) and isinstance(
            restore_shape_state.get("states"), list
        ):
            for state in restore_shape_state.get("states", []):
                self._restore_shape_state(state)
            return
        if not isinstance(restore_shape_state, dict):
            return
        shape = restore_shape_state.get("shape")
        if shape is None:
            return
        try:
            shape.label = restore_shape_state.get("label")
            shape.group_id = restore_shape_state.get("group_id")
            self._refresh_current_shape_views(shape)
        except Exception:
            logger.debug("Failed to restore shape state.", exc_info=True)

    def _finish_shape_action_success(
        self, action: str, original_frame: int, final_updated_frame
    ) -> None:
        if action == "rename & propagate":
            self._reload_annotation_view(final_updated_frame)
        else:
            self.main_window.set_frame_number(original_frame)

        if hasattr(self.main_window, "statusBar"):
            self.main_window.statusBar().showMessage(
                f"{action.capitalize()} completed."
            )
        QtWidgets.QMessageBox.information(
            self.main_window,
            f"{action.capitalize()} Complete",
            f"The shape has been {action}ed in future frames.",
        )
        self.accept()

    def _prediction_store_candidates(self):
        main_window = self.main_window
        store_path = getattr(main_window, "_prediction_store_path", None)
        if store_path:
            yield Path(store_path)

        for folder_attr in ("video_results_folder", "annotation_dir"):
            folder = getattr(main_window, folder_attr, None)
            if not folder:
                continue
            folder = Path(folder)
            yield AnnotationStore.for_frame_path(
                folder / f"{folder.name}_000000000.json"
            ).store_path

    def _resolve_last_available_prediction_frame(
        self, fallback_frame: int | None = None
    ) -> int | None:
        seen_paths = set()
        for store_path in self._prediction_store_candidates():
            if store_path in seen_paths:
                continue
            seen_paths.add(store_path)
            try:
                store = AnnotationStore(Path(store_path))
                if not store.store_path.exists():
                    continue
                frames = [int(frame) for frame in store.iter_frames()]
                if frames:
                    return max(frames)
            except Exception:
                logger.debug(
                    "Failed to resolve last predicted frame from %s",
                    store_path,
                    exc_info=True,
                )

        if fallback_frame is None:
            return None
        try:
            return int(fallback_frame)
        except Exception:
            return None

    def _resolve_last_available_prediction_frame_for_label_file(
        self, label_file: str, fallback_frame: int | None = None
    ) -> int | None:
        try:
            store = AnnotationStore.for_frame_path(label_file)
            if store.store_path.exists():
                frames = [int(frame) for frame in store.iter_frames()]
                if frames:
                    return max(frames)
        except Exception:
            logger.debug(
                "Failed to resolve last predicted frame from annotation file %s",
                label_file,
                exc_info=True,
            )
        return self._resolve_last_available_prediction_frame(fallback_frame)

    def _reload_annotation_view(self, frame_number: int | None = None) -> None:
        main_window = self.main_window
        try:
            if frame_number is None:
                return
            resolved_frame = int(frame_number)
            if hasattr(main_window, "set_frame_number"):
                main_window.set_frame_number(resolved_frame)
            if hasattr(main_window, "loadPredictShapes"):
                main_window.loadPredictShapes(
                    resolved_frame, getattr(main_window, "filename", "")
                )
                return
            frame_path = Path(getattr(main_window, "filename", "") or "")
            if frame_path.exists() and hasattr(main_window, "loadFile"):
                main_window.loadFile(str(frame_path))
        except Exception:
            logger.debug(
                "Failed to reload annotation view after rename.", exc_info=True
            )

    def _resolve_propagation_end_frame(self) -> int:
        target_frame = int(self.frame_spin.value())
        last_available_frame = self._resolve_last_available_prediction_frame()
        if last_available_frame is None:
            return target_frame
        return min(target_frame, int(last_available_frame))

    def _resolve_action_end_frame(
        self, action: str, label_file: str | None = None
    ) -> int:
        _ = action
        _ = label_file
        # Honor the user-selected range directly. The default value already
        # resolves to "next manual seed - 1" or "video end" when no future
        # manual seed exists.
        return int(self.frame_spin.value())

    def _label_file_for_frame(self, frame: int) -> str:
        main_window = self.main_window
        frame_path = None
        if hasattr(main_window, "_frame_image_path"):
            try:
                frame_path = main_window._frame_image_path(int(frame))
            except Exception:
                frame_path = None
        if frame_path is None:
            results_folder = getattr(main_window, "video_results_folder", None)
            if results_folder:
                folder = Path(results_folder)
                frame_path = folder / f"{folder.name}_{int(frame):09}.png"
            else:
                filename = getattr(main_window, "filename", None)
                if filename:
                    frame_path = Path(filename).with_name(
                        f"{Path(filename).stem.rsplit('_', 1)[0]}_{int(frame):09}.png"
                    )
        if frame_path is None:
            return ""
        return main_window._getLabelFile(str(frame_path))

    def _rename_shape_in_label_file(
        self, lf, label_file, reference_shape, new_label, new_group_id
    ):
        matches = self._match_shapes(lf.shapes, reference_shape)
        if not matches:
            return 0

        if self._shape_group_id(reference_shape) is not None:
            matched_ids = {id(shape) for shape in matches}
        else:
            matched_ids = {id(matches[0])}

        updated = 0
        updated_shapes = []
        for shape in lf.shapes:
            if id(shape) in matched_ids:
                updated_shapes.append(
                    self._update_shape_record(shape, new_label, new_group_id)
                )
                updated += 1
            else:
                updated_shapes.append(shape)
        lf.shapes = updated_shapes
        self._save_shape_file(label_file, lf)
        return updated

    def _rename_matching_shapes_in_label_file(
        self, lf, label_file, reference_shape, new_label, new_group_id
    ):
        reference_label = self._shape_label(reference_shape)
        matched_ids = {
            id(shape)
            for shape in lf.shapes
            if self._shape_label(shape) == reference_label
        }
        if not matched_ids:
            return 0

        updated = 0
        updated_shapes = []
        for shape in lf.shapes:
            if id(shape) in matched_ids:
                updated_shapes.append(
                    self._update_shape_record(shape, new_label, new_group_id)
                )
                updated += 1
            else:
                updated_shapes.append(shape)
        lf.shapes = updated_shapes
        self._save_shape_file(label_file, lf)
        return updated

    def _switch_matching_shape_labels_in_label_file(
        self,
        lf,
        label_file,
        *,
        original_label,
        original_group_id,
        new_label,
        new_group_id,
    ):
        updated = 0
        updated_shapes = []
        for shape in lf.shapes:
            shape_label = self._shape_label(shape)
            if shape_label == original_label:
                updated_shapes.append(
                    self._update_shape_record(shape, new_label, new_group_id)
                )
                updated += 1
            elif shape_label == new_label:
                updated_shapes.append(
                    self._update_shape_record(shape, original_label, original_group_id)
                )
                updated += 1
            else:
                updated_shapes.append(shape)
        if updated:
            lf.shapes = updated_shapes
            self._save_shape_file(label_file, lf)
        return updated

    def compute_centroid(self, shape):
        """Compute the centroid of a shape (works for object or dict with 'points')."""
        if isinstance(shape, dict):
            pts = shape.get("points", [])
        else:
            pts = [(pt.x(), pt.y()) for pt in shape.points]
        if not pts:
            return (0, 0)
        xs, ys = zip(*pts)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def load_or_create_label_file(self, label_file):
        """
        Load a label file if it exists; otherwise, create a new one with default metadata.
        Returns a LabelFile instance.
        """
        main_window = self.main_window
        existing = self._load_existing_label_file(label_file)
        if existing is not None:
            return existing

        if not osp.exists(label_file):
            logger.info(f"Label file {label_file} not found. Creating a new one.")
            lf = LabelFile()
            lf.filename = label_file
            try:
                lf.imageHeight = main_window.image.height()
                lf.imageWidth = main_window.image.width()
            except Exception:
                lf.imageHeight = None
                lf.imageWidth = None
            lf.shapes = []
            lf.caption = ""
            lf.flags = {}
            lf.otherData = {}
        else:
            try:
                lf = LabelFile(label_file, is_video_frame=True)
            except Exception as e:
                logger.error(f"Error loading label file {label_file}: {e}")
                return None
        return lf

    def _load_existing_label_file(self, label_file):
        target_kind, store, frame = self._resolve_annotation_target(label_file)
        if target_kind == "store" and store is not None:
            if frame is None:
                return None
            try:
                record = store.get_frame_fast(int(frame))
                if record is None:
                    record = store.get_frame(int(frame))
                if record is None:
                    return None
            except Exception:
                logger.debug(
                    "Failed to inspect store-backed annotation %s",
                    label_file,
                    exc_info=True,
                )
                return None
            return self._label_file_from_record(record, label_file)

        if not osp.exists(label_file):
            return None
        try:
            return LabelFile(label_file, is_video_frame=True)
        except Exception as exc:
            logger.error(f"Error loading label file {label_file}: {exc}")
            return None

    def _preload_store_records(self, frame_label_files):
        store_frames: dict[Path, set[int]] = {}
        label_targets = {}
        for frame, label_file in frame_label_files:
            if not label_file:
                continue
            target_kind, store, target_frame = self._resolve_annotation_target(
                label_file
            )
            label_targets[label_file] = (target_kind, store, target_frame)
            if target_kind != "store" or store is None or target_frame is None:
                continue
            store_frames.setdefault(store.store_path, set()).add(int(target_frame))

        preload = {}
        for store_path, frames in store_frames.items():
            store = AnnotationStore(Path(store_path))
            records = store.get_frames_fast(frames)
            for frame in frames:
                if int(frame) in records:
                    continue
                record = store.get_frame(int(frame))
                if record is not None:
                    records[int(frame)] = record
            preload[store_path] = records
        return label_targets, preload

    def _load_existing_label_file_from_preload(
        self, label_file, label_targets, preloaded_store_records
    ):
        target_kind, store, frame = label_targets.get(
            label_file, self._resolve_annotation_target(label_file)
        )
        if target_kind == "store" and store is not None:
            if frame is None:
                return None
            record = preloaded_store_records.get(store.store_path, {}).get(int(frame))
            if record is None:
                return None
            return self._label_file_from_record(record, label_file)
        return self._load_existing_label_file(label_file)

    def _label_file_from_record(self, record, label_file):
        lf = LabelFile()
        lf.filename = label_file
        lf.flags = record.get("flags") or {}
        lf.shapes = record.get("shapes") or []
        lf.imagePath = record.get("imagePath")
        lf.imageData = record.get("imageData")
        lf.imageHeight = record.get("imageHeight")
        lf.imageWidth = record.get("imageWidth")
        lf.caption = record.get("caption")
        lf.otherData = dict(record.get("otherData") or {})
        lf.is_video_frame = True
        return lf

    def _execute_shape_range_action(
        self,
        *,
        action: str,
        selected_shape_record,
        reference_shape,
        new_label: str,
        new_group_id,
        current_label_file: str,
        original_frame: int,
        frame_label_files,
        label_switch=None,
        stop_event=None,
    ):
        frame_numbers = [
            int(frame) for frame, label_file in frame_label_files if label_file
        ]
        range_text = (
            f"{min(frame_numbers)}-{max(frame_numbers)}" if frame_numbers else "none"
        )
        logger.info(
            "Starting shape action '%s' over frame range %s (%s target frame(s)).",
            action,
            range_text,
            len(frame_numbers),
        )
        updated_frames = []
        skipped_frames = []
        store_batch_count = 0
        json_batch_count = 0
        canceled = False
        self._annotation_store_batch = {}
        self._annotation_json_batch = {}
        self._annotation_target_cache = {}
        label_targets, preloaded_store_records = self._preload_store_records(
            frame_label_files
        )
        if stop_event is not None and stop_event.is_set():
            canceled = True
        if not canceled and action in {"rename & propagate", "switch labels"}:
            current_lf = self._load_existing_label_file(current_label_file)
            if current_lf is not None:
                if isinstance(label_switch, dict):
                    self._switch_matching_shape_labels_in_label_file(
                        current_lf,
                        current_label_file,
                        original_label=label_switch["original_label"],
                        original_group_id=label_switch["original_group_id"],
                        new_label=new_label,
                        new_group_id=new_group_id,
                    )
                else:
                    self._rename_matching_shapes_in_label_file(
                        current_lf,
                        current_label_file,
                        reference_shape,
                        new_label,
                        new_group_id,
                    )
            else:
                logger.warning(
                    "Shape action '%s' skipped current frame %s because no existing annotation record was found.",
                    action,
                    original_frame,
                )

        final_updated_frame = (
            original_frame
            if action in {"rename & propagate", "switch labels"}
            else None
        )
        try:
            for frame, label_file in frame_label_files:
                if stop_event is not None and stop_event.is_set():
                    canceled = True
                    break
                if not label_file:
                    continue
                target_kind, _, _ = label_targets.get(
                    label_file, self._resolve_annotation_target(label_file)
                )
                if action == "propagate":
                    lf = self._load_existing_label_file_from_preload(
                        label_file, label_targets, preloaded_store_records
                    )
                    if lf is None:
                        if target_kind == "store":
                            logger.debug(
                                "Skipping store-backed frame %s during propagate because the existing record could not be loaded.",
                                frame,
                            )
                            skipped_frames.append(frame)
                            continue
                        lf = self.load_or_create_label_file(label_file)
                else:
                    lf = self._load_existing_label_file_from_preload(
                        label_file, label_targets, preloaded_store_records
                    )
                if lf is None:
                    skipped_frames.append(frame)
                    continue

                shapes = lf.shapes
                frame_saved = False

                if action == "propagate":
                    new_shape_dict = dict(selected_shape_record)
                    if self._shape_group_id(new_shape_dict) is None:
                        new_shape_dict["group_id"] = self._shape_group_id(
                            selected_shape_record
                        )
                    matches = self._match_shapes(shapes, reference_shape)
                    if not matches:
                        shapes.append(new_shape_dict)
                    else:
                        match_ids = {id(shape) for shape in matches}
                        replaced = False
                        updated_shapes = []
                        for existing_shape in shapes:
                            if id(existing_shape) in match_ids:
                                updated_shapes.append(new_shape_dict)
                                replaced = True
                            else:
                                updated_shapes.append(existing_shape)
                        if not replaced:
                            updated_shapes.append(new_shape_dict)
                        shapes = updated_shapes
                elif action in {"rename & propagate", "switch labels"}:
                    if isinstance(label_switch, dict):
                        self._switch_matching_shape_labels_in_label_file(
                            lf,
                            label_file,
                            original_label=label_switch["original_label"],
                            original_group_id=label_switch["original_group_id"],
                            new_label=new_label,
                            new_group_id=new_group_id,
                        )
                    else:
                        self._rename_matching_shapes_in_label_file(
                            lf,
                            label_file,
                            reference_shape,
                            new_label,
                            new_group_id,
                        )
                    shapes = lf.shapes
                    frame_saved = True
                elif action == "delete":
                    logger.debug(
                        "Deleting shape with label: %s | Details: %s",
                        self._shape_label(reference_shape),
                        reference_shape,
                    )
                    shapes = [
                        s
                        for s in shapes
                        if not self._shape_matches_reference(s, reference_shape)
                    ]

                lf.shapes = shapes

                if not frame_saved:
                    self._save_shape_file(label_file, lf)
                final_updated_frame = frame
                updated_frames.append(frame)

            if stop_event is not None and stop_event.is_set():
                canceled = True
            if canceled:
                discarded_store_count, discarded_json_count = (
                    self._discard_annotation_batches()
                )
                logger.info(
                    "Canceled shape action '%s' over frame range %s: updated=%s skipped=%s pending_store_batch_discarded=%s pending_json_batch_discarded=%s.",
                    action,
                    range_text,
                    len(updated_frames),
                    len(skipped_frames),
                    discarded_store_count,
                    discarded_json_count,
                )
                return {
                    "final_updated_frame": final_updated_frame,
                    "updated_frames": len(updated_frames),
                    "skipped_frames": len(skipped_frames),
                    "canceled": True,
                }

            if isinstance(self._annotation_store_batch, dict):
                store_batch_count = sum(
                    len(updates) for updates in self._annotation_store_batch.values()
                )
            if isinstance(self._annotation_json_batch, dict):
                json_batch_count = len(self._annotation_json_batch)
            self._flush_annotation_batches()
            logger.info(
                "Completed shape action '%s' over frame range %s: updated=%s skipped=%s store_batched=%s json_batched=%s.",
                action,
                range_text,
                len(updated_frames),
                len(skipped_frames),
                store_batch_count,
                json_batch_count,
            )
        finally:
            self._annotation_store_batch = None
            self._annotation_json_batch = None
            self._annotation_target_cache = None

        return {
            "final_updated_frame": final_updated_frame,
            "updated_frames": len(updated_frames),
            "skipped_frames": len(skipped_frames),
            "canceled": False,
        }

    def do_action(self):
        item = self.shape_list.currentItem()
        if not item:
            QtWidgets.QMessageBox.warning(
                self, "No Selection", "Please select a shape."
            )
            return

        selected_shape = item.data(QtCore.Qt.UserRole)
        # 'propagate', 'delete', or 'define proximity event'
        action = self.action_combo.currentText().lower()
        main_window = self.main_window
        reference_shape = shape_to_dict(selected_shape)

        # If action is "delete", warn the user about irreversibility.
        if action == "delete":
            reply = QtWidgets.QMessageBox.question(
                self,
                "Confirm Deletion",
                "Are you sure you want to delete this shape? This action cannot be undone.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                # User decided not to delete, so exit the method.
                return

        if action in ["propagate", "rename & propagate", "switch labels", "delete"]:
            new_label = self.rename_line.text().strip()
            if action in {"rename & propagate", "switch labels"}:
                if not new_label:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Missing Input",
                        "Please choose a target shape label to switch/rename.",
                    )
                    return
                validate_label = getattr(main_window, "validateLabel", None)
                if callable(validate_label) and not validate_label(new_label):
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Label", f"Invalid label '{new_label}'."
                    )
                    return

            current_label_file = main_window._getLabelFile(main_window.filename)
            original_frame = self.current_frame
            action_end_frame = self._resolve_action_end_frame(
                action, current_label_file
            )
            logger.info(
                f"{action.capitalize()} shape '{selected_shape.label}' from frame {self.current_frame + 1} to {action_end_frame}"
            )
            start_frame = original_frame if action == "delete" else original_frame + 1
            frame_label_files = [
                (frame, self._label_file_for_frame(frame))
                for frame in range(start_frame, action_end_frame + 1)
            ]
            new_group_id = self._shape_group_id(selected_shape)
            restore_shape_state = None
            label_switch = None
            if action in {"rename & propagate", "switch labels"}:
                original_label = self._shape_label(selected_shape)
                original_group_id = new_group_id
                label_switch_target = self._find_selected_label_switch_target(
                    selected_shape, new_label
                )
                restore_shape_state = {
                    "shape": selected_shape,
                    "label": original_label,
                    "group_id": original_group_id,
                }
                if action == "switch labels" and label_switch_target is None:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Switch Requires Two Shapes",
                        "Select exactly two shapes with different labels, then choose the other label.",
                    )
                    return
                if label_switch_target is not None:
                    target_original_label = self._shape_label(label_switch_target)
                    target_original_group_id = self._shape_group_id(label_switch_target)
                    missing_group_count = int(original_group_id is None) + int(
                        target_original_group_id is None
                    )
                    allocated_group_ids = self._allocate_group_ids(missing_group_count)
                    if original_group_id is None:
                        original_group_id = allocated_group_ids.pop(0)
                    if target_original_group_id is None:
                        target_original_group_id = allocated_group_ids.pop(0)

                    new_group_id = target_original_group_id
                    label_switch = {
                        "original_label": original_label,
                        "original_group_id": original_group_id,
                    }
                    restore_shape_state = {
                        "states": [
                            restore_shape_state,
                            {
                                "shape": label_switch_target,
                                "label": target_original_label,
                                "group_id": self._shape_group_id(label_switch_target),
                            },
                        ]
                    }
                    label_switch_target.label = original_label
                    label_switch_target.group_id = original_group_id
                    self._refresh_current_shape_views(label_switch_target)
                elif new_group_id is None and action == "rename & propagate":
                    new_group_id = self._allocate_group_id()
                selected_shape.label = new_label
                selected_shape.group_id = new_group_id
                self._refresh_current_shape_views(selected_shape)

            selected_shape_record = shape_to_dict(selected_shape)

            def _task(stop_event=None):
                return self._execute_shape_range_action(
                    action=action,
                    selected_shape_record=selected_shape_record,
                    reference_shape=reference_shape,
                    new_label=new_label,
                    new_group_id=new_group_id,
                    current_label_file=current_label_file,
                    original_frame=original_frame,
                    frame_label_files=frame_label_files,
                    label_switch=label_switch,
                    stop_event=stop_event,
                )

            frame_count = len(frame_label_files)
            if action in {"rename & propagate", "switch labels"}:
                frame_count += 1
            if self._should_run_shape_action_in_background(frame_count):
                self.apply_btn.setEnabled(False)
                self._start_shape_action_worker(
                    _task,
                    action=action,
                    original_frame=original_frame,
                    restore_shape_state=restore_shape_state,
                )
                self.accept()
                return

            result = _task()
            final_updated_frame = result.get("final_updated_frame")
            self._finish_shape_action_success(
                action, original_frame, final_updated_frame
            )

        elif action == "define proximity event":
            target_group = self.target_group_combo.currentText()
            event_name = self.event_name_line.text().strip()
            if not event_name:
                QtWidgets.QMessageBox.warning(
                    self, "Missing Input", "Please enter an event name."
                )
                return
            proximity_threshold = self.proximity_threshold_spin.value()
            rule_type = self.rule_type_combo.currentText().lower()  # "any" or "all"
            event_start_frame = self.event_start_frame_spin.value()
            event_end_frame = self.event_end_frame_spin.value()
            if event_start_frame > event_end_frame:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Frame Range",
                    "Event start frame must be less than or equal to event end frame.",
                )
                return

            frames_updated = 0
            # Evaluate the proximity event for each frame in the given range.
            for frame in range(event_start_frame, event_end_frame + 1):
                main_window.set_frame_number(frame)
                label_file = main_window._getLabelFile(main_window.filename)
                lf = self.load_or_create_label_file(label_file)
                if lf is None:
                    continue

                # Filter target shapes from the current label file.
                target_shapes = []
                for shape in lf.shapes:
                    if target_group.lower() == "all others":
                        if shape.get("label") != selected_shape.label:
                            target_shapes.append(shape)
                    else:
                        if shape.get("label") == target_group:
                            target_shapes.append(shape)

                # Use Shapely to check proximity or overlap.
                if target_shapes:
                    proximity_results = [
                        are_polygons_close_or_overlap(
                            selected_shape, target, proximity_threshold
                        )
                        for target in target_shapes
                    ]
                    if rule_type == "any":
                        triggered = any(proximity_results)
                    else:  # rule_type == "all"
                        triggered = all(proximity_results)
                else:
                    triggered = False

                if triggered:
                    lf.flags[event_name] = True
                    # only keep the event flag is true
                    lf.flags = {k: v for k, v in lf.flags.items() if v}
                    frames_updated += 1
                    lf.save(
                        label_file,
                        lf.shapes,
                        lf.imagePath,
                        getattr(lf, "imageHeight", None),
                        getattr(lf, "imageWidth", None),
                        lf.imageData,
                        lf.otherData,
                        lf.flags,
                        lf.caption,
                    )
                    logger.info(f"Frame {frame} updated with event flag: {event_name}.")

            main_window.set_frame_number(self.current_frame)
            QtWidgets.QMessageBox.information(
                self,
                "Define Proximity Event Complete",
                f"Event '{event_name}' applied to {frames_updated} frame(s) where the condition was met.",
            )
            self.accept()
