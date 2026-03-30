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
            ["Propagate", "Rename & Propagate", "Delete", "Define Proximity Event"]
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
        rename_layout = QtWidgets.QHBoxLayout(self.rename_widget)
        rename_layout.setContentsMargins(0, 0, 0, 0)
        self.rename_label = QtWidgets.QLabel("Rename to:", self.rename_widget)
        self.rename_line = QtWidgets.QLineEdit(self.rename_widget)
        self.rename_line.setPlaceholderText("New label")
        rename_layout.addWidget(self.rename_label)
        rename_layout.addWidget(self.rename_line, 1)
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
        rename_action = current_action == "rename & propagate"
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
            return 0
        return max(candidates) + 1

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
        if path.lower().endswith(".ndjson"):
            return "store", AnnotationStore(Path(path)), None
        if not path.lower().endswith(".json"):
            return "json", None, None

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
                return "store", store, frame_value

            if isinstance(raw.get("shapes"), list) and len(raw.get("shapes")) > 0:
                return "json", None, None

        if fallback_store.store_path.exists() and frame is not None:
            try:
                if fallback_store.get_frame(int(frame)) is not None:
                    return "store", fallback_store, int(frame)
            except Exception:
                logger.debug(
                    "Failed to inspect fallback annotation store for %s",
                    path,
                    exc_info=True,
                )

        return "json", None, None

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
            logger.info(
                "Saving frame %s via annotation store %s (%s)",
                frame,
                store.store_path,
                label_file,
            )
            store.update_frame(int(frame), self._build_labelme_record(lf))
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
        target_frame = int(self.frame_spin.value())
        if action == "rename & propagate":
            if label_file:
                resolved = self._resolve_last_available_prediction_frame_for_label_file(
                    label_file
                )
                if resolved is not None:
                    return min(target_frame, int(resolved))
            return self._resolve_propagation_end_frame()
        return target_frame

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

        if action in ["propagate", "rename & propagate", "delete"]:
            new_label = self.rename_line.text().strip()
            if action == "rename & propagate":
                if not new_label:
                    QtWidgets.QMessageBox.warning(
                        self, "Missing Input", "Please enter a new label name."
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
            # if delete start from the current frame
            if action == "delete":
                self.current_frame -= 1

            renamed_shape = selected_shape
            new_group_id = self._shape_group_id(selected_shape)
            if action == "rename & propagate":
                if new_group_id is None:
                    new_group_id = self._allocate_group_id()
                renamed_shape.label = new_label
                renamed_shape.group_id = new_group_id
                self._refresh_current_shape_views(renamed_shape)

                current_lf = self._load_existing_label_file(current_label_file)
                if current_lf is not None:
                    self._rename_matching_shapes_in_label_file(
                        current_lf,
                        current_label_file,
                        reference_shape,
                        new_label,
                        new_group_id,
                    )
                else:
                    logger.warning(
                        "Rename & propagate skipped current frame %s because no existing annotation record was found.",
                        original_frame,
                    )

            final_updated_frame = (
                original_frame if action == "rename & propagate" else None
            )
            for frame in range(self.current_frame + 1, action_end_frame + 1):
                label_file = self._label_file_for_frame(frame)
                if not label_file:
                    continue
                target_kind, _, _ = self._resolve_annotation_target(label_file)
                if action == "propagate":
                    lf = self._load_existing_label_file(label_file)
                    if lf is None:
                        if target_kind == "store":
                            logger.warning(
                                "Skipping store-backed frame %s during propagate because the existing record could not be loaded.",
                                frame,
                            )
                            continue
                        lf = self.load_or_create_label_file(label_file)
                else:
                    lf = self._load_existing_label_file(label_file)
                if lf is None:
                    continue

                shapes = lf.shapes
                frame_saved = False

                if action == "propagate":
                    new_shape = (
                        selected_shape.copy()
                        if hasattr(selected_shape, "copy")
                        else selected_shape
                    )
                    if self._shape_group_id(new_shape) is None:
                        new_shape.group_id = self._shape_group_id(selected_shape)
                    new_shape_dict = shape_to_dict(new_shape)
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
                elif action == "rename & propagate":
                    if new_group_id is None:
                        new_group_id = self._allocate_group_id()
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
                    # Convert the shape to a dictionary for detailed logging.
                    shape_details = reference_shape
                    logger.info(
                        f"Deleting shape with label: {selected_shape.label} | Details: {shape_details}"
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
                logger.info(f"Frame {frame} updated with action: {action}.")

            if action == "rename & propagate":
                self._reload_annotation_view(final_updated_frame)
            else:
                main_window.set_frame_number(original_frame)
            QtWidgets.QMessageBox.information(
                self,
                f"{action.capitalize()} Complete",
                f"The shape has been {action}ed in future frames.",
            )
            self.accept()

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
