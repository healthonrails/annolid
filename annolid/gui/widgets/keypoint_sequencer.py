from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from annolid.annotation.pose_schema import PoseSchema
from annolid.gui.keypoint_catalog import (
    extract_labels_from_uniq_label_list,
    merge_keypoint_lists,
    normalize_keypoint_names,
)


class _SkeletonPreviewWidget(QtWidgets.QWidget):
    """Lightweight skeleton preview for keypoint sequence selection."""

    keypointClicked = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._schema: Optional[PoseSchema] = None
        self._active_keypoints: set[str] = set()
        self._pixel_points: dict[str, QtCore.QPointF] = {}
        self.setMinimumHeight(170)

    def set_schema(self, schema: Optional[PoseSchema]) -> None:
        self._schema = schema
        self.update()

    def set_active_keypoints(self, keypoints: List[str]) -> None:
        self._active_keypoints = {str(k).strip() for k in keypoints if str(k).strip()}
        self.update()

    @staticmethod
    def _x_hint(name: str) -> float:
        lowered = name.lower()
        if "left" in lowered or lowered.startswith("l_") or lowered.endswith("_l"):
            return 0.25
        if "right" in lowered or lowered.startswith("r_") or lowered.endswith("_r"):
            return 0.75
        return 0.5

    def _layout_points(self, keypoints: List[str]) -> dict[str, QtCore.QPointF]:
        points: dict[str, QtCore.QPointF] = {}
        if not keypoints:
            return points
        n = len(keypoints)
        for idx, kp in enumerate(keypoints):
            y = (idx + 1) / (n + 1)
            x = self._x_hint(kp)
            points[kp] = QtCore.QPointF(x, y)
        return points

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.fillRect(rect, QtGui.QColor("#f5f7fb"))

        schema = self._schema
        keypoints = list(getattr(schema, "keypoints", []) or [])
        if not keypoints:
            painter.setPen(QtGui.QColor("#6b7280"))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No pose schema loaded")
            return

        points = self._layout_points(keypoints)
        pixel_points: dict[str, QtCore.QPointF] = {}
        for kp, p in points.items():
            px = rect.left() + p.x() * rect.width()
            py = rect.top() + p.y() * rect.height()
            pixel_points[kp] = QtCore.QPointF(px, py)
        self._pixel_points = pixel_points

        edge_pen = QtGui.QPen(QtGui.QColor("#94a3b8"))
        edge_pen.setWidth(2)
        painter.setPen(edge_pen)
        for raw_a, raw_b in list(getattr(schema, "edges", []) or []):
            _inst_a, a = schema.strip_instance_prefix(str(raw_a))
            _inst_b, b = schema.strip_instance_prefix(str(raw_b))
            p1 = pixel_points.get(a)
            p2 = pixel_points.get(b)
            if p1 is not None and p2 is not None:
                painter.drawLine(p1, p2)

        for kp in keypoints:
            point = pixel_points.get(kp)
            if point is None:
                continue
            active = kp in self._active_keypoints
            color = QtGui.QColor("#0ea5e9" if active else "#9ca3af")
            painter.setBrush(color)
            painter.setPen(QtGui.QPen(QtGui.QColor("#1f2937"), 1))
            painter.drawEllipse(point, 5, 5)
            painter.setPen(QtGui.QColor("#111827"))
            painter.drawText(
                QtCore.QRectF(point.x() + 6, point.y() - 10, 130, 20),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                kp,
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.pos()
        hit_name: Optional[str] = None
        best_dist = float("inf")
        for name, point in self._pixel_points.items():
            dx = float(point.x() - pos.x())
            dy = float(point.y() - pos.y())
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                hit_name = name
        if hit_name is not None and best_dist <= 12.0:
            self.keypointClicked.emit(hit_name)
            event.accept()
            return
        super().mousePressEvent(event)


class _LockableKeypointListWidget(QtWidgets.QListWidget):
    """List widget with a small lock click target per row."""

    lockToggleRequested = QtCore.Signal(QtWidgets.QListWidgetItem)
    LOCK_HIT_WIDTH = 44

    @staticmethod
    def _lock_rect(item_rect: QtCore.QRect) -> QtCore.QRect:
        # Lock marker hit-zone at row start (icon + padding).
        return QtCore.QRect(
            item_rect.left(),
            item_rect.top(),
            _LockableKeypointListWidget.LOCK_HIT_WIDTH,
            item_rect.height(),
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.pos())
        if item is not None:
            rect = self.visualItemRect(item)
            lock_rect = self._lock_rect(rect)
            if lock_rect.contains(event.pos()):
                self.lockToggleRequested.emit(item)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.pos())
        if item is not None and self._lock_rect(self.visualItemRect(item)).contains(
            event.pos()
        ):
            self.viewport().setCursor(QtCore.Qt.PointingHandCursor)
            self.setToolTip("Click to lock/unlock this keypoint in sequence order.")
        else:
            self.viewport().setCursor(QtCore.Qt.ArrowCursor)
            self.setToolTip("")
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)
        self.setToolTip("")
        super().leaveEvent(event)


class KeypointSequencerWidget(QtWidgets.QWidget):
    """Dock widget content for sequential keypoint point labeling."""

    poseSchemaChanged = QtCore.Signal(object, str)
    ROLE_KEYPOINT_NAME = QtCore.Qt.UserRole + 1

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._schema: Optional[PoseSchema] = None
        self._schema_path: Optional[str] = None
        self._next_index: int = 0
        self._keypoint_order: List[str] = []
        self._locked_keypoint_order: List[str] = []
        self._suspend_schema_signal: bool = False
        self._config_updating: bool = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        schema_row = QtWidgets.QHBoxLayout()
        schema_row.addWidget(QtWidgets.QLabel("Schema:"))
        self.schema_path_edit = QtWidgets.QLineEdit(self)
        self.schema_path_edit.setPlaceholderText("pose_schema.json (or .yaml)")
        self.schema_path_edit.editingFinished.connect(self._on_schema_path_edited)
        schema_row.addWidget(self.schema_path_edit, 1)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_schema)
        load_btn = QtWidgets.QPushButton("Load")
        load_btn.clicked.connect(self._load_schema_clicked)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self._save_schema_clicked)
        save_as_btn = QtWidgets.QPushButton("Save As…")
        save_as_btn.clicked.connect(self._save_schema_as_clicked)
        schema_row.addWidget(browse_btn)
        schema_row.addWidget(load_btn)
        schema_row.addWidget(save_btn)
        schema_row.addWidget(save_as_btn)
        layout.addLayout(schema_row)

        config_row = QtWidgets.QHBoxLayout()
        config_row.addWidget(QtWidgets.QLabel("Instance prefixes:"))
        self.instance_prefixes_edit = QtWidgets.QLineEdit(self)
        self.instance_prefixes_edit.setPlaceholderText(
            "instances/prefixes (comma separated)"
        )
        self.instance_prefixes_edit.editingFinished.connect(
            self._on_instance_config_changed
        )
        config_row.addWidget(self.instance_prefixes_edit, 1)
        config_row.addWidget(QtWidgets.QLabel("Separator:"))
        self.instance_separator_edit = QtWidgets.QLineEdit(self)
        self.instance_separator_edit.setFixedWidth(70)
        self.instance_separator_edit.setText("_")
        self.instance_separator_edit.editingFinished.connect(
            self._on_instance_config_changed
        )
        config_row.addWidget(self.instance_separator_edit)
        self.normalize_prefixes_btn = QtWidgets.QPushButton("Normalize prefixes")
        self.normalize_prefixes_btn.clicked.connect(self._normalize_prefixed_schema)
        config_row.addWidget(self.normalize_prefixes_btn)
        layout.addLayout(config_row)

        self.enable_checkbox = QtWidgets.QCheckBox(
            "Enable click-to-label keypoint sequence"
        )
        self.enable_checkbox.setChecked(False)
        self.enable_checkbox.toggled.connect(self._on_sequence_mode_toggled)
        layout.addWidget(self.enable_checkbox)

        self.autosave_checkbox = QtWidgets.QCheckBox("Auto-save after each point click")
        self.autosave_checkbox.setChecked(True)
        layout.addWidget(self.autosave_checkbox)

        instance_row = QtWidgets.QHBoxLayout()
        instance_row.addWidget(QtWidgets.QLabel("Instance:"))
        self.instance_combo = QtWidgets.QComboBox()
        self.instance_combo.currentIndexChanged.connect(self.reset_sequence)
        instance_row.addWidget(self.instance_combo, 1)
        layout.addLayout(instance_row)

        self.preview = _SkeletonPreviewWidget(self)
        self.preview.keypointClicked.connect(self._on_preview_keypoint_clicked)
        layout.addWidget(self.preview)

        self.edge_hint_label = QtWidgets.QLabel(
            "Edge editor: click two keypoints in the preview to add/remove an edge."
        )
        self.edge_hint_label.setWordWrap(True)
        layout.addWidget(self.edge_hint_label)

        self.edges_list = QtWidgets.QListWidget(self)
        self.edges_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.edges_list)

        edge_controls = QtWidgets.QHBoxLayout()
        remove_edge_btn = QtWidgets.QPushButton("Remove selected edge(s)")
        clear_edge_btn = QtWidgets.QPushButton("Clear edges")
        remove_edge_btn.clicked.connect(self._remove_selected_edges)
        clear_edge_btn.clicked.connect(self._clear_edges)
        edge_controls.addWidget(remove_edge_btn)
        edge_controls.addWidget(clear_edge_btn)
        layout.addLayout(edge_controls)

        symmetry_label = QtWidgets.QLabel("Symmetry pairs (left <-> right):")
        layout.addWidget(symmetry_label)
        self.symmetry_list = QtWidgets.QListWidget(self)
        self.symmetry_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        layout.addWidget(self.symmetry_list)

        symmetry_add_row = QtWidgets.QHBoxLayout()
        self.symmetry_a_combo = QtWidgets.QComboBox(self)
        self.symmetry_b_combo = QtWidgets.QComboBox(self)
        add_pair_btn = QtWidgets.QPushButton("Add pair")
        add_pair_btn.clicked.connect(self._add_symmetry_pair_from_combos)
        symmetry_add_row.addWidget(self.symmetry_a_combo, 1)
        symmetry_add_row.addWidget(self.symmetry_b_combo, 1)
        symmetry_add_row.addWidget(add_pair_btn)
        layout.addLayout(symmetry_add_row)

        symmetry_controls = QtWidgets.QHBoxLayout()
        remove_pair_btn = QtWidgets.QPushButton("Remove selected pair(s)")
        remove_pair_btn.clicked.connect(self._remove_selected_symmetry_pairs)
        clear_pairs_btn = QtWidgets.QPushButton("Clear pairs")
        clear_pairs_btn.clicked.connect(self._clear_symmetry_pairs)
        auto_pairs_btn = QtWidgets.QPushButton("Auto symmetry")
        auto_pairs_btn.clicked.connect(self._auto_fill_symmetry_pairs)
        symmetry_controls.addWidget(remove_pair_btn)
        symmetry_controls.addWidget(clear_pairs_btn)
        symmetry_controls.addWidget(auto_pairs_btn)
        layout.addLayout(symmetry_controls)

        flip_row = QtWidgets.QHBoxLayout()
        flip_row.addWidget(QtWidgets.QLabel("flip_idx preview:"))
        self.flip_preview_label = QtWidgets.QLabel("")
        flip_row.addWidget(self.flip_preview_label, 1)
        layout.addLayout(flip_row)

        layout.addWidget(QtWidgets.QLabel("Active keypoints (checked = in sequence):"))
        self.keypoint_list = _LockableKeypointListWidget(self)
        self.keypoint_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.keypoint_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.keypoint_list.model().rowsMoved.connect(self._on_keypoint_order_changed)
        self.keypoint_list.itemChanged.connect(self._on_keypoint_checks_changed)
        self.keypoint_list.lockToggleRequested.connect(self._toggle_lock_for_item)
        layout.addWidget(self.keypoint_list, 1)
        self.keypoint_lock_hint = QtWidgets.QLabel(
            "Tip: click the small lock icon area left of a keypoint to lock/unlock its sequence order."
        )
        self.keypoint_lock_hint.setWordWrap(True)
        layout.addWidget(self.keypoint_lock_hint)

        add_row = QtWidgets.QHBoxLayout()
        self.add_keypoint_edit = QtWidgets.QLineEdit(self)
        self.add_keypoint_edit.setPlaceholderText(
            "Add keypoint (e.g. nose, left_ear, right_ear)"
        )
        add_keypoint_btn = QtWidgets.QPushButton("Add")
        add_keypoint_btn.clicked.connect(self._add_keypoint_from_input)
        self.load_from_labels_btn = QtWidgets.QPushButton("Load from Labels Dock")
        self.load_from_labels_btn.clicked.connect(self._load_from_labels_dock)
        self.add_keypoint_edit.returnPressed.connect(self._add_keypoint_from_input)
        add_row.addWidget(self.add_keypoint_edit, 1)
        add_row.addWidget(add_keypoint_btn)
        add_row.addWidget(self.load_from_labels_btn)
        layout.addLayout(add_row)

        controls_row = QtWidgets.QHBoxLayout()
        remove_btn = QtWidgets.QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_keypoints)
        move_up_btn = QtWidgets.QPushButton("Move up")
        move_up_btn.clicked.connect(lambda: self._move_selected_keypoints(-1))
        move_down_btn = QtWidgets.QPushButton("Move down")
        move_down_btn.clicked.connect(lambda: self._move_selected_keypoints(1))
        select_all_btn = QtWidgets.QPushButton("Select all")
        clear_btn = QtWidgets.QPushButton("Clear")
        reset_btn = QtWidgets.QPushButton("Reset sequence")
        controls_row.addWidget(remove_btn)
        controls_row.addWidget(move_up_btn)
        controls_row.addWidget(move_down_btn)
        select_all_btn.clicked.connect(self._select_all_keypoints)
        clear_btn.clicked.connect(self._clear_all_keypoints)
        reset_btn.clicked.connect(self.reset_sequence)
        controls_row.addWidget(select_all_btn)
        controls_row.addWidget(clear_btn)
        controls_row.addWidget(reset_btn)
        layout.addLayout(controls_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._pending_edge_start: Optional[str] = None
        self._update_status_text()

    def set_pose_schema(
        self,
        schema: Optional[PoseSchema],
        schema_path: Optional[str] = None,
        *,
        emit_change: bool = False,
    ) -> None:
        previous_instance = self.instance_combo.currentData()
        previous_active = self.active_keypoints()
        self._schema = schema
        if schema_path:
            self._schema_path = str(schema_path)
            self.schema_path_edit.setText(self._schema_path)
        self._keypoint_order = merge_keypoint_lists(
            list(getattr(schema, "keypoints", []) or []), self._keypoint_order
        )
        self._sync_schema_keypoints_from_order()
        self._rebuild_instance_options(previous_instance=previous_instance)
        self._refresh_instance_config_from_schema()
        self._rebuild_keypoint_list()
        if previous_active:
            self.set_active_keypoints(previous_active)
        self.preview.set_schema(schema)
        self._rebuild_edges_list_from_schema()
        self._rebuild_symmetry_list_from_schema()
        self._refresh_symmetry_combo_choices()
        self._refresh_lock_icons()
        self._update_flip_preview()
        has_keypoints = bool(self._keypoint_order)
        if not has_keypoints:
            self.enable_checkbox.setChecked(False)
            self.enable_checkbox.setEnabled(False)
        else:
            self.enable_checkbox.setEnabled(True)
        self._apply_sequencer_settings_from_schema(schema)
        self.reset_sequence()
        if emit_change:
            self._emit_schema_changed()

    def set_active_keypoints(self, keypoints: List[str]) -> None:
        desired = {str(k).strip() for k in keypoints if str(k).strip()}
        for i in range(self.keypoint_list.count()):
            item = self.keypoint_list.item(i)
            item.setCheckState(
                QtCore.Qt.Checked
                if self._item_keypoint_name(item) in desired
                else QtCore.Qt.Unchecked
            )
        self._on_keypoint_checks_changed()

    def load_keypoints_from_labels(self, labels: List[str]) -> None:
        cleaned = normalize_keypoint_names(labels)
        if not cleaned:
            return
        self._ensure_schema()
        self._keypoint_order = merge_keypoint_lists(self._keypoint_order, cleaned)
        self._sync_schema_keypoints_from_order()
        self._rebuild_keypoint_list()
        self.set_active_keypoints(cleaned)

    def auto_save_on_click(self) -> bool:
        return bool(self.autosave_checkbox.isChecked())

    def is_sequence_enabled(self) -> bool:
        return bool(self.enable_checkbox.isChecked()) and bool(
            self.sequence_active_keypoints()
        )

    def _item_keypoint_name(self, item: Optional[QtWidgets.QListWidgetItem]) -> str:
        if item is None:
            return ""
        raw = item.data(self.ROLE_KEYPOINT_NAME)
        if raw is None:
            text = str(item.text() or "").strip()
            if text.startswith("[L] ") or text.startswith("[ ] "):
                return text[4:].strip()
            if text.startswith("🔒 ") or text.startswith("🔓 "):
                return text[2:].strip()
            if text.startswith("⭕ "):
                return text[2:].strip()
            return text
        return str(raw).strip()

    def _display_text_for_keypoint(self, name: str) -> str:
        marker = "🔒" if name in set(self._locked_keypoint_order) else "⭕"
        return f"{marker} {name}"

    def _update_item_display(self, item: Optional[QtWidgets.QListWidgetItem]) -> None:
        if item is None:
            return
        name = self._item_keypoint_name(item)
        if not name:
            return
        item.setData(self.ROLE_KEYPOINT_NAME, name)
        item.setText(self._display_text_for_keypoint(name))

    def active_keypoints(self) -> List[str]:
        names: List[str] = []
        for i in range(self.keypoint_list.count()):
            item = self.keypoint_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                text = self._item_keypoint_name(item)
                if text:
                    names.append(text)
        return names

    def locked_keypoints(self) -> List[str]:
        return list(self._locked_keypoint_order)

    def sequence_active_keypoints(self) -> List[str]:
        # Sequence labeling is lock-only by design.
        return list(self._locked_keypoint_order)

    def keypoint_order(self) -> List[str]:
        return list(self._keypoint_order)

    def schema_path(self) -> Optional[str]:
        return self._schema_path

    def current_schema(self) -> Optional[PoseSchema]:
        return self._schema

    def consume_next_label(self) -> Optional[str]:
        active = self.sequence_active_keypoints()
        if not active:
            self._update_status_text()
            return None
        idx = self._next_index % len(active)
        base_name = active[idx]
        self._next_index = (idx + 1) % len(active)
        self._update_status_text()
        return self._compose_label(base_name)

    def reset_sequence(self) -> None:
        self._next_index = 0
        self._update_status_text()

    def _on_sequence_mode_toggled(self, enabled: bool) -> None:
        if bool(enabled):
            self.reset_sequence()
            return
        self._update_status_text()

    @staticmethod
    def _clean_saved_keypoint_list(value: object) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(k).strip() for k in value if str(k).strip()]

    def _apply_sequencer_settings_from_schema(
        self, schema: Optional[PoseSchema]
    ) -> None:
        if schema is None:
            return
        raw = getattr(schema, "sequencer", None)
        if not isinstance(raw, dict) or not raw:
            return

        valid = {str(k).strip() for k in self._keypoint_order if str(k).strip()}
        saved_locked = self._clean_saved_keypoint_list(raw.get("locked_keypoints"))
        if saved_locked:
            self._locked_keypoint_order = [k for k in saved_locked if k in valid]
            self._refresh_lock_icons()

        saved_active = self._clean_saved_keypoint_list(raw.get("active_keypoints"))
        if saved_active:
            self.set_active_keypoints(saved_active)

        if "auto_save_on_click" in raw:
            try:
                self.autosave_checkbox.setChecked(bool(raw.get("auto_save_on_click")))
            except Exception:
                pass

        if "selected_instance" in raw:
            selected_instance = raw.get("selected_instance")
            if selected_instance in (None, ""):
                self.instance_combo.setCurrentIndex(0)
            else:
                idx = self.instance_combo.findData(str(selected_instance))
                if idx >= 0:
                    self.instance_combo.setCurrentIndex(idx)

        if "enabled" in raw and self.enable_checkbox.isEnabled():
            self.enable_checkbox.setChecked(bool(raw.get("enabled")))

    def _capture_sequencer_settings(self) -> dict:
        selected_instance = self.instance_combo.currentData()
        return {
            "locked_keypoints": list(self._locked_keypoint_order),
            "active_keypoints": self.active_keypoints(),
            "enabled": bool(self.enable_checkbox.isChecked()),
            "auto_save_on_click": bool(self.autosave_checkbox.isChecked()),
            "selected_instance": (
                str(selected_instance).strip() if selected_instance else None
            ),
        }

    def _compose_label(self, keypoint: str) -> str:
        schema = self._schema
        if schema is None or not schema.instances:
            return keypoint
        current_instance = self.instance_combo.currentData()
        if not current_instance:
            return keypoint
        return f"{schema.instance_prefix(str(current_instance))}{keypoint}"

    def _ensure_schema(self) -> PoseSchema:
        if self._schema is None:
            self._schema = PoseSchema()
        if not getattr(self._schema, "instance_separator", ""):
            self._schema.instance_separator = "_"
        return self._schema

    def _refresh_instance_config_from_schema(self) -> None:
        schema = self._schema
        self._config_updating = True
        try:
            if schema is None:
                self.instance_prefixes_edit.setText("")
                self.instance_separator_edit.setText("_")
                return
            instances = [
                str(i).strip() for i in list(schema.instances or []) if str(i).strip()
            ]
            self.instance_prefixes_edit.setText(",".join(instances))
            self.instance_separator_edit.setText(
                str(getattr(schema, "instance_separator", "_") or "_")
            )
        finally:
            self._config_updating = False

    def _on_instance_config_changed(self) -> None:
        if self._config_updating:
            return
        schema = self._ensure_schema()
        raw_instances = str(self.instance_prefixes_edit.text() or "")
        instances = [
            token.strip().rstrip("_")
            for token in raw_instances.replace(";", ",").split(",")
            if token.strip()
        ]
        deduped: List[str] = []
        for inst in instances:
            if inst and inst not in deduped:
                deduped.append(inst)
        schema.instances = deduped
        schema.instance_separator = (
            str(self.instance_separator_edit.text() or "").strip() or "_"
        )
        previous_instance = self.instance_combo.currentData()
        self._rebuild_instance_options(previous_instance=previous_instance)
        self._rebuild_edges_list_from_schema()
        self._rebuild_symmetry_list_from_schema()
        self._update_flip_preview()
        self.preview.update()
        self._emit_schema_changed()

    def _normalize_prefixed_schema(self) -> None:
        schema = self._schema
        if schema is None:
            return
        schema.normalize_prefixed_keypoints()
        self._keypoint_order = list(getattr(schema, "keypoints", []) or [])
        self._suspend_schema_signal = True
        try:
            self.set_pose_schema(schema, self._schema_path, emit_change=False)
        finally:
            self._suspend_schema_signal = False
        self._emit_schema_changed()

    @staticmethod
    def _looks_like_instance_prefixed_schema(schema: PoseSchema) -> bool:
        keypoints = [
            str(k).strip() for k in list(getattr(schema, "keypoints", []) or [])
        ]
        if not keypoints:
            return False
        if list(getattr(schema, "instances", []) or []):
            return False
        sep = str(getattr(schema, "instance_separator", "_") or "_")
        if not sep:
            sep = "_"
        if not all(sep in kp for kp in keypoints):
            return False

        prefixes: set[str] = set()
        bases: set[str] = set()
        for kp in keypoints:
            inst, base = kp.split(sep, 1)
            inst = inst.strip().rstrip("_")
            base = base.strip()
            if not inst or not base:
                return False
            prefixes.add(inst)
            bases.add(base)
        # Conservative: only auto-normalize when structure clearly looks expanded
        # across multiple instances and multiple base keypoints.
        if len(prefixes) < 2 or len(bases) < 2:
            return False
        return len(prefixes) * len(bases) == len(keypoints)

    def _rebuild_instance_options(
        self, previous_instance: Optional[str] = None
    ) -> None:
        self.instance_combo.blockSignals(True)
        self.instance_combo.clear()
        self.instance_combo.addItem("Default", None)
        schema = self._schema
        if schema is not None:
            for inst in list(getattr(schema, "instances", []) or []):
                name = str(inst).strip()
                if name:
                    self.instance_combo.addItem(name, name)
        if previous_instance:
            idx = self.instance_combo.findData(previous_instance)
            if idx >= 0:
                self.instance_combo.setCurrentIndex(idx)
        self.instance_combo.blockSignals(False)

    def _rebuild_keypoint_list(self) -> None:
        previous = set(self.active_keypoints())
        self.keypoint_list.blockSignals(True)
        self.keypoint_list.clear()
        keypoints = list(self._keypoint_order)
        for kp in keypoints:
            text = str(kp).strip()
            if not text:
                continue
            item = QtWidgets.QListWidgetItem()
            item.setData(self.ROLE_KEYPOINT_NAME, text)
            item.setText(self._display_text_for_keypoint(text))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.Checked
                if not previous or text in previous
                else QtCore.Qt.Unchecked
            )
            self.keypoint_list.addItem(item)
        self.keypoint_list.blockSignals(False)
        self._on_keypoint_checks_changed()
        self._refresh_symmetry_combo_choices()
        self._refresh_lock_icons()

    def _add_keypoint_from_input(self) -> None:
        text = str(self.add_keypoint_edit.text() or "").strip()
        if not text:
            return
        self.add_keypoint_edit.clear()
        tokens = [tok.strip() for tok in text.replace(";", ",").split(",")]
        self.load_keypoints_from_labels(tokens)
        self.enable_checkbox.setEnabled(bool(self._keypoint_order))

    def _discover_keypoints_from_labels_dock(self) -> List[str]:
        window = self.window()
        uniq = getattr(window, "uniqLabelList", None)
        return extract_labels_from_uniq_label_list(uniq)

    def _load_from_labels_dock(self) -> None:
        labels = self._discover_keypoints_from_labels_dock()
        self.load_keypoints_from_labels(labels)

    def _select_all_keypoints(self) -> None:
        for i in range(self.keypoint_list.count()):
            self.keypoint_list.item(i).setCheckState(QtCore.Qt.Checked)
        self._on_keypoint_checks_changed()

    def _clear_all_keypoints(self) -> None:
        for i in range(self.keypoint_list.count()):
            self.keypoint_list.item(i).setCheckState(QtCore.Qt.Unchecked)
        self._on_keypoint_checks_changed()

    def _on_keypoint_checks_changed(self) -> None:
        self._sync_keypoint_order_from_ui()
        active = self.sequence_active_keypoints()
        if active:
            self._next_index %= len(active)
        else:
            self._next_index = 0
        self.preview.set_active_keypoints(active)
        self._update_status_text()

    def _on_keypoint_order_changed(self, *args) -> None:
        _ = args
        self._sync_keypoint_order_from_ui()
        self._sync_locked_order_to_keypoint_order()
        self._refresh_lock_icons()
        self.reset_sequence()

    def _sync_keypoint_order_from_ui(self) -> None:
        ordered: List[str] = []
        for i in range(self.keypoint_list.count()):
            item = self.keypoint_list.item(i)
            if item is None:
                continue
            text = self._item_keypoint_name(item)
            if text:
                ordered.append(text)
        self._keypoint_order = normalize_keypoint_names(ordered)
        self._refresh_lock_icons()
        self._sync_schema_keypoints_from_order()

    def _sync_schema_keypoints_from_order(self) -> None:
        schema = self._schema
        if schema is None:
            return
        schema.keypoints = list(self._keypoint_order)
        valid = set(schema.keypoints)

        def _pair_is_valid(a_raw, b_raw) -> bool:
            if a_raw == b_raw:
                return False
            _inst_a, a = schema.strip_instance_prefix(str(a_raw or ""))
            _inst_b, b = schema.strip_instance_prefix(str(b_raw or ""))
            return bool(a in valid and b in valid and a != b)

        schema.edges = [
            (a, b)
            for (a, b) in list(getattr(schema, "edges", []) or [])
            if _pair_is_valid(a, b)
        ]
        schema.symmetry_pairs = [
            (a, b)
            for (a, b) in list(getattr(schema, "symmetry_pairs", []) or [])
            if _pair_is_valid(a, b)
        ]
        self._rebuild_edges_list_from_schema()
        self._rebuild_symmetry_list_from_schema()
        self._refresh_symmetry_combo_choices()
        self._update_flip_preview()
        self._emit_schema_changed()

    def _remove_selected_keypoints(self) -> None:
        rows = sorted(
            {idx.row() for idx in self.keypoint_list.selectedIndexes()}, reverse=True
        )
        if not rows:
            return
        for row in rows:
            item = self.keypoint_list.takeItem(row)
            if item is not None:
                del item
        self._sync_keypoint_order_from_ui()
        self._prune_locked_keypoints()
        self._refresh_lock_icons()
        has_keypoints = bool(self._keypoint_order)
        self.enable_checkbox.setEnabled(has_keypoints)
        if not has_keypoints:
            self.enable_checkbox.setChecked(False)
        self.reset_sequence()

    def _move_selected_keypoints(self, direction: int) -> None:
        if direction not in (-1, 1):
            return
        selected_rows = sorted(
            {idx.row() for idx in self.keypoint_list.selectedIndexes()}
        )
        if not selected_rows:
            return
        if direction < 0:
            iterator = selected_rows
        else:
            iterator = list(reversed(selected_rows))
        for row in iterator:
            target = row + direction
            if target < 0 or target >= self.keypoint_list.count():
                continue
            item = self.keypoint_list.takeItem(row)
            if item is None:
                continue
            self.keypoint_list.insertItem(target, item)
            item.setSelected(True)
        self._sync_keypoint_order_from_ui()
        self._sync_locked_order_to_keypoint_order()
        self._refresh_lock_icons()
        self.reset_sequence()

    def _prune_locked_keypoints(self) -> None:
        valid = {str(k).strip() for k in self._keypoint_order if str(k).strip()}
        self._locked_keypoint_order = [
            kp for kp in self._locked_keypoint_order if kp in valid
        ]

    def _sync_locked_order_to_keypoint_order(self) -> None:
        self._prune_locked_keypoints()
        locked = set(self._locked_keypoint_order)
        self._locked_keypoint_order = [
            kp for kp in self._keypoint_order if kp in locked
        ]

    def _refresh_lock_icons(self) -> None:
        for i in range(self.keypoint_list.count()):
            item = self.keypoint_list.item(i)
            if item is None:
                continue
            self._update_item_display(item)

    def _toggle_lock_for_item(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is None:
            return
        name = self._item_keypoint_name(item)
        if not name:
            return
        if name in self._locked_keypoint_order:
            self._locked_keypoint_order = [
                kp for kp in self._locked_keypoint_order if kp != name
            ]
        else:
            self._locked_keypoint_order.append(name)
        self._refresh_lock_icons()
        self.reset_sequence()

    def _lock_selected_keypoints(self) -> None:
        selected_rows = sorted(
            {idx.row() for idx in self.keypoint_list.selectedIndexes()}
        )
        if not selected_rows:
            return
        changed = False
        for row in selected_rows:
            item = self.keypoint_list.item(row)
            if item is None:
                continue
            name = self._item_keypoint_name(item)
            if name and name not in self._locked_keypoint_order:
                self._locked_keypoint_order.append(name)
                changed = True
        if changed:
            self._refresh_lock_icons()
            self.reset_sequence()

    def _update_status_text(self) -> None:
        active = self.sequence_active_keypoints()
        if not active:
            self.status_label.setText(
                "No active keypoints selected. Check keypoints to enable sequence labeling."
            )
            return

        idx = self._next_index % len(active)
        next_label = self._compose_label(active[idx])
        mode_text = "Enabled" if self.enable_checkbox.isChecked() else "Disabled"
        edge_text = ""
        if self._pending_edge_start:
            edge_text = f" | Edge start: {self._pending_edge_start}"
        lock_text = ""
        if self._locked_keypoint_order:
            lock_text = f" | Locked: {len(self._locked_keypoint_order)}"
        self.status_label.setText(
            f"{mode_text} | Next label: {next_label} ({idx + 1}/{len(active)}){lock_text}{edge_text}"
        )

    def _emit_schema_changed(self) -> None:
        if self._suspend_schema_signal:
            return
        schema = self._schema
        if schema is None:
            return
        self.poseSchemaChanged.emit(schema, str(self._schema_path or ""))

    def _on_schema_path_edited(self) -> None:
        text = str(self.schema_path_edit.text() or "").strip()
        self._schema_path = text or None

    def _edges_as_pairs(self) -> List[tuple[str, str]]:
        schema = self._schema
        if schema is None:
            return []
        pairs: List[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for raw_a, raw_b in list(getattr(schema, "edges", []) or []):
            _inst_a, a = schema.strip_instance_prefix(str(raw_a or ""))
            _inst_b, b = schema.strip_instance_prefix(str(raw_b or ""))
            a = str(a or "").strip()
            b = str(b or "").strip()
            if not a or not b or a == b:
                continue
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
        return pairs

    def _rebuild_edges_list_from_schema(self) -> None:
        self.edges_list.clear()
        for a, b in self._edges_as_pairs():
            self.edges_list.addItem(f"{a} - {b}")

    def _on_preview_keypoint_clicked(self, name: str) -> None:
        if not name:
            return
        if self._pending_edge_start is None:
            self._pending_edge_start = name
            self._update_status_text()
            return
        first = self._pending_edge_start
        self._pending_edge_start = None
        if first == name:
            self._update_status_text()
            return
        self._toggle_edge(first, name)
        self._update_status_text()

    def _toggle_edge(self, a: str, b: str) -> None:
        schema = self._schema
        if schema is None:
            schema = PoseSchema()
            self._schema = schema
        current = self._edges_as_pairs()
        key = (a, b) if a <= b else (b, a)
        if key in current:
            current = [pair for pair in current if pair != key]
        else:
            current.append(key)
        schema.edges = [(left, right) for left, right in current]
        self._rebuild_edges_list_from_schema()
        self.preview.update()
        self._emit_schema_changed()

    def _remove_selected_edges(self) -> None:
        selected = self.edges_list.selectedItems()
        if not selected:
            return
        remove_keys = set()
        for item in selected:
            text = str(item.text() or "").strip()
            if " - " not in text:
                continue
            left, right = [part.strip() for part in text.split(" - ", 1)]
            if left and right and left != right:
                key = (left, right) if left <= right else (right, left)
                remove_keys.add(key)
        if not remove_keys:
            return
        schema = self._schema
        if schema is None:
            return
        kept = [pair for pair in self._edges_as_pairs() if pair not in remove_keys]
        schema.edges = [(a, b) for a, b in kept]
        self._rebuild_edges_list_from_schema()
        self.preview.update()
        self._emit_schema_changed()

    def _clear_edges(self) -> None:
        schema = self._schema
        if schema is None:
            return
        schema.edges = []
        self._rebuild_edges_list_from_schema()
        self.preview.update()
        self._emit_schema_changed()

    def _symmetry_pairs_as_pairs(self) -> List[Tuple[str, str]]:
        schema = self._schema
        if schema is None:
            return []
        pairs: List[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for raw_a, raw_b in list(getattr(schema, "symmetry_pairs", []) or []):
            _inst_a, a = schema.strip_instance_prefix(str(raw_a or ""))
            _inst_b, b = schema.strip_instance_prefix(str(raw_b or ""))
            a = str(a or "").strip()
            b = str(b or "").strip()
            if not a or not b or a == b:
                continue
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
        return pairs

    def _rebuild_symmetry_list_from_schema(self) -> None:
        self.symmetry_list.clear()
        for a, b in self._symmetry_pairs_as_pairs():
            self.symmetry_list.addItem(f"{a} <-> {b}")

    def _refresh_symmetry_combo_choices(self) -> None:
        keypoints = list(self._keypoint_order)
        for combo in (self.symmetry_a_combo, self.symmetry_b_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(keypoints)
            if current:
                idx = combo.findText(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _set_symmetry_pairs(self, pairs: Sequence[Tuple[str, str]]) -> None:
        schema = self._ensure_schema()
        schema.symmetry_pairs = [(str(a).strip(), str(b).strip()) for a, b in pairs]
        self._rebuild_symmetry_list_from_schema()
        self._update_flip_preview()
        self._emit_schema_changed()

    def _add_symmetry_pair_from_combos(self) -> None:
        a = str(self.symmetry_a_combo.currentText() or "").strip()
        b = str(self.symmetry_b_combo.currentText() or "").strip()
        if not a or not b or a == b:
            return
        current = self._symmetry_pairs_as_pairs()
        key = (a, b) if a <= b else (b, a)
        if key not in current:
            current.append(key)
        self._set_symmetry_pairs(current)

    def _remove_selected_symmetry_pairs(self) -> None:
        selected = self.symmetry_list.selectedItems()
        if not selected:
            return
        remove_keys: set[Tuple[str, str]] = set()
        for item in selected:
            text = str(item.text() or "").strip()
            if " <-> " not in text:
                continue
            left, right = [part.strip() for part in text.split(" <-> ", 1)]
            if left and right and left != right:
                key = (left, right) if left <= right else (right, left)
                remove_keys.add(key)
        if not remove_keys:
            return
        kept = [
            pair for pair in self._symmetry_pairs_as_pairs() if pair not in remove_keys
        ]
        self._set_symmetry_pairs(kept)

    def _clear_symmetry_pairs(self) -> None:
        self._set_symmetry_pairs([])

    def _auto_fill_symmetry_pairs(self) -> None:
        keypoints = list(self._keypoint_order)
        if not keypoints:
            return
        pairs = PoseSchema.infer_symmetry_pairs(keypoints)
        self._set_symmetry_pairs(pairs)

    def _update_flip_preview(self) -> None:
        schema = self._schema
        if schema is None:
            self.flip_preview_label.setText("")
            return
        try:
            flip = schema.compute_flip_idx(schema.expand_keypoints())
        except Exception:
            flip = None
        self.flip_preview_label.setText(str(flip) if flip else "")

    def _browse_schema(self) -> None:
        start = self.schema_path_edit.text().strip() or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select pose schema file",
            start,
            "Pose schema (*.json *.yaml *.yml);;All files (*)",
        )
        if path:
            self.schema_path_edit.setText(path)
            self._schema_path = path

    def _load_schema_clicked(self) -> None:
        path = self.schema_path_edit.text().strip()
        if not path:
            return
        self.load_schema_from_path(path, quiet=False)

    def _save_schema_clicked(self) -> None:
        path = self.schema_path_edit.text().strip() or str(self._schema_path or "")
        if not path:
            self._save_schema_as_clicked()
            return
        self.save_schema_to_path(path, quiet=False)

    def _save_schema_as_clicked(self) -> None:
        start = self.schema_path_edit.text().strip() or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save pose schema",
            start,
            "Pose schema (*.json *.yaml *.yml)",
        )
        if not path:
            return
        self.schema_path_edit.setText(path)
        self._schema_path = path
        self.save_schema_to_path(path, quiet=False)

    def load_schema_from_path(self, path: str, *, quiet: bool = True) -> bool:
        path_text = str(path or "").strip()
        if not path_text:
            return False
        schema_path = Path(path_text).expanduser()
        if not schema_path.exists():
            if not quiet:
                QtWidgets.QMessageBox.warning(
                    self, "Schema not found", f"Schema not found:\n{schema_path}"
                )
            return False
        try:
            schema = PoseSchema.load(schema_path)
        except Exception as exc:
            if not quiet:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Load failed",
                    f"Failed to load schema:\n{schema_path}\n\n{exc}",
                )
            return False
        if self._looks_like_instance_prefixed_schema(schema):
            try:
                schema.normalize_prefixed_keypoints()
            except Exception:
                pass
        self._suspend_schema_signal = True
        try:
            self.set_pose_schema(schema, str(schema_path), emit_change=False)
        finally:
            self._suspend_schema_signal = False
        self._emit_schema_changed()
        return True

    def save_schema_to_path(self, path: str, *, quiet: bool = True) -> bool:
        path_text = str(path or "").strip()
        if not path_text:
            return False
        schema = self._schema or PoseSchema()
        self._schema = schema
        self._sync_schema_keypoints_from_order()
        schema.sequencer = self._capture_sequencer_settings()
        schema_path = Path(path_text).expanduser()
        try:
            schema.save(schema_path)
        except Exception as exc:
            if not quiet:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Save failed",
                    f"Failed to save schema:\n{schema_path}\n\n{exc}",
                )
            return False
        self._schema_path = str(schema_path)
        self.schema_path_edit.setText(self._schema_path)
        self._emit_schema_changed()
        if not quiet:
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Pose schema saved:\n{schema_path}"
            )
        return True
