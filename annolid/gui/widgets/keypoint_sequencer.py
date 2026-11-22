from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from annolid.annotation.pose_schema import (
    PoseSchema,
    dedup_undirected_pairs,
    get_pose_schema_preset,
    list_pose_schema_presets,
)
from annolid.gui.keypoint_catalog import (
    extract_labels_from_uniq_label_list,
    merge_keypoint_lists,
    normalize_keypoint_names,
)
from annolid.gui.qt_compat import configure_compact_combo


class _SkeletonPreviewWidget(QtWidgets.QWidget):
    """Interactive anatomical skeleton preview for edge editing and sequence state."""

    keypointClicked = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._schema: Optional[PoseSchema] = None
        self._active_keypoints: set[str] = set()
        self._pending_edge_start: Optional[str] = None
        self._next_keypoint: Optional[str] = None
        self._hovered_keypoint: Optional[str] = None
        self._pixel_points: dict[str, QtCore.QPointF] = {}
        self._normalized_points: dict[str, QtCore.QPointF] = {}
        self.setMouseTracking(True)
        self.setMinimumHeight(180)

    def set_schema(self, schema: Optional[PoseSchema]) -> None:
        self._schema = schema
        keypoints = list(getattr(schema, "keypoints", []) or [])
        if self._pending_edge_start and self._pending_edge_start not in keypoints:
            self._pending_edge_start = None
        if self._next_keypoint and self._next_keypoint not in keypoints:
            self._next_keypoint = None
        if self._hovered_keypoint and self._hovered_keypoint not in keypoints:
            self._hovered_keypoint = None
        row_count = self._estimate_row_count(keypoints)
        target_height = max(180, min(320, 26 * max(1, row_count) + 34))
        self.setMinimumHeight(target_height)
        self.update()

    def set_active_keypoints(self, keypoints: List[str]) -> None:
        self._active_keypoints = {str(k).strip() for k in keypoints if str(k).strip()}
        self.update()

    def set_pending_edge_start(self, keypoint: Optional[str]) -> None:
        cleaned = str(keypoint or "").strip() or None
        if cleaned != self._pending_edge_start:
            self._pending_edge_start = cleaned
            self.update()

    def set_next_keypoint(self, keypoint: Optional[str]) -> None:
        cleaned = str(keypoint or "").strip() or None
        if cleaned != self._next_keypoint:
            self._next_keypoint = cleaned
            self.update()

    @staticmethod
    def _x_hint(name: str) -> float:
        lowered = name.lower()
        is_left = (
            "left" in lowered
            or lowered.startswith(("l_", "l-", "lt_"))
            or lowered.endswith(("_l", "-l", "_lt"))
        )
        is_right = (
            "right" in lowered
            or lowered.startswith(("r_", "r-", "rt_"))
            or lowered.endswith(("_r", "-r", "_rt"))
        )
        if not is_left and not is_right:
            return 0.50
        # Slight horizontal stagger between torso (shoulder/hip) and outer arm
        # joints (elbow/wrist) so vertical torso bones do not pass behind arms.
        is_outer_arm = any(
            tok in lowered for tok in ("elbow", "wrist", "forearm", "hand", "wing")
        )
        is_inner_torso = any(tok in lowered for tok in ("shoulder", "hip", "eye"))
        if is_outer_arm:
            return 0.34 if is_left else 0.66
        if is_inner_torso:
            return 0.42 if is_left else 0.58
        return 0.38 if is_left else 0.62

    def _symmetry_lookup(self, keypoints: List[str]) -> dict[str, str]:
        partner: dict[str, str] = {}
        valid = set(keypoints)
        pairs: List[Tuple[str, str]] = []
        if self._schema is not None:
            pairs.extend(self._schema.base_symmetry_pairs())
        pairs.extend(PoseSchema.infer_symmetry_pairs(keypoints))
        for a, b in pairs:
            if a in valid and b in valid and a != b:
                if a not in partner and b not in partner:
                    partner[a] = b
                    partner[b] = a
        return partner

    def _build_layout_rows(self, keypoints: List[str]) -> List[List[Tuple[str, float]]]:
        partner = self._symmetry_lookup(keypoints)
        rows: List[List[Tuple[str, float]]] = []
        placed: set[str] = set()
        for kp in keypoints:
            if kp in placed:
                continue
            mate = partner.get(kp)
            if mate and mate not in placed:
                x_kp = self._x_hint(kp)
                x_mate = self._x_hint(mate)
                if x_kp <= x_mate:
                    left_name, right_name = kp, mate
                    left_x = x_kp if x_kp < 0.5 else 0.38
                    right_x = x_mate if x_mate > 0.5 else 0.62
                else:
                    left_name, right_name = mate, kp
                    left_x = x_mate if x_mate < 0.5 else 0.38
                    right_x = x_kp if x_kp > 0.5 else 0.62
                rows.append([(left_name, left_x), (right_name, right_x)])
                placed.add(left_name)
                placed.add(right_name)
            else:
                rows.append([(kp, self._x_hint(kp))])
                placed.add(kp)
        return rows

    def _estimate_row_count(self, keypoints: List[str]) -> int:
        if not keypoints:
            return 0
        return len(self._build_layout_rows(keypoints))

    def _layout_points(self, keypoints: List[str]) -> dict[str, QtCore.QPointF]:
        points: dict[str, QtCore.QPointF] = {}
        if not keypoints:
            return points
        rows = self._build_layout_rows(keypoints)
        n_rows = len(rows)
        for row_idx, row in enumerate(rows):
            y = (row_idx + 1) / (n_rows + 1)
            for kp, x in row:
                points[kp] = QtCore.QPointF(x, y)
        return points

    def _hit_test(self, pos: QtCore.QPoint) -> Optional[str]:
        hit_name: Optional[str] = None
        best_dist = float("inf")
        for name, point in self._pixel_points.items():
            dx = float(point.x() - pos.x())
            dy = float(point.y() - pos.y())
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                hit_name = name
        return hit_name if (hit_name is not None and best_dist <= 13.0) else None

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        pal = self.palette()
        is_dark = pal.color(QtGui.QPalette.Window).lightness() < 128
        bg_color = QtGui.QColor("#1e2430") if is_dark else QtGui.QColor("#f5f7fb")
        border_color = QtGui.QColor("#334155") if is_dark else QtGui.QColor("#cbd5e1")
        text_color = QtGui.QColor("#e2e8f0") if is_dark else QtGui.QColor("#0f172a")
        muted_color = QtGui.QColor("#94a3b8") if is_dark else QtGui.QColor("#64748b")
        edge_color = QtGui.QColor("#38bdf8") if is_dark else QtGui.QColor("#0284c7")
        sym_color = QtGui.QColor("#a78bfa") if is_dark else QtGui.QColor("#8b5cf6")

        rect = self.rect().adjusted(6, 6, -6, -6)
        painter.setPen(QtGui.QPen(border_color, 1))
        painter.setBrush(bg_color)
        painter.drawRoundedRect(rect, 6, 6)

        schema = self._schema
        keypoints = list(getattr(schema, "keypoints", []) or [])
        if not keypoints:
            self._pixel_points = {}
            self._normalized_points = {}
            painter.setPen(muted_color)
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No pose schema loaded")
            return

        inner = rect.adjusted(10, 10, -10, -10)
        points = self._layout_points(keypoints)
        self._normalized_points = points
        pixel_points: dict[str, QtCore.QPointF] = {}
        for kp, p in points.items():
            px = inner.left() + p.x() * inner.width()
            py = inner.top() + p.y() * inner.height()
            pixel_points[kp] = QtCore.QPointF(px, py)
        self._pixel_points = pixel_points

        # 1. Draw symmetry pairs subtly beneath skeleton edges.
        sym_pairs = schema.base_symmetry_pairs() if schema is not None else []
        if sym_pairs:
            sym_pen = QtGui.QPen(sym_color)
            sym_pen.setWidthF(1.2)
            sym_pen.setStyle(QtCore.Qt.DotLine)
            painter.setPen(sym_pen)
            for a, b in sym_pairs:
                p1 = pixel_points.get(a)
                p2 = pixel_points.get(b)
                if p1 is not None and p2 is not None:
                    painter.drawLine(p1, p2)

        # 2. Draw skeleton edges (emphasize edges incident to pending/hovered keypoint).
        edges = schema.base_edges() if schema is not None else []
        for a, b in edges:
            p1 = pixel_points.get(a)
            p2 = pixel_points.get(b)
            if p1 is None or p2 is None:
                continue
            emphasized = a in {
                self._pending_edge_start,
                self._hovered_keypoint,
            } or b in {self._pending_edge_start, self._hovered_keypoint}
            edge_pen = QtGui.QPen(QtGui.QColor("#f59e0b") if emphasized else edge_color)
            edge_pen.setWidthF(2.6 if emphasized else 2.0)
            edge_pen.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(edge_pen)
            painter.drawLine(p1, p2)

        # 3. Draw keypoint nodes and non-clipping labels.
        font = painter.font()
        font.setPointSize(
            max(8, min(10, font.pointSize() if font.pointSize() > 0 else 9))
        )
        painter.setFont(font)
        fm = painter.fontMetrics()

        for kp in keypoints:
            point = pixel_points.get(kp)
            norm_pt = points.get(kp)
            if point is None or norm_pt is None:
                continue

            is_pending = kp == self._pending_edge_start
            is_next = kp == self._next_keypoint
            is_hovered = kp == self._hovered_keypoint
            is_active = kp in self._active_keypoints

            if is_pending:
                ring_pen = QtGui.QPen(QtGui.QColor("#f59e0b"), 2.0, QtCore.Qt.DashLine)
                painter.setPen(ring_pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(point, 9.5, 9.5)
            elif is_next:
                ring_pen = QtGui.QPen(QtGui.QColor("#10b981"), 2.0)
                painter.setPen(ring_pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(point, 9.0, 9.0)
            elif is_hovered:
                ring_pen = QtGui.QPen(edge_color, 1.5)
                painter.setPen(ring_pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(point, 8.5, 8.5)

            if is_pending:
                node_fill = QtGui.QColor("#f59e0b")
            elif is_next:
                node_fill = QtGui.QColor("#10b981")
            elif is_active:
                node_fill = QtGui.QColor("#0ea5e9")
            else:
                node_fill = QtGui.QColor("#64748b" if is_dark else "#9ca3af")

            radius = 5.5 if (is_pending or is_hovered) else 4.8
            painter.setBrush(node_fill)
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor("#f8fafc" if is_dark else "#1e293b"),
                    1.2,
                )
            )
            painter.drawEllipse(point, radius, radius)

            painter.setPen(text_color)
            if norm_pt.x() < 0.48:
                avail_w = max(24.0, point.x() - rect.left() - 16.0)
                elided = fm.elidedText(kp, QtCore.Qt.ElideRight, int(avail_w))
                label_rect = QtCore.QRectF(
                    rect.left() + 4.0, point.y() - 9.0, avail_w, 18.0
                )
                painter.drawText(
                    label_rect, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, elided
                )
            else:
                label_offset = 12.0 if (is_pending or is_next or is_hovered) else 9.0
                avail_w = max(24.0, rect.right() - point.x() - label_offset - 4.0)
                elided = fm.elidedText(kp, QtCore.Qt.ElideRight, int(avail_w))
                label_rect = QtCore.QRectF(
                    point.x() + label_offset, point.y() - 9.0, avail_w, 18.0
                )
                painter.drawText(
                    label_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, elided
                )

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        hit = self._hit_test(event.pos())
        if hit != self._hovered_keypoint:
            self._hovered_keypoint = hit
            if hit is not None:
                self.setCursor(QtCore.Qt.PointingHandCursor)
                if self._pending_edge_start and self._pending_edge_start != hit:
                    self.setToolTip(
                        f"{hit} — click to toggle edge with '{self._pending_edge_start}'"
                    )
                else:
                    self.setToolTip(f"{hit} — click to start/finish a skeleton edge")
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
                self.setToolTip("")
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hovered_keypoint is not None:
            self._hovered_keypoint = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            self.setToolTip("")
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        hit_name = self._hit_test(event.pos())
        if hit_name is not None:
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
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget(scroll)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        schema_path_row = QtWidgets.QHBoxLayout()
        schema_path_row.setContentsMargins(0, 0, 0, 0)
        schema_path_row.setSpacing(4)
        schema_path_row.addWidget(QtWidgets.QLabel("Schema:"))
        self.schema_path_edit = QtWidgets.QLineEdit(self)
        self.schema_path_edit.setMinimumWidth(60)
        self.schema_path_edit.setPlaceholderText("pose_schema.json / .yaml")
        self.schema_path_edit.editingFinished.connect(self._on_schema_path_edited)
        schema_path_row.addWidget(self.schema_path_edit, 1)
        layout.addLayout(schema_path_row)

        schema_btn_row = QtWidgets.QHBoxLayout()
        schema_btn_row.setContentsMargins(0, 0, 0, 0)
        schema_btn_row.setSpacing(4)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_schema)
        load_btn = QtWidgets.QPushButton("Load")
        load_btn.clicked.connect(self._load_schema_clicked)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self._save_schema_clicked)
        save_as_btn = QtWidgets.QPushButton("Save As…")
        save_as_btn.clicked.connect(self._save_schema_as_clicked)
        for btn in (browse_btn, load_btn, save_btn, save_as_btn):
            btn.setMinimumWidth(42)
            schema_btn_row.addWidget(btn)
        layout.addLayout(schema_btn_row)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(4)
        preset_row.addWidget(QtWidgets.QLabel("Preset:"))
        self.preset_combo = QtWidgets.QComboBox(self)
        configure_compact_combo(self.preset_combo, min_chars=10, popup_min_width=210)
        self.preset_combo.addItem("Select preset…", None)
        for preset_name in list_pose_schema_presets():
            self.preset_combo.addItem(preset_name, preset_name)
        apply_preset_btn = QtWidgets.QPushButton("Apply")
        apply_preset_btn.setToolTip(
            "Apply built-in skeleton preset (keypoints, edges, and symmetry pairs)"
        )
        apply_preset_btn.clicked.connect(self._apply_preset_clicked)
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(apply_preset_btn)
        layout.addLayout(preset_row)

        prefix_row = QtWidgets.QHBoxLayout()
        prefix_row.setContentsMargins(0, 0, 0, 0)
        prefix_row.setSpacing(4)
        prefix_row.addWidget(QtWidgets.QLabel("Prefixes:"))
        self.instance_prefixes_edit = QtWidgets.QLineEdit(self)
        self.instance_prefixes_edit.setMinimumWidth(60)
        self.instance_prefixes_edit.setPlaceholderText("e.g. mouse_1, mouse_2")
        self.instance_prefixes_edit.editingFinished.connect(
            self._on_instance_config_changed
        )
        prefix_row.addWidget(self.instance_prefixes_edit, 1)
        layout.addLayout(prefix_row)

        sep_row = QtWidgets.QHBoxLayout()
        sep_row.setContentsMargins(0, 0, 0, 0)
        sep_row.setSpacing(4)
        sep_row.addWidget(QtWidgets.QLabel("Sep:"))
        self.instance_separator_edit = QtWidgets.QLineEdit(self)
        self.instance_separator_edit.setFixedWidth(44)
        self.instance_separator_edit.setText("_")
        self.instance_separator_edit.editingFinished.connect(
            self._on_instance_config_changed
        )
        sep_row.addWidget(self.instance_separator_edit)
        self.normalize_prefixes_btn = QtWidgets.QPushButton("Normalize prefixes")
        self.normalize_prefixes_btn.setMinimumWidth(60)
        self.normalize_prefixes_btn.clicked.connect(self._normalize_prefixed_schema)
        sep_row.addWidget(self.normalize_prefixes_btn, 1)
        layout.addLayout(sep_row)

        self.enable_checkbox = QtWidgets.QCheckBox("Enable click-to-label sequence")
        self.enable_checkbox.setChecked(False)
        self.enable_checkbox.toggled.connect(self._on_sequence_mode_toggled)
        layout.addWidget(self.enable_checkbox)

        self.autosave_checkbox = QtWidgets.QCheckBox("Auto-save after point click")
        self.autosave_checkbox.setChecked(True)
        layout.addWidget(self.autosave_checkbox)

        instance_row = QtWidgets.QHBoxLayout()
        instance_row.setContentsMargins(0, 0, 0, 0)
        instance_row.setSpacing(4)
        instance_row.addWidget(QtWidgets.QLabel("Instance:"))
        self.instance_combo = QtWidgets.QComboBox()
        configure_compact_combo(self.instance_combo, min_chars=10, popup_min_width=200)
        self.instance_combo.currentIndexChanged.connect(self.reset_sequence)
        instance_row.addWidget(self.instance_combo, 1)
        layout.addLayout(instance_row)

        self.preview = _SkeletonPreviewWidget(self)
        self.preview.keypointClicked.connect(self._on_preview_keypoint_clicked)
        layout.addWidget(self.preview)

        self.edge_hint_label = QtWidgets.QLabel(
            "Edge editor: click two keypoints in preview to add/remove an edge."
        )
        self.edge_hint_label.setWordWrap(True)
        layout.addWidget(self.edge_hint_label)

        self.edges_list = QtWidgets.QListWidget(self)
        self.edges_list.setMinimumHeight(64)
        self.edges_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.edges_list)

        edge_add_row = QtWidgets.QHBoxLayout()
        edge_add_row.setContentsMargins(0, 0, 0, 0)
        edge_add_row.setSpacing(4)
        self.edge_a_combo = QtWidgets.QComboBox(self)
        self.edge_b_combo = QtWidgets.QComboBox(self)
        configure_compact_combo(self.edge_a_combo, min_chars=6, popup_min_width=180)
        configure_compact_combo(self.edge_b_combo, min_chars=6, popup_min_width=180)
        add_edge_btn = QtWidgets.QPushButton("Add")
        add_edge_btn.setToolTip("Add skeleton edge between selected keypoints")
        add_edge_btn.clicked.connect(self._add_edge_from_combos)
        edge_add_row.addWidget(self.edge_a_combo, 1)
        edge_add_row.addWidget(self.edge_b_combo, 1)
        edge_add_row.addWidget(add_edge_btn)
        layout.addLayout(edge_add_row)

        edge_controls = QtWidgets.QHBoxLayout()
        edge_controls.setContentsMargins(0, 0, 0, 0)
        edge_controls.setSpacing(4)
        remove_edge_btn = QtWidgets.QPushButton("Remove")
        remove_edge_btn.setToolTip("Remove selected skeleton edge(s)")
        clear_edge_btn = QtWidgets.QPushButton("Clear")
        clear_edge_btn.setToolTip("Clear all skeleton edges")
        auto_edge_btn = QtWidgets.QPushButton("Auto")
        auto_edge_btn.setToolTip("Auto-infer anatomical skeleton edges from keypoints")
        remove_edge_btn.clicked.connect(self._remove_selected_edges)
        clear_edge_btn.clicked.connect(self._clear_edges)
        auto_edge_btn.clicked.connect(self._auto_fill_edges)
        for btn in (remove_edge_btn, clear_edge_btn, auto_edge_btn):
            btn.setMinimumWidth(46)
            edge_controls.addWidget(btn, 1)
        layout.addLayout(edge_controls)

        symmetry_label = QtWidgets.QLabel("Symmetry pairs (left <-> right):")
        layout.addWidget(symmetry_label)
        self.symmetry_list = QtWidgets.QListWidget(self)
        self.symmetry_list.setMinimumHeight(64)
        self.symmetry_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        layout.addWidget(self.symmetry_list)

        symmetry_add_row = QtWidgets.QHBoxLayout()
        symmetry_add_row.setContentsMargins(0, 0, 0, 0)
        symmetry_add_row.setSpacing(4)
        self.symmetry_a_combo = QtWidgets.QComboBox(self)
        self.symmetry_b_combo = QtWidgets.QComboBox(self)
        configure_compact_combo(self.symmetry_a_combo, min_chars=6, popup_min_width=180)
        configure_compact_combo(self.symmetry_b_combo, min_chars=6, popup_min_width=180)
        add_pair_btn = QtWidgets.QPushButton("Add")
        add_pair_btn.setToolTip("Add selected left/right symmetry pair")
        add_pair_btn.clicked.connect(self._add_symmetry_pair_from_combos)
        symmetry_add_row.addWidget(self.symmetry_a_combo, 1)
        symmetry_add_row.addWidget(self.symmetry_b_combo, 1)
        symmetry_add_row.addWidget(add_pair_btn)
        layout.addLayout(symmetry_add_row)

        symmetry_controls = QtWidgets.QHBoxLayout()
        symmetry_controls.setContentsMargins(0, 0, 0, 0)
        symmetry_controls.setSpacing(4)
        remove_pair_btn = QtWidgets.QPushButton("Remove")
        remove_pair_btn.setToolTip("Remove selected symmetry pair(s)")
        remove_pair_btn.clicked.connect(self._remove_selected_symmetry_pairs)
        clear_pairs_btn = QtWidgets.QPushButton("Clear")
        clear_pairs_btn.setToolTip("Clear all symmetry pairs")
        clear_pairs_btn.clicked.connect(self._clear_symmetry_pairs)
        auto_pairs_btn = QtWidgets.QPushButton("Auto")
        auto_pairs_btn.setToolTip("Auto-detect left/right symmetry pairs")
        auto_pairs_btn.clicked.connect(self._auto_fill_symmetry_pairs)
        for btn in (remove_pair_btn, clear_pairs_btn, auto_pairs_btn):
            btn.setMinimumWidth(46)
            symmetry_controls.addWidget(btn, 1)
        layout.addLayout(symmetry_controls)

        flip_row = QtWidgets.QHBoxLayout()
        flip_row.setContentsMargins(0, 0, 0, 0)
        flip_row.setSpacing(4)
        flip_row.addWidget(QtWidgets.QLabel("flip_idx:"))
        self.flip_preview_label = QtWidgets.QLabel("")
        self.flip_preview_label.setWordWrap(True)
        flip_row.addWidget(self.flip_preview_label, 1)
        layout.addLayout(flip_row)

        layout.addWidget(QtWidgets.QLabel("Active keypoints (checked = in sequence):"))
        self.keypoint_list = _LockableKeypointListWidget(self)
        self.keypoint_list.setMinimumHeight(96)
        self.keypoint_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.keypoint_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.keypoint_list.model().rowsMoved.connect(self._on_keypoint_order_changed)
        self.keypoint_list.itemChanged.connect(self._on_keypoint_checks_changed)
        self.keypoint_list.lockToggleRequested.connect(self._toggle_lock_for_item)
        layout.addWidget(self.keypoint_list, 1)
        self.keypoint_lock_hint = QtWidgets.QLabel(
            "Tip: click left of a keypoint to lock/unlock its sequence order."
        )
        self.keypoint_lock_hint.setWordWrap(True)
        layout.addWidget(self.keypoint_lock_hint)

        add_row = QtWidgets.QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        add_row.setSpacing(4)
        self.add_keypoint_edit = QtWidgets.QLineEdit(self)
        self.add_keypoint_edit.setMinimumWidth(60)
        self.add_keypoint_edit.setPlaceholderText("Add keypoint (e.g. nose, tail_base)")
        add_keypoint_btn = QtWidgets.QPushButton("Add")
        add_keypoint_btn.clicked.connect(self._add_keypoint_from_input)
        self.load_from_labels_btn = QtWidgets.QPushButton("From Labels")
        self.load_from_labels_btn.setToolTip("Import keypoints from the Labels dock")
        self.load_from_labels_btn.clicked.connect(self._load_from_labels_dock)
        self.add_keypoint_edit.returnPressed.connect(self._add_keypoint_from_input)
        add_row.addWidget(self.add_keypoint_edit, 1)
        add_row.addWidget(add_keypoint_btn)
        add_row.addWidget(self.load_from_labels_btn)
        layout.addLayout(add_row)

        controls_grid = QtWidgets.QGridLayout()
        controls_grid.setContentsMargins(0, 0, 0, 0)
        controls_grid.setHorizontalSpacing(4)
        controls_grid.setVerticalSpacing(4)
        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.setToolTip("Remove selected keypoint(s)")
        remove_btn.clicked.connect(self._remove_selected_keypoints)
        move_up_btn = QtWidgets.QPushButton("Up")
        move_up_btn.setToolTip("Move selected keypoint(s) up")
        move_up_btn.clicked.connect(lambda: self._move_selected_keypoints(-1))
        move_down_btn = QtWidgets.QPushButton("Down")
        move_down_btn.setToolTip("Move selected keypoint(s) down")
        move_down_btn.clicked.connect(lambda: self._move_selected_keypoints(1))
        select_all_btn = QtWidgets.QPushButton("Select all")
        clear_btn = QtWidgets.QPushButton("Clear")
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.setToolTip("Reset sequence cursor to the first active keypoint")
        select_all_btn.clicked.connect(self._select_all_keypoints)
        clear_btn.clicked.connect(self._clear_all_keypoints)
        reset_btn.clicked.connect(self.reset_sequence)
        for idx, btn in enumerate(
            (
                remove_btn,
                move_up_btn,
                move_down_btn,
                select_all_btn,
                clear_btn,
                reset_btn,
            )
        ):
            btn.setMinimumWidth(46)
            controls_grid.addWidget(btn, idx // 3, idx % 3)
        layout.addLayout(controls_grid)

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
        replace_order: bool = False,
    ) -> None:
        previous_instance = self.instance_combo.currentData()
        previous_active = self.active_keypoints()
        self._schema = schema
        self._pending_edge_start = None
        self.preview.set_pending_edge_start(None)
        if schema_path:
            self._schema_path = str(schema_path)
            self.schema_path_edit.setText(self._schema_path)
        incoming_kps = list(getattr(schema, "keypoints", []) or [])
        if replace_order:
            self._keypoint_order = normalize_keypoint_names(incoming_kps)
            self._prune_locked_keypoints()
        else:
            self._keypoint_order = merge_keypoint_lists(
                incoming_kps, self._keypoint_order
            )
        self._sync_schema_keypoints_from_order()
        self._rebuild_instance_options(previous_instance=previous_instance)
        self._refresh_instance_config_from_schema()
        self._rebuild_keypoint_list()
        if previous_active and not replace_order:
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

    def apply_preset(self, preset_name: str) -> bool:
        """Apply a named built-in skeleton preset while preserving instance settings."""
        name = str(preset_name or "").strip()
        if not name:
            return False
        try:
            preset_schema = get_pose_schema_preset(name)
        except KeyError:
            return False
        if self._schema is not None:
            preset_schema.instances = list(self._schema.instances or [])
            preset_schema.instance_separator = str(
                getattr(self._schema, "instance_separator", "_") or "_"
            )
        self._suspend_schema_signal = True
        try:
            self.set_pose_schema(
                preset_schema,
                self._schema_path,
                emit_change=False,
                replace_order=True,
            )
            idx = self.preset_combo.findData(name)
            if idx >= 0:
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentIndex(idx)
                self.preset_combo.blockSignals(False)
        finally:
            self._suspend_schema_signal = False
        self._emit_schema_changed()
        return True

    def _apply_preset_clicked(self) -> None:
        preset_name = self.preset_combo.currentData()
        if preset_name:
            self.apply_preset(str(preset_name))

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
        if self._pending_edge_start and self._pending_edge_start not in valid:
            self._pending_edge_start = None
            self.preview.set_pending_edge_start(None)
        self._rebuild_edges_list_from_schema()
        self._rebuild_symmetry_list_from_schema()
        self._refresh_symmetry_combo_choices()
        self.preview.set_schema(schema)
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
        edge_text = (
            f" | Edge start: {self._pending_edge_start}"
            if self._pending_edge_start
            else ""
        )
        if not active:
            self.preview.set_next_keypoint(None)
            self.status_label.setText(
                f"Lock keypoints (🔒) to enable sequence labeling.{edge_text}"
            )
            return

        idx = self._next_index % len(active)
        next_kp = active[idx]
        self.preview.set_next_keypoint(
            next_kp if self.enable_checkbox.isChecked() else None
        )
        next_label = self._compose_label(next_kp)
        mode_text = "Enabled" if self.enable_checkbox.isChecked() else "Disabled"
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
        return schema.base_edges()

    def _rebuild_edges_list_from_schema(self) -> None:
        self.edges_list.clear()
        for a, b in self._edges_as_pairs():
            item = QtWidgets.QListWidgetItem(f"{a} - {b}")
            item.setData(QtCore.Qt.UserRole, (a, b))
            self.edges_list.addItem(item)

    def _on_preview_keypoint_clicked(self, name: str) -> None:
        if not name:
            return
        if self._pending_edge_start is None:
            self._pending_edge_start = name
            self.preview.set_pending_edge_start(name)
            self._update_status_text()
            return
        first = self._pending_edge_start
        self._pending_edge_start = None
        self.preview.set_pending_edge_start(None)
        if first == name:
            self._update_status_text()
            return
        self._toggle_edge(first, name)
        self._update_status_text()

    def _toggle_edge(self, a: str, b: str) -> None:
        schema = self._ensure_schema()
        current = self._edges_as_pairs()
        target_ukey = (a, b) if a < b else (b, a)
        matched = [
            pair
            for pair in current
            if ((pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0]))
            == target_ukey
        ]
        if matched:
            current = [
                pair
                for pair in current
                if ((pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0]))
                != target_ukey
            ]
        else:
            current.append((a, b))
        schema.edges = dedup_undirected_pairs(current)
        self._rebuild_edges_list_from_schema()
        self.preview.set_schema(schema)
        self._emit_schema_changed()

    def _add_edge_from_combos(self) -> None:
        a = str(self.edge_a_combo.currentText() or "").strip()
        b = str(self.edge_b_combo.currentText() or "").strip()
        if not a or not b or a == b:
            return
        schema = self._ensure_schema()
        current = self._edges_as_pairs()
        target_ukey = (a, b) if a < b else (b, a)
        existing_ukeys = {
            (left, right) if left < right else (right, left) for left, right in current
        }
        if target_ukey not in existing_ukeys:
            current.append((a, b))
        schema.edges = dedup_undirected_pairs(current)
        self._rebuild_edges_list_from_schema()
        self.preview.set_schema(schema)
        self._emit_schema_changed()

    def _auto_fill_edges(self) -> None:
        keypoints = list(self._keypoint_order)
        if len(keypoints) < 2:
            return
        schema = self._ensure_schema()
        inferred = PoseSchema.infer_edges(keypoints, preset="auto")
        if inferred:
            schema.edges = dedup_undirected_pairs(inferred)
            self._pending_edge_start = None
            self.preview.set_pending_edge_start(None)
            self._rebuild_edges_list_from_schema()
            self.preview.set_schema(schema)
            self._emit_schema_changed()

    def _remove_selected_edges(self) -> None:
        selected = self.edges_list.selectedItems()
        if not selected:
            return
        remove_ukeys: set[tuple[str, str]] = set()
        for item in selected:
            stored = item.data(QtCore.Qt.UserRole)
            if isinstance(stored, (tuple, list)) and len(stored) == 2:
                left, right = str(stored[0]).strip(), str(stored[1]).strip()
            else:
                text = str(item.text() or "").strip()
                if " - " not in text:
                    continue
                left, right = [part.strip() for part in text.split(" - ", 1)]
            if left and right and left != right:
                remove_ukeys.add((left, right) if left < right else (right, left))
        if not remove_ukeys:
            return
        schema = self._schema
        if schema is None:
            return
        kept = [
            pair
            for pair in self._edges_as_pairs()
            if ((pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0]))
            not in remove_ukeys
        ]
        schema.edges = dedup_undirected_pairs(kept)
        self._rebuild_edges_list_from_schema()
        self.preview.set_schema(schema)
        self._emit_schema_changed()

    def _clear_edges(self) -> None:
        schema = self._schema
        if schema is None:
            return
        schema.edges = []
        self._pending_edge_start = None
        self.preview.set_pending_edge_start(None)
        self._rebuild_edges_list_from_schema()
        self.preview.set_schema(schema)
        self._update_status_text()
        self._emit_schema_changed()

    def _symmetry_pairs_as_pairs(self) -> List[Tuple[str, str]]:
        schema = self._schema
        if schema is None:
            return []
        return schema.base_symmetry_pairs()

    def _rebuild_symmetry_list_from_schema(self) -> None:
        self.symmetry_list.clear()
        for a, b in self._symmetry_pairs_as_pairs():
            item = QtWidgets.QListWidgetItem(f"{a} <-> {b}")
            item.setData(QtCore.Qt.UserRole, (a, b))
            self.symmetry_list.addItem(item)

    def _refresh_symmetry_combo_choices(self) -> None:
        keypoints = list(self._keypoint_order)
        for combo in (
            self.edge_a_combo,
            self.edge_b_combo,
            self.symmetry_a_combo,
            self.symmetry_b_combo,
        ):
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
        schema.symmetry_pairs = dedup_undirected_pairs(pairs)
        self._rebuild_symmetry_list_from_schema()
        self.preview.set_schema(schema)
        self._update_flip_preview()
        self._emit_schema_changed()

    def _add_symmetry_pair_from_combos(self) -> None:
        a = str(self.symmetry_a_combo.currentText() or "").strip()
        b = str(self.symmetry_b_combo.currentText() or "").strip()
        if not a or not b or a == b:
            return
        current = self._symmetry_pairs_as_pairs()
        ukey = (a, b) if a < b else (b, a)
        existing = {
            (left, right) if left < right else (right, left) for left, right in current
        }
        if ukey not in existing:
            current.append((a, b))
        self._set_symmetry_pairs(current)

    def _remove_selected_symmetry_pairs(self) -> None:
        selected = self.symmetry_list.selectedItems()
        if not selected:
            return
        remove_ukeys: set[Tuple[str, str]] = set()
        for item in selected:
            stored = item.data(QtCore.Qt.UserRole)
            if isinstance(stored, (tuple, list)) and len(stored) == 2:
                left, right = str(stored[0]).strip(), str(stored[1]).strip()
            else:
                text = str(item.text() or "").strip()
                if " <-> " not in text:
                    continue
                left, right = [part.strip() for part in text.split(" <-> ", 1)]
            if left and right and left != right:
                remove_ukeys.add((left, right) if left < right else (right, left))
        if not remove_ukeys:
            return
        kept = [
            pair
            for pair in self._symmetry_pairs_as_pairs()
            if ((pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0]))
            not in remove_ukeys
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
            self.set_pose_schema(
                schema,
                str(schema_path),
                emit_change=False,
                replace_order=bool(schema.keypoints),
            )
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
