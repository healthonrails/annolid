from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from qtpy import QtCore, QtWidgets

from annolid.annotation.pose_schema import PoseSchema
from annolid.utils.logger import logger


class PoseSchemaDialog(QtWidgets.QDialog):
    """Edit a PoseSchema (keypoint order, symmetry pairs, and edges)."""

    def __init__(
        self,
        *,
        keypoints: Optional[List[str]] = None,
        schema: Optional[PoseSchema] = None,
        schema_path: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle("Pose Schema (Keypoints)")
        self.setModal(True)
        self.resize(760, 520)

        self._schema = schema if schema is not None else PoseSchema()
        if keypoints:
            self._schema.keypoints = list(dict.fromkeys(
                [kp.strip() for kp in keypoints if kp and kp.strip()]))
        self._schema_path = schema_path

        self._build_ui()

        if schema_path:
            try:
                self._load_schema(Path(schema_path))
            except Exception:
                logger.debug("Failed to preload pose schema.", exc_info=True)

        self._refresh_from_schema()

    @property
    def schema(self) -> PoseSchema:
        return self._schema

    @property
    def schema_path(self) -> Optional[str]:
        return self._schema_path

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # File row
        file_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("pose_schema.json (or .yaml)")
        if self._schema_path:
            self.path_edit.setText(self._schema_path)
        browse_btn = QtWidgets.QPushButton("Browse…")
        load_btn = QtWidgets.QPushButton("Load")
        save_btn = QtWidgets.QPushButton("Save")
        save_as_btn = QtWidgets.QPushButton("Save As…")

        browse_btn.clicked.connect(self._browse_schema)
        load_btn.clicked.connect(self._on_load_clicked)
        save_btn.clicked.connect(self._on_save_clicked)
        save_as_btn.clicked.connect(self._on_save_as_clicked)

        file_row.addWidget(QtWidgets.QLabel("Schema file:"))
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(browse_btn)
        file_row.addWidget(load_btn)
        file_row.addWidget(save_btn)
        file_row.addWidget(save_as_btn)
        layout.addLayout(file_row)

        # Instance prefix support (optional)
        instance_row = QtWidgets.QHBoxLayout()
        self.instance_edit = QtWidgets.QLineEdit()
        self.instance_edit.setPlaceholderText(
            "Instances/prefixes (comma separated), e.g. intruder,resident"
        )
        self.instance_edit.textChanged.connect(self._update_flip_preview)

        self.separator_edit = QtWidgets.QLineEdit()
        self.separator_edit.setFixedWidth(60)
        self.separator_edit.setPlaceholderText("_")
        self.separator_edit.setText(
            getattr(self._schema, "instance_separator", "_") or "_")
        self.separator_edit.textChanged.connect(self._update_flip_preview)

        normalize_btn = QtWidgets.QPushButton("Normalize prefixes")
        normalize_btn.clicked.connect(self._normalize_prefixed_schema)

        instance_row.addWidget(QtWidgets.QLabel("Instance prefixes:"))
        instance_row.addWidget(self.instance_edit, 1)
        instance_row.addWidget(QtWidgets.QLabel("Separator:"))
        instance_row.addWidget(self.separator_edit)
        instance_row.addWidget(normalize_btn)
        layout.addLayout(instance_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left panel: keypoints
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        kp_header = QtWidgets.QHBoxLayout()
        kp_header.addWidget(QtWidgets.QLabel("Keypoints (order matters)"))
        kp_header.addStretch(1)
        self.auto_pairs_btn = QtWidgets.QPushButton("Auto symmetry")
        self.auto_pairs_btn.clicked.connect(self._auto_fill_symmetry)
        kp_header.addWidget(self.auto_pairs_btn)
        left_layout.addLayout(kp_header)

        self.kp_list = QtWidgets.QListWidget()
        self.kp_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.kp_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.kp_list.model().rowsMoved.connect(self._on_keypoints_changed)
        self.kp_list.model().rowsInserted.connect(self._on_keypoints_changed)
        self.kp_list.model().rowsRemoved.connect(self._on_keypoints_changed)
        left_layout.addWidget(self.kp_list, 1)

        kp_controls = QtWidgets.QHBoxLayout()
        self.kp_add_edit = QtWidgets.QLineEdit()
        self.kp_add_edit.setPlaceholderText("Add keypoint (e.g. left_ear)")
        kp_add_btn = QtWidgets.QPushButton("Add")
        kp_remove_btn = QtWidgets.QPushButton("Remove")
        kp_add_btn.clicked.connect(self._add_keypoint)
        kp_remove_btn.clicked.connect(self._remove_selected_keypoints)
        kp_controls.addWidget(self.kp_add_edit, 1)
        kp_controls.addWidget(kp_add_btn)
        kp_controls.addWidget(kp_remove_btn)
        left_layout.addLayout(kp_controls)

        splitter.addWidget(left)

        # Right panel: pairs and edges
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        self.sym_table = self._build_pair_table(
            "Symmetry pairs (left ↔ right)")
        sym_controls = self._build_table_controls(
            self.sym_table, add_text="Add pair", remove_text="Remove pair")
        right_layout.addWidget(self.sym_table, 1)
        right_layout.addLayout(sym_controls)

        self.edge_table = self._build_pair_table("Edges (optional)")
        edge_controls = self._build_table_controls(
            self.edge_table, add_text="Add edge", remove_text="Remove edge")
        right_layout.addWidget(self.edge_table, 1)
        right_layout.addLayout(edge_controls)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        # Preview row
        preview = QtWidgets.QHBoxLayout()
        self.flip_preview = QtWidgets.QLabel("")
        self.flip_preview.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        preview.addWidget(QtWidgets.QLabel("flip_idx preview:"))
        preview.addWidget(self.flip_preview, 1)
        layout.addLayout(preview)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    @staticmethod
    def _build_pair_table(title: str) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        table = QtWidgets.QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["A", "B"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(table)
        box._table = table  # type: ignore[attr-defined]
        return box

    def _build_table_controls(
        self,
        box: QtWidgets.QGroupBox,
        *,
        add_text: str,
        remove_text: str,
    ) -> QtWidgets.QHBoxLayout:
        # type: ignore[attr-defined]
        table: QtWidgets.QTableWidget = box._table
        controls = QtWidgets.QHBoxLayout()
        controls.addStretch(1)
        add_btn = QtWidgets.QPushButton(add_text)
        rm_btn = QtWidgets.QPushButton(remove_text)
        add_btn.clicked.connect(lambda: self._add_pair_row(table))
        rm_btn.clicked.connect(lambda: self._remove_selected_row(table))
        controls.addWidget(add_btn)
        controls.addWidget(rm_btn)
        return controls

    def _browse_schema(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select pose schema file",
            self.path_edit.text().strip() or str(Path.home()),
            "Pose schema (*.json *.yaml *.yml);;All files (*)",
        )
        if path:
            self.path_edit.setText(path)

    def _on_load_clicked(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            return
        self._load_schema(Path(path))
        self._refresh_from_schema()

    def _on_save_clicked(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            self._on_save_as_clicked()
            return
        self._sync_schema_from_ui()
        self._schema.save(path)
        self._schema_path = path
        QtWidgets.QMessageBox.information(
            self, "Saved", f"Pose schema saved to:\n{path}")

    def _on_save_as_clicked(self) -> None:
        start_dir = self.path_edit.text().strip() or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save pose schema",
            start_dir,
            "Pose schema (*.json *.yaml *.yml)",
        )
        if not path:
            return
        self.path_edit.setText(path)
        self._on_save_clicked()

    def _load_schema(self, path: Path) -> None:
        schema = PoseSchema.load(path)
        self._schema = schema
        self._schema_path = str(path)
        self.path_edit.setText(str(path))

    def _refresh_from_schema(self) -> None:
        self.instance_edit.setText(
            ", ".join(self._schema.instances) if getattr(
                self._schema, "instances", None) else ""
        )
        self.separator_edit.setText(
            getattr(self._schema, "instance_separator", "_") or "_")
        self._set_keypoints(self._schema.keypoints)
        # type: ignore[attr-defined]
        self._set_pair_table(self.sym_table._table,
                             self._schema.symmetry_pairs)
        # type: ignore[attr-defined]
        self._set_pair_table(self.edge_table._table, self._schema.edges)
        self._update_flip_preview()

    def _set_keypoints(self, keypoints: List[str]) -> None:
        self.kp_list.blockSignals(True)
        try:
            self.kp_list.clear()
            for kp in keypoints:
                self.kp_list.addItem(kp)
        finally:
            self.kp_list.blockSignals(False)
        self._refresh_table_choices()

    def _keypoints(self) -> List[str]:
        return [self.kp_list.item(i).text().strip() for i in range(self.kp_list.count()) if self.kp_list.item(i).text().strip()]

    def _make_combo(self, current: Optional[str] = None) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItem("")
        for kp in self._keypoints():
            combo.addItem(kp)
        if current:
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(self._update_flip_preview)
        return combo

    def _add_pair_row(self, table: QtWidgets.QTableWidget) -> None:
        row = table.rowCount()
        table.insertRow(row)
        table.setCellWidget(row, 0, self._make_combo())
        table.setCellWidget(row, 1, self._make_combo())
        self._update_flip_preview()

    def _remove_selected_row(self, table: QtWidgets.QTableWidget) -> None:
        row = table.currentRow()
        if row < 0:
            return
        table.removeRow(row)
        self._update_flip_preview()

    def _set_pair_table(self, table: QtWidgets.QTableWidget, pairs: List[Tuple[str, str]]) -> None:
        table.setRowCount(0)
        for a, b in pairs:
            row = table.rowCount()
            table.insertRow(row)
            table.setCellWidget(row, 0, self._make_combo(a))
            table.setCellWidget(row, 1, self._make_combo(b))

    def _pairs_from_table(self, table: QtWidgets.QTableWidget) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for row in range(table.rowCount()):
            # type: ignore[union-attr]
            a = table.cellWidget(row, 0).currentText().strip()
            # type: ignore[union-attr]
            b = table.cellWidget(row, 1).currentText().strip()
            if not a or not b or a == b:
                continue
            pairs.append((a, b))
        return pairs

    def _refresh_table_choices(self) -> None:
        # type: ignore[attr-defined]
        for table in (self.sym_table._table, self.edge_table._table):
            for row in range(table.rowCount()):
                for col in (0, 1):
                    combo: QtWidgets.QComboBox = table.cellWidget(
                        row, col)  # type: ignore[assignment]
                    current = combo.currentText()
                    combo.blockSignals(True)
                    combo.clear()
                    combo.addItem("")
                    for kp in self._keypoints():
                        combo.addItem(kp)
                    idx = combo.findText(current)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
        self._update_flip_preview()

    def _on_keypoints_changed(self, *args) -> None:
        self._refresh_table_choices()

    def _add_keypoint(self) -> None:
        kp = self.kp_add_edit.text().strip()
        if not kp:
            return
        existing = set(self._keypoints())
        if kp in existing:
            self.kp_add_edit.clear()
            return
        self.kp_list.addItem(kp)
        self.kp_add_edit.clear()
        self._refresh_table_choices()

    def _remove_selected_keypoints(self) -> None:
        rows = sorted({idx.row()
                      for idx in self.kp_list.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            self.kp_list.takeItem(row)
        self._refresh_table_choices()

    def _auto_fill_symmetry(self) -> None:
        keypoints = self._keypoints()
        pairs = PoseSchema.infer_symmetry_pairs(keypoints)
        # type: ignore[attr-defined]
        self._set_pair_table(self.sym_table._table, pairs)
        self._update_flip_preview()

    def _update_flip_preview(self) -> None:
        self._sync_schema_from_ui(update_preview_only=True)
        order = self._schema.expand_keypoints() if hasattr(
            self._schema, "expand_keypoints") else self._schema.keypoints
        flip = self._schema.compute_flip_idx(order)
        self.flip_preview.setText(str(flip) if flip else "")

    def _sync_schema_from_ui(self, update_preview_only: bool = False) -> None:
        instances = [
            item.strip().rstrip("_")
            for item in self.instance_edit.text().split(",")
            if item.strip()
        ]
        self._schema.instances = instances
        self._schema.instance_separator = self.separator_edit.text().strip() or "_"
        self._schema.keypoints = self._keypoints()
        self._schema.symmetry_pairs = self._pairs_from_table(
            self.sym_table._table)  # type: ignore[attr-defined]
        self._schema.edges = self._pairs_from_table(
            self.edge_table._table)  # type: ignore[attr-defined]
        if not update_preview_only:
            order = self._schema.expand_keypoints()
            self._schema.flip_idx = self._schema.compute_flip_idx(order)

    def _on_accept(self) -> None:
        self._sync_schema_from_ui(update_preview_only=False)
        if not self._schema.keypoints:
            QtWidgets.QMessageBox.warning(
                self, "Missing keypoints", "Please define at least one keypoint.")
            return
        self.accept()

    def _normalize_prefixed_schema(self) -> None:
        """Convert a schema that uses fully-qualified keypoint labels into base+instances."""
        self._sync_schema_from_ui(update_preview_only=True)
        if not self._schema.instances:
            inferred = []
            for kp in self._schema.keypoints:
                inst, _ = self._schema.strip_instance_prefix(kp)
                if inst and inst not in inferred:
                    inferred.append(inst)
            self._schema.instances = inferred
        self._schema.normalize_prefixed_keypoints()
        self._refresh_from_schema()
