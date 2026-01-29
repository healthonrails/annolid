from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from qtpy import QtCore, QtWidgets

from annolid.annotation.pose_schema import PoseSchema
from annolid.utils.logger import logger

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

Pair = Tuple[str, str]


@dataclass(frozen=True)
class ParsedPair:
    a: str
    b: str


class PoseSchemaDialog(QtWidgets.QDialog):
    """
    Edit a PoseSchema (keypoint order, symmetry pairs, and edges).

    Key fixes vs original:
    - Robustly loads edges/symmetry_pairs even if PoseSchema.load() doesn't hydrate them
      (reads raw JSON/YAML and forces fields into schema).
    - Robust pair/edge coercion (list/tuple/dict/string forms).
    - Better L/R inference fallback for eyeL/eyeR, wingL/wingR, forelegL1/forelegR1, etc.
    - Drops invalid edges/pairs referencing missing keypoints instead of crashing.
    """

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
        self._schema_path = schema_path
        # When True, UI is being populated from schema; ignore reactive updates that
        # would otherwise sync empty tables back into the schema.
        self._refreshing_ui = False

        if keypoints:
            cleaned = [kp.strip() for kp in keypoints if kp and kp.strip()]
            self._schema.keypoints = list(dict.fromkeys(cleaned))

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

    # -----------------------------
    # UI
    # -----------------------------

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
            getattr(self._schema, "instance_separator", "_") or "_"
        )
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
        self.kp_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
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

        self.sym_box, self.sym_table = self._build_pair_table(
            "Symmetry pairs (left ↔ right)"
        )
        sym_controls = self._build_table_controls(
            self.sym_table, add_text="Add pair", remove_text="Remove pair"
        )
        right_layout.addWidget(self.sym_box, 1)
        right_layout.addLayout(sym_controls)

        self.edge_box, self.edge_table = self._build_pair_table("Edges (optional)")
        edge_controls = self._build_table_controls(
            self.edge_table, add_text="Add edge", remove_text="Remove edge"
        )
        right_layout.addWidget(self.edge_box, 1)
        right_layout.addLayout(edge_controls)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        # Preview row
        preview = QtWidgets.QHBoxLayout()
        self.flip_preview = QtWidgets.QLabel("")
        self.flip_preview.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
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
    def _build_pair_table(
        title: str,
    ) -> Tuple[QtWidgets.QGroupBox, QtWidgets.QTableWidget]:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        table = QtWidgets.QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["A", "B"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(table)
        return box, table

    def _build_table_controls(
        self,
        table: QtWidgets.QTableWidget,
        *,
        add_text: str,
        remove_text: str,
    ) -> QtWidgets.QHBoxLayout:
        controls = QtWidgets.QHBoxLayout()
        controls.addStretch(1)
        add_btn = QtWidgets.QPushButton(add_text)
        rm_btn = QtWidgets.QPushButton(remove_text)
        add_btn.clicked.connect(lambda: self._add_pair_row(table))
        rm_btn.clicked.connect(lambda: self._remove_selected_row(table))
        controls.addWidget(add_btn)
        controls.addWidget(rm_btn)
        return controls

    # -----------------------------
    # File actions
    # -----------------------------

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
        try:
            self._load_schema(Path(path))
        except Exception:
            logger.exception("Failed to load pose schema: %s", path)
            QtWidgets.QMessageBox.critical(
                self, "Load failed", f"Failed to load:\n{path}"
            )
            return
        self._refresh_from_schema()

    def _on_save_clicked(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            self._on_save_as_clicked()
            return

        self._sync_schema_from_ui(update_preview_only=False)
        try:
            self._schema.save(path)
        except Exception:
            logger.exception("Failed to save pose schema: %s", path)
            QtWidgets.QMessageBox.critical(
                self, "Save failed", f"Failed to save:\n{path}"
            )
            return

        self._schema_path = path
        QtWidgets.QMessageBox.information(
            self, "Saved", f"Pose schema saved to:\n{path}"
        )

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

    # -----------------------------
    # Robust loading/hydration
    # -----------------------------

    @staticmethod
    def _read_schema_dict(path: Path) -> dict:
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("PyYAML not installed (needed to read .yaml/.yml)")
            obj = yaml.safe_load(text)
        else:
            obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}

    @staticmethod
    def _first_present(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
        for k in keys:
            if k in d:
                return d[k]
        return None

    def _load_schema(self, path: Path) -> None:
        """
        Load schema via PoseSchema.load(), then force-hydrate key fields from raw JSON/YAML
        in case PoseSchema.load() ignores some keys (common cause of "edges not loaded").
        """
        schema = PoseSchema.load(path)

        raw: Dict[str, Any] = {}
        try:
            raw = self._read_schema_dict(path)
        except Exception:
            logger.debug("Failed to read raw schema dict for hydration.", exc_info=True)

        if raw:
            # keypoints
            raw_kps = self._first_present(raw, ["keypoints", "nodes", "landmarks"])
            if isinstance(raw_kps, list) and raw_kps:
                schema.keypoints = [str(x).strip() for x in raw_kps if str(x).strip()]

            # symmetry pairs: accept alternate key names
            raw_sym = self._first_present(
                raw, ["symmetry_pairs", "symmetry", "pairs", "lr_pairs"]
            )
            if raw_sym is not None:
                schema.symmetry_pairs = self._coerce_pairs(raw_sym)

            # edges: accept alternate key names
            raw_edges = self._first_present(
                raw, ["edges", "edge_list", "skeleton", "links", "connections"]
            )
            if raw_edges is not None:
                schema.edges = self._coerce_pairs(raw_edges)

            # instances + separator (optional)
            raw_instances = self._first_present(raw, ["instances", "instance_prefixes"])
            if raw_instances is not None and hasattr(schema, "instances"):
                if isinstance(raw_instances, list):
                    schema.instances = [
                        str(x).strip() for x in raw_instances if str(x).strip()
                    ]

            raw_sep = self._first_present(raw, ["instance_separator", "separator"])
            if raw_sep is not None and hasattr(schema, "instance_separator"):
                schema.instance_separator = str(raw_sep).strip() or "_"

            # flip_idx (optional)
            raw_flip = self._first_present(raw, ["flip_idx", "flipIndex"])
            if raw_flip is not None and hasattr(schema, "flip_idx"):
                schema.flip_idx = raw_flip

        self._schema = schema
        self._schema_path = str(path)
        self.path_edit.setText(str(path))

        # Normalize & validate loaded content
        self._coerce_loaded_schema()

    def _coerce_loaded_schema(self) -> None:
        """Normalize loaded schema and ensure edges/pairs are valid and not silently lost."""
        # keypoints
        self._schema.keypoints = [
            str(k).strip() for k in (self._schema.keypoints or []) if str(k).strip()
        ]
        kps = set(self._schema.keypoints)

        # instances/separator safety
        if not hasattr(self._schema, "instances") or self._schema.instances is None:
            self._schema.instances = []
        if (
            not hasattr(self._schema, "instance_separator")
            or not self._schema.instance_separator
        ):
            self._schema.instance_separator = "_"

        # symmetry pairs + edges coercion and filtering
        self._schema.symmetry_pairs = self._filter_pairs_by_keypoints(
            self._coerce_pairs(getattr(self._schema, "symmetry_pairs", None)), kps
        )
        self._schema.edges = self._filter_pairs_by_keypoints(
            self._coerce_pairs(getattr(self._schema, "edges", None)), kps
        )

        # compute flip_idx (optional)
        try:
            order = self._expand_keypoints_safe()
            self._schema.flip_idx = self._compute_flip_idx_safe(order)
        except Exception:
            logger.debug(
                "Failed to compute flip_idx during load coercion.", exc_info=True
            )

    def _filter_pairs_by_keypoints(
        self, pairs: List[Pair], keypoints_set: Set[str]
    ) -> List[Pair]:
        out: List[Pair] = []
        seen: Set[Tuple[str, str]] = set()
        for a, b in pairs:
            a = (a or "").strip()
            b = (b or "").strip()
            if not a or not b or a == b:
                continue
            if a not in keypoints_set or b not in keypoints_set:
                logger.debug(
                    "Dropping pair referencing missing keypoints: (%s, %s)", a, b
                )
                continue
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b))
        return out

    # -----------------------------
    # Schema <-> UI syncing
    # -----------------------------

    def _refresh_from_schema(self) -> None:
        self._refreshing_ui = True
        try:
            self.instance_edit.setText(
                ", ".join(getattr(self._schema, "instances", []) or [])
            )
            self.separator_edit.setText(
                getattr(self._schema, "instance_separator", "_") or "_"
            )

            # Keypoints first so the pair/edge combo boxes have choices.
            self._set_keypoints(list(self._schema.keypoints or []))
            self._set_pair_table(
                self.sym_table, self._coerce_pairs(self._schema.symmetry_pairs)
            )
            self._set_pair_table(
                self.edge_table, self._coerce_pairs(self._schema.edges)
            )
            self._refresh_table_choices()
        finally:
            self._refreshing_ui = False
        self._update_flip_preview()

    def _sync_schema_from_ui(self, update_preview_only: bool = False) -> None:
        instances = [
            item.strip().rstrip("_")
            for item in self.instance_edit.text().split(",")
            if item.strip()
        ]
        self._schema.instances = instances
        self._schema.instance_separator = self.separator_edit.text().strip() or "_"
        self._schema.keypoints = self._keypoints()

        self._schema.symmetry_pairs = self._pairs_from_table(self.sym_table)
        self._schema.edges = self._pairs_from_table(self.edge_table)

        if not update_preview_only:
            order = self._expand_keypoints_safe()
            self._schema.flip_idx = self._compute_flip_idx_safe(order)

    def _on_accept(self) -> None:
        self._sync_schema_from_ui(update_preview_only=False)
        if not self._schema.keypoints:
            QtWidgets.QMessageBox.warning(
                self, "Missing keypoints", "Please define at least one keypoint."
            )
            return
        self.accept()

    # -----------------------------
    # Keypoints UI
    # -----------------------------

    def _set_keypoints(self, keypoints: List[str]) -> None:
        # Block both the widget and its underlying model signals.
        # QListWidget emits rowsInserted/rowsRemoved/rowsMoved from its model even
        # when the widget's signals are blocked; those were triggering
        # _update_flip_preview() mid-refresh and clearing loaded pairs/edges.
        model = self.kp_list.model()
        model.blockSignals(True)
        self.kp_list.blockSignals(True)
        try:
            self.kp_list.clear()
            for kp in keypoints:
                kps = (kp or "").strip()
                if kps:
                    self.kp_list.addItem(kps)
        finally:
            self.kp_list.blockSignals(False)
            model.blockSignals(False)
        self._refresh_table_choices()

    def _keypoints(self) -> List[str]:
        out: List[str] = []
        for i in range(self.kp_list.count()):
            txt = self.kp_list.item(i).text().strip()
            if txt:
                out.append(txt)
        return out

    def _on_keypoints_changed(self, *args: Any) -> None:
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
        rows = sorted(
            {idx.row() for idx in self.kp_list.selectedIndexes()}, reverse=True
        )
        if not rows:
            return
        for row in rows:
            self.kp_list.takeItem(row)
        self._refresh_table_choices()

    # -----------------------------
    # Pairs / Edges UI
    # -----------------------------

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

    def _set_pair_table(self, table: QtWidgets.QTableWidget, pairs: List[Pair]) -> None:
        table.setRowCount(0)
        for a, b in pairs:
            row = table.rowCount()
            table.insertRow(row)
            table.setCellWidget(row, 0, self._make_combo(a))
            table.setCellWidget(row, 1, self._make_combo(b))

    def _pairs_from_table(self, table: QtWidgets.QTableWidget) -> List[Pair]:
        pairs: List[Pair] = []
        seen: Set[Tuple[str, str]] = set()
        kps = set(self._keypoints())

        for row in range(table.rowCount()):
            w0 = table.cellWidget(row, 0)
            w1 = table.cellWidget(row, 1)
            if not isinstance(w0, QtWidgets.QComboBox) or not isinstance(
                w1, QtWidgets.QComboBox
            ):
                continue

            a = w0.currentText().strip()
            b = w1.currentText().strip()
            if not a or not b or a == b:
                continue
            if a not in kps or b not in kps:
                continue

            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, b))
        return pairs

    def _refresh_table_choices(self) -> None:
        kps = self._keypoints()
        for table in (self.sym_table, self.edge_table):
            for row in range(table.rowCount()):
                for col in (0, 1):
                    combo = table.cellWidget(row, col)
                    if not isinstance(combo, QtWidgets.QComboBox):
                        continue
                    current = combo.currentText()
                    combo.blockSignals(True)
                    combo.clear()
                    combo.addItem("")
                    for kp in kps:
                        combo.addItem(kp)
                    idx = combo.findText(current)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
        self._update_flip_preview()

    # -----------------------------
    # Pair coercion (robust)
    # -----------------------------

    def _coerce_pairs(self, raw: Any) -> List[Pair]:
        """
        Accepts many shapes:
          - List[Tuple[str,str]] / List[List[str,str]]
          - Dicts: {"a": "...", "b": "..."} or {"source": "...", "target": "..."}
          - Strings: "a-b", "a,b", "a->b"
        Returns clean, de-duplicated List[Pair].
        """
        if not raw:
            return []

        if isinstance(raw, (list, tuple)):
            items = list(raw)
        else:
            items = [raw]

        out: List[Pair] = []
        seen: Set[Tuple[str, str]] = set()

        for item in items:
            parsed = self._parse_pair_item(item)
            if not parsed:
                continue
            a = parsed.a.strip()
            b = parsed.b.strip()
            if not a or not b or a == b:
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            out.append((a, b))

        return out

    def _parse_pair_item(self, item: Any) -> Optional[ParsedPair]:
        try:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                return ParsedPair(str(item[0]), str(item[1]))

            if isinstance(item, dict):
                for ka, kb in (
                    ("a", "b"),
                    ("A", "B"),
                    ("source", "target"),
                    ("src", "dst"),
                ):
                    if ka in item and kb in item:
                        return ParsedPair(str(item[ka]), str(item[kb]))
                if len(item) == 2:
                    vals = list(item.values())
                    return ParsedPair(str(vals[0]), str(vals[1]))

            if isinstance(item, str):
                s = item.strip()
                parts = re.split(r"\s*(?:->|—|–|-|,|:|/|\|)\s*", s)
                if len(parts) >= 2:
                    return ParsedPair(parts[0], parts[1])
        except Exception:
            logger.debug("Failed to parse pair item: %r", item, exc_info=True)
        return None

    # -----------------------------
    # Auto symmetry + flip preview
    # -----------------------------

    def _auto_fill_symmetry(self) -> None:
        keypoints = self._keypoints()

        pairs: List[Pair] = []
        try:
            pairs = PoseSchema.infer_symmetry_pairs(keypoints)
        except Exception:
            logger.debug(
                "PoseSchema.infer_symmetry_pairs failed; using fallback.", exc_info=True
            )

        if not pairs:
            pairs = self._infer_lr_pairs_fallback(keypoints)

        self._set_pair_table(self.sym_table, pairs)
        self._update_flip_preview()

    def _update_flip_preview(self) -> None:
        if getattr(self, "_refreshing_ui", False):
            return
        self._sync_schema_from_ui(update_preview_only=True)
        order = self._expand_keypoints_safe()
        flip = self._compute_flip_idx_safe(order)
        self.flip_preview.setText(str(flip) if flip else "")

    def _expand_keypoints_safe(self) -> List[str]:
        try:
            if hasattr(self._schema, "expand_keypoints"):
                return list(self._schema.expand_keypoints())
        except Exception:
            logger.debug(
                "expand_keypoints() failed; using raw keypoints.", exc_info=True
            )
        return list(self._schema.keypoints or [])

    def _compute_flip_idx_safe(self, order: Sequence[str]) -> Optional[List[int]]:
        try:
            if hasattr(self._schema, "compute_flip_idx"):
                flip = self._schema.compute_flip_idx(list(order))
                return flip if flip else None
        except Exception:
            logger.debug("compute_flip_idx() failed.", exc_info=True)
        return None

    # -----------------------------
    # L/R inference fallback
    # -----------------------------

    _LR_SUFFIX = re.compile(
        r"^(?P<base>.*?)(?P<side>[LR])(?P<tail>\d*)$", re.IGNORECASE
    )
    _LR_SEP_SUFFIX = re.compile(
        r"^(?P<base>.*?)(?:[_\-\.\s])(?P<side>[LR])(?P<tail>\d*)$", re.IGNORECASE
    )

    def _infer_lr_pairs_fallback(self, keypoints: Sequence[str]) -> List[Pair]:
        buckets: Dict[Tuple[str, str], Dict[str, str]] = {}

        for kp in keypoints:
            kp_str = (kp or "").strip()
            if not kp_str:
                continue
            side, base, tail = self._split_lr(kp_str)
            if not side or not base:
                continue
            key = (base.lower(), tail)
            buckets.setdefault(key, {})[side] = kp_str

        pairs: List[Pair] = []
        for (_base, _tail), lr in buckets.items():
            left = lr.get("L")
            right = lr.get("R")
            if left and right:
                pairs.append((left, right))

        order = {kp: i for i, kp in enumerate(keypoints)}
        pairs.sort(key=lambda p: min(order.get(p[0], 10**9), order.get(p[1], 10**9)))
        return pairs

    def _split_lr(self, label: str) -> Tuple[Optional[str], Optional[str], str]:
        s = label.strip()
        low = s.lower()

        # wordy forms: left_eye / right_eye / eye_left / eye_right
        if low.startswith(("left_", "left-", "left ")):
            base = re.sub(r"^left[_\-\.\s]+", "", s, flags=re.IGNORECASE)
            return "L", base, ""
        if low.startswith(("right_", "right-", "right ")):
            base = re.sub(r"^right[_\-\.\s]+", "", s, flags=re.IGNORECASE)
            return "R", base, ""
        if low.endswith(("_left", "-left")) or low.endswith(" left"):
            base = re.sub(r"[_\-\.\s]+left$", "", s, flags=re.IGNORECASE)
            return "L", base, ""
        if low.endswith(("_right", "-right")) or low.endswith(" right"):
            base = re.sub(r"[_\-\.\s]+right$", "", s, flags=re.IGNORECASE)
            return "R", base, ""

        # suffix: eyeL, eyeR, forelegL1, forelegR1
        m = self._LR_SUFFIX.match(s)
        if m:
            side = m.group("side").upper()
            base = (m.group("base") or "").strip()
            tail = (m.group("tail") or "").strip()
            if side in ("L", "R") and base:
                return side, base, tail

        # separated suffix: eye_L, eye-R
        m = self._LR_SEP_SUFFIX.match(s)
        if m:
            side = m.group("side").upper()
            base = (m.group("base") or "").strip()
            tail = (m.group("tail") or "").strip()
            if side in ("L", "R") and base:
                return side, base, tail

        return None, None, ""

    # -----------------------------
    # Prefix normalization
    # -----------------------------

    def _normalize_prefixed_schema(self) -> None:
        """Convert a schema that uses fully-qualified keypoint labels into base+instances."""
        self._sync_schema_from_ui(update_preview_only=True)

        if not getattr(self._schema, "instances", None):
            inferred: List[str] = []
            for kp in self._schema.keypoints:
                try:
                    inst, _ = self._schema.strip_instance_prefix(kp)
                except Exception:
                    continue
                if inst and inst not in inferred:
                    inferred.append(inst)
            self._schema.instances = inferred

        try:
            self._schema.normalize_prefixed_keypoints()
        except Exception:
            logger.debug("normalize_prefixed_keypoints() failed.", exc_info=True)

        self._coerce_loaded_schema()
        self._refresh_from_schema()
