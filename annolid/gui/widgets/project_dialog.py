from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from qtpy import QtCore, QtWidgets

from annolid.behavior.boris_io import (
    BorisProjectImport,
    import_boris_project,
    write_boris_project,
)
from annolid.behavior.project_schema import validate_schema
from annolid.domain import (
    BehaviorDefinition,
    CategoryDefinition,
    IndependentVariableDefinition,
    ModifierDefinition,
    ProjectSchema,
    SubjectDefinition,
    load_behavior_spec,
    save_behavior_spec,
)


class ProjectDialog(QtWidgets.QDialog):
    """BORIS-compatible schema & ethogram editor for Annolid project metadata."""

    def __init__(
        self,
        schema: ProjectSchema,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Project Schema & BORIS Ethogram")
        self.resize(1080, 680)
        self._schema = copy.deepcopy(schema)
        self._imported_boris: Optional[BorisProjectImport] = None
        self._syncing_matrix = False

        main_layout = QtWidgets.QVBoxLayout(self)

        # Top-level I/O toolbar
        io_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(io_layout)

        self.import_button = QtWidgets.QPushButton("Import Schema…")
        self.import_button.setToolTip("Import Annolid JSON or YAML project schema")
        self.import_button.clicked.connect(self._import_schema)
        io_layout.addWidget(self.import_button)

        self.export_button = QtWidgets.QPushButton("Export Schema…")
        self.export_button.setToolTip("Export Annolid JSON or YAML project schema")
        self.export_button.clicked.connect(self._export_schema)
        io_layout.addWidget(self.export_button)

        self.import_boris_button = QtWidgets.QPushButton("Import BORIS (.boris)…")
        self.import_boris_button.setToolTip(
            "Import ethogram, subjects, modifiers, exclusions, independent variables, and observations from a .boris file"
        )
        self.import_boris_button.clicked.connect(self._import_boris)
        io_layout.addWidget(self.import_boris_button)

        self.export_boris_button = QtWidgets.QPushButton("Export BORIS (.boris)…")
        self.export_boris_button.setToolTip(
            "Export current ethogram, subjects, modifiers, exclusions, and independent variables to a .boris file"
        )
        self.export_boris_button.clicked.connect(self._export_boris)
        io_layout.addWidget(self.export_boris_button)

        io_layout.addStretch(1)

        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget, 1)

        self._build_behavior_tab()
        self._build_exclusion_matrix_tab()
        self._build_subject_tab()
        self._build_modifier_tab()
        self._build_category_tab()
        self._build_independent_variables_tab()

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        self._validation_label = QtWidgets.QLabel("")
        self._validation_label.setWordWrap(True)
        self._validation_label.setStyleSheet(
            "QLabel { color: #92400e; background-color: #fef3c7; border: 1px solid #f59e0b; border-radius: 4px; padding: 6px; }"
        )
        self._validation_label.hide()
        main_layout.addWidget(self._validation_label)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self._populate_tables()
        self._refresh_validation_banner()

    @property
    def imported_boris_project(self) -> Optional[BorisProjectImport]:
        return self._imported_boris

    # ------------------------------------------------------------------ #
    # Tab builders
    # ------------------------------------------------------------------ #
    def _build_category_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.categories_table = QtWidgets.QTableWidget(0, 4)
        self.categories_table.setHorizontalHeaderLabels(
            ["ID", "Name", "Color (#RRGGBB)", "Description"]
        )
        self.categories_table.horizontalHeader().setStretchLastSection(True)
        self.categories_table.itemChanged.connect(
            lambda _item: self._refresh_validation_banner()
        )
        layout.addWidget(self.categories_table)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        add_btn = QtWidgets.QPushButton("Add Category")
        add_btn.clicked.connect(
            lambda: self._insert_row(self.categories_table, ["", "", "#2E7D32", ""])
        )
        buttons.addWidget(add_btn)
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(lambda: self._remove_selected(self.categories_table))
        buttons.addWidget(remove_btn)
        buttons.addStretch(1)

        self.tab_widget.addTab(tab, "Categories")

    def _build_modifier_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.modifiers_table = QtWidgets.QTableWidget(0, 6)
        self.modifiers_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Name",
                "Group / Set",
                "Mode (multi/single)",
                "Key Binding",
                "Description",
            ]
        )
        self.modifiers_table.horizontalHeader().setStretchLastSection(True)
        self.modifiers_table.itemChanged.connect(
            lambda _item: self._refresh_validation_banner()
        )
        layout.addWidget(self.modifiers_table)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        add_btn = QtWidgets.QPushButton("Add Modifier")
        add_btn.clicked.connect(
            lambda: self._insert_row(
                self.modifiers_table, ["", "", "Default", "multi", "", ""]
            )
        )
        buttons.addWidget(add_btn)
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(lambda: self._remove_selected(self.modifiers_table))
        buttons.addWidget(remove_btn)
        buttons.addStretch(1)

        self.tab_widget.addTab(tab, "Modifiers")

    def _build_subject_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.subjects_table = QtWidgets.QTableWidget(0, 5)
        self.subjects_table.setHorizontalHeaderLabels(
            ["ID", "Name", "Key Binding", "Color (#RRGGBB)", "Description"]
        )
        self.subjects_table.horizontalHeader().setStretchLastSection(True)
        self.subjects_table.itemChanged.connect(
            lambda _item: self._refresh_validation_banner()
        )
        layout.addWidget(self.subjects_table)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        add_btn = QtWidgets.QPushButton("Add Subject")
        add_btn.clicked.connect(
            lambda: self._insert_row(self.subjects_table, ["", "", "", "", ""])
        )
        buttons.addWidget(add_btn)
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(lambda: self._remove_selected(self.subjects_table))
        buttons.addWidget(remove_btn)

        self.add_subjects_as_modifiers_btn = QtWidgets.QPushButton(
            "Add Subjects as Modifiers"
        )
        self.add_subjects_as_modifiers_btn.setToolTip(
            "Create a single-select 'Target Subject' modifier set from defined subjects for dyadic social interactions"
        )
        self.add_subjects_as_modifiers_btn.clicked.connect(
            self._add_subjects_as_modifiers
        )
        buttons.addWidget(self.add_subjects_as_modifiers_btn)
        buttons.addStretch(1)

        self.tab_widget.addTab(tab, "Subjects")

    def _build_behavior_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.behaviors_table = QtWidgets.QTableWidget(0, 9)
        self.behaviors_table.setHorizontalHeaderLabels(
            [
                "Code",
                "Name",
                "Description",
                "Category ID",
                "Modifier IDs (comma separated)",
                "Key Binding",
                "Is State (true/false)",
                "Excluded Codes (comma separated)",
                "Color (#RRGGBB)",
            ]
        )
        self.behaviors_table.horizontalHeader().setStretchLastSection(True)
        self.behaviors_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Interactive
        )
        self.behaviors_table.itemChanged.connect(
            lambda _item: self._refresh_validation_banner()
        )
        layout.addWidget(self.behaviors_table)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        add_state_btn = QtWidgets.QPushButton("Add State Behavior")
        add_state_btn.clicked.connect(
            lambda: self._insert_row(
                self.behaviors_table,
                ["", "", "", "", "", "", "true", "", ""],
            )
        )
        buttons.addWidget(add_state_btn)

        add_point_btn = QtWidgets.QPushButton("Add Point Behavior")
        add_point_btn.clicked.connect(
            lambda: self._insert_row(
                self.behaviors_table,
                ["", "", "", "", "", "", "false", "", ""],
            )
        )
        buttons.addWidget(add_point_btn)

        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(lambda: self._remove_selected(self.behaviors_table))
        buttons.addWidget(remove_btn)
        buttons.addStretch(1)

        self.tab_widget.addTab(tab, "Behaviors (Ethogram)")

    def _build_exclusion_matrix_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        hint = QtWidgets.QLabel(
            "Check pairs of behaviors that are mutually exclusive for the same focal subject. "
            "When 'Auto-stop Excluded' is active during coding, starting one state automatically closes active excluded states."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.exclusion_matrix_table = QtWidgets.QTableWidget(0, 0)
        self.exclusion_matrix_table.itemChanged.connect(
            self._on_exclusion_matrix_item_changed
        )
        layout.addWidget(self.exclusion_matrix_table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        excl_all_states_btn = QtWidgets.QPushButton(
            "Exclude All State Behaviors From Each Other"
        )
        excl_all_states_btn.clicked.connect(self._exclude_all_states)
        clear_excl_btn = QtWidgets.QPushButton("Clear All Exclusions")
        clear_excl_btn.clicked.connect(self._clear_all_exclusions)
        btn_row.addWidget(excl_all_states_btn)
        btn_row.addWidget(clear_excl_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._exclusion_tab_index = self.tab_widget.addTab(tab, "Exclusion Matrix")

    def _build_independent_variables_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.independent_vars_table = QtWidgets.QTableWidget(0, 7)
        self.independent_vars_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Name",
                "Type (text/numeric/value_from_list/timestamp)",
                "Default Value",
                "Allowed Values (comma separated)",
                "Session Value",
                "Description",
            ]
        )
        self.independent_vars_table.horizontalHeader().setStretchLastSection(True)
        self.independent_vars_table.itemChanged.connect(
            lambda _item: self._refresh_validation_banner()
        )
        layout.addWidget(self.independent_vars_table)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        add_btn = QtWidgets.QPushButton("Add Variable")
        add_btn.clicked.connect(
            lambda: self._insert_row(
                self.independent_vars_table,
                ["", "", "text", "", "", "", ""],
            )
        )
        buttons.addWidget(add_btn)
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(
            lambda: self._remove_selected(self.independent_vars_table)
        )
        buttons.addWidget(remove_btn)
        buttons.addStretch(1)

        self.tab_widget.addTab(tab, "Independent Variables")

    # ------------------------------------------------------------------ #
    # Table & Matrix helpers
    # ------------------------------------------------------------------ #
    def _insert_row(self, table: QtWidgets.QTableWidget, values: List[str]) -> None:
        row = table.rowCount()
        table.insertRow(row)
        for col, value in enumerate(values):
            table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
        self._refresh_validation_banner()

    def _remove_selected(self, table: QtWidgets.QTableWidget) -> None:
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)
        self._refresh_validation_banner()

    def _on_tab_changed(self, index: int) -> None:
        if index == getattr(self, "_exclusion_tab_index", -1):
            self._rebuild_exclusion_matrix_from_behaviors_table()

    def _behavior_rows_snapshot(self) -> List[Tuple[int, str, bool, Set[str]]]:
        out: List[Tuple[int, str, bool, Set[str]]] = []
        for r in range(self.behaviors_table.rowCount()):
            code_item = self.behaviors_table.item(r, 0)
            code = code_item.text().strip() if code_item else ""
            if not code:
                continue
            state_item = self.behaviors_table.item(r, 6)
            is_state = (
                (state_item.text().strip().lower() != "false") if state_item else True
            )
            excl_item = self.behaviors_table.item(r, 7)
            excl_raw = excl_item.text().strip() if excl_item else ""
            excl_set = {x.strip() for x in excl_raw.split(",") if x.strip()}
            out.append((r, code, is_state, excl_set))
        return out

    def _rebuild_exclusion_matrix_from_behaviors_table(self) -> None:
        snap = self._behavior_rows_snapshot()
        codes = [code for _, code, _, _ in snap]
        pairs: Set[Tuple[str, str]] = set()
        for _, code, _, excl_set in snap:
            for other in excl_set:
                if other != code:
                    pairs.add((code, other))
                    pairs.add((other, code))

        self._syncing_matrix = True
        try:
            n = len(codes)
            self.exclusion_matrix_table.clear()
            self.exclusion_matrix_table.setRowCount(n)
            self.exclusion_matrix_table.setColumnCount(n)
            self.exclusion_matrix_table.setHorizontalHeaderLabels(codes)
            self.exclusion_matrix_table.setVerticalHeaderLabels(codes)
            for r_i, code_r in enumerate(codes):
                for c_i, code_c in enumerate(codes):
                    item = QtWidgets.QTableWidgetItem("")
                    if r_i == c_i:
                        item.setFlags(QtCore.Qt.NoItemFlags)
                        item.setBackground(QtCore.Qt.lightGray)
                    else:
                        item.setFlags(
                            QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable
                        )
                        item.setCheckState(
                            QtCore.Qt.Checked
                            if (code_r, code_c) in pairs
                            else QtCore.Qt.Unchecked
                        )
                    self.exclusion_matrix_table.setItem(r_i, c_i, item)
            self.exclusion_matrix_table.resizeColumnsToContents()
        finally:
            self._syncing_matrix = False

    def _on_exclusion_matrix_item_changed(
        self, item: QtWidgets.QTableWidgetItem
    ) -> None:
        if self._syncing_matrix:
            return
        r = item.row()
        c = item.column()
        if r == c:
            return
        self._syncing_matrix = True
        try:
            mirror = self.exclusion_matrix_table.item(c, r)
            if mirror is not None:
                mirror.setCheckState(item.checkState())
            self._write_matrix_back_to_behaviors_table()
        finally:
            self._syncing_matrix = False
        self._refresh_validation_banner()

    def _write_matrix_back_to_behaviors_table(self) -> None:
        n = self.exclusion_matrix_table.rowCount()
        codes = [
            (
                self.exclusion_matrix_table.horizontalHeaderItem(i).text()
                if self.exclusion_matrix_table.horizontalHeaderItem(i)
                else ""
            )
            for i in range(n)
        ]
        excl_map: Dict[str, List[str]] = {code: [] for code in codes if code}
        for r in range(n):
            for c in range(n):
                if r == c:
                    continue
                cell = self.exclusion_matrix_table.item(r, c)
                if cell and cell.checkState() == QtCore.Qt.Checked:
                    if codes[r] and codes[c]:
                        excl_map[codes[r]].append(codes[c])

        with QtCore.QSignalBlocker(self.behaviors_table):
            for row in range(self.behaviors_table.rowCount()):
                c_item = self.behaviors_table.item(row, 0)
                code = c_item.text().strip() if c_item else ""
                if code in excl_map:
                    self.behaviors_table.setItem(
                        row,
                        7,
                        QtWidgets.QTableWidgetItem(", ".join(sorted(excl_map[code]))),
                    )

    def _exclude_all_states(self) -> None:
        snap = self._behavior_rows_snapshot()
        state_codes = {code for _, code, is_state, _ in snap if is_state}
        self._syncing_matrix = True
        try:
            n = self.exclusion_matrix_table.rowCount()
            for r in range(n):
                code_r = (
                    self.exclusion_matrix_table.horizontalHeaderItem(r).text()
                    if self.exclusion_matrix_table.horizontalHeaderItem(r)
                    else ""
                )
                for c in range(n):
                    if r == c:
                        continue
                    code_c = (
                        self.exclusion_matrix_table.horizontalHeaderItem(c).text()
                        if self.exclusion_matrix_table.horizontalHeaderItem(c)
                        else ""
                    )
                    cell = self.exclusion_matrix_table.item(r, c)
                    if (
                        cell is not None
                        and code_r in state_codes
                        and code_c in state_codes
                    ):
                        cell.setCheckState(QtCore.Qt.Checked)
            self._write_matrix_back_to_behaviors_table()
        finally:
            self._syncing_matrix = False

    def _clear_all_exclusions(self) -> None:
        self._syncing_matrix = True
        try:
            n = self.exclusion_matrix_table.rowCount()
            for r in range(n):
                for c in range(n):
                    if r == c:
                        continue
                    cell = self.exclusion_matrix_table.item(r, c)
                    if cell is not None:
                        cell.setCheckState(QtCore.Qt.Unchecked)
            self._write_matrix_back_to_behaviors_table()
        finally:
            self._syncing_matrix = False

    def _add_subjects_as_modifiers(self) -> None:
        current_schema = self._collect_schema()
        if not current_schema.subjects:
            QtWidgets.QMessageBox.information(
                self,
                "Subjects Required",
                "Add at least one subject in the Subjects tab first.",
            )
            return
        enriched = current_schema.with_subjects_as_modifiers()
        self._schema = enriched
        self._populate_tables()
        self._refresh_validation_banner()

    def _populate_tables(self) -> None:
        def fill(table: QtWidgets.QTableWidget, rows: List[List[str]]) -> None:
            with QtCore.QSignalBlocker(table):
                table.setRowCount(0)
                for r_idx, row_values in enumerate(rows):
                    table.insertRow(r_idx)
                    for col, value in enumerate(row_values):
                        table.setItem(r_idx, col, QtWidgets.QTableWidgetItem(value))

        fill(
            self.categories_table,
            [
                [
                    category.id,
                    category.name,
                    category.color or "",
                    category.description or "",
                ]
                for category in self._schema.categories
            ],
        )
        fill(
            self.modifiers_table,
            [
                [
                    modifier.id,
                    modifier.name,
                    modifier.group or "",
                    modifier.selection_mode or "multi",
                    modifier.key_binding or "",
                    modifier.description or "",
                ]
                for modifier in self._schema.modifiers
            ],
        )
        fill(
            self.subjects_table,
            [
                [
                    subject.id,
                    subject.name,
                    subject.key_binding or "",
                    subject.color or "",
                    subject.description or "",
                ]
                for subject in self._schema.subjects
            ],
        )
        fill(
            self.behaviors_table,
            [
                [
                    behavior.code,
                    behavior.name,
                    behavior.description or "",
                    behavior.category_id or "",
                    ", ".join(behavior.modifier_ids),
                    behavior.key_binding or "",
                    "true" if behavior.is_state else "false",
                    ", ".join(behavior.exclusive_with),
                    behavior.color or "",
                ]
                for behavior in self._schema.behaviors
            ],
        )
        obs_vals = self._schema.observation_variables or {}
        fill(
            self.independent_vars_table,
            [
                [
                    iv.id,
                    iv.name,
                    iv.var_type or "text",
                    iv.default_value or "",
                    ", ".join(iv.allowed_values),
                    str(
                        obs_vals.get(
                            iv.name, obs_vals.get(iv.id, iv.default_value or "")
                        )
                    ),
                    iv.description or "",
                ]
                for iv in self._schema.independent_variables
            ],
        )
        self._rebuild_exclusion_matrix_from_behaviors_table()

    # ------------------------------------------------------------------ #
    # Schema serialization & validation helpers
    # ------------------------------------------------------------------ #
    def _collect_schema(self) -> ProjectSchema:
        def values(table: QtWidgets.QTableWidget, row: int) -> List[str]:
            return [
                (table.item(row, col).text().strip() if table.item(row, col) else "")
                for col in range(table.columnCount())
            ]

        categories: List[CategoryDefinition] = []
        for row in range(self.categories_table.rowCount()):
            vals = values(self.categories_table, row)
            cid = vals[0]
            if cid:
                categories.append(
                    CategoryDefinition(
                        id=cid,
                        name=vals[1] or cid,
                        color=vals[2] or None,
                        description=vals[3] or None,
                    )
                )

        modifiers: List[ModifierDefinition] = []
        for row in range(self.modifiers_table.rowCount()):
            vals = values(self.modifiers_table, row)
            mid = vals[0]
            if not mid:
                continue
            if len(vals) >= 6:
                name, group, mode, key_b, desc = (
                    vals[1],
                    vals[2],
                    vals[3],
                    vals[4],
                    vals[5],
                )
            else:
                name = vals[1] if len(vals) > 1 else mid
                desc = vals[2] if len(vals) > 2 else ""
                group, mode, key_b = "", "multi", ""
            modifiers.append(
                ModifierDefinition(
                    id=mid,
                    name=name or mid,
                    description=desc or None,
                    group=group or None,
                    selection_mode=(mode or "multi").lower(),
                    key_binding=key_b or None,
                )
            )

        subjects: List[SubjectDefinition] = []
        for row in range(self.subjects_table.rowCount()):
            vals = values(self.subjects_table, row)
            sid = vals[0]
            if not sid:
                continue
            if len(vals) >= 5:
                name, key_b, color, desc = vals[1], vals[2], vals[3], vals[4]
            else:
                name = vals[1] if len(vals) > 1 else sid
                desc = vals[2] if len(vals) > 2 else ""
                key_b, color = "", ""
            subjects.append(
                SubjectDefinition(
                    id=sid,
                    name=name or sid,
                    description=desc or None,
                    key_binding=key_b or None,
                    color=color or None,
                )
            )

        behaviors: List[BehaviorDefinition] = []
        for row in range(self.behaviors_table.rowCount()):
            vals = values(self.behaviors_table, row)
            code = vals[0]
            if not code:
                continue
            name = vals[1]
            description = vals[2]
            category_id = vals[3]
            modifier_ids = vals[4]
            key_binding = vals[5]
            is_state = vals[6]
            excl_raw = vals[7] if len(vals) > 7 else ""
            color_raw = vals[8] if len(vals) > 8 else ""

            modifiers_list = [
                item.strip() for item in modifier_ids.split(",") if item.strip()
            ]
            excl_list = [
                item.strip()
                for item in excl_raw.split(",")
                if item.strip() and item.strip() != code
            ]
            behaviors.append(
                BehaviorDefinition(
                    code=code,
                    name=name or code,
                    description=description or None,
                    category_id=category_id or None,
                    modifier_ids=modifiers_list,
                    key_binding=key_binding or None,
                    is_state=is_state.lower() != "false",
                    exclusive_with=excl_list,
                    color=color_raw or None,
                )
            )

        indep_vars: List[IndependentVariableDefinition] = []
        obs_vars: Dict[str, str] = {}
        for row in range(self.independent_vars_table.rowCount()):
            vals = values(self.independent_vars_table, row)
            iv_id = vals[0]
            if not iv_id:
                continue
            iv_name = vals[1] or iv_id
            iv_type = (vals[2] or "text").strip().lower()
            iv_default = vals[3] or None
            iv_allowed = [x.strip() for x in vals[4].split(",") if x.strip()]
            iv_sess_val = vals[5]
            iv_desc = vals[6] or None
            indep_vars.append(
                IndependentVariableDefinition(
                    id=iv_id,
                    name=iv_name,
                    var_type=iv_type,
                    default_value=iv_default,
                    allowed_values=iv_allowed,
                    description=iv_desc,
                )
            )
            if iv_sess_val:
                obs_vars[iv_name] = iv_sess_val

        return ProjectSchema(
            behaviors=behaviors,
            categories=categories,
            modifiers=modifiers,
            subjects=subjects,
            pose_schema_path=getattr(self._schema, "pose_schema_path", None),
            pose_schema=getattr(self._schema, "pose_schema", None),
            version=self._schema.version,
            independent_variables=indep_vars,
            observation_variables=obs_vars,
        )

    def _refresh_validation_banner(self) -> None:
        if not hasattr(self, "_validation_label"):
            return
        schema = self._collect_schema()
        warnings = validate_schema(schema)
        if warnings:
            self._validation_label.setText(
                "Schema warnings:\n• " + "\n• ".join(warnings[:5])
            )
            self._validation_label.show()
        else:
            self._validation_label.hide()
            self._validation_label.clear()

    def get_schema(self) -> ProjectSchema:
        return self._collect_schema()

    # ------------------------------------------------------------------ #
    # Import / export actions
    # ------------------------------------------------------------------ #
    def _import_schema(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Project Schema",
            "",
            "Schema & BORIS Files (*.json *.yaml *.yml *.boris);;All Files (*)",
        )
        if not path_str:
            return
        resolved = Path(path_str)
        if resolved.suffix.lower() == ".boris":
            self._load_boris_path(resolved)
            return
        try:
            schema, _ = load_behavior_spec(path=resolved)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Import Failed",
                f"Unable to import schema:\n{exc}",
            )
            return
        self._imported_boris = None
        self._schema = schema
        self._populate_tables()
        self._refresh_validation_banner()

    def _export_schema(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Project Schema",
            "",
            "Schema Files (*.json *.yaml *.yml);;All Files (*)",
        )
        if not path_str:
            return
        schema = self._collect_schema()
        try:
            save_behavior_spec(schema, Path(path_str))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Failed",
                f"Unable to export schema:\n{exc}",
            )

    def _import_boris(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import BORIS Project (.boris)",
            "",
            "BORIS Project Files (*.boris *.json);;All Files (*)",
        )
        if not path_str:
            return
        self._load_boris_path(Path(path_str))

    def _load_boris_path(self, path: Path) -> None:
        try:
            imported = import_boris_project(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "BORIS Import Failed",
                f"Unable to import BORIS project:\n{exc}",
            )
            return
        self._imported_boris = imported
        self._schema = imported.schema
        self._populate_tables()
        self._refresh_validation_banner()

    def _export_boris(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export BORIS Project (.boris)",
            "project.boris",
            "BORIS Project Files (*.boris);;JSON Files (*.json)",
        )
        if not path_str:
            return
        schema = self._collect_schema()
        try:
            write_boris_project(Path(path_str), schema)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "BORIS Export Failed",
                f"Unable to export BORIS project:\n{exc}",
            )
