from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional

from qtpy import QtWidgets

from annolid.core.behavior.spec import (
    BehaviorDefinition,
    CategoryDefinition,
    ModifierDefinition,
    ProjectSchema,
    SubjectDefinition,
    load_behavior_spec,
    save_behavior_spec,
)


class ProjectDialog(QtWidgets.QDialog):
    """BORIS-inspired schema editor for Annolid project metadata."""

    def __init__(
        self,
        schema: ProjectSchema,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Project Schema")
        self.resize(960, 640)
        self._schema = copy.deepcopy(schema)

        main_layout = QtWidgets.QVBoxLayout(self)

        # Top-level buttons
        io_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(io_layout)

        self.import_button = QtWidgets.QPushButton("Import Schema…")
        self.import_button.clicked.connect(self._import_schema)
        io_layout.addWidget(self.import_button)

        self.export_button = QtWidgets.QPushButton("Export Schema…")
        self.export_button.clicked.connect(self._export_schema)
        io_layout.addWidget(self.export_button)

        io_layout.addStretch(1)

        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget, 1)

        self._build_category_tab()
        self._build_modifier_tab()
        self._build_subject_tab()
        self._build_behavior_tab()

        # Buttons at bottom
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self._populate_tables()

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
        layout.addWidget(self.categories_table)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.clicked.connect(
            lambda: self._insert_row(self.categories_table, ["", "", "", ""])
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

        self.modifiers_table = QtWidgets.QTableWidget(0, 3)
        self.modifiers_table.setHorizontalHeaderLabels(["ID", "Name", "Description"])
        self.modifiers_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.modifiers_table)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.clicked.connect(
            lambda: self._insert_row(self.modifiers_table, ["", "", ""])
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

        self.subjects_table = QtWidgets.QTableWidget(0, 3)
        self.subjects_table.setHorizontalHeaderLabels(["ID", "Name", "Description"])
        self.subjects_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.subjects_table)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.clicked.connect(
            lambda: self._insert_row(self.subjects_table, ["", "", ""])
        )
        buttons.addWidget(add_btn)
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(lambda: self._remove_selected(self.subjects_table))
        buttons.addWidget(remove_btn)
        buttons.addStretch(1)

        self.tab_widget.addTab(tab, "Subjects")

    def _build_behavior_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.behaviors_table = QtWidgets.QTableWidget(0, 7)
        self.behaviors_table.setHorizontalHeaderLabels(
            [
                "Code",
                "Name",
                "Description",
                "Category ID",
                "Modifier IDs (comma separated)",
                "Key Binding",
                "Is State (true/false)",
            ]
        )
        self.behaviors_table.horizontalHeader().setStretchLastSection(True)
        self.behaviors_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Interactive
        )
        layout.addWidget(self.behaviors_table)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.clicked.connect(
            lambda: self._insert_row(
                self.behaviors_table, ["", "", "", "", "", "", "true"]
            )
        )
        buttons.addWidget(add_btn)
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(lambda: self._remove_selected(self.behaviors_table))
        buttons.addWidget(remove_btn)
        buttons.addStretch(1)

        self.tab_widget.addTab(tab, "Behaviors")

    # ------------------------------------------------------------------ #
    # Table helpers
    # ------------------------------------------------------------------ #
    def _insert_row(self, table: QtWidgets.QTableWidget, values: List[str]) -> None:
        row = table.rowCount()
        table.insertRow(row)
        for col, value in enumerate(values):
            table.setItem(row, col, QtWidgets.QTableWidgetItem(value))

    def _remove_selected(self, table: QtWidgets.QTableWidget) -> None:
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)

    def _populate_tables(self) -> None:
        def fill(table: QtWidgets.QTableWidget, rows: List[List[str]]) -> None:
            table.setRowCount(0)
            for row_values in rows:
                self._insert_row(table, row_values)

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
                ]
                for behavior in self._schema.behaviors
            ],
        )

    # ------------------------------------------------------------------ #
    # Schema serialization helpers
    # ------------------------------------------------------------------ #
    def _collect_schema(self) -> ProjectSchema:
        def values(table: QtWidgets.QTableWidget, row: int) -> List[str]:
            return [
                (table.item(row, col).text().strip() if table.item(row, col) else "")
                for col in range(table.columnCount())
            ]

        categories: List[CategoryDefinition] = []
        for row in range(self.categories_table.rowCount()):
            cid, name, color, description = values(self.categories_table, row)
            if cid:
                categories.append(
                    CategoryDefinition(
                        id=cid,
                        name=name or cid,
                        color=color or None,
                        description=description or None,
                    )
                )

        modifiers: List[ModifierDefinition] = []
        for row in range(self.modifiers_table.rowCount()):
            mid, name, description = values(self.modifiers_table, row)
            if mid:
                modifiers.append(
                    ModifierDefinition(
                        id=mid,
                        name=name or mid,
                        description=description or None,
                    )
                )

        subjects: List[SubjectDefinition] = []
        for row in range(self.subjects_table.rowCount()):
            sid, name, description = values(self.subjects_table, row)
            if sid:
                subjects.append(
                    SubjectDefinition(
                        id=sid,
                        name=name or sid,
                        description=description or None,
                    )
                )

        behaviors: List[BehaviorDefinition] = []
        for row in range(self.behaviors_table.rowCount()):
            (
                code,
                name,
                description,
                category_id,
                modifier_ids,
                key_binding,
                is_state,
            ) = values(self.behaviors_table, row)
            if not code:
                continue
            modifiers_list = [
                item.strip() for item in modifier_ids.split(",") if item.strip()
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
                )
            )

        return ProjectSchema(
            behaviors=behaviors,
            categories=categories,
            modifiers=modifiers,
            subjects=subjects,
            pose_schema_path=getattr(self._schema, "pose_schema_path", None),
            pose_schema=getattr(self._schema, "pose_schema", None),
            version=self._schema.version,
        )

    def get_schema(self) -> ProjectSchema:
        return self._collect_schema()

    # ------------------------------------------------------------------ #
    # Import/export actions
    # ------------------------------------------------------------------ #
    def _import_schema(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Project Schema",
            "",
            "Schema Files (*.json *.yaml *.yml);;All Files (*)",
        )
        if not path_str:
            return
        try:
            schema, _ = load_behavior_spec(path=Path(path_str))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Import Failed",
                f"Unable to import schema:\n{exc}",
            )
            return
        self._schema = schema
        self._populate_tables()

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
