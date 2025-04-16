from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QMenu,
    QShortcut,
    QStyle,
    QTableWidget,
    QUndoStack,
)
from qtpy.QtCore import QPropertyAnimation
from typing import Dict, List
import re


class FlagTable(QTableWidget):
    COLUMN_NAME = 0
    COLUMN_ACTIVE = 1
    COLUMN_START = 2
    COLUMN_END = 3

    def __init__(self, rows=0, columns=4, parent=None):
        super().__init__(rows, columns, parent)
        self.undo_stack = QUndoStack(self)
        self._row_order = list(range(rows))  # Track row order
        self._setup_ui()
        self._setup_shortcuts()
        self._setup_sorting()

    def _setup_ui(self):
        self.setHorizontalHeaderLabels(["Behavior", "Active", "Start", "End"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Get system selection color
        palette = self.palette()
        highlight_color = palette.color(palette.Highlight)
        highlight_rgba = f"rgba({highlight_color.red()}, {highlight_color.green()}, {highlight_color.blue()}, 0.3)"

        self.setStyleSheet(f"""
            QTableWidget {{
                background-color: transparent;
                gridline-color: #e0e0e0;
            }}
            QTableWidget::item:selected {{
                background-color: {highlight_rgba};
                color: palette(text);
            }}
            QTableWidget::item:hover:!selected {{
                background-color: rgba(227, 242, 253, 0.3);
            }}
        """)

    def _setup_shortcuts(self):
        # Set up keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_stack.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.undo_stack.redo)
        QShortcut(QKeySequence("Delete"), self, self.delete_selected)
        QShortcut(QKeySequence("Ctrl+N"), self, self.add_flag)

    def _setup_sorting(self):
        # Enable sorting and set initial sort state
        self.setSortingEnabled(True)
        self.horizontalHeader().setSortIndicatorShown(True)
        self.horizontalHeader().sortIndicatorChanged.connect(self._handle_sort)

    def _handle_sort(self, column, order):
        # Get all rows data
        rows_data = []
        for row in range(self.rowCount()):
            row_data = []
            for col in range(self.columnCount()):
                widget = self.cellWidget(row, col)
                if isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, QtWidgets.QLineEdit):
                    value = widget.text()
                else:
                    value = ""
                row_data.append(value)
            rows_data.append((row_data, row))

        # Sort based on column
        reverse = order == Qt.DescendingOrder
        rows_data.sort(key=lambda x: x[0][column], reverse=reverse)

        # Apply new order
        self._row_order = [x[1] for x in rows_data]
        self._update_row_order()

    def _update_row_order(self):
        self.blockSignals(True)
        for new_idx, old_idx in enumerate(self._row_order):
            if new_idx != old_idx:
                for col in range(self.columnCount()):
                    widget = self.cellWidget(old_idx, col)
                    if widget:
                        self.removeCellWidget(old_idx, col)
                        self.setCellWidget(new_idx, col, widget)
        self.blockSignals(False)

    def _sort_by_active_state(self):
        """Sort table to move active flags to top"""
        rows_data = []
        for row in range(self.rowCount()):
            checkbox = self.cellWidget(row, self.COLUMN_ACTIVE)
            name = self.cellWidget(row, self.COLUMN_NAME).text()
            is_active = checkbox.isChecked() if checkbox else False
            rows_data.append((is_active, name, row))

        # Sort by active state (True first) then by name
        rows_data.sort(key=lambda x: (-int(x[0]), x[1]))
        self._row_order = [x[2] for x in rows_data]
        self._update_row_order()

    def delete_selected(self):
        rows = set(item.row() for item in self.selectedItems())
        for row in sorted(rows, reverse=True):
            self.removeRow(row)
        self._row_order = list(range(self.rowCount()))
        self._update_row_order()

    def add_flag(self, name="", value=False):
        row = self.rowCount()
        self.insertRow(row)

        name_editor = QtWidgets.QLineEdit(name)
        name_editor.setPlaceholderText("Enter flag name...")
        self.setCellWidget(row, self.COLUMN_NAME, name_editor)

        checkbox = QCheckBox()
        checkbox.setChecked(value)
        checkbox.stateChanged.connect(
            lambda state: self._update_checkbox_icon(checkbox, state == Qt.Checked))
        self.setCellWidget(row, self.COLUMN_ACTIVE, checkbox)

        start_button = QtWidgets.QPushButton("Start")
        start_button.clicked.connect(self._handle_start_button_click)
        self.setCellWidget(row, self.COLUMN_START, start_button)

        end_button = QtWidgets.QPushButton("End")
        end_button.clicked.connect(self._handle_end_button_click)
        self.setCellWidget(row, self.COLUMN_END, end_button)

        # Store the row for later use in button handlers
        start_button.setProperty("row", row)
        end_button.setProperty("row", row)

        self._row_order = list(range(self.rowCount()))
        self._update_row_order()

    def delete_selected_row(self):
        """Delete the currently selected row"""
        current_row = self.currentRow()
        if current_row >= 0:
            self.removeRow(current_row)
        self._row_order = list(range(self.rowCount()))
        self._update_row_order()

    def add_new_row(self):
        """Add a new empty row"""
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, self.COLUMN_NAME, QtWidgets.QTableWidgetItem(""))
        self.setItem(row, self.COLUMN_ACTIVE, QtWidgets.QTableWidgetItem(""))
        self._row_order = list(range(self.rowCount()))
        self._update_row_order()

    def save_changes(self):
        """Save current table state"""
        flags = {}
        for row in range(self.rowCount()):
            name_item = self.item(row, self.COLUMN_NAME)
            value_item = self.item(row, self.COLUMN_ACTIVE)
            if name_item and value_item:
                flags[name_item.text()] = bool(value_item.checkState())
        return flags

    def _animate_checkbox(self, checkbox, state):
        animation = QPropertyAnimation(checkbox, b"geometry")
        animation.setDuration(200)

        # Store current geometry
        current_geo = checkbox.geometry()

        # Calculate end geometry based on state
        if state:
            end_geo = current_geo.adjusted(-2, -2, 2, 2)
        else:
            end_geo = current_geo.adjusted(2, 2, -2, -2)

        # Set animation properties
        animation.setStartValue(current_geo)
        animation.setEndValue(end_geo)
        animation.start()

    def _update_checkbox_icon(self, checkbox, state):
        if state:
            checkbox.setIcon(self.style().standardIcon(
                QStyle.SP_DialogApplyButton))
            checkbox.setStyleSheet("QCheckBox { background-color: #e8f5e9; }")
            checkbox.setToolTip("Flag is active (Click to deactivate)")
        else:
            checkbox.setIcon(self.style().standardIcon(
                QStyle.SP_DialogCancelButton))
            checkbox.setStyleSheet("QCheckBox { background-color: #ffebee; }")
            checkbox.setToolTip("Flag is inactive (Click to activate)")
        self._animate_checkbox(checkbox, state)

    def _handle_start_button_click(self):
        """Handler for the start button click."""
        button = self.sender()
        row = button.property("row")
        checkbox = self.cellWidget(row, self.COLUMN_ACTIVE)
        if checkbox:
            checkbox.setChecked(True)

    def _handle_end_button_click(self):
        """Handler for the end button click."""
        button = self.sender()
        row = button.property("row")
        checkbox = self.cellWidget(row, self.COLUMN_ACTIVE)
        if checkbox:
            checkbox.setChecked(False)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addAction("Add", self.add_flag)
        menu.addAction("Delete Selected", self.delete_selected)
        menu.exec_(event.globalPos())


class FlagTableWidget(QtWidgets.QWidget):
    flagsSaved = QtCore.Signal(dict)
    startButtonClicked = QtCore.Signal(str)
    endButtonClicked = QtCore.Signal(str)
    rowSelected = QtCore.Signal(str)
    flagToggled = QtCore.Signal(str, bool)

    COLUMN_NAME = 0
    COLUMN_ACTIVE = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._flags: Dict[str, bool] = {}
        self.last_selected_row: int | None = None
        self._table = FlagTable()
        self._table.setHorizontalHeaderLabels(
            ["Behavior", "Active", "Start", "End"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QTableWidget.AllEditTriggers)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self._table.clicked.connect(self._handle_table_clicked)
        buttons_layout = QtWidgets.QHBoxLayout()
        self.add_flag_button = QtWidgets.QPushButton("Add")
        self.add_flag_button.clicked.connect(self.add_row)
        buttons_layout.addWidget(self.add_flag_button)
        self.remove_flag_button = QtWidgets.QPushButton("Remove")
        self.remove_flag_button.clicked.connect(self.remove_selected_row)
        buttons_layout.addWidget(self.remove_flag_button)
        self.clear_all_button = QtWidgets.QPushButton("Clear All")
        self.clear_all_button.clicked.connect(self.clear_all_rows)
        buttons_layout.addWidget(self.clear_all_button)
        self.save_all_button = QtWidgets.QPushButton("Save All")
        self.save_all_button.clicked.connect(self.save_all_flags)
        buttons_layout.addWidget(self.save_all_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._table)
        layout.addLayout(buttons_layout)

        # add keyboard shortcuts
        shortcuts = [
            # “S” calls handle_start_button on whatever row is focused.
            ("S", lambda: self._trigger_start_on_current_row()),
            # “E” calls handle_end_button on the focused row.
            ("E", lambda: self._trigger_end_on_current_row()),
            # “Ctrl+S” saves all flags.
            ("Ctrl+S", self.save_all_flags),      # save flags
            # “Ctrl+D” deletes the selected row.
            ("Ctrl+D", self.remove_selected_row),  # delete selected
            # “Ctrl+L” clears the entire table after confirmation.
            ("Ctrl+L", self.clear_all_rows),      # clear all
        ]
        for seq, handler in shortcuts:
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(handler)

    def _trigger_start_on_current_row(self):
        row = self._table.currentRow()
        if row >= 0:
            self.handle_start_button(row)

    def _trigger_end_on_current_row(self):
        row = self._table.currentRow()
        if row >= 0:
            self.handle_end_button(row)

    def _handle_table_clicked(self, index):
        """Emit the flag name of the selected row."""
        row = index.row()
        self.last_selected_row = row
        item_name_widget = self._table.cellWidget(row, self.COLUMN_NAME)
        if isinstance(item_name_widget, QtWidgets.QLineEdit):
            flag_name = item_name_widget.text().strip()
            if flag_name:
                self.rowSelected.emit(flag_name)

    def loadFlags(self, flags: Dict[str, bool]):
        """Load or update flags without adding duplicates."""
        existing_flags = self._get_existing_flag_names()
        for flag_name, flag_value in flags.items():
            if flag_name not in existing_flags:
                self.add_row(flag_name, flag_value)
            else:
                self._update_row_value(flag_name, flag_value)
        self._table.resizeColumnsToContents()

    def add_row(self, name: str = "", value: bool = False):
        """Add a new row if the flag name doesn't already exist."""
        existing_flags = self._get_existing_flag_names()
        name = str(name) if name else ""
        if name and not re.match(r"^[a-zA-Z_]", name):
            QtWidgets.QMessageBox.warning(
                self, "Error", "Behavior must start with a letter or underscore."
            )
            return
        if name and name in existing_flags:
            QtWidgets.QMessageBox.warning(
                self, "Duplicate behavior", f"Behavior '{name}' already exists."
            )
            return
        row = self._table.rowCount()
        self._table.insertRow(row)
        # Flag Name
        name_editor = QtWidgets.QLineEdit(name)
        name_editor.setPlaceholderText("Enter behavior name...")
        self._table.setCellWidget(row, self.COLUMN_NAME, name_editor)
        name_editor.setAccessibleName(f"Behavior name: {name}")
        # Flag Value with CheckBox and Icons
        flag_checkbox = QCheckBox()
        flag_checkbox.setChecked(value)
        self._update_checkbox_icon(flag_checkbox, value)  # Set initial icon
        # Change icon when the checkbox toggles
        flag_checkbox.stateChanged.connect(
            lambda state: self._update_checkbox_icon(flag_checkbox, state == Qt.Checked))
        flag_checkbox.stateChanged.connect(
            lambda state, row=row: self.flagToggled.emit(
                self._table.cellWidget(row, self.COLUMN_NAME).text().strip(),
                state == Qt.Checked
            )
        )
        self._table.setCellWidget(row, self.COLUMN_ACTIVE, flag_checkbox)
        flag_checkbox.setAccessibleDescription(f"Toggle state for {name}")
        # Start Button
        start_button = QtWidgets.QPushButton("Start")
        start_button.clicked.connect(lambda: self.handle_start_button(row))
        self._table.setCellWidget(row, 2, start_button)
        start_button.setAccessibleDescription(f"Start behavior {name}")
        # End Button
        end_button = QtWidgets.QPushButton("End")
        end_button.clicked.connect(lambda: self.handle_end_button(row))
        self._table.setCellWidget(row, 3, end_button)
        end_button.setAccessibleDescription(f"End behavior {name}")
        self._table._row_order = list(range(self._table.rowCount()))
        self._table._update_row_order()

    def _update_checkbox_icon(self, checkbox, state):
        if state:
            checkbox.setIcon(self.style().standardIcon(
                QStyle.SP_DialogApplyButton))
            checkbox.setStyleSheet("QCheckBox { background-color: #e8f5e9; }")
            checkbox.setToolTip("Behavior is active (Click to deactivate)")
            self._animate_checkbox(checkbox, True)
        else:
            checkbox.setIcon(self.style().standardIcon(
                QStyle.SP_DialogCancelButton))
            checkbox.setStyleSheet("QCheckBox { background-color: #ffebee; }")
            checkbox.setToolTip("Behavior is inactive (Click to activate)")
            self._animate_checkbox(checkbox, False)

    def _animate_checkbox(self, checkbox, state):
        animation = QPropertyAnimation(checkbox, b"geometry")
        animation.setDuration(200)
        current_geo = checkbox.geometry()
        if state:
            end_geo = current_geo.adjusted(-2, -2, 2, 2)
        else:
            end_geo = current_geo.adjusted(2, 2, -2, -2)
        animation.setStartValue(current_geo)
        animation.setEndValue(end_geo)
        animation.start()

    def _get_existing_flag_names(self) -> Dict[str, int]:
        existing_flags = {}
        for row in range(self._table.rowCount()):
            name_widget = self._table.cellWidget(row, self.COLUMN_NAME)
            if isinstance(name_widget, QtWidgets.QLineEdit):
                name = name_widget.text().strip()
                if name:
                    existing_flags[name] = row
        return existing_flags

    def _update_row_value(self, name: str, value: bool):
        """Update the value of an existing behavior."""
        existing_flags = self._get_existing_flag_names()
        row = existing_flags.get(name)
        if row is not None:
            value_widget = self._table.cellWidget(row, self.COLUMN_ACTIVE)
            value_widget.setChecked(value)
            self._update_checkbox_icon(value_widget, value)

    def remove_selected_row(self):
        """Remove the selected row."""
        selected_items = self._table.selectionModel().selectedRows()
        if selected_items:
            for index in sorted(selected_items, reverse=True):
                self._table.removeRow(index.row())
        else:
            QtWidgets.QMessageBox.warning(
                self, "Error", "No row selected to remove.")
        self._table._row_order = list(range(self._table.rowCount()))
        self._table._update_row_order()

    def clear_all_rows(self):
        """Clear all rows with confirmation."""
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Confirm Clear All",
            "Are you sure you want to clear all rows?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            self._table.setRowCount(0)
            self._table._row_order = []
            self._table._update_row_order()

    def handle_start_button(self, row):
        """Handle the Start button click."""
        item_name = self._table.cellWidget(row, self.COLUMN_NAME)
        if isinstance(item_name, QtWidgets.QLineEdit):
            flag_name = item_name.text().strip()
            if flag_name:
                self.startButtonClicked.emit(flag_name)
                checkbox = self._table.cellWidget(row, self.COLUMN_ACTIVE)
                if checkbox:
                    checkbox.setChecked(True)

    def handle_end_button(self, row):
        """Handle the End button click."""
        item_name = self._table.cellWidget(row, self.COLUMN_NAME)
        if isinstance(item_name, QtWidgets.QLineEdit):
            flag_name = item_name.text().strip()
            if flag_name:
                self.endButtonClicked.emit(flag_name)
                checkbox = self._table.cellWidget(row, self.COLUMN_ACTIVE)
                if checkbox:
                    checkbox.setChecked(False)

    def save_all_flags(self):
        """Collect and save all flags."""
        flags = {}
        for row in range(self._table.rowCount()):
            name_widget = self._table.cellWidget(row, self.COLUMN_NAME)
            value_widget = self._table.cellWidget(row, self.COLUMN_ACTIVE)
            if isinstance(name_widget, QtWidgets.QLineEdit) and isinstance(value_widget, QCheckBox):
                flag_name = name_widget.text().strip()
                flag_value = value_widget.isChecked()
                if flag_name:
                    flags[flag_name] = flag_value
                else:
                    self.show_error(f"Invalid input at row {row + 1}")
                    return
        self._flags = flags
        self.flagsSaved.emit(self._flags)
        QtWidgets.QMessageBox.information(
            self, "Success", "All flags have been saved!")

    def show_error(self, message: str):
        """Show an error message."""
        QtWidgets.QMessageBox.warning(self, "Error", message)

    def clear(self):
        """ Clears the table """
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setHorizontalHeaderLabels(
            ["Behavior", "Active", "Start", "End"])
