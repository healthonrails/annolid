from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from typing import Dict


class FlagTableWidget(QtWidgets.QWidget):
    flagsSaved = QtCore.Signal(dict)  # Emit all flags at once
    startButtonClicked = QtCore.Signal(str)  # Emit when "Start" is clicked
    endButtonClicked = QtCore.Signal(str)  # Emit when "End" is clicked
    rowSelected = QtCore.Signal(str)  # Emit flag name when a row is selected

    COLUMN_NAME = 0
    COLUMN_VALUE = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._flags: Dict[str, bool] = {}

        # Table setup
        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Flag Name", "Value", "Start", "End"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QTableWidget.AllEditTriggers)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self._table.clicked.connect(self._handle_table_clicked)

        # Buttons Layout
        buttons_layout = QtWidgets.QHBoxLayout()

        self.add_flag_button = QtWidgets.QPushButton("New")
        self.add_flag_button.clicked.connect(self.add_row)
        buttons_layout.addWidget(self.add_flag_button)

        self.remove_flag_button = QtWidgets.QPushButton("Remove Selected")
        self.remove_flag_button.clicked.connect(self.remove_selected_row)
        buttons_layout.addWidget(self.remove_flag_button)

        self.clear_all_button = QtWidgets.QPushButton("Clear All")
        self.clear_all_button.clicked.connect(self.clear_all_rows)
        buttons_layout.addWidget(self.clear_all_button)

        self.save_all_button = QtWidgets.QPushButton("Save")
        self.save_all_button.clicked.connect(self.save_all_flags)
        buttons_layout.addWidget(self.save_all_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._table)
        layout.addLayout(buttons_layout)

    def _handle_table_clicked(self, index):
        """Emit the flag name of the selected row."""
        row = index.row()
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
        # Ensure name is a string and value is a boolean
        name = str(name) if name else ""  # Make sure 'name' is a string
        value = "True" if value else "False"  # Ensure value is "True" or "False"

        if name and name in existing_flags:
            QtWidgets.QMessageBox.warning(
                self, "Duplicate Flag", f"Flag '{name}' already exists."
            )
            return

        row = self._table.rowCount()
        self._table.insertRow(row)

        # Flag Name
        name_editor = QtWidgets.QLineEdit(name)
        name_editor.setPlaceholderText("Enter flag name...")
        self._table.setCellWidget(row, self.COLUMN_NAME, name_editor)

        # Flag Value
        value_editor = QtWidgets.QLineEdit("True" if value else "False")
        value_editor.setPlaceholderText("True or False")
        self._table.setCellWidget(row, self.COLUMN_VALUE, value_editor)

        # Start Button
        start_button = QtWidgets.QPushButton("Start")
        start_button.clicked.connect(lambda: self.handle_start_button(row))
        self._table.setCellWidget(row, 2, start_button)

        # End Button
        end_button = QtWidgets.QPushButton("End")
        end_button.clicked.connect(lambda: self.handle_end_button(row))
        self._table.setCellWidget(row, 3, end_button)

    def _get_existing_flag_names(self) -> Dict[str, int]:
        """Retrieve existing flag names in the table."""
        existing_flags = {}
        for row in range(self._table.rowCount()):
            name_widget = self._table.cellWidget(row, self.COLUMN_NAME)
            if isinstance(name_widget, QtWidgets.QLineEdit):
                name = name_widget.text().strip()
                if name:
                    existing_flags[name] = row
        return existing_flags

    def _update_row_value(self, name: str, value: bool):
        """Update the value of an existing flag."""
        existing_flags = self._get_existing_flag_names()
        row = existing_flags.get(name)
        if row is not None:
            value_widget = self._table.cellWidget(row, self.COLUMN_VALUE)
            if isinstance(value_widget, QtWidgets.QLineEdit):
                value_widget.setText("True" if value else "False")

    def remove_selected_row(self):
        """Remove the selected row."""
        selected_items = self._table.selectionModel().selectedRows()
        if selected_items:
            for index in sorted(selected_items, reverse=True):
                self._table.removeRow(index.row())
        else:
            QtWidgets.QMessageBox.warning(
                self, "Error", "No row selected to remove.")

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

    def handle_start_button(self, row):
        """Handle the Start button click."""
        item_name = self._table.cellWidget(row, self.COLUMN_NAME)
        if isinstance(item_name, QtWidgets.QLineEdit):
            flag_name = item_name.text().strip()
            if flag_name:
                self.startButtonClicked.emit(flag_name)

    def handle_end_button(self, row):
        """Handle the End button click."""
        item_name = self._table.cellWidget(row, self.COLUMN_NAME)
        if isinstance(item_name, QtWidgets.QLineEdit):
            flag_name = item_name.text().strip()
            if flag_name:
                self.endButtonClicked.emit(flag_name)

    def save_all_flags(self):
        """Collect and save all flags."""
        flags = {}
        for row in range(self._table.rowCount()):
            name_widget = self._table.cellWidget(row, self.COLUMN_NAME)
            value_widget = self._table.cellWidget(row, self.COLUMN_VALUE)
            if isinstance(name_widget, QtWidgets.QLineEdit) and isinstance(value_widget, QtWidgets.QLineEdit):
                flag_name = name_widget.text().strip()
                flag_value = value_widget.text().strip().lower()
                if flag_name and flag_value in {"true", "false"}:
                    flags[flag_name] = flag_value == "true"
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
