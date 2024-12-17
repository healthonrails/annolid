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
            ["Flag Name", "Value", "Start", "End"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QTableWidget.AllEditTriggers)

        # Select entire rows
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        # Connect the row selection
        self._table.clicked.connect(self._handle_table_clicked)

        # Buttons Layout
        buttons_layout = QtWidgets.QHBoxLayout()

        # Add New Flag Button
        self.add_flag_button = QtWidgets.QPushButton("New")
        self.add_flag_button.clicked.connect(self.add_row)
        buttons_layout.addWidget(self.add_flag_button)

        # Save All Button
        self.save_all_button = QtWidgets.QPushButton("Save")
        self.save_all_button.clicked.connect(self.save_all_flags)
        buttons_layout.addWidget(self.save_all_button)

        # Main Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._table)
        layout.addLayout(buttons_layout)

    def _handle_table_clicked(self, index):
        """Handle clicks on table rows and emit the flag name."""
        row = index.row()
        item_name_widget = self._table.cellWidget(row, self.COLUMN_NAME)

        if isinstance(item_name_widget, QtWidgets.QLineEdit):
            flag_name = item_name_widget.text().strip()
            if flag_name:
                self.rowSelected.emit(flag_name)

    def loadFlags(self, flags: Dict[str, bool]):
        """Load initial flags into the table or update existing flags."""
        self._flags = flags
        row_count = self._table.rowCount()

        # Create a dictionary to easily access flags in the table by their name
        table_flags = {}
        for row in range(row_count):
            item = self._table.item(row, 0)
            if item:
                table_flags[item.text()] = row

        for key, value in flags.items():
            if key in table_flags:
                # Update Existing Row
                row = table_flags[key]
                item_value = self._table.item(row, 1)
                if item_value:
                    item_value.setText("True" if value else "False")
            else:
                # Add a New Row for the new flag
                self.add_row(key, value)

        # Resize columns to fit content
        self._table.resizeColumnsToContents()

    def add_row(self, name: str = "", value: bool = False):
        """Add a new row for editing."""
        row = self._table.rowCount()
        self._table.insertRow(row)

        # Ensure name is a string and value is a boolean
        name = str(name) if name else ""  # Make sure 'name' is a string
        value = "True" if value else "False"  # Ensure value is "True" or "False"

        # Flag Name
        name_editor = QtWidgets.QLineEdit(name)
        name_editor.setPlaceholderText("Enter flag name...")
        self._table.setCellWidget(row, self.COLUMN_NAME, name_editor)

        # Flag Value
        value_editor = QtWidgets.QLineEdit(value)
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
        """Collect all flags and emit them."""
        flags = {}
        for row in range(self._table.rowCount()):
            name_editor = self._table.cellWidget(row, self.COLUMN_NAME)
            value_editor = self._table.cellWidget(row, self.COLUMN_VALUE)

            if not isinstance(name_editor, QtWidgets.QLineEdit) or not isinstance(value_editor, QtWidgets.QLineEdit):
                continue

            flag_name = name_editor.text().strip()
            flag_value = value_editor.text().strip().lower()

            # Validate inputs
            if not flag_name:
                self.show_error(f"Row {row + 1}: Flag name cannot be empty.")
                return
            if flag_value not in {"true", "false"}:
                self.show_error(
                    f"Row {row + 1}: Value must be 'True' or 'False'.")
                return
            if flag_name in flags:
                self.show_error(f"Duplicate flag name: {flag_name}")
                return

            flags[flag_name] = flag_value == "true"

        # Save the flags and emit the signal
        self._flags = flags
        self.flagsSaved.emit(self._flags)
        QtWidgets.QMessageBox.information(
            self, "Success", "All flags have been saved!")

    def show_error(self, message: str):
        """Display an error message."""
        QtWidgets.QMessageBox.warning(self, "Error", message)

    @property
    def flags(self) -> Dict[str, bool]:
        """Return all flags as a dictionary."""
        return self._flags
