from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
import typing
import copy


class FlagTableWidget(QtWidgets.QWidget):
    flagsChanged = QtCore.Signal()
    startButtonClicked = QtCore.Signal(str)
    endButtonClicked = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["Flag Name", "Value", "Start", "End"]
        )
        self._table.verticalHeader().setVisible(False)
        self._flags: typing.Dict[str, bool] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._table)
        self.setLayout(layout)

    @property
    def flags(self):
        # return a copy to avoid external changes
        return copy.deepcopy(self._flags)

    def set_flags(self, flags):
        self._flags = flags

    def loadFlags(self, flags: typing.Dict[str, bool]):
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
                item_value.setText("True" if value else "False")
            else:
                # Add a new Row
                row = self._table.rowCount()
                self._table.insertRow(row)

                item_name = QtWidgets.QTableWidgetItem(key)
                item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
                item_value = QtWidgets.QTableWidgetItem(
                    "True" if value else "False")
                item_value.setFlags(item_value.flags() & ~Qt.ItemIsEditable)

                start_button = QtWidgets.QPushButton("Start")
                start_button.setProperty("flag_name", key)
                start_button.clicked.connect(self.handle_button_click)

                end_button = QtWidgets.QPushButton("End")
                end_button.setProperty("flag_name", key)
                end_button.clicked.connect(self.handle_button_click)

                self._table.setItem(row, 0, item_name)
                self._table.setItem(row, 1, item_value)
                self._table.setCellWidget(row, 2, start_button)
                self._table.setCellWidget(row, 3, end_button)

        self._table.resizeColumnsToContents()

    def handle_button_click(self):
        sender = self.sender()
        flag_name = sender.property("flag_name")
        if sender.text() == "Start":
            self._flags[flag_name] = True
            self.startButtonClicked.emit(flag_name)  # Emit Start signal
        elif sender.text() == "End":
            self._flags[flag_name] = False
            self.endButtonClicked.emit(flag_name)  # Emit End signal

        # Update the table value
        rows = self._table.rowCount()
        for row in range(rows):
            item = self._table.item(row, 0)
            if item and item.text() == flag_name:
                item_value = self._table.item(row, 1)
                if item_value:
                    item_value.setText(
                        "True" if self._flags[flag_name] else "False"
                    )
                break

        self.flagsChanged.emit()
