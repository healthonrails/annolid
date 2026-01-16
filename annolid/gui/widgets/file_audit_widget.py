from qtpy import QtWidgets, QtCore, QtGui
from pathlib import Path
from typing import List, Dict, Any


class FileAuditWidget(QtWidgets.QWidget):
    """Widget to display and manage scanned annotation files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Filters
        filter_layout = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QRadioButton("All")
        self.btn_valid = QtWidgets.QRadioButton("Valid")
        self.btn_issues = QtWidgets.QRadioButton("Issues Only")
        self.btn_all.setChecked(True)

        self.btn_all.toggled.connect(self._apply_filter)
        self.btn_valid.toggled.connect(self._apply_filter)
        self.btn_issues.toggled.connect(self._apply_filter)

        filter_layout.addWidget(QtWidgets.QLabel("Filter:"))
        filter_layout.addWidget(self.btn_all)
        filter_layout.addWidget(self.btn_valid)
        filter_layout.addWidget(self.btn_issues)
        filter_layout.addStretch()
        self.layout.addLayout(filter_layout)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Status", "Filename", "Image", "Shapes"])
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.layout.addWidget(self.table)

        # Data storage
        self.items: List[Dict[str, Any]] = []

    def set_items(self, items: List[Dict[str, Any]]):
        self.items = items
        self._apply_filter()

    def _apply_filter(self):
        mode = "all"
        if self.btn_valid.isChecked():
            mode = "valid"
        elif self.btn_issues.isChecked():
            mode = "issues"

        self.table.setRowCount(0)

        filtered = []
        for item in self.items:
            is_valid = item["status"] == "valid"
            if mode == "valid" and not is_valid:
                continue
            if mode == "issues" and is_valid:
                continue
            filtered.append(item)

        self.table.setRowCount(len(filtered))
        for row, item in enumerate(filtered):
            # Status Icon
            status_item = QtWidgets.QTableWidgetItem()
            if item["status"] == "valid":
                status_item.setText("✅")
                status_item.setToolTip("Valid annotation pair")
            elif item["status"] == "missing_image":
                status_item.setText("❌")
                status_item.setToolTip("Missing image file")
            else:
                status_item.setText("⚠️")
                status_item.setToolTip(item["status"])
            status_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 0, status_item)

            # Filename
            fname_item = QtWidgets.QTableWidgetItem(item["json_path"].name)
            fname_item.setToolTip(str(item["json_path"]))
            self.table.setItem(row, 1, fname_item)

            # Image existing?
            img_text = "Yes" if item.get("image_path") else "Missing"
            img_item = QtWidgets.QTableWidgetItem(img_text)
            if not item.get("image_path"):
                img_item.setForeground(QtGui.QColor("red"))
            self.table.setItem(row, 2, img_item)

            # Shape count
            shape_item = QtWidgets.QTableWidgetItem(str(item.get("shape_count", 0)))
            self.table.setItem(row, 3, shape_item)

            # Store data user role
            fname_item.setData(QtCore.Qt.UserRole, item)

    def _show_context_menu(self, position):
        menu = QtWidgets.QMenu()
        selected = self.table.selectedItems()
        if not selected:
            return

        # Get item data from the Filename column (column 1)
        row = selected[0].row()
        item_data = self.table.item(row, 1).data(QtCore.Qt.UserRole)
        json_path: Path = item_data["json_path"]

        action_reveal = menu.addAction("Reveal in Finder/Explorer")
        action_delete = menu.addAction("Delete JSON File")

        action = menu.exec_(self.table.viewport().mapToGlobal(position))

        if action == action_reveal:
            self._reveal_file(json_path)
        elif action == action_delete:
            self._delete_file(json_path, row)

    def _reveal_file(self, path: Path):
        url = QtCore.QUrl.fromLocalFile(str(path.parent))
        QtGui.QDesktopServices.openUrl(url)

    def _delete_file(self, path: Path, row: int):
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {path.name}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            try:
                path.unlink()
                self.table.removeRow(row)
                # Remove from internal list
                self.items = [i for i in self.items if i["json_path"] != path]
                # Re-apply filter to update view correctly
                self._apply_filter()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to delete: {e}")
