import sys
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
)
from annolid.annotation.labelme2csv import convert_json_to_csv


class LabelmeJsonToCsvDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Convert Labelme JSON files to CSV file")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        folder_label = QLabel("JSON folder", self)
        layout.addWidget(folder_label)

        self.txt_selected_folder = QLineEdit(self)
        self.txt_selected_folder.setReadOnly(True)
        layout.addWidget(self.txt_selected_folder)

        self.btn_select_folder = QPushButton("Select JSON folder", self)
        self.btn_select_folder.clicked.connect(self.select_folder)
        layout.addWidget(self.btn_select_folder)

        self.generate_tracking_checkbox = QCheckBox("Generate *_tracking.csv", self)
        self.generate_tracking_checkbox.setChecked(True)
        layout.addWidget(self.generate_tracking_checkbox)

        self.generate_tracked_checkbox = QCheckBox("Generate *_tracked.csv", self)
        self.generate_tracked_checkbox.setChecked(True)
        layout.addWidget(self.generate_tracked_checkbox)

        buttons_row = QHBoxLayout()
        self.btn_run = QPushButton("Run", self)
        self.btn_run.clicked.connect(self.run_conversion)
        self.btn_run.setEnabled(False)
        buttons_row.addWidget(self.btn_run)
        self.btn_close = QPushButton("Close", self)
        self.btn_close.clicked.connect(self.close)
        buttons_row.addWidget(self.btn_close)
        layout.addLayout(buttons_row)

    @staticmethod
    def _default_tracking_csv_path(json_folder_path: str) -> str:
        return f"{json_folder_path}_tracking.csv"

    @staticmethod
    def _default_tracked_csv_path(json_folder_path: str) -> str:
        return f"{json_folder_path}_tracked.csv"

    def select_folder(self):
        folder_dialog = QFileDialog(self)
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setNameFilter("JSON files (*.json)")
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.setWindowTitle("Select JSON folder")
        if folder_dialog.exec_():
            folder_path = folder_dialog.selectedFiles()[0]
            self.json_folder_path = folder_path
            self.txt_selected_folder.setText(self.json_folder_path)
            self.btn_run.setEnabled(True)

    def run_conversion(self):
        if not (
            self.generate_tracking_checkbox.isChecked()
            or self.generate_tracked_checkbox.isChecked()
        ):
            QMessageBox.warning(
                self,
                "No Output Selected",
                "Select at least one output file type.",
            )
            return

        try:
            tracking_csv_path = self._default_tracking_csv_path(self.json_folder_path)
            tracked_csv_path = self._default_tracked_csv_path(self.json_folder_path)
            convert_json_to_csv(
                self.json_folder_path,
                csv_file=tracking_csv_path,
                tracked_csv_file=(
                    tracked_csv_path
                    if self.generate_tracked_checkbox.isChecked()
                    else None
                ),
                include_tracking_output=self.generate_tracking_checkbox.isChecked(),
            )
            QMessageBox.information(
                self, "Success", "Conversion completed successfully."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = LabelmeJsonToCsvDialog()
    dialog.show()
    sys.exit(app.exec_())
