import sys
from qtpy.QtWidgets import (QApplication, QDialog,
                            QPushButton, QFileDialog,
                            QMessageBox, QLineEdit)
from annolid.annotation.labelme2csv import convert_json_to_csv


class LabelmeJsonToCsvDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Convert Labelme JSON files to CSV file')
        self.setGeometry(100, 100, 400, 200)

        self.txt_selected_folder = QLineEdit(self)
        self.txt_selected_folder.setGeometry(50, 20, 300, 20)
        self.txt_selected_folder.setReadOnly(True)

        self.btn_select_folder = QPushButton('Select JSON folder', self)
        self.btn_select_folder.setGeometry(50, 50, 200, 30)
        self.btn_select_folder.clicked.connect(self.select_folder)

        self.btn_run = QPushButton('Run', self)
        self.btn_run.setGeometry(50, 100, 100, 30)
        self.btn_run.clicked.connect(self.run_conversion)
        self.btn_run.setEnabled(False)

        self.btn_close = QPushButton('Close', self)
        self.btn_close.setGeometry(200, 100, 100, 30)
        self.btn_close.clicked.connect(self.close)

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
        try:
            convert_json_to_csv(self.json_folder_path)
            QMessageBox.information(
                self, "Success", "Conversion completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = LabelmeJsonToCsvDialog()
    dialog.show()
    sys.exit(app.exec_())
