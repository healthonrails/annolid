import sys
from qtpy.QtWidgets import (QApplication, QDialog,
                            QPushButton, QFileDialog,
                            QMessageBox, QLineEdit)
from annolid.annotation.sleap2labelme import convert_sleap_h5_to_labelme


class ConvertSleapDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Convert H5 to Labelme')
        self.setGeometry(100, 100, 400, 200)

        self.txt_selected_file = QLineEdit(self)
        self.txt_selected_file.setGeometry(50, 20, 300, 20)
        self.txt_selected_file.setReadOnly(True)

        self.btn_select_file = QPushButton('Select H5 File', self)
        self.btn_select_file.setGeometry(50, 50, 200, 30)
        self.btn_select_file.clicked.connect(self.select_file)

        self.btn_run = QPushButton('Run', self)
        self.btn_run.setGeometry(50, 100, 100, 30)
        self.btn_run.clicked.connect(self.run_conversion)
        self.btn_run.setEnabled(False)

        self.btn_close = QPushButton('Close', self)
        self.btn_close.setGeometry(200, 100, 100, 30)
        self.btn_close.clicked.connect(self.close)

    def select_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("H5 Files (*.h5)")
        file_dialog.setWindowTitle("Select H5 File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.h5_file_path = file_paths[0]
                self.txt_selected_file.setText(self.h5_file_path)
                self.btn_run.setEnabled(True)

    def run_conversion(self):
        try:
            convert_sleap_h5_to_labelme(self.h5_file_path)
            QMessageBox.information(
                self, "Success", "Conversion completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = ConvertSleapDialog()
    dialog.show()
    sys.exit(app.exec_())
