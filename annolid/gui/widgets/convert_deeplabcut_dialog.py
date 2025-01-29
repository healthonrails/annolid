import sys
from qtpy.QtWidgets import (
    QApplication, QDialog, QPushButton, QFileDialog, QMessageBox, QLineEdit, QCheckBox)
from pathlib import Path
from annolid.annotation.deeplabcut2labelme import deeplabcut_to_labelme_json


class ConvertDLCDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Convert DeepLabCut CSV to LabelMe')
        self.setGeometry(100, 100, 500, 200)

        self.txt_selected_file = QLineEdit(self)
        self.txt_selected_file.setGeometry(50, 20, 400, 20)
        self.txt_selected_file.setReadOnly(True)

        self.btn_select_file = QPushButton('Select Video File', self)
        self.btn_select_file.setGeometry(50, 50, 200, 30)
        self.btn_select_file.clicked.connect(self.select_file)

        self.chk_multi_animal = QCheckBox('Multi-Animal Tracking', self)
        self.chk_multi_animal.setGeometry(50, 90, 200, 20)

        self.btn_run = QPushButton('Run', self)
        self.btn_run.setGeometry(50, 120, 100, 30)
        self.btn_run.clicked.connect(self.run_conversion)
        self.btn_run.setEnabled(False)

        self.btn_close = QPushButton('Close', self)
        self.btn_close.setGeometry(200, 120, 100, 30)
        self.btn_close.clicked.connect(self.close)

    def select_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov)")
        file_dialog.setWindowTitle("Select Video File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.video_file_path = file_paths[0]
                self.txt_selected_file.setText(self.video_file_path)
                self.btn_run.setEnabled(True)

    def run_conversion(self):
        try:
            output_dir = str(Path(self.video_file_path).with_suffix(''))
            deeplabcut_to_labelme_json(
                self.video_file_path, output_dir, self.chk_multi_animal.isChecked())
            QMessageBox.information(
                self, "Success", "Conversion completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = ConvertDLCDialog()
    dialog.show()
    sys.exit(app.exec_())
