from qtpy.QtWidgets import (QVBoxLayout, QPushButton,
                            QFileDialog, QLineEdit, QLabel, QDialog,
                            QMessageBox)

from annolid.postprocessing.skeletonization import main as extract_shape_keypoints


class ExtractShapeKeyPointsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Extract Keypoints")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.selectFolderBtn = QPushButton("Select Folder")
        self.selectFolderBtn.clicked.connect(self.selectFolder)
        layout.addWidget(self.selectFolderBtn)

        self.folderLabel = QLabel()
        layout.addWidget(self.folderLabel)

        self.instanceNamesEdit = QLineEdit()
        self.instanceNamesEdit.setPlaceholderText(
            "Enter instance names separated by space")
        layout.addWidget(self.instanceNamesEdit)

        self.runBtn = QPushButton("Run")
        self.runBtn.clicked.connect(self.extract_keypoints)
        layout.addWidget(self.runBtn)

        self.setLayout(layout)

    def selectFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folderLabel.setText(folder)

    def extract_keypoints(self):
        input_folder = self.folderLabel.text()
        instance_names = self.instanceNamesEdit.text().split()
        extract_shape_keypoints(input_folder, instance_names)
        # Display message to the user
        QMessageBox.information(self, "Processing Complete", "Processing is complete.")
        self.accept()
