from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from annolid.postprocessing.skeletonization import main as extract_shape_keypoints


class ExtractShapeKeyPointsDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        start_dir: Optional[str | Path] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extract Keypoints from Masks (Skeletonization)")
        self._start_dir = str(start_dir) if start_dir else ""
        self.initUI()
        if self._start_dir:
            self.folderLabel.setText(self._start_dir)

    def initUI(self) -> None:
        layout = QVBoxLayout()

        self.selectFolderBtn = QPushButton("Select Folder")
        self.selectFolderBtn.clicked.connect(self.selectFolder)
        layout.addWidget(self.selectFolderBtn)

        self.folderLabel = QLabel()
        self.folderLabel.setWordWrap(True)
        layout.addWidget(self.folderLabel)

        self.instanceNamesEdit = QLineEdit()
        self.instanceNamesEdit.setPlaceholderText(
            "Enter instance names separated by space or comma"
        )
        layout.addWidget(self.instanceNamesEdit)

        self.runBtn = QPushButton("Run")
        self.runBtn.clicked.connect(self.extract_keypoints)
        layout.addWidget(self.runBtn)

        self.setLayout(layout)

    def selectFolder(self) -> None:
        start = self.folderLabel.text().strip() or self._start_dir or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if folder:
            self.folderLabel.setText(folder)

    def extract_keypoints(self) -> None:
        input_folder = self.folderLabel.text().strip()
        if not input_folder or not Path(input_folder).expanduser().is_dir():
            QMessageBox.warning(
                self,
                "Invalid Folder",
                "Please select an existing folder containing LabelMe JSON files.",
            )
            return
        raw_names = self.instanceNamesEdit.text().replace(",", " ")
        instance_names = [tok.strip() for tok in raw_names.split() if tok.strip()]
        if not instance_names:
            QMessageBox.warning(
                self,
                "Missing Instance Names",
                "Please enter at least one instance label to extract skeleton keypoints from.",
            )
            return
        try:
            summary = extract_shape_keypoints(input_folder, instance_names)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Extraction Failed",
                f"Failed to extract skeleton keypoints:\n{exc}",
            )
            return
        files_count = (
            int(summary.get("processed_files", 0)) if isinstance(summary, dict) else 0
        )
        points_count = (
            int(summary.get("points_written", 0)) if isinstance(summary, dict) else 0
        )
        QMessageBox.information(
            self,
            "Processing Complete",
            (
                f"Processed {files_count} LabelMe JSON file(s) and "
                f"added/updated {points_count} skeleton keypoint(s)."
            ),
        )
        self.accept()
