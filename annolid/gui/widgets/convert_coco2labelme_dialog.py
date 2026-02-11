from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from qtpy import QtWidgets


@dataclass(frozen=True)
class ConvertCOCO2LabelMeConfig:
    annotations_dir: Path
    output_dir: Path
    images_dir: Optional[Path]
    recursive: bool
    link_mode: str


class ConvertCOCO2LabelMeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, start_dir: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Convert COCO -> LabelMe Dataset")
        self.setModal(True)
        self.resize(700, 280)

        self._start_dir = start_dir or str(Path.home())
        self._config: Optional[ConvertCOCO2LabelMeConfig] = None
        self._build_ui()
        self._wire()
        self._update_state()

    @property
    def config(self) -> Optional[ConvertCOCO2LabelMeConfig]:
        return self._config

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()

        self.annotations_edit = QtWidgets.QLineEdit()
        self.annotations_edit.setPlaceholderText(
            "Directory containing COCO *.json files"
        )
        row_ann = QtWidgets.QHBoxLayout()
        row_ann.addWidget(self.annotations_edit, 1)
        self.annotations_btn = QtWidgets.QPushButton("Browse...")
        row_ann.addWidget(self.annotations_btn)
        form.addRow("Annotations dir:", row_ann)

        self.images_edit = QtWidgets.QLineEdit()
        self.images_edit.setPlaceholderText(
            "Optional image root (leave empty for auto-resolve)"
        )
        row_img = QtWidgets.QHBoxLayout()
        row_img.addWidget(self.images_edit, 1)
        self.images_btn = QtWidgets.QPushButton("Browse...")
        row_img.addWidget(self.images_btn)
        form.addRow("Images dir:", row_img)

        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("Output LabelMe dataset directory")
        row_out = QtWidgets.QHBoxLayout()
        row_out.addWidget(self.output_edit, 1)
        self.output_btn = QtWidgets.QPushButton("Browse...")
        row_out.addWidget(self.output_btn)
        form.addRow("Output dir:", row_out)

        self.recursive_check = QtWidgets.QCheckBox("Find COCO JSON recursively")
        self.recursive_check.setChecked(True)
        form.addRow("", self.recursive_check)

        self.link_mode_combo = QtWidgets.QComboBox()
        self.link_mode_combo.addItems(["hardlink", "copy", "symlink"])
        self.link_mode_combo.setCurrentText("hardlink")
        form.addRow("Image copy mode:", self.link_mode_combo)

        root.addLayout(form)

        note = QtWidgets.QLabel(
            "The converter saves each image and its LabelMe sidecar JSON together in the output folder."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.ok_btn = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.ok_btn.setText("Convert")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _wire(self) -> None:
        self.annotations_btn.clicked.connect(self._browse_annotations)
        self.images_btn.clicked.connect(self._browse_images)
        self.output_btn.clicked.connect(self._browse_output)
        self.annotations_edit.textChanged.connect(self._update_state)
        self.output_edit.textChanged.connect(self._update_state)

    def _browse_annotations(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select COCO annotations directory", self._start_dir
        )
        if not folder:
            return
        self.annotations_edit.setText(folder)
        if not self.output_edit.text().strip():
            self.output_edit.setText(str(Path(folder).parent / "labelme_dataset"))

    def _browse_images(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select images directory (optional)", self._start_dir
        )
        if folder:
            self.images_edit.setText(folder)

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory", self._start_dir
        )
        if folder:
            self.output_edit.setText(folder)

    def _update_state(self) -> None:
        annotations_ok = bool(self.annotations_edit.text().strip())
        output_ok = bool(self.output_edit.text().strip())
        self.ok_btn.setEnabled(annotations_ok and output_ok)

    def _accept(self) -> None:
        annotations_dir = Path(self.annotations_edit.text().strip()).expanduser()
        output_dir = Path(self.output_edit.text().strip()).expanduser()
        if not annotations_dir.exists() or not annotations_dir.is_dir():
            QtWidgets.QMessageBox.warning(
                self, "Invalid Input", "Please select a valid annotations directory."
            )
            return
        images_text = self.images_edit.text().strip()
        images_dir = Path(images_text).expanduser() if images_text else None
        if images_dir is not None and (
            not images_dir.exists() or not images_dir.is_dir()
        ):
            QtWidgets.QMessageBox.warning(
                self, "Invalid Input", "Images directory does not exist."
            )
            return
        self._config = ConvertCOCO2LabelMeConfig(
            annotations_dir=annotations_dir.resolve(),
            output_dir=output_dir.resolve(),
            images_dir=(images_dir.resolve() if images_dir else None),
            recursive=bool(self.recursive_check.isChecked()),
            link_mode=str(self.link_mode_combo.currentText()),
        )
        self.accept()
