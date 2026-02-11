from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy import QtCore, QtWidgets

from annolid.datasets.labelme_collection import resolve_image_path


class ConvertCOODialog(QtWidgets.QDialog):
    """User-friendly dialog for exporting LabelMe annotations to COCO."""

    def __init__(self, annotation_dir: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Export LabelMe -> COCO")
        self.setModal(True)
        self.resize(760, 420)

        # Public attributes used by existing workflow code.
        self.annotation_dir: Optional[str] = annotation_dir
        self.out_dir: Optional[str] = None
        self.label_list_text: Optional[str] = None
        self.num_train_frames: float = 0.7  # ratio for labelme2coco.convert(...)
        self.output_mode: str = "segmentation"

        self._build_ui()
        self._connect_signals()

        if annotation_dir:
            self.anno_dir_edit.setText(str(annotation_dir))
            self._auto_fill_output_dir()

        self._update_summary()

    def _build_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)
        main.setSpacing(12)

        intro = QtWidgets.QLabel(
            "Choose a LabelMe annotation folder and export a COCO train/valid dataset.\n"
            "Train split controls how many samples go to train; the rest go to valid."
        )
        intro.setWordWrap(True)
        main.addWidget(intro)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.anno_dir_edit = QtWidgets.QLineEdit(self)
        self.anno_dir_edit.setPlaceholderText(
            "Folder containing LabelMe *.json and images"
        )
        anno_btn = QtWidgets.QPushButton("Browse...", self)
        anno_btn.clicked.connect(self._browse_annotation_dir)
        anno_row = QtWidgets.QHBoxLayout()
        anno_row.addWidget(self.anno_dir_edit, 1)
        anno_row.addWidget(anno_btn)
        form.addRow("Annotation dir:", self._wrap_row(anno_row))

        self.out_dir_edit = QtWidgets.QLineEdit(self)
        self.out_dir_edit.setPlaceholderText(
            "Optional. Defaults to <annotation_dir>_coco_dataset"
        )
        out_btn = QtWidgets.QPushButton("Browse...", self)
        out_btn.clicked.connect(self._browse_output_dir)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.out_dir_edit, 1)
        out_row.addWidget(out_btn)
        form.addRow("Output dir:", self._wrap_row(out_row))

        self.labels_edit = QtWidgets.QLineEdit(self)
        self.labels_edit.setPlaceholderText(
            "Optional labels.txt. Leave empty to auto-detect labels from JSON."
        )
        labels_btn = QtWidgets.QPushButton("Browse...", self)
        labels_btn.clicked.connect(self._browse_labels_file)
        labels_row = QtWidgets.QHBoxLayout()
        labels_row.addWidget(self.labels_edit, 1)
        labels_row.addWidget(labels_btn)
        form.addRow("Labels file:", self._wrap_row(labels_row))

        split_box = QtWidgets.QWidget(self)
        split_layout = QtWidgets.QHBoxLayout(split_box)
        split_layout.setContentsMargins(0, 0, 0, 0)
        self.train_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.train_slider.setMinimum(10)
        self.train_slider.setMaximum(95)
        self.train_slider.setValue(70)
        self.train_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.train_slider.setTickInterval(5)
        self.train_percent_spin = QtWidgets.QSpinBox(self)
        self.train_percent_spin.setRange(10, 95)
        self.train_percent_spin.setValue(70)
        self.train_percent_spin.setSuffix("%")
        split_layout.addWidget(self.train_slider, 1)
        split_layout.addWidget(self.train_percent_spin, 0)
        form.addRow("Train split:", split_box)

        self.mode_combo = QtWidgets.QComboBox(self)
        self.mode_combo.addItem(
            "Segmentation (COCO polygons per shape)", userData="segmentation"
        )
        self.mode_combo.addItem(
            "Keypoints (strict COCO pose annotations)", userData="keypoints"
        )
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.setToolTip(
            "Segmentation mode exports polygon masks per shape label. "
            "Keypoints mode exports one annotation per instance with keypoints/num_keypoints."
        )
        form.addRow("Output mode:", self.mode_combo)

        main.addLayout(form)

        self.summary_label = QtWidgets.QLabel(self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #555;")
        main.addWidget(self.summary_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main.addWidget(buttons)

        self._button_box = buttons

    def _connect_signals(self) -> None:
        self.anno_dir_edit.textChanged.connect(self._on_annotation_dir_changed)
        self.out_dir_edit.textChanged.connect(self._update_summary)
        self.labels_edit.textChanged.connect(self._update_summary)
        self.train_slider.valueChanged.connect(self.train_percent_spin.setValue)
        self.train_percent_spin.valueChanged.connect(self.train_slider.setValue)
        self.train_percent_spin.valueChanged.connect(self._update_summary)
        self.mode_combo.currentIndexChanged.connect(self._update_summary)

    @staticmethod
    def _wrap_row(layout: QtWidgets.QHBoxLayout) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def _browse_annotation_dir(self) -> None:
        start = self.anno_dir_edit.text().strip() or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select LabelMe annotation directory", start
        )
        if folder:
            self.anno_dir_edit.setText(folder)

    def _browse_output_dir(self) -> None:
        start = self.out_dir_edit.text().strip() or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory", start
        )
        if folder:
            self.out_dir_edit.setText(folder)

    def _browse_labels_file(self) -> None:
        start = self.labels_edit.text().strip() or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select labels file (optional)",
            start,
            "Text files (*.txt);;All files (*)",
        )
        if path:
            self.labels_edit.setText(path)

    def _on_annotation_dir_changed(self) -> None:
        self._auto_fill_output_dir()
        self._update_summary()

    def _auto_fill_output_dir(self) -> None:
        if self.out_dir_edit.text().strip():
            return
        src = self.anno_dir_edit.text().strip()
        if not src:
            return
        src_path = Path(src).expanduser()
        self.out_dir_edit.setText(
            str(src_path.parent / f"{src_path.name}_coco_dataset")
        )

    def _count_valid_pairs(self, annotation_dir: Path) -> tuple[int, int]:
        total_json = 0
        valid_pairs = 0
        for jf in annotation_dir.glob("*.json"):
            total_json += 1
            image_path = resolve_image_path(jf)
            if image_path is not None and image_path.exists():
                valid_pairs += 1
        return total_json, valid_pairs

    def _update_summary(self) -> None:
        src = self.anno_dir_edit.text().strip()
        if not src:
            self.summary_label.setText(
                "Select an annotation directory to preview split."
            )
            return

        src_path = Path(src).expanduser()
        if not src_path.is_dir():
            self.summary_label.setText("Annotation directory does not exist.")
            return

        total_json, valid_pairs = self._count_valid_pairs(src_path)
        train_ratio = float(self.train_percent_spin.value()) / 100.0
        train_n = int(round(valid_pairs * train_ratio))
        valid_n = max(0, valid_pairs - train_n)
        mode = str(self.mode_combo.currentData() or "segmentation")
        mode_text = (
            "strict COCO keypoints"
            if mode == "keypoints"
            else "COCO instance segmentation"
        )
        self.summary_label.setText(
            f"Detected {total_json} JSON files, {valid_pairs} valid JSON+image pairs. "
            f"Estimated split: train={train_n}, valid={valid_n}. Export mode: {mode_text}."
        )

    def accept(self) -> None:
        annotation_dir = self.anno_dir_edit.text().strip()
        output_dir = self.out_dir_edit.text().strip()
        labels_file = self.labels_edit.text().strip()

        if not annotation_dir:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing annotation directory",
                "Please select an annotation directory.",
            )
            return
        src_path = Path(annotation_dir).expanduser()
        if not src_path.is_dir():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid annotation directory",
                f"Directory not found:\n{src_path}",
            )
            return

        total_json, valid_pairs = self._count_valid_pairs(src_path)
        if total_json == 0 or valid_pairs == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No valid annotations",
                "No valid LabelMe JSON/image pairs found in the selected directory.",
            )
            return

        if labels_file:
            labels_path = Path(labels_file).expanduser()
            if not labels_path.is_file():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid labels file",
                    f"Labels file not found:\n{labels_path}",
                )
                return
            self.label_list_text = str(labels_path)
        else:
            self.label_list_text = None

        self.annotation_dir = str(src_path)
        self.out_dir = str(Path(output_dir).expanduser()) if output_dir else None
        self.num_train_frames = float(self.train_percent_spin.value()) / 100.0
        self.output_mode = str(self.mode_combo.currentData() or "segmentation")
        super().accept()
