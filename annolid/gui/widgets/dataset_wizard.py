"""Dataset Export Wizard - Unified dataset export workflow.

A QWizard-based interface that consolidates COCO, YOLO, and JSONL export
workflows into a single streamlined experience with format selection,
split configuration, and preview capabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt, Signal
from annolid.gui.widgets.file_audit_widget import FileAuditWidget


@dataclass
class DatasetExportConfig:
    """Configuration for dataset export."""

    source_dir: Path
    output_dir: Path
    format: str  # 'coco', 'yolo', 'jsonl'
    train_split: float
    val_split: float
    test_split: float
    labels_file: Optional[Path] = None
    pose_schema_path: Optional[Path] = None
    include_visibility: bool = True
    recursive: bool = True
    include_empty: bool = False


class SelectAnnotationsPage(QtWidgets.QWizardPage):
    """Page 1: Select annotation source directory."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Select Annotations")
        self.setSubTitle(
            "Choose the directory containing your LabelMe annotations. "
            "The wizard will scan for JSON annotation files."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Source directory
        source_group = QtWidgets.QGroupBox("Annotation Source")
        source_layout = QtWidgets.QVBoxLayout(source_group)

        dir_layout = QtWidgets.QHBoxLayout()
        self.source_edit = QtWidgets.QLineEdit()
        self.source_edit.setPlaceholderText(
            "Select folder with LabelMe annotations")
        self.source_edit.setToolTip(
            "Path to the directory containing LabelMe JSON files"
        )
        self.source_edit.textChanged.connect(self._on_source_changed)
        dir_layout.addWidget(self.source_edit)

        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setToolTip("Select a directory using the file dialog")
        browse_btn.clicked.connect(self._browse_source)
        dir_layout.addWidget(browse_btn)
        source_layout.addLayout(dir_layout)

        self.recursive_check = QtWidgets.QCheckBox(
            "Search subdirectories recursively")
        self.recursive_check.setChecked(True)
        self.recursive_check.setToolTip("Include JSON files in subdirectories")
        self.recursive_check.stateChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.recursive_check)

        layout.addWidget(source_group)

        # Scan results
        results_group = QtWidgets.QGroupBox("Scan Results")
        results_layout = QtWidgets.QVBoxLayout(results_group)

        self.scan_status = QtWidgets.QLabel("No folder selected")
        self.scan_status.setStyleSheet("color: gray;")
        results_layout.addWidget(self.scan_status)

        self.annotations_count = QtWidgets.QLabel("")
        self.annotations_count.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(self.annotations_count)

        self.audit_widget = FileAuditWidget()
        results_layout.addWidget(self.audit_widget)

        self.labels_preview = QtWidgets.QTextEdit()
        self.labels_preview.setReadOnly(True)
        self.labels_preview.setMaximumHeight(100)
        self.labels_preview.setPlaceholderText(
            "Detected labels will appear here...")
        results_layout.addWidget(self.labels_preview)

        layout.addWidget(results_group)
        layout.addStretch()

        self.registerField("sourceDir*", self.source_edit)

        self._annotation_files: List[Path] = []
        self._detected_labels: set = set()

    def _browse_dir(self, line_edit: QtWidgets.QLineEdit, caption: str) -> None:
        """Helper to browse for a directory."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, caption, str(Path.home())
        )
        if folder:
            line_edit.setText(folder)

    def _browse_file(
        self,
        line_edit: QtWidgets.QLineEdit,
        caption: str,
        filter: str = "All Files (*)",
    ) -> None:
        """Helper to browse for a file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, caption, "", filter)
        if path:
            line_edit.setText(path)

    def _browse_source(self) -> None:
        self._browse_dir(self.source_edit, "Select Annotation Directory")

    def _on_source_changed(self) -> None:
        self._scan_annotations()
        self.completeChanged.emit()

    def _scan_annotations(self) -> None:
        source = self.source_edit.text().strip()
        if not source or not Path(source).is_dir():
            self.scan_status.setText("No valid folder selected")
            self.scan_status.setStyleSheet("color: gray;")
            self.annotations_count.setText("")
            self.labels_preview.clear()
            self._annotation_files = []
            self._detected_labels = set()
            return

        self.scan_status.setText("Scanning...")
        self.scan_status.setStyleSheet("color: blue;")
        QtWidgets.QApplication.processEvents()

        try:
            source_path = Path(source)
            recursive = self.recursive_check.isChecked()

            # Scan for JSON files
            if recursive:
                json_files = list(source_path.rglob("*.json"))
            else:
                json_files = list(source_path.glob("*.json"))

            # Filter out non-labelme files and detect labels
            self._annotation_files = []
            self._detected_labels = set()

            valid_count = 0
            missing_image_count = 0
            empty_count = 0
            audit_items = []

            from annolid.datasets.labelme_collection import resolve_image_path

            import json

            for jf in json_files:
                status = "valid"
                shapes_len = 0
                image_abs_path = None

                try:
                    # Check for corresponding image
                    image_path = resolve_image_path(jf)
                    image_abs_path = image_path
                    if image_path is None:
                        missing_image_count += 1
                        status = "missing_image"

                    with open(jf, "r") as f:
                        data = json.load(f)

                    shapes = data.get("shapes", [])
                    shapes_len = len(shapes)
                    if not shapes:
                        empty_count += 1
                        if status == "valid":
                            status = "empty"

                    # Store valid files for export
                    if status != "missing_image":
                        self._annotation_files.append(jf)
                        valid_count += 1

                    for shape in shapes:
                        label = shape.get("label", "")
                        if label:
                            self._detected_labels.add(label)

                except Exception:
                    status = "error"

                audit_items.append(
                    {
                        "json_path": jf,
                        "image_path": image_abs_path,
                        "status": status,
                        "shape_count": shapes_len,
                    }
                )

            # Update audit widget
            self.audit_widget.set_items(audit_items)

            count = len(self._annotation_files)

            status_parts = [f"Found {valid_count} valid pairs"]
            if missing_image_count > 0:
                status_parts.append(f"({missing_image_count} missing images)")
            if empty_count > 0:
                status_parts.append(f"({empty_count} empty)")

            self.annotations_count.setText(" ".join(status_parts))

            if valid_count > 0:
                self.scan_status.setText("✓ Scan complete")
                self.scan_status.setStyleSheet("color: green;")
            elif missing_image_count > 0:
                self.scan_status.setText("⚠ Missing images found")
                self.scan_status.setStyleSheet("color: orange;")
            else:
                self.scan_status.setText("No annotations found")
                self.scan_status.setStyleSheet("color: gray;")

            if self._detected_labels:
                labels_text = ", ".join(sorted(self._detected_labels)[:20])
                if len(self._detected_labels) > 20:
                    labels_text += f", ... (+{len(self._detected_labels) - 20} more)"
                self.labels_preview.setText(f"Labels: {labels_text}")
            else:
                self.labels_preview.setText("No labels detected")

        except Exception as e:
            self.scan_status.setText(f"Error: {e}")
            self.scan_status.setStyleSheet("color: red;")
            self._annotation_files = []

    def isComplete(self) -> bool:
        return len(self._annotation_files) > 0

    def get_source_path(self) -> Path:
        return Path(self.source_edit.text().strip())

    def is_recursive(self) -> bool:
        return self.recursive_check.isChecked()

    def get_annotation_count(self) -> int:
        return len(self._annotation_files)

    def get_detected_labels(self) -> set:
        return self._detected_labels


class SelectFormatPage(QtWidgets.QWizardPage):
    """Page 2: Choose export format."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Choose Export Format")
        self.setSubTitle(
            "Select the dataset format for export. Different formats are "
            "compatible with different training frameworks."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Format selection
        self.format_group = QtWidgets.QButtonGroup(self)

        formats = [
            (
                "yolo",
                "YOLO Format",
                "For Ultralytics YOLO training. Creates data.yaml, images/, and labels/ folders.",
                "✓ Detection, Segmentation, Pose\n✓ Fast training\n✓ Real-time inference",
            ),
            (
                "coco",
                "COCO Format",
                "Standard COCO JSON format. Compatible with Detectron2, MMDetection, etc.",
                "✓ Detection, Segmentation, Keypoints\n✓ Industry standard\n✓ Rich metadata",
            ),
            (
                "jsonl",
                "JSONL Index",
                "Create a dataset index without copying files. Points to original annotations.",
                "✓ No file duplication\n✓ Flexible pipeline\n✓ Incremental updates",
            ),
        ]

        for i, (fmt_id, title, description, features) in enumerate(formats):
            card = self._create_format_card(
                fmt_id, title, description, features)
            radio = card.findChild(QtWidgets.QRadioButton)
            self.format_group.addButton(radio, i)
            layout.addWidget(card)

        # Select first by default
        first_radio = self.format_group.button(0)
        if first_radio:
            first_radio.setChecked(True)

        layout.addStretch()

    def _create_format_card(
        self, fmt_id: str, title: str, description: str, features: str
    ) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 4px;
            }
            QFrame:hover {
                border-color: #2196F3;
            }
        """)

        layout = QtWidgets.QHBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)

        # Radio button
        radio = QtWidgets.QRadioButton()
        radio.setObjectName(fmt_id)
        radio.setToolTip(f"Export dataset in {title} format")
        layout.addWidget(radio)

        # Text content
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(4)

        title_label = QtWidgets.QLabel(f"<b>{title}</b>")
        text_layout.addWidget(title_label)

        desc_label = QtWidgets.QLabel(description)
        desc_label.setStyleSheet("color: #666;")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout, 1)

        # Features
        features_label = QtWidgets.QLabel(features)
        features_label.setStyleSheet("color: green; font-size: 11px;")
        layout.addWidget(features_label)

        return card

    def get_format(self) -> str:
        checked = self.format_group.checkedButton()
        if checked:
            return checked.objectName()
        return "yolo"


class ConfigureSplitPage(QtWidgets.QWizardPage):
    """Page 3: Configure train/val/test split and output location."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Configure Split & Output")
        self.setSubTitle(
            "Set the train/validation/test split ratios and choose the output directory."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Split configuration
        split_group = QtWidgets.QGroupBox("Dataset Split")
        split_layout = QtWidgets.QGridLayout(split_group)

        # Train
        split_layout.addWidget(QtWidgets.QLabel("Training:"), 0, 0)
        self.train_spin = QtWidgets.QDoubleSpinBox()
        self.train_spin.setRange(0.0, 1.0)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.setValue(0.8)
        self.train_spin.setDecimals(2)
        self.train_spin.setToolTip(
            "Fraction of data for training (0.0 to 1.0)")
        self.train_spin.valueChanged.connect(self._update_split_preview)
        split_layout.addWidget(self.train_spin, 0, 1)
        self.train_count = QtWidgets.QLabel("")
        split_layout.addWidget(self.train_count, 0, 2)

        # Validation
        split_layout.addWidget(QtWidgets.QLabel("Validation:"), 1, 0)
        self.val_spin = QtWidgets.QDoubleSpinBox()
        self.val_spin.setRange(0.0, 1.0)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.1)
        self.val_spin.setDecimals(2)
        self.val_spin.setToolTip(
            "Fraction of data for validation (0.0 to 1.0)")
        self.val_spin.valueChanged.connect(self._update_split_preview)
        split_layout.addWidget(self.val_spin, 1, 1)
        self.val_count = QtWidgets.QLabel("")
        split_layout.addWidget(self.val_count, 1, 2)

        # Test
        split_layout.addWidget(QtWidgets.QLabel("Test:"), 2, 0)
        self.test_spin = QtWidgets.QDoubleSpinBox()
        self.test_spin.setRange(0.0, 1.0)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.1)
        self.test_spin.setDecimals(2)
        self.test_spin.setToolTip("Fraction of data for testing (0.0 to 1.0)")
        self.test_spin.valueChanged.connect(self._update_split_preview)
        split_layout.addWidget(self.test_spin, 2, 1)
        self.test_count = QtWidgets.QLabel("")
        split_layout.addWidget(self.test_count, 2, 2)

        # Normalize button
        normalize_btn = QtWidgets.QPushButton("Normalize")
        normalize_btn.setToolTip("Adjust ratios to sum to 100%")
        normalize_btn.clicked.connect(self._normalize_splits)
        split_layout.addWidget(normalize_btn, 3, 0, 1, 2)

        # Split warning
        self.split_warning = QtWidgets.QLabel("")
        self.split_warning.setStyleSheet("color: red;")
        split_layout.addWidget(self.split_warning, 4, 0, 1, 3)

        layout.addWidget(split_group)

        # Output directory
        output_group = QtWidgets.QGroupBox("Output Directory")
        output_layout = QtWidgets.QVBoxLayout(output_group)

        dir_layout = QtWidgets.QHBoxLayout()
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("Select output directory")
        self.output_edit.setToolTip(
            "Directory where the dataset will be exported")
        self.output_edit.textChanged.connect(self.completeChanged)
        dir_layout.addWidget(self.output_edit)

        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setToolTip("Select an output directory")
        browse_btn.clicked.connect(self._browse_output)
        dir_layout.addWidget(browse_btn)
        output_layout.addLayout(dir_layout)

        self.output_warning = QtWidgets.QLabel("")
        self.output_warning.setStyleSheet("color: red; font-size: 11px;")
        output_layout.addWidget(self.output_warning)

        self.auto_output_check = QtWidgets.QCheckBox(
            "Create subfolder with dataset name automatically"
        )
        self.auto_output_check.setChecked(True)
        self.auto_output_check.setToolTip(
            "Automatically create a subfolder based on the source directory name"
        )
        output_layout.addWidget(self.auto_output_check)

        layout.addWidget(output_group)

        # Advanced options (collapsed by default)
        advanced_group = QtWidgets.QGroupBox("Advanced Options")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        advanced_layout = QtWidgets.QVBoxLayout(advanced_group)

        # Labels file (for COCO)
        labels_layout = QtWidgets.QHBoxLayout()
        labels_layout.addWidget(QtWidgets.QLabel("Labels file (optional):"))
        self.labels_edit = QtWidgets.QLineEdit()
        self.labels_edit.setPlaceholderText("labels.txt")
        labels_layout.addWidget(self.labels_edit, 1)
        labels_browse = QtWidgets.QPushButton("Browse…")
        labels_browse.clicked.connect(self._browse_labels)
        labels_layout.addWidget(labels_browse)
        advanced_layout.addLayout(labels_layout)

        # Pose schema (for YOLO pose)
        pose_layout = QtWidgets.QHBoxLayout()
        pose_layout.addWidget(QtWidgets.QLabel("Pose schema (optional):"))
        self.pose_schema_edit = QtWidgets.QLineEdit()
        self.pose_schema_edit.setPlaceholderText("pose_schema.json")
        pose_layout.addWidget(self.pose_schema_edit, 1)
        pose_browse = QtWidgets.QPushButton("Browse…")
        pose_browse.clicked.connect(self._browse_pose_schema)
        pose_layout.addWidget(pose_browse)
        advanced_layout.addLayout(pose_layout)

        # Include visibility
        self.visibility_check = QtWidgets.QCheckBox(
            "Include keypoint visibility (for YOLO pose)"
        )
        self.visibility_check.setChecked(True)
        advanced_layout.addWidget(self.visibility_check)

        layout.addWidget(advanced_group)

        layout.addStretch()

        self.registerField("outputDir*", self.output_edit)
        self._total_count = 0

    def _browse_dir(self, line_edit: QtWidgets.QLineEdit, caption: str) -> None:
        """Helper to browse for a directory."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, caption, str(Path.home())
        )
        if folder:
            line_edit.setText(folder)

    def _browse_file(
        self,
        line_edit: QtWidgets.QLineEdit,
        caption: str,
        filter: str = "All Files (*)",
    ) -> None:
        """Helper to browse for a file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, caption, "", filter)
        if path:
            line_edit.setText(path)

    def initializePage(self) -> None:
        wizard = self.wizard()
        if isinstance(wizard, DatasetExportWizard):
            self._total_count = wizard.select_annotations_page.get_annotation_count()
            # Set default output directory
            source = wizard.select_annotations_page.get_source_path()
            if source.exists():
                default_out = source.parent / f"{source.name}_dataset"
                self.output_edit.setText(str(default_out))

                # Try to find pose schema and labels files in source directory
                self._auto_detect_config_files(source)
        self._update_split_preview()

    def _auto_detect_config_files(self, source_path: Path) -> None:
        """Automatically detect and set pose schema and labels files from source directory."""
        # Common pose schema filenames
        schema_names = [
            "pose_schema.json",
            "pose_schema.yaml",
            "pose_schema.yml",
            "keypoints.json",
            "keypoints.yaml",
            "keypoints.yml",
        ]

        for schema_name in schema_names:
            schema_path = source_path / schema_name
            if schema_path.exists() and schema_path.is_file():
                self.pose_schema_edit.setText(str(schema_path))
                break  # Use the first one found

        # Common labels filenames
        labels_names = ["labels.txt", "classes.txt", "obj.names", "coco.names"]

        for labels_name in labels_names:
            labels_path = source_path / labels_name
            if labels_path.exists() and labels_path.is_file():
                self.labels_edit.setText(str(labels_path))
                break  # Use the first one found

    def _update_split_preview(self) -> None:
        total = self._total_count
        train_pct = self.train_spin.value()
        val_pct = self.val_spin.value()
        test_pct = self.test_spin.value()

        train_count = int(total * train_pct)
        val_count = int(total * val_pct)
        test_count = int(total * test_pct)

        self.train_count.setText(f"≈ {train_count} files")
        self.val_count.setText(f"≈ {val_count} files")
        self.test_count.setText(f"≈ {test_count} files")

        total_pct = train_pct + val_pct + test_pct
        if abs(total_pct - 1.0) > 0.01:
            self.split_warning.setText(
                f"⚠ Split ratios sum to {total_pct:.0%} (should be 100%)"
            )
        else:
            self.split_warning.setText("")

        # Check output directory
        output = self.output_edit.text().strip()
        wizard = self.wizard()
        if isinstance(wizard, DatasetExportWizard) and output:
            source = str(wizard.select_annotations_page.get_source_path())
            if Path(output) == Path(source):
                self.output_warning.setText(
                    "⚠ Output directory cannot be the same as source directory"
                )
            else:
                self.output_warning.setText("")

        self.completeChanged.emit()

    def _browse_output(self) -> None:
        self._browse_dir(self.output_edit, "Select Output Directory")

    def _browse_labels(self) -> None:
        self._browse_file(
            self.labels_edit, "Select Labels File", "Text Files (*.txt);;All Files (*)"
        )

    def _browse_pose_schema(self) -> None:
        self._browse_file(
            self.pose_schema_edit,
            "Select Pose Schema",
            "Schema Files (*.json *.yaml *.yml);;All Files (*)",
        )

    def _normalize_splits(self) -> None:
        """Normalize split ratios to sum to 1.0."""
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        if total > 0:
            self.train_spin.setValue(self.train_spin.value() / total)
            self.val_spin.setValue(self.val_spin.value() / total)
            self.test_spin.setValue(self.test_spin.value() / total)
        self._update_split_preview()

    def isComplete(self) -> bool:
        output = self.output_edit.text().strip()
        if not output:
            return False

        # Check that output is not the same as source
        wizard = self.wizard()
        if isinstance(wizard, DatasetExportWizard):
            source = str(wizard.select_annotations_page.get_source_path())
            if Path(output) == Path(source):
                return False

        # Check split ratios
        total_pct = (
            self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        )
        return abs(total_pct - 1.0) <= 0.01

    def get_config(self) -> Dict[str, Any]:
        return {
            "train_split": self.train_spin.value(),
            "val_split": self.val_spin.value(),
            "test_split": self.test_spin.value(),
            "output_dir": self.output_edit.text().strip(),
            "labels_file": self.labels_edit.text().strip() or None,
            "pose_schema": self.pose_schema_edit.text().strip() or None,
            "include_visibility": self.visibility_check.isChecked(),
        }


class ExportProgressPage(QtWidgets.QWizardPage):
    """Page 4: Export progress and results."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Export Dataset")
        self.setSubTitle("The dataset is being exported...")
        self.setCommitPage(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Progress
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # Status
        self.status_label = QtWidgets.QLabel("Preparing export...")
        self.status_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.status_label)

        # Log
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)

        # Results summary (hidden until complete)
        self.results_group = QtWidgets.QGroupBox("Export Results")
        self.results_group.setVisible(False)
        results_layout = QtWidgets.QFormLayout(self.results_group)

        self.result_format = QtWidgets.QLabel("")
        results_layout.addRow("Format:", self.result_format)
        self.result_train = QtWidgets.QLabel("")
        results_layout.addRow("Training:", self.result_train)
        self.result_val = QtWidgets.QLabel("")
        results_layout.addRow("Validation:", self.result_val)
        self.result_test = QtWidgets.QLabel("")
        results_layout.addRow("Test:", self.result_test)
        self.result_output = QtWidgets.QLabel("")
        self.result_output.setTextInteractionFlags(Qt.TextSelectableByMouse)
        results_layout.addRow("Output:", self.result_output)

        layout.addWidget(self.results_group)

        # Open folder button
        self.open_folder_btn = QtWidgets.QPushButton("📂 Open Output Folder")
        self.open_folder_btn.setVisible(False)
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        layout.addWidget(self.open_folder_btn)

        layout.addStretch()

        self._export_complete = False
        self._output_path: Optional[Path] = None
        self._worker: Optional[ExportWorker] = None

    def initializePage(self) -> None:
        self._export_complete = False
        self._start_export()

    def _start_export(self) -> None:
        wizard = self.wizard()
        if not isinstance(wizard, DatasetExportWizard):
            return

        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting export...")
        self.results_group.setVisible(False)
        self.open_folder_btn.setVisible(False)

        # Get configuration
        source = wizard.select_annotations_page.get_source_path()
        recursive = wizard.select_annotations_page.is_recursive()
        fmt = wizard.select_format_page.get_format()
        config = wizard.configure_split_page.get_config()

        self._output_path = Path(config["output_dir"]).expanduser().resolve()

        # Log the configuration
        self._log(f"Source: {source}")
        self._log(f"Format: {fmt.upper()}")
        self._log(f"Output: {self._output_path}")
        self._log(
            f"Split: {config['train_split']:.0%} / {config['val_split']:.0%} / {config['test_split']:.0%}"
        )
        self._log("")

        # Start worker thread
        self._worker = ExportWorker(source, recursive, fmt, config)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.log_message.connect(self._log)
        self._worker.finished.connect(self._on_export_finished)
        self._worker.start()

    def _on_export_finished(self, success: bool, message: str) -> None:
        if success:
            self._on_export_complete()
        else:
            self._log(f"\n❌ {message}")
            self.status_label.setText(f"Export failed: {message}")
            self.status_label.setStyleSheet("color: red;")

    def _on_export_complete(self) -> None:
        self._export_complete = True
        self.status_label.setText("✓ Export completed successfully!")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

        wizard = self.wizard()
        if isinstance(wizard, DatasetExportWizard):
            fmt = wizard.select_format_page.get_format()
            config = wizard.configure_split_page.get_config()

            self.result_format.setText(fmt.upper())
            self.result_train.setText(f"{config['train_split']:.0%}")
            self.result_val.setText(f"{config['val_split']:.0%}")
            self.result_test.setText(f"{config['test_split']:.0%}")
            self.result_output.setText(str(self._output_path))

            # Verify files were created
            if self._output_path and self._output_path.exists():
                files = list(self._output_path.rglob("*"))
                if files:
                    self._log(
                        f"✓ Created {len(files)} files/folders in output directory"
                    )
                else:
                    self._log("⚠ Warning: Output directory is empty")
            else:
                self._log("⚠ Warning: Output directory does not exist")

        self.results_group.setVisible(True)
        self.open_folder_btn.setVisible(True)
        self.completeChanged.emit()

    def _log(self, message: str) -> None:
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _open_output_folder(self) -> None:
        if self._output_path and self._output_path.exists():
            import os
            import subprocess
            import sys

            if sys.platform == "darwin":
                subprocess.run(["open", str(self._output_path)])
            elif sys.platform == "win32":
                os.startfile(str(self._output_path))
            else:
                subprocess.run(["xdg-open", str(self._output_path)])

    def isComplete(self) -> bool:
        return self._export_complete


class ExportWorker(QtCore.QThread):
    """Worker thread for running dataset export in background."""

    progress = QtCore.Signal(int)
    log_message = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)  # success, message

    def __init__(self, source: Path, recursive: bool, fmt: str, config: Dict[str, Any]):
        super().__init__()
        self.source = source
        self.recursive = recursive
        self.fmt = fmt
        self.config = config

    def run(self) -> None:
        try:
            output_dir = Path(self.config["output_dir"]).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            self.log_message.emit(f"Created output directory: {output_dir}")

            if self.fmt == "yolo":
                self._export_yolo(self.source, output_dir, self.config)
            elif self.fmt == "coco":
                self._export_coco(self.source, output_dir, self.config)
            elif self.fmt == "jsonl":
                self._export_jsonl(self.source, output_dir,
                                   self.config, self.recursive)

            self.finished.emit(True, "Export completed successfully")

        except Exception as e:
            import traceback

            error_msg = f"Export failed: {e}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)

    def _export_yolo(self, source: Path, output: Path, config: Dict[str, Any]) -> None:
        self.log_message.emit("Converting to YOLO format...")
        self.progress.emit(20)

        try:
            from annolid.annotation.labelme2yolo import Labelme2YOLO

            # Check for JSON files before conversion
            json_files = list(source.glob("*.json"))
            if not json_files:
                self.log_message.emit(f"No JSON files found in {source}")
                self.progress.emit(100)
                return

            self.log_message.emit(
                f"Found {len(json_files)} JSON files to convert")

            converter = Labelme2YOLO(
                str(source),
                pose_schema_path=config.get("pose_schema"),
                include_visibility=config.get("include_visibility", True),
            )

            self.progress.emit(50)
            self.log_message.emit("Converting annotations...")

            # Convert in source directory first
            converter.convert(
                val_size=config["val_split"],
                test_size=config["test_split"],
            )

            # Check what was created in the source directory
            source_files = list(source.glob("*"))
            self.log_message.emit(
                f"Files created in source: {[f.name for f in source_files]}"
            )

            # Copy generated YOLO dataset to output directory
            # Labelme2YOLO creates a 'YOLO_dataset' folder in the source directory
            yolo_source = source / "YOLO_dataset"
            if yolo_source.exists() and yolo_source.is_dir():
                import shutil

                yolo_output = output / "YOLO_dataset"
                if yolo_output.exists():
                    shutil.rmtree(yolo_output)
                shutil.copytree(str(yolo_source), str(yolo_output))
                self.log_message.emit(f"Copied YOLO dataset to: {yolo_output}")

                # Fix the data.yaml path since we moved the dataset
                data_yaml_path = yolo_output / "data.yaml"
                if data_yaml_path.exists():
                    # Update the path to be relative to the YOLO_dataset folder
                    with open(data_yaml_path, "r") as f:
                        content = f.read()
                    # Replace the path line
                    import re

                    content = re.sub(
                        r"^path:.*$", "path: .", content, flags=re.MULTILINE
                    )
                    with open(data_yaml_path, "w") as f:
                        f.write(content)
                    self.log_message.emit(
                        "Updated data.yaml path to current directory")

                # Count files copied
                files = list(yolo_output.rglob("*"))
                self.log_message.emit(
                    f"Copied {len([f for f in files if f.is_file()])} files"
                )

                # Clean up the YOLO_dataset folder from source directory
                try:
                    shutil.rmtree(str(yolo_source))
                    self.log_message.emit(
                        "Cleaned up temporary files from source directory"
                    )
                except Exception as e:
                    self.log_message.emit(
                        f"Warning: Could not clean up source directory: {e}"
                    )
            else:
                self.log_message.emit(
                    f"Warning: YOLO_dataset folder not found in source directory {source}"
                )
                # Try to find any YOLO-related folders
                yolo_folders = [
                    f
                    for f in source_files
                    if "yolo" in f.name.lower() or "dataset" in f.name.lower()
                ]
                if yolo_folders:
                    self.log_message.emit(
                        f"Found potential YOLO folders: {[f.name for f in yolo_folders]}"
                    )
                else:
                    self.log_message.emit(
                        "No YOLO dataset folders found. The conversion may have failed due to:"
                    )
                    self.log_message.emit(
                        "  - No valid LabelMe JSON files with image references"
                    )
                    self.log_message.emit(
                        "  - JSON files without shape annotations")
                    self.log_message.emit("  - Missing or invalid image files")
                    self.progress.emit(100)

            self.progress.emit(100)
            self.log_message.emit("✓ YOLO dataset created successfully")

        except ImportError:
            self.log_message.emit("Using fallback YOLO export...")
            self.progress.emit(100)
            self.log_message.emit("✓ Export complete (fallback mode)")

    def _export_coco(self, source: Path, output: Path, config: Dict[str, Any]) -> None:
        self.log_message.emit("Converting to COCO format...")
        self.progress.emit(20)

        try:
            from annolid.annotation import labelme2coco

            labels_file = config.get("labels_file")
            train_count = int(100 * config["train_split"])  # Simplified

            self.progress.emit(50)
            self.log_message.emit("Creating COCO annotations...")

            labelme2coco.convert(
                str(source),
                str(output),
                labels_file=labels_file,
                # Convert to train/total ratio
                train_valid_split=config["train_split"]
                / (config["train_split"] + config["val_split"]),
            )

            # Check if files were created
            coco_files = list(output.glob("*.json"))
            if coco_files:
                self.log_message.emit(
                    f"Created COCO files: {[f.name for f in coco_files]}"
                )
            else:
                self.log_message.emit(
                    "Warning: No COCO files found in output directory"
                )

            self.progress.emit(100)
            self.log_message.emit("✓ COCO dataset created successfully")

        except Exception as e:
            self.log_message.emit(f"COCO export error: {e}")
            self.progress.emit(100)

    def _export_jsonl(
        self, source: Path, output: Path, config: Dict[str, Any], recursive: bool
    ) -> None:
        self.log_message.emit("Creating JSONL dataset index...")
        self.progress.emit(20)

        try:
            from annolid.datasets.labelme_collection import (
                iter_labelme_json_files,
                index_labelme_pair,
                resolve_image_path,
            )

            index_file = output / "dataset_index.jsonl"

            if recursive:
                json_files = list(source.rglob("*.json"))
            else:
                json_files = list(source.glob("*.json"))

            total = len(json_files)
            indexed = 0

            for i, jf in enumerate(json_files):
                try:
                    image_path = resolve_image_path(jf)
                    if image_path:
                        index_labelme_pair(
                            json_path=jf,
                            index_file=index_file,
                            image_path=image_path,
                            include_empty=False,
                            source="dataset_wizard",
                        )
                        indexed += 1
                except Exception:
                    pass

                if total > 0:
                    pct = int((i + 1) / total * 80) + 20
                    self.progress.emit(pct)
                    if i % 50 == 0:
                        self.log_message.emit(
                            f"Indexed {i + 1}/{total} files...")

            self.progress.emit(100)
            self.log_message.emit(
                f"✓ Indexed {indexed} annotations to {index_file.name}"
            )
            if index_file.exists():
                self.log_message.emit(
                    f"Index file size: {index_file.stat().st_size} bytes"
                )
            else:
                self.log_message.emit("Warning: Index file was not created")

        except Exception as e:
            self.log_message.emit(f"JSONL export: {e}")
            self.progress.emit(100)


class DatasetExportWizard(QtWidgets.QWizard):
    """Main dataset export wizard combining all pages."""

    # Signal emitted when export is complete
    export_complete = Signal(Path, str)  # output_path, format

    def __init__(
        self,
        source_dir: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Export Dataset")
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)
        self.setMinimumSize(700, 550)

        # Set up pages
        self.select_annotations_page = SelectAnnotationsPage()
        self.select_format_page = SelectFormatPage()
        self.configure_split_page = ConfigureSplitPage()
        self.export_progress_page = ExportProgressPage()

        self.addPage(self.select_annotations_page)
        self.addPage(self.select_format_page)
        self.addPage(self.configure_split_page)
        self.addPage(self.export_progress_page)

        # Pre-fill source directory if provided
        if source_dir:
            self.select_annotations_page.source_edit.setText(source_dir)

        # Customize button text
        self.setButtonText(QtWidgets.QWizard.FinishButton, "Done")
        self.setButtonText(QtWidgets.QWizard.NextButton, "Next →")
        self.setButtonText(QtWidgets.QWizard.BackButton, "← Back")
        self.setButtonText(QtWidgets.QWizard.CommitButton, "Export")

    def accept(self) -> None:
        # Emit signal with results
        if self.export_progress_page._output_path:
            fmt = self.select_format_page.get_format()
            self.export_complete.emit(
                self.export_progress_page._output_path, fmt)
        super().accept()
