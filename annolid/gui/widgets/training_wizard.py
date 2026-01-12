"""Training Wizard - Streamlined model training workflow.

A QWizard-based interface that guides users through dataset selection,
backend choice, parameter configuration, and training launch with
integrated progress monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, Signal

from annolid.utils.devices import get_device
from annolid.gui.models_registry import PATCH_SIMILARITY_MODELS, PATCH_SIMILARITY_DEFAULT_MODEL


YOLO_TASKS = {
    "Detection": "",
    "Instance Segmentation": "-seg",
    "Pose Estimation": "-pose",
}

YOLO_SIZES = ("n", "s", "m", "l", "x")


class SelectDatasetPage(QtWidgets.QWizardPage):
    """Page 1: Select training dataset."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Select Dataset")
        self.setSubTitle(
            "Choose the dataset configuration file for training. "
            "Use the Dataset Export wizard to create one if needed."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Dataset file selection
        file_group = QtWidgets.QGroupBox("Dataset Configuration")
        file_layout = QtWidgets.QVBoxLayout(file_group)

        # YOLO data.yaml
        yolo_layout = QtWidgets.QHBoxLayout()
        yolo_layout.addWidget(QtWidgets.QLabel("YOLO data.yaml:"))
        self.yolo_data_edit = QtWidgets.QLineEdit()
        self.yolo_data_edit.setPlaceholderText("Select data.yaml file")
        self.yolo_data_edit.textChanged.connect(self._validate_dataset)
        yolo_layout.addWidget(self.yolo_data_edit, 1)
        browse_yolo = QtWidgets.QPushButton("Browse…")
        browse_yolo.clicked.connect(self._browse_yolo_data)
        yolo_layout.addWidget(browse_yolo)
        file_layout.addLayout(yolo_layout)

        # COCO annotations (alternative)
        coco_layout = QtWidgets.QHBoxLayout()
        coco_layout.addWidget(QtWidgets.QLabel("Or COCO folder:"))
        self.coco_dir_edit = QtWidgets.QLineEdit()
        self.coco_dir_edit.setPlaceholderText(
            "Folder with train.json/val.json")
        self.coco_dir_edit.textChanged.connect(self._validate_dataset)
        coco_layout.addWidget(self.coco_dir_edit, 1)
        browse_coco = QtWidgets.QPushButton("Browse…")
        browse_coco.clicked.connect(self._browse_coco_dir)
        coco_layout.addWidget(browse_coco)
        file_layout.addLayout(coco_layout)

        layout.addWidget(file_group)

        # Dataset info preview
        info_group = QtWidgets.QGroupBox("Dataset Information")
        info_layout = QtWidgets.QFormLayout(info_group)

        self.info_classes = QtWidgets.QLabel("—")
        info_layout.addRow("Classes:", self.info_classes)
        self.info_train = QtWidgets.QLabel("—")
        info_layout.addRow("Training images:", self.info_train)
        self.info_val = QtWidgets.QLabel("—")
        info_layout.addRow("Validation images:", self.info_val)
        self.info_type = QtWidgets.QLabel("—")
        info_layout.addRow("Task type:", self.info_type)

        layout.addWidget(info_group)

        # Quick create button
        create_layout = QtWidgets.QHBoxLayout()
        create_layout.addStretch()
        self.create_dataset_btn = QtWidgets.QPushButton("Create Dataset…")
        self.create_dataset_btn.setToolTip("Open the Dataset Export wizard")
        self.create_dataset_btn.clicked.connect(self._open_dataset_wizard)
        create_layout.addWidget(self.create_dataset_btn)
        layout.addLayout(create_layout)

        layout.addStretch()

        self.registerField("datasetPath*", self.yolo_data_edit)

    def _browse_yolo_data(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select YOLO data.yaml", "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self.yolo_data_edit.setText(path)
            self.coco_dir_edit.clear()

    def _browse_coco_dir(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select COCO Dataset Folder"
        )
        if folder:
            self.coco_dir_edit.setText(folder)
            self.yolo_data_edit.clear()

    def _validate_dataset(self) -> None:
        yolo_path = self.yolo_data_edit.text().strip()
        coco_path = self.coco_dir_edit.text().strip()

        self.info_classes.setText("—")
        self.info_train.setText("—")
        self.info_val.setText("—")
        self.info_type.setText("—")

        if yolo_path and Path(yolo_path).exists():
            self._parse_yolo_yaml(Path(yolo_path))
        elif coco_path and Path(coco_path).is_dir():
            self._parse_coco_dir(Path(coco_path))

        self.completeChanged.emit()

    def _parse_yolo_yaml(self, path: Path) -> None:
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)

            names = data.get('names', [])
            if isinstance(names, dict):
                names = list(names.values())
            self.info_classes.setText(
                f"{len(names)}: {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")

            # Check for image counts
            train_path = data.get('train', '')
            val_path = data.get('val', '')
            base = path.parent

            train_count = self._count_images(
                base / train_path) if train_path else 0
            val_count = self._count_images(base / val_path) if val_path else 0

            self.info_train.setText(str(train_count) if train_count else "—")
            self.info_val.setText(str(val_count) if val_count else "—")

            # Detect task type
            if 'kpt_shape' in data:
                self.info_type.setText("Pose Estimation")
            elif any('segment' in str(v).lower() for v in data.values()):
                self.info_type.setText("Instance Segmentation")
            else:
                self.info_type.setText("Detection")

        except Exception as e:
            self.info_classes.setText(f"Error: {e}")

    def _parse_coco_dir(self, path: Path) -> None:
        try:
            train_json = path / "train.json"
            val_json = path / "val.json"

            if train_json.exists():
                import json
                with open(train_json, 'r') as f:
                    data = json.load(f)
                cats = data.get('categories', [])
                self.info_classes.setText(f"{len(cats)} categories")
                self.info_train.setText(
                    f"{len(data.get('images', []))} images")
                self.info_type.setText("COCO Detection/Segmentation")

            if val_json.exists():
                import json
                with open(val_json, 'r') as f:
                    data = json.load(f)
                self.info_val.setText(f"{len(data.get('images', []))} images")

        except Exception as e:
            self.info_classes.setText(f"Error: {e}")

    def _count_images(self, path: Path) -> int:
        if not path.exists():
            return 0
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        if path.is_file():
            # It's a list file
            return sum(1 for line in path.read_text().splitlines() if line.strip())
        return sum(1 for f in path.iterdir() if f.suffix.lower() in extensions)

    def _open_dataset_wizard(self) -> None:
        from annolid.gui.widgets.dataset_wizard import DatasetExportWizard
        wizard = DatasetExportWizard(parent=self)
        if wizard.exec_() == QtWidgets.QDialog.Accepted:
            # If export was successful, try to find the data.yaml
            if wizard.export_progress_page._output_path:
                data_yaml = wizard.export_progress_page._output_path / "data.yaml"
                if data_yaml.exists():
                    self.yolo_data_edit.setText(str(data_yaml))

    def isComplete(self) -> bool:
        yolo_path = self.yolo_data_edit.text().strip()
        coco_path = self.coco_dir_edit.text().strip()
        return (yolo_path and Path(yolo_path).exists()) or \
               (coco_path and Path(coco_path).is_dir())

    def get_dataset_path(self) -> str:
        return self.yolo_data_edit.text().strip() or self.coco_dir_edit.text().strip()

    def get_dataset_type(self) -> str:
        if self.yolo_data_edit.text().strip():
            return "yolo"
        return "coco"


class SelectBackendPage(QtWidgets.QWizardPage):
    """Page 2: Choose training backend (YOLO, DINO KPSEG, MaskRCNN)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Choose Training Backend")
        self.setSubTitle(
            "Select the model architecture and training framework. "
            "Each backend has different strengths and use cases."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        self.backend_group = QtWidgets.QButtonGroup(self)

        backends = [
            (
                "yolo",
                "🚀 YOLO (Ultralytics)",
                "Fast, state-of-the-art detection and segmentation. "
                "Best for real-time inference and most use cases.",
                ["Detection", "Segmentation", "Pose Estimation",
                    "Fast Training", "Real-time Inference"],
                True,  # recommended
            ),
            (
                "dino_kpseg",
                "🦕 DINO KPSEG",
                "Self-supervised keypoint detection using DINOv2 features. "
                "Excellent for few-shot learning with limited annotations.",
                ["Keypoint Detection", "Few-shot Learning", "Transfer Learning"],
                False,
            ),
            (
                "maskrcnn",
                "🎭 Mask R-CNN (Detectron2)",
                "Classic instance segmentation architecture. "
                "Good for complex segmentation tasks with multiple instances.",
                ["Instance Segmentation", "Object Detection", "Research Baseline"],
                False,
            ),
        ]

        for i, (backend_id, title, description, features, recommended) in enumerate(backends):
            card = self._create_backend_card(
                backend_id, title, description, features, recommended
            )
            radio = card.findChild(QtWidgets.QRadioButton)
            self.backend_group.addButton(radio, i)
            layout.addWidget(card)

        # Select YOLO by default
        first_radio = self.backend_group.button(0)
        if first_radio:
            first_radio.setChecked(True)

        layout.addStretch()

    def _create_backend_card(
        self,
        backend_id: str,
        title: str,
        description: str,
        features: List[str],
        recommended: bool,
    ) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setFrameStyle(QtWidgets.QFrame.StyledPanel)

        border_color = "#2196F3" if recommended else "#e0e0e0"
        card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 2px solid {border_color};
                border-radius: 8px;
                padding: 4px;
            }}
            QFrame:hover {{
                border-color: #1976D2;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)

        # Header with radio and title
        header = QtWidgets.QHBoxLayout()
        radio = QtWidgets.QRadioButton()
        radio.setObjectName(backend_id)
        header.addWidget(radio)

        title_label = QtWidgets.QLabel(f"<b>{title}</b>")
        header.addWidget(title_label)

        if recommended:
            badge = QtWidgets.QLabel("Recommended")
            badge.setStyleSheet("""
                background-color: #2196F3;
                color: white;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 11px;
            """)
            header.addWidget(badge)

        header.addStretch()
        layout.addLayout(header)

        # Description
        desc = QtWidgets.QLabel(description)
        desc.setStyleSheet("color: #666; margin-left: 24px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Features
        features_text = " • ".join(features)
        features_label = QtWidgets.QLabel(f"✓ {features_text}")
        features_label.setStyleSheet(
            "color: green; font-size: 11px; margin-left: 24px;")
        layout.addWidget(features_label)

        return card

    def get_backend(self) -> str:
        checked = self.backend_group.checkedButton()
        if checked:
            return checked.objectName()
        return "yolo"


class ConfigureParametersPage(QtWidgets.QWizardPage):
    """Page 3: Configure training parameters."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Configure Training")
        self.setSubTitle(
            "Set the training parameters. Default values work well for most cases."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Create stacked widget for backend-specific options
        self.stack = QtWidgets.QStackedWidget()

        # YOLO options
        self.yolo_widget = self._create_yolo_options()
        self.stack.addWidget(self.yolo_widget)

        # DINO KPSEG options
        self.dino_widget = self._create_dino_options()
        self.stack.addWidget(self.dino_widget)

        # MaskRCNN options
        self.maskrcnn_widget = self._create_maskrcnn_options()
        self.stack.addWidget(self.maskrcnn_widget)

        layout.addWidget(self.stack)

        # Common options
        common_group = QtWidgets.QGroupBox("Output")
        common_layout = QtWidgets.QFormLayout(common_group)

        dir_layout = QtWidgets.QHBoxLayout()
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText("Training outputs (optional)")
        dir_layout.addWidget(self.output_dir_edit, 1)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output)
        dir_layout.addWidget(browse_btn)
        common_layout.addRow("Output directory:", dir_layout)

        layout.addWidget(common_group)

    def initializePage(self) -> None:
        wizard = self.wizard()
        if isinstance(wizard, TrainingWizard):
            backend = wizard.select_backend_page.get_backend()
            if backend == "yolo":
                self.stack.setCurrentWidget(self.yolo_widget)
            elif backend == "dino_kpseg":
                self.stack.setCurrentWidget(self.dino_widget)
            else:
                self.stack.setCurrentWidget(self.maskrcnn_widget)

    def _create_yolo_options(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Model selection
        model_group = QtWidgets.QGroupBox("YOLO Model")
        model_layout = QtWidgets.QHBoxLayout(model_group)

        model_layout.addWidget(QtWidgets.QLabel("Task:"))
        self.yolo_task_combo = QtWidgets.QComboBox()
        self.yolo_task_combo.addItems(list(YOLO_TASKS.keys()))
        self.yolo_task_combo.setCurrentText("Instance Segmentation")
        model_layout.addWidget(self.yolo_task_combo)

        model_layout.addWidget(QtWidgets.QLabel("Size:"))
        self.yolo_size_combo = QtWidgets.QComboBox()
        self.yolo_size_combo.addItems([s.upper() for s in YOLO_SIZES])
        model_layout.addWidget(self.yolo_size_combo)

        self.yolo_model_label = QtWidgets.QLabel("yolo11n-seg.pt")
        self.yolo_model_label.setStyleSheet("font-weight: bold;")
        model_layout.addWidget(self.yolo_model_label)

        self.yolo_task_combo.currentTextChanged.connect(
            self._update_yolo_model)
        self.yolo_size_combo.currentTextChanged.connect(
            self._update_yolo_model)

        model_layout.addStretch()
        layout.addWidget(model_group)

        # Training params
        train_group = QtWidgets.QGroupBox("Training Parameters")
        train_layout = QtWidgets.QFormLayout(train_group)

        # Device
        self.yolo_device_combo = QtWidgets.QComboBox()
        self._populate_device_combo(self.yolo_device_combo)
        train_layout.addRow("Device:", self.yolo_device_combo)

        # Epochs
        self.yolo_epochs_spin = QtWidgets.QSpinBox()
        self.yolo_epochs_spin.setRange(1, 1000)
        self.yolo_epochs_spin.setValue(100)
        train_layout.addRow("Epochs:", self.yolo_epochs_spin)

        # Batch size
        self.yolo_batch_spin = QtWidgets.QSpinBox()
        self.yolo_batch_spin.setRange(1, 128)
        self.yolo_batch_spin.setValue(8)
        train_layout.addRow("Batch size:", self.yolo_batch_spin)

        # Image size
        self.yolo_imgsz_spin = QtWidgets.QSpinBox()
        self.yolo_imgsz_spin.setRange(320, 1280)
        self.yolo_imgsz_spin.setSingleStep(32)
        self.yolo_imgsz_spin.setValue(640)
        train_layout.addRow("Image size:", self.yolo_imgsz_spin)

        layout.addWidget(train_group)

        # Advanced (collapsible)
        advanced_group = QtWidgets.QGroupBox("Advanced Options")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        advanced_layout = QtWidgets.QFormLayout(advanced_group)

        self.yolo_lr_spin = QtWidgets.QDoubleSpinBox()
        self.yolo_lr_spin.setRange(0.0001, 0.1)
        self.yolo_lr_spin.setDecimals(4)
        self.yolo_lr_spin.setValue(0.01)
        advanced_layout.addRow("Learning rate:", self.yolo_lr_spin)

        self.yolo_patience_spin = QtWidgets.QSpinBox()
        self.yolo_patience_spin.setRange(1, 500)
        self.yolo_patience_spin.setValue(100)
        advanced_layout.addRow("Early stop patience:", self.yolo_patience_spin)

        self.yolo_cache_check = QtWidgets.QCheckBox("Cache images in RAM")
        advanced_layout.addRow("", self.yolo_cache_check)

        layout.addWidget(advanced_group)
        layout.addStretch()

        return widget

    def _create_dino_options(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Model selection
        model_group = QtWidgets.QGroupBox("DINO Backbone")
        model_layout = QtWidgets.QFormLayout(model_group)

        self.dino_model_combo = QtWidgets.QComboBox()
        for cfg in PATCH_SIMILARITY_MODELS:
            self.dino_model_combo.addItem(cfg.display_name, cfg.identifier)
        model_layout.addRow("Model:", self.dino_model_combo)

        self.dino_short_side_spin = QtWidgets.QSpinBox()
        self.dino_short_side_spin.setRange(224, 2048)
        self.dino_short_side_spin.setSingleStep(32)
        self.dino_short_side_spin.setValue(768)
        model_layout.addRow("Short side:", self.dino_short_side_spin)

        layout.addWidget(model_group)

        # Training params
        train_group = QtWidgets.QGroupBox("Training Parameters")
        train_layout = QtWidgets.QFormLayout(train_group)

        self.dino_epochs_spin = QtWidgets.QSpinBox()
        self.dino_epochs_spin.setRange(1, 500)
        self.dino_epochs_spin.setValue(100)
        train_layout.addRow("Epochs:", self.dino_epochs_spin)

        self.dino_batch_spin = QtWidgets.QSpinBox()
        self.dino_batch_spin.setRange(1, 64)
        self.dino_batch_spin.setValue(8)
        train_layout.addRow("Batch size:", self.dino_batch_spin)

        self.dino_lr_spin = QtWidgets.QDoubleSpinBox()
        self.dino_lr_spin.setRange(0.0001, 0.01)
        self.dino_lr_spin.setDecimals(4)
        self.dino_lr_spin.setValue(0.002)
        train_layout.addRow("Learning rate:", self.dino_lr_spin)

        self.dino_radius_spin = QtWidgets.QDoubleSpinBox()
        self.dino_radius_spin.setRange(1.0, 20.0)
        self.dino_radius_spin.setValue(6.0)
        train_layout.addRow("Keypoint radius (px):", self.dino_radius_spin)

        layout.addWidget(train_group)
        layout.addStretch()

        return widget

    def _create_maskrcnn_options(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Training params
        train_group = QtWidgets.QGroupBox("Training Parameters")
        train_layout = QtWidgets.QFormLayout(train_group)

        self.maskrcnn_iter_spin = QtWidgets.QSpinBox()
        self.maskrcnn_iter_spin.setRange(100, 50000)
        self.maskrcnn_iter_spin.setSingleStep(100)
        self.maskrcnn_iter_spin.setValue(2000)
        train_layout.addRow("Max iterations:", self.maskrcnn_iter_spin)

        self.maskrcnn_batch_spin = QtWidgets.QSpinBox()
        self.maskrcnn_batch_spin.setRange(1, 32)
        self.maskrcnn_batch_spin.setValue(2)
        train_layout.addRow("Batch size:", self.maskrcnn_batch_spin)

        layout.addWidget(train_group)
        layout.addStretch()

        return widget

    def _populate_device_combo(self, combo: QtWidgets.QComboBox) -> None:
        options = [("Auto (recommended)", "")]
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                options.append(("Apple MPS", "mps"))
            if torch.cuda.is_available():
                options.insert(1, ("CUDA (GPU 0)", "0"))
        except ImportError:
            pass
        options.append(("CPU", "cpu"))

        for label, value in options:
            combo.addItem(label, userData=value)

    def _update_yolo_model(self) -> None:
        task = self.yolo_task_combo.currentText()
        size = self.yolo_size_combo.currentText().lower()
        suffix = YOLO_TASKS.get(task, "-seg")
        model_name = f"yolo11{size}{suffix}.pt"
        self.yolo_model_label.setText(model_name)

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if folder:
            self.output_dir_edit.setText(folder)

    def get_config(self) -> Dict[str, Any]:
        wizard = self.wizard()
        backend = wizard.select_backend_page.get_backend(
        ) if isinstance(wizard, TrainingWizard) else "yolo"

        config = {
            "backend": backend,
            "output_dir": self.output_dir_edit.text().strip() or None,
        }

        if backend == "yolo":
            task = self.yolo_task_combo.currentText()
            size = self.yolo_size_combo.currentText().lower()
            suffix = YOLO_TASKS.get(task, "-seg")

            config.update({
                "model": f"yolo11{size}{suffix}.pt",
                "epochs": self.yolo_epochs_spin.value(),
                "batch": self.yolo_batch_spin.value(),
                "imgsz": self.yolo_imgsz_spin.value(),
                "device": self.yolo_device_combo.currentData() or "",
                "lr0": self.yolo_lr_spin.value(),
                "patience": self.yolo_patience_spin.value(),
                "cache": self.yolo_cache_check.isChecked(),
            })

        elif backend == "dino_kpseg":
            config.update({
                "model": self.dino_model_combo.currentData(),
                "epochs": self.dino_epochs_spin.value(),
                "batch": self.dino_batch_spin.value(),
                "short_side": self.dino_short_side_spin.value(),
                "lr": self.dino_lr_spin.value(),
                "radius_px": self.dino_radius_spin.value(),
            })

        elif backend == "maskrcnn":
            config.update({
                "max_iterations": self.maskrcnn_iter_spin.value(),
                "batch": self.maskrcnn_batch_spin.value(),
            })

        return config


class TrainingSummaryPage(QtWidgets.QWizardPage):
    """Page 4: Review and launch training."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Review & Launch")
        self.setSubTitle(
            "Review your training configuration and click Start to begin training."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Summary
        self.summary_text = QtWidgets.QTextBrowser()
        self.summary_text.setOpenExternalLinks(False)
        self.summary_text.setMaximumHeight(300)
        layout.addWidget(self.summary_text)

        # Options
        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)

        self.open_dashboard_check = QtWidgets.QCheckBox(
            "Open training dashboard to monitor progress"
        )
        self.open_dashboard_check.setChecked(True)
        options_layout.addWidget(self.open_dashboard_check)

        self.open_tensorboard_check = QtWidgets.QCheckBox(
            "Launch TensorBoard for visualization"
        )
        options_layout.addWidget(self.open_tensorboard_check)

        layout.addWidget(options_group)

        # Estimated time
        self.time_estimate = QtWidgets.QLabel("")
        self.time_estimate.setStyleSheet("color: gray;")
        layout.addWidget(self.time_estimate)

        layout.addStretch()

    def initializePage(self) -> None:
        wizard = self.wizard()
        if not isinstance(wizard, TrainingWizard):
            return

        dataset_path = wizard.select_dataset_page.get_dataset_path()
        backend = wizard.select_backend_page.get_backend()
        config = wizard.configure_params_page.get_config()

        # Build summary HTML
        html = []
        html.append("<h3>📊 Dataset</h3>")
        html.append(f"<p>{Path(dataset_path).name}</p>")

        html.append("<h3>🧠 Training Backend</h3>")
        backend_names = {
            "yolo": "YOLO (Ultralytics)",
            "dino_kpseg": "DINO KPSEG",
            "maskrcnn": "Mask R-CNN (Detectron2)",
        }
        html.append(f"<p><b>{backend_names.get(backend, backend)}</b></p>")

        html.append("<h3>⚙️ Parameters</h3>")
        html.append("<ul>")
        if backend == "yolo":
            html.append(
                f"<li>Model: {config.get('model', 'yolo11n-seg.pt')}</li>")
            html.append(f"<li>Epochs: {config.get('epochs', 100)}</li>")
            html.append(f"<li>Batch size: {config.get('batch', 8)}</li>")
            html.append(f"<li>Image size: {config.get('imgsz', 640)}</li>")
        elif backend == "dino_kpseg":
            html.append(f"<li>Model: {config.get('model', 'DINOv2')}</li>")
            html.append(f"<li>Epochs: {config.get('epochs', 100)}</li>")
            html.append(f"<li>Learning rate: {config.get('lr', 0.002)}</li>")
        else:
            html.append(
                f"<li>Max iterations: {config.get('max_iterations', 2000)}</li>")
        html.append("</ul>")

        if config.get('output_dir'):
            html.append("<h3>📁 Output</h3>")
            html.append(f"<p>{config['output_dir']}</p>")

        self.summary_text.setHtml("".join(html))

        # Estimate time
        epochs = config.get('epochs', config.get('max_iterations', 100))
        self.time_estimate.setText(
            f"Estimated training time: varies based on dataset size and hardware"
        )

    def should_open_dashboard(self) -> bool:
        return self.open_dashboard_check.isChecked()

    def should_open_tensorboard(self) -> bool:
        return self.open_tensorboard_check.isChecked()


class TrainingWizard(QtWidgets.QWizard):
    """Main training wizard combining all pages."""

    # Signal emitted when training should start
    training_requested = Signal(dict)  # config dict

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Train Model")
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)
        self.setMinimumSize(700, 550)

        # Set up pages
        self.select_dataset_page = SelectDatasetPage()
        self.select_backend_page = SelectBackendPage()
        self.configure_params_page = ConfigureParametersPage()
        self.summary_page = TrainingSummaryPage()

        self.addPage(self.select_dataset_page)
        self.addPage(self.select_backend_page)
        self.addPage(self.configure_params_page)
        self.addPage(self.summary_page)

        # Pre-fill dataset if provided
        if dataset_path:
            self.select_dataset_page.yolo_data_edit.setText(dataset_path)

        # Customize buttons
        self.setButtonText(QtWidgets.QWizard.FinishButton, "🚀 Start Training")
        self.setButtonText(QtWidgets.QWizard.NextButton, "Next →")
        self.setButtonText(QtWidgets.QWizard.BackButton, "← Back")

    def accept(self) -> None:
        # Build complete config and emit signal
        config = self.configure_params_page.get_config()
        config["dataset_path"] = self.select_dataset_page.get_dataset_path()
        config["dataset_type"] = self.select_dataset_page.get_dataset_type()
        config["open_dashboard"] = self.summary_page.should_open_dashboard()
        config["open_tensorboard"] = self.summary_page.should_open_tensorboard()

        self.training_requested.emit(config)
        super().accept()

    def get_training_config(self) -> Dict[str, Any]:
        """Get the complete training configuration."""
        config = self.configure_params_page.get_config()
        config["dataset_path"] = self.select_dataset_page.get_dataset_path()
        config["dataset_type"] = self.select_dataset_page.get_dataset_type()
        return config
