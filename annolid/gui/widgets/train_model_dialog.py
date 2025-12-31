from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from qtpy import QtCore, QtWidgets

from annolid.utils.devices import get_device


YOLO11_TASK_SUFFIXES = {
    "Detection": "",
    "Instance Segmentation": "-seg",
    "Pose Estimation": "-pose",
    "Oriented Detection": "-obb",
    "Classification": "-cls",
}
YOLO11_MODEL_SIZES = ("n", "s", "m", "l", "x")


class TrainModelDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Train Models")

        # -------------------- Defaults (consumed by annolid/gui/app.py)
        self.batch_size = 8
        self.algo = "MaskRCNN"
        self.config_file = None
        self.out_dir = None
        self.max_iterations = 2000
        self.trained_model = None

        # YOLO-only defaults
        self.image_size = 640
        self.epochs = 100
        self.yolo_model_file = "yolo11n-seg.pt"
        self.yolo_device = self._default_yolo_device()
        self.yolo_plots = True

        # YOLO hyperparams (Advanced tab)
        self.yolo_lr0 = 0.01
        self.yolo_lrf = 0.01
        self.yolo_weight_decay = 0.0005
        self.yolo_patience = 100
        self.yolo_close_mosaic = 10
        self.yolo_optimizer = "auto"
        self.yolo_cos_lr = False
        self.yolo_cache = False  # False | True | "disk"

        self._build_ui()
        self.update_ui_for_algorithm()

        self.resize(760, 520)
        self.setSizeGripEnabled(True)

    # -------------------- Public API used by app.py
    def get_yolo_train_overrides(self) -> Dict[str, Any]:
        return {
            "lr0": float(self.yolo_lr0),
            "lrf": float(self.yolo_lrf),
            "weight_decay": float(self.yolo_weight_decay),
            "patience": int(self.yolo_patience),
            "close_mosaic": int(self.yolo_close_mosaic),
            "optimizer": str(self.yolo_optimizer),
            "cos_lr": bool(self.yolo_cos_lr),
            "cache": self.yolo_cache,
        }

    # -------------------- UI construction
    def _build_ui(self) -> None:
        self._build_algo_selector()
        self._build_shared_inputs()
        self._build_yolo_model_selector()
        self._build_device_selector()
        self._build_basic_tab()
        self._build_advanced_tab()
        self._build_buttons()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.groupBoxAlgo)
        layout.addWidget(self.tabs)
        layout.addWidget(self.buttonbox)

    def _build_algo_selector(self) -> None:
        self.groupBoxAlgo = QtWidgets.QGroupBox("Training Backend", self)
        row = QtWidgets.QHBoxLayout(self.groupBoxAlgo)

        self.radio_btn_maskrcnn = QtWidgets.QRadioButton(
            "MaskRCNN", self.groupBoxAlgo)
        self.radio_btn_maskrcnn.setChecked(True)
        self.radio_btn_maskrcnn.toggled.connect(self.on_radio_button_checked)
        row.addWidget(self.radio_btn_maskrcnn)

        self.radio_btn_yolo = QtWidgets.QRadioButton("YOLO", self.groupBoxAlgo)
        self.radio_btn_yolo.toggled.connect(self.on_radio_button_checked)
        row.addWidget(self.radio_btn_yolo)

        row.addStretch(1)

    def _build_shared_inputs(self) -> None:
        # Training inputs
        self.batch_spin = QtWidgets.QSpinBox(self)
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(int(self.batch_size))
        self.batch_spin.valueChanged.connect(
            lambda v: setattr(self, "batch_size", int(v)))

        self.imgsz_spin = QtWidgets.QSpinBox(self)
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(int(self.image_size))
        self.imgsz_spin.valueChanged.connect(
            lambda v: setattr(self, "image_size", int(v)))

        self.epochs_spin = QtWidgets.QSpinBox(self)
        self.epochs_spin.setRange(1, 300)
        self.epochs_spin.setValue(int(self.epochs))
        self.epochs_spin.valueChanged.connect(
            lambda v: setattr(self, "epochs", int(v)))

        self.max_iter_spin = QtWidgets.QSpinBox(self)
        self.max_iter_spin.setRange(100, 20000)
        self.max_iter_spin.setSingleStep(100)
        self.max_iter_spin.setValue(int(self.max_iterations))
        self.max_iter_spin.valueChanged.connect(
            lambda v: setattr(self, "max_iterations", int(v)))

        self.yolo_plots_checkbox = QtWidgets.QCheckBox(
            "Save training plots (curves PNG, confusion matrix, etc.)", self)
        self.yolo_plots_checkbox.setChecked(bool(self.yolo_plots))
        self.yolo_plots_checkbox.stateChanged.connect(
            lambda _=None: setattr(self, "yolo_plots", bool(
                self.yolo_plots_checkbox.isChecked()))
        )

        # File inputs (shared)
        self.configFileLineEdit = QtWidgets.QLineEdit(self)
        self.configFileButton = QtWidgets.QPushButton("Browse…", self)
        self.configFileButton.clicked.connect(
            self.on_config_file_button_clicked)

        self.trainedModelLineEdit = QtWidgets.QLineEdit(self)
        self.trainedModelButton = QtWidgets.QPushButton("Browse…", self)
        self.trainedModelButton.clicked.connect(
            self.on_trained_model_button_clicked)

        self.outDirLineEdit = QtWidgets.QLineEdit(self)
        self.outDirButton = QtWidgets.QPushButton("Select…", self)
        self.outDirButton.clicked.connect(self.on_out_dir_button_clicked)

    def _build_yolo_model_selector(self) -> None:
        self.yolo_model_groupbox = QtWidgets.QGroupBox("YOLO Weights", self)
        grid = QtWidgets.QGridLayout(self.yolo_model_groupbox)

        self.yolo_task_combo = QtWidgets.QComboBox(self.yolo_model_groupbox)
        self.yolo_task_combo.addItems(list(YOLO11_TASK_SUFFIXES.keys()))
        self.yolo_task_combo.currentTextChanged.connect(
            self.update_yolo_model_filename)

        self.yolo_size_combo = QtWidgets.QComboBox(self.yolo_model_groupbox)
        self.yolo_size_combo.addItems([s.upper() for s in YOLO11_MODEL_SIZES])
        self.yolo_size_combo.currentTextChanged.connect(
            self.update_yolo_model_filename)

        self.yolo_model_label = QtWidgets.QLabel(self.yolo_model_groupbox)
        self.yolo_model_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)

        grid.addWidget(QtWidgets.QLabel("Task"), 0, 0)
        grid.addWidget(self.yolo_task_combo, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Model Size"), 0, 2)
        grid.addWidget(self.yolo_size_combo, 0, 3)
        grid.addWidget(self.yolo_model_label, 1, 0, 1, 4)

        self.yolo_task_combo.setCurrentText("Instance Segmentation")
        self.yolo_size_combo.setCurrentText("N")
        self.update_yolo_model_filename()

    def _default_yolo_device(self) -> str:
        device = str(get_device() or "cpu").strip().lower()
        if device == "cuda":
            return "0"
        if device in {"cpu", "mps"}:
            return device
        return "cpu"

    def _build_device_selector(self) -> None:
        self.yolo_device_combo = QtWidgets.QComboBox(self)
        options = [
            ("Auto (recommended)", ""),
            ("CPU", "cpu"),
        ]
        try:
            import torch  # type: ignore

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                options.insert(1, ("Apple MPS", "mps"))
            if torch.cuda.is_available():
                options.insert(1, ("CUDA (GPU 0)", "0"))
        except Exception:
            pass

        default_index = 0
        for idx, (label, value) in enumerate(options):
            self.yolo_device_combo.addItem(label, userData=value)
            if str(value) == str(self.yolo_device):
                default_index = idx
        self.yolo_device_combo.setCurrentIndex(default_index)
        self.yolo_device_combo.currentIndexChanged.connect(
            self._on_device_changed)

    def _build_basic_tab(self) -> None:
        self.tabs = QtWidgets.QTabWidget(self)
        self.basic_tab = QtWidgets.QWidget(self.tabs)
        self.tabs.addTab(self.basic_tab, "Basic")

        grid = QtWidgets.QGridLayout(self.basic_tab)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        self.training_groupbox = QtWidgets.QGroupBox(
            "Training", self.basic_tab)
        self._training_form = QtWidgets.QFormLayout(self.training_groupbox)
        self._training_form.setLabelAlignment(QtCore.Qt.AlignRight)
        self._training_form.setFormAlignment(QtCore.Qt.AlignTop)

        self._training_form.addRow("Device", self.yolo_device_combo)
        self._training_form.addRow("Batch size", self.batch_spin)
        self._training_form.addRow("Image size", self.imgsz_spin)
        self._training_form.addRow("Epochs", self.epochs_spin)
        self._training_form.addRow("Max iterations", self.max_iter_spin)
        self._training_form.addRow("", self.yolo_plots_checkbox)

        self.io_groupbox = QtWidgets.QGroupBox("Data & Output", self.basic_tab)
        self._io_form = QtWidgets.QFormLayout(self.io_groupbox)
        self._io_form.setLabelAlignment(QtCore.Qt.AlignRight)
        self._io_form.setFormAlignment(QtCore.Qt.AlignTop)

        self._io_config_row = self._make_row(
            self.configFileLineEdit, self.configFileButton)
        self._io_model_row = self._make_row(
            self.trainedModelLineEdit, self.trainedModelButton)
        self._io_out_row = self._make_row(
            self.outDirLineEdit, self.outDirButton)

        self._io_config_label = QtWidgets.QLabel(
            "Dataset YAML", self.io_groupbox)
        self._io_model_label = QtWidgets.QLabel(
            "Resume from (.pt)", self.io_groupbox)
        self._io_out_label = QtWidgets.QLabel(
            "Output directory", self.io_groupbox)

        self._io_form.addRow(self._io_config_label, self._io_config_row)
        self._io_form.addRow(self._io_model_label, self._io_model_row)
        self._io_form.addRow(self._io_out_label, self._io_out_row)

        grid.addWidget(self.yolo_model_groupbox, 0, 0, 1, 2)
        grid.addWidget(self.training_groupbox, 1, 0, 1, 1)
        grid.addWidget(self.io_groupbox, 1, 1, 1, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    def _build_advanced_tab(self) -> None:
        self.advanced_tab = QtWidgets.QWidget(self.tabs)
        self.tabs.addTab(self.advanced_tab, "Advanced")

        layout = QtWidgets.QVBoxLayout(self.advanced_tab)
        layout.setContentsMargins(10, 10, 10, 10)

        self.yolo_hyperparams_groupbox = self._build_yolo_hyperparams_groupbox()
        scroll = QtWidgets.QScrollArea(self.advanced_tab)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self.yolo_hyperparams_groupbox)
        layout.addWidget(scroll)

    def _build_buttons(self) -> None:
        qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonbox = QtWidgets.QDialogButtonBox(qbtn, parent=self)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

    # -------------------- Helpers
    def _make_row(self, line_edit: QtWidgets.QLineEdit, button: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        layout.addWidget(button, 0)
        return row

    def _set_form_row_visible(self, form: QtWidgets.QFormLayout, field: QtWidgets.QWidget, visible: bool) -> None:
        label = form.labelForField(field)
        if label is not None:
            label.setVisible(visible)
        field.setVisible(visible)

    # -------------------- Callbacks
    def on_radio_button_checked(self):
        radio_btn = self.sender()
        if radio_btn.isChecked():
            self.algo = radio_btn.text()
            self.update_ui_for_algorithm()

    def update_ui_for_algorithm(self):
        yolo_selected = self.algo == "YOLO"

        self.yolo_model_groupbox.setVisible(yolo_selected)
        self.tabs.setTabEnabled(1, bool(yolo_selected))

        # Training form rows
        self._set_form_row_visible(
            self._training_form, self.yolo_device_combo, yolo_selected)
        self._set_form_row_visible(
            self._training_form, self.imgsz_spin, yolo_selected)
        self._set_form_row_visible(
            self._training_form, self.epochs_spin, yolo_selected)
        self._set_form_row_visible(
            self._training_form, self.max_iter_spin, not yolo_selected)
        self._set_form_row_visible(
            self._training_form, self.yolo_plots_checkbox, yolo_selected)

        # IO labels
        self._io_config_label.setText(
            "Dataset YAML" if yolo_selected else "Config file")
        self._io_model_label.setText(
            "Resume from (.pt)" if yolo_selected else "Model weights (.pt)")

        self.configFileButton.setText(
            "Browse YAML…" if yolo_selected else "Browse…")

    def update_yolo_model_filename(self):
        task_text = self.yolo_task_combo.currentText()
        size_text = self.yolo_size_combo.currentText().lower()
        suffix = YOLO11_TASK_SUFFIXES.get(task_text, "")
        self.yolo_model_file = f"yolo11{size_text}{suffix}.pt"
        self.yolo_model_label.setText(
            f"Selected weights: {self.yolo_model_file}")

    def _on_device_changed(self) -> None:
        value = self.yolo_device_combo.currentData()
        self.yolo_device = str(value or "").strip().lower()

    def on_config_file_button_clicked(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if self.algo == "YOLO":
            file_dialog.setNameFilter("YAML files (*.yaml *.yml)")
        else:
            file_dialog.setNameFilter("All files (*)")

        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.config_file = selected_files[0]
                self.configFileLineEdit.setText(self.config_file)

    def on_trained_model_button_clicked(self):
        self.trained_model, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Select Model Weights",
            directory=str(Path()),
            filter="YOLO/Model Weights (*.pt *.pth);;All Files (*.*)",
        )
        if self.trained_model:
            self.trainedModelLineEdit.setText(self.trained_model)

    def on_out_dir_button_clicked(self):
        self.out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if self.out_dir:
            self.outDirLineEdit.setText(self.out_dir)

    # -------------------- Hyperparams (Advanced tab)
    def _build_yolo_hyperparams_groupbox(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Hyperparameters", self)
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        lr0 = QtWidgets.QDoubleSpinBox(box)
        lr0.setDecimals(6)
        lr0.setRange(0.000001, 1.0)
        lr0.setSingleStep(0.001)
        lr0.setValue(float(self.yolo_lr0))
        lr0.valueChanged.connect(lambda v: setattr(self, "yolo_lr0", float(v)))

        lrf = QtWidgets.QDoubleSpinBox(box)
        lrf.setDecimals(6)
        lrf.setRange(0.000001, 1.0)
        lrf.setSingleStep(0.001)
        lrf.setValue(float(self.yolo_lrf))
        lrf.valueChanged.connect(lambda v: setattr(self, "yolo_lrf", float(v)))

        wd = QtWidgets.QDoubleSpinBox(box)
        wd.setDecimals(6)
        wd.setRange(0.0, 0.1)
        wd.setSingleStep(0.0001)
        wd.setValue(float(self.yolo_weight_decay))
        wd.valueChanged.connect(lambda v: setattr(
            self, "yolo_weight_decay", float(v)))

        patience = QtWidgets.QSpinBox(box)
        patience.setRange(0, 1000)
        patience.setValue(int(self.yolo_patience))
        patience.valueChanged.connect(
            lambda v: setattr(self, "yolo_patience", int(v)))

        close_mosaic = QtWidgets.QSpinBox(box)
        close_mosaic.setRange(0, 100)
        close_mosaic.setValue(int(self.yolo_close_mosaic))
        close_mosaic.valueChanged.connect(
            lambda v: setattr(self, "yolo_close_mosaic", int(v)))

        optimizer = QtWidgets.QComboBox(box)
        optimizer_options = [
            ("Auto", "auto"),
            ("SGD", "SGD"),
            ("Adam", "Adam"),
            ("AdamW", "AdamW"),
        ]
        default_idx = 0
        for idx, (label, value) in enumerate(optimizer_options):
            optimizer.addItem(label, userData=value)
            if str(value).lower() == str(self.yolo_optimizer).lower():
                default_idx = idx
        optimizer.setCurrentIndex(default_idx)
        optimizer.currentIndexChanged.connect(lambda _=None: setattr(
            self, "yolo_optimizer", str(optimizer.currentData())))

        cos_lr = QtWidgets.QCheckBox("Cosine LR schedule", box)
        cos_lr.setChecked(bool(self.yolo_cos_lr))
        cos_lr.stateChanged.connect(lambda _=None: setattr(
            self, "yolo_cos_lr", bool(cos_lr.isChecked())))

        cache = QtWidgets.QComboBox(box)
        cache_options = [
            ("Off", False),
            ("RAM", True),
            ("Disk", "disk"),
        ]
        cache_default_idx = 0
        for idx, (label, value) in enumerate(cache_options):
            cache.addItem(label, userData=value)
            if value == self.yolo_cache:
                cache_default_idx = idx
        cache.setCurrentIndex(cache_default_idx)
        cache.currentIndexChanged.connect(lambda _=None: setattr(
            self, "yolo_cache", cache.currentData()))

        form.addRow("Learning rate (lr0)", lr0)
        form.addRow("Final LR fraction (lrf)", lrf)
        form.addRow("Weight decay", wd)
        form.addRow("Early stop patience", patience)
        form.addRow("Close mosaic (epoch)", close_mosaic)
        form.addRow("Optimizer", optimizer)
        form.addRow("Dataset cache", cache)
        form.addRow("", cos_lr)
        return box

    # -------------------- Accept
    def accept(self) -> None:
        if self.algo == "YOLO" and not self.config_file:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Please select a dataset YAML file.")
            return
        super().accept()
