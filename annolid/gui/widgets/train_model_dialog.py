from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from qtpy import QtCore, QtWidgets

from annolid.utils.devices import get_device
from annolid.gui.models_registry import PATCH_SIMILARITY_MODELS, PATCH_SIMILARITY_DEFAULT_MODEL


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

        # DinoKPSEG defaults
        self.dino_model_name = PATCH_SIMILARITY_DEFAULT_MODEL
        self.dino_short_side = 768
        self.dino_layers = "-1"
        self.dino_radius_px = 6.0
        self.dino_hidden_dim = 128
        self.dino_head_type = "conv"
        self.dino_attn_heads = 4
        self.dino_attn_layers = 1
        self.dino_lr_pair_loss_weight = 0.0
        self.dino_lr_pair_margin_px = 0.0
        self.dino_lr_side_loss_weight = 0.0
        self.dino_lr_side_loss_margin = 0.0
        self.dino_lr = 0.002
        self.dino_threshold = 0.4
        self.dino_bce_type = "bce"
        self.dino_focal_alpha = 0.25
        self.dino_focal_gamma = 2.0
        self.dino_coord_warmup_epochs = 0
        self.dino_radius_schedule = "none"
        self.dino_radius_start_px = float(self.dino_radius_px)
        self.dino_radius_end_px = float(self.dino_radius_px)
        self.dino_overfit_n = 0
        self.dino_cache_features = True
        self.dino_patience = 10
        self.dino_min_delta = 0.0
        self.dino_min_epochs = 10
        self.dino_best_metric = "pck@8px"
        self.dino_early_stop_metric = "auto"
        self.dino_pck_weighted_weights = "1,1,1,1"
        self.dino_augment_enabled = False
        self.dino_hflip_prob = 0.5
        self.dino_degrees = 0.0
        self.dino_translate = 0.0
        self.dino_scale = 0.0
        self.dino_brightness = 0.0
        self.dino_contrast = 0.0
        self.dino_saturation = 0.0
        self.dino_seed = -1
        self.dino_tb_add_graph = False
        self.dino_tb_projector = True
        self.dino_tb_projector_split = "val"
        self.dino_tb_projector_max_images = 64
        self.dino_tb_projector_max_patches = 4000
        self.dino_tb_projector_per_image_per_keypoint = 3
        self.dino_tb_projector_pos_threshold = 0.35
        self.dino_tb_projector_crop_px = 96
        self.dino_tb_projector_sprite_border_px = 3
        self.dino_tb_projector_add_negatives = False
        self.dino_tb_projector_neg_threshold = 0.02
        self.dino_tb_projector_negatives_per_image = 6

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

        self.radio_btn_dino_kpseg = QtWidgets.QRadioButton(
            "DINO KPSEG", self.groupBoxAlgo)
        self.radio_btn_dino_kpseg.toggled.connect(self.on_radio_button_checked)
        row.addWidget(self.radio_btn_dino_kpseg)

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

    def _build_dino_model_selector(self) -> None:
        self.dino_model_groupbox = QtWidgets.QGroupBox("DINO Backbone", self)
        form = QtWidgets.QFormLayout(self.dino_model_groupbox)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self.dino_model_combo = QtWidgets.QComboBox(self.dino_model_groupbox)
        for cfg in PATCH_SIMILARITY_MODELS:
            self.dino_model_combo.addItem(cfg.display_name, cfg.identifier)
        idx = self.dino_model_combo.findData(self.dino_model_name)
        if idx >= 0:
            self.dino_model_combo.setCurrentIndex(idx)
        self.dino_model_combo.currentIndexChanged.connect(
            lambda _=None: setattr(self, "dino_model_name", str(
                self.dino_model_combo.currentData() or "").strip())
        )

        short_side = QtWidgets.QSpinBox(self.dino_model_groupbox)
        short_side.setRange(224, 2048)
        short_side.setSingleStep(32)
        short_side.setValue(int(self.dino_short_side))
        short_side.valueChanged.connect(
            lambda v: setattr(self, "dino_short_side", int(v)))

        layers = QtWidgets.QLineEdit(self.dino_model_groupbox)
        layers.setPlaceholderText("-1 or -2,-1")
        layers.setText(str(self.dino_layers))
        layers.textChanged.connect(
            lambda text: setattr(self, "dino_layers", str(text).strip())
        )

        form.addRow("Model", self.dino_model_combo)
        form.addRow("Short side", short_side)
        form.addRow("Layers", layers)

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
        self._build_dino_model_selector()
        grid.addWidget(self.dino_model_groupbox, 0, 0, 1, 2)
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

        self.dino_hyperparams_groupbox = self._build_dino_hyperparams_groupbox()
        self.dino_augment_groupbox = self._build_dino_augment_groupbox()
        scroll = QtWidgets.QScrollArea(self.advanced_tab)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        container = QtWidgets.QWidget(self.advanced_tab)
        stack = QtWidgets.QVBoxLayout(container)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.addWidget(self.yolo_hyperparams_groupbox)
        stack.addWidget(self.dino_hyperparams_groupbox)
        stack.addWidget(self.dino_augment_groupbox)
        stack.addStretch(1)
        scroll.setWidget(container)
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
        dino_selected = self.algo == "DINO KPSEG"
        classic_selected = not (yolo_selected or dino_selected)

        self.yolo_model_groupbox.setVisible(yolo_selected)
        self.dino_model_groupbox.setVisible(dino_selected)
        self.tabs.setTabEnabled(1, bool(yolo_selected or dino_selected))
        self.yolo_hyperparams_groupbox.setVisible(yolo_selected)
        self.dino_hyperparams_groupbox.setVisible(dino_selected)
        self.dino_augment_groupbox.setVisible(dino_selected)

        # Training form rows
        self._set_form_row_visible(
            self._training_form, self.yolo_device_combo, yolo_selected or dino_selected)
        self._set_form_row_visible(
            self._training_form, self.imgsz_spin, yolo_selected)
        self._set_form_row_visible(
            self._training_form, self.epochs_spin, yolo_selected or dino_selected)
        self._set_form_row_visible(
            self._training_form, self.max_iter_spin, classic_selected)
        self._set_form_row_visible(
            self._training_form, self.yolo_plots_checkbox, yolo_selected)

        # Batch size: DinoKPSEG supports padded batching (may increase memory usage).
        self.batch_spin.setEnabled(True)
        if dino_selected:
            self.batch_spin.setToolTip(
                "DINO KPSEG supports batch_size > 1 via padded feature grids.\n"
                "Higher batch sizes use more memory; reduce if you hit OOM."
            )
        else:
            self.batch_spin.setToolTip("")

        # IO labels
        self._io_config_label.setText(
            "Dataset YAML" if (yolo_selected or dino_selected) else "Config file")
        self._io_model_label.setText(
            "Resume from (.pt)" if yolo_selected else "Model weights (.pt)")

        self.configFileButton.setText(
            "Browse YAML…" if (yolo_selected or dino_selected) else "Browse…")

        # Resume row only for YOLO / classic trainers.
        self._set_form_row_visible(
            self._io_form, self._io_model_row, bool(yolo_selected))

        self._update_dino_augment_enabled_state()

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
        if self.algo in {"YOLO", "DINO KPSEG"}:
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

    def _build_dino_hyperparams_groupbox(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("DINO KPSEG Hyperparameters", self)
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        head_type = QtWidgets.QComboBox(box)
        head_type.addItem("Conv head (fast)", userData="conv")
        head_type.addItem("Attention head (relational)", userData="attn")
        head_type.addItem("Hybrid head (conv + attn)", userData="hybrid")
        head_default_idx = head_type.findData(
            str(getattr(self, "dino_head_type", "conv")))
        if head_default_idx >= 0:
            head_type.setCurrentIndex(head_default_idx)

        def _on_head_changed(_=None) -> None:
            setattr(self, "dino_head_type", str(
                head_type.currentData() or "conv"))
            self._update_dino_head_controls()

        head_type.currentIndexChanged.connect(_on_head_changed)
        self._dino_head_type_combo = head_type

        attn_heads = QtWidgets.QSpinBox(box)
        attn_heads.setRange(1, 32)
        attn_heads.setValue(int(getattr(self, "dino_attn_heads", 4)))
        attn_heads.valueChanged.connect(
            lambda v: setattr(self, "dino_attn_heads", int(v)))
        self._dino_attn_heads_spin = attn_heads

        attn_layers = QtWidgets.QSpinBox(box)
        attn_layers.setRange(1, 8)
        attn_layers.setValue(int(getattr(self, "dino_attn_layers", 1)))
        attn_layers.valueChanged.connect(
            lambda v: setattr(self, "dino_attn_layers", int(v)))
        self._dino_attn_layers_spin = attn_layers

        pair_w = QtWidgets.QDoubleSpinBox(box)
        pair_w.setDecimals(6)
        pair_w.setRange(0.0, 10.0)
        pair_w.setSingleStep(0.01)
        pair_w.setValue(float(getattr(self, "dino_lr_pair_loss_weight", 0.0)))
        pair_w.valueChanged.connect(lambda v: setattr(
            self, "dino_lr_pair_loss_weight", float(v)))
        self._dino_pair_loss_weight = pair_w

        pair_margin = QtWidgets.QDoubleSpinBox(box)
        pair_margin.setDecimals(2)
        pair_margin.setRange(0.0, 256.0)
        pair_margin.setSingleStep(1.0)
        pair_margin.setValue(
            float(getattr(self, "dino_lr_pair_margin_px", 0.0)))
        pair_margin.valueChanged.connect(lambda v: setattr(
            self, "dino_lr_pair_margin_px", float(v)))
        self._dino_pair_margin_px = pair_margin

        side_w = QtWidgets.QDoubleSpinBox(box)
        side_w.setDecimals(6)
        side_w.setRange(0.0, 10.0)
        side_w.setSingleStep(0.01)
        side_w.setValue(float(getattr(self, "dino_lr_side_loss_weight", 0.0)))
        side_w.valueChanged.connect(lambda v: setattr(
            self, "dino_lr_side_loss_weight", float(v)))
        self._dino_side_loss_weight = side_w

        side_margin = QtWidgets.QDoubleSpinBox(box)
        side_margin.setDecimals(3)
        side_margin.setRange(0.0, 1.0)
        side_margin.setSingleStep(0.05)
        side_margin.setValue(
            float(getattr(self, "dino_lr_side_loss_margin", 0.0)))
        side_margin.valueChanged.connect(lambda v: setattr(
            self, "dino_lr_side_loss_margin", float(v)))
        self._dino_side_loss_margin = side_margin

        lr = QtWidgets.QDoubleSpinBox(box)
        lr.setDecimals(6)
        lr.setRange(1e-6, 1.0)
        lr.setSingleStep(0.0005)
        lr.setValue(float(self.dino_lr))
        lr.valueChanged.connect(lambda v: setattr(self, "dino_lr", float(v)))

        radius = QtWidgets.QDoubleSpinBox(box)
        radius.setDecimals(2)
        radius.setRange(1.0, 64.0)
        radius.setSingleStep(0.5)
        radius.setValue(float(self.dino_radius_px))
        radius.valueChanged.connect(
            lambda v: setattr(self, "dino_radius_px", float(v)))

        hidden = QtWidgets.QSpinBox(box)
        hidden.setRange(16, 2048)
        hidden.setSingleStep(16)
        hidden.setValue(int(self.dino_hidden_dim))
        hidden.valueChanged.connect(
            lambda v: setattr(self, "dino_hidden_dim", int(v)))

        thr = QtWidgets.QDoubleSpinBox(box)
        thr.setDecimals(3)
        thr.setRange(0.01, 0.99)
        thr.setSingleStep(0.01)
        thr.setValue(float(self.dino_threshold))
        thr.valueChanged.connect(
            lambda v: setattr(self, "dino_threshold", float(v)))

        bce_type = QtWidgets.QComboBox(box)
        bce_type.addItem("Standard BCE", userData="bce")
        bce_type.addItem("Focal BCE", userData="focal")
        bce_default = str(getattr(self, "dino_bce_type", "bce") or "bce")
        bce_idx = bce_type.findData(bce_default)
        if bce_idx >= 0:
            bce_type.setCurrentIndex(bce_idx)

        def _on_bce_changed(_=None) -> None:
            setattr(self, "dino_bce_type", str(
                bce_type.currentData() or "bce"))
            self._update_dino_focal_controls()

        bce_type.currentIndexChanged.connect(_on_bce_changed)

        focal_alpha = QtWidgets.QDoubleSpinBox(box)
        focal_alpha.setDecimals(3)
        focal_alpha.setRange(0.0, 1.0)
        focal_alpha.setSingleStep(0.05)
        focal_alpha.setValue(float(getattr(self, "dino_focal_alpha", 0.25)))
        focal_alpha.valueChanged.connect(
            lambda v: setattr(self, "dino_focal_alpha", float(v)))

        focal_gamma = QtWidgets.QDoubleSpinBox(box)
        focal_gamma.setDecimals(3)
        focal_gamma.setRange(0.0, 10.0)
        focal_gamma.setSingleStep(0.1)
        focal_gamma.setValue(float(getattr(self, "dino_focal_gamma", 2.0)))
        focal_gamma.valueChanged.connect(
            lambda v: setattr(self, "dino_focal_gamma", float(v)))

        self._dino_focal_controls = [focal_alpha, focal_gamma]

        coord_warmup = QtWidgets.QSpinBox(box)
        coord_warmup.setRange(0, 1000)
        coord_warmup.setValue(
            int(getattr(self, "dino_coord_warmup_epochs", 0)))
        coord_warmup.valueChanged.connect(
            lambda v: setattr(self, "dino_coord_warmup_epochs", int(v)))

        radius_schedule = QtWidgets.QComboBox(box)
        radius_schedule.addItem("None", userData="none")
        radius_schedule.addItem("Linear", userData="linear")
        radius_default = str(
            getattr(self, "dino_radius_schedule", "none") or "none")
        radius_idx = radius_schedule.findData(radius_default)
        if radius_idx >= 0:
            radius_schedule.setCurrentIndex(radius_idx)

        def _on_radius_schedule_changed(_=None) -> None:
            setattr(self, "dino_radius_schedule", str(
                radius_schedule.currentData() or "none"))
            self._update_dino_radius_schedule_controls()

        radius_schedule.currentIndexChanged.connect(
            _on_radius_schedule_changed)

        radius_start = QtWidgets.QDoubleSpinBox(box)
        radius_start.setDecimals(2)
        radius_start.setRange(1.0, 128.0)
        radius_start.setSingleStep(0.5)
        radius_start.setValue(
            float(getattr(self, "dino_radius_start_px", self.dino_radius_px)))
        radius_start.valueChanged.connect(
            lambda v: setattr(self, "dino_radius_start_px", float(v)))

        radius_end = QtWidgets.QDoubleSpinBox(box)
        radius_end.setDecimals(2)
        radius_end.setRange(1.0, 128.0)
        radius_end.setSingleStep(0.5)
        radius_end.setValue(
            float(getattr(self, "dino_radius_end_px", self.dino_radius_px)))
        radius_end.valueChanged.connect(
            lambda v: setattr(self, "dino_radius_end_px", float(v)))

        self._dino_radius_controls = [radius_start, radius_end]

        overfit_n = QtWidgets.QSpinBox(box)
        overfit_n.setRange(0, 100000)
        overfit_n.setSingleStep(10)
        overfit_n.setValue(int(getattr(self, "dino_overfit_n", 0)))
        overfit_n.setToolTip(
            "Debug mode: use N training images and mirror into validation.")
        overfit_n.valueChanged.connect(
            lambda v: setattr(self, "dino_overfit_n", int(v)))

        cache = QtWidgets.QCheckBox("Cache frozen DINO features to disk", box)
        cache.setChecked(bool(self.dino_cache_features))
        cache.stateChanged.connect(lambda _=None: setattr(
            self, "dino_cache_features", bool(cache.isChecked())))

        tb_graph = QtWidgets.QCheckBox(
            "Log model graph to TensorBoard (slow)", box)
        tb_graph.setChecked(bool(self.dino_tb_add_graph))
        tb_graph.stateChanged.connect(lambda _=None: setattr(
            self, "dino_tb_add_graph", bool(tb_graph.isChecked())))

        tb_projector = QtWidgets.QCheckBox(
            "Write TensorBoard Projector embeddings (interactive)", box
        )
        tb_projector.setChecked(bool(getattr(self, "dino_tb_projector", True)))

        def _on_projector_changed(_=None) -> None:
            setattr(self, "dino_tb_projector", bool(tb_projector.isChecked()))
            self._update_dino_projector_enabled_state()

        tb_projector.stateChanged.connect(_on_projector_changed)

        tb_projector_split = QtWidgets.QComboBox(box)
        tb_projector_split.addItem("Validation (recommended)", userData="val")
        tb_projector_split.addItem("Train", userData="train")
        tb_projector_split.addItem("Both", userData="both")
        split_default = str(
            getattr(self, "dino_tb_projector_split", "val") or "val")
        idx = tb_projector_split.findData(split_default)
        if idx >= 0:
            tb_projector_split.setCurrentIndex(idx)
        tb_projector_split.currentIndexChanged.connect(
            lambda _=None: setattr(self, "dino_tb_projector_split", str(
                tb_projector_split.currentData() or "val"))
        )

        tb_max_images = QtWidgets.QSpinBox(box)
        tb_max_images.setRange(1, 10_000)
        tb_max_images.setValue(
            int(getattr(self, "dino_tb_projector_max_images", 64)))
        tb_max_images.valueChanged.connect(
            lambda v: setattr(self, "dino_tb_projector_max_images", int(v))
        )

        tb_max_patches = QtWidgets.QSpinBox(box)
        tb_max_patches.setRange(64, 200_000)
        tb_max_patches.setSingleStep(256)
        tb_max_patches.setValue(
            int(getattr(self, "dino_tb_projector_max_patches", 4000)))
        tb_max_patches.valueChanged.connect(
            lambda v: setattr(self, "dino_tb_projector_max_patches", int(v))
        )

        tb_per_image_per_kpt = QtWidgets.QSpinBox(box)
        tb_per_image_per_kpt.setRange(0, 1000)
        tb_per_image_per_kpt.setValue(
            int(getattr(self, "dino_tb_projector_per_image_per_keypoint", 3)))
        tb_per_image_per_kpt.valueChanged.connect(
            lambda v: setattr(
                self, "dino_tb_projector_per_image_per_keypoint", int(v))
        )

        tb_pos_thr = QtWidgets.QDoubleSpinBox(box)
        tb_pos_thr.setDecimals(3)
        tb_pos_thr.setRange(0.0, 1.0)
        tb_pos_thr.setSingleStep(0.05)
        tb_pos_thr.setValue(
            float(getattr(self, "dino_tb_projector_pos_threshold", 0.35)))
        tb_pos_thr.valueChanged.connect(
            lambda v: setattr(
                self, "dino_tb_projector_pos_threshold", float(v))
        )

        tb_crop = QtWidgets.QSpinBox(box)
        tb_crop.setRange(16, 512)
        tb_crop.setSingleStep(16)
        tb_crop.setValue(int(getattr(self, "dino_tb_projector_crop_px", 96)))
        tb_crop.valueChanged.connect(
            lambda v: setattr(self, "dino_tb_projector_crop_px", int(v))
        )

        tb_border = QtWidgets.QSpinBox(box)
        tb_border.setRange(0, 16)
        tb_border.setValue(
            int(getattr(self, "dino_tb_projector_sprite_border_px", 3)))
        tb_border.valueChanged.connect(
            lambda v: setattr(
                self, "dino_tb_projector_sprite_border_px", int(v))
        )

        tb_add_negs = QtWidgets.QCheckBox(
            "Also sample background patches", box)
        tb_add_negs.setChecked(
            bool(getattr(self, "dino_tb_projector_add_negatives", False)))
        tb_add_negs.stateChanged.connect(
            lambda _=None: setattr(
                self, "dino_tb_projector_add_negatives", bool(tb_add_negs.isChecked()))
        )

        tb_neg_thr = QtWidgets.QDoubleSpinBox(box)
        tb_neg_thr.setDecimals(3)
        tb_neg_thr.setRange(0.0, 1.0)
        tb_neg_thr.setSingleStep(0.01)
        tb_neg_thr.setValue(
            float(getattr(self, "dino_tb_projector_neg_threshold", 0.02)))
        tb_neg_thr.valueChanged.connect(
            lambda v: setattr(
                self, "dino_tb_projector_neg_threshold", float(v))
        )

        tb_negs_per_img = QtWidgets.QSpinBox(box)
        tb_negs_per_img.setRange(0, 10_000)
        tb_negs_per_img.setValue(
            int(getattr(self, "dino_tb_projector_negatives_per_image", 6)))
        tb_negs_per_img.valueChanged.connect(
            lambda v: setattr(
                self, "dino_tb_projector_negatives_per_image", int(v))
        )

        self._dino_projector_controls = [
            tb_projector_split,
            tb_max_images,
            tb_max_patches,
            tb_per_image_per_kpt,
            tb_pos_thr,
            tb_crop,
            tb_border,
            tb_add_negs,
            tb_neg_thr,
            tb_negs_per_img,
        ]

        patience = QtWidgets.QSpinBox(box)
        patience.setRange(0, 1000)
        patience.setValue(int(self.dino_patience))
        patience.valueChanged.connect(
            lambda v: setattr(self, "dino_patience", int(v)))

        min_delta = QtWidgets.QDoubleSpinBox(box)
        min_delta.setDecimals(6)
        min_delta.setRange(0.0, 1.0)
        min_delta.setSingleStep(0.0001)
        min_delta.setValue(float(self.dino_min_delta))
        min_delta.valueChanged.connect(
            lambda v: setattr(self, "dino_min_delta", float(v)))

        min_epochs = QtWidgets.QSpinBox(box)
        min_epochs.setRange(0, 1000)
        min_epochs.setValue(int(self.dino_min_epochs))
        min_epochs.valueChanged.connect(
            lambda v: setattr(self, "dino_min_epochs", int(v)))

        best_metric = QtWidgets.QComboBox(box)
        best_metric.addItem("PCK@8px (pose)", "pck@8px")
        best_metric.addItem("Weighted PCK (2/4/8/16px)", "pck_weighted")
        best_metric.addItem("Val loss", "val_loss")
        best_metric.addItem("Train loss (no val)", "train_loss")
        best_metric.setCurrentIndex(
            max(0, best_metric.findData(
                getattr(self, "dino_best_metric", "pck@8px")))
        )
        best_metric.currentIndexChanged.connect(
            lambda _=None: (
                setattr(
                    self,
                    "dino_best_metric",
                    str(best_metric.currentData() or "pck@8px"),
                ),
                self._update_dino_model_selection_controls(),
            )
        )

        early_stop_metric = QtWidgets.QComboBox(box)
        early_stop_metric.addItem("Same as best (auto)", "auto")
        early_stop_metric.addItem("PCK@8px (pose)", "pck@8px")
        early_stop_metric.addItem("Weighted PCK (2/4/8/16px)", "pck_weighted")
        early_stop_metric.addItem("Val loss", "val_loss")
        early_stop_metric.addItem("Train loss (no val)", "train_loss")
        early_stop_metric.setCurrentIndex(
            max(
                0,
                early_stop_metric.findData(
                    getattr(self, "dino_early_stop_metric", "auto")
                ),
            )
        )
        early_stop_metric.currentIndexChanged.connect(
            lambda _=None: (
                setattr(
                    self,
                    "dino_early_stop_metric",
                    str(early_stop_metric.currentData() or "auto"),
                ),
                self._update_dino_model_selection_controls(),
            )
        )

        pck_weights = QtWidgets.QLineEdit(box)
        pck_weights.setPlaceholderText("1,1,1,1")
        pck_weights.setText(
            str(getattr(self, "dino_pck_weighted_weights", "1,1,1,1")))
        pck_weights.textChanged.connect(
            lambda text: setattr(
                self, "dino_pck_weighted_weights", str(text).strip())
        )

        self._dino_best_metric_combo = best_metric
        self._dino_early_stop_metric_combo = early_stop_metric
        self._dino_pck_weighted_weights_edit = pck_weights

        form.addRow("Head type", head_type)
        form.addRow("Attn heads", attn_heads)
        form.addRow("Attn layers", attn_layers)
        form.addRow("Pair loss weight", pair_w)
        form.addRow("Pair margin (px)", pair_margin)
        form.addRow("LR side loss weight", side_w)
        form.addRow("LR side loss margin", side_margin)
        form.addRow("Learning rate", lr)
        form.addRow("Circle radius (px)", radius)
        form.addRow("Radius schedule", radius_schedule)
        form.addRow("Radius start (px)", radius_start)
        form.addRow("Radius end (px)", radius_end)
        form.addRow("Head hidden dim", hidden)
        form.addRow("Mask threshold", thr)
        form.addRow("Mask BCE type", bce_type)
        form.addRow("Focal alpha", focal_alpha)
        form.addRow("Focal gamma", focal_gamma)
        form.addRow("Coord warmup (epochs)", coord_warmup)
        form.addRow("Overfit N (debug)", overfit_n)
        form.addRow("Best checkpoint metric", best_metric)
        form.addRow("Early stop metric", early_stop_metric)
        form.addRow("Weighted PCK weights", pck_weights)
        form.addRow("Early stop patience (0=off)", patience)
        form.addRow("Early stop min delta", min_delta)
        form.addRow("Early stop min epochs", min_epochs)
        form.addRow("", tb_graph)
        form.addRow("", tb_projector)
        form.addRow("Projector split", tb_projector_split)
        form.addRow("Projector max images", tb_max_images)
        form.addRow("Projector max patches", tb_max_patches)
        form.addRow("Projector patches / image / kpt", tb_per_image_per_kpt)
        form.addRow("Projector pos threshold", tb_pos_thr)
        form.addRow("Projector crop (px)", tb_crop)
        form.addRow("Projector border (px)", tb_border)
        form.addRow("", tb_add_negs)
        form.addRow("Projector neg threshold", tb_neg_thr)
        form.addRow("Projector negatives / image", tb_negs_per_img)
        form.addRow("", cache)
        self._update_dino_head_controls()
        self._update_dino_projector_enabled_state()
        self._update_dino_focal_controls()
        self._update_dino_radius_schedule_controls()
        self._update_dino_model_selection_controls()
        return box

    def _update_dino_model_selection_controls(self) -> None:
        best_combo = getattr(self, "_dino_best_metric_combo", None)
        stop_combo = getattr(self, "_dino_early_stop_metric_combo", None)
        weights_edit = getattr(self, "_dino_pck_weighted_weights_edit", None)
        if best_combo is None or stop_combo is None or weights_edit is None:
            return
        best_val = str(best_combo.currentData() or "")
        stop_val = str(stop_combo.currentData() or "")
        enabled = (best_val == "pck_weighted") or (stop_val == "pck_weighted")
        try:
            weights_edit.setEnabled(enabled)
        except Exception:
            pass

    def _update_dino_head_controls(self) -> None:
        head_type = str(getattr(self, "dino_head_type", "conv")
                        or "conv").strip().lower()
        attn_enabled = head_type in {"attn", "hybrid"}
        for w in (getattr(self, "_dino_attn_heads_spin", None), getattr(self, "_dino_attn_layers_spin", None)):
            if w is not None:
                w.setEnabled(attn_enabled)

    def _update_dino_projector_enabled_state(self) -> None:
        enabled = bool(getattr(self, "dino_tb_projector", True))
        controls = getattr(self, "_dino_projector_controls", None)
        if not controls:
            return
        for widget in controls:
            try:
                widget.setEnabled(enabled)
            except Exception:
                pass

    def _update_dino_focal_controls(self) -> None:
        bce_type = str(getattr(self, "dino_bce_type", "bce")
                       or "bce").strip().lower()
        enabled = bce_type == "focal"
        controls = getattr(self, "_dino_focal_controls", None)
        if not controls:
            return
        for widget in controls:
            try:
                widget.setEnabled(enabled)
            except Exception:
                pass

    def _update_dino_radius_schedule_controls(self) -> None:
        schedule = str(getattr(self, "dino_radius_schedule", "none")
                       or "none").strip().lower()
        enabled = schedule != "none"
        controls = getattr(self, "_dino_radius_controls", None)
        if not controls:
            return
        for widget in controls:
            try:
                widget.setEnabled(enabled)
            except Exception:
                pass

    def _build_dino_augment_groupbox(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("DINO KPSEG Augmentations", self)
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        enable = QtWidgets.QCheckBox(
            "Enable YOLO-like pose augmentations", box)
        enable.setChecked(bool(self.dino_augment_enabled))
        enable.stateChanged.connect(
            lambda _=None: setattr(
                self, "dino_augment_enabled", bool(enable.isChecked()))
        )
        enable.stateChanged.connect(
            lambda _=None: self._update_dino_augment_enabled_state())

        hflip = QtWidgets.QDoubleSpinBox(box)
        hflip.setDecimals(3)
        hflip.setRange(0.0, 1.0)
        hflip.setSingleStep(0.05)
        hflip.setValue(float(self.dino_hflip_prob))
        hflip.valueChanged.connect(lambda v: setattr(
            self, "dino_hflip_prob", float(v)))

        degrees = QtWidgets.QDoubleSpinBox(box)
        degrees.setDecimals(2)
        degrees.setRange(0.0, 45.0)
        degrees.setSingleStep(1.0)
        degrees.setValue(float(self.dino_degrees))
        degrees.valueChanged.connect(
            lambda v: setattr(self, "dino_degrees", float(v)))

        translate = QtWidgets.QDoubleSpinBox(box)
        translate.setDecimals(3)
        translate.setRange(0.0, 0.5)
        translate.setSingleStep(0.01)
        translate.setValue(float(self.dino_translate))
        translate.valueChanged.connect(
            lambda v: setattr(self, "dino_translate", float(v)))

        scale = QtWidgets.QDoubleSpinBox(box)
        scale.setDecimals(3)
        scale.setRange(0.0, 0.9)
        scale.setSingleStep(0.01)
        scale.setValue(float(self.dino_scale))
        scale.valueChanged.connect(
            lambda v: setattr(self, "dino_scale", float(v)))

        brightness = QtWidgets.QDoubleSpinBox(box)
        brightness.setDecimals(3)
        brightness.setRange(0.0, 1.0)
        brightness.setSingleStep(0.01)
        brightness.setValue(float(self.dino_brightness))
        brightness.valueChanged.connect(
            lambda v: setattr(self, "dino_brightness", float(v)))

        contrast = QtWidgets.QDoubleSpinBox(box)
        contrast.setDecimals(3)
        contrast.setRange(0.0, 1.0)
        contrast.setSingleStep(0.01)
        contrast.setValue(float(self.dino_contrast))
        contrast.valueChanged.connect(
            lambda v: setattr(self, "dino_contrast", float(v)))

        saturation = QtWidgets.QDoubleSpinBox(box)
        saturation.setDecimals(3)
        saturation.setRange(0.0, 1.0)
        saturation.setSingleStep(0.01)
        saturation.setValue(float(self.dino_saturation))
        saturation.valueChanged.connect(
            lambda v: setattr(self, "dino_saturation", float(v)))

        seed = QtWidgets.QSpinBox(box)
        seed.setRange(-1, 2_147_483_647)
        seed.setValue(int(self.dino_seed))
        seed.valueChanged.connect(lambda v: setattr(self, "dino_seed", int(v)))

        self._dino_augment_controls = [
            hflip,
            degrees,
            translate,
            scale,
            brightness,
            contrast,
            saturation,
        ]

        form.addRow("", enable)
        form.addRow("HFlip prob", hflip)
        form.addRow("Rotate degrees", degrees)
        form.addRow("Translate (frac)", translate)
        form.addRow("Scale (frac)", scale)
        form.addRow("Brightness", brightness)
        form.addRow("Contrast", contrast)
        form.addRow("Saturation", saturation)
        form.addRow("Seed (-1=random)", seed)
        return box

    def _update_dino_augment_enabled_state(self) -> None:
        enabled = bool(getattr(self, "dino_augment_enabled", False))
        controls = getattr(self, "_dino_augment_controls", None)
        if not controls:
            return
        for widget in controls:
            try:
                widget.setEnabled(enabled)
            except Exception:
                pass

    # -------------------- Accept
    def accept(self) -> None:
        if self.algo in {"YOLO", "DINO KPSEG"} and not self.config_file:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Please select a dataset YAML file.")
            return
        super().accept()
