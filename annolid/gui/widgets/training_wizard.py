"""
Training Wizard - Streamlined model training workflow.

A QWizard-based interface that guides users through dataset selection,
backend choice, parameter configuration, and training launch with
integrated progress monitoring.

GUI goals (this patch):
- Modern, cohesive theme (single QSS)
- Clear dataset selection (YOLO vs COCO tabs, drag/drop, inline validation)
- Clickable backend “cards” with persistent selected styling
- Presets + tidy parameter forms
- Polished review screen + “Copy config” button
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, Signal

from annolid.gui.models_registry import PATCH_SIMILARITY_MODELS


YOLO_TASKS = {
    "Detection": "",
    "Instance Segmentation": "-seg",
    "Pose Estimation": "-pose",
}
YOLO_SIZES = ("n", "s", "m", "l", "x")


# ---------------------------
# Theme + small UI primitives
# ---------------------------

def apply_training_wizard_theme(app: QtWidgets.QApplication, wizard: QtWidgets.QWizard) -> None:
    """Apply a modern dark theme with a consistent accent color."""
    app.setStyle("Fusion")

    # Slightly nicer default font
    f = app.font()
    if f.pointSize() < 11:
        f.setPointSize(11)
    app.setFont(f)

    qss = """
    /* ========= Global ========= */
    QWidget {
        color: #E9EEF5;
        background: #0F141A;
        font-size: 11pt;
    }

    QWizard {
        background: #0F141A;
    }

    /* Wizard page title/subtitle (we add these labels ourselves) */
    QLabel[role="title"] {
        font-size: 20pt;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: 0.3px;
    }
    QLabel[role="subtitle"] {
        color: #A7B3C2;
        font-size: 10.7pt;
    }

    QLabel[muted="true"] { color: #A7B3C2; }
    QLabel[good="true"]  { color: #4CD97B; }
    QLabel[bad="true"]   { color: #FF5C7A; }

    /* ========= Containers ========= */
    QGroupBox {
        border: 1px solid #283241;
        border-radius: 14px;
        margin-top: 14px;
        padding: 12px;
        background: #121922;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 10px;
        margin-left: 6px;
        color: #CFE3FF;
        font-weight: 700;
    }

    /* “Card” frames */
    QFrame[card="true"] {
        background: #121922;
        border: 1px solid #283241;
        border-radius: 16px;
    }
    QFrame[card="true"]:hover {
        border: 1px solid #2B7FFF;
    }
    QFrame[cardSelected="true"] {
        border: 1px solid #2B7FFF;
        background: #121E2C;
    }

    /* ========= Inputs ========= */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextBrowser {
        background: #0D1218;
        border: 1px solid #283241;
        border-radius: 12px;
        padding: 8px 10px;
        selection-background-color: #2B7FFF;
    }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextBrowser:focus {
        border: 1px solid #2B7FFF;
    }

    QTextBrowser {
        padding: 12px;
    }

    /* ========= Buttons ========= */
    QPushButton {
        background: #182230;
        border: 1px solid #2A3A50;
        border-radius: 12px;
        padding: 8px 12px;
        font-weight: 700;
    }
    QPushButton:hover { border-color: #2B7FFF; }
    QPushButton:pressed { background: #142033; }

    /* Primary action */
    QPushButton[primary="true"] {
        background: #2B7FFF;
        border: none;
        color: white;
        padding: 9px 14px;
        border-radius: 12px;
    }
    QPushButton[primary="true"]:hover {
        background: #3B8BFF;
    }

    /* ========= Tabs ========= */
    QTabWidget::pane {
        border: 1px solid #283241;
        border-radius: 14px;
        top: -1px;
        background: #121922;
    }
    QTabBar::tab {
        background: #121922;
        border: 1px solid #283241;
        padding: 8px 14px;
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
        margin-right: 6px;
        color: #A7B3C2;
        font-weight: 700;
    }
    QTabBar::tab:selected {
        color: #FFFFFF;
        border-bottom-color: #121922;
        background: #121E2C;
    }

    /* ========= Checkbox ========= */
    QCheckBox { spacing: 10px; }
    """

    app.setStyleSheet(qss)

    wizard.setWizardStyle(QtWidgets.QWizard.ModernStyle)
    wizard.setOption(QtWidgets.QWizard.HaveHelpButton, False)
    wizard.setOption(QtWidgets.QWizard.IndependentPages, False)


class ClickableCard(QtWidgets.QFrame):
    clicked = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setCursor(QtGui.QCursor(Qt.PointingHandCursor))

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)

    def setSelected(self, selected: bool) -> None:
        self.setProperty("cardSelected", selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class PathPicker(QtWidgets.QWidget):
    changed = QtCore.Signal(str)

    def __init__(self, label: str, placeholder: str, mode: str = "file", parent=None):
        super().__init__(parent)
        self._mode = mode  # "file" or "dir"

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        self._label = QtWidgets.QLabel(label)
        self._label.setProperty("muted", True)
        top.addWidget(self._label)
        top.addStretch()

        self.status = QtWidgets.QLabel("—")
        self.status.setProperty("muted", True)
        top.addWidget(self.status)
        root.addLayout(top)

        row = QtWidgets.QHBoxLayout()
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setClearButtonEnabled(True)
        self.edit.textChanged.connect(self._on_text_changed)

        # Drag/drop support
        self.edit.setAcceptDrops(True)
        self.edit.installEventFilter(self)

        row.addWidget(self.edit, 1)

        self.browse = QtWidgets.QPushButton("Browse…")
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.browse)

        root.addLayout(row)

    def path(self) -> str:
        return self.edit.text().strip()

    def setPath(self, p: str) -> None:
        self.edit.setText(p)

    def setStatus(self, text: str, ok: Optional[bool] = None) -> None:
        self.status.setText(text)
        self.status.setProperty("good", ok is True)
        self.status.setProperty("bad", ok is False)
        self.status.setProperty("muted", ok is None)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.status.update()

    def _browse(self) -> None:
        if self._mode == "file":
            p, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select File", "", "All Files (*)"
            )
        else:
            p = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Folder")
        if p:
            self.setPath(p)

    def _on_text_changed(self, t: str) -> None:
        self.changed.emit(t.strip())

    def eventFilter(self, obj, event):
        if obj is self.edit:
            if event.type() == QtCore.QEvent.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            if event.type() == QtCore.QEvent.Drop:
                urls = event.mimeData().urls()
                if urls:
                    self.setPath(urls[0].toLocalFile())
                return True
        return super().eventFilter(obj, event)


def _header(title: str, subtitle: str) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    l = QtWidgets.QVBoxLayout(w)
    l.setContentsMargins(0, 0, 0, 0)
    l.setSpacing(6)

    t = QtWidgets.QLabel(title)
    t.setProperty("role", "title")
    s = QtWidgets.QLabel(subtitle)
    s.setProperty("role", "subtitle")
    s.setWordWrap(True)

    l.addWidget(t)
    l.addWidget(s)
    return w


# ---------------------------
# Page 1: Dataset selection
# ---------------------------

class SelectDatasetPage(QtWidgets.QWizardPage):
    """Select YOLO data.yaml or COCO folder, with preview + validation."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Select Dataset")
        self.setSubTitle("")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setSpacing(14)

        outer.addWidget(_header(
            "Dataset",
            "Pick a YOLO data.yaml file or a COCO folder. You can drag & drop a file/folder into the field."
        ))

        source_group = QtWidgets.QGroupBox("Dataset Source")
        source_layout = QtWidgets.QVBoxLayout(source_group)

        self.source_tabs = QtWidgets.QTabWidget()
        self.source_tabs.setDocumentMode(True)
        self.source_tabs.currentChanged.connect(
            lambda _: self._validate_dataset())

        # YOLO tab
        yolo_tab = QtWidgets.QWidget()
        yolo_l = QtWidgets.QVBoxLayout(yolo_tab)
        yolo_l.setSpacing(10)

        self.yolo_picker = PathPicker(
            "YOLO data.yaml",
            "Select data.yaml (e.g., /path/to/data.yaml)",
            mode="file",
        )
        self.yolo_picker.changed.connect(self._on_yolo_changed)
        yolo_l.addWidget(self.yolo_picker)

        hint = QtWidgets.QLabel(
            "Tip: Use the Dataset Export wizard if you need to generate a YOLO dataset.")
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        yolo_l.addWidget(hint)

        yolo_l.addStretch()
        self.source_tabs.addTab(yolo_tab, "YOLO")

        # COCO tab
        coco_tab = QtWidgets.QWidget()
        coco_l = QtWidgets.QVBoxLayout(coco_tab)
        coco_l.setSpacing(10)

        self.coco_picker = PathPicker(
            "COCO folder",
            "Folder containing train.json (and optionally val.json)",
            mode="dir",
        )
        self.coco_picker.changed.connect(self._on_coco_changed)
        coco_l.addWidget(self.coco_picker)

        hint2 = QtWidgets.QLabel(
            "Expected files: train.json and (optional) val.json")
        hint2.setProperty("muted", True)
        hint2.setWordWrap(True)
        coco_l.addWidget(hint2)

        coco_l.addStretch()
        self.source_tabs.addTab(coco_tab, "COCO")

        source_layout.addWidget(self.source_tabs)
        outer.addWidget(source_group)

        # Preview
        info_group = QtWidgets.QGroupBox("Preview")
        form = QtWidgets.QFormLayout(info_group)
        form.setLabelAlignment(Qt.AlignLeft)

        self.info_classes = QtWidgets.QLabel("—")
        self.info_classes.setProperty("muted", True)
        self.info_train = QtWidgets.QLabel("—")
        self.info_train.setProperty("muted", True)
        self.info_val = QtWidgets.QLabel("—")
        self.info_val.setProperty("muted", True)
        self.info_type = QtWidgets.QLabel("—")
        self.info_type.setProperty("muted", True)

        form.addRow("Classes:", self.info_classes)
        form.addRow("Training images:", self.info_train)
        form.addRow("Validation images:", self.info_val)
        form.addRow("Task type:", self.info_type)

        outer.addWidget(info_group)

        # Bottom actions
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch()
        self.create_dataset_btn = QtWidgets.QPushButton("Create Dataset…")
        self.create_dataset_btn.setToolTip("Open the Dataset Export wizard")
        self.create_dataset_btn.clicked.connect(self._open_dataset_wizard)
        actions.addWidget(self.create_dataset_btn)
        outer.addLayout(actions)

        outer.addStretch()

        self._validate_dataset()

    def _on_yolo_changed(self, _: str) -> None:
        # switching tabs shouldn’t keep stale status
        self._validate_dataset()

    def _on_coco_changed(self, _: str) -> None:
        self._validate_dataset()

    def _reset_preview(self) -> None:
        # Some signals may fire during construction before these widgets
        # are created; guard against AttributeError by checking presence.
        try:
            self.info_classes.setText("—")
            self.info_train.setText("—")
            self.info_val.setText("—")
            self.info_type.setText("—")
        except AttributeError:
            return

    def _validate_dataset(self) -> None:
        self._reset_preview()

        if self.get_dataset_type() == "yolo":
            p = Path(self.yolo_picker.path())
            if not self.yolo_picker.path():
                self.yolo_picker.setStatus("—", None)
            elif p.exists() and p.is_file():
                self.yolo_picker.setStatus("Found ✓", True)
                self._parse_yolo_yaml(p)
            else:
                self.yolo_picker.setStatus("Not found", False)

            # Keep COCO status muted when not active
            if self.coco_picker.path():
                self.coco_picker.setStatus("—", None)

        else:
            d = Path(self.coco_picker.path())
            if not self.coco_picker.path():
                self.coco_picker.setStatus("—", None)
            elif d.exists() and d.is_dir():
                train_json = d / "train.json"
                if train_json.exists():
                    self.coco_picker.setStatus("OK ✓", True)
                    self._parse_coco_dir(d)
                else:
                    self.coco_picker.setStatus("Missing train.json", False)
                    # still parse what we can
                    self._parse_coco_dir(d)
            else:
                self.coco_picker.setStatus("Not found", False)

            # Keep YOLO status muted when not active
            if self.yolo_picker.path():
                self.yolo_picker.setStatus("—", None)

        self.completeChanged.emit()

    def _parse_yolo_yaml(self, path: Path) -> None:
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            names = data.get("names", [])
            if isinstance(names, dict):
                names = list(names.values())
            if not isinstance(names, list):
                names = [names]
            preview = [str(n) for n in names[:6]]
            self.info_classes.setText(
                f"{len(names)}: {', '.join(preview)}{'…' if len(names) > 6 else ''}"
            )

            train_path = data.get("train", "")
            val_path = data.get("val", "")
            base = path.parent

            train_count = self._count_images(
                base / train_path) if train_path else 0
            val_count = self._count_images(base / val_path) if val_path else 0

            self.info_train.setText(str(train_count) if train_count else "—")
            self.info_val.setText(str(val_count) if val_count else "—")

            if "kpt_shape" in data:
                self.info_type.setText("Pose Estimation")
            elif any("segment" in str(v).lower() for v in data.values()):
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
                with open(train_json, "r") as f:
                    data = json.load(f)
                cats = data.get("categories", [])
                self.info_classes.setText(f"{len(cats)} categories")
                self.info_train.setText(
                    f"{len(data.get('images', []))} images")
                self.info_type.setText("COCO Detection/Segmentation")

            if val_json.exists():
                import json
                with open(val_json, "r") as f:
                    data = json.load(f)
                self.info_val.setText(f"{len(data.get('images', []))} images")

        except Exception as e:
            self.info_classes.setText(f"Error: {e}")

    def _count_images(self, path: Path) -> int:
        if not path.exists():
            return 0
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if path.is_file():
            # list file
            return sum(1 for line in path.read_text().splitlines() if line.strip())
        return sum(1 for f in path.iterdir() if f.suffix.lower() in extensions)

    def _open_dataset_wizard(self) -> None:
        from annolid.gui.widgets.dataset_wizard import DatasetExportWizard
        wizard = DatasetExportWizard(parent=self)
        if wizard.exec_() == QtWidgets.QDialog.Accepted:
            if getattr(wizard.export_progress_page, "_output_path", None):
                outp = wizard.export_progress_page._output_path
                data_yaml = Path(outp) / "data.yaml"
                if data_yaml.exists():
                    self.source_tabs.setCurrentIndex(0)
                    self.yolo_picker.setPath(str(data_yaml))

    def isComplete(self) -> bool:
        p = self.get_dataset_path().strip()
        if not p:
            return False
        if self.get_dataset_type() == "yolo":
            return Path(p).exists() and Path(p).is_file()
        # coco
        d = Path(p)
        return d.exists() and d.is_dir() and (d / "train.json").exists()

    def get_dataset_path(self) -> str:
        return self.yolo_picker.path() if self.get_dataset_type() == "yolo" else self.coco_picker.path()

    def get_dataset_type(self) -> str:
        return "yolo" if self.source_tabs.currentIndex() == 0 else "coco"


# ---------------------------
# Page 2: Backend selection
# ---------------------------

class SelectBackendPage(QtWidgets.QWizardPage):
    """Choose training backend."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Choose Backend")
        self.setSubTitle("")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(14)

        layout.addWidget(_header(
            "Training Backend",
            "Pick a training engine. YOLO is the default for most workflows."
        ))

        self.backend_group = QtWidgets.QButtonGroup(self)
        self.backend_cards: List[ClickableCard] = []

        backends = [
            (
                "yolo",
                "YOLO (Ultralytics)",
                "Fast, strong default for detection / segmentation / pose. Great for production + real-time.",
                ["Detection", "Segmentation", "Pose",
                    "Fast training", "Real-time inference"],
                True,
            ),
            (
                "dino_kpseg",
                "DINO KPSEG",
                "Self-supervised keypoint discovery using DINO features. Excellent for limited labels / few-shot setups.",
                ["Keypoints", "Few-shot", "Transfer learning"],
                False,
            ),
            (
                "maskrcnn",
                "Mask R-CNN (Detectron2)",
                "Classic instance segmentation baseline. Useful for research comparisons and certain segmentation tasks.",
                ["Instance segmentation", "Research baseline"],
                False,
            ),
        ]

        for i, (backend_id, title, desc, feats, recommended) in enumerate(backends):
            card, radio = self._make_backend_card(
                backend_id, title, desc, feats, recommended)
            self.backend_group.addButton(radio, i)
            self.backend_cards.append(card)
            layout.addWidget(card)

        # Default selection: YOLO
        first_radio = self.backend_group.button(0)
        if first_radio:
            first_radio.setChecked(True)

        layout.addStretch()

    def _make_backend_card(
        self,
        backend_id: str,
        title: str,
        description: str,
        features: List[str],
        recommended: bool,
    ) -> tuple[ClickableCard, QtWidgets.QRadioButton]:

        card = ClickableCard()
        root = QtWidgets.QVBoxLayout(card)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        # Header
        header = QtWidgets.QHBoxLayout()
        radio = QtWidgets.QRadioButton()
        radio.setObjectName(backend_id)
        header.addWidget(radio)

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setStyleSheet("font-size: 13pt; font-weight: 800;")
        header.addWidget(title_lbl)

        if recommended:
            badge = QtWidgets.QLabel("Recommended")
            badge.setStyleSheet("""
                background: #2B7FFF;
                color: white;
                padding: 2px 10px;
                border-radius: 10px;
                font-size: 10pt;
                font-weight: 800;
            """)
            header.addWidget(badge)

        header.addStretch()
        root.addLayout(header)

        # Description
        desc = QtWidgets.QLabel(description)
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        desc.setStyleSheet("margin-left: 26px;")
        root.addWidget(desc)

        # Features
        feats_lbl = QtWidgets.QLabel(" • ".join(features))
        feats_lbl.setWordWrap(True)
        feats_lbl.setProperty("muted", True)
        feats_lbl.setStyleSheet("margin-left: 26px;")
        root.addWidget(feats_lbl)

        # Card click selects radio; radio toggles card selected styling
        card.clicked.connect(radio.click)
        radio.toggled.connect(lambda on, c=card: c.setSelected(on))

        return card, radio

    def get_backend(self) -> str:
        checked = self.backend_group.checkedButton()
        return checked.objectName() if checked else "yolo"


# ---------------------------
# Page 3: Configure training
# ---------------------------

class ConfigureParametersPage(QtWidgets.QWizardPage):
    """Backend-specific parameter configuration + shared output options."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Configure")
        self.setSubTitle("")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(14)

        layout.addWidget(_header(
            "Training Settings",
            "Defaults are good. Use presets for quick tuning, or expand advanced options."
        ))

        self.stack = QtWidgets.QStackedWidget()
        self.yolo_widget = self._create_yolo_options()
        self.dino_widget = self._create_dino_options()
        self.maskrcnn_widget = self._create_maskrcnn_options()

        self.stack.addWidget(self.yolo_widget)
        self.stack.addWidget(self.dino_widget)
        self.stack.addWidget(self.maskrcnn_widget)

        layout.addWidget(self.stack)

        # Output options (common)
        common_group = QtWidgets.QGroupBox("Output")
        common_layout = QtWidgets.QFormLayout(common_group)

        dir_layout = QtWidgets.QHBoxLayout()
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText(
            "Optional: choose where to store outputs (runs, checkpoints, logs)")
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

    # ---- YOLO ----

    def _create_yolo_options(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Model selection
        model_group = QtWidgets.QGroupBox("YOLO Model")
        model_layout = QtWidgets.QGridLayout(model_group)
        model_layout.setHorizontalSpacing(12)
        model_layout.setVerticalSpacing(10)

        self.yolo_preset = QtWidgets.QComboBox()
        self.yolo_preset.addItems(
            ["Balanced (recommended)", "Fast", "Accurate"])
        self.yolo_preset.currentTextChanged.connect(self._apply_yolo_preset)
        model_layout.addWidget(QtWidgets.QLabel("Preset:"), 0, 0)
        model_layout.addWidget(self.yolo_preset, 0, 1, 1, 3)

        self.yolo_task_combo = QtWidgets.QComboBox()
        self.yolo_task_combo.addItems(list(YOLO_TASKS.keys()))
        self.yolo_task_combo.setCurrentText("Instance Segmentation")

        self.yolo_size_combo = QtWidgets.QComboBox()
        self.yolo_size_combo.addItems([s.upper() for s in YOLO_SIZES])

        self.yolo_model_label = QtWidgets.QLabel("yolo11n-seg.pt")
        self.yolo_model_label.setStyleSheet("font-weight: 900;")

        model_layout.addWidget(QtWidgets.QLabel("Task:"), 1, 0)
        model_layout.addWidget(self.yolo_task_combo, 1, 1)
        model_layout.addWidget(QtWidgets.QLabel("Size:"), 1, 2)
        model_layout.addWidget(self.yolo_size_combo, 1, 3)

        model_layout.addWidget(QtWidgets.QLabel("Model:"), 2, 0)
        model_layout.addWidget(self.yolo_model_label, 2, 1, 1, 3)

        self.yolo_task_combo.currentTextChanged.connect(
            self._update_yolo_model)
        self.yolo_size_combo.currentTextChanged.connect(
            self._update_yolo_model)

        layout.addWidget(model_group)

        # Training params
        train_group = QtWidgets.QGroupBox("Core Parameters")
        train_layout = QtWidgets.QFormLayout(train_group)

        self.yolo_device_combo = QtWidgets.QComboBox()
        self._populate_device_combo(self.yolo_device_combo)
        train_layout.addRow("Device:", self.yolo_device_combo)

        self.yolo_epochs_spin = QtWidgets.QSpinBox()
        self.yolo_epochs_spin.setRange(1, 1000)
        self.yolo_epochs_spin.setValue(100)
        train_layout.addRow("Epochs:", self.yolo_epochs_spin)

        self.yolo_batch_spin = QtWidgets.QSpinBox()
        self.yolo_batch_spin.setRange(1, 128)
        self.yolo_batch_spin.setValue(8)
        train_layout.addRow("Batch size:", self.yolo_batch_spin)

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
        self.yolo_lr_spin.setRange(0.0001, 0.2)
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

        self._update_yolo_model()
        self._apply_yolo_preset(self.yolo_preset.currentText())

        return widget

    def _apply_yolo_preset(self, name: str) -> None:
        # Friendly defaults that “feel smart”
        if "Fast" in name:
            self.yolo_epochs_spin.setValue(50)
            self.yolo_batch_spin.setValue(16)
            self.yolo_imgsz_spin.setValue(640)
            self.yolo_lr_spin.setValue(0.01)
            self.yolo_patience_spin.setValue(50)
        elif "Accurate" in name:
            self.yolo_epochs_spin.setValue(200)
            self.yolo_batch_spin.setValue(4)
            self.yolo_imgsz_spin.setValue(960)
            self.yolo_lr_spin.setValue(0.01)
            self.yolo_patience_spin.setValue(120)
        else:
            self.yolo_epochs_spin.setValue(100)
            self.yolo_batch_spin.setValue(8)
            self.yolo_imgsz_spin.setValue(640)
            self.yolo_lr_spin.setValue(0.01)
            self.yolo_patience_spin.setValue(100)

    # ---- DINO KPSEG ----

    def _create_dino_options(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

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
        self.dino_lr_spin.setRange(0.0001, 0.05)
        self.dino_lr_spin.setDecimals(4)
        self.dino_lr_spin.setValue(0.002)
        train_layout.addRow("Learning rate:", self.dino_lr_spin)

        self.dino_radius_spin = QtWidgets.QDoubleSpinBox()
        self.dino_radius_spin.setRange(1.0, 30.0)
        self.dino_radius_spin.setValue(6.0)
        train_layout.addRow("Keypoint radius (px):", self.dino_radius_spin)

        layout.addWidget(train_group)
        layout.addStretch()
        return widget

    # ---- Mask R-CNN ----

    def _create_maskrcnn_options(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

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

    # ---- helpers ----

    def _populate_device_combo(self, combo: QtWidgets.QComboBox) -> None:
        options = [("Auto (recommended)", "")]
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                options.append(("Apple MPS", "mps"))
            if torch.cuda.is_available():
                # show GPU name if possible
                try:
                    name = torch.cuda.get_device_name(0)
                    options.insert(1, (f"CUDA (GPU 0) — {name}", "0"))
                except Exception:
                    options.insert(1, ("CUDA (GPU 0)", "0"))
        except ImportError:
            pass
        options.append(("CPU", "cpu"))

        combo.clear()
        for label, value in options:
            combo.addItem(label, userData=value)

    def _update_yolo_model(self) -> None:
        task = self.yolo_task_combo.currentText()
        size = self.yolo_size_combo.currentText().lower()
        suffix = YOLO_TASKS.get(task, "-seg")
        self.yolo_model_label.setText(f"yolo11{size}{suffix}.pt")

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if folder:
            self.output_dir_edit.setText(folder)

    def get_config(self) -> Dict[str, Any]:
        wizard = self.wizard()
        backend = wizard.select_backend_page.get_backend(
        ) if isinstance(wizard, TrainingWizard) else "yolo"

        config: Dict[str, Any] = {
            "backend": backend,
            "output_dir": self.output_dir_edit.text().strip() or None,
        }

        if backend == "yolo":
            task = self.yolo_task_combo.currentText()
            size = self.yolo_size_combo.currentText().lower()
            suffix = YOLO_TASKS.get(task, "-seg")

            config.update({
                "model": f"yolo11{size}{suffix}.pt",
                "preset": self.yolo_preset.currentText(),
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

        else:  # maskrcnn
            config.update({
                "max_iterations": self.maskrcnn_iter_spin.value(),
                "batch": self.maskrcnn_batch_spin.value(),
            })

        return config


# ---------------------------
# Page 4: Review + launch
# ---------------------------

class TrainingSummaryPage(QtWidgets.QWizardPage):
    """Review configuration, choose monitoring options, then start training."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Review")
        self.setSubTitle("")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(14)

        layout.addWidget(_header(
            "Review & Launch",
            "Double-check settings, then start training."
        ))

        self.summary_text = QtWidgets.QTextBrowser()
        self.summary_text.setOpenExternalLinks(False)
        self.summary_text.setMinimumHeight(260)
        layout.addWidget(self.summary_text)

        # Quick actions row
        quick = QtWidgets.QHBoxLayout()
        self.copy_btn = QtWidgets.QPushButton("Copy configuration")
        self.copy_btn.clicked.connect(self._copy_summary_text)
        quick.addWidget(self.copy_btn)

        quick.addStretch()
        layout.addLayout(quick)

        options_group = QtWidgets.QGroupBox("Monitoring")
        options_layout = QtWidgets.QVBoxLayout(options_group)

        self.open_dashboard_check = QtWidgets.QCheckBox(
            "Open training dashboard to monitor progress")
        self.open_dashboard_check.setChecked(True)
        options_layout.addWidget(self.open_dashboard_check)

        self.open_tensorboard_check = QtWidgets.QCheckBox(
            "Launch TensorBoard for visualization")
        options_layout.addWidget(self.open_tensorboard_check)

        layout.addWidget(options_group)

        self.time_estimate = QtWidgets.QLabel(
            "Estimated time: varies with dataset size and hardware.")
        self.time_estimate.setProperty("muted", True)
        layout.addWidget(self.time_estimate)

        layout.addStretch()

    def _copy_summary_text(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.summary_text.toPlainText())

    def initializePage(self) -> None:
        wizard = self.wizard()
        if not isinstance(wizard, TrainingWizard):
            return

        dataset_path = wizard.select_dataset_page.get_dataset_path()
        dataset_type = wizard.select_dataset_page.get_dataset_type()
        backend = wizard.select_backend_page.get_backend()
        config = wizard.configure_params_page.get_config()

        backend_names = {
            "yolo": "YOLO (Ultralytics)",
            "dino_kpseg": "DINO KPSEG",
            "maskrcnn": "Mask R-CNN (Detectron2)",
        }

        # Build clean “review screen” HTML
        def row(k: str, v: str) -> str:
            return f"""
            <tr>
              <td style="padding: 6px 10px; color:#A7B3C2; width: 180px; font-weight:700;">{k}</td>
              <td style="padding: 6px 10px; color:#E9EEF5; font-weight:700;">{v}</td>
            </tr>
            """

        ds_name = Path(dataset_path).name if dataset_path else "—"
        ds_kind = "YOLO" if dataset_type == "yolo" else "COCO"
        bk = backend_names.get(backend, backend)

        params_rows = []
        if backend == "yolo":
            params_rows += [
                row("Model", str(config.get("model", "yolo11n-seg.pt"))),
                row("Preset", str(config.get("preset", "Balanced"))),
                row("Epochs", str(config.get("epochs", 100))),
                row("Batch", str(config.get("batch", 8))),
                row("Image size", str(config.get("imgsz", 640))),
                row("Device", str(config.get("device", "auto") or "auto")),
            ]
        elif backend == "dino_kpseg":
            params_rows += [
                row("Model", str(config.get("model", "DINO"))),
                row("Epochs", str(config.get("epochs", 100))),
                row("Batch", str(config.get("batch", 8))),
                row("Short side", str(config.get("short_side", 768))),
                row("LR", str(config.get("lr", 0.002))),
                row("Radius (px)", str(config.get("radius_px", 6.0))),
            ]
        else:
            params_rows += [
                row("Max iterations", str(config.get("max_iterations", 2000))),
                row("Batch", str(config.get("batch", 2))),
            ]

        out_dir = config.get("output_dir") or "Default runs directory"

        html = f"""
        <div style="font-family: -apple-system, Segoe UI, Roboto, Arial;">
          <div style="margin-bottom: 10px;">
            <span style="font-size: 15pt; font-weight: 900; color: #FFFFFF;">Configuration</span>
          </div>

          <div style="background:#0D1218; border:1px solid #283241; border-radius:14px; padding:12px; margin-bottom:12px;">
            <div style="color:#CFE3FF; font-weight:900; margin-bottom:8px;">Dataset</div>
            <table style="width:100%; border-collapse:collapse;">
              {row("Type", ds_kind)}
              {row("File/Folder", ds_name)}
            </table>
          </div>

          <div style="background:#0D1218; border:1px solid #283241; border-radius:14px; padding:12px; margin-bottom:12px;">
            <div style="color:#CFE3FF; font-weight:900; margin-bottom:8px;">Backend</div>
            <table style="width:100%; border-collapse:collapse;">
              {row("Training engine", bk)}
            </table>
          </div>

          <div style="background:#0D1218; border:1px solid #283241; border-radius:14px; padding:12px; margin-bottom:12px;">
            <div style="color:#CFE3FF; font-weight:900; margin-bottom:8px;">Parameters</div>
            <table style="width:100%; border-collapse:collapse;">
              {''.join(params_rows)}
            </table>
          </div>

          <div style="background:#0D1218; border:1px solid #283241; border-radius:14px; padding:12px;">
            <div style="color:#CFE3FF; font-weight:900; margin-bottom:8px;">Output</div>
            <table style="width:100%; border-collapse:collapse;">
              {row("Directory", str(out_dir))}
            </table>
          </div>
        </div>
        """

        self.summary_text.setHtml(html)

    def should_open_dashboard(self) -> bool:
        return self.open_dashboard_check.isChecked()

    def should_open_tensorboard(self) -> bool:
        return self.open_tensorboard_check.isChecked()


# ---------------------------
# Main wizard
# ---------------------------

class TrainingWizard(QtWidgets.QWizard):
    """Main training wizard combining all pages."""

    training_requested = Signal(dict)  # config dict

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        app = QtWidgets.QApplication.instance()
        if app:
            apply_training_wizard_theme(app, self)

        self.setWindowTitle("Training Wizard")
        self.setMinimumSize(860, 640)

        # Pages
        self.select_dataset_page = SelectDatasetPage()
        self.select_backend_page = SelectBackendPage()
        self.configure_params_page = ConfigureParametersPage()
        self.summary_page = TrainingSummaryPage()

        self.addPage(self.select_dataset_page)
        self.addPage(self.select_backend_page)
        self.addPage(self.configure_params_page)
        self.addPage(self.summary_page)

        # Prefill dataset if provided
        if dataset_path:
            self.select_dataset_page.source_tabs.setCurrentIndex(0)
            self.select_dataset_page.yolo_picker.setPath(dataset_path)

        # Buttons
        self.setButtonText(QtWidgets.QWizard.FinishButton, "Start Training")
        self.setButtonText(QtWidgets.QWizard.NextButton, "Next →")
        self.setButtonText(QtWidgets.QWizard.BackButton, "← Back")

        # Make finish button look primary
        finish_btn = self.button(QtWidgets.QWizard.FinishButton)
        if finish_btn:
            finish_btn.setProperty("primary", True)
            finish_btn.style().unpolish(finish_btn)
            finish_btn.style().polish(finish_btn)

    def accept(self) -> None:
        config = self.configure_params_page.get_config()
        config["dataset_path"] = self.select_dataset_page.get_dataset_path()
        config["dataset_type"] = self.select_dataset_page.get_dataset_type()
        config["open_dashboard"] = self.summary_page.should_open_dashboard()
        config["open_tensorboard"] = self.summary_page.should_open_tensorboard()

        self.training_requested.emit(config)
        super().accept()

    def get_training_config(self) -> Dict[str, Any]:
        config = self.configure_params_page.get_config()
        config["dataset_path"] = self.select_dataset_page.get_dataset_path()
        config["dataset_type"] = self.select_dataset_page.get_dataset_type()
        return config
