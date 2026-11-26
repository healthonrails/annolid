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

import html as _html
from pathlib import Path
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, Signal

from annolid.gui.widgets.dino_training_ui_shared import (
    apply_dino_bce_control_state,
    apply_dino_head_control_state,
    configure_dino_model_combo,
    create_dino_head_loss_controls,
)
from annolid.gui.widgets.yolo_training_ui_shared import (
    compute_weight_for_selectors,
    default_yolo_plots_enabled,
    format_yolo_weight_display,
    get_yolo_train_preset,
    parse_yolo_weight_spec,
    populate_yolo_device_combo,
    populate_yolo_family_combo,
    populate_yolo_preset_combo,
    populate_yolo_size_combo,
    populate_yolo_task_combo,
    populate_yolo_weight_combo,
    select_or_insert_yolo_weight,
    task_key_to_label,
    yolo_registry_weight_configs,
)
from annolid.segmentation.dino_kpseg import defaults as dino_defaults
from annolid.yolo import (
    YOLO_MODEL_SIZES as YOLO11_MODEL_SIZES,
    YOLO_TASK_SUFFIXES as YOLO11_TASK_SUFFIXES,  # noqa: F401
    recommend_preset_for_dataset,
)

YOLO_TASKS = {
    "Detection": "",
    "Instance Segmentation": "-seg",
    "Pose Estimation": "-pose",
}
YOLO_SIZES = YOLO11_MODEL_SIZES


# ---------------------------
# Theme + small UI primitives
# ---------------------------


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
            p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if p:
            self.setPath(p)

    def _on_text_changed(self, t: str) -> None:
        self.changed.emit(t.strip())

    def eventFilter(self, obj, event):
        if obj is self.edit:
            if event.type() == QtCore.QEvent.DragEnter:
                if event.mimeData().hasUrls():
                    event.setDropAction(QtCore.Qt.CopyAction)
                    event.accept()
                    return True
            if event.type() == QtCore.QEvent.Drop:
                urls = event.mimeData().urls()
                if urls:
                    self.setPath(urls[0].toLocalFile())
                event.setDropAction(QtCore.Qt.CopyAction)
                event.accept()
                return True
        return super().eventFilter(obj, event)


def _header(title: str, subtitle: str) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    t = QtWidgets.QLabel(title)
    t.setProperty("role", "title")
    s = QtWidgets.QLabel(subtitle)
    s.setProperty("role", "subtitle")
    s.setWordWrap(True)

    layout.addWidget(t)
    layout.addWidget(s)
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

        outer.addWidget(
            _header(
                "Dataset",
                "Pick a YOLO data.yaml file or a COCO folder. You can drag & drop a file/folder into the field.",
            )
        )

        source_group = QtWidgets.QGroupBox("Dataset Source")
        source_layout = QtWidgets.QVBoxLayout(source_group)

        self.source_tabs = QtWidgets.QTabWidget()
        self.source_tabs.setDocumentMode(True)

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
            "Tip: Use the Dataset Export wizard if you need to generate a YOLO dataset."
        )
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

        hint2 = QtWidgets.QLabel("Expected files: train.json and (optional) val.json")
        hint2.setProperty("muted", True)
        hint2.setWordWrap(True)
        coco_l.addWidget(hint2)

        coco_l.addStretch()
        self.source_tabs.addTab(coco_tab, "COCO")

        # LabelMe tab (DinoKPSEG)
        labelme_tab = QtWidgets.QWidget()
        labelme_l = QtWidgets.QVBoxLayout(labelme_tab)
        labelme_l.setSpacing(10)

        self.labelme_picker = PathPicker(
            "LabelMe spec.yaml",
            "Select spec.yaml (LabelMe)",
            mode="file",
        )
        self.labelme_picker.changed.connect(self._on_labelme_changed)
        labelme_l.addWidget(self.labelme_picker)

        hint3 = QtWidgets.QLabel(
            "Spec must include kpt_shape and keypoint_names. Train/val can be dirs or JSONL."
        )
        hint3.setProperty("muted", True)
        hint3.setWordWrap(True)
        labelme_l.addWidget(hint3)

        labelme_l.addStretch()
        self.source_tabs.addTab(labelme_tab, "LabelMe")

        source_layout.addWidget(self.source_tabs)
        # Connect after pickers/tabs are created to avoid triggering
        # validation before attributes exist during construction.
        self.source_tabs.currentChanged.connect(lambda _: self._validate_dataset())
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

    def _on_labelme_changed(self, _: str) -> None:
        self._validate_dataset()

    def _reset_preview(self) -> None:
        # Some signals may fire during construction before these widgets
        # are created; guard against AttributeError by checking presence.
        self._dataset_summary = None
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
            raw_path = self.yolo_picker.path()
            p = Path(raw_path)
            if not raw_path:
                self.yolo_picker.setStatus("—", None)
            elif p.exists() and (p.is_file() or p.is_dir()):
                self._parse_yolo_yaml(p)
            else:
                self.yolo_picker.setStatus("Not found", False)

            if self.coco_picker.path():
                self.coco_picker.setStatus("—", None)
            if self.labelme_picker.path():
                self.labelme_picker.setStatus("—", None)

        elif self.get_dataset_type() == "coco":
            raw_path = self.coco_picker.path()
            d = Path(raw_path)
            if not raw_path:
                self.coco_picker.setStatus("—", None)
            elif d.exists() and d.is_dir():
                self._parse_coco_dir(d)
            else:
                self.coco_picker.setStatus("Not found", False)

            if self.yolo_picker.path():
                self.yolo_picker.setStatus("—", None)
            if self.labelme_picker.path():
                self.labelme_picker.setStatus("—", None)

        else:
            p = Path(self.labelme_picker.path())
            if not self.labelme_picker.path():
                self.labelme_picker.setStatus("—", None)
            elif p.exists() and p.is_file():
                self.labelme_picker.setStatus("Found ✓", True)
                self._parse_labelme_yaml(p)
            else:
                self.labelme_picker.setStatus("Not found", False)

            if self.yolo_picker.path():
                self.yolo_picker.setStatus("—", None)
            if self.coco_picker.path():
                self.coco_picker.setStatus("—", None)

        self.completeChanged.emit()

    def _apply_dataset_summary_preview(
        self, summary: Any, picker: PathPicker, *, fallback_type: str
    ) -> None:
        self._dataset_summary = summary
        if summary.is_valid:
            picker.setStatus("OK ✓", True)
        else:
            issue = summary.issues[0] if summary.issues else "Invalid dataset"
            picker.setStatus(issue[:42], False)

        names = list(summary.class_names)
        if names:
            preview = [str(n) for n in names[:6]]
            self.info_classes.setText(
                f"{len(names)}: {', '.join(preview)}{'…' if len(names) > 6 else ''}"
            )
        else:
            self.info_classes.setText("—")

        self.info_train.setText(
            f"{summary.train_images} images" if summary.train_images else "—"
        )
        self.info_val.setText(
            f"{summary.val_images} images" if summary.val_images else "—"
        )
        task_label = (
            task_key_to_label(summary.task) if summary.task != "unknown" else None
        )
        if summary.kpt_shape:
            self.info_type.setText(
                f"{task_label or 'Pose Estimation'} ({summary.kpt_shape[0]} kpts)"
            )
        else:
            self.info_type.setText(task_label or fallback_type)

    def _parse_yolo_yaml(self, path: Path) -> None:
        try:
            from annolid.yolo.dataset_prep import inspect_yolo_dataset

            summary = inspect_yolo_dataset(path)
            self._apply_dataset_summary_preview(
                summary, self.yolo_picker, fallback_type="YOLO Detection"
            )
        except Exception as e:
            self.yolo_picker.setStatus("Invalid YAML", False)
            self.info_classes.setText(f"Error: {e}")

    def _parse_labelme_yaml(self, path: Path) -> None:
        try:
            import yaml

            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}

            keypoint_names = data.get("keypoint_names") or data.get("kpt_names") or []
            if isinstance(keypoint_names, dict):
                keypoint_names = list(keypoint_names.values())
            if not isinstance(keypoint_names, list):
                keypoint_names = [keypoint_names]
            preview = [str(n) for n in keypoint_names[:6]]
            self.info_classes.setText(
                f"{len(keypoint_names)}: {', '.join(preview)}{'…' if len(keypoint_names) > 6 else ''}"
            )

            base = path.parent
            train_value = data.get("train")
            val_value = data.get("val")
            train_count = self._count_labelme_items(train_value, base)
            val_count = self._count_labelme_items(val_value, base)

            self.info_train.setText(str(train_count) if train_count else "—")
            self.info_val.setText(str(val_count) if val_count else "—")
            self.info_type.setText("Pose Estimation (LabelMe)")

        except Exception as e:
            self.info_classes.setText(f"Error: {e}")

    def _parse_coco_dir(self, path: Path) -> None:
        try:
            from annolid.yolo.dataset_prep import inspect_yolo_dataset

            summary = inspect_yolo_dataset(path)
            self._apply_dataset_summary_preview(
                summary,
                self.coco_picker,
                fallback_type="COCO Detection/Segmentation",
            )
        except Exception as e:
            self.coco_picker.setStatus("Invalid COCO", False)
            self.info_classes.setText(f"Error: {e}")

    def _count_images(self, path: Path) -> int:
        if not path.exists():
            return 0
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if path.is_file():
            return sum(1 for line in path.read_text().splitlines() if line.strip())
        return sum(1 for f in path.iterdir() if f.suffix.lower() in extensions)

    def _count_labelme_items(self, entry: object, base: Path) -> int:
        if not entry:
            return 0
        if isinstance(entry, (list, tuple)):
            return sum(self._count_labelme_items(item, base) for item in entry)
        try:
            raw = str(entry).strip()
        except Exception:
            return 0
        if not raw:
            return 0
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = base / p
        if not p.exists():
            return 0
        if p.is_dir():
            return sum(1 for f in p.rglob("*.json"))
        if p.is_file():
            if p.suffix.lower() in {".txt", ".list", ".jsonl"}:
                return sum(
                    1
                    for line in p.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            if p.suffix.lower() == ".json":
                return 1
        return 0

    def _open_dataset_wizard(self) -> None:
        from annolid.gui.widgets.dataset_wizard import DatasetExportWizard

        wizard = DatasetExportWizard(parent=self)
        wizard.export_complete.connect(self._on_dataset_export_complete)
        wizard.exec_()

    def _on_dataset_export_complete(self, output_path: Path, format_type: str) -> None:
        """Handle dataset export completion by updating the matching dataset tab."""
        if format_type == "yolo":
            self.source_tabs.setCurrentIndex(0)
            data_yaml = output_path / "YOLO_dataset" / "data.yaml"
            if data_yaml.exists():
                self.yolo_picker.setPath(str(data_yaml))
            else:
                data_yaml = output_path / "data.yaml"
                if data_yaml.exists():
                    self.yolo_picker.setPath(str(data_yaml))
                else:
                    self.yolo_picker.setPath(str(output_path))

        elif format_type == "coco":
            self.source_tabs.setCurrentIndex(1)
            self.coco_picker.setPath(str(output_path))

        elif format_type == "jsonl":
            self.source_tabs.setCurrentIndex(0)
            index_file = output_path / "dataset_index.jsonl"
            if index_file.exists():
                self.yolo_picker.setPath(str(index_file))
            else:
                self.yolo_picker.setPath(str(output_path))

    def get_dataset_summary(self) -> Any:
        return getattr(self, "_dataset_summary", None)

    def isComplete(self) -> bool:
        p = self.get_dataset_path().strip()
        if not p:
            return False
        dtype = self.get_dataset_type()
        if dtype == "labelme":
            return Path(p).exists() and Path(p).is_file()
        summary = getattr(self, "_dataset_summary", None)
        if summary is not None:
            return bool(summary.is_valid)
        if dtype == "yolo":
            return Path(p).exists() and (Path(p).is_file() or Path(p).is_dir())
        d = Path(p)
        return (
            d.exists()
            and d.is_dir()
            and (
                (d / "train.json").exists()
                or (d / "annotations" / "train.json").exists()
                or (d / "annotations" / "instances_train.json").exists()
                or (d / "annotations" / "person_keypoints_train.json").exists()
                or (d / "train" / "annotations.json").exists()
            )
        )

    def get_dataset_path(self) -> str:
        dtype = self.get_dataset_type()
        if dtype == "yolo":
            return self.yolo_picker.path()
        if dtype == "labelme":
            return self.labelme_picker.path()
        return self.coco_picker.path()

    def get_dataset_type(self) -> str:
        idx = self.source_tabs.currentIndex()
        if idx == 0:
            return "yolo"
        if idx == 1:
            return "coco"
        return "labelme"


# ---------------------------
# YOLO model selection helpers (delegating to shared catalog)
# ---------------------------


def _yolo_weight_configs() -> List[Any]:
    """Return trainable YOLO configs from MODEL_REGISTRY."""
    return yolo_registry_weight_configs(include_yoloe=False)


def _default_yolo_weight_from_registry() -> str:
    cfgs = _yolo_weight_configs()
    return str(cfgs[0].weight_file) if cfgs else "yolo11n-seg.pt"


def _compute_yolo_weight_name(
    task_text: str, size_text: str, *, family: str = "yolo11"
) -> str:
    return compute_weight_for_selectors(family=family, task=task_text, size=size_text)


def _try_select_registry_weight(weight_file: str, combo: QtWidgets.QComboBox) -> bool:
    """Select preset combo item matching weight_file. Returns True if matched."""
    target = str(weight_file or "").strip().lower()
    if not target:
        return False
    for i in range(combo.count()):
        wf = str(combo.itemData(i) or "").strip().lower()
        if wf == target:
            combo.setCurrentIndex(i)
            return True
    return False


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

        layout.addWidget(
            _header(
                "Training Backend",
                "Pick a training engine. YOLO is the default for most workflows.",
            )
        )

        self.backend_group = QtWidgets.QButtonGroup(self)
        self.backend_cards: List[ClickableCard] = []

        backends = [
            (
                "yolo",
                "YOLO (Ultralytics)",
                "Fast, strong default for detection / segmentation / pose. Great for production + real-time.",
                [
                    "Detection",
                    "Segmentation",
                    "Pose",
                    "Fast training",
                    "Real-time inference",
                ],
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
                backend_id, title, desc, feats, recommended
            )
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

        layout.addWidget(
            _header(
                "Training Settings",
                "Defaults are good. Use presets for quick tuning, or expand advanced options.",
            )
        )

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
            "Optional: choose where to store outputs (runs, checkpoints, logs)"
        )
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
                self._sync_yolo_defaults_from_dataset(
                    wizard.select_dataset_page.get_dataset_summary()
                )
            elif backend == "dino_kpseg":
                self.stack.setCurrentWidget(self.dino_widget)
            else:
                self.stack.setCurrentWidget(self.maskrcnn_widget)

    def _sync_yolo_defaults_from_dataset(self, summary: Any) -> None:
        if summary is None or not getattr(summary, "is_valid", False):
            return
        dataset_key = str(getattr(summary, "source_path", ""))
        if dataset_key and dataset_key == getattr(self, "_synced_yolo_dataset_key", ""):
            return
        self._synced_yolo_dataset_key = dataset_key

        task = str(getattr(summary, "task", "") or "").strip().lower()
        task_label = task_key_to_label(task)
        if task_label and self.yolo_task_combo.currentText() != task_label:
            # Only auto-sync task when user hasn't picked a local checkpoint path.
            current_weight = str(self.yolo_weights_combo.currentData() or "").strip()
            if not (current_weight and Path(current_weight).is_file()):
                self.yolo_task_combo.setCurrentText(task_label)

        train_images = int(getattr(summary, "train_images", 0) or 0)
        if train_images > 0:
            rec = recommend_preset_for_dataset(train_images)
            self._apply_yolo_train_preset(rec.key)

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

        # Preset + Recent trained runs dropdown
        self.yolo_weights_combo = QtWidgets.QComboBox()
        populate_yolo_weight_combo(
            self.yolo_weights_combo,
            default_weight=_default_yolo_weight_from_registry(),
            include_recent_runs=True,
        )

        # Optional browse button for custom weight path
        self.yolo_weights_browse_btn = QtWidgets.QPushButton("Browse…")
        self.yolo_weights_browse_btn.clicked.connect(self._browse_yolo_weights)

        preset_row = QtWidgets.QWidget()
        preset_row_layout = QtWidgets.QHBoxLayout(preset_row)
        preset_row_layout.setContentsMargins(0, 0, 0, 0)
        preset_row_layout.addWidget(self.yolo_weights_combo, 1)
        preset_row_layout.addWidget(self.yolo_weights_browse_btn, 0)

        model_layout.addWidget(QtWidgets.QLabel("Weights / Recent:"), 0, 0)
        model_layout.addWidget(preset_row, 0, 1, 1, 5)

        # Family + Task + Size selectors
        self.yolo_family_combo = QtWidgets.QComboBox()
        populate_yolo_family_combo(self.yolo_family_combo, default="yolo11")
        self.yolo_family_combo.currentIndexChanged.connect(
            self._update_yolo_model_from_task_size
        )

        self.yolo_task_combo = QtWidgets.QComboBox()
        populate_yolo_task_combo(self.yolo_task_combo, default_task="seg")
        self.yolo_task_combo.currentTextChanged.connect(
            self._update_yolo_model_from_task_size
        )

        self.yolo_size_combo = QtWidgets.QComboBox()
        populate_yolo_size_combo(self.yolo_size_combo, default="n")
        self.yolo_size_combo.currentTextChanged.connect(
            self._update_yolo_model_from_task_size
        )

        self.yolo_model_label = QtWidgets.QLabel()
        self.yolo_model_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.yolo_model_label.setStyleSheet("font-weight: 900;")

        self.yolo_resume_check = QtWidgets.QCheckBox(
            "Resume interrupted run (exact epoch & optimizer state from checkpoint)"
        )
        self.yolo_resume_check.setChecked(False)
        self.yolo_resume_check.setEnabled(False)
        self.yolo_resume_check.setToolTip(
            "Enable when continuing an interrupted run from last.pt. "
            "Leave unchecked to fine-tune from a checkpoint on a new schedule or dataset."
        )

        model_layout.addWidget(QtWidgets.QLabel("Family:"), 1, 0)
        model_layout.addWidget(self.yolo_family_combo, 1, 1)
        model_layout.addWidget(QtWidgets.QLabel("Task:"), 1, 2)
        model_layout.addWidget(self.yolo_task_combo, 1, 3)
        model_layout.addWidget(QtWidgets.QLabel("Size:"), 1, 4)
        model_layout.addWidget(self.yolo_size_combo, 1, 5)

        model_layout.addWidget(QtWidgets.QLabel("Selected:"), 2, 0)
        model_layout.addWidget(self.yolo_model_label, 2, 1, 1, 5)
        model_layout.addWidget(self.yolo_resume_check, 3, 0, 1, 6)

        self.yolo_weights_combo.currentIndexChanged.connect(
            self._on_yolo_preset_changed
        )

        layout.addWidget(model_group)

        # Training params
        train_group = QtWidgets.QGroupBox("Core Parameters")
        train_layout = QtWidgets.QFormLayout(train_group)

        self.yolo_train_preset_combo = QtWidgets.QComboBox()
        populate_yolo_preset_combo(
            self.yolo_train_preset_combo, default_key="balanced", include_custom=True
        )
        self.yolo_train_preset_combo.currentIndexChanged.connect(
            self._on_yolo_train_preset_changed
        )
        train_layout.addRow("Fine-tune preset:", self.yolo_train_preset_combo)

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
        advanced_group = QtWidgets.QGroupBox("Advanced Fine-tuning Options")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        advanced_layout = QtWidgets.QFormLayout(advanced_group)

        self.yolo_lr_spin = QtWidgets.QDoubleSpinBox()
        self.yolo_lr_spin.setRange(0.00001, 0.2)
        self.yolo_lr_spin.setDecimals(5)
        self.yolo_lr_spin.setSingleStep(0.001)
        self.yolo_lr_spin.setValue(0.01)
        advanced_layout.addRow("Learning rate (lr0):", self.yolo_lr_spin)

        self.yolo_lrf_spin = QtWidgets.QDoubleSpinBox()
        self.yolo_lrf_spin.setRange(0.0001, 1.0)
        self.yolo_lrf_spin.setDecimals(4)
        self.yolo_lrf_spin.setSingleStep(0.005)
        self.yolo_lrf_spin.setValue(0.01)
        advanced_layout.addRow("Final LR fraction (lrf):", self.yolo_lrf_spin)

        self.yolo_patience_spin = QtWidgets.QSpinBox()
        self.yolo_patience_spin.setRange(0, 500)
        self.yolo_patience_spin.setValue(50)
        advanced_layout.addRow("Early stop patience:", self.yolo_patience_spin)

        self.yolo_freeze_spin = QtWidgets.QSpinBox()
        self.yolo_freeze_spin.setRange(0, 24)
        self.yolo_freeze_spin.setValue(0)
        self.yolo_freeze_spin.setToolTip(
            "Freeze leading backbone layers (e.g. 10) for small lab datasets. 0 trains all layers."
        )
        advanced_layout.addRow("Freeze backbone layers:", self.yolo_freeze_spin)

        self.yolo_close_mosaic_spin = QtWidgets.QSpinBox()
        self.yolo_close_mosaic_spin.setRange(0, 100)
        self.yolo_close_mosaic_spin.setValue(10)
        advanced_layout.addRow(
            "Close mosaic (last N epochs):", self.yolo_close_mosaic_spin
        )

        self.yolo_optimizer_combo = QtWidgets.QComboBox()
        for label, value in (
            ("Auto", "auto"),
            ("SGD", "SGD"),
            ("Adam", "Adam"),
            ("AdamW", "AdamW"),
        ):
            self.yolo_optimizer_combo.addItem(label, userData=value)
        advanced_layout.addRow("Optimizer:", self.yolo_optimizer_combo)

        self.yolo_cos_lr_check = QtWidgets.QCheckBox("Use cosine LR schedule")
        self.yolo_cos_lr_check.setChecked(True)
        advanced_layout.addRow("", self.yolo_cos_lr_check)

        self.yolo_cache_check = QtWidgets.QCheckBox("Cache images in RAM")
        advanced_layout.addRow("", self.yolo_cache_check)

        self.yolo_plots_check = QtWidgets.QCheckBox(
            "Save Ultralytics training plots (PNG curves, confusion matrix)"
        )
        self.yolo_plots_check.setChecked(default_yolo_plots_enabled())
        advanced_layout.addRow("", self.yolo_plots_check)

        layout.addWidget(advanced_group)
        layout.addStretch()

        self._on_yolo_preset_changed()
        self._apply_yolo_train_preset("balanced")

        return widget

    # ---- DINO KPSEG ----

    def _create_dino_options(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setToolTip(
            "Core: backbone/training basics\n"
            "Head Type: choose relational/conv/multitask\n"
            "Loss: mask loss setup\n"
            "Runtime: execution guidance"
        )

        # ---------------- Core tab ----------------
        core_tab = QtWidgets.QWidget()
        core_layout = QtWidgets.QVBoxLayout(core_tab)
        core_layout.setContentsMargins(8, 8, 8, 8)
        core_layout.setSpacing(12)
        core_hint = QtWidgets.QLabel(
            "Set backbone and core optimization settings used in most runs."
        )
        core_hint.setWordWrap(True)
        core_hint.setProperty("muted", True)
        core_layout.addWidget(core_hint)

        model_group = QtWidgets.QGroupBox("DINO Backbone")
        model_layout = QtWidgets.QFormLayout(model_group)

        self.dino_model_combo = QtWidgets.QComboBox()
        configure_dino_model_combo(
            self.dino_model_combo,
            default_identifier=str(dino_defaults.MODEL_NAME),
        )
        model_layout.addRow("Model:", self.dino_model_combo)

        self.dino_short_side_spin = QtWidgets.QSpinBox()
        self.dino_short_side_spin.setRange(224, 2048)
        self.dino_short_side_spin.setSingleStep(32)
        self.dino_short_side_spin.setValue(int(dino_defaults.SHORT_SIDE))
        model_layout.addRow("Short side:", self.dino_short_side_spin)

        self.dino_layers_edit = QtWidgets.QLineEdit()
        self.dino_layers_edit.setPlaceholderText("-1 or -2,-1")
        self.dino_layers_edit.setText(str(dino_defaults.LAYERS))
        model_layout.addRow("Layers:", self.dino_layers_edit)
        core_layout.addWidget(model_group)

        train_group = QtWidgets.QGroupBox("Training Parameters")
        train_layout = QtWidgets.QFormLayout(train_group)

        self.dino_epochs_spin = QtWidgets.QSpinBox()
        self.dino_epochs_spin.setRange(1, 500)
        self.dino_epochs_spin.setValue(210)
        train_layout.addRow("Epochs:", self.dino_epochs_spin)

        self.dino_batch_spin = QtWidgets.QSpinBox()
        self.dino_batch_spin.setRange(1, 64)
        self.dino_batch_spin.setValue(int(dino_defaults.BATCH))
        train_layout.addRow("Batch size:", self.dino_batch_spin)

        self.dino_lr_spin = QtWidgets.QDoubleSpinBox()
        self.dino_lr_spin.setRange(0.000001, 0.05)
        self.dino_lr_spin.setDecimals(6)
        self.dino_lr_spin.setValue(5e-4)
        train_layout.addRow("Learning rate:", self.dino_lr_spin)

        self.dino_freeze_bn_combo = QtWidgets.QComboBox()
        self.dino_freeze_bn_combo.addItem("Auto (<64 images)", userData=None)
        self.dino_freeze_bn_combo.addItem("Enabled", userData=True)
        self.dino_freeze_bn_combo.addItem("Disabled", userData=False)
        train_layout.addRow("Freeze BatchNorm stats:", self.dino_freeze_bn_combo)

        self.dino_radius_spin = QtWidgets.QDoubleSpinBox()
        self.dino_radius_spin.setRange(1.0, 30.0)
        self.dino_radius_spin.setValue(float(dino_defaults.RADIUS_PX))
        train_layout.addRow("Keypoint radius (px):", self.dino_radius_spin)
        core_layout.addWidget(train_group)
        core_layout.addStretch()

        # ---------------- Head tab ----------------
        head_tab = QtWidgets.QWidget()
        head_layout_root = QtWidgets.QVBoxLayout(head_tab)
        head_layout_root.setContentsMargins(8, 8, 8, 8)
        head_layout_root.setSpacing(12)
        head_hint = QtWidgets.QLabel(
            "Pick a head type first. Use Relational for strongest pose modeling, "
            "Conv for speed, Multitask when you need auxiliary object/box/mask heads."
        )
        head_hint.setWordWrap(True)
        head_hint.setProperty("muted", True)
        head_layout_root.addWidget(head_hint)

        head_group = QtWidgets.QGroupBox("Head Type")
        head_layout = QtWidgets.QFormLayout(head_group)
        controls = create_dino_head_loss_controls(
            head_group,
            head_type="conv",
            attn_heads=int(dino_defaults.ATTN_HEADS),
            attn_layers=int(dino_defaults.ATTN_LAYERS),
            hidden_dim=int(dino_defaults.HIDDEN_DIM),
            threshold=float(dino_defaults.THRESHOLD),
            bce_type=str(dino_defaults.BCE_TYPE),
            focal_alpha=float(dino_defaults.FOCAL_ALPHA),
            focal_gamma=float(dino_defaults.FOCAL_GAMMA),
            obj_loss_weight=float(dino_defaults.OBJ_LOSS_WEIGHT),
            box_loss_weight=float(dino_defaults.BOX_LOSS_WEIGHT),
            inst_loss_weight=float(dino_defaults.INST_LOSS_WEIGHT),
            multitask_aux_warmup_epochs=int(dino_defaults.MULTITASK_AUX_WARMUP_EPOCHS),
        )

        self.dino_head_type_combo = controls.head_type_combo
        self.dino_head_type_combo.currentIndexChanged.connect(
            self._on_dino_head_type_changed
        )
        head_layout.addRow("Head type:", self.dino_head_type_combo)
        head_layout_root.addWidget(head_group)

        self.dino_head_detail_tabs = QtWidgets.QTabWidget()
        self.dino_head_detail_tabs.setDocumentMode(True)
        self.dino_head_detail_tabs.setToolTip(
            "Relational: attention-based keypoint reasoning\n"
            "Conv: fastest/simple option\n"
            "Multitask: adds object/box/instance auxiliary losses"
        )

        relational_tab = QtWidgets.QWidget()
        relational_form = QtWidgets.QFormLayout(relational_tab)
        relational_tab.setToolTip(
            "Best default when left-right relationships and long-range context matter."
        )
        self.dino_attn_heads_spin = controls.attn_heads_spin
        relational_form.addRow("Relational heads:", self.dino_attn_heads_spin)
        self.dino_attn_layers_spin = controls.attn_layers_spin
        relational_form.addRow("Relational layers:", self.dino_attn_layers_spin)
        self.dino_head_detail_tabs.addTab(relational_tab, "Relational")

        conv_tab = QtWidgets.QWidget()
        conv_tab.setToolTip(
            "Fastest head for lightweight experiments and quick iteration."
        )
        conv_layout = QtWidgets.QVBoxLayout(conv_tab)
        conv_hint = QtWidgets.QLabel(
            "Conv head is faster and simpler. No extra relational or multitask settings."
        )
        conv_hint.setWordWrap(True)
        conv_hint.setProperty("muted", True)
        conv_layout.addWidget(conv_hint)
        conv_layout.addStretch()
        self.dino_head_detail_tabs.addTab(conv_tab, "Conv")

        multitask_tab = QtWidgets.QWidget()
        multitask_tab.setToolTip(
            "Use when you want auxiliary objectness/box/instance losses."
        )
        multitask_form = QtWidgets.QFormLayout(multitask_tab)
        self.dino_hidden_dim_spin = controls.hidden_dim_spin
        multitask_form.addRow("Hidden dim:", self.dino_hidden_dim_spin)

        self.dino_obj_loss_weight_spin = controls.obj_loss_weight_spin
        multitask_form.addRow("Objectness loss weight:", self.dino_obj_loss_weight_spin)

        self.dino_box_loss_weight_spin = controls.box_loss_weight_spin
        multitask_form.addRow("Box loss weight:", self.dino_box_loss_weight_spin)

        self.dino_inst_loss_weight_spin = controls.inst_loss_weight_spin
        multitask_form.addRow("Instance loss weight:", self.dino_inst_loss_weight_spin)

        self.dino_multitask_aux_warmup_spin = controls.multitask_aux_warmup_spin
        multitask_form.addRow(
            "Multitask aux warmup (epochs):", self.dino_multitask_aux_warmup_spin
        )
        self.dino_head_detail_tabs.addTab(multitask_tab, "Multitask")

        head_layout_root.addWidget(self.dino_head_detail_tabs)
        head_layout_root.addStretch()

        self._dino_multitask_controls = [
            self.dino_obj_loss_weight_spin,
            self.dino_box_loss_weight_spin,
            self.dino_inst_loss_weight_spin,
            self.dino_multitask_aux_warmup_spin,
        ]

        # ---------------- Loss tab ----------------
        loss_tab = QtWidgets.QWidget()
        loss_layout_root = QtWidgets.QVBoxLayout(loss_tab)
        loss_layout_root.setContentsMargins(8, 8, 8, 8)
        loss_layout_root.setSpacing(12)
        loss_hint = QtWidgets.QLabel(
            "Tune mask decision threshold and BCE/focal loss behavior."
        )
        loss_hint.setWordWrap(True)
        loss_hint.setProperty("muted", True)
        loss_layout_root.addWidget(loss_hint)

        advanced_group = QtWidgets.QGroupBox("Advanced")
        advanced_layout = QtWidgets.QFormLayout(advanced_group)

        self.dino_threshold_spin = controls.threshold_spin
        advanced_layout.addRow("Mask threshold:", self.dino_threshold_spin)

        self.dino_bce_type_combo = controls.bce_type_combo
        self.dino_bce_type_combo.currentIndexChanged.connect(
            self._update_dino_focal_controls
        )
        advanced_layout.addRow("Mask BCE type:", self.dino_bce_type_combo)

        self.dino_focal_alpha_spin = controls.focal_alpha_spin
        advanced_layout.addRow("Focal alpha:", self.dino_focal_alpha_spin)

        self.dino_focal_gamma_spin = controls.focal_gamma_spin
        advanced_layout.addRow("Focal gamma:", self.dino_focal_gamma_spin)
        self._dino_focal_controls = list(controls.focal_controls)

        self.dino_cache_features_check = QtWidgets.QCheckBox(
            "Cache frozen DINO features to disk"
        )
        self.dino_cache_features_check.setChecked(True)
        advanced_layout.addRow("", self.dino_cache_features_check)

        loss_layout_root.addWidget(advanced_group)
        loss_layout_root.addStretch()

        # ---------------- Runtime tab ----------------
        runtime_tab = QtWidgets.QWidget()
        runtime_layout = QtWidgets.QVBoxLayout(runtime_tab)
        runtime_layout.setContentsMargins(8, 8, 8, 8)
        runtime_layout.setSpacing(12)
        runtime_hint_top = QtWidgets.QLabel(
            "Wizard keeps runtime options simple; use Train Models dialog for full controls."
        )
        runtime_hint_top.setWordWrap(True)
        runtime_hint_top.setProperty("muted", True)
        runtime_layout.addWidget(runtime_hint_top)

        runtime_group = QtWidgets.QGroupBox("Runtime Guidance")
        runtime_form = QtWidgets.QFormLayout(runtime_group)
        runtime_hint = QtWidgets.QLabel(
            "Use Train Models dialog for full runtime controls:\n"
            "- Auto safe mode\n"
            "- CPU/DataLoader thread tuning\n"
            "- TensorBoard projector options\n"
            "- Post-training report automation"
        )
        runtime_hint.setWordWrap(True)
        runtime_hint.setProperty("muted", True)
        runtime_form.addRow(runtime_hint)
        runtime_layout.addWidget(runtime_group)
        runtime_layout.addStretch()

        tabs.addTab(core_tab, "Core")
        tabs.addTab(head_tab, "Head Type")
        tabs.addTab(loss_tab, "Loss")
        tabs.addTab(runtime_tab, "Runtime")
        layout.addWidget(tabs)

        self._sync_dino_head_detail_tab()
        self._update_dino_head_controls()
        self._update_dino_focal_controls()
        return widget

    def _on_dino_head_type_changed(self) -> None:
        self._sync_dino_head_detail_tab()
        self._update_dino_head_controls()

    def _sync_dino_head_detail_tab(self) -> None:
        tabs = getattr(self, "dino_head_detail_tabs", None)
        combo = getattr(self, "dino_head_type_combo", None)
        if tabs is None or combo is None:
            return
        head_type = str(combo.currentData() or "").strip().lower()
        index_by_type = {
            "relational": 0,
            "conv": 1,
            "multitask": 2,
        }
        idx = index_by_type.get(head_type, 0)
        try:
            tabs.setCurrentIndex(int(idx))
        except Exception:
            pass

    def _update_dino_head_controls(self) -> None:
        head_type = str(self.dino_head_type_combo.currentData() or "").strip().lower()
        relational_controls = [
            self.dino_attn_heads_spin,
            self.dino_attn_layers_spin,
        ]
        apply_dino_head_control_state(
            head_type=head_type,
            relational_controls=relational_controls,
            multitask_controls=list(getattr(self, "_dino_multitask_controls", [])),
        )

    def _update_dino_focal_controls(self) -> None:
        bce_type = str(self.dino_bce_type_combo.currentData() or "bce").strip().lower()
        apply_dino_bce_control_state(
            bce_type=bce_type,
            focal_controls=list(getattr(self, "_dino_focal_controls", [])),
        )

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
        populate_yolo_device_combo(combo)

    def _on_yolo_train_preset_changed(self) -> None:
        key = str(self.yolo_train_preset_combo.currentData() or "").strip().lower()
        if not key or key == "custom":
            return
        self._apply_yolo_train_preset(key)

    def _apply_yolo_train_preset(self, preset_key: str) -> None:
        try:
            preset = get_yolo_train_preset(preset_key)
        except ValueError:
            return
        idx = self.yolo_train_preset_combo.findData(preset.key)
        if idx >= 0 and self.yolo_train_preset_combo.currentIndex() != idx:
            self.yolo_train_preset_combo.blockSignals(True)
            self.yolo_train_preset_combo.setCurrentIndex(idx)
            self.yolo_train_preset_combo.blockSignals(False)

        self.yolo_epochs_spin.setValue(int(preset.epochs))
        self.yolo_batch_spin.setValue(int(preset.batch))
        self.yolo_imgsz_spin.setValue(int(preset.imgsz))
        self.yolo_lr_spin.setValue(float(preset.lr0))
        self.yolo_lrf_spin.setValue(float(preset.lrf))
        self.yolo_patience_spin.setValue(int(preset.patience))
        self.yolo_freeze_spin.setValue(int(preset.freeze))
        self.yolo_close_mosaic_spin.setValue(int(preset.close_mosaic))
        self.yolo_cos_lr_check.setChecked(bool(preset.cos_lr))
        self.yolo_cache_check.setChecked(bool(preset.cache))

        for i in range(self.yolo_optimizer_combo.count()):
            if (
                str(self.yolo_optimizer_combo.itemData(i)).lower()
                == str(preset.optimizer).lower()
            ):
                self.yolo_optimizer_combo.setCurrentIndex(i)
                break

    def _update_yolo_model(self) -> None:
        family = str(self.yolo_family_combo.currentData() or "yolo11")
        task = self.yolo_task_combo.currentText()
        size = self.yolo_size_combo.currentText()
        computed = _compute_yolo_weight_name(task, size, family=family)
        self.yolo_model_label.setText(computed)

    def _on_yolo_preset_changed(self) -> None:
        """Handle weight dropdown change - sync family/task/size combos and resume state."""
        wf = str(self.yolo_weights_combo.currentData() or "").strip()
        if wf:
            parsed = parse_yolo_weight_spec(wf)
            if parsed is not None:
                family, task_label, size = parsed
                self.yolo_family_combo.blockSignals(True)
                fam_idx = self.yolo_family_combo.findData(family)
                if fam_idx >= 0:
                    self.yolo_family_combo.setCurrentIndex(fam_idx)
                self.yolo_family_combo.blockSignals(False)

                self.yolo_task_combo.blockSignals(True)
                self.yolo_task_combo.setCurrentText(task_label)
                self.yolo_task_combo.blockSignals(False)

                self.yolo_size_combo.blockSignals(True)
                self.yolo_size_combo.setCurrentText(size)
                self.yolo_size_combo.blockSignals(False)

        is_local_ckpt = bool(wf and Path(wf).expanduser().is_file())
        self.yolo_resume_check.setEnabled(is_local_ckpt)
        if not is_local_ckpt and self.yolo_resume_check.isChecked():
            self.yolo_resume_check.setChecked(False)
        self._update_yolo_model_label()

    def _update_yolo_model_from_task_size(self) -> None:
        """Update model selection when family/task/size combos change."""
        family = str(self.yolo_family_combo.currentData() or "yolo11")
        task = self.yolo_task_combo.currentText()
        size = self.yolo_size_combo.currentText()
        expected_name = _compute_yolo_weight_name(task, size, family=family)

        self.yolo_weights_combo.blockSignals(True)
        select_or_insert_yolo_weight(self.yolo_weights_combo, expected_name)
        self.yolo_weights_combo.blockSignals(False)
        self.yolo_resume_check.setEnabled(False)
        self.yolo_resume_check.setChecked(False)
        self._update_yolo_model_label()

    def _update_yolo_model_label(self) -> None:
        """Update the model label based on current selection."""
        wf = str(self.yolo_weights_combo.currentData() or "").strip()
        if not wf:
            self.yolo_model_label.setText("Selected weights: (none)")
            return
        display = format_yolo_weight_display(wf)
        self.yolo_model_label.setText(f"Selected weights: {display or wf}")

    def _parse_task_size_from_weight(
        self, weight_file: str
    ) -> tuple[str | None, str | None]:
        parsed = parse_yolo_weight_spec(weight_file)
        if parsed is None:
            return (None, None)
        _family, task_label, size = parsed
        return (task_label, size)

    def _browse_yolo_weights(self) -> None:
        """Browse for custom YOLO weights file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select YOLO Weights", "", "Model Files (*.pt *.pth);;All Files (*)"
        )
        if path:
            select_or_insert_yolo_weight(
                self.yolo_weights_combo,
                path,
                display_label=f"Custom: {Path(path).name}",
            )
            self._on_yolo_preset_changed()

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if folder:
            self.output_dir_edit.setText(folder)

    def get_config(self) -> Dict[str, Any]:
        wizard = self.wizard()
        backend = (
            wizard.select_backend_page.get_backend()
            if isinstance(wizard, TrainingWizard)
            else "yolo"
        )

        config: Dict[str, Any] = {
            "backend": backend,
            "output_dir": self.output_dir_edit.text().strip() or None,
        }

        if backend == "yolo":
            preset_key = str(self.yolo_train_preset_combo.currentData() or "custom")
            config.update(
                {
                    "model": str(self.yolo_weights_combo.currentData() or ""),
                    "preset": self.yolo_train_preset_combo.currentText(),
                    "preset_key": preset_key,
                    "epochs": self.yolo_epochs_spin.value(),
                    "batch": self.yolo_batch_spin.value(),
                    "imgsz": self.yolo_imgsz_spin.value(),
                    "device": self.yolo_device_combo.currentData() or "",
                    "lr0": self.yolo_lr_spin.value(),
                    "lrf": self.yolo_lrf_spin.value(),
                    "patience": self.yolo_patience_spin.value(),
                    "freeze": self.yolo_freeze_spin.value(),
                    "close_mosaic": self.yolo_close_mosaic_spin.value(),
                    "optimizer": str(self.yolo_optimizer_combo.currentData() or "auto"),
                    "cos_lr": self.yolo_cos_lr_check.isChecked(),
                    "cache": self.yolo_cache_check.isChecked(),
                    "plots": self.yolo_plots_check.isChecked(),
                    "resume": self.yolo_resume_check.isChecked(),
                }
            )

        elif backend == "dino_kpseg":
            dataset_type = wizard.select_dataset_page.get_dataset_type()
            if dataset_type == "labelme":
                data_format = "labelme"
            elif dataset_type == "coco":
                data_format = "coco"
            else:
                data_format = "yolo"
            config.update(
                {
                    "model": self.dino_model_combo.currentData(),
                    "epochs": self.dino_epochs_spin.value(),
                    "batch": self.dino_batch_spin.value(),
                    "short_side": self.dino_short_side_spin.value(),
                    "layers": self.dino_layers_edit.text().strip()
                    or str(dino_defaults.LAYERS),
                    "lr": self.dino_lr_spin.value(),
                    "freeze_bn": self.dino_freeze_bn_combo.currentData(),
                    "radius_px": self.dino_radius_spin.value(),
                    "hidden_dim": self.dino_hidden_dim_spin.value(),
                    "head_type": str(self.dino_head_type_combo.currentData() or ""),
                    "attn_heads": self.dino_attn_heads_spin.value(),
                    "attn_layers": self.dino_attn_layers_spin.value(),
                    "threshold": self.dino_threshold_spin.value(),
                    "bce_type": str(self.dino_bce_type_combo.currentData() or "bce"),
                    "focal_alpha": self.dino_focal_alpha_spin.value(),
                    "focal_gamma": self.dino_focal_gamma_spin.value(),
                    "cache_features": self.dino_cache_features_check.isChecked(),
                    "obj_loss_weight": self.dino_obj_loss_weight_spin.value(),
                    "box_loss_weight": self.dino_box_loss_weight_spin.value(),
                    "inst_loss_weight": self.dino_inst_loss_weight_spin.value(),
                    "multitask_aux_warmup_epochs": self.dino_multitask_aux_warmup_spin.value(),
                    "data_format": data_format,
                }
            )

        else:  # maskrcnn
            config.update(
                {
                    "max_iterations": self.maskrcnn_iter_spin.value(),
                    "batch": self.maskrcnn_batch_spin.value(),
                }
            )

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

        layout.addWidget(
            _header("Review & Launch", "Double-check settings, then start training.")
        )

        self.summary_text = QtWidgets.QTextBrowser()
        self.summary_text.setOpenExternalLinks(False)
        self.summary_text.setMinimumHeight(260)
        layout.addWidget(self.summary_text)

        # Quick actions row
        quick = QtWidgets.QHBoxLayout()
        self.copy_btn = QtWidgets.QPushButton("Copy configuration")
        self.copy_btn.clicked.connect(self._copy_summary_text)
        quick.addWidget(self.copy_btn)

        self.colab_btn = QtWidgets.QPushButton("Run on Google Colab…")
        self.colab_btn.setToolTip(
            "Open Google Colab Studio prefilled with this training configuration"
        )
        self.colab_btn.clicked.connect(self._open_in_colab)
        quick.addWidget(self.colab_btn)

        quick.addStretch()
        layout.addLayout(quick)

        options_group = QtWidgets.QGroupBox("Monitoring")
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

        self.time_estimate = QtWidgets.QLabel(
            "Estimated time: varies with dataset size and hardware."
        )
        self.time_estimate.setProperty("muted", True)
        layout.addWidget(self.time_estimate)

        layout.addStretch()

    def _copy_summary_text(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.summary_text.toPlainText())

    def _open_in_colab(self) -> None:
        wizard = self.wizard()
        if isinstance(wizard, TrainingWizard):
            wizard.request_colab_launch()

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
            safe_k = _html.escape(str(k or ""), quote=True)
            safe_v = _html.escape(str(v or ""), quote=True)
            return f"""
            <tr>
              <td style="padding: 6px 10px; color:#A7B3C2; width: 180px; font-weight:700;">{safe_k}</td>
              <td style="padding: 6px 10px; color:#E9EEF5; font-weight:700;">{safe_v}</td>
            </tr>
            """

        ds_name = Path(dataset_path).name if dataset_path else "—"
        if dataset_type == "yolo":
            ds_kind = "YOLO"
        elif dataset_type == "labelme":
            ds_kind = "LabelMe"
        else:
            ds_kind = "COCO"
        bk = backend_names.get(backend, backend)

        params_rows = []
        if backend == "yolo":
            mode_label = (
                "Resume interrupted run"
                if bool(config.get("resume", False))
                else "Fine-tune / Train"
            )
            params_rows += [
                row("Model", str(config.get("model", "yolo11n-seg.pt"))),
                row("Mode", mode_label),
                row("Preset", str(config.get("preset", "Balanced"))),
                row("Epochs", str(config.get("epochs", 100))),
                row("Batch", str(config.get("batch", 8))),
                row("Image size", str(config.get("imgsz", 640))),
                row("Device", str(config.get("device", "auto") or "auto")),
                row("LR (lr0)", str(config.get("lr0", 0.01))),
                row("Patience", str(config.get("patience", 50))),
                row("Freeze layers", str(config.get("freeze", 0))),
                row("Optimizer", str(config.get("optimizer", "auto"))),
                row("Cosine LR", "Yes" if config.get("cos_lr", True) else "No"),
                row("RAM cache", "Yes" if config.get("cache", False) else "No"),
                row("Plots", "Yes" if config.get("plots", False) else "No"),
            ]
        elif backend == "dino_kpseg":
            freeze_bn_val = config.get("freeze_bn", None)
            if freeze_bn_val is None:
                freeze_bn_text = "Auto"
            else:
                freeze_bn_text = "Enabled" if bool(freeze_bn_val) else "Disabled"
            params_rows += [
                row("Model", str(config.get("model", "DINO"))),
                row("Epochs", str(config.get("epochs", 210))),
                row("Batch", str(config.get("batch", 8))),
                row(
                    "Short side",
                    str(config.get("short_side", dino_defaults.SHORT_SIDE)),
                ),
                row("Layers", str(config.get("layers", dino_defaults.LAYERS))),
                row("LR", str(config.get("lr", 5e-4))),
                row("Freeze BN stats", freeze_bn_text),
                row("Radius (px)", str(config.get("radius_px", 6.0))),
                row("Head type", str(config.get("head_type", dino_defaults.HEAD_TYPE))),
                row(
                    "Relational heads",
                    str(config.get("attn_heads", dino_defaults.ATTN_HEADS)),
                ),
                row(
                    "Relational layers",
                    str(config.get("attn_layers", dino_defaults.ATTN_LAYERS)),
                ),
                row("BCE type", str(config.get("bce_type", dino_defaults.BCE_TYPE))),
                row("Dataset format", str(config.get("data_format", "auto"))),
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
              {"".join(params_rows)}
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
    colab_requested = Signal(dict)  # config dict for Google Colab Studio

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        # Theme application is handled at app startup/settings level.
        # The wizard should not apply themes directly.

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
            ds_candidate = Path(dataset_path).expanduser()
            if ds_candidate.is_dir() and not (ds_candidate / "data.yaml").is_file():
                self.select_dataset_page.source_tabs.setCurrentIndex(1)
                self.select_dataset_page.coco_picker.setPath(dataset_path)
            else:
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

    def request_colab_launch(self) -> None:
        config = self.get_training_config()
        self.colab_requested.emit(config)
        self.reject()
