"""Inference Wizard - Streamlined inference/tracking workflow.

A QWizard-based interface that guides users through model selection,
video/segment configuration, and inference execution with progress
monitoring.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt, Signal

from annolid.utils.logger import logger


class SelectModelPage(QtWidgets.QWizardPage):
    """Page 1: Select the trained model for inference."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Select Model")
        self.setSubTitle(
            "Choose the trained model to use for inference. "
            "You can use YOLO, DINO KPSEG, or Detectron2 models."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Model type selection
        type_group = QtWidgets.QGroupBox("Model Type")
        type_layout = QtWidgets.QVBoxLayout(type_group)

        self.type_group = QtWidgets.QButtonGroup(self)

        types = [
            ("yolo", "🚀 YOLO", "Ultralytics YOLO models (.pt files)"),
            ("dino_kpseg", "🦕 DINO KPSEG", "DINO keypoint/segmentation checkpoints"),
            ("detectron2", "🎭 Detectron2", "Detectron2 models (.pth files)"),
            ("predictions", "📊 Predictions CSV", "Apply existing prediction results"),
        ]

        for i, (type_id, label, description) in enumerate(types):
            row = QtWidgets.QHBoxLayout()
            radio = QtWidgets.QRadioButton(label)
            radio.setObjectName(type_id)
            self.type_group.addButton(radio, i)
            row.addWidget(radio)
            desc_label = QtWidgets.QLabel(description)
            desc_label.setStyleSheet("color: gray;")
            row.addWidget(desc_label)
            row.addStretch()
            type_layout.addLayout(row)

        # Select YOLO by default
        self.type_group.button(0).setChecked(True)
        self.type_group.buttonClicked.connect(self._on_type_changed)

        layout.addWidget(type_group)

        # Model file selection
        file_group = QtWidgets.QGroupBox("Model File")
        file_layout = QtWidgets.QVBoxLayout(file_group)

        model_layout = QtWidgets.QHBoxLayout()
        self.model_edit = QtWidgets.QLineEdit()
        self.model_edit.setPlaceholderText("Select trained model file")
        self.model_edit.textChanged.connect(self._validate_model)
        model_layout.addWidget(self.model_edit, 1)

        self.browse_btn = QtWidgets.QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_model)
        model_layout.addWidget(self.browse_btn)
        file_layout.addLayout(model_layout)

        # Model info
        self.model_info = QtWidgets.QLabel("")
        self.model_info.setStyleSheet("color: gray;")
        file_layout.addWidget(self.model_info)

        layout.addWidget(file_group)

        # Config file (for Detectron2)
        self.config_group = QtWidgets.QGroupBox("Configuration File (Optional)")
        config_layout = QtWidgets.QHBoxLayout(self.config_group)
        self.config_edit = QtWidgets.QLineEdit()
        self.config_edit.setPlaceholderText("Select config.yaml (for Detectron2)")
        config_layout.addWidget(self.config_edit, 1)
        config_browse = QtWidgets.QPushButton("Browse…")
        config_browse.clicked.connect(self._browse_config)
        config_layout.addWidget(config_browse)
        self.config_group.setVisible(False)
        layout.addWidget(self.config_group)

        # Recent models
        recent_group = QtWidgets.QGroupBox("Recent Models")
        recent_layout = QtWidgets.QVBoxLayout(recent_group)
        self.recent_list = QtWidgets.QListWidget()
        self.recent_list.setMaximumHeight(100)
        self.recent_list.itemClicked.connect(self._select_recent)
        recent_layout.addWidget(self.recent_list)
        layout.addWidget(recent_group)

        layout.addStretch()

        self.registerField("modelPath*", self.model_edit)

    def _on_type_changed(self, button: QtWidgets.QAbstractButton) -> None:
        model_type = button.objectName()
        self.config_group.setVisible(model_type == "detectron2")
        self._validate_model()
        self.completeChanged.emit()

    def _browse_model(self) -> None:
        model_type = self.get_model_type()

        if model_type == "yolo":
            filter_str = "YOLO Models (*.pt);;All Files (*)"
        elif model_type == "dino_kpseg":
            filter_str = "Checkpoints (*.pt *.pth *.ckpt);;All Files (*)"
        elif model_type == "detectron2":
            filter_str = "Detectron2 Models (*.pth);;All Files (*)"
        else:
            filter_str = "CSV Files (*.csv);;All Files (*)"

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model", "", filter_str
        )
        if path:
            self.model_edit.setText(path)

    def _browse_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Config", "", "Config Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self.config_edit.setText(path)

    def _validate_model(self) -> None:
        path = self.model_edit.text().strip()
        if not path:
            self.model_info.setText("")
            return

        if Path(path).exists():
            size = Path(path).stat().st_size / (1024 * 1024)
            self.model_info.setText(f"✓ Model found ({size:.1f} MB)")
            self.model_info.setStyleSheet("color: green;")
        else:
            self.model_info.setText("❌ File not found")
            self.model_info.setStyleSheet("color: red;")

        self.completeChanged.emit()

    def _select_recent(self, item: QtWidgets.QListWidgetItem) -> None:
        self.model_edit.setText(item.text())

    def isComplete(self) -> bool:
        path = self.model_edit.text().strip()
        return bool(path and Path(path).exists())

    def get_model_type(self) -> str:
        checked = self.type_group.checkedButton()
        return checked.objectName() if checked else "yolo"

    def get_model_path(self) -> str:
        return self.model_edit.text().strip()

    def get_config_path(self) -> Optional[str]:
        path = self.config_edit.text().strip()
        return path if path else None

    def set_recent_models(self, models: List[str]) -> None:
        self.recent_list.clear()
        for m in models[:5]:
            if Path(m).exists():
                self.recent_list.addItem(m)


class SelectVideosPage(QtWidgets.QWizardPage):
    """Page 2: Select videos to process."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Select Videos")
        self.setSubTitle(
            "Choose the videos to run inference on. You can select multiple "
            "videos or an entire folder."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Input mode selection
        mode_layout = QtWidgets.QHBoxLayout()
        self.mode_group = QtWidgets.QButtonGroup(self)

        self.single_video_radio = QtWidgets.QRadioButton("Single Video")
        self.single_video_radio.setChecked(True)
        self.mode_group.addButton(self.single_video_radio, 0)
        mode_layout.addWidget(self.single_video_radio)

        self.multi_video_radio = QtWidgets.QRadioButton("Multiple Videos")
        self.mode_group.addButton(self.multi_video_radio, 1)
        mode_layout.addWidget(self.multi_video_radio)

        self.folder_radio = QtWidgets.QRadioButton("Folder of Videos")
        self.mode_group.addButton(self.folder_radio, 2)
        mode_layout.addWidget(self.folder_radio)

        mode_layout.addStretch()
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        layout.addLayout(mode_layout)

        # Stack for different input modes
        self.input_stack = QtWidgets.QStackedWidget()

        # Single video input
        single_widget = QtWidgets.QWidget()
        single_layout = QtWidgets.QVBoxLayout(single_widget)
        single_layout.setContentsMargins(0, 0, 0, 0)

        single_file_layout = QtWidgets.QHBoxLayout()
        self.single_video_edit = QtWidgets.QLineEdit()
        self.single_video_edit.setPlaceholderText("Select video file")
        self.single_video_edit.textChanged.connect(self.completeChanged)
        single_file_layout.addWidget(self.single_video_edit, 1)
        browse_single = QtWidgets.QPushButton("Browse…")
        browse_single.clicked.connect(self._browse_single_video)
        single_file_layout.addWidget(browse_single)
        single_layout.addLayout(single_file_layout)

        self.input_stack.addWidget(single_widget)

        # Multiple videos input
        multi_widget = QtWidgets.QWidget()
        multi_layout = QtWidgets.QVBoxLayout(multi_widget)
        multi_layout.setContentsMargins(0, 0, 0, 0)

        self.video_list = QtWidgets.QListWidget()
        self.video_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        multi_layout.addWidget(self.video_list, 1)

        multi_btn_layout = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add Videos…")
        add_btn.clicked.connect(self._add_videos)
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_videos)
        multi_btn_layout.addWidget(add_btn)
        multi_btn_layout.addWidget(remove_btn)
        multi_btn_layout.addStretch()
        multi_layout.addLayout(multi_btn_layout)

        self.input_stack.addWidget(multi_widget)

        # Folder input
        folder_widget = QtWidgets.QWidget()
        folder_layout = QtWidgets.QVBoxLayout(folder_widget)
        folder_layout.setContentsMargins(0, 0, 0, 0)

        folder_file_layout = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("Select folder containing videos")
        self.folder_edit.textChanged.connect(self._on_folder_changed)
        folder_file_layout.addWidget(self.folder_edit, 1)
        browse_folder = QtWidgets.QPushButton("Browse…")
        browse_folder.clicked.connect(self._browse_folder)
        folder_file_layout.addWidget(browse_folder)
        folder_layout.addLayout(folder_file_layout)

        self.folder_info = QtWidgets.QLabel("")
        self.folder_info.setStyleSheet("color: gray;")
        folder_layout.addWidget(self.folder_info)

        self.input_stack.addWidget(folder_widget)

        layout.addWidget(self.input_stack)

        # Segment options
        segment_group = QtWidgets.QGroupBox("Segment Options")
        segment_layout = QtWidgets.QVBoxLayout(segment_group)

        self.full_video_check = QtWidgets.QCheckBox("Process full video(s)")
        self.full_video_check.setChecked(True)
        self.full_video_check.stateChanged.connect(self._toggle_segment_options)
        segment_layout.addWidget(self.full_video_check)

        self.segment_container = QtWidgets.QWidget()
        seg_layout = QtWidgets.QFormLayout(self.segment_container)
        seg_layout.setContentsMargins(20, 0, 0, 0)

        self.start_frame_spin = QtWidgets.QSpinBox()
        self.start_frame_spin.setRange(0, 999999)
        self.start_frame_spin.setValue(0)
        seg_layout.addRow("Start frame:", self.start_frame_spin)

        self.end_frame_spin = QtWidgets.QSpinBox()
        self.end_frame_spin.setRange(0, 999999)
        self.end_frame_spin.setValue(0)
        self.end_frame_spin.setSpecialValueText("End of video")
        seg_layout.addRow("End frame:", self.end_frame_spin)

        self.segment_container.setVisible(False)
        segment_layout.addWidget(self.segment_container)

        layout.addWidget(segment_group)

        layout.addStretch()

    def _on_mode_changed(self, button: QtWidgets.QAbstractButton) -> None:
        idx = self.mode_group.id(button)
        self.input_stack.setCurrentIndex(idx)
        self.completeChanged.emit()

    def _toggle_segment_options(self, state: int) -> None:
        self.segment_container.setVisible(state != Qt.Checked)

    def _browse_single_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        if path:
            self.single_video_edit.setText(path)

    def _add_videos(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Videos",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        for f in files:
            existing = [
                self.video_list.item(i).text() for i in range(self.video_list.count())
            ]
            if f not in existing:
                self.video_list.addItem(f)
        self.completeChanged.emit()

    def _remove_videos(self) -> None:
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))
        self.completeChanged.emit()

    def _browse_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder:
            self.folder_edit.setText(folder)

    def _on_folder_changed(self) -> None:
        path = self.folder_edit.text().strip()
        if path and Path(path).is_dir():
            extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            count = sum(
                1 for f in Path(path).iterdir() if f.suffix.lower() in extensions
            )
            self.folder_info.setText(f"Found {count} video files")
        else:
            self.folder_info.setText("")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        mode = self.mode_group.checkedId()
        if mode == 0:  # Single
            path = self.single_video_edit.text().strip()
            return bool(path and Path(path).exists())
        elif mode == 1:  # Multiple
            return self.video_list.count() > 0
        else:  # Folder
            path = self.folder_edit.text().strip()
            return bool(path and Path(path).is_dir())

    def get_videos(self) -> List[str]:
        mode = self.mode_group.checkedId()
        if mode == 0:  # Single
            path = self.single_video_edit.text().strip()
            return [path] if path else []
        elif mode == 1:  # Multiple
            return [
                self.video_list.item(i).text() for i in range(self.video_list.count())
            ]
        else:  # Folder
            path = self.folder_edit.text().strip()
            if not path or not Path(path).is_dir():
                return []
            extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            return [
                str(f) for f in Path(path).iterdir() if f.suffix.lower() in extensions
            ]

    def is_full_video(self) -> bool:
        return self.full_video_check.isChecked()

    def get_segment(self) -> Optional[tuple]:
        if self.full_video_check.isChecked():
            return None
        return (self.start_frame_spin.value(), self.end_frame_spin.value())

    def set_single_video(self, path: str) -> None:
        self.single_video_radio.setChecked(True)
        self.input_stack.setCurrentIndex(0)
        self.single_video_edit.setText(path)


class ConfigureInferencePage(QtWidgets.QWizardPage):
    """Page 3: Configure inference parameters."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Configure Inference")
        self.setSubTitle(
            "Set the inference parameters for running the model on your videos."
        )

        self._settings = QtCore.QSettings("Annolid", "Annolid")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Detection settings
        detect_group = QtWidgets.QGroupBox("Detection Settings")
        detect_layout = QtWidgets.QFormLayout(detect_group)

        self.score_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.01, 1.0)
        self.score_threshold_spin.setSingleStep(0.05)
        self.score_threshold_spin.setValue(0.25)
        detect_layout.addRow("Confidence threshold:", self.score_threshold_spin)

        self.top_k_spin = QtWidgets.QSpinBox()
        self.top_k_spin.setRange(1, 1000)
        self.top_k_spin.setValue(100)
        detect_layout.addRow("Max detections per frame:", self.top_k_spin)

        layout.addWidget(detect_group)

        # Tracking settings
        track_group = QtWidgets.QGroupBox("Tracking Settings")
        track_layout = QtWidgets.QFormLayout(track_group)

        self.enable_tracking_check = QtWidgets.QCheckBox("Enable object tracking")
        self.enable_tracking_check.setChecked(True)
        track_layout.addRow("", self.enable_tracking_check)

        self.tracker_combo = QtWidgets.QComboBox()
        self.tracker_combo.addItems(
            [
                "ByteTrack",
                "BoT-SORT",
                "OC-SORT",
                "CUTIE (Segmentation)",
            ]
        )
        track_layout.addRow("Tracker:", self.tracker_combo)

        layout.addWidget(track_group)

        # Output settings
        output_group = QtWidgets.QGroupBox("Output Settings")
        output_layout = QtWidgets.QFormLayout(output_group)

        dir_layout = QtWidgets.QHBoxLayout()
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText("Output directory (optional)")
        dir_layout.addWidget(self.output_dir_edit, 1)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output)
        dir_layout.addWidget(browse_btn)
        output_layout.addRow("Output folder:", dir_layout)

        self.save_video_check = QtWidgets.QCheckBox("Save annotated video")
        self.save_video_check.setChecked(False)
        output_layout.addRow("", self.save_video_check)

        self.save_csv_check = QtWidgets.QCheckBox("Save tracking results as CSV")
        self.save_csv_check.setChecked(True)
        output_layout.addRow("", self.save_csv_check)

        self.save_labelme_check = QtWidgets.QCheckBox("Save as LabelMe annotations")
        self.save_labelme_check.setChecked(True)
        output_layout.addRow("", self.save_labelme_check)

        layout.addWidget(output_group)

        pose_group = QtWidgets.QGroupBox("Pose Settings")
        pose_layout = QtWidgets.QFormLayout(pose_group)

        self.save_pose_bbox_check = QtWidgets.QCheckBox(
            "Save pose bounding boxes (YOLO pose)"
        )
        self.save_pose_bbox_check.setChecked(
            self._settings.value("pose/save_bbox", True, type=bool)
        )
        self.save_pose_bbox_check.toggled.connect(self._persist_pose_bbox_setting)
        pose_layout.addRow("", self.save_pose_bbox_check)

        self.show_pose_edges_check = QtWidgets.QCheckBox("Show pose skeleton (edges)")
        self.show_pose_edges_check.setChecked(
            self._settings.value("pose/show_edges", True, type=bool)
        )
        self.show_pose_edges_check.toggled.connect(self._toggle_pose_edges_setting)
        pose_layout.addRow("", self.show_pose_edges_check)

        layout.addWidget(pose_group)

        dino_group = QtWidgets.QGroupBox("DINO KPSEG Settings")
        dino_layout = QtWidgets.QFormLayout(dino_group)
        self.dino_group = dino_group

        self.dino_tta_hflip_check = QtWidgets.QCheckBox("Enable horizontal flip TTA")
        self.dino_tta_hflip_check.setChecked(
            self._settings.value("dino_kpseg/tta_hflip", False, type=bool)
        )
        self.dino_tta_hflip_check.toggled.connect(self._persist_dino_settings)
        dino_layout.addRow("", self.dino_tta_hflip_check)

        self.dino_tta_merge_combo = QtWidgets.QComboBox()
        self.dino_tta_merge_combo.addItems(["mean", "max"])
        dino_merge = (
            str(
                self._settings.value("dino_kpseg/tta_merge", "mean", type=str) or "mean"
            )
            .strip()
            .lower()
        )
        merge_idx = self.dino_tta_merge_combo.findText(
            dino_merge if dino_merge in {"mean", "max"} else "mean"
        )
        self.dino_tta_merge_combo.setCurrentIndex(max(0, merge_idx))
        self.dino_tta_merge_combo.currentIndexChanged.connect(
            self._persist_dino_settings
        )
        dino_layout.addRow("TTA merge:", self.dino_tta_merge_combo)

        self.dino_min_score_spin = QtWidgets.QDoubleSpinBox()
        self.dino_min_score_spin.setRange(0.0, 1.0)
        self.dino_min_score_spin.setSingleStep(0.05)
        self.dino_min_score_spin.setDecimals(3)
        self.dino_min_score_spin.setValue(
            float(
                self._settings.value("dino_kpseg/min_keypoint_score", 0.0, type=float)
            )
        )
        self.dino_min_score_spin.valueChanged.connect(self._persist_dino_settings)
        dino_layout.addRow("Min keypoint score:", self.dino_min_score_spin)

        layout.addWidget(dino_group)
        layout.addStretch()

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if folder:
            self.output_dir_edit.setText(folder)

    def _persist_pose_bbox_setting(self, checked: bool) -> None:
        try:
            self._settings.setValue("pose/save_bbox", bool(checked))
        except Exception:
            pass

        wizard = self.wizard()
        parent = wizard.parent() if wizard is not None else None
        try:
            menu_controller = getattr(parent, "menu_controller", None)
            actions = getattr(menu_controller, "_actions", None)
            if isinstance(actions, dict):
                action = actions.get("toggle_pose_bbox_save")
                if isinstance(action, QtWidgets.QAction):
                    if action.isChecked() != bool(checked):
                        action.blockSignals(True)
                        action.setChecked(bool(checked))
                        action.blockSignals(False)
        except Exception:
            pass

    def _toggle_pose_edges_setting(self, checked: bool) -> None:
        try:
            self._settings.setValue("pose/show_edges", bool(checked))
        except Exception:
            pass

        wizard = self.wizard()
        parent = wizard.parent() if wizard is not None else None
        if parent is not None and hasattr(parent, "toggle_pose_edges_display"):
            try:
                parent.toggle_pose_edges_display(bool(checked))
            except Exception:
                pass

        try:
            menu_controller = getattr(parent, "menu_controller", None)
            actions = getattr(menu_controller, "_actions", None)
            if isinstance(actions, dict):
                action = actions.get("toggle_pose_edges")
                if isinstance(action, QtWidgets.QAction):
                    if action.isChecked() != bool(checked):
                        action.blockSignals(True)
                        action.setChecked(bool(checked))
                        action.blockSignals(False)
        except Exception:
            pass

    def initializePage(self) -> None:
        super().initializePage()
        wizard = self.wizard()
        model_type = ""
        if wizard is not None and hasattr(wizard, "select_model_page"):
            try:
                model_type = str(wizard.select_model_page.get_model_type() or "")
            except Exception:
                model_type = ""
        self.dino_group.setVisible(model_type == "dino_kpseg")

    def _persist_dino_settings(self, *_args: Any) -> None:
        try:
            self._settings.setValue(
                "dino_kpseg/tta_hflip", bool(self.dino_tta_hflip_check.isChecked())
            )
            self._settings.setValue(
                "dino_kpseg/tta_merge", str(self.dino_tta_merge_combo.currentText())
            )
            self._settings.setValue(
                "dino_kpseg/min_keypoint_score",
                float(self.dino_min_score_spin.value()),
            )
        except Exception:
            pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "score_threshold": self.score_threshold_spin.value(),
            "top_k": self.top_k_spin.value(),
            "enable_tracking": self.enable_tracking_check.isChecked(),
            "tracker": self.tracker_combo.currentText(),
            "output_dir": self.output_dir_edit.text().strip() or None,
            "save_video": self.save_video_check.isChecked(),
            "save_csv": self.save_csv_check.isChecked(),
            "save_labelme": self.save_labelme_check.isChecked(),
            "save_pose_bbox": self.save_pose_bbox_check.isChecked(),
            "dino_kpseg_tta_hflip": self.dino_tta_hflip_check.isChecked(),
            "dino_kpseg_tta_merge": str(self.dino_tta_merge_combo.currentText()),
            "dino_kpseg_min_keypoint_score": float(self.dino_min_score_spin.value()),
        }


class InferenceWorker(QtCore.QObject):
    """Run inference jobs in a background thread."""

    progress = Signal(int)
    frame_progress = Signal(int, int)
    log = Signal(str)
    error = Signal(str)
    video_started = Signal(str, int, int)
    video_finished = Signal(str, bool)
    finished = Signal(dict)

    _TRACKER_MAP = {
        "ByteTrack": "bytetrack.yaml",
        "BoT-SORT": "botsort.yaml",
        "OC-SORT": "ocsort.yaml",
    }

    def __init__(
        self, config: Dict[str, Any], parent: Optional[QtCore.QObject] = None
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._stop_event = threading.Event()
        self._processed_videos = 0
        self._processed_frames = 0
        self._errors: List[str] = []
        self._output_dirs: List[Path] = []
        self._processor = None
        self._segmentor = None

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()

    @QtCore.Slot()
    def run(self) -> None:
        videos = [v for v in self._config.get("videos", []) if v]
        total_videos = len(videos)
        if total_videos == 0:
            self.error.emit("No videos selected for inference.")
            self.finished.emit(self._build_summary(total_videos))
            return

        if not self._config.get("save_labelme", True):
            self.log.emit(
                "LabelMe output is required for this workflow; ignoring the toggle."
            )
        if self._config.get("save_video"):
            self.log.emit("Annotated video export is not available in this workflow.")

        for idx, video in enumerate(videos, start=1):
            if self._stop_event.is_set():
                break
            video_path = Path(video)
            if not video_path.exists():
                msg = f"Missing video: {video_path}"
                self.error.emit(msg)
                self._errors.append(msg)
                continue

            self.video_started.emit(str(video_path), idx, total_videos)
            success = False
            try:
                success = self._process_video(
                    video_path=video_path,
                    video_index=idx,
                    total_videos=total_videos,
                )
            except Exception as exc:
                msg = f"{video_path.name}: {exc}"
                logger.error(msg, exc_info=True)
                self.error.emit(msg)
                self._errors.append(msg)

            self.video_finished.emit(str(video_path), success)
            self.progress.emit(int((idx / max(total_videos, 1)) * 100))
            if success:
                self._processed_videos += 1

        self.finished.emit(self._build_summary(total_videos))

    def _build_summary(self, total_videos: int) -> Dict[str, Any]:
        return {
            "total_videos": total_videos,
            "processed_videos": self._processed_videos,
            "processed_frames": self._processed_frames,
            "errors": list(self._errors),
            "output_dirs": [str(p) for p in self._output_dirs],
            "stopped": self._stop_event.is_set(),
        }

    def _process_video(
        self,
        *,
        video_path: Path,
        video_index: int,
        total_videos: int,
    ) -> bool:
        model_type = (self._config.get("model_type") or "").lower()
        if model_type in ("yolo", "dino_kpseg"):
            return self._run_yolo_inference(
                video_path, video_index, total_videos, model_type
            )
        if model_type == "detectron2":
            return self._run_detectron2(video_path)
        if model_type == "predictions":
            return self._run_predictions(video_path, video_index, total_videos)
        raise ValueError(f"Unsupported model type: {model_type}")

    def _resolve_output_dir(
        self, video_path: Path, *, suffix: Optional[str] = None
    ) -> Path:
        output_root = self._config.get("output_dir")
        if output_root:
            output_root = Path(output_root)
            output_root.mkdir(parents=True, exist_ok=True)
            output_dir = output_root / video_path.stem
        else:
            output_dir = video_path.with_suffix("")
        if suffix:
            output_dir = Path(f"{output_dir}{suffix}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _resolve_segment(self, total_frames: int) -> tuple[int, Optional[int]]:
        segment = self._config.get("segment")
        if not segment:
            return 0, total_frames - 1 if total_frames > 0 else None
        start_frame, end_frame = segment
        try:
            start_frame = max(0, int(start_frame))
        except (TypeError, ValueError):
            start_frame = 0
        try:
            end_frame = int(end_frame)
        except (TypeError, ValueError):
            end_frame = None
        if end_frame is not None and end_frame <= 0:
            end_frame = None
        if end_frame is None and total_frames > 0:
            end_frame = total_frames - 1
        if end_frame is not None and end_frame < start_frame:
            end_frame = start_frame
        return start_frame, end_frame

    def _resolve_tracker(self) -> Optional[str]:
        tracker_name = self._config.get("tracker")
        return self._TRACKER_MAP.get(tracker_name, None)

    def _make_progress_callback(self, video_index: int, total_videos: int):
        last_emit = {"time": 0.0}

        def _callback(processed: int, total: int) -> None:
            if total <= 0:
                return
            if self._stop_event.is_set():
                return
            now = time.monotonic()
            if processed < total and now - last_emit["time"] < 0.2:
                return
            last_emit["time"] = now
            overall = ((video_index - 1) + (processed / total)) / max(total_videos, 1)
            self.progress.emit(int(overall * 100))
            self.frame_progress.emit(int(processed), int(total))

        return _callback

    def _get_total_frames(self, video_path: Path) -> int:
        try:
            import cv2
        except Exception:
            return 0
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                return 0
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        finally:
            cap.release()

    def _run_yolo_inference(
        self,
        video_path: Path,
        video_index: int,
        total_videos: int,
        model_type: str,
    ) -> bool:
        from annolid.segmentation.yolos import InferenceProcessor
        from annolid.annotation import labelme2csv

        model_path = self._config.get("model_path")
        if not model_path:
            raise ValueError("Missing model path.")
        dino_cfg = None
        if model_type == "dino_kpseg":
            dino_cfg = {
                "tta_hflip": bool(self._config.get("dino_kpseg_tta_hflip", False)),
                "tta_merge": str(self._config.get("dino_kpseg_tta_merge", "mean")),
                "min_keypoint_score": float(
                    self._config.get("dino_kpseg_min_keypoint_score", 0.0)
                ),
                "stabilize_lr": True,
            }

        if (
            self._processor is None
            or getattr(self._processor, "model_type", "") != model_type
        ):
            resolved_type = "dinokpseg" if model_type == "dino_kpseg" else "yolo"
            self._processor = InferenceProcessor(
                model_name=model_path,
                model_type=resolved_type,
                dino_kpseg_inference_config=dino_cfg,
            )
        elif model_type == "dino_kpseg":
            try:
                self._processor.set_dino_kpseg_inference_config(dino_cfg)
            except Exception:
                pass

        total_frames = self._get_total_frames(video_path)
        start_frame, end_frame = self._resolve_segment(total_frames)
        output_dir = self._resolve_output_dir(video_path)
        progress_callback = self._make_progress_callback(video_index, total_videos)

        self._output_dirs.append(output_dir)
        tracker_name = self._config.get("tracker")
        tracker = self._resolve_tracker()
        if tracker is None and tracker_name and tracker_name not in self._TRACKER_MAP:
            self.log.emit(f"Tracker '{tracker_name}' is not supported; using default.")
        message = self._processor.run_inference(
            source=str(video_path),
            start_frame=start_frame,
            end_frame=end_frame,
            step=1,
            skip_existing=True,
            pred_worker=self,
            stop_event=self._stop_event,
            output_directory=output_dir,
            progress_callback=progress_callback,
            enable_tracking=bool(self._config.get("enable_tracking", True)),
            tracker=tracker,
            save_pose_bbox=self._config.get("save_pose_bbox"),
        )

        frame_count = self._parse_frame_count(message)
        if frame_count is not None:
            self._processed_frames += frame_count

        if isinstance(message, str) and message.startswith("Error:"):
            raise RuntimeError(message)
        if isinstance(message, str) and message.startswith("Stopped"):
            self._stop_event.set()
            return False

        if self._config.get("save_csv"):
            csv_path = output_dir.parent / f"{output_dir.name}_tracking.csv"
            tracked_csv = output_dir.parent / f"{output_dir.name}_tracked.csv"
            self.log.emit(f"Generating CSV: {csv_path.name}")
            labelme2csv.convert_json_to_csv(
                json_folder=str(output_dir),
                csv_file=str(csv_path),
                tracked_csv_file=str(tracked_csv),
                stop_event=self._stop_event,
            )

        return True

    def _run_detectron2(self, video_path: Path) -> bool:
        from annolid.inference.predict import Segmentor

        model_path = self._config.get("model_path")
        if not model_path:
            raise ValueError("Missing Detectron2 model path.")
        config_path = self._config.get("config_path")
        dataset_dir = (
            Path(config_path).parent if config_path else Path(model_path).parent
        )
        output_dir = self._resolve_output_dir(video_path)
        self._output_dirs.append(output_dir)

        if self._segmentor is None:
            self._segmentor = Segmentor(
                dataset_dir=str(dataset_dir),
                model_pth_path=model_path,
                score_threshold=self._config.get("score_threshold", 0.25),
                model_config=config_path,
            )
        self._segmentor.on_video(
            str(video_path),
            skip_frames=1,
            tracking=bool(self._config.get("enable_tracking", True)),
            output_dir=str(output_dir),
        )
        return True

    def _run_predictions(
        self,
        video_path: Path,
        video_index: int,
        total_videos: int,
    ) -> bool:
        from annolid.postprocessing.quality_control import TracksResults

        csv_path = self._config.get("model_path")
        if not csv_path:
            raise ValueError("Missing predictions CSV.")
        if not Path(csv_path).exists():
            raise ValueError(f"Predictions CSV not found: {csv_path}")
        if total_videos > 1 and video_index == 1:
            self.log.emit("Using the same predictions CSV for multiple videos.")

        output_dir = self._resolve_output_dir(
            video_path, suffix="_tracking_results_labelme"
        )
        self._output_dirs.append(output_dir)
        trs = TracksResults(str(video_path), str(csv_path))
        generator = trs.to_labelme_json(str(output_dir))

        for progress, message in generator:
            if self._stop_event.is_set():
                break
            overall = ((video_index - 1) + (progress / 100.0)) / max(total_videos, 1)
            self.progress.emit(int(overall * 100))
            self.frame_progress.emit(int(progress), 100)
            if message:
                self.log.emit(str(message))

        try:
            trs.clean_up()
        except Exception:
            pass
        return not self._stop_event.is_set()

    @staticmethod
    def _parse_frame_count(message: Optional[str]) -> Optional[int]:
        if not message or "#" not in str(message):
            return None
        try:
            _label, count = str(message).split("#", 1)
            return int(count)
        except (ValueError, TypeError):
            return None


class InferenceProgressPage(QtWidgets.QWizardPage):
    """Page 4: Inference progress and results."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Running Inference")
        self.setSubTitle("Processing videos with the selected model...")
        self.setCommitPage(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Current video
        self.current_video_label = QtWidgets.QLabel("Preparing...")
        self.current_video_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.current_video_label)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # Frame progress
        self.frame_label = QtWidgets.QLabel("")
        layout.addWidget(self.frame_label)

        # Log
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)

        # Results summary
        self.results_group = QtWidgets.QGroupBox("Results")
        self.results_group.setVisible(False)
        results_layout = QtWidgets.QFormLayout(self.results_group)

        self.result_videos = QtWidgets.QLabel("")
        results_layout.addRow("Videos processed:", self.result_videos)
        self.result_detections = QtWidgets.QLabel("")
        results_layout.addRow("Total detections:", self.result_detections)
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

        self._complete = False
        self._output_path: Optional[Path] = None
        self._worker: Optional[InferenceWorker] = None
        self._worker_thread: Optional[QtCore.QThread] = None
        self._running = False

    def initializePage(self) -> None:
        self._complete = False
        self.log_text.clear()
        self._log("Initializing inference...")
        self.progress_bar.setValue(0)
        self.results_group.setVisible(False)
        self.open_folder_btn.setVisible(False)
        self.current_video_label.setText("Preparing...")
        self.frame_label.setText("")

        # Start inference after a short delay
        QtCore.QTimer.singleShot(200, self._start_inference)

    def _start_inference(self) -> None:
        wizard = self.wizard()
        if not isinstance(wizard, InferenceWizard):
            return

        config = wizard.get_inference_config()
        model_path = config.get("model_path")
        videos = config.get("videos", [])

        self._output_path = (
            Path(config["output_dir"]) if config.get("output_dir") else None
        )

        if model_path:
            self._log(f"Model: {Path(model_path).name}")
        self._log(f"Videos: {len(videos)}")
        if config.get("score_threshold") is not None:
            self._log(f"Threshold: {config['score_threshold']}")
        if str(config.get("model_type", "")).lower() == "dino_kpseg":
            self._log(
                "DINO KPSEG: "
                f"tta_hflip={bool(config.get('dino_kpseg_tta_hflip', False))}, "
                f"tta_merge={str(config.get('dino_kpseg_tta_merge', 'mean'))}, "
                f"min_score={float(config.get('dino_kpseg_min_keypoint_score', 0.0)):.3f}"
            )
        self._log("")

        self._cleanup_worker()
        self._running = True

        self._worker_thread = QtCore.QThread(self)
        self._worker = InferenceWorker(config)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.frame_progress.connect(self._on_frame_progress)
        self._worker.log.connect(self._log)
        self._worker.error.connect(self._on_error)
        self._worker.video_started.connect(self._on_video_started)
        self._worker.video_finished.connect(self._on_video_finished)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        self._worker_thread.start()

    def _on_complete(self, video_count: int) -> None:
        self._complete = True
        self.current_video_label.setText("✓ Inference complete!")
        self.current_video_label.setStyleSheet("color: green; font-weight: bold;")
        self.progress_bar.setValue(100)
        self.frame_label.setText("")

        self.result_videos.setText(str(video_count))
        self.result_detections.setText("—")  # Would be actual count
        if self._output_path:
            self.result_output.setText(str(self._output_path))

        self.results_group.setVisible(True)
        self.open_folder_btn.setVisible(bool(self._output_path))

        self.completeChanged.emit()

    def _on_worker_finished(self, summary: Dict[str, Any]) -> None:
        self._running = False
        self._complete = True
        processed = summary.get("processed_videos", 0)
        total = summary.get("total_videos", 0)
        stopped = summary.get("stopped", False)

        if stopped:
            self.current_video_label.setText("Stopped")
            self.current_video_label.setStyleSheet("color: #b36b00; font-weight: bold;")
        else:
            self.current_video_label.setText("✓ Inference complete!")
            self.current_video_label.setStyleSheet("color: green; font-weight: bold;")

        self.progress_bar.setValue(100)
        self.frame_label.setText("")
        self.result_videos.setText(f"{processed}/{total}")
        self.result_detections.setText("—")

        output_dirs = summary.get("output_dirs") or []
        if self._output_path:
            self.result_output.setText(str(self._output_path))
            self._output_path = Path(self._output_path)
        elif len(output_dirs) == 1:
            self._output_path = Path(output_dirs[0])
            self.result_output.setText(output_dirs[0])
        elif output_dirs:
            self.result_output.setText("Multiple folders")

        self.results_group.setVisible(True)
        self.open_folder_btn.setVisible(bool(self._output_path))
        self.completeChanged.emit()

    def _log(self, message: str) -> None:
        self.log_text.append(message)

    def _on_error(self, message: str) -> None:
        self._log(f"❌ {message}")

    def _on_video_started(self, video_path: str, idx: int, total: int) -> None:
        self.current_video_label.setText(
            f"Processing: {Path(video_path).name} ({idx}/{total})"
        )
        self.frame_label.setText("")
        self._log(f"Processing {Path(video_path).name}...")

    def _on_video_finished(self, video_path: str, success: bool) -> None:
        status = "✓ Completed" if success else "⚠ Incomplete"
        self._log(f"{status} {Path(video_path).name}")

    def _on_frame_progress(self, current: int, total: int) -> None:
        if total <= 0:
            self.frame_label.setText("")
            return
        self.frame_label.setText(f"Frame {current}/{total}")

    def _cleanup_worker(self) -> None:
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        self._worker = None
        self._worker_thread = None

    def request_stop(self) -> None:
        if self._worker:
            self._log("Stop requested. Finishing current task...")
            self._worker.request_stop()

    def is_running(self) -> bool:
        return self._running

    def _open_output_folder(self) -> None:
        import subprocess
        import sys

        if self._output_path and self._output_path.exists():
            if sys.platform == "darwin":
                subprocess.run(["open", str(self._output_path)])
            elif sys.platform == "win32":
                import os

                os.startfile(str(self._output_path))
            else:
                subprocess.run(["xdg-open", str(self._output_path)])

    def isComplete(self) -> bool:
        return self._complete


class InferenceWizard(QtWidgets.QWizard):
    """Main inference wizard combining all pages."""

    # Signal emitted when inference is requested
    inference_requested = Signal(dict)  # config dict

    def __init__(
        self,
        model_path: Optional[str] = None,
        video_path: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Run Inference")
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)
        self.setMinimumSize(700, 550)

        # Set up pages
        self.select_model_page = SelectModelPage()
        self.select_videos_page = SelectVideosPage()
        self.configure_page = ConfigureInferencePage()
        self.progress_page = InferenceProgressPage()

        self.addPage(self.select_model_page)
        self.addPage(self.select_videos_page)
        self.addPage(self.configure_page)
        self.addPage(self.progress_page)

        # Pre-fill if provided
        if model_path:
            self.select_model_page.model_edit.setText(model_path)
        if video_path:
            self.select_videos_page.set_single_video(video_path)

        # Customize buttons
        self.setButtonText(QtWidgets.QWizard.FinishButton, "Done")
        self.setButtonText(QtWidgets.QWizard.NextButton, "Next →")
        self.setButtonText(QtWidgets.QWizard.BackButton, "← Back")
        self.setButtonText(QtWidgets.QWizard.CommitButton, "▶ Run")

    def get_inference_config(self) -> Dict[str, Any]:
        """Get the complete inference configuration."""
        config = self.configure_page.get_config()
        config["model_type"] = self.select_model_page.get_model_type()
        config["model_path"] = self.select_model_page.get_model_path()
        config["config_path"] = self.select_model_page.get_config_path()
        config["videos"] = self.select_videos_page.get_videos()
        config["full_video"] = self.select_videos_page.is_full_video()
        config["segment"] = self.select_videos_page.get_segment()
        return config

    def reject(self) -> None:
        if self.progress_page.is_running():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Stop Inference",
                "Inference is still running. Stop after the current task?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            self.progress_page.request_stop()
            return
        super().reject()
