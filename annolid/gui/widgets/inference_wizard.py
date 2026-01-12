"""Inference Wizard - Streamlined inference/tracking workflow.

A QWizard-based interface that guides users through model selection,
video/segment configuration, and inference execution with progress
monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, Signal


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
            ("predictions", "📊 Predictions CSV",
             "Apply existing prediction results"),
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
        self.config_group = QtWidgets.QGroupBox(
            "Configuration File (Optional)")
        config_layout = QtWidgets.QHBoxLayout(self.config_group)
        self.config_edit = QtWidgets.QLineEdit()
        self.config_edit.setPlaceholderText(
            "Select config.yaml (for Detectron2)")
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
            self, "Select Config", "",
            "Config Files (*.yaml *.yml);;All Files (*)"
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
        self.video_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
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
        self.full_video_check.stateChanged.connect(
            self._toggle_segment_options)
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
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if path:
            self.single_video_edit.setText(path)

    def _add_videos(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Videos", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        for f in files:
            existing = [
                self.video_list.item(i).text()
                for i in range(self.video_list.count())
            ]
            if f not in existing:
                self.video_list.addItem(f)
        self.completeChanged.emit()

    def _remove_videos(self) -> None:
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))
        self.completeChanged.emit()

    def _browse_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Video Folder"
        )
        if folder:
            self.folder_edit.setText(folder)

    def _on_folder_changed(self) -> None:
        path = self.folder_edit.text().strip()
        if path and Path(path).is_dir():
            extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            count = sum(
                1 for f in Path(path).iterdir()
                if f.suffix.lower() in extensions
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
                self.video_list.item(i).text()
                for i in range(self.video_list.count())
            ]
        else:  # Folder
            path = self.folder_edit.text().strip()
            if not path or not Path(path).is_dir():
                return []
            extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            return [
                str(f) for f in Path(path).iterdir()
                if f.suffix.lower() in extensions
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

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Detection settings
        detect_group = QtWidgets.QGroupBox("Detection Settings")
        detect_layout = QtWidgets.QFormLayout(detect_group)

        self.score_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.01, 1.0)
        self.score_threshold_spin.setSingleStep(0.05)
        self.score_threshold_spin.setValue(0.25)
        detect_layout.addRow("Confidence threshold:",
                             self.score_threshold_spin)

        self.top_k_spin = QtWidgets.QSpinBox()
        self.top_k_spin.setRange(1, 1000)
        self.top_k_spin.setValue(100)
        detect_layout.addRow("Max detections per frame:", self.top_k_spin)

        layout.addWidget(detect_group)

        # Tracking settings
        track_group = QtWidgets.QGroupBox("Tracking Settings")
        track_layout = QtWidgets.QFormLayout(track_group)

        self.enable_tracking_check = QtWidgets.QCheckBox(
            "Enable object tracking")
        self.enable_tracking_check.setChecked(True)
        track_layout.addRow("", self.enable_tracking_check)

        self.tracker_combo = QtWidgets.QComboBox()
        self.tracker_combo.addItems([
            "ByteTrack",
            "BoT-SORT",
            "OC-SORT",
            "CUTIE (Segmentation)",
        ])
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

        self.save_csv_check = QtWidgets.QCheckBox(
            "Save tracking results as CSV")
        self.save_csv_check.setChecked(True)
        output_layout.addRow("", self.save_csv_check)

        self.save_labelme_check = QtWidgets.QCheckBox(
            "Save as LabelMe annotations")
        self.save_labelme_check.setChecked(True)
        output_layout.addRow("", self.save_labelme_check)

        layout.addWidget(output_group)
        layout.addStretch()

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if folder:
            self.output_dir_edit.setText(folder)

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
        }


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

    def initializePage(self) -> None:
        self._complete = False
        self._log("Initializing inference...")
        self.progress_bar.setValue(0)
        self.results_group.setVisible(False)
        self.open_folder_btn.setVisible(False)

        # Start inference after a short delay
        QtCore.QTimer.singleShot(500, self._start_inference)

    def _start_inference(self) -> None:
        wizard = self.wizard()
        if not isinstance(wizard, InferenceWizard):
            return

        model_type = wizard.select_model_page.get_model_type()
        model_path = wizard.select_model_page.get_model_path()
        videos = wizard.select_videos_page.get_videos()
        config = wizard.configure_page.get_config()

        self._output_path = Path(
            config["output_dir"]) if config["output_dir"] else None

        self._log(f"Model: {Path(model_path).name}")
        self._log(f"Videos: {len(videos)}")
        self._log(f"Threshold: {config['score_threshold']}")
        self._log("")

        # Simulate inference progress (actual implementation would use workers)
        total_videos = len(videos)
        for i, video in enumerate(videos):
            self.current_video_label.setText(f"Processing: {Path(video).name}")
            self._log(f"Processing {Path(video).name}...")

            # Simulate frame-by-frame progress
            for pct in range(0, 101, 10):
                self.progress_bar.setValue(pct)
                self.frame_label.setText(f"Frame {pct}%")
                QtWidgets.QApplication.processEvents()
                QtCore.QThread.msleep(50)  # Simulate processing

            self._log(f"✓ Completed {Path(video).name}")

        self._on_complete(len(videos))

    def _on_complete(self, video_count: int) -> None:
        self._complete = True
        self.current_video_label.setText("✓ Inference complete!")
        self.current_video_label.setStyleSheet(
            "color: green; font-weight: bold;")
        self.progress_bar.setValue(100)
        self.frame_label.setText("")

        self.result_videos.setText(str(video_count))
        self.result_detections.setText("—")  # Would be actual count
        if self._output_path:
            self.result_output.setText(str(self._output_path))

        self.results_group.setVisible(True)
        self.open_folder_btn.setVisible(bool(self._output_path))

        self.completeChanged.emit()

    def _log(self, message: str) -> None:
        self.log_text.append(message)

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
