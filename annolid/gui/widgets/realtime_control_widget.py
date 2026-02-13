from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

from qtpy import QtCore, QtWidgets

from annolid.gui.realtime_launch import build_realtime_launch_payload
from annolid.gui.models_registry import MODEL_REGISTRY, ModelConfig
from annolid.realtime.config import Config as RealtimeConfig


class RealtimeControlWidget(QtWidgets.QWidget):
    """
    Compact control panel for configuring and launching realtime inference.
    Emits `start_requested(RealtimeConfig, extras)` and `stop_requested()` so the
    embedding window can manage the worker lifecycle.
    """

    start_requested = QtCore.Signal(object, dict)
    stop_requested = QtCore.Signal()

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        config: Optional[dict] = None,
        label_provider: Optional[Callable[[], Iterable[str]]] = None,
    ):
        super().__init__(parent)
        self._config = config or {}
        self._label_provider = label_provider
        self._custom_model_path: Optional[Path] = None

        self._build_ui()
        self._load_defaults()
        self.set_running(False)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # --- Model Selection -------------------------------------------------
        model_group = QtWidgets.QGroupBox(self.tr("Model"))
        model_layout = QtWidgets.QGridLayout(model_group)
        self.model_combo = QtWidgets.QComboBox()
        for model in MODEL_REGISTRY:
            self.model_combo.addItem(model.display_name, model)
        self.model_combo.addItem(self.tr("Custom…"), "__custom__")
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)

        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_path_edit.setPlaceholderText(
            self.tr(
                "Select or enter a YOLO model file (*.pt, *.onnx, *.engine, *.mlpackage)"
            )
        )
        browse_model_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        browse_model_btn.clicked.connect(self._browse_model_file)

        model_layout.addWidget(QtWidgets.QLabel(self.tr("Preset")), 0, 0)
        model_layout.addWidget(self.model_combo, 0, 1, 1, 2)
        model_layout.addWidget(QtWidgets.QLabel(self.tr("Model Path")), 1, 0)
        model_layout.addWidget(self.model_path_edit, 1, 1)
        model_layout.addWidget(browse_model_btn, 1, 2)
        main_layout.addWidget(model_group)

        # --- Source Configuration -------------------------------------------
        source_group = QtWidgets.QGroupBox(self.tr("Source & Publisher"))
        source_form = QtWidgets.QFormLayout(source_group)

        self.camera_edit = QtWidgets.QLineEdit()
        self.camera_edit.setPlaceholderText(self.tr("e.g. 0, rtsp://, filename"))
        browse_camera_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        browse_camera_btn.clicked.connect(self._browse_camera_source)

        camera_layout = QtWidgets.QHBoxLayout()
        camera_layout.addWidget(self.camera_edit)
        camera_layout.addWidget(browse_camera_btn)

        self.server_edit = QtWidgets.QLineEdit()
        self.server_edit.setPlaceholderText("localhost")

        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(0, 65535)

        self.publisher_edit = QtWidgets.QLineEdit()
        self.publisher_edit.setPlaceholderText("tcp://*:5555")

        self.subscriber_edit = QtWidgets.QLineEdit()
        self.subscriber_edit.setPlaceholderText("tcp://127.0.0.1:5555")

        self.targets_edit = QtWidgets.QLineEdit()
        self.targets_edit.setPlaceholderText(
            self.tr("Comma separated behaviours (leave blank for all)")
        )

        source_form.addRow(self.tr("Camera / Stream"), camera_layout)
        source_form.addRow(self.tr("Remote Server"), self.server_edit)
        source_form.addRow(self.tr("Remote Port"), self.port_spin)
        source_form.addRow(self.tr("Publisher Bind"), self.publisher_edit)
        source_form.addRow(self.tr("Subscriber Address"), self.subscriber_edit)
        source_form.addRow(self.tr("Target Behaviours"), self.targets_edit)
        main_layout.addWidget(source_group)

        # --- Performance / Thresholds ---------------------------------------
        perf_group = QtWidgets.QGroupBox(self.tr("Performance & Thresholds"))
        perf_form = QtWidgets.QFormLayout(perf_group)

        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(160, 4096)
        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(120, 4096)

        self.max_fps_spin = QtWidgets.QDoubleSpinBox()
        self.max_fps_spin.setRange(1.0, 120.0)
        self.max_fps_spin.setDecimals(1)

        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setDecimals(3)
        self.confidence_spin.setSingleStep(0.01)

        perf_form.addRow(self.tr("Frame Width"), self.width_spin)
        perf_form.addRow(self.tr("Frame Height"), self.height_spin)
        perf_form.addRow(self.tr("Max FPS"), self.max_fps_spin)
        perf_form.addRow(self.tr("Confidence"), self.confidence_spin)
        main_layout.addWidget(perf_group)

        # --- Output options --------------------------------------------------
        output_group = QtWidgets.QGroupBox(self.tr("Output Options"))
        output_layout = QtWidgets.QGridLayout(output_group)
        self.publish_frames_check = QtWidgets.QCheckBox(
            self.tr("Publish frames to GUI")
        )
        self.publish_annotated_check = QtWidgets.QCheckBox(
            self.tr("Send annotated frames")
        )
        self.enable_eye_control = QtWidgets.QCheckBox(self.tr("Enable Eye Control"))
        self.enable_hand_control = QtWidgets.QCheckBox(self.tr("Enable Hand Control"))
        self.classify_eye_blinks = QtWidgets.QCheckBox(self.tr("Classify Eye Blinks"))
        self.blink_ear_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.blink_ear_threshold_spin.setRange(0.05, 0.60)
        self.blink_ear_threshold_spin.setDecimals(3)
        self.blink_ear_threshold_spin.setSingleStep(0.005)
        self.blink_min_frames_spin = QtWidgets.QSpinBox()
        self.blink_min_frames_spin.setRange(1, 30)
        self.blink_min_frames_spin.setValue(2)
        self.publish_frames_check.toggled.connect(self._on_publish_frames_toggled)
        self.log_check = QtWidgets.QCheckBox(self.tr("Log detections to NDJSON"))
        self.log_check.toggled.connect(self._on_log_toggled)

        self.log_path_edit = QtWidgets.QLineEdit()
        self.log_path_edit.setPlaceholderText(
            self.tr("Directory or *.ndjson file (optional)")
        )
        self.log_path_edit.setEnabled(False)

        browse_log_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        browse_log_btn.clicked.connect(self._browse_log_path)
        browse_log_btn.setEnabled(False)
        self._log_browse_btn = browse_log_btn

        # Viewer selection
        self.viewer_combo = QtWidgets.QComboBox()
        self.viewer_combo.addItem(self.tr("PyQt Canvas"), "pyqt")
        self.viewer_combo.addItem(self.tr("Three.js Viewer"), "threejs")

        output_layout.addWidget(self.publish_frames_check, 0, 0, 1, 3)
        output_layout.addWidget(self.publish_annotated_check, 1, 0, 1, 3)
        output_layout.addWidget(self.enable_eye_control, 2, 0, 1, 1)
        output_layout.addWidget(self.enable_hand_control, 2, 1, 1, 2)
        output_layout.addWidget(self.classify_eye_blinks, 3, 0, 1, 3)
        output_layout.addWidget(
            QtWidgets.QLabel(self.tr("Blink EAR Threshold")), 4, 0, 1, 1
        )
        output_layout.addWidget(self.blink_ear_threshold_spin, 4, 1, 1, 2)
        output_layout.addWidget(
            QtWidgets.QLabel(self.tr("Blink Min Frames")), 5, 0, 1, 1
        )
        output_layout.addWidget(self.blink_min_frames_spin, 5, 1, 1, 2)
        output_layout.addWidget(self.log_check, 6, 0, 1, 3)
        output_layout.addWidget(self.log_path_edit, 6, 0, 1, 2)
        output_layout.addWidget(browse_log_btn, 6, 2)
        output_layout.addWidget(QtWidgets.QLabel(self.tr("Preferred Viewer")), 7, 0)
        output_layout.addWidget(self.viewer_combo, 7, 1, 1, 2)
        main_layout.addWidget(output_group)

        # --- Buttons / Status ------------------------------------------------
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        self.start_button = QtWidgets.QPushButton(self.tr("Start Realtime"))
        self.stop_button = QtWidgets.QPushButton(self.tr("Stop"))
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self.stop_requested.emit)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        main_layout.addLayout(button_layout)

        self.status_label = QtWidgets.QLabel(self.tr("Idle"))
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)
        main_layout.addStretch()

    # ------------------------------------------------------------------
    # Defaults & helpers
    # ------------------------------------------------------------------
    def _load_defaults(self):
        defaults = (self._config.get("realtime") or {}) if self._config else {}

        initial_model = defaults.get("model_weight") or defaults.get("model")
        if initial_model:
            for index in range(self.model_combo.count()):
                data = self.model_combo.itemData(index)
                if isinstance(data, ModelConfig) and data.weight_file == initial_model:
                    self.model_combo.setCurrentIndex(index)
                    break
            else:
                # fallback to custom
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
                self.model_path_edit.setText(str(initial_model))
        else:
            # select first preset by default
            self.model_combo.setCurrentIndex(0)
            preset = self.model_combo.currentData()
            if isinstance(preset, ModelConfig):
                self.model_path_edit.setText(str(preset.weight_file))

        self.camera_edit.setText(str(defaults.get("camera_index", "0")))
        self.server_edit.setText(str(defaults.get("server_address", "localhost")))
        self.port_spin.setValue(int(defaults.get("server_port", 5002)))
        self.publisher_edit.setText(
            str(defaults.get("publisher_address", "tcp://*:5555"))
        )
        self.subscriber_edit.setText(
            str(defaults.get("subscriber_address", "tcp://127.0.0.1:5555"))
        )

        targets = defaults.get("targets")
        if isinstance(targets, list):
            targets = ", ".join(map(str, targets))
        elif not targets and self._label_provider:
            possible = list(self._label_provider() or [])
            if possible:
                targets = ", ".join(possible)
        self.targets_edit.setText(str(targets or ""))

        self.width_spin.setValue(int(defaults.get("frame_width", 1280)))
        self.height_spin.setValue(int(defaults.get("frame_height", 960)))
        self.max_fps_spin.setValue(float(defaults.get("max_fps", 30.0)))
        self.confidence_spin.setValue(
            float(
                defaults.get("confidence_threshold", defaults.get("confidence", 0.25))
            )
        )
        viewer_type = str(defaults.get("viewer_type", "threejs") or "threejs")
        viewer_index = self.viewer_combo.findData(viewer_type)
        if viewer_index >= 0:
            self.viewer_combo.setCurrentIndex(viewer_index)

        self.publish_frames_check.setChecked(bool(defaults.get("publish_frames", True)))
        self.publish_annotated_check.setChecked(
            bool(
                defaults.get(
                    "publish_annotated_frames", defaults.get("publish_annotated", True)
                )
            )
        )
        self.enable_eye_control.setChecked(
            bool(defaults.get("enable_eye_control", False))
        )
        self.enable_hand_control.setChecked(
            bool(defaults.get("enable_hand_control", False))
        )

        self.log_check.setChecked(bool(defaults.get("log_to_ndjson", False)))
        self.classify_eye_blinks.setChecked(
            bool(defaults.get("classify_eye_blinks", False))
        )
        self.blink_ear_threshold_spin.setValue(
            float(defaults.get("blink_ear_threshold", 0.21))
        )
        self.blink_min_frames_spin.setValue(
            int(defaults.get("blink_min_consecutive_frames", 2))
        )
        log_path = defaults.get("log_path") or defaults.get("ndjson_path") or ""
        if log_path:
            self.log_path_edit.setText(str(log_path))
        self._on_publish_frames_toggled(self.publish_frames_check.isChecked())
        self._update_blink_controls()

    # UI slots
    def _on_model_combo_changed(self, index: int):
        data = self.model_combo.itemData(index)
        if data == "__custom__":
            self._browse_model_file()
            self._update_blink_controls()
            return

        if isinstance(data, ModelConfig):
            self.model_path_edit.setText(str(data.weight_file))
            self._custom_model_path = None
        self._update_blink_controls()

    def _update_blink_controls(self) -> None:
        model_text = str(self.model_path_edit.text() or "").strip().lower()
        is_face_model = (
            "mediapipe_face" in model_text or "face_landmarker" in model_text
        )
        self.classify_eye_blinks.setEnabled(is_face_model)
        self.blink_ear_threshold_spin.setEnabled(is_face_model)
        self.blink_min_frames_spin.setEnabled(is_face_model)
        if not is_face_model:
            self.classify_eye_blinks.setChecked(False)

    def _browse_model_file(self):
        last_path = self.model_path_edit.text().strip() or str(Path.home())
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select YOLO Model"),
            last_path,
            self.tr("YOLO Models (*.pt *.onnx *.engine *.mlpackage);;All Files (*.*)"),
        )
        if selected:
            self._custom_model_path = Path(selected)
            self.model_path_edit.setText(selected)
            custom_index = self.model_combo.count() - 1
            if self.model_combo.currentIndex() != custom_index:
                self.model_combo.setCurrentIndex(custom_index)

    def _browse_camera_source(self):
        last_path = self.camera_edit.text().strip() or str(Path.home())
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Video File"),
            last_path,
            self.tr("Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*.*)"),
        )
        if selected:
            self.camera_edit.setText(selected)

    def _on_log_toggled(self, checked: bool):
        self.log_path_edit.setEnabled(checked)
        self._log_browse_btn.setEnabled(checked)

    def _browse_log_path(self):
        initial = self.log_path_edit.text().strip() or str(Path.home())
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select log directory"), initial
        )
        if directory:
            self.log_path_edit.setText(directory)

    def _on_publish_frames_toggled(self, checked: bool):
        self.publish_annotated_check.setEnabled(checked)
        if not checked:
            self.publish_annotated_check.setChecked(False)

    # ------------------------------------------------------------------
    # State exposed to parent
    # ------------------------------------------------------------------
    def set_running(self, running: bool):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def set_status_text(self, text: str):
        self.status_label.setText(text)

    def set_stopping(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_start_clicked(self):
        try:
            config, extras = self._build_runtime_config()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self, self.tr("Invalid Realtime Configuration"), str(exc)
            )
            return

        self.set_running(True)
        self.set_status_text(self.tr("Starting realtime inference…"))
        self.start_requested.emit(config, extras)

    def _build_runtime_config(self) -> Tuple[RealtimeConfig, dict]:
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            raise ValueError(self.tr("Please select a YOLO model file."))

        publisher = self.publisher_edit.text().strip() or "tcp://*:5555"
        subscriber = self.subscriber_edit.text().strip() or "tcp://127.0.0.1:5555"

        targets_text = self.targets_edit.text().strip()
        return build_realtime_launch_payload(
            camera_source=self.camera_edit.text().strip(),
            model_name=model_path,
            target_behaviors_csv=targets_text,
            confidence_threshold=float(self.confidence_spin.value()),
            viewer_type=str(self.viewer_combo.currentData() or "threejs"),
            enable_eye_control=self.enable_eye_control.isChecked(),
            enable_hand_control=self.enable_hand_control.isChecked(),
            classify_eye_blinks=self.classify_eye_blinks.isChecked(),
            blink_ear_threshold=float(self.blink_ear_threshold_spin.value()),
            blink_min_consecutive_frames=int(self.blink_min_frames_spin.value()),
            subscriber_address=subscriber,
            log_enabled=self.log_check.isChecked(),
            log_path=self.log_path_edit.text().strip(),
            server_address=self.server_edit.text().strip() or "localhost",
            server_port=int(self.port_spin.value()),
            publisher_address=publisher,
            frame_width=int(self.width_spin.value()),
            frame_height=int(self.height_spin.value()),
            max_fps=float(self.max_fps_spin.value()),
            publish_frames=self.publish_frames_check.isChecked(),
            publish_annotated_frames=self.publish_annotated_check.isChecked(),
        )
