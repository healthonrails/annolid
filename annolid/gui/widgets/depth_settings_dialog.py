from qtpy import QtWidgets


class DepthSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Video Depth Anything Settings"))
        self.setModal(True)
        layout = QtWidgets.QFormLayout(self)

        self.encoder_combo = QtWidgets.QComboBox()
        self.encoder_combo.addItems(["vitl", "vitb", "vits"])
        layout.addRow(self.tr("Encoder"), self.encoder_combo)

        self.input_size_spin = QtWidgets.QSpinBox()
        self.input_size_spin.setRange(256, 1024)
        self.input_size_spin.setSingleStep(2)
        layout.addRow(self.tr("Input size"), self.input_size_spin)

        self.max_res_spin = QtWidgets.QSpinBox()
        self.max_res_spin.setRange(512, 2048)
        self.max_res_spin.setSingleStep(16)
        layout.addRow(self.tr("Max resolution"), self.max_res_spin)

        self.max_len_spin = QtWidgets.QSpinBox()
        self.max_len_spin.setRange(-1, 20000)
        self.max_len_spin.setSingleStep(1)
        layout.addRow(self.tr("Max frames (-1 = unlimited)"), self.max_len_spin)

        self.target_fps_spin = QtWidgets.QSpinBox()
        self.target_fps_spin.setRange(-1, 240)
        layout.addRow(self.tr("Target FPS (-1 = original)"), self.target_fps_spin)

        self.metric_chk = QtWidgets.QCheckBox(self.tr("Metric depth model"))
        self.fp32_chk = QtWidgets.QCheckBox(self.tr("Infer with FP32"))
        self.grayscale_chk = QtWidgets.QCheckBox(self.tr("Grayscale overlay"))
        self.save_video_chk = QtWidgets.QCheckBox(self.tr("Save depth video"))
        self.save_frames_chk = QtWidgets.QCheckBox(self.tr("Save depth frames"))
        self.save_point_cloud_chk = QtWidgets.QCheckBox(
            self.tr("Save point cloud CSVs")
        )
        self.streaming_chk = QtWidgets.QCheckBox(self.tr("Streaming mode"))
        self.streaming_chk.setChecked(True)

        flag_layout = QtWidgets.QVBoxLayout()
        flag_layout.addWidget(self.metric_chk)
        flag_layout.addWidget(self.fp32_chk)
        flag_layout.addWidget(self.grayscale_chk)
        flag_layout.addWidget(self.save_video_chk)
        flag_layout.addWidget(self.save_frames_chk)
        flag_layout.addWidget(self.save_point_cloud_chk)
        flag_layout.addWidget(self.streaming_chk)
        layout.addRow(self.tr("Options"), flag_layout)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self._apply_config(config or {})

    def _apply_config(self, cfg):
        self.encoder_combo.setCurrentText(cfg.get("encoder", "vits"))
        self.input_size_spin.setValue(cfg.get("input_size", 518))
        self.max_res_spin.setValue(cfg.get("max_res", 1280))
        self.max_len_spin.setValue(cfg.get("max_len", -1))
        self.target_fps_spin.setValue(cfg.get("target_fps", -1))
        self.metric_chk.setChecked(cfg.get("metric", False))
        self.fp32_chk.setChecked(cfg.get("fp32", False))
        self.grayscale_chk.setChecked(cfg.get("grayscale", False))
        self.save_video_chk.setChecked(cfg.get("save_depth_video", False))
        self.save_frames_chk.setChecked(cfg.get("save_depth_frames", False))
        self.save_point_cloud_chk.setChecked(
            cfg.get("save_point_clouds", False)
        )
        self.streaming_chk.setChecked(cfg.get("streaming", True))

    def values(self):
        return {
            "encoder": self.encoder_combo.currentText(),
            "input_size": self.input_size_spin.value(),
            "max_res": self.max_res_spin.value(),
            "max_len": self.max_len_spin.value(),
            "target_fps": self.target_fps_spin.value(),
            "metric": self.metric_chk.isChecked(),
            "fp32": self.fp32_chk.isChecked(),
            "grayscale": self.grayscale_chk.isChecked(),
            "save_depth_video": self.save_video_chk.isChecked(),
            "save_depth_frames": self.save_frames_chk.isChecked(),
            "save_point_clouds": self.save_point_cloud_chk.isChecked(),
            "streaming": self.streaming_chk.isChecked(),
        }
