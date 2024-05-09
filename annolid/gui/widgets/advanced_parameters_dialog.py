from qtpy.QtWidgets import (QDialog, QVBoxLayout,
                            QLabel, QDoubleSpinBox,
                            QPushButton, QCheckBox,
                            QSpinBox,
                            )


class AdvancedParametersDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Parameters")

        # Initialize epsilon value, T_max value, automatic pause flag, CPU only flag,
        # save video with color mask flag, and auto recovery of missing instances flag
        self.epsilon_value = 2.0
        self.t_max_value = 5
        self.automatic_pause_enabled = False
        self.cpu_only_enabled = False
        self.save_video_with_color_mask = False
        self.auto_recovery_missing_instances = False
        self.compute_optical_flow = True

        # Create layout
        layout = QVBoxLayout()

        # Add epsilon parameter
        epsilon_label = QLabel(
            "Epsilon Value Selection for Polygon Approximation (default 2.0)")
        self.epsilon_spinbox = QDoubleSpinBox()
        self.epsilon_spinbox.setRange(0.0, 10.0)
        self.epsilon_spinbox.setSingleStep(0.1)
        self.epsilon_spinbox.setValue(self.epsilon_value)
        layout.addWidget(epsilon_label)
        layout.addWidget(self.epsilon_spinbox)

        # Add T_max parameter
        t_max_label = QLabel(
            "T_max Value Selection (5 to 20)")
        self.t_max_spinbox = QSpinBox()
        self.t_max_spinbox.setRange(5, 20)
        self.t_max_spinbox.setSingleStep(1)
        self.t_max_spinbox.setValue(self.t_max_value)
        layout.addWidget(t_max_label)
        layout.addWidget(self.t_max_spinbox)

        # Add checkbox for automatic pause
        self.automatic_pause_checkbox = QCheckBox(
            "Automatic Pause on Error Detection")
        self.automatic_pause_checkbox.setChecked(self.automatic_pause_enabled)
        layout.addWidget(self.automatic_pause_checkbox)

        # Add checkbox for CPU only usage
        self.cpu_only_checkbox = QCheckBox(
            "Use CPU Only")
        self.cpu_only_checkbox.setChecked(self.cpu_only_enabled)
        layout.addWidget(self.cpu_only_checkbox)

        # Add checkbox for saving video with color mask
        self.save_video_with_color_mask_checkbox = QCheckBox(
            "Save Video with Color Mask")
        self.save_video_with_color_mask_checkbox.setChecked(
            self.save_video_with_color_mask)
        layout.addWidget(self.save_video_with_color_mask_checkbox)

        # Add checkbox for auto recovery of missing instances
        self.auto_recovery_missing_instances_checkbox = QCheckBox(
            "Auto Recovery of Missing Instances")
        self.auto_recovery_missing_instances_checkbox.setChecked(
            self.auto_recovery_missing_instances)
        layout.addWidget(self.auto_recovery_missing_instances_checkbox)

        self.compute_optical_flow_checkbox = QCheckBox(
            "Compute Motion Index based on optical flow over instance mask"
        )
        self.compute_optical_flow_checkbox.setChecked(
            self.compute_optical_flow
        )
        layout.addWidget(self.compute_optical_flow_checkbox)
        # Add button to apply parameters
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_parameters)
        layout.addWidget(apply_button)

        self.setLayout(layout)
        self.show()

    def apply_parameters(self):
        # Get epsilon value from spinbox
        self.epsilon_value = self.epsilon_spinbox.value()
        # Get T_max value from spinbox
        self.t_max_value = self.t_max_spinbox.value()
        # Get automatic pause setting
        self.automatic_pause_enabled = self.automatic_pause_checkbox.isChecked()
        # Get CPU only usage setting
        self.cpu_only_enabled = self.cpu_only_checkbox.isChecked()
        # Get save video with color mask setting
        self.save_video_with_color_mask = self.save_video_with_color_mask_checkbox.isChecked()
        # Get auto recovery of missing instances setting
        self.auto_recovery_missing_instances = self.auto_recovery_missing_instances_checkbox.isChecked()
        self.compute_optical_flow = self.compute_optical_flow_checkbox.isChecked()
        # Close the dialog
        self.close()

    def get_epsilon_value(self):
        return self.epsilon_value

    def get_t_max_value(self):
        return self.t_max_value

    def is_automatic_pause_enabled(self):
        return self.automatic_pause_enabled

    def is_cpu_only_enabled(self):
        return self.cpu_only_enabled

    def is_save_video_with_color_mask_enabled(self):
        return self.save_video_with_color_mask

    def is_auto_recovery_missing_instances_enabled(self):
        return self.auto_recovery_missing_instances

    def is_compute_optiocal_flow_enabled(self):
        return self.compute_optical_flow
