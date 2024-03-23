from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QDoubleSpinBox, QPushButton, QCheckBox


class AdvancedParametersDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Parameters")

        # Initialize epsilon value and automatic pause flag
        self.epsilon_value = 2.0
        self.automatic_pause_enabled = False

        # Create layout
        layout = QVBoxLayout()

        # Add epsilon parameter
        epsilon_label = QLabel(
            "Epsilon Value Selection for Polygon Approximation(default 2.0)")
        self.epsilon_spinbox = QDoubleSpinBox()
        self.epsilon_spinbox.setRange(0.0, 10.0)
        self.epsilon_spinbox.setSingleStep(0.1)
        self.epsilon_spinbox.setValue(self.epsilon_value)
        layout.addWidget(epsilon_label)
        layout.addWidget(self.epsilon_spinbox)

        # Add checkbox for automatic pause
        self.automatic_pause_checkbox = QCheckBox(
            "Automatic Pause on Error Detection")
        self.automatic_pause_checkbox.setChecked(self.automatic_pause_enabled)
        layout.addWidget(self.automatic_pause_checkbox)

        # Add button to apply parameters
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_parameters)
        layout.addWidget(apply_button)

        self.setLayout(layout)
        self.show()

    def apply_parameters(self):
        # Get epsilon value from spinbox
        self.epsilon_value = self.epsilon_spinbox.value()
        # Get automatic pause setting
        self.automatic_pause_enabled = self.automatic_pause_checkbox.isChecked()
        # Close the dialog
        self.close()

    def get_epsilon_value(self):
        return self.epsilon_value

    def is_automatic_pause_enabled(self):
        return self.automatic_pause_enabled
