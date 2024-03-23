from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QDoubleSpinBox, QPushButton, QCheckBox


from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QDoubleSpinBox, QCheckBox, QPushButton, QSpinBox)


class AdvancedParametersDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Parameters")

        # Initialize epsilon value, T_max value, and automatic pause flag
        self.epsilon_value = 2.0
        self.t_max_value = 5
        self.automatic_pause_enabled = False

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
        # Close the dialog
        self.close()

    def get_epsilon_value(self):
        return self.epsilon_value

    def get_t_max_value(self):
        return self.t_max_value

    def is_automatic_pause_enabled(self):
        return self.automatic_pause_enabled
