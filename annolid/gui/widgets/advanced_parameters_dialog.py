from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QDoubleSpinBox, QPushButton


class AdvancedParametersDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Parameters")

        # Initialize epsilon value
        self.epsilon_value = 2.0

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

        # Add button to apply parameters
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_parameters)
        layout.addWidget(apply_button)

        self.setLayout(layout)
        self.show()

    def apply_parameters(self):
        # Get epsilon value from spinbox
        self.epsilon_value = self.epsilon_spinbox.value()
        # Close the dialog
        self.close()

    def get_epsilon_value(self):
        return self.epsilon_value
