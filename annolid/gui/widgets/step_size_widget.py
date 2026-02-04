from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class StepSizeWidget(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int)

    def __init__(self, value=1):
        super(StepSizeWidget, self).__init__()

        # Step Size Spin Box
        self.step_size_spin_box = QtWidgets.QSpinBox()
        self.step_size_spin_box.setRange(-1000, 1000)
        self.set_value(value)
        self.step_size_spin_box.setToolTip("Video Step Size")
        self.step_size_spin_box.setStatusTip(self.step_size_spin_box.toolTip())
        self.step_size_spin_box.setAlignment(QtCore.Qt.AlignCenter)
        self.step_size_spin_box.setFixedWidth(60)
        self.step_size_spin_box.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Checkbox for indicating occlusion
        self.occclusion_checkbox = QtWidgets.QCheckBox(self.tr("Auto"))
        self.occclusion_checkbox.setChecked(True)
        self.occclusion_checkbox.setToolTip(
            self.tr("Enable automatic occlusion handling")
        )
        self.occclusion_checkbox.setFocusPolicy(QtCore.Qt.NoFocus)

        # Predict Button
        self.predict_button = QtWidgets.QPushButton(self.tr("Pred"))
        self.predict_button.setStyleSheet("background-color: green; color: white;")
        self.predict_button.setToolTip(self.tr("Run prediction on the next frame"))
        self.predict_button.setFocusPolicy(QtCore.Qt.NoFocus)

        # Connect valueChanged signal of QSpinBox to self.valueChanged
        self.step_size_spin_box.valueChanged.connect(self.emit_value_changed)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.setAlignment(QtCore.Qt.AlignHCenter)
        layout.addWidget(self.occclusion_checkbox, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.step_size_spin_box, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.predict_button, alignment=QtCore.Qt.AlignHCenter)
        self.setLayout(layout)
        self.predict_button.setFixedWidth(self.step_size_spin_box.sizeHint().width())

    def set_value(self, value):
        self.step_size_spin_box.setValue(value)

    def emit_value_changed(self, value):
        self.valueChanged.emit(value)

    def minimumSizeHint(self):
        height = super(StepSizeWidget, self).minimumSizeHint().height()
        fm = QtGui.QFontMetrics(self.font())
        width = fm.width(str(self.step_size_spin_box.maximum()))
        return QtCore.QSize(width, height)
