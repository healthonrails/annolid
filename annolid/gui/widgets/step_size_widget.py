from qtpy import QtCore
from qtpy import QtWidgets


class _AutoResizePushButton(QtWidgets.QPushButton):
    def __init__(self, *args, on_changed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_changed = on_changed

    def setText(self, text):  # type: ignore[override]
        super().setText(text)
        if callable(self._on_changed):
            self._on_changed()

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            if callable(self._on_changed):
                self._on_changed()


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
        self.predict_button = _AutoResizePushButton(
            self.tr("Pred"), on_changed=self._sync_control_widths
        )
        self.predict_button.setStyleSheet(
            "background-color: green; color: white; padding: 2px 2px;"
        )
        self.predict_button.setToolTip(self.tr("Run prediction on the next frame"))
        self.predict_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.predict_button.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )

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
        self._sync_control_widths()

    def _sync_control_widths(self) -> None:
        """
        Ensure the embedded controls have enough width for their current text.

        This widget is embedded into a QToolBar via QWidgetAction; the toolbar may
        reuse the widget's size hint, so we keep the width in sync when the
        predict button label changes (e.g. Pred/Stop/Stopping...).
        """
        widths = [
            int(self.occclusion_checkbox.sizeHint().width()),
            int(self.step_size_spin_box.sizeHint().width()),
            int(self.predict_button.sizeHint().width()),
        ]
        target = max(60, *widths)

        self.occclusion_checkbox.setMinimumWidth(target)
        self.step_size_spin_box.setFixedWidth(target)
        self.predict_button.setMinimumWidth(target)
        self.setMinimumWidth(target)
        self.updateGeometry()

    def set_value(self, value):
        self.step_size_spin_box.setValue(value)

    def emit_value_changed(self, value):
        self.valueChanged.emit(value)

    def minimumSizeHint(self):
        base = super(StepSizeWidget, self).minimumSizeHint()
        if self.layout() is None:
            return base
        hint = self.layout().sizeHint()
        return QtCore.QSize(
            max(base.width(), hint.width()), max(base.height(), hint.height())
        )
