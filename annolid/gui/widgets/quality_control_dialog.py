from pathlib import Path
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class QualityControlDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        super(QualityControlDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Convert tracking results to labelme format")
        self.video_file = None
        self.tracking_results = None
        self.skip_num_frames = 30
        self._syncing_skip_controls = False

        self.slider()

        self.label1 = QtWidgets.QLabel(
            f"Please type or select number of frames to skip default={self.skip_num_frames}"
        )

        self.framesLineEdit = QtWidgets.QLineEdit(self)
        self.framesLineEdit.setText(str(self.skip_num_frames))
        self.framesLineEdit.setValidator(QtGui.QIntValidator(1, 2_147_483_647, self))
        self.framesLineEdit.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.framesLineEdit.textChanged.connect(self.onFramesTextChanged)

        qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonbox = QtWidgets.QDialogButtonBox(qbtn)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self.groupBoxVideoFiles = QtWidgets.QGroupBox("Please choose a video file")
        self.inputVideoFileLineEdit = QtWidgets.QLineEdit(self)
        self.inputVideoFileLineEdit.textChanged.connect(
            lambda txt: setattr(self, "video_file", txt.strip() or None)
        )
        self.inputVideoFileButton = QtWidgets.QPushButton("Open", self)
        self.inputVideoFileButton.clicked.connect(self.onInputVideoFileButtonClicked)
        video_hboxLayOut = QtWidgets.QHBoxLayout()

        video_hboxLayOut.addWidget(self.inputVideoFileLineEdit)
        video_hboxLayOut.addWidget(self.inputVideoFileButton)
        self.groupBoxVideoFiles.setLayout(video_hboxLayOut)

        hboxLayOut = QtWidgets.QHBoxLayout()

        self.groupBoxFiles = QtWidgets.QGroupBox(
            "Please choose tracking results CSV file"
        )
        self.inputFileLineEdit = QtWidgets.QLineEdit(self)
        self.inputFileLineEdit.textChanged.connect(
            lambda txt: setattr(self, "tracking_results", txt.strip() or None)
        )
        self.inputFileButton = QtWidgets.QPushButton("Open", self)
        self.inputFileButton.clicked.connect(self.onInputFileButtonClicked)
        hboxLayOut.addWidget(self.inputFileLineEdit)
        hboxLayOut.addWidget(self.inputFileButton)
        self.groupBoxFiles.setLayout(hboxLayOut)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.groupBoxVideoFiles)
        vbox.addWidget(self.groupBoxFiles)
        vbox.addWidget(self.label1)
        vbox.addWidget(self.framesLineEdit)
        vbox.addWidget(self.slider)
        vbox.addWidget(self.buttonbox)

        self.setLayout(vbox)

    def onInputVideoFileButtonClicked(self):
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self, caption="Open video file", directory=str(Path()), filter="*"
        )
        if selected:
            self.video_file = selected
            self.inputVideoFileLineEdit.setText(selected)

    def onInputFileButtonClicked(self):
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open tracking results CSV file",
            directory=str(Path()),
            filter="*",
        )
        if selected:
            self.tracking_results = selected
            self.inputFileLineEdit.setText(selected)

    def slider(self):
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1000)
        self.slider.setValue(int(self.skip_num_frames))
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.onSliderChange)

    def onSliderChange(self, position):
        if self._syncing_skip_controls:
            return
        self._syncing_skip_controls = True
        try:
            self.skip_num_frames = (
                int(position) if position and str(position).isdigit() else 1
            )
            self.framesLineEdit.setText(str(self.skip_num_frames))
            self.label1.setText(
                f"You have selected to skip {self.skip_num_frames} frames each time."
            )
        finally:
            self._syncing_skip_controls = False

    def onFramesTextChanged(self, text: str) -> None:
        if self._syncing_skip_controls:
            return
        cleaned = str(text or "").strip()
        if not cleaned.isdigit():
            return
        value = max(1, int(cleaned))
        self._syncing_skip_controls = True
        try:
            self.skip_num_frames = value
            if 1 <= value <= self.slider.maximum():
                self.slider.setValue(value)
            self.label1.setText(
                f"You have selected to skip {self.skip_num_frames} frames each time."
            )
        finally:
            self._syncing_skip_controls = False
