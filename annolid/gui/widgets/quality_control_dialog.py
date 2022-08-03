from pathlib import Path
from qtpy import QtCore
from qtpy import QtWidgets


class QualityControlDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        super(QualityControlDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Convert tracking results to labelme format")
        self.video_file = None
        self.tracking_results = None
        self.skip_num_frames = 30

        self.slider()

        self.label1 = QtWidgets.QLabel(
            f"Please type or select number of frames to skip default={self.skip_num_frames}")

        self.framesLineEdit = QtWidgets.QLineEdit(self)
        self.framesLineEdit.setText(str(self.skip_num_frames))

        qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonbox = QtWidgets.QDialogButtonBox(qbtn)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self.groupBoxVideoFiles = QtWidgets.QGroupBox(
            "Please choose a video file")
        self.inputVideoFileLineEdit = QtWidgets.QLineEdit(self)
        self.inputVideoFileButton = QtWidgets.QPushButton('Open', self)
        self.inputVideoFileButton.clicked.connect(
            self.onInputVideoFileButtonClicked)
        video_hboxLayOut = QtWidgets.QHBoxLayout()

        video_hboxLayOut.addWidget(self.inputVideoFileLineEdit)
        video_hboxLayOut.addWidget(self.inputVideoFileButton)
        self.groupBoxVideoFiles.setLayout(video_hboxLayOut)

        hboxLayOut = QtWidgets.QHBoxLayout()

        self.groupBoxFiles = QtWidgets.QGroupBox(
            "Please choose tracking results CSV file")
        self.inputFileLineEdit = QtWidgets.QLineEdit(self)
        self.inputFileButton = QtWidgets.QPushButton('Open', self)
        self.inputFileButton.clicked.connect(
            self.onInputFileButtonClicked)
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
        self.show()

    def onInputVideoFileButtonClicked(self):
        self.video_file, filter = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open video file",
            directory=str(Path()),
            filter='*'

        )
        if self.video_file is not None:
            self.inputVideoFileLineEdit.setText(self.video_file)

    def onInputFileButtonClicked(self):
        self.tracking_results, filter = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open tracking results CSV file",
            directory=str(Path()),
            filter='*'

        )
        if self.tracking_results is not None:
            self.inputFileLineEdit.setText(self.tracking_results)

    def onInputFileButtonClicked(self):
        self.tracking_results, filter = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open tracking results CSV file",
            directory=str(Path()),
            filter='*'

        )
        if self.tracking_results is not None:
            self.inputFileLineEdit.setText(self.tracking_results)

    def slider(self):
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1000)
        self.slider.setValue(10)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.onSliderChange)

    def onSliderChange(self, position):
        self.skip_num_frames = int(position) if position and str(
            position).isdigit() else 1
        self.framesLineEdit.setText(str(position))

        self.label1.setText(
            f"You have selected to skip {str(self.skip_num_frames)} frames each time.")
