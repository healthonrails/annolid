from pathlib import Path
from qtpy import QtCore
from qtpy import QtWidgets


class ExtractFrameDialog(QtWidgets.QDialog):
    def __init__(self, video_file=None, *args, **kwargs):
        super(ExtractFrameDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Extract Frames from a video")
        self.raidoButtons()
        self.slider()
        self.num_frames = 100
        self.algo = 'random'
        self.video_file = video_file
        self.out_dir = None
        self.start_sconds = None
        self.end_seconds = None
        self.sub_clip = False

        qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonbox = QtWidgets.QDialogButtonBox(qbtn)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self.label1 = QtWidgets.QLabel(
            f"Please type or select number of frames default={self.num_frames} or use -1 for all frames")
        self.inputFileLineEdit = QtWidgets.QLineEdit(self)
        if self.video_file is not None:
            self.inputFileLineEdit.setText(self.video_file)
        self.framesLineEdit = QtWidgets.QLineEdit(self)
        self.framesLineEdit.setText(str(self.num_frames))
        self.inputFileButton = QtWidgets.QPushButton('Open', self)
        self.inputFileButton.clicked.connect(self.onInputFileButtonClicked)
        self.framesLineEdit.textChanged.connect(self.onSliderChange)

        hboxLayOut = QtWidgets.QHBoxLayout()

        self.groupBoxFiles = QtWidgets.QGroupBox("Please choose a video file")
        hboxLayOut.addWidget(self.inputFileLineEdit)
        hboxLayOut.addWidget(self.inputFileButton)
        self.groupBoxFiles.setLayout(hboxLayOut)

        self.groupBoxSubClip = QtWidgets.QGroupBox(
            "Please type the start seconds and end seconds for the video clip (Optional)")
        self.startSecondsLabel = QtWidgets.QLabel(self)
        self.startSecondsLabel.setText("Start seconds:")
        self.startSecondsLineEdit = QtWidgets.QLineEdit(self)
        self.startSecondsLineEdit.textChanged.connect(
            self.onCutClipStartTimeChanged)
        self.endSecondsLabel = QtWidgets.QLabel(self)
        self.endSecondsLabel.setText("End seconds:")
        self.endSecondsLineEdit = QtWidgets.QLineEdit(self)
        self.endSecondsLineEdit.textChanged.connect(
            self.onCutClipEndTimeChanged)

        hboxLayOutSubClip = QtWidgets.QHBoxLayout()
        hboxLayOutSubClip.addWidget(self.startSecondsLabel)
        hboxLayOutSubClip.addWidget(self.startSecondsLineEdit)
        hboxLayOutSubClip.addWidget(self.endSecondsLabel)
        hboxLayOutSubClip.addWidget(self.endSecondsLineEdit)
        self.groupBoxSubClip.setLayout(hboxLayOutSubClip)

        self.groupBoxOutDir = QtWidgets.QGroupBox(
            "Please choose output directory (Optional)")
        self.outFileDirEdit = QtWidgets.QLineEdit(self)
        self.outDirButton = QtWidgets.QPushButton('Select', self)
        self.outDirButton.clicked.connect(self.onOutDirButtonClicked)
        hboxLayOutDir = QtWidgets.QHBoxLayout()
        hboxLayOutDir.addWidget(self.outFileDirEdit)
        hboxLayOutDir.addWidget(self.outDirButton)
        self.groupBoxOutDir.setLayout(hboxLayOutDir)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.groupBox)
        vbox.addWidget(self.label1)
        vbox.addWidget(self.framesLineEdit)
        vbox.addWidget(self.slider)
        vbox.addWidget(self.groupBoxFiles)
        vbox.addWidget(self.groupBoxSubClip)
        vbox.addWidget(self.groupBoxOutDir)
        vbox.addWidget(self.buttonbox)

        self.setLayout(vbox)
        self.show()

    def onInputFileButtonClicked(self):
        self.video_file, filter = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open video file",
            directory=str(Path()),
            filter='*'

        )
        if self.video_file is not None:
            self.inputFileLineEdit.setText(self.video_file)

    def onOutDirButtonClicked(self):
        self.out_dir = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                  "Select Directory")
        if self.out_dir is not None:
            self.outFileDirEdit.setText(self.out_dir)

    def raidoButtons(self):
        self.groupBox = QtWidgets.QGroupBox("Please choose an algorithm")
        hboxLayOut = QtWidgets.QHBoxLayout()
        self.radio_btn1 = QtWidgets.QRadioButton("random")
        self.radio_btn1.setChecked(True)
        self.radio_btn1.toggled.connect(self.onRadioButtonChecked)
        hboxLayOut.addWidget(self.radio_btn1)
        self.radio_btn2 = QtWidgets.QRadioButton("keyframes")
        self.radio_btn2.toggled.connect(self.onRadioButtonChecked)
        hboxLayOut.addWidget(self.radio_btn2)
        self.radio_btn3 = QtWidgets.QRadioButton("flow")
        self.radio_btn3.toggled.connect(self.onRadioButtonChecked)
        hboxLayOut.addWidget(self.radio_btn3)
        self.groupBox.setLayout(hboxLayOut)

    def slider(self):
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(-1)
        self.slider.setMaximum(1000)
        self.slider.setValue(10)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.onSliderChange)

    def onRadioButtonChecked(self):
        radio_btn = self.sender()
        if radio_btn.isChecked():
            self.algo = radio_btn.text()
            if self.algo == 'keyframes':
                self.slider.setDisabled(True)
                self.slider.hide()
                self.label1.setVisible(False)

                self.framesLineEdit.hide()
                self.framesLineEdit.setVisible(False)
            else:
                self.slider.setEnabled(True)
                if self.slider.isHidden():
                    self.slider.setVisible(True)
                if self.label1.isHidden():
                    self.label1.setVisible(True)
                if self.framesLineEdit.isHidden():
                    self.framesLineEdit.setVisible(True)

    def onSliderChange(self, position):
        self.num_frames = int(position) if position and str(
            position).isdigit() else -1
        self.framesLineEdit.setText(str(position))
        if self.num_frames == -1:
            self.label1.setText(
                f"You have selected to extract all the frames."
            )
        else:
            self.label1.setText(
                f"You have selected {str(self.num_frames)} frames.")

    def onCutClipStartTimeChanged(self):
        self.start_sconds = self.startSecondsLineEdit.text()
        if not self.start_sconds.isdigit():
            QtWidgets.QMessageBox.about(self,
                                        "invalid start seconds",
                                        "Please enter a vaild int number for start seconds")

        if self.start_sconds.isdigit():
            self.start_sconds = int(self.start_sconds)

    def onCutClipEndTimeChanged(self):
        self.end_seconds = self.endSecondsLineEdit.text()

        if not self.end_seconds.isdigit():
            QtWidgets.QMessageBox.about(self,
                                        "invalid end seconds",
                                        "Please enter a vaild int number for end seconds")

        if self.end_seconds.isdigit():
            self.end_seconds = int(self.end_seconds)
