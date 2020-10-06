from pathlib import Path
from qtpy import QtCore
from qtpy import QtWidgets


class ExtractFrameDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        super(ExtractFrameDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Extract Frames from a video")
        self.raidoButtons()
        self.slider()
        self.num_frames = 100
        self.algo = 'random'
        self.video_file = None
        self.out_dir = None

        qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonbox = QtWidgets.QDialogButtonBox(qbtn)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self.label1 = QtWidgets.QLabel(f"Please select number of frames")
        self.inputFileLineEdit = QtWidgets.QLineEdit(self)
        self.inputFileButton = QtWidgets.QPushButton('Open', self)
        self.inputFileButton.clicked.connect(self.onInputFileButtonClicked)

        hboxLayOut = QtWidgets.QHBoxLayout()

        self.groupBoxFiles = QtWidgets.QGroupBox("Please choose a video file")
        hboxLayOut.addWidget(self.inputFileLineEdit)
        hboxLayOut.addWidget(self.inputFileButton)
        self.groupBoxFiles.setLayout(hboxLayOut)

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
        vbox.addWidget(self.slider)
        vbox.addWidget(self.groupBoxFiles)
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
        self.slider.setMinimum(1)
        self.slider.setMaximum(1000)
        self.slider.setValue(10)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(5)
        self.slider.valueChanged.connect(self.onSliderChange)

    def onRadioButtonChecked(self):
        radio_btn = self.sender()
        if radio_btn.isChecked():
            self.algo = radio_btn.text()
            if self.algo == 'keyframes':
                self.slider.setDisabled(True)
                self.slider.hide()
                self.label1.setVisible(False)
            else:
                self.slider.setEnabled(True)
                if self.slider.isHidden():
                    self.slider.setVisible(True)
                if self.label1.isHidden():
                    self.label1.setVisible(True)

    def onSliderChange(self):
        self.num_frames = self.slider.value()
        self.label1.setText(
            f"You selected {str(self.num_frames)} frames")
