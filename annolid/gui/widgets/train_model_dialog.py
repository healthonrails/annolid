from pathlib import Path
from qtpy import QtCore
from qtpy import QtWidgets


class TrainModelDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        super(TrainModelDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Train models")
        self.raidoButtons()
        self.slider()
        self.batch_size = 8
        self.algo = 'MaskRCNN'
        self.config_file = None
        self.out_dir = None

        qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonbox = QtWidgets.QDialogButtonBox(qbtn)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self.label1 = QtWidgets.QLabel(f"Please select batch size")
        self.inputFileLineEdit = QtWidgets.QLineEdit(self)
        self.inputFileButton = QtWidgets.QPushButton('Open', self)
        self.inputFileButton.clicked.connect(self.onInputFileButtonClicked)

        hboxLayOut = QtWidgets.QHBoxLayout()

        self.groupBoxFiles = QtWidgets.QGroupBox("Please choose a config file")
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

        if self.algo == 'MaskRCNN':
            self.label1.hide()
            self.slider.hide()

        vbox.addWidget(self.groupBoxFiles)
        vbox.addWidget(self.groupBoxOutDir)
        vbox.addWidget(self.buttonbox)

        self.setLayout(vbox)
        self.show()

    def onInputFileButtonClicked(self):
        self.config_file, filter = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open config file",
            directory=str(Path()),
            filter='*'

        )
        if self.config_file is not None:
            self.inputFileLineEdit.setText(self.config_file)

    def onOutDirButtonClicked(self):
        self.out_dir = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                  "Select Directory")
        if self.out_dir is not None:
            self.outFileDirEdit.setText(self.out_dir)

    def raidoButtons(self):
        self.groupBox = QtWidgets.QGroupBox("Please choose a model")
        hboxLayOut = QtWidgets.QHBoxLayout()
        self.radio_btn1 = QtWidgets.QRadioButton("MaskRCNN")
        self.radio_btn1.setChecked(True)
        self.radio_btn1.toggled.connect(self.onRadioButtonChecked)
        hboxLayOut.addWidget(self.radio_btn1)
        self.radio_btn2 = QtWidgets.QRadioButton("YOLACT")
        self.radio_btn2.toggled.connect(self.onRadioButtonChecked)
        self.radio_btn2.setEnabled(True)
        hboxLayOut.addWidget(self.radio_btn2)
        self.radio_btn3 = QtWidgets.QRadioButton("YOLOv5")
        self.radio_btn3.toggled.connect(self.onRadioButtonChecked)
        self.radio_btn3.setEnabled(False)
        hboxLayOut.addWidget(self.radio_btn3)
        self.groupBox.setLayout(hboxLayOut)

    def slider(self):
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(128)
        self.slider.setValue(8)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.onSliderChange)

    def onRadioButtonChecked(self):
        radio_btn = self.sender()
        if radio_btn.isChecked():
            self.algo = radio_btn.text()

        if self.algo == 'YOLACT':
            self.label1.show()
            self.slider.show()
        elif self.algo == 'MaskRCNN':
            self.label1.hide()
            self.slider.hide()

    def onSliderChange(self):
        self.batch_size = self.slider.value()
        self.label1.setText(
            f"You selected {str(self.batch_size)} as batch size")
