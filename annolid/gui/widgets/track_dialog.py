from pathlib import Path
from qtpy import QtCore
from qtpy import QtWidgets


class TrackDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        super(TrackDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Track animals and objects")
        self.raidoButtons()
        self.slider()
        self.top_k_slider()
        self.score_threshold = 0.15
        self.algo = 'Detectron2'
        self.config_file = None
        self.out_dir = None
        self.video_file = None
        self.top_k = 100
        self.trained_model = None
        self.raidoButtons()

        qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonbox = QtWidgets.QDialogButtonBox(qbtn)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self.groupBoxModelFiles = QtWidgets.QGroupBox(
            "Please select a trained pth model file")
        self.inputModelFileLineEdit = QtWidgets.QLineEdit(self)
        self.inputModelFileButton = QtWidgets.QPushButton('Open', self)
        self.inputModelFileButton.clicked.connect(
            self.onInputModelFileButtonClicked)
        model_hboxLayOut = QtWidgets.QHBoxLayout()

        model_hboxLayOut.addWidget(self.inputModelFileLineEdit)
        model_hboxLayOut.addWidget(self.inputModelFileButton)
        self.groupBoxModelFiles.setLayout(model_hboxLayOut)

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

        self.label1 = QtWidgets.QLabel(f"Please select class score threshold")
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

        self.label2 = QtWidgets.QLabel(
            "Please select tok k segmentations"
        )

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.groupBox)
        vbox.addWidget(self.groupBoxVideoFiles)
        vbox.addWidget(self.groupBoxModelFiles)
        vbox.addWidget(self.label1)
        vbox.addWidget(self.slider)
        vbox.addWidget(self.groupBoxFiles)
        vbox.addWidget(self.label2)
        vbox.addWidget(self.top_k_slider)
        vbox.addWidget(self.groupBoxOutDir)
        vbox.addWidget(self.buttonbox)

        self.setLayout(vbox)
        self.show()

    def raidoButtons(self):
        self.groupBox = QtWidgets.QGroupBox("Please choose a model type")
        hboxLayOut = QtWidgets.QHBoxLayout()
        self.radio_btn1 = QtWidgets.QRadioButton("Detectron2")
        self.radio_btn1.setChecked(True)
        self.radio_btn1.toggled.connect(self.onRadioButtonChecked)
        hboxLayOut.addWidget(self.radio_btn1)
        self.radio_btn2 = QtWidgets.QRadioButton("YOLACT")
        self.radio_btn2.toggled.connect(self.onRadioButtonChecked)
        self.radio_btn2.setEnabled(True)
        hboxLayOut.addWidget(self.radio_btn2)
        self.groupBox.setLayout(hboxLayOut)

    def onInputModelFileButtonClicked(self):
        self.trained_model, fiter = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open trained model file",
            directory=str(Path()),
            filter='*'
        )
        if self.trained_model is not None:
            self.inputModelFileLineEdit.setText(self.trained_model)

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

    def onRadioButtonChecked(self):
        radio_btn = self.sender()
        if radio_btn.isChecked():
            self.algo = radio_btn.text()

    def slider(self):
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(15)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.onSliderChange)

    def top_k_slider(self):
        self.top_k_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.top_k_slider.setMinimum(0)
        self.top_k_slider.setMaximum(200)
        self.top_k_slider.setValue(20)
        self.top_k_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.top_k_slider.setTickInterval(1)
        self.top_k_slider.valueChanged.connect(self.onTopKSliderChange)

    def onTopKSliderChange(self):
        self.top_k = self.top_k_slider.value()
        self.label2.setText(
            f"You selected top {str(self.top_k)} segmentations"
        )

    def onSliderChange(self):
        self.score_threshold = self.slider.value() / 100
        self.label1.setText(
            f"You selected {str(self.score_threshold)} as score threshold")
