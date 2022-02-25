import glob
import os.path as osp
from pathlib import Path
from qtpy import QtCore
from qtpy import QtWidgets


class ConvertCOODialog(QtWidgets.QDialog):
    def __init__(self, annotation_dir=None, *args, **kwargs):
        super(ConvertCOODialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Convert to COCO format datasets ")
        self.annotation_dir = annotation_dir
        self.out_dir = None
        self.slider()
        self.label_list_text = None

        self.num_train_frames = 100

        qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonbox = QtWidgets.QDialogButtonBox(qbtn)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self.label1 = QtWidgets.QLabel(
            f"Please type or select number of frames for training default={self.num_train_frames}")

        self.trainFramesLineEdit = QtWidgets.QLineEdit(self)
        self.trainFramesLineEdit.setText(str(self.num_train_frames))
        self.trainFramesLineEdit.textChanged.connect(self.onSliderChange)

        hboxLayOut = QtWidgets.QHBoxLayout()
        vbox = QtWidgets.QVBoxLayout()

        self.groupBoxFiles = QtWidgets.QGroupBox(
            f"Please select annotation directory")
        self.annoFileLineEdit = QtWidgets.QLineEdit(self)

        if self.annotation_dir is not None:
            self.annoFileLineEdit.setText(str(self.annotation_dir))

        self.annoFileButton = QtWidgets.QPushButton(
            'Open Annotation Directory', self)
        self.annoFileButton.clicked.connect(
            self.onOutAnnoDirButtonClicked)
        hboxLayOut.addWidget(self.annoFileLineEdit)
        hboxLayOut.addWidget(self.annoFileButton)
        self.groupBoxFiles.setLayout(hboxLayOut)

        hboxLabelLayOut = QtWidgets.QVBoxLayout()

        self.groupBoxLabelFiles = QtWidgets.QGroupBox(
            "Please choose a label text file")
        self.inputLabelFileLineEdit = QtWidgets.QLineEdit(self)
        self.inputLabelFileButton = QtWidgets.QPushButton(
            'Open Labels File', self)
        self.inputLabelFileButton.clicked.connect(
            self.onInputFileButtonClicked)
        hboxLabelLayOut.addWidget(self.inputLabelFileLineEdit)
        hboxLabelLayOut.addWidget(self.inputLabelFileButton)
        self.groupBoxLabelFiles.setLayout(hboxLabelLayOut)

        self.groupBoxOutDir = QtWidgets.QGroupBox(
            "Please choose output directory (Optional)")
        self.outFileDirEdit = QtWidgets.QLineEdit(self)
        self.outDirButton = QtWidgets.QPushButton(
            'Select Output Directory', self)
        self.outDirButton.clicked.connect(self.onOutDirButtonClicked)
        hboxLayOutDir = QtWidgets.QHBoxLayout()
        hboxLayOutDir.addWidget(self.outFileDirEdit)
        hboxLayOutDir.addWidget(self.outDirButton)
        self.groupBoxOutDir.setLayout(hboxLayOutDir)

        vbox.addWidget(self.groupBoxFiles)
        vbox.addWidget(self.groupBoxLabelFiles)
        vbox.addWidget(self.label1)
        vbox.addWidget(self.trainFramesLineEdit)
        vbox.addWidget(self.slider)
        vbox.addWidget(self.groupBoxOutDir)
        vbox.addWidget(self.buttonbox)

        self.setLayout(vbox)
        self.show()

    def onInputFileButtonClicked(self):
        self.label_list_text, filter = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open labels txt file",
            directory=str(Path()),
            filter='*'

        )
        if self.label_list_text is not None:
            self.inputLabelFileLineEdit.setText(self.label_list_text)

    def onOutDirButtonClicked(self):
        self.out_dir = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                  "Select Directory")
        if self.out_dir is not None:
            self.outFileDirEdit.setText(self.out_dir)

    def onOutAnnoDirButtonClicked(self):
        self.annotation_dir = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                         "Select Directory")
        if self.annotation_dir is not None:
            self.annoFileLineEdit.setText(self.annotation_dir)
            self.num_train_frames = len(
                glob.glob(osp.join(self.annotation_dir, '*.json')))
            self.trainFramesLineEdit.setText(str(self.num_train_frames))

    def slider(self):
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1000)
        self.slider.setValue(100)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.onSliderChange)

    def onSliderChange(self, position):
        self.num_train_frames = int(position) if position and str(
            position).isdigit() else 0

        self.trainFramesLineEdit.setText(str(position))
        self.label1.setText(
            f"{str(self.num_train_frames)} frames are available for training"
        )
