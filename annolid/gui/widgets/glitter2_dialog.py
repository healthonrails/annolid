from pathlib import Path
from qtpy import QtCore
from qtpy import QtWidgets


class Glitter2Dialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        super(Glitter2Dialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Convert to Glitter2 nix format")
        self.video_file = None
        self.tracking_results = None
        self.out_nix_csv_file = None
        self.zone_info_json = None

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

        self.groupBoxZoneFiles = QtWidgets.QGroupBox(
            "Please select a zone info json format file (Optional)"
        )
        self.inputZoneInfoLineEdit = QtWidgets.QLineEdit(self)
        self.inputZoneInfoButton = QtWidgets.QPushButton('Open', self)

        self.inputZoneInfoButton.clicked.connect(
            self.onInputZoneInfoButtonClicked
        )
        hboxZoneInfoLayout = QtWidgets.QHBoxLayout()
        hboxZoneInfoLayout.addWidget(self.inputZoneInfoLineEdit)
        hboxZoneInfoLayout.addWidget(self.inputZoneInfoButton)
        self.groupBoxZoneFiles.setLayout(hboxZoneInfoLayout)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.groupBoxVideoFiles)
        vbox.addWidget(self.groupBoxFiles)
        vbox.addWidget(self.groupBoxZoneFiles)
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

    def onInputZoneInfoButtonClicked(self):
        self.zone_info_json, filter = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open zone info json file",
            directory=str(Path()),
            filter='*'

        )
        if self.zone_info_json is not None:
            self.inputZoneInfoLineEdit.setText(self.zone_info_json)
