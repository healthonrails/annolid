import sys
from pathlib import Path
import functools
from qtpy import QtCore
from qtpy import QtWidgets
from labelme.app import MainWindow
from labelme.utils import newIcon
from labelme.utils import newAction
from labelme import utils
from labelme.config import get_config
from annolid.annotation import labelme2coco
from annolid.data import videos
__appname__ = 'Annolid'
__version__ = "1.0.0"


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


class AnnolidWindow(MainWindow):
    def __init__(self,
                 config=None
                 ):
        super(AnnolidWindow, self).__init__()

        self.flag_dock.setVisible(True)
        self.label_dock.setVisible(True)
        self.shape_dock.setVisible(True)
        self.file_dock.setVisible(True)
        self.here = Path(__file__).resolve().parent
        action = functools.partial(newAction, self)

        coco = action(
            self.tr("&Convert to COCO format"),
            self.coco,
            'Ctrl+C+O',
            "coco",
            self.tr("Convert to COCO format"),
        )

        frames = action(
            self.tr("&Extract frames"),
            self.frames,
            'Ctrl+Shift+E',
            "Extract frames",
            self.tr("Extract frames frome a video"),
        )

        self.menus = utils.struct(
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            coco=self.menu(self.tr("&COCO")),
            frames=self.menu(self.tr("&Extract Frames")),

        )

        _action_tools = list(self.actions.tool)
        _action_tools.append(coco)
        _action_tools.append(frames)
        self.actions.tool = tuple(_action_tools)
        self.tools.clear()
        utils.addActions(self.tools, self.actions.tool)
        utils.addActions(self.menus.coco, (coco,))
        utils.addActions(self.menus.frames, (frames,))
        self.statusBar().showMessage(self.tr("%s started.") % __appname__)
        self.statusBar().show()
        self.setWindowTitle(__appname__)
        self.settings = QtCore.QSettings("Annolid", 'Annolid')

    def frames(self):

        dlg = ExtractFrameDialog()
        video_file = None
        out_dir = None

        if dlg.exec_():
            video_file = dlg.video_file
            num_frames = dlg.num_frames
            algo = dlg.algo
            out_dir = dlg.out_dir

        if video_file is None:
            return

        videos.extract_frames(
            video_file,
            num_frames=num_frames,
            algo=algo,
            out_dir=out_dir
        )

        if out_dir is None:
            out_frames_dir = str(Path(video_file).resolve().with_suffix(''))
        else:
            out_frames_dir = str(Path(out_dir) / Paht(video_file).name)

        QtWidgets.QMessageBox.about(self,
                                    "Finished",
                                    f"Done! Results are in folder: \
                                         {out_frames_dir}")
        self.statusBar().showMessage(
            self.tr(f"Finshed extracting frames."))
        self.importDirImages(out_frames_dir)

    def coco(self):
        if self.filename is None:
            QtWidgets.QMessageBox.about(self,
                                        "No input file or directory",
                                        f"Please check and open the  \
                                        files or directories.")
            return

        if self.output_dir is None:
            self.output_dir = Path(self.filename).parent

        labels_file = str(self.here.parent / 'annotation' /
                          'labels_custom.txt')
        out_anno_dir = self.output_dir.parent / \
            (self.output_dir.name + '_coco_dataset')
        labelme2coco.convert(
            str(self.output_dir),
            output_annotated_dir=str(out_anno_dir),
            labels_file=labels_file
        )
        self.statusBar().showMessage(self.tr("%s ...") % "converting")
        QtWidgets.QMessageBox.about(self,
                                    "Finished",
                                    f"Done! Results are in folder: \
                                         {str(out_anno_dir)}")
        self.statusBar().showMessage(self.tr("%s Done.") % "converting")


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("icon"))
    win = AnnolidWindow()

    win.show()
    win.raise_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
