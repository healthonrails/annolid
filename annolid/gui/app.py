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


class AnnolidWindow(MainWindow):
    def __init__(self,
                 config=None
                 ):
        super(AnnolidWindow, self).__init__()
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
        utils.addActions(self.tools,self.actions.tool)
        utils.addActions(self.menus.coco, (coco,))
        utils.addActions(self.menus.frames, (frames,))
        self.statusBar().showMessage(self.tr("%s started.") % __appname__)
        self.statusBar().show()
        self.setWindowTitle(__appname__)
        self.settings = QtCore.QSettings("Annolid", 'Annolid')

    def frames(self):

        path = Path(self.filename) if self.filename else Path(".")
        video_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr(f"{__appname__} - Choose a video file"),
            str(path.resolve())
        )

        if not video_file:
            return

        videos.extract_frames(
            video_file,
            num_frames=100,
            algo='random'
        )

        out_frames_dir = str(Path(video_file).resolve().with_suffix(''))

        QtWidgets.QMessageBox.about(self,
                                    "Finished",
                                    f"Done! Results are in folder: \
                                         {out_frames_dir}")
        self.statusBar().showMessage(
            self.tr(f"Finshed extracting frames."))

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
