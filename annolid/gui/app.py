import sys
import time
import torch
import codecs
import argparse
from pathlib import Path
import functools
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy import QtGui
import requests
import subprocess
from labelme.app import MainWindow
from labelme.utils import newIcon
from labelme.utils import newAction
from labelme import utils
from labelme.config import get_config
from annolid.annotation import labelme2coco
from annolid.data import videos
from annolid.gui.widgets import ExtractFrameDialog
from annolid.gui.widgets import ConvertCOODialog
from annolid.gui.widgets import TrainModelDialog
from annolid.gui.widgets import Glitter2Dialog
from annolid.gui.widgets import TrackDialog
from qtpy.QtWebEngineWidgets import QWebEngineView
from annolid.postprocessing.glitter import tracks2nix
from annolid.gui.widgets import ProgressingWindow
import webbrowser
__appname__ = 'Annolid'
__version__ = "1.0.1"


def start_tensorboard(log_dir=None,
                      tensorboard_url='http://localhost:6006'):

    process = None
    if log_dir is None:
        here = Path(__file__).parent
        log_dir = here.parent.resolve() / "runs" / "logs"
    try:
        r = requests.get(tensorboard_url)
    except requests.exceptions.ConnectionError:
        process = subprocess.Popen(
            ['tensorboard', f'--logdir={str(log_dir)}'])
        time.sleep(8)
    return process


class VisualizationWindow(QtWidgets.QDialog):

    def __init__(self):
        super(VisualizationWindow, self).__init__()
        self.setWindowTitle("Visualization Tensorboard")
        self.process = start_tensorboard()
        self.browser = QWebEngineView()
        self.browser.setUrl(QtCore.QUrl(self.tensorboar_url))
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.browser)
        self.setLayout(vbox)
        self.show()

    def closeEvent(self, event):
        if self.process is not None:
            time.sleep(3)
            self.process.kill()
        event.accept()


class AnnolidWindow(MainWindow):
    """Annolid Main Window based on Labelme.
    """

    def __init__(self,
                 config=None
                 ):

        self.config = config
        super(AnnolidWindow, self).__init__(config=self.config)

        self.flag_dock.setVisible(True)
        self.label_dock.setVisible(True)
        self.shape_dock.setVisible(True)
        self.file_dock.setVisible(True)
        self.here = Path(__file__).resolve().parent
        action = functools.partial(newAction, self)

        coco = action(
            self.tr("&COCO format"),
            self.coco,
            'Ctrl+C+O',
            "coco",
            self.tr("Convert to COCO format"),
        )

        coco.setIcon(QtGui.QIcon(str(
            self.here / "icons/coco.png")))

        save_labeles = action(
            self.tr("&Save labels"),
            self.save_labels,
            'Ctrl+Shift+L',
            'Save Labels',
            self.tr("Save labels to txt file")
        )

        save_labeles.setIcon(QtGui.QIcon(
            str(self.here/"icons/label_list.png")
        ))

        frames = action(
            self.tr("&Extract frames"),
            self.frames,
            'Ctrl+Shift+E',
            "Extract frames",
            self.tr("Extract frames frome a video"),
        )

        models = action(
            self.tr("&Train models"),
            self.models,
            "Ctrl+Shift+T",
            "Train models",
            self.tr("Train neural networks")
        )
        models.setIcon(QtGui.QIcon(str(
            self.here / "icons/models.png")))

        frames.setIcon(QtGui.QIcon(str(
            self.here / "icons/extract_frames.png")))

        tracks = action(
            self.tr("&Track Animals"),
            self.tracks,
            "Ctrl+Shift+O",
            "Track Animals",
            self.tr("Track animals and Objects")
        )

        tracks.setIcon(QtGui.QIcon(str(
            self.here / 'icons/track.png'
        )))

        glitter2 = action(
            self.tr("&Glitter2"),
            self.glitter2,
            "Ctrl+Shift+G",
            self.tr("Convert to Glitter2 nix format")
        )

        glitter2.setIcon(QtGui.QIcon(str(
            self.here / 'icons/glitter2_logo.png'
        )))

        visualization = action(
            self.tr("&Visualization"),
            self.visualization,
            'Ctrl+Shift+V',
            "Visualization",
            self.tr("Visualization results"),
        )

        visualization.setIcon(QtGui.QIcon(str(
            self.here / "icons/visualization.png")))

        self.menus = utils.struct(
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            frames=self.menu(self.tr("&Extract Frames")),
            coco=self.menu(self.tr("&COCO")),
            models=self.menu(self.tr("&Train models")),
            visualization=self.menu(self.tr("&Visualization")),
            tracks=self.menu(self.tr("&Track Animals")),
            glitter2=self.menu(self.tr("&Glitter2")),
            save_labels=self.menu(self.tr("&Save Labels")),
        )

        _action_tools = list(self.actions.tool)
        _action_tools.insert(0, frames)
        _action_tools.append(coco)
        _action_tools.append(models)
        _action_tools.append(visualization)
        _action_tools.append(tracks)
        _action_tools.append(glitter2)
        _action_tools.append(save_labeles)
        self.actions.tool = tuple(_action_tools)
        self.tools.clear()
        utils.addActions(self.tools, self.actions.tool)
        utils.addActions(self.menus.frames, (frames,))
        utils.addActions(self.menus.coco, (coco,))
        utils.addActions(self.menus.models, (models,))
        utils.addActions(self.menus.visualization, (visualization,))
        utils.addActions(self.menus.tracks, (tracks,))
        utils.addActions(self.menus.glitter2, (glitter2,))
        utils.addActions(self.menus.save_labels, (save_labeles,))
        self.statusBar().showMessage(self.tr("%s started.") % __appname__)
        self.statusBar().show()
        self.setWindowTitle(__appname__)
        self.settings = QtCore.QSettings("Annolid", 'Annolid')

    def popLabelListMenu(self, point):
        try:
            self.menus.labelList.exec_(self.labelList.mapToGlobal(point))
        except AttributeError:
            return

    def save_labels(self):
        """Save the labels into a selected text file.
        """
        file_name, extension = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save labels file",
            str(self.here.parent / 'annotation'),
            filter='*.txt'
        )

        if Path(file_name).is_file() or Path(file_name).parent.is_dir():
            labels_text_list = ['__ignore__', '_background_']
            for l in self.labelList:
                label_name = l.text().split()[0]
                labels_text_list.append(label_name)

            with open(file_name, 'w') as lt:
                for ltl in labels_text_list:
                    lt.writelines(ltl+'\n')
        else:
            return

    def frames(self):
        """Extract frames based on the selected algos. 
        """
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
            out_frames_dir = str(Path(out_dir) / Path(video_file).name)

        QtWidgets.QMessageBox.about(self,
                                    "Finished",
                                    f"Done! Results are in folder: \
                                         {out_frames_dir}")
        self.statusBar().showMessage(
            self.tr(f"Finshed extracting frames."))
        self.importDirImages(out_frames_dir)

    def tracks(self):
        dlg = TrackDialog()
        config_file = None
        out_dir = None
        score_threshold = 0.15
        algo = "YOLACT"
        video_file = None
        model_path = None
        top_k = 100
        video_multiframe = 1
        display_mask = False
        out_video_file = None

        if dlg.exec_():
            config_file = dlg.config_file
            score_threshold = 0.15
            algo = dlg.algo
            out_dir = dlg.out_dir
            video_file = dlg.video_file
            model_path = dlg.trained_model

        if video_file is None:
            return

        out_video_file = str(Path(video_file).name)
        out_video_file = f"tracked_{out_video_file}"

        if config_file is None:
            return
        if not torch.cuda.is_available():
            QtWidgets.QMessageBox.about(self,
                                        "Not GPU available",
                                        "At least one GPU  is required to train models.")
            return

        subprocess.Popen(['annolid-track',
                          f'--trained_model={model_path}',
                          f'--config={config_file}',
                          f'--score_threshold={score_threshold}',
                          f'--top_k={top_k}',
                          f'--video_multiframe={video_multiframe}',
                          f'--video={video_file}|{out_video_file}',
                          f'--mot',
                          f'--display_mask={display_mask}'
                          ]
                         )

        if out_dir is None:
            out_runs_dir = Path(__file__).parent.parent / 'runs'
        else:
            out_runs_dir = Path(out_dir) / Path(config_file).name / 'runs'

        out_runs_dir.mkdir(exist_ok=True, parents=True)

        QtWidgets.QMessageBox.about(self,
                                    "Started",
                                    f"Results are in folder: \
                                         {str(out_runs_dir)}")
        self.statusBar().showMessage(
            self.tr(f"Tracking..."))

    def models(self):

        dlg = TrainModelDialog()
        config_file = None
        out_dir = None

        if dlg.exec_():
            config_file = dlg.config_file
            batch_size = dlg.batch_size
            algo = dlg.algo
            out_dir = dlg.out_dir

        if config_file is None:
            return

        # start training models
        if not torch.cuda.is_available():
            QtWidgets.QMessageBox.about(self,
                                        "Not GPU available",
                                        "At least one GPU  is required to train models.")
            return

        subprocess.Popen(['annolid-train',
                          f'--config={config_file}',
                          f'--batch_size={batch_size}'])

        process = start_tensorboard()

        if out_dir is None:
            out_runs_dir = Path(__file__).parent.parent / 'runs'
        else:
            out_runs_dir = Path(out_dir) / Path(config_file).name / 'runs'

        out_runs_dir.mkdir(exist_ok=True, parents=True)

        QtWidgets.QMessageBox.about(self,
                                    "Started",
                                    f"Results are in folder: \
                                         {str(out_runs_dir)}")
        self.statusBar().showMessage(
            self.tr(f"Training..."))

    def coco(self):
        """
        Convert Labelme annotations to COCO format.
        """
        output_dir = None
        labels_file = None
        input_anno_dir = None
        coco_dlg = ConvertCOODialog()
        if coco_dlg.exec_():
            input_anno_dir = coco_dlg.annotation_dir
            labels_file = coco_dlg.label_list_text
            output_dir = coco_dlg.out_dir
        else:
            return

        if input_anno_dir is None:
            QtWidgets.QMessageBox.about(self,
                                        "No input file or directory",
                                        f"Please check and open the  \
                                        files or directories.")
            return

        if output_dir is None:
            self.output_dir = Path(input_anno_dir).parent / \
                (Path(input_anno_dir).name + '_coco_dataset')

        else:
            self.output_dir = output_dir

        if labels_file is None:
            labels_file = str(self.here.parent / 'annotation' /
                              'labels_custom.txt')

        label_gen = labelme2coco.convert(
            str(input_anno_dir),
            output_annotated_dir=str(self.output_dir),
            labels_file=labels_file
        )
        pw = ProgressingWindow(label_gen)
        if pw.exec_():
            pw.runner_thread.terminate()

        self.statusBar().showMessage(self.tr("%s ...") % "converting")
        QtWidgets.QMessageBox.about(self,
                                    "Finished",
                                    f"Done! Results are in folder: \
                                            {str(self.output_dir)}")
        self.statusBar().showMessage(self.tr("%s Done.") % "converting")

    def visualization(self):
        try:
            url = 'http://localhost:6006/'
            process = start_tensorboard(tensorboard_url=url)
            webbrowser.open(url)
        except Exception:
            vdlg = VisualizationWindow()
            if vdlg.exec_():
                pass

    def glitter2(self):

        video_file = None
        tracking_results = None
        out_nix_csv_file = None

        g_dialog = Glitter2Dialog()
        if g_dialog.exec_():
            video_file = g_dialog.video_file
            tracking_results = g_dialog.tracking_results
            out_nix_csv_file = g_dialog.out_nix_csv_file
        else:
            return

        if video_file is None or tracking_results is None:
            QtWidgets.QMessageBox.about(self,
                                        "No input video or tracking results",
                                        f"Please check and open the  \
                                        files.")
            return

        if out_nix_csv_file is None:
            out_nix_csv_file = tracking_results.replace('.csv', '_nix.csv')

        tracks2nix(
            video_file,
            tracking_results,
            out_nix_csv_file
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="show version"
    )

    parser.add_argument(
        '--labels',
        default=argparse.SUPPRESS,
        help="comma separated list of labels or file containing labels"
    )

    default_config_file = str(Path.home() / '.labelmerc')
    parser.add_argument(
        '--config',
        dest="config",
        default=default_config_file,
        help=f"config file or yaml format string default {default_config_file}"
    )

    args = parser.parse_args()

    if hasattr(args, "labels"):
        if Path(args.labels).is_file():
            with codecs.open(args.labels,
                             'r', encoding='utf-8'
                             ) as f:
                args.labels = [line.strip()
                               for line in f if line.strip()
                               ]
        else:
            args.labels = [
                line for line in args.labels.split(',')
                if line
            ]

    config_from_args = args.__dict__
    config_from_args.pop("version")
    config_file_or_yaml = config_from_args.pop("config")

    config = get_config(config_file_or_yaml, config_from_args)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("icon"))
    win = AnnolidWindow(config=config)

    win.show()
    win.raise_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
