import sys
import os
import re
import csv
import os.path as osp
import time
import html
import shutil
import pandas as pd
from collections import deque
import torch
import codecs
import imgviz
import argparse
from pathlib import Path
import functools
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtWidgets
from qtpy import QtGui
from labelme import PY2
from labelme import QT5
import PIL
from PIL import ImageQt
import requests
import subprocess
from annolid.gui.shape import Shape
from labelme.app import MainWindow
from labelme.utils import newAction
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import LabelListWidgetItem
from labelme.label_file import LabelFileError
from labelme.label_file import LabelFile
from labelme import utils
from labelme.widgets import ToolBar
from annolid.configs import get_config
from annolid.gui.widgets.canvas import Canvas
from annolid.annotation import labelme2coco
from annolid.data import videos
from annolid.gui.widgets import ExtractFrameDialog
from annolid.gui.widgets import ConvertCOODialog
from annolid.gui.widgets import TrainModelDialog
from annolid.gui.widgets import Glitter2Dialog
from annolid.gui.widgets import QualityControlDialog
from annolid.gui.widgets import TrackDialog
from qtpy.QtWebEngineWidgets import QWebEngineView
from annolid.postprocessing.glitter import tracks2nix
from annolid.postprocessing.quality_control import TracksResults
from annolid.gui.widgets import ProgressingWindow
import webbrowser
import atexit
import qimage2ndarray
from annolid.gui.widgets.video_slider import VideoSlider, VideoSliderMark
from annolid.gui.widgets.step_size_widget import StepSizeWidget
from annolid.postprocessing.quality_control import pred_dict_to_labelme
from annolid.annotation.keypoints import save_labels
from annolid.annotation.timestamps import convert_frame_number_to_time
__appname__ = 'Annolid'
__version__ = "1.1.3"


LABEL_COLORMAP = imgviz.label_colormap(value=200)


class FlexibleWorker(QtCore.QObject):
    start = QtCore.Signal()

    def __init__(self, function, *args, **kwargs):
        super(FlexibleWorker, self).__init__()

        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.function(*self.args, **self.kwargs)


class LoadFrameThread(QtCore.QObject):
    """Thread for loading video frames. 
    """
    res_frame = QtCore.Signal(QtGui.QImage)
    process = QtCore.Signal()

    frame_queue = []
    request_waiting_time = 1
    reload_times = None
    previous_process_time = 0
    video_loader = None

    def __init__(self, *args, **kwargs):
        super(LoadFrameThread, self).__init__(*args, **kwargs)
        self.working_lock = QtCore.QMutex()
        self.current_load_times = deque(maxlen=5)

        self.process.connect(self.load)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.load)
        self.timer.start(20)

    def load(self):
        self.previous_process_time = time.time()
        if not self.frame_queue:
            return

        self.working_lock.lock()
        if not self.frame_queue:
            return

        frame_number = self.frame_queue[-1]
        self.frame_queue = []

        try:
            t_start = time.time()
            frame = self.video_loader.load_frame(frame_number)
            self.current_load_times.append(time.time() - t_start)
            average_load_time = sum(self.current_load_times) / \
                len(self.current_load_times)
            self.request_waiting_time = average_load_time
        except Exception:
            frame = None

        self.working_lock.unlock()
        if frame is not None:
            qimage = qimage2ndarray.array2qimage(frame)
            self.res_frame.emit(qimage)

    def request(self, frame_number):
        self.frame_queue.append(frame_number)

        t_last = time.time() - self.previous_process_time

        if t_last > self.request_waiting_time:
            self.previous_process_time = time.time()
            self.process.emit()


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
        self._df = None
        self.label_stats = {}
        self.shape_hash_ids = {}
        self.changed_json_stats = {}
        self._pred_res_folder_suffix = '_tracking_results_labelme'
        self.frame_number = 0
        self.video_loader = None
        self.video_file = None
        self.isPlaying = False
        self._time_stamp = ''
        self.timestamp_dict = dict()
        self.annotation_dir = None
        self.highlighted_mark = None
        self.step_size = 1
        self.stepSizeWidget = StepSizeWidget()

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
            sam=self._config["sam"]
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }

        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.setCentralWidget(scrollArea)

        self.createPolygonSAMMode = action(
            self.tr("AI Polygons"),
            self.segmentAnything,
            icon="objects",
            tip=self.tr("Start creating polygons with segment anything"),
        )

        open_video = action(
            self.tr("&Open Video"),
            self.openVideo,
            None,
            "Open Video",
            self.tr("Open video")
        )
        open_video.setIcon(QtGui.QIcon(
            str(
                self.here / "icons/open_video.png"
            )
        ))

        step_size = QtWidgets.QWidgetAction(self)
        step_size.setIcon(QtGui.QIcon(
            str(
                self.here / "icons/fast_forward.png"
            )
        ))

        step_size.setDefaultWidget(self.stepSizeWidget)

        self.stepSizeWidget.setWhatsThis(
            self.tr(
                "Step for the next or prev image. e.g. 30"
            )
        )
        self.stepSizeWidget.setEnabled(False)

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

        quality_control = action(
            self.tr("&Quality Control"),
            self.quality_control,
            "Ctrl+Shift+G",
            self.tr("Convert to tracking results to labelme format")
        )

        quality_control.setIcon(QtGui.QIcon(str(
            self.here / 'icons/quality_control.png'
        )))

        visualization = action(
            self.tr("&Visualization"),
            self.visualization,
            'Ctrl+Shift+V',
            "Visualization",
            self.tr("Visualization results"),
        )

        colab = action(
            self.tr("&Open in Colab"),
            self.train_on_colab,
            icon="Colab",
            tip=self.tr("Open in Colab"),
        )

        colab.setIcon(QtGui.QIcon(str(
            self.here / "icons/colab.png")))
        shortcuts = self._config["shortcuts"]

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )

        edit = action(
            self.tr("&Edit Label"),
            self.editLabel,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )
        # Lavel list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))

        visualization.setIcon(QtGui.QIcon(str(
            self.here / "icons/visualization.png")))

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
            labelList=labelMenu,
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            frames=self.menu(self.tr("&Extract Frames")),
            open_video=self.menu(self.tr("&Open Video")),
            coco=self.menu(self.tr("&COCO")),
            models=self.menu(self.tr("&Train models")),
            visualization=self.menu(self.tr("&Visualization")),
            tracks=self.menu(self.tr("&Track Animals")),
            glitter2=self.menu(self.tr("&Glitter2")),
            save_labels=self.menu(self.tr("&Save Labels")),
            quality_control=self.menu(self.tr("&Quality Control")),
            colab=self.menu(self.tr("&Open in Colab")),
        )

        _action_tools = list(self.actions.tool)
        _action_tools.insert(0, frames)
        _action_tools.insert(1, open_video)
        _action_tools.insert(2, step_size)
        _action_tools.append(self.createPolygonSAMMode)
        _action_tools.append(coco)
        _action_tools.append(models)
        _action_tools.append(tracks)
        _action_tools.append(glitter2)
        _action_tools.append(save_labeles)
        _action_tools.append(quality_control)
        _action_tools.append(colab)
        _action_tools.append(visualization)

        self.actions.tool = tuple(_action_tools)
        self.tools.clear()
        utils.addActions(self.tools, self.actions.tool)
        utils.addActions(self.menus.frames, (frames,))
        utils.addActions(self.menus.open_video, (open_video,))
        utils.addActions(self.menus.coco, (coco,))
        utils.addActions(self.menus.models, (models,))
        utils.addActions(self.menus.visualization, (visualization,))
        utils.addActions(self.menus.tracks, (tracks,))
        utils.addActions(self.menus.glitter2, (glitter2,))
        utils.addActions(self.menus.save_labels, (save_labeles,))
        utils.addActions(self.menus.quality_control, (quality_control,))
        utils.addActions(self.menus.colab, (colab,))
        self.statusBar().showMessage(self.tr("%s started.") % __appname__)
        self.statusBar().show()
        self.setWindowTitle(__appname__)
        self.settings = QtCore.QSettings("Annolid", 'Annolid')
        self.video_results_folder = None
        self.seekbar = None

        self.frame_worker = QtCore.QThread()
        self.frame_loader = LoadFrameThread()
        self.seg_pred_thread = QtCore.QThread()
        self.seg_train_thread = QtCore.QThread()
        self.destroyed.connect(self.clean_up)
        self.stepSizeWidget.valueChanged.connect(self.update_step_size)
        atexit.register(self.clean_up)
        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

    def update_step_size(self, value):
        self.step_size = value

    def segmentAnything(self,):
        try:
            self.toggleDrawMode(False, createMode="polygonSAM")
            self.canvas.loadSamPredictor()
        except ImportError as e:
            print(e)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState()

    def playVideo(self, isPlaying=False):
        self.isPlaying = isPlaying
        if self.video_loader is None:
            return
        if self.isPlaying:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.openNextImg)
            if self.fps:
                self.timer.start(1 / self.fps)
            else:
                # handle the zero FPS case here, e.g.:
                self.timer.start(1/30)
        else:
            self.timer.stop()

    def startPlaying(self):
        self.playVideo(isPlaying=True)

    def stopPlaying(self):
        self.playVideo(isPlaying=False)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            self.actions.createMode.setEnabled(True)
            self.actions.createRectangleMode.setEnabled(True)
            self.actions.createCircleMode.setEnabled(True)
            self.actions.createLineMode.setEnabled(True)
            self.actions.createPointMode.setEnabled(True)
            self.actions.createLineStripMode.setEnabled(True)
            self.createPolygonSAMMode.setEnabled(True)
        else:
            if createMode == "polygon":
                self.actions.createMode.setEnabled(False)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
                self.createPolygonSAMMode.setEnabled(True)
            elif createMode == "rectangle":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(False)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
                self.createPolygonSAMMode.setEnabled(True)
            elif createMode == "line":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(False)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
                self.createPolygonSAMMode.setEnabled(True)
            elif createMode == "point":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(False)
                self.actions.createLineStripMode.setEnabled(True)
                self.createPolygonSAMMode.setEnabled(True)
            elif createMode == "circle":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(False)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
                self.createPolygonSAMMode.setEnabled(True)
            elif createMode == "linestrip":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(False)
                self.createPolygonSAMMode.setEnabled(True)
            elif createMode == "polygonSAM":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
                self.createPolygonSAMMode.setEnabled(False)
            else:
                raise ValueError("Unsupported createMode: %s" % createMode)
        self.actions.editMode.setEnabled(not edit)

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)
        self.uniqLabelList.clear()
        # clear the file list
        self.fileListWidget.clear()
        if self.video_loader is not None:
            self.video_loader = None
            self.num_frames = None
            self.video_file = None
            self.annotation_dir = None
            self.statusBar().removeWidget(self.seekbar)
            self.statusBar().removeWidget(self.saveButton)
            self.statusBar().removeWidget(self.playButton)
            self.seekbar = None
            self._df = None
            self.label_stats = {}
            self.shape_hash_ids = {}
            self.changed_json_stats = {}
            self._pred_res_folder_suffix = '_tracking_results_labelme'
            self.frame_number = 0
            self.step_size = 1
            self.video_results_folder = None
            self.timestamp_dict = dict()
            self.isPlaying = False
            self._time_stamp = ''
            self.saveButton = None
            self.playButton = None
            self.timer = None

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        extensions.append('.json')
        self.only_json_files = True

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = osp.join(root, file)
                    if self.only_json_files and not file.lower().endswith('.json'):
                        self.only_json_files = False
                    images.append(relativePath)
        images.sort(key=lambda x: x.lower())
        return images

    def _addItem(self, filename, label_file):
        item = QtWidgets.QListWidgetItem(filename)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
            label_file
        ):
            item.setCheckState(Qt.Checked)
        else:
            item.setCheckState(Qt.Unchecked)
        if not self.fileListWidget.findItems(filename, Qt.MatchExactly):
            self.fileListWidget.addItem(item)

    def _getLabelFile(self, filename):
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        return label_file

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if self.canvas.hShape and not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.annotation_dir = dirpath
        self.filename = None
        self.fileListWidget.clear()
        for filename in self.scanAllImages(dirpath):
            if pattern and pattern not in filename:
                continue
            label_file = self._getLabelFile(filename)

            if not filename.endswith('.json') or self.only_json_files:
                self._addItem(filename, label_file)
        self.openNextImg(load=load)

    def _get_rgb_by_label(self, label):
        if self._config["shape_color"] == "auto":
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]

    def _update_shape_color(self, shape):

        if not self.uniqLabelList.findItemByLabel(shape.label):
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)
        return r, g, b

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        shape_points_hash = hash(
            str(sorted(shape.points, key=lambda point: point.x())))
        self.shape_hash_ids[shape_points_hash] = self.shape_hash_ids.get(
            shape_points_hash, 0) + 1
        if self.shape_hash_ids[shape_points_hash] <= 1:
            self.label_stats[text] = self.label_stats.get(text, 0) + 1
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        item = self.uniqLabelList.findItemByLabel(shape.label)
        if item is None:
            item = self.uniqLabelList.createItemFromLabel(
                shape.label
            )
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(
                item, f"{shape.label} [{self.label_stats.get(text,0)} instance]", rgb)
        else:
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(
                item, f"{shape.label} [{self.label_stats.get(text,0)} instances]", rgb)

        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        r, g, b = self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), r, g, b
            )
        )

    def editLabel(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")

        if not self.canvas.editing():
            return
        if not item:
            item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label, flags=shape.flags,
            group_id=shape.group_id,
            description=shape.description,
        )
        if text is None:
            return
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id
        shape.description = description

        r, g, b = self._update_shape_color(shape)

        if shape.group_id is None:
            item.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    shape.label, r, g, b))
        else:
            item.setText("{} ({})".format(shape.label, shape.group_id))
        self.setDirty()
        if not self.uniqLabelList.findItemByLabel(shape.label):
            item = QtWidgets.QListWidgetItem()
            item.setData(Qt.UserRole, shape.label)
            self.uniqLabelList.addItem(item)

    def _saveImageFile(self, filename):
        image_filename = filename.replace('.json', '.png')
        imgage_jpg_file = filename.replace('.json', '.jpg')
        # save png if there is no png or jpg image in the folder
        if (not Path(image_filename).exists()
                and not Path(imgage_jpg_file).exists()):
            img = utils.img_data_to_arr(self.imageData)
            imgviz.io.imsave(image_filename, img)
        return image_filename

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            if len(s.points) <= 1:
                s.shape_type = 'point'
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=Path(imagePath).name,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(
                self.imagePath, Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def _saveFile(self, filename):
        json_saved = self.saveLabels(filename)
        if filename and json_saved:
            self.changed_json_stats[filename] = self.changed_json_stats.get(
                filename, 0) + 1
            if (self.changed_json_stats[filename] >= 1
                    and self._pred_res_folder_suffix in filename):

                changed_folder = str(Path(filename).parent)
                if '_edited' not in changed_folder:
                    changed_folder = Path(changed_folder + '_edited')
                else:
                    changed_folder = Path(changed_folder)
                if not changed_folder.exists():
                    changed_folder.mkdir(exist_ok=True, parents=True)
                changed_filename = changed_folder / Path(filename).name
                _ = self.saveLabels(str(changed_filename))
                _ = self._saveImageFile(
                    str(changed_filename).replace('.json', '.png'))
            image_filename = self._saveImageFile(filename)
            self.imageList.append(image_filename)
            self.addRecentFile(filename)
            label_file = self._getLabelFile(filename)
            self._addItem(image_filename, label_file)
            self.setClean()

    def getLabelFile(self):
        if str(self.filename).lower().endswith(".json"):
            label_file = str(self.filename)
        else:
            label_file = osp.splitext(str(self.filename))[0] + ".json"

        return label_file

    def popLabelListMenu(self, point):
        try:
            self.menus.labelList.exec_(self.labelList.mapToGlobal(point))
        except AttributeError:
            return

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = self.getTitle(clean=False)
        self.setWindowTitle(title)

    def getTitle(self, clean=True):
        title = __appname__
        _filename = os.path.basename(self.filename)
        if self.video_loader:
            if self.frame_number:
                self._time_stamp = convert_frame_number_to_time(
                    self.frame_number)
                if clean:
                    title = f"{title}-Video Timestamp:{self._time_stamp}-total:{len(self.timestamp_dict.keys())}"
                    title = f"{title}-Frame_number:{self.frame_number}-File Name:{_filename}"
                else:
                    title = f"{title}-Video Timestamp:{self._time_stamp}"

                    title = f"{title}-Frame_number:{self.frame_number}-{_filename}*"
            else:
                if clean:
                    title = "{} - {}".format(title, _filename)
                else:
                    title = "{} - {}*".format(title, _filename)
        return title

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = self.getTitle()
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def save_labels(self):
        """Save the labels into a selected text file.
        """
        file_name, extension = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save labels file",
            str(self.here.parent / 'annotation'),
            filter='*.txt'
        )

        # return if user cancels the operation
        if len(file_name) < 1:
            return

        if Path(file_name).is_file() or Path(file_name).parent.is_dir():
            labels_text_list = ['__ignore__', '_background_']
            for i in range(self.uniqLabelList.count()):
                # get all the label names from the unique list
                label_name = self.uniqLabelList.item(
                    i).data(QtCore.Qt.UserRole)
                labels_text_list.append(label_name)

            with open(file_name, 'w') as lt:
                for ltl in labels_text_list:
                    lt.writelines(ltl+'\n')

    def frames(self):
        """Extract frames based on the selected algos.
        """
        dlg = ExtractFrameDialog(self.video_file)
        video_file = None
        out_dir = None

        if dlg.exec_():
            video_file = dlg.video_file
            num_frames = dlg.num_frames
            algo = dlg.algo
            out_dir = dlg.out_dir
            start_seconds = dlg.start_sconds
            end_seconds = dlg.end_seconds
            sub_clip = isinstance(
                start_seconds, int) and isinstance(end_seconds, int)

        if video_file is None:
            return
        out_frames_gen = videos.extract_frames(
            video_file,
            num_frames=num_frames,
            algo=algo,
            out_dir=out_dir,
            sub_clip=sub_clip,
            start_seconds=start_seconds,
            end_seconds=end_seconds
        )

        pw = ProgressingWindow(out_frames_gen)
        if pw.exec_():
            pw.runner_thread.terminate()

        if out_dir is None:
            out_frames_dir = str(
                Path(video_file).resolve().with_suffix(''))
        else:
            out_frames_dir = str(Path(out_dir) / Path(video_file).name)

        if start_seconds is not None and end_seconds is not None:
            out_frames_dir = f"{out_frames_dir}_{start_seconds}_{end_seconds}"
        out_frames_dir = f"{out_frames_dir}_{algo}"

        self.annotation_dir = out_frames_dir

        QtWidgets.QMessageBox.about(self,
                                    "Finished",
                                    f"Done! Results are in folder: \
                                         {out_frames_dir}")
        self.statusBar().showMessage(
            self.tr(f"Finshed extracting frames."))
        self.importDirImages(out_frames_dir)

    def tracks(self):
        """
        Track animals using the train models for a video
        The tracking results CSV file will be saved on the disk.
        """

        dlg = TrackDialog()
        config_file = None
        out_dir = None
        score_threshold = 0.15
        algo = None
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

        if algo == 'Detectron2':
            try:
                import detectron2
            except ImportError:
                QtWidgets.QMessageBox.about(self,
                                            "Detectron2 is not installed",
                                            "Please check the docs and install it.")
                return
            out_folder = Path(video_file).with_suffix('')
            if out_folder.exists():
                QtWidgets.QMessageBox.about(self,
                                            f"You folder {str(out_folder)} is not empty.",
                                            "Please backup your data or rename it.")

            from annolid.inference.predict import Segmentor
            dataset_dir = str(Path(config_file).parent)
            segmentor = Segmentor(
                dataset_dir,
                model_path,
                score_threshold)

            try:
                self.seg_pred_thread.start()
                self.pred_worker = FlexibleWorker(
                    function=segmentor.on_video, video_path=video_file)
                self.pred_worker.moveToThread(self.seg_pred_thread)
                self.pred_worker.start.connect(self.pred_worker.run)
                self.pred_worker.start.emit()
                out_result_dir = Path(video_file).with_suffix('')
            except AttributeError:
                out_result_dir = segmentor.on_video(video_file)
            QtWidgets.QMessageBox.about(self,
                                        "Running",
                                        f"Results will be saved to folder: \
                                         {out_result_dir} \
                                        Please do not close Annolid GUI"
                                        )
            self.importDirImages(out_result_dir)

        if algo == 'YOLACT':
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
        """
        Train a model with the provided dataset and selected params
        """

        dlg = TrainModelDialog()
        config_file = None
        out_dir = None
        max_iterations = 2000
        batch_size = 8
        model_path = None

        if dlg.exec_():
            config_file = dlg.config_file
            batch_size = dlg.batch_size
            algo = dlg.algo
            out_dir = dlg.out_dir
            max_iterations = dlg.max_iterations
            model_path = dlg.trained_model

        if config_file is None:
            return

        if algo == 'YOLACT':
            # start training models
            if not torch.cuda.is_available():
                QtWidgets.QMessageBox.about(self,
                                            "Not GPU available",
                                            "At least one GPU  is required to train models.")
                return

            subprocess.Popen(['annolid-train',
                              f'--config={config_file}',
                              f'--batch_size={batch_size}'])

            if out_dir is None:
                out_runs_dir = Path(__file__).parent.parent / 'runs'
            else:
                out_runs_dir = Path(out_dir) / Path(config_file).name / 'runs'

            out_runs_dir.mkdir(exist_ok=True, parents=True)
            process = start_tensorboard(log_dir=out_runs_dir)
            QtWidgets.QMessageBox.about(self,
                                        "Started",
                                        f"Results are in folder: \
                                         {str(out_runs_dir)}")

        elif algo == 'MaskRCNN':
            from annolid.segmentation.maskrcnn.detectron2_train import Segmentor
            dataset_dir = str(Path(config_file).parent)
            segmentor = Segmentor(dataset_dir, out_dir,
                                  max_iterations=max_iterations,
                                  batch_size=batch_size,
                                  model_pth_path=model_path
                                  )
            out_runs_dir = segmentor.out_put_dir
            process = start_tensorboard(log_dir=out_runs_dir)
            try:
                self.seg_train_thread.start()
                train_worker = FlexibleWorker(function=segmentor.train)
                train_worker.moveToThread(self.seg_train_thread)
                train_worker.start.connect(train_worker.run)
                train_worker.start.emit()
            except Exception:
                segmentor.train()

            QtWidgets.QMessageBox.about(self,
                                        "Started.",
                                        f"Training in background... \
                                        Results will be saved to folder: \
                                         {str(out_runs_dir)} \
                                         Please do not close Annolid GUI."
                                        )
            self.statusBar().showMessage(
                self.tr(f"Training..."))

    def handle_uniq_label_list_selection_change(self):
        selected_items = self.uniqLabelList.selectedItems()
        if selected_items:
            self.add_highlighted_mark()
        else:
            self.add_highlighted_mark(mark_type='event_end',
                                      color='red')

    def add_highlighted_mark(self, val=None,
                             mark_type='event_start',
                             color='green'):
        """Adds a new highlighted mark with a green filled color to the slider."""
        if val is None:
            val = self.frame_number
        else:
            val = int(val)
        highlighted_mark = VideoSliderMark(mark_type=mark_type,
                                           val=val, _color=color)
        self.timestamp_dict[(val, mark_type)
                            ] = self._time_stamp if self._time_stamp else convert_frame_number_to_time(val)
        self.seekbar.addMark(highlighted_mark)
        return highlighted_mark

    def remove_highlighted_mark(self):
        if self.highlighted_mark is not None:
            self.seekbar.setValue(self.highlighted_mark.val)
            if self.isPlaying:
                self.togglePlay()
            self.seekbar.removeMark(self.highlighted_mark)
            _item = (self.highlighted_mark.val,
                     self.highlighted_mark.mark_type)
            if _item in self.timestamp_dict:
                del self.timestamp_dict[_item]
                self.highlighted_mark = None
        else:
            _curr_val = self.seekbar.value()
            _marks = list(self.seekbar.getMarks())
            for i in range(len(_marks)):
                if _curr_val == _marks[i].val:
                    self.seekbar.removeMark(_marks[i])
                    _key = (_curr_val, _marks[i].mark_type)
                    if _key in self.timestamp_dict:
                        del self.timestamp_dict[_key]

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if self.seekbar is not None:
            if event.key() == Qt.Key_Right:
                next_pos = self.seekbar.value()+1
                self.seekbar.setValue(next_pos)
                self.seekbar.valueChanged.emit(next_pos)
            elif event.key() == Qt.Key_Left:
                prev_pos = self.seekbar.value()-1
                self.seekbar.setValue(prev_pos)
                self.seekbar.valueChanged.emit(prev_pos)
            elif event.key() == Qt.Key_0:
                self.seekbar.setValue(0)
            elif event.key() == Qt.Key_Space:
                self.togglePlay()
            elif event.key() == Qt.Key_S:
                event_type = 'event_start'
                self.highlighted_mark = self.add_highlighted_mark(
                    self.frame_number, mark_type=event_type)
            elif event.key() == Qt.Key_E:
                event_type = 'event_end'
                self.highlighted_mark = self.add_highlighted_mark(
                    self.frame_number, mark_type=event_type)
            elif event.key() == Qt.Key_R:
                self.remove_highlighted_mark()
            elif event.key() == Qt.Key_Q:
                self.seekbar.setValue(self.seekbar._val_max)
            else:
                event.ignore()

    def saveTimestampList(self):
        # Open file dialog to get file path
        default_timestamp_csv_file = str(
            os.path.dirname(self.filename)) + '_timestamps.csv'
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setDefaultSuffix('.csv')
        file_path, _ = file_dialog.getSaveFileName(self, "Save Timestamps", default_timestamp_csv_file,
                                                   "CSV files (*.csv)")
        if file_path:
            with open(file_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Frame_number'])
                for _key in sorted(self.timestamp_dict.keys()):
                    writer.writerow([self.timestamp_dict[_key], _key])
            QtWidgets.QMessageBox.information(
                self, "Timestamps saved", "Timestamps saved successfully!")

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        return super().keyReleaseEvent(event)

    def togglePlay(self):
        if self.isPlaying:
            self.stopPlaying()
            self.playButton.setIcon(
                QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.playButton.setText("Play")
        else:
            self.startPlaying()
            self.playButton.setIcon(
                QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
            self.playButton.setText("Pause")

    def set_frame_number(self, frame_number):
        self.frame_number = frame_number
        self.filename = self.video_results_folder / \
            f"{str(self.video_results_folder.name)}_{self.frame_number:09}.png"
        self.current_frame_time_stamp = self.video_loader.get_time_stamp()
        self.frame_loader.request(frame_number)

    def load_tracking_results(self, cur_video_folder, video_filename):
        """Load tracking results from CSV files in the given folder that match the video filename."""
        _tracking_csv_file = None
        _video_name = Path(video_filename).stem
        for tr in cur_video_folder.glob('*.csv'):
            if not tr.is_file() or not tr.suffix == '.csv':
                continue

            if 'timestamp' in tr.name and _video_name in tr.name:
                _df_timestamps = pd.read_csv(str(tr))
                # iterate over the rows of the DataFrame
                for _, row in _df_timestamps.iterrows():
                    timestamp = row['Timestamp']
                    frame_time = row['Frame_number']
                    if isinstance(frame_time, int):
                        frame_number = frame_time
                        mark_type = 'event_start'
                    else:
                        frame_number, mark_type = eval(frame_time)
                    # add the timestamp and frame_number to the dictionary
                    self.timestamp_dict[(
                        frame_number, mark_type)] = timestamp
            if 'tracking' in tr.name and _video_name in tr.name and '_nix' not in tr.name:
                _tracking_csv_file = tr
            elif '_labels' in tr.name and _video_name in tr.name:
                self._df = pd.read_csv(tr)
                self._df.rename(
                    columns={'Unnamed: 0': 'frame_number'}, inplace=True)

        if _tracking_csv_file is not None:
            self._df = pd.read_csv(_tracking_csv_file)
            self._df = self._df.drop(columns=['Unnamed: 0'], errors='ignore')

    def openVideo(self, _value=False):
        """open a video for annotaiton frame by frame

        Args:
            _value (bool, optional):  Defaults to False.
        """
        video_path = Path(self.filename).parent if self.filename else "."
        formats = ["*.*"]
        filters = self.tr(f"Video files {formats[0]}")
        video_filename = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr(f"{__appname__} - Choose Video"),
            str(video_path),
            filters,
        )
        if QT5:
            video_filename, _ = video_filename

        video_filename = str(video_filename)
        self.stepSizeWidget.setEnabled(True)
        if self.seekbar:
            self.statusBar().removeWidget(self.seekbar)

        if video_filename:
            cur_video_folder = Path(video_filename).parent
            # go over all the tracking csv files
            # use the first matched file with video name
            # and segmentation
            self.load_tracking_results(cur_video_folder, video_filename)

            self.video_results_folder = Path(video_filename).with_suffix('')

            self.video_results_folder.mkdir(
                exist_ok=True,
                parents=True
            )
            self.annotation_dir = self.video_results_folder
            self.video_file = video_filename
            self.video_loader = videos.CV2Video(video_filename)
            self.fps = self.video_loader.get_fps()

            self.num_frames = self.video_loader.total_frames()
            self.seekbar = VideoSlider()
            self.seekbar.keyPress.connect(self.keyPressEvent)
            self.seekbar.keyRelease.connect(self.keyReleaseEvent)

            # load the first frame
            self.set_frame_number(self.frame_number)

            self.actions.openNextImg.setEnabled(True)
            self.openNextImg(load=True)

            self.actions.openPrevImg.setEnabled(True)

            self.seekbar.valueChanged.connect(lambda f: self.set_frame_number(
                self.seekbar.value()
            ))

            self.frame_loader.video_loader = self.video_loader

            self.frame_loader.moveToThread(self.frame_worker)

            self.frame_worker.start(priority=QtCore.QThread.IdlePriority)

            self.frame_loader.res_frame.connect(
                lambda qimage: self.image_to_canvas(
                    qimage, self.filename, self.frame_number)
            )

            self.seekbar.setMinimum(0)
            self.seekbar.setMaximum(self.num_frames-1)
            self.seekbar.setEnabled(True)
            self.seekbar.resizeEvent()
            self.seekbar.setTooltipCallable(self.tooltip_callable)
            # Create a play button
            self.playButton = QtWidgets.QPushButton("Play", self)
            self.playButton.setIcon(
                QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.playButton.clicked.connect(self.togglePlay)
            # create save button
            self.saveButton = QtWidgets.QPushButton("Save Timestamps", self)
            self.saveButton.clicked.connect(self.saveTimestampList)

            # Add the play button to the status bar
            self.statusBar().addWidget(self.playButton)
            self.statusBar().addWidget(self.seekbar, stretch=1)
            self.statusBar().addWidget(self.saveButton)
            if self.timestamp_dict:
                for frame_number, mark_type in self.timestamp_dict.keys():
                    self.add_highlighted_mark(val=frame_number,
                                              mark_type=mark_type)

    def tooltip_callable(self, val):
        return f"Frame:{val},Time:{convert_frame_number_to_time(val)}"

    def image_to_canvas(self, qimage, filename, frame_number):
        self.resetState()
        self.canvas.setEnabled(True)
        if isinstance(filename, str):
            filename = Path(filename)
        self.imagePath = str(filename.parent)
        self.filename = str(filename)
        self.image = qimage
        imageData = ImageQt.fromqimage(qimage)

        self.imageData = utils.img_pil_to_data(imageData)
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage))
        flags = {k: False for k in self._config["flags"] or []}
        self.loadFlags(flags)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness constrast values
        dialog = BrightnessContrastDialog(
            imageData,
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recentFiles:
            brightness, _ = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recentFiles:
            _, contrast = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightnessContrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            dialog.onNewValue(None)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.loadPredictShapes(frame_number, filename)
        return True

    def clean_up(self):
        if self.frame_worker is not None:
            try:
                self.frame_worker.quit()
                self.frame_worker.wait()
            except RuntimeError:
                print("Thank you!")
        if self.seg_train_thread is not None:
            try:
                self.seg_train_thread.quit()
                self.seg_train_thread.wait()
            except RuntimeError:
                print("See you next time!")

        if self.seg_pred_thread is not None:
            try:
                self.seg_pred_thread.quit()
                self.seg_pred_thread.wait()
            except RuntimeError:
                print("Bye!")

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data
            s.append(shape)

        self.loadShapes(s)
        return s

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def loadPredictShapes(self, frame_number, filename):

        label_json_file = str(filename).replace(".png", ".json")
        if self._df is not None and not Path(filename).exists():
            df_cur = self._df[self._df.frame_number == frame_number]
            frame_label_list = []
            pd.options.mode.chained_assignment = None
            for row in df_cur.to_dict(orient='records'):
                if 'x1' not in row:
                    row['x1'] = 2
                    row['y1'] = 2
                    row['x2'] = 4
                    row['y2'] = 4
                    row['class_score'] = 1
                    df_cur.drop('frame_number', axis=1, inplace=True)
                    try:
                        instance_names = df_cur.apply(lambda row:
                                                      df_cur.columns[[i for i in range(
                                                          len(row)) if row[i] > 0][0]
                                                      ],
                                                      axis=1).tolist()
                        row['instance_name'] = '_'.join(instance_names)
                    except IndexError:
                        row['instance_name'] = 'unknown'
                    row['segmentation'] = None
                pred_label_list = pred_dict_to_labelme(row)
                frame_label_list += pred_label_list

            save_labels(label_json_file,
                        str(filename),
                        frame_label_list,
                        self.video_loader.height,
                        self.video_loader.width,
                        imageData=self.imageData,
                        save_image_to_json=False
                        )

        if Path(label_json_file).exists():
            try:
                self.labelFile = LabelFile(label_json_file)
                if self.labelFile:
                    shapes = self.loadLabels(self.labelFile.shapes)
            except Exception:
                print(f'Count not load label json file {label_json_file}')

            if not Path(filename).exists():
                os.remove(label_json_file)

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if Qt.KeyboardModifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if self.video_loader is not None:
            if self.frame_number < self.num_frames:
                if self.step_size + self.frame_number <= self.num_frames:
                    self.frame_number += self.step_size
                else:
                    self.frame_number += 1
            else:
                self.frame_number = self.num_frames
            self.set_frame_number(self.frame_number)
            # update the seekbar value
            self.seekbar.setValue(self.frame_number)
            self.uniqLabelList.itemSelectionChanged.connect(
                self.handle_uniq_label_list_selection_change)

        elif len(self.imageList) <= 0:
            return
        else:
            filename = None
            if self.filename is None:
                filename = self.imageList[0]
            else:
                currIndex = self.imageList.index(self.filename)
                if currIndex + 1 < len(self.imageList):
                    filename = self.imageList[currIndex + 1]
                else:
                    filename = self.imageList[-1]
            self.filename = filename

            if self.filename and load:
                self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if Qt.KeyboardModifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if self.video_loader is not None:
            if self.frame_number > 1:
                if self.frame_number - self.step_size >= 1:
                    self.frame_number -= self.step_size
                else:
                    self.frame_number -= 1
            else:
                self.frame_number = 0
            self.set_frame_number(self.frame_number)
            # update the seekbar value
            self.seekbar.setValue(self.frame_number)

        elif len(self.imageList) <= 0:
            return
        else:
            if self.filename is None:
                return
            currIndex = self.imageList.index(self.filename)
            if currIndex - 1 >= 0:
                filename = self.imageList[currIndex - 1]
                if filename:
                    self.loadFile(filename)

        self._config["keep_prev"] = keep_prev

    def coco(self):
        """
        Convert Labelme annotations to COCO format.
        """
        coco_dlg = ConvertCOODialog(annotation_dir=self.annotation_dir)
        output_dir = None
        labels_file = None
        input_anno_dir = None
        num_train_frames = 0.7
        if coco_dlg.exec_():
            input_anno_dir = coco_dlg.annotation_dir
            labels_file = coco_dlg.label_list_text
            output_dir = coco_dlg.out_dir
            num_train_frames = coco_dlg.num_train_frames
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
            labels_file=labels_file,
            train_valid_split=num_train_frames
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
        try:
            shutil.make_archive(str(self.output_dir),
                                'zip', self.output_dir.parent, self.output_dir.stem)
        except:
            print("Failed to create the zip file")

    def visualization(self):
        try:
            url = 'http://localhost:6006/'
            process = start_tensorboard(tensorboard_url=url)
            webbrowser.open(url)
        except Exception:
            vdlg = VisualizationWindow()
            if vdlg.exec_():
                pass

    def train_on_colab(self):
        url = "https://colab.research.google.com/github/healthonrails/annolid/blob/main/docs/tutorials/Annolid_on_Detectron2_Tutorial.ipynb"
        webbrowser.open(url)

    def quality_control(self):
        video_file = None
        tracking_results = None
        skip_num_frames = None
        qc_dialog = QualityControlDialog()

        if qc_dialog.exec_():
            video_file = qc_dialog.video_file
            tracking_results = qc_dialog.tracking_results
            skip_num_frames = qc_dialog.skip_num_frames
        else:
            return

        if video_file is None or tracking_results is None:
            QtWidgets.QMessageBox.about(self,
                                        "No input video or tracking results",
                                        f"Please check and open the  \
                                        files.")
            return
        out_dir = f"{str(Path(video_file).with_suffix(''))}{self._pred_res_folder_suffix}"
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        trs = TracksResults(video_file, tracking_results)
        label_json_gen = trs.to_labelme_json(str(out_dir),
                                             skip_frames=skip_num_frames)

        try:
            if label_json_gen is not None:
                pwj = ProgressingWindow(label_json_gen)
                if pwj.exec_():
                    trs._is_running = False
                    pwj.running_submitted.emit('stopped')
                    pwj.runner_thread.terminate()
                    pwj.runner_thread.quit()
        except Exception:
            pass
        finally:
            self.importDirImages(str(out_dir))

    def newShape(self):
        """Pop-up and give focus to the label editor.
        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""
        if self._config["display_label_popup"] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id, description = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)
        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            shapes = self.canvas.setLastLabel(text, flags)
            for shape in shapes:
                shape.group_id = group_id
                shape.description = description
                self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def glitter2(self):
        """
        overlay the predicted masks and bboxes on the inference video
        and convert the nix format for editing with glitter2 package
        https://github.com/matham/glitter2
        """

        video_file = None
        tracking_results = None
        out_nix_csv_file = None
        zone_info_json = None
        score_threshold = None
        motion_threshold = None
        pretrained_model = None
        subject_names = None
        behaviors = None

        g_dialog = Glitter2Dialog()
        if g_dialog.exec_():
            video_file = g_dialog.video_file
            tracking_results = g_dialog.tracking_results
            out_nix_csv_file = g_dialog.out_nix_csv_file
            zone_info_json = g_dialog.zone_info_json
            score_threshold = g_dialog.score_threshold
            motion_threshold = g_dialog.motion_threshold
            pretrained_model = g_dialog.pretrained_model
            subject_names = g_dialog.subject_names
            behavior_names = g_dialog.behaviors
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
            out_nix_csv_file,
            zone_info=zone_info_json,
            score_threshold=score_threshold,
            motion_threshold=motion_threshold,
            pretrained_model=pretrained_model,
            subject_names=subject_names,
            behavior_names=behavior_names
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
    annolid_icon = QtGui.QIcon(
        str(Path(__file__).resolve().parent / "icons/icon_annolid.png"))
    app.setWindowIcon(annolid_icon)
    win = AnnolidWindow(config=config)

    win.show()
    win.raise_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
