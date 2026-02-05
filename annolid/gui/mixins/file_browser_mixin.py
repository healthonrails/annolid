from __future__ import annotations

import os
import os.path as osp

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from annolid.gui.label_file import LabelFile
from annolid.gui.window_base import AnnolidToolBar, utils


class FileBrowserMixin:
    """Toolbar creation and directory/image list workflows."""

    def toolbar(self, title, actions=None):
        toolbar = AnnolidToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QtCore.QSize(32, 32))
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

        extensions.append(".json")
        self.only_json_files = True

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = osp.join(root, file)
                    if self.only_json_files and not file.lower().endswith(".json"):
                        self.only_json_files = False
                    images.append(relativePath)
        images.sort(key=lambda x: x.lower())
        return images

    def _addItem(self, filename, label_file):
        item = QtWidgets.QListWidgetItem(filename)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        has_label = False
        if QtCore.QFile.exists(label_file):
            if LabelFile.is_label_file(label_file):
                has_label = True
        elif self._annotation_store_has_frame(label_file):
            has_label = True

        item.setCheckState(Qt.Checked)
        item.setData(Qt.UserRole, bool(has_label))
        if not has_label:
            item.setForeground(QtGui.QBrush(QtGui.QColor(160, 160, 160)))
        if not self.fileListWidget.findItems(filename, Qt.MatchExactly):
            self.fileListWidget.addItem(item)
            try:
                self._apply_file_search_filter()
            except Exception:
                pass

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
        self.imageList = []
        blocker = QtCore.QSignalBlocker(self.fileListWidget)
        try:
            self.fileListWidget.clear()
            for filename in self.scanAllImages(dirpath):
                if pattern and pattern not in filename:
                    continue
                label_file = self._getLabelFile(filename)

                if not filename.endswith(".json") or self.only_json_files:
                    self.imageList.append(filename)
                    self._addItem(filename, label_file)
        finally:
            del blocker
        self.openNextImg(load=load)
