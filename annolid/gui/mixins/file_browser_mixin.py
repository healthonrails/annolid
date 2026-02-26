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

    def _addItem(self, filename, label_file, *, apply_updates=True):
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
            if apply_updates:
                try:
                    self._apply_file_search_filter()
                except Exception:
                    pass
                try:
                    if hasattr(self, "_apply_file_dock_sort"):
                        self._apply_file_dock_sort()
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

        self._cancel_import_dir_scan()
        self.lastOpenDir = dirpath
        self.annotation_dir = dirpath
        self.filename = None
        self.imageList = []
        blocker = QtCore.QSignalBlocker(self.fileListWidget)
        try:
            self.fileListWidget.clear()
            if hasattr(self, "_reset_file_dock_incremental_state"):
                self._reset_file_dock_incremental_state()
        finally:
            del blocker
        self._start_import_dir_scan(dirpath, pattern=pattern, load=load)

    def _cancel_import_dir_scan(self) -> None:
        current = int(getattr(self, "_dir_scan_token", 0))
        had_active_scan = getattr(self, "_dir_scan_file_iter", None) is not None
        self._dir_scan_token = current + 1
        self._dir_scan_file_iter = None
        self._dir_scan_load = False
        self._dir_scan_first_loaded = False
        if had_active_scan and hasattr(self, "_set_file_scan_status"):
            self._set_file_scan_status(self.tr("Scan canceled"), visible=True)

    def _start_import_dir_scan(self, dirpath, *, pattern=None, load=True) -> None:
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        extensions.append(".json")
        self.only_json_files = True
        self._dir_scan_extensions = tuple(extensions)
        self._dir_scan_pattern = str(pattern) if pattern else ""
        self._dir_scan_load = bool(load)
        self._dir_scan_first_loaded = False
        self._dir_scan_loaded_count = 0
        self._dir_scan_file_iter = (
            osp.join(root, file)
            for root, _dirs, files in os.walk(dirpath)
            for file in files
            if file.lower().endswith(self._dir_scan_extensions)
        )
        self._dir_scan_token = int(getattr(self, "_dir_scan_token", 0)) + 1
        if hasattr(self, "_set_file_scan_status"):
            self._set_file_scan_status(self.tr("Scanning..."), visible=True)
        token = self._dir_scan_token
        QtCore.QTimer.singleShot(0, lambda: self._process_import_dir_scan_chunk(token))

    def _process_import_dir_scan_chunk(self, token: int) -> None:
        if token != int(getattr(self, "_dir_scan_token", 0)):
            return
        file_iter = getattr(self, "_dir_scan_file_iter", None)
        if file_iter is None:
            return
        batch_limit = 1500
        loaded_entries = []
        processed = 0
        for _ in range(batch_limit):
            try:
                filename = next(file_iter)
            except StopIteration:
                self._dir_scan_file_iter = None
                break
            processed += 1
            if self.only_json_files and not str(filename).lower().endswith(".json"):
                self.only_json_files = False
                # Match previous behavior: if the folder has real images, do not
                # keep JSON files in the dock list.
                self.imageList = [
                    path
                    for path in self.imageList
                    if not str(path).lower().endswith(".json")
                ]
                if hasattr(self, "_remove_json_items_from_file_dock"):
                    self._remove_json_items_from_file_dock()
                loaded_entries = [
                    (path, label)
                    for path, label in loaded_entries
                    if not str(path).lower().endswith(".json")
                ]

            pattern = str(getattr(self, "_dir_scan_pattern", "") or "")
            if pattern and pattern not in filename:
                continue
            if str(filename).lower().endswith(".json") and not self.only_json_files:
                continue

            label_file = self._getLabelFile(filename)
            self.imageList.append(filename)
            loaded_entries.append((filename, label_file))

        if loaded_entries:
            if hasattr(self, "_append_file_dock_entries"):
                self._append_file_dock_entries(loaded_entries)
            else:
                for filename, label_file in loaded_entries:
                    self._addItem(filename, label_file)
        self._dir_scan_loaded_count = int(
            getattr(self, "_dir_scan_loaded_count", 0)
        ) + len(loaded_entries)
        if hasattr(self, "_update_file_scan_progress"):
            self._update_file_scan_progress(
                int(getattr(self, "_dir_scan_loaded_count", 0))
            )

        if (
            bool(getattr(self, "_dir_scan_load", False))
            and not bool(getattr(self, "_dir_scan_first_loaded", False))
            and self.imageList
        ):
            self._dir_scan_first_loaded = True
            self.openNextImg(load=True)

        if getattr(self, "_dir_scan_file_iter", None) is not None:
            QtCore.QTimer.singleShot(
                0, lambda: self._process_import_dir_scan_chunk(token)
            )
        else:
            count = len(self.imageList)
            if hasattr(self, "_set_file_scan_idle"):
                self._set_file_scan_idle(count=count, hide=False)
            self.statusBar().showMessage(
                self.tr("Loaded %1 files from %2")
                .replace("%1", str(count))
                .replace("%2", str(self.lastOpenDir or "")),
                2500,
            )
