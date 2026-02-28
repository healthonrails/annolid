from __future__ import annotations

import os
import os.path as osp
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from annolid.gui.label_file import LabelFile
from annolid.gui.window_base import AnnolidToolBar, utils


class _ImportDirScanWorker(QtCore.QObject):
    batchReady = QtCore.Signal(int, object)
    finished = QtCore.Signal(int)
    failed = QtCore.Signal(int, str)

    def __init__(
        self,
        *,
        token: int,
        dirpath: str,
        extensions: tuple[str, ...],
        pattern: str = "",
        batch_size: int = 300,
    ) -> None:
        super().__init__()
        self._token = int(token)
        self._dirpath = str(dirpath)
        self._extensions = tuple(extensions)
        self._pattern = str(pattern or "")
        self._batch_size = max(50, int(batch_size))
        self._stopped = False

    @QtCore.Slot()
    def stop(self) -> None:
        self._stopped = True

    @QtCore.Slot()
    def run(self) -> None:
        try:
            batch: list[str] = []
            for root, _dirs, files in os.walk(self._dirpath):
                if self._stopped:
                    break
                for file in files:
                    if self._stopped:
                        break
                    if not file.lower().endswith(self._extensions):
                        continue
                    filename = osp.join(root, file)
                    if self._pattern and self._pattern not in filename:
                        continue
                    batch.append(filename)
                    if len(batch) >= self._batch_size:
                        self.batchReady.emit(self._token, list(batch))
                        batch = []
            if batch and not self._stopped:
                self.batchReady.emit(self._token, list(batch))
            self.finished.emit(self._token)
        except Exception as exc:
            self.failed.emit(self._token, str(exc))


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
        known_paths = getattr(self, "_known_file_paths", None)
        if known_paths is None:
            known_paths = set()
            self._known_file_paths = known_paths
        if filename in known_paths:
            return

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
        self.fileListWidget.addItem(item)
        known_paths.add(filename)
        if apply_updates:
            try:
                self._apply_file_search_filter()
            except Exception:
                pass
            try:
                if hasattr(self, "_apply_file_dock_sort") and bool(
                    getattr(self, "_file_dock_sort_enabled", False)
                ):
                    self._apply_file_dock_sort()
            except Exception:
                pass

    @staticmethod
    def _resolve_pending_restore_candidate(
        pending_path: str, available_paths: list[str]
    ) -> str:
        pending = str(pending_path or "").strip()
        if not pending or not available_paths:
            return ""
        if pending in available_paths:
            return pending

        try:
            pending_obj = Path(pending)
            pending_stem = pending_obj.stem.lower()
            pending_suffix = pending_obj.suffix.lower()
        except Exception:
            return ""

        stem_matches = [
            p for p in available_paths if Path(p).stem.lower() == pending_stem
        ]
        if not stem_matches:
            return ""
        if pending_suffix == ".json":
            for path in stem_matches:
                if Path(path).suffix.lower() != ".json":
                    return path
        else:
            for path in stem_matches:
                if Path(path).suffix.lower() == ".json":
                    return path
        return stem_matches[0]

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
        self._known_file_paths = set()
        blocker = QtCore.QSignalBlocker(self.fileListWidget)
        try:
            self.fileListWidget.clear()
            if hasattr(self, "_reset_file_dock_incremental_state"):
                self._reset_file_dock_incremental_state()
        finally:
            del blocker
        if hasattr(self, "_update_file_selection_counter"):
            self._update_file_selection_counter()
        self._start_import_dir_scan(dirpath, pattern=pattern, load=load)

    def _cancel_import_dir_scan(self) -> None:
        current = int(getattr(self, "_dir_scan_token", 0))
        had_active_scan = bool(getattr(self, "_dir_scan_running", False))
        self._dir_scan_token = current + 1
        self._stop_import_dir_scan_worker()
        self._dir_scan_load = False
        self._dir_scan_first_loaded = False
        self._dir_scan_pending_restore = ""
        self._dir_scan_wait_for_pending_restore = False
        self._dir_scan_running = False
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
        self._dir_scan_running = True
        self._dir_scan_pending_restore = ""
        self._dir_scan_wait_for_pending_restore = False
        if bool(load):
            resolver = getattr(self, "_pending_last_worked_file_for_directory", None)
            if callable(resolver):
                try:
                    pending = str(resolver(str(dirpath or "")) or "")
                except Exception:
                    pending = ""
                self._dir_scan_pending_restore = pending
                self._dir_scan_wait_for_pending_restore = bool(pending)
        self._dir_scan_token = int(getattr(self, "_dir_scan_token", 0)) + 1
        if hasattr(self, "_set_file_scan_status"):
            self._set_file_scan_status(self.tr("Scanning..."), visible=True)
        self._start_import_dir_scan_worker(dirpath)

    def _start_import_dir_scan_worker(self, dirpath: str) -> None:
        self._stop_import_dir_scan_worker()
        token = int(getattr(self, "_dir_scan_token", 0))
        worker = _ImportDirScanWorker(
            token=token,
            dirpath=dirpath,
            extensions=tuple(getattr(self, "_dir_scan_extensions", ()) or ()),
            pattern=str(getattr(self, "_dir_scan_pattern", "") or ""),
            batch_size=320,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.batchReady.connect(self._on_import_dir_scan_batch)
        worker.finished.connect(self._on_import_dir_scan_finished)
        worker.failed.connect(self._on_import_dir_scan_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._dir_scan_worker = worker
        self._dir_scan_thread = thread
        thread.start()

    def _stop_import_dir_scan_worker(self) -> None:
        worker = getattr(self, "_dir_scan_worker", None)
        thread = getattr(self, "_dir_scan_thread", None)
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
        if thread is not None:
            try:
                thread.quit()
                thread.wait(250)
            except Exception:
                pass
        self._dir_scan_worker = None
        self._dir_scan_thread = None

    @QtCore.Slot(int, object)
    def _on_import_dir_scan_batch(self, token: int, paths_obj: object) -> None:
        if token != int(getattr(self, "_dir_scan_token", 0)):
            return
        if not bool(getattr(self, "_dir_scan_running", False)):
            return
        if not isinstance(paths_obj, list):
            return
        self._consume_import_dir_scan_paths(paths_obj)

    @QtCore.Slot(int)
    def _on_import_dir_scan_finished(self, token: int) -> None:
        if token != int(getattr(self, "_dir_scan_token", 0)):
            return
        if (
            bool(getattr(self, "_dir_scan_load", False))
            and not bool(getattr(self, "_dir_scan_first_loaded", False))
            and self.imageList
        ):
            self._dir_scan_first_loaded = True
            self.openNextImg(load=True)
        self._dir_scan_running = False
        self._dir_scan_worker = None
        self._dir_scan_thread = None
        self._dir_scan_pending_restore = ""
        self._dir_scan_wait_for_pending_restore = False
        count = len(self.imageList)
        if hasattr(self, "_set_file_scan_idle"):
            self._set_file_scan_idle(count=count, hide=False)
        self.statusBar().showMessage(
            self.tr("Loaded %1 files from %2")
            .replace("%1", str(count))
            .replace("%2", str(self.lastOpenDir or "")),
            2500,
        )

    @QtCore.Slot(int, str)
    def _on_import_dir_scan_failed(self, token: int, error_text: str) -> None:
        if token != int(getattr(self, "_dir_scan_token", 0)):
            return
        self._dir_scan_running = False
        self._dir_scan_worker = None
        self._dir_scan_thread = None
        self._dir_scan_pending_restore = ""
        self._dir_scan_wait_for_pending_restore = False
        if hasattr(self, "_set_file_scan_status"):
            self._set_file_scan_status(
                self.tr("Scan error: %1").replace("%1", str(error_text or "unknown")),
                visible=True,
            )
        self.statusBar().showMessage(
            self.tr("Directory scan failed: %1").replace(
                "%1", str(error_text or "unknown")
            ),
            4000,
        )

    def _consume_import_dir_scan_paths(self, paths: list[str]) -> None:
        loaded_entries = []
        for filename in paths:
            if self.only_json_files and not str(filename).lower().endswith(".json"):
                self.only_json_files = False
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
            pending_for_dir = str(getattr(self, "_dir_scan_pending_restore", "") or "")
            restore_candidate = self._resolve_pending_restore_candidate(
                pending_for_dir, self.imageList
            )
            if restore_candidate:
                self._dir_scan_first_loaded = True
                self._dir_scan_pending_restore = ""
                self._dir_scan_wait_for_pending_restore = False
                self.loadFile(restore_candidate)
                self._set_current_file_item(restore_candidate)
                if self.caption_widget is not None:
                    self.caption_widget.set_image_path(restore_candidate)
                clear_pending = getattr(self, "_clear_pending_last_worked_file", None)
                if callable(clear_pending):
                    try:
                        clear_pending()
                    except Exception:
                        pass
                return
            if bool(getattr(self, "_dir_scan_wait_for_pending_restore", False)):
                return
            self._dir_scan_first_loaded = True
            self.openNextImg(load=True)
