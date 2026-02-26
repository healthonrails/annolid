from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy import QtCore, QtGui, QtWidgets


class FileDockMixin:
    """Files dock UI and file-management interactions."""

    def _init_file_dock_ui(self) -> None:
        self._file_dock_pending_entries: list[tuple[str, str]] = []
        self._file_dock_batch_size = 200
        self._file_dock_loading = False
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.fileSearchWidget = QtWidgets.QLineEdit()
        self.fileSearchWidget.setClearButtonEnabled(True)
        self.fileSearchWidget.setPlaceholderText(self.tr("Search files..."))
        self.fileSearchWidget.textChanged.connect(
            lambda _text: self._apply_file_search_filter()
        )
        self.fileSortCombo = QtWidgets.QComboBox()
        self.fileSortCombo.addItem(self.tr("Name"), "name")
        self.fileSortCombo.addItem(self.tr("Date"), "date")
        self.fileSortCombo.addItem(self.tr("Size"), "size")
        self.fileSortCombo.addItem(self.tr("Type"), "type")
        self.fileSortCombo.setToolTip(self.tr("Sort files by selected field"))
        self.fileSortCombo.currentIndexChanged.connect(
            lambda _idx: self._apply_file_dock_sort()
        )
        self.fileSortOrderButton = QtWidgets.QToolButton()
        self.fileSortOrderButton.setCheckable(True)
        self.fileSortOrderButton.setChecked(False)
        self.fileSortOrderButton.setToolTip(self.tr("Toggle descending sort"))
        self.fileSortOrderButton.setText(self.tr("Asc"))
        self.fileSortOrderButton.toggled.connect(self._on_file_sort_order_toggled)
        self.fileScanStatusLabel = QtWidgets.QLabel(self.tr("Idle"))
        self.fileScanStatusLabel.setObjectName("fileScanStatusLabel")
        self.fileScanStatusLabel.setStyleSheet("color: rgb(140, 140, 140);")
        self.fileScanStatusLabel.setVisible(False)

        self.file_dock = QtWidgets.QDockWidget(self.tr("Files"), self)
        self.file_dock.setObjectName("fileDock")
        file_container = QtWidgets.QWidget(self.file_dock)
        file_layout = QtWidgets.QVBoxLayout(file_container)
        file_layout.setContentsMargins(6, 6, 6, 6)
        file_layout.setSpacing(4)
        search_row = QtWidgets.QHBoxLayout()
        search_row.setContentsMargins(0, 0, 0, 0)
        search_row.setSpacing(4)
        search_row.addWidget(self.fileSearchWidget, 1)
        search_row.addWidget(self.fileSortCombo, 0)
        search_row.addWidget(self.fileSortOrderButton, 0)
        search_row.addWidget(self.fileScanStatusLabel, 0)
        file_layout.addLayout(search_row)
        file_actions_row = QtWidgets.QHBoxLayout()
        file_actions_row.setContentsMargins(0, 0, 0, 0)
        file_actions_row.setSpacing(4)
        self.fileOpenButton = QtWidgets.QToolButton(file_container)
        self.fileOpenButton.setText(self.tr("Open"))
        self.fileOpenButton.setToolTip(self.tr("Open selected file"))
        self.fileOpenButton.clicked.connect(self._open_selected_file_from_dock)
        file_actions_row.addWidget(self.fileOpenButton)
        self.fileRenameButton = QtWidgets.QToolButton(file_container)
        self.fileRenameButton.setText(self.tr("Rename"))
        self.fileRenameButton.setToolTip(self.tr("Rename selected file"))
        self.fileRenameButton.clicked.connect(self._rename_selected_file_from_dock)
        file_actions_row.addWidget(self.fileRenameButton)
        self.fileDeleteButton = QtWidgets.QToolButton(file_container)
        self.fileDeleteButton.setText(self.tr("Delete"))
        self.fileDeleteButton.setToolTip(self.tr("Delete selected file"))
        self.fileDeleteButton.clicked.connect(self._delete_selected_file_from_dock)
        file_actions_row.addWidget(self.fileDeleteButton)
        self.fileRevealButton = QtWidgets.QToolButton(file_container)
        self.fileRevealButton.setText(self.tr("Reveal"))
        self.fileRevealButton.setToolTip(self.tr("Open containing folder"))
        self.fileRevealButton.clicked.connect(self._reveal_selected_file_in_folder)
        file_actions_row.addWidget(self.fileRevealButton)
        self.fileRefreshButton = QtWidgets.QToolButton(file_container)
        self.fileRefreshButton.setText(self.tr("Refresh"))
        self.fileRefreshButton.setToolTip(
            self.tr("Refresh file list from current folder")
        )
        self.fileRefreshButton.clicked.connect(self._refresh_file_dock_listing)
        file_actions_row.addWidget(self.fileRefreshButton)
        file_actions_row.addStretch(1)
        file_layout.addLayout(file_actions_row)
        file_layout.addWidget(self.fileListWidget)
        self.file_dock.setWidget(file_container)

        self.fileListWidget.customContextMenuRequested.connect(
            self._show_file_dock_context_menu
        )
        self.fileListWidget.itemDoubleClicked.connect(
            self._on_file_dock_item_double_clicked
        )
        self.fileListWidget.verticalScrollBar().valueChanged.connect(
            self._on_file_list_scroll_value_changed
        )
        self._setup_file_dock_shortcuts()

    def _set_file_scan_status(self, text: str, *, visible: bool = True) -> None:
        label = getattr(self, "fileScanStatusLabel", None)
        if label is None:
            return
        label.setText(str(text or ""))
        label.setVisible(bool(visible))

    def _set_file_scan_idle(
        self, *, count: Optional[int] = None, hide: bool = False
    ) -> None:
        if hide:
            self._set_file_scan_status("", visible=False)
            return
        if count is None:
            self._set_file_scan_status(self.tr("Idle"), visible=True)
            return
        self._set_file_scan_status(
            self.tr("Loaded %1").replace("%1", str(int(max(0, count)))),
            visible=True,
        )

    def _update_file_scan_progress(self, loaded_count: int) -> None:
        self._set_file_scan_status(
            self.tr("Scanning: %1").replace("%1", str(int(max(0, loaded_count)))),
            visible=True,
        )

    def _setup_file_dock_shortcuts(self) -> None:
        # Scoped to the Files dock widget so they do not conflict with canvas/label
        # shortcuts when focus is elsewhere.
        rename_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_F2), self.fileListWidget
        )
        rename_shortcut.setContext(QtCore.Qt.WidgetShortcut)
        rename_shortcut.activated.connect(self._rename_selected_file_from_dock)

        delete_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Delete), self.fileListWidget
        )
        delete_shortcut.setContext(QtCore.Qt.WidgetShortcut)
        delete_shortcut.activated.connect(self._delete_selected_file_from_dock)

        self._file_dock_shortcuts = [rename_shortcut, delete_shortcut]

    def _apply_file_search_filter(self) -> None:
        """Filter file dock list items based on the current search text."""
        try:
            query = str(self.fileSearchWidget.text() or "").strip().lower()
        except Exception:
            query = ""
        for idx in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(idx)
            if item is None:
                continue
            if not query:
                item.setHidden(False)
                continue
            try:
                hay = str(item.text() or "").lower()
            except Exception:
                hay = ""
            item.setHidden(query not in hay)

    def _on_file_sort_order_toggled(self, checked: bool) -> None:
        self.fileSortOrderButton.setText(self.tr("Desc") if checked else self.tr("Asc"))
        self._apply_file_dock_sort()

    def _file_sort_mode(self) -> str:
        mode = self.fileSortCombo.currentData()
        text = str(mode or "").strip().lower()
        return text or "name"

    def _path_sort_key(self, path: Optional[Path], mode: str, fallback_text: str):
        display_text = str(fallback_text or "")
        if mode == "date":
            try:
                mtime = (
                    path.stat().st_mtime if path is not None and path.exists() else 0.0
                )
            except Exception:
                mtime = 0.0
            return (mtime, display_text.lower())
        if mode == "size":
            try:
                size = path.stat().st_size if path is not None and path.exists() else -1
            except Exception:
                size = -1
            return (size, display_text.lower())
        if mode == "type":
            suffix = str(path.suffix or "").lower() if path is not None else ""
            return (suffix, display_text.lower())
        # default name
        name = path.name if path is not None else display_text
        return (str(name).lower(), str(path or display_text).lower())

    def _item_sort_key(self, item, mode: str):
        path = self._item_path_from_file_dock(item)
        display_text = str(item.text() or "")
        role_type = str(item.data(QtCore.Qt.UserRole + 1) or "")
        if mode == "date":
            try:
                mtime = (
                    path.stat().st_mtime if path is not None and path.exists() else 0.0
                )
            except Exception:
                mtime = 0.0
            return (mtime, display_text.lower())
        if mode == "size":
            try:
                size = path.stat().st_size if path is not None and path.exists() else -1
            except Exception:
                size = -1
            return (size, display_text.lower())
        if mode == "type":
            suffix = ""
            if role_type:
                suffix = role_type.lower()
            elif path is not None:
                suffix = str(path.suffix or "").lower()
            return (suffix, display_text.lower())
        return self._path_sort_key(path, mode, display_text)

    def _apply_file_dock_sort(self) -> None:
        mode = self._file_sort_mode()
        descending = bool(self.fileSortOrderButton.isChecked())
        if self._file_dock_pending_entries:
            self._file_dock_pending_entries.sort(
                key=lambda entry: self._path_sort_key(
                    Path(entry[0]).expanduser(), mode, entry[0]
                ),
                reverse=descending,
            )
        count = self.fileListWidget.count()
        if count <= 1:
            self._apply_file_search_filter()
            return
        current = self._selected_file_dock_item()
        current_path = str(self._item_path_from_file_dock(current) or "")
        items = [self.fileListWidget.takeItem(0) for _ in range(count)]
        items = [it for it in items if it is not None]
        items.sort(key=lambda it: self._item_sort_key(it, mode), reverse=descending)
        for item in items:
            self.fileListWidget.addItem(item)
        if current_path:
            for idx in range(self.fileListWidget.count()):
                item = self.fileListWidget.item(idx)
                if item is None:
                    continue
                if str(self._item_path_from_file_dock(item) or "") == current_path:
                    self.fileListWidget.setCurrentItem(item)
                    break
        self._apply_file_search_filter()

    def _reset_file_dock_incremental_state(self) -> None:
        self._file_dock_pending_entries = []
        self._file_dock_loading = False

    def _append_file_dock_entries(self, entries: list[tuple[str, str]]) -> None:
        if not entries:
            return
        self._file_dock_pending_entries.extend(list(entries))
        self._apply_file_dock_sort()
        if self.fileListWidget.count() == 0:
            self._load_more_file_dock_items()

    def _remove_json_items_from_file_dock(self) -> None:
        # Used when a directory contains real images: remove JSON-only entries.
        for idx in range(self.fileListWidget.count() - 1, -1, -1):
            item = self.fileListWidget.item(idx)
            if item is None:
                continue
            path = self._item_path_from_file_dock(item)
            if path is None or path.suffix.lower() != ".json":
                continue
            removed = self.fileListWidget.takeItem(idx)
            if removed is not None:
                del removed
        if self._file_dock_pending_entries:
            self._file_dock_pending_entries = [
                (filename, label_file)
                for filename, label_file in self._file_dock_pending_entries
                if Path(filename).suffix.lower() != ".json"
            ]

    def _queue_file_dock_entries(self, entries: list[tuple[str, str]]) -> None:
        self._reset_file_dock_incremental_state()
        self._file_dock_pending_entries = list(entries or [])
        self._apply_file_dock_sort()
        self._load_more_file_dock_items(batch_size=max(self._file_dock_batch_size, 300))

    def _load_more_file_dock_items(self, batch_size: Optional[int] = None) -> None:
        if self._file_dock_loading:
            return
        if not self._file_dock_pending_entries:
            return
        self._file_dock_loading = True
        try:
            limit = max(1, int(batch_size or self._file_dock_batch_size))
            loaded = 0
            while self._file_dock_pending_entries and loaded < limit:
                filename, label_file = self._file_dock_pending_entries.pop(0)
                if hasattr(self, "_addItem"):
                    self._addItem(filename, label_file, apply_updates=False)
                loaded += 1
            self._apply_file_search_filter()
        finally:
            self._file_dock_loading = False

        pending = len(self._file_dock_pending_entries)
        if pending > 0:
            self.statusBar().showMessage(
                self.tr("Loaded %1 files. Scroll to load %2 more.")
                .replace("%1", str(self.fileListWidget.count()))
                .replace("%2", str(pending)),
                2500,
            )
        else:
            self.statusBar().showMessage(
                self.tr("Loaded %1 files.").replace(
                    "%1", str(self.fileListWidget.count())
                ),
                1500,
            )

    def _on_file_list_scroll_value_changed(self, value: int) -> None:
        if not self._file_dock_pending_entries:
            return
        bar = self.fileListWidget.verticalScrollBar()
        if bar.maximum() <= 0:
            self._load_more_file_dock_items()
            return
        # Start loading next chunk before reaching absolute end to keep scrolling smooth.
        if value >= max(0, bar.maximum() - 10):
            self._load_more_file_dock_items()

    def _selected_file_dock_item(self):
        return self.fileListWidget.currentItem()

    def _item_path_from_file_dock(self, item) -> Optional[Path]:
        if item is None:
            return None
        raw = item.data(QtCore.Qt.UserRole)
        role_type = item.data(QtCore.Qt.UserRole + 1)
        if role_type == "pdf" and raw:
            text = str(raw).strip()
        elif isinstance(raw, str) and raw:
            text = str(raw).strip()
        else:
            text = str(item.text() or "").strip()
        if not text:
            return None
        try:
            return Path(text).expanduser()
        except Exception:
            return None

    def _show_file_dock_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.fileListWidget.itemAt(pos)
        menu = QtWidgets.QMenu(self.fileListWidget)
        open_action = menu.addAction(self.tr("Open"))
        rename_action = menu.addAction(self.tr("Rename..."))
        delete_action = menu.addAction(self.tr("Delete..."))
        reveal_action = menu.addAction(self.tr("Reveal in Folder"))
        menu.addSeparator()
        sort_by_name = menu.addAction(self.tr("Sort by Name"))
        sort_by_date = menu.addAction(self.tr("Sort by Date"))
        sort_by_size = menu.addAction(self.tr("Sort by Size"))
        sort_by_type = menu.addAction(self.tr("Sort by Type"))
        toggle_order = menu.addAction(
            self.tr("Sort Descending")
            if not self.fileSortOrderButton.isChecked()
            else self.tr("Sort Ascending")
        )
        menu.addSeparator()
        refresh_action = menu.addAction(self.tr("Refresh"))
        if item is None:
            open_action.setEnabled(False)
            rename_action.setEnabled(False)
            delete_action.setEnabled(False)
            reveal_action.setEnabled(False)
        action = menu.exec_(self.fileListWidget.mapToGlobal(pos))
        if action is None:
            return
        if action is open_action:
            self._open_selected_file_from_dock(item)
        elif action is rename_action:
            self._rename_selected_file_from_dock(item)
        elif action is delete_action:
            self._delete_selected_file_from_dock(item)
        elif action is reveal_action:
            self._reveal_selected_file_in_folder(item)
        elif action is sort_by_name:
            self.fileSortCombo.setCurrentIndex(self.fileSortCombo.findData("name"))
        elif action is sort_by_date:
            self.fileSortCombo.setCurrentIndex(self.fileSortCombo.findData("date"))
        elif action is sort_by_size:
            self.fileSortCombo.setCurrentIndex(self.fileSortCombo.findData("size"))
        elif action is sort_by_type:
            self.fileSortCombo.setCurrentIndex(self.fileSortCombo.findData("type"))
        elif action is toggle_order:
            self.fileSortOrderButton.setChecked(
                not self.fileSortOrderButton.isChecked()
            )
        elif action is refresh_action:
            self._refresh_file_dock_listing()

    def _on_file_dock_item_double_clicked(self, item) -> None:
        self._open_selected_file_from_dock(item)

    def _open_selected_file_from_dock(self, item=None) -> None:
        selected = item or self._selected_file_dock_item()
        path = self._item_path_from_file_dock(selected)
        if path is None:
            return
        path_text = str(path)
        if not path.exists():
            self.statusBar().showMessage(self.tr("Missing file: %s") % path_text, 4000)
            return
        role_type = (
            selected.data(QtCore.Qt.UserRole + 1) if selected is not None else ""
        )
        if role_type == "pdf":
            try:
                if hasattr(self, "_pdf_manager") and self._pdf_manager is not None:
                    self._pdf_manager.show_pdf_in_viewer(path_text)
                    return
            except Exception:
                pass
        self.loadFile(path_text)

    def _rename_selected_file_from_dock(self, item=None) -> None:
        selected = item or self._selected_file_dock_item()
        path = self._item_path_from_file_dock(selected)
        if path is None:
            return
        if not path.exists():
            self.statusBar().showMessage(self.tr("Missing file: %s") % str(path), 4000)
            return
        current_name = path.name
        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            self.tr("Rename File"),
            self.tr("New file name:"),
            text=current_name,
        )
        if not ok:
            return
        new_name = str(new_name or "").strip()
        if not new_name or new_name == current_name:
            return
        if Path(new_name).name != new_name:
            self.errorMessage(
                self.tr("Rename Error"),
                self.tr("Please provide a file name without path separators."),
            )
            return
        target = path.with_name(new_name)
        if target.exists():
            self.errorMessage(
                self.tr("Rename Error"),
                self.tr("Target already exists: %s") % str(target),
            )
            return
        try:
            path.rename(target)
        except Exception as exc:
            self.errorMessage(self.tr("Rename Error"), str(exc))
            return
        if selected is not None:
            role_type = selected.data(QtCore.Qt.UserRole + 1)
            if role_type == "pdf":
                selected.setText(target.name)
                selected.setData(QtCore.Qt.UserRole, str(target))
                selected.setToolTip(str(target))
            else:
                selected.setText(str(target))
                selected.setToolTip(str(target))
        self._apply_file_dock_sort()
        old_text = str(path)
        new_text = str(target)
        self.imageList = [new_text if p == old_text else p for p in self.imageList]
        if self.filename == old_text:
            self.filename = new_text
            self.setWindowTitle(f"Annolid - {Path(self.filename).name}")
        self.statusBar().showMessage(
            self.tr("Renamed %s -> %s") % (current_name, target.name), 3000
        )

    def _delete_selected_file_from_dock(self, item=None) -> None:
        selected = item or self._selected_file_dock_item()
        path = self._item_path_from_file_dock(selected)
        if path is None:
            return
        if not path.exists():
            self.statusBar().showMessage(self.tr("Missing file: %s") % str(path), 4000)
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Delete File"),
            self.tr("Delete this file?\n%1").replace("%1", str(path)),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        deleting_current = self.filename == str(path)
        if deleting_current and hasattr(self, "mayContinue") and not self.mayContinue():
            return
        try:
            path.unlink()
        except Exception as exc:
            self.errorMessage(self.tr("Delete Error"), str(exc))
            return
        row = self.fileListWidget.row(selected) if selected is not None else -1
        if selected is not None and row >= 0:
            removed = self.fileListWidget.takeItem(row)
            if removed is not None:
                del removed
        path_text = str(path)
        self.imageList = [p for p in self.imageList if p != path_text]
        if deleting_current:
            self.resetState()
            self.filename = None
            self.imagePath = None
            self.setWindowTitle("Annolid")
        self._apply_file_dock_sort()
        self.statusBar().showMessage(self.tr("Deleted: %s") % path.name, 3000)

    def _reveal_selected_file_in_folder(self, item=None) -> None:
        selected = item or self._selected_file_dock_item()
        path = self._item_path_from_file_dock(selected)
        if path is None:
            return
        target_dir = path.parent if path.parent.exists() else path
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(target_dir.resolve()))
        )

    def _refresh_file_dock_listing(self) -> None:
        if not self.lastOpenDir or not hasattr(self, "importDirImages"):
            return
        current = self.filename
        self.importDirImages(self.lastOpenDir, load=False)
        self._apply_file_dock_sort()
        if current:
            matches = self.fileListWidget.findItems(current, QtCore.Qt.MatchExactly)
            if matches:
                self.fileListWidget.setCurrentItem(matches[0])
        self.statusBar().showMessage(self.tr("File list refreshed"), 2000)
