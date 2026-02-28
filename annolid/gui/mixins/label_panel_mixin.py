from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    keypoint_visibility_from_shape_object,
    set_keypoint_visibility_on_shape_object,
)
from annolid.gui.shape import Shape
from annolid.gui.window_base import AnnolidLabelListItem


class LabelPanelMixin:
    """Label/file list synchronization and keypoint visibility controls."""

    def _resolve_pdf_manager(self):
        manager = getattr(self, "pdf_manager", None)
        if manager is not None:
            return manager
        return getattr(self, "_pdf_manager", None)

    def _rebuild_unique_label_list(self) -> None:
        selected = set()
        try:
            for it in self.uniqLabelList.selectedItems():
                val = it.data(QtCore.Qt.UserRole)
                if val:
                    selected.add(str(val))
        except Exception:
            selected = set()

        counts: dict[str, int] = {}
        for shape in getattr(self.canvas, "shapes", []) or []:
            label = str(getattr(shape, "label", "") or "").strip()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1

        self.uniqLabelList.blockSignals(True)
        try:
            self.uniqLabelList.clear()
            for label in sorted(counts.keys(), key=lambda s: s.lower()):
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                count = counts[label]
                display = f"{label} [{count}]"
                self.uniqLabelList.setItemLabel(item, display, rgb)
                if label in selected:
                    item.setSelected(True)
        finally:
            self.uniqLabelList.blockSignals(False)

    def _setup_label_list_connections(self) -> None:
        if getattr(self, "_label_list_connections_setup", False):
            return
        self._label_list_connections_setup = True

        def on_selection_changed() -> None:
            if getattr(self, "_noSelectionSlot", False):
                return
            shapes = []
            for it in self.labelList.selectedItems():
                try:
                    shape = it.shape()
                except Exception:
                    shape = None
                if shape is not None:
                    shapes.append(shape)
            try:
                self._noSelectionSlot = True
                self.canvas.selectShapes(shapes)
            finally:
                self._noSelectionSlot = False

        try:
            self.labelList.itemSelectionChanged.disconnect()
        except Exception:
            pass
        self.labelList.itemSelectionChanged.connect(on_selection_changed)

        try:
            self.labelList.itemDoubleClicked.disconnect()
        except Exception:
            pass
        self.labelList.itemDoubleClicked.connect(self.editLabel)

        def on_shape_visibility_changed(shape: Shape, visible: bool) -> None:
            if shape is None:
                return
            try:
                shape.visible = bool(visible)
            except Exception:
                pass
            try:
                self.canvas.setShapeVisible(shape, bool(visible))
            except Exception:
                try:
                    self.canvas.update()
                except Exception:
                    pass
            try:
                self.setDirty()
            except Exception:
                pass

        def on_shape_delete_requested(shape: Shape) -> None:
            if shape is None:
                return
            try:
                self._noSelectionSlot = True
                self.canvas.selectShapes([shape])
                # Keep the label list selection aligned so subsequent actions
                # (e.g. Delete key) behave predictably.
                for idx in range(self.labelList.count()):
                    it = self.labelList.item(idx)
                    try:
                        if isinstance(it, AnnolidLabelListItem) and it.shape() is shape:
                            it.setSelected(True)
                        elif it is not None:
                            it.setSelected(False)
                    except Exception:
                        continue
            finally:
                self._noSelectionSlot = False
            try:
                self.deleteSelectedShapes()
            except Exception:
                pass

        def on_shapes_delete_requested(shapes: list[Shape]) -> None:
            shape_list = [s for s in (shapes or []) if s is not None]
            if not shape_list:
                return
            try:
                self._noSelectionSlot = True
                self.canvas.selectShapes(shape_list)
                selected_ids = {id(s) for s in shape_list}
                for idx in range(self.labelList.count()):
                    it = self.labelList.item(idx)
                    try:
                        shape = (
                            it.shape() if isinstance(it, AnnolidLabelListItem) else None
                        )
                        it.setSelected(shape is not None and id(shape) in selected_ids)
                    except Exception:
                        continue
            finally:
                self._noSelectionSlot = False
            try:
                self.deleteSelectedShapes()
            except Exception:
                pass

        try:
            self.labelList.shapeVisibilityChanged.disconnect()
        except Exception:
            pass
        try:
            self.labelList.shapeDeleteRequested.disconnect()
        except Exception:
            pass
        try:
            self.labelList.shapesDeleteRequested.disconnect()
        except Exception:
            pass
        try:
            self.labelList.shapeVisibilityChanged.connect(on_shape_visibility_changed)
            self.labelList.shapeDeleteRequested.connect(on_shape_delete_requested)
            self.labelList.shapesDeleteRequested.connect(on_shapes_delete_requested)
        except Exception:
            # If the list widget doesn't expose these signals, ignore.
            pass

        # Label Instances dock shortcuts:
        # - Select all instances in the list
        # - Delete selected instances directly from the dock
        if not getattr(self, "_label_list_shortcuts_setup", False):
            self._label_list_shortcuts_setup = True
            self._label_list_shortcuts = []

            def _delete_selected_from_label_list() -> None:
                if self.labelList.selectedItems():
                    self.deleteSelectedShapes()

            delete_seqs = (
                QtGui.QKeySequence(Qt.Key_Delete),
                QtGui.QKeySequence(Qt.Key_Backspace),
                QtGui.QKeySequence("Meta+Backspace"),
            )
            for seq in delete_seqs:
                shortcut = QtWidgets.QShortcut(seq, self.labelList)
                shortcut.setContext(Qt.WidgetWithChildrenShortcut)
                shortcut.activated.connect(_delete_selected_from_label_list)
                self._label_list_shortcuts.append(shortcut)

            select_all = QtWidgets.QShortcut(
                QtGui.QKeySequence.SelectAll, self.labelList
            )
            select_all.setContext(Qt.WidgetWithChildrenShortcut)
            select_all.activated.connect(self.labelList.selectAll)
            self._label_list_shortcuts.append(select_all)

    def _setup_file_list_connections(self) -> None:
        if getattr(self, "_file_list_connections_setup", False):
            return
        self._file_list_connections_setup = True

        self.fileListWidget.currentItemChanged.connect(
            self._on_file_list_current_item_changed
        )

        self.fileListWidget.itemChanged.connect(self._on_file_list_item_changed)

    def _checked_file_paths(self) -> list[str]:
        checked: list[str] = []
        for idx in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(idx)
            if item is None:
                continue
            if item.isHidden():
                continue
            if item.checkState() != Qt.Unchecked:
                checked.append(item.text())
        return checked

    def _try_open_pdf_from_file_list_path(self, path_text: str) -> bool:
        path = Path(str(path_text or "").strip())
        if str(path.suffix or "").lower() != ".pdf":
            return False
        try:
            manager = self._resolve_pdf_manager()
            if manager is not None:
                manager.show_pdf_in_viewer(str(path))
                return True
        except Exception:
            return False
        return False

    def _set_current_file_item(self, path: str) -> None:
        if not path:
            return
        matches = self.fileListWidget.findItems(path, Qt.MatchExactly)
        if not matches:
            return
        current = self.fileListWidget.currentItem()
        if current is matches[0]:
            return
        blocker = QtCore.QSignalBlocker(self.fileListWidget)
        try:
            self.fileListWidget.setCurrentItem(matches[0])
            self.fileListWidget.scrollToItem(
                matches[0], QtWidgets.QAbstractItemView.PositionAtCenter
            )
        finally:
            del blocker
        try:
            if hasattr(self, "_update_file_selection_counter"):
                self._update_file_selection_counter()
        except Exception:
            pass

    def _nearest_checked_file_path(self, row: int) -> str | None:
        count = self.fileListWidget.count()
        if count <= 0:
            return None

        for idx in range(max(0, row), count):
            it = self.fileListWidget.item(idx)
            if it is not None and not it.isHidden() and it.checkState() != Qt.Unchecked:
                return it.text()
        for idx in range(min(row - 1, count - 1), -1, -1):
            it = self.fileListWidget.item(idx)
            if it is not None and not it.isHidden() and it.checkState() != Qt.Unchecked:
                return it.text()
        return None

    def _on_file_list_current_item_changed(self, current, previous) -> None:
        try:
            if hasattr(self, "_update_file_selection_counter"):
                self._update_file_selection_counter()
        except Exception:
            pass
        if current is None:
            return
        if self.video_loader is not None:
            return

        path = current.text()
        if not path:
            return
        if current.checkState() == Qt.Unchecked:
            if self.filename is not None and not self.mayContinue():
                blocker = QtCore.QSignalBlocker(self.fileListWidget)
                try:
                    self.fileListWidget.setCurrentItem(previous)
                finally:
                    del blocker
                return
            self.resetState()
            self.setWindowTitle("Annolid")
            return
        if self.filename == path:
            return

        if not self.mayContinue():
            blocker = QtCore.QSignalBlocker(self.fileListWidget)
            try:
                self.fileListWidget.setCurrentItem(previous)
            finally:
                del blocker
            return

        if self._try_open_pdf_from_file_list_path(path):
            return
        self.loadFile(path)
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(self.filename)
        self._update_frame_display_and_emit_update()

    def _on_file_list_item_changed(self, item) -> None:
        if item is None:
            return
        if self.video_loader is not None:
            return
        path = item.text()
        if not path:
            return

        if item.checkState() == Qt.Unchecked:
            if self.filename == path:
                fallback_path = self._nearest_checked_file_path(
                    self.fileListWidget.row(item)
                )
                if fallback_path:
                    self._set_current_file_item(fallback_path)
                    if not self._try_open_pdf_from_file_list_path(fallback_path):
                        self.loadFile(fallback_path)
                        if self.caption_widget is not None:
                            self.caption_widget.set_image_path(self.filename)
                        self._update_frame_display_and_emit_update()
                else:
                    self.resetState()
                    self.setWindowTitle("Annolid")
            return

        if (
            item.checkState() != Qt.Unchecked
            and self.fileListWidget.currentItem() is item
        ):
            if not self.mayContinue():
                return
            if self._try_open_pdf_from_file_list_path(path):
                return
            self.loadFile(path)
            if self.caption_widget is not None:
                self.caption_widget.set_image_path(self.filename)
            self._update_frame_display_and_emit_update()

    def _set_label_list_item_text(
        self,
        item: AnnolidLabelListItem,
        *,
        base_text: str,
        marker: str,
        rgb: tuple[int, int, int],
    ) -> None:
        r, g, b = rgb
        item.setText(f"{base_text} {marker}")
        item.setForeground(QtGui.QBrush(QtGui.QColor(r, g, b)))

    def addLabel(self, shape, *, rebuild_unique: bool = True):
        def marker_for_shape(shape_obj: Shape) -> str:
            if str(getattr(shape_obj, "shape_type", "") or "").lower() != "point":
                return "●"
            visibility = keypoint_visibility_from_shape_object(shape_obj)
            return "○" if visibility == int(KeypointVisibility.OCCLUDED) else "●"

        if shape.group_id is None:
            text = str(shape.label)
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = AnnolidLabelListItem(text, shape)
        try:
            # Per-shape visibility toggle (checkbox in front of each label).
            visible = bool(getattr(shape, "visible", True))
            role = getattr(
                self.labelList, "VISIBILITY_STATE_ROLE", int(Qt.UserRole) + 10
            )
            with QtCore.QSignalBlocker(self.labelList):
                label_list_item.setFlags(
                    label_list_item.flags()
                    | Qt.ItemIsUserCheckable
                    | Qt.ItemIsSelectable
                )
                label_list_item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
                label_list_item.setData(role, bool(visible))
        except Exception:
            pass
        self.labelList.addItem(label_list_item)

        self.labelDialog.addLabelHistory(str(shape.label))
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        rgb = self._update_shape_color(shape)
        self._set_label_list_item_text(
            label_list_item,
            base_text=text,
            marker=marker_for_shape(shape),
            rgb=rgb,
        )
        if rebuild_unique:
            self._rebuild_unique_label_list()

    def propagateSelectedShape(self):
        from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

        if not self.canvas.selectedShapes:
            QtWidgets.QMessageBox.information(
                self, "No Shape Selected", "Please select a shape first."
            )
            return

        selected_shape = self.canvas.selectedShapes[0]
        current_frame = self.frame_number
        max_frame = self.num_frames - 1

        dialog = ShapePropagationDialog(
            self.canvas, self, current_frame, max_frame, parent=self
        )

        for i in range(dialog.shape_list.count()):
            item = dialog.shape_list.item(i)
            if item.data(QtCore.Qt.UserRole) == selected_shape:
                dialog.shape_list.setCurrentRow(i)
                break

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.statusBar().showMessage("Shape propagation completed.")
        else:
            self.statusBar().showMessage("Shape propagation canceled.")

    def editLabel(self, item=None):
        if item and not isinstance(item, AnnolidLabelListItem):
            raise TypeError("item must be AnnolidLabelListItem type")

        if not self.canvas.editing():
            return
        if not item:
            item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        shape_flags = shape.flags or {}
        safe_flags = {k: bool(v) for k, v in shape_flags.items()}
        text, flags, group_id, description = self.labelDialog.popUp(
            text=str(shape.label),
            flags=safe_flags,
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

        rgb = self._update_shape_color(shape)

        base_text = (
            str(shape.label)
            if shape.group_id is None
            else "{} ({})".format(shape.label, shape.group_id)
        )
        marker = "●"
        if str(getattr(shape, "shape_type", "") or "").lower() == "point":
            visibility = keypoint_visibility_from_shape_object(shape)
            marker = "○" if visibility == int(KeypointVisibility.OCCLUDED) else "●"
        self._set_label_list_item_text(
            item,
            base_text=base_text,
            marker=marker,
            rgb=rgb,
        )
        self.setDirty()
        self._rebuild_unique_label_list()

    def _selected_shapes_for_keypoint_visibility(self) -> list[Shape]:
        shapes = list(getattr(self.canvas, "selectedShapes", None) or [])
        if shapes:
            return shapes
        try:
            item = self.currentItem()
        except Exception:
            item = None
        if isinstance(item, AnnolidLabelListItem):
            shape = item.shape()
            if shape is not None:
                return [shape]
        return []

    def _refresh_label_list_items_for_shapes(self, shapes: list[Shape]) -> None:
        if not shapes:
            return
        target_ids = {id(s) for s in shapes}
        for item in self.labelList:
            if not isinstance(item, AnnolidLabelListItem):
                continue
            shape = item.shape()
            if shape is None or id(shape) not in target_ids:
                continue
            rgb = self._update_shape_color(shape)
            base_text = (
                str(shape.label)
                if shape.group_id is None
                else "{} ({})".format(shape.label, shape.group_id)
            )
            marker = "●"
            if str(getattr(shape, "shape_type", "") or "").lower() == "point":
                visibility = keypoint_visibility_from_shape_object(shape)
                marker = "○" if visibility == int(KeypointVisibility.OCCLUDED) else "●"
            self._set_label_list_item_text(
                item,
                base_text=base_text,
                marker=marker,
                rgb=rgb,
            )

    def set_selected_keypoint_visibility(self, visible: bool) -> None:
        shapes = [
            s
            for s in self._selected_shapes_for_keypoint_visibility()
            if str(getattr(s, "shape_type", "") or "").lower() == "point"
        ]
        if not shapes:
            self.statusBar().showMessage(
                "Select one or more keypoint (point) shapes first."
            )
            return
        target = KeypointVisibility.VISIBLE if visible else KeypointVisibility.OCCLUDED
        for shape in shapes:
            set_keypoint_visibility_on_shape_object(shape, int(target))
        self._refresh_label_list_items_for_shapes(shapes)
        self.canvas.update()
        self.setDirty()

    def toggle_selected_keypoint_visibility(self) -> None:
        shapes = [
            s
            for s in self._selected_shapes_for_keypoint_visibility()
            if str(getattr(s, "shape_type", "") or "").lower() == "point"
        ]
        if not shapes:
            self.statusBar().showMessage(
                "Select one or more keypoint (point) shapes first."
            )
            return
        for shape in shapes:
            current = keypoint_visibility_from_shape_object(shape)
            target = (
                KeypointVisibility.VISIBLE
                if current == int(KeypointVisibility.OCCLUDED)
                else KeypointVisibility.OCCLUDED
            )
            set_keypoint_visibility_on_shape_object(shape, int(target))
        self._refresh_label_list_items_for_shapes(shapes)
        self.canvas.update()
        self.setDirty()
