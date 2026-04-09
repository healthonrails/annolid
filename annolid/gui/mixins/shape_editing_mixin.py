from __future__ import annotations

from qtpy import QtCore
from qtpy.QtCore import Qt

from annolid.gui.polygon_tools import collapse_polygon_shape
from annolid.gui.polygon_tools import is_collapsed_polygon
from annolid.gui.polygon_tools import restore_polygon_shape


class ShapeEditingMixin:
    """Shape list/canvas editing actions."""

    def _sync_shape_visibility_without_signal(self, shape, visible: bool) -> bool:
        visible_flag = bool(visible)
        synced = False

        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "setShapeVisible"):
            try:
                canvas.setShapeVisible(
                    shape,
                    visible_flag,
                    emit_selection=False,
                )
                synced = True
            except TypeError:
                try:
                    canvas.setShapeVisible(shape, visible_flag)
                    synced = True
                except Exception:
                    pass
            except Exception:
                pass

        large_view = getattr(self, "large_image_view", None)
        if large_view is not None and hasattr(large_view, "setShapeVisible"):
            try:
                large_view.setShapeVisible(
                    shape,
                    visible_flag,
                    emit_selection=False,
                )
                synced = True
            except TypeError:
                try:
                    large_view.setShapeVisible(shape, visible_flag)
                    synced = True
                except Exception:
                    pass
            except Exception:
                pass

        try:
            item = self._label_list_item_for_shape(shape)
        except Exception:
            item = None
        if item is not None:
            try:
                with QtCore.QSignalBlocker(self.labelList):
                    item.setCheckState(Qt.Checked if visible_flag else Qt.Unchecked)
                    item.setData(
                        self.labelList.VISIBILITY_STATE_ROLE,
                        bool(visible_flag),
                    )
            except Exception:
                pass

        return synced

    def _active_shape_editor(self):
        if getattr(self, "_active_image_view", "canvas") == "tiled":
            editor = getattr(self, "large_image_view", None)
            if editor is not None and hasattr(editor, "setLastLabel"):
                return editor
        return self.canvas

    def remLabels(self, shapes) -> None:
        super().remLabels(shapes)
        self._rebuild_unique_label_list()

    def deleteSelectedShapes(self, _value=False) -> None:
        # Prefer explicit selection from Label Instances dock when available.
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return

        active_editor = self._active_shape_editor()
        selected_shapes = []
        for item in self.labelList.selectedItems():
            try:
                shape = item.shape()
            except Exception:
                shape = None
            if shape is not None:
                selected_shapes.append(shape)
        if selected_shapes:
            try:
                canvas.selectShapes(selected_shapes)
            except Exception:
                pass

        if active_editor is not canvas:
            editor_selection = list(getattr(active_editor, "selectedShapes", []) or [])
            if editor_selection:
                try:
                    canvas.selectShapes(editor_selection)
                except Exception:
                    pass

        deleted = canvas.deleteSelected() or []
        if deleted and active_editor is not canvas:
            try:
                if hasattr(active_editor, "set_selected_shapes"):
                    active_editor.set_selected_shapes([])
                elif hasattr(active_editor, "selectedShapes"):
                    active_editor.selectedShapes = []
            except Exception:
                pass
            try:
                if hasattr(active_editor, "set_shapes"):
                    active_editor.set_shapes(list(getattr(canvas, "shapes", []) or []))
            except Exception:
                pass

        if deleted:
            self.remLabels(deleted)
            self.setDirty()
            try:
                self.labelList.clearSelection()
            except Exception:
                pass

    def duplicateSelectedShapes(self, _value=False) -> None:
        duplicated = self.canvas.duplicateSelectedShapes() or []
        if not duplicated:
            return
        existing_ids: set[int] = set()
        for idx in range(self.labelList.count()):
            try:
                shape = self.labelList.item(idx).shape()
                if shape is not None:
                    existing_ids.add(id(shape))
            except Exception:
                continue
        for shape in duplicated:
            if shape is not None and id(shape) not in existing_ids:
                self.addLabel(shape, rebuild_unique=False)
        self._rebuild_unique_label_list()
        self.shapeSelectionChanged(duplicated)
        self.setDirty()

    def _selected_shapes_for_polygon_tools(self) -> list:
        shapes = []
        editor = self._active_shape_editor()
        for candidate in (
            list(getattr(editor, "selectedShapes", []) or []),
            list(getattr(self.canvas, "selectedShapes", []) or []),
        ):
            for shape in candidate:
                if shape is not None and shape not in shapes:
                    shapes.append(shape)
        for item in self.labelList.selectedItems():
            try:
                shape = item.shape()
            except Exception:
                shape = None
            if shape is not None and shape not in shapes:
                shapes.append(shape)
        current = self.currentItem()
        if current is not None:
            try:
                shape = current.shape()
            except Exception:
                shape = None
            if shape is not None and shape not in shapes:
                shapes.append(shape)
        return shapes

    def canCollapseSelectedPolygons(self) -> bool:
        for shape in self._selected_shapes_for_polygon_tools():
            if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
                continue
            if bool(getattr(shape, "visible", True)) and not is_collapsed_polygon(
                shape
            ):
                return True
        return False

    def canRestoreSelectedPolygons(self) -> bool:
        for shape in self._selected_shapes_for_polygon_tools():
            if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
                continue
            if is_collapsed_polygon(shape):
                return True
        return False

    def canInferCurrentLargeImagePagePolygons(self) -> bool:
        if not bool(getattr(self, "_has_large_image_page_navigation", lambda: False)()):
            return False
        for shape in getattr(self.canvas, "shapes", []) or []:
            if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
                continue
            if bool(getattr(shape, "visible", True)) and not is_collapsed_polygon(
                shape
            ):
                return False
        return True

    def collapseSelectedPolygons(self, _value=False) -> int:
        collapsed = 0
        synced_any = False
        shapes = [
            shape
            for shape in self._selected_shapes_for_polygon_tools()
            if str(getattr(shape, "shape_type", "") or "").lower() == "polygon"
        ]
        if not shapes:
            return 0
        for shape in shapes:
            if not bool(getattr(shape, "visible", True)):
                continue
            if is_collapsed_polygon(shape):
                continue
            if not collapse_polygon_shape(shape):
                continue
            collapsed += 1
            synced_any = (
                self._sync_shape_visibility_without_signal(shape, False) or synced_any
            )
        if collapsed:
            if not synced_any:
                self._refresh_shape_views()
            self._rebuild_unique_label_list()
            self.setDirty()
        return collapsed

    def restoreSelectedPolygons(self, _value=False) -> int:
        restored = 0
        synced_any = False
        shapes = [
            shape
            for shape in self._selected_shapes_for_polygon_tools()
            if str(getattr(shape, "shape_type", "") or "").lower() == "polygon"
        ]
        if not shapes:
            return 0
        for shape in shapes:
            if not is_collapsed_polygon(shape):
                continue
            if not restore_polygon_shape(shape):
                continue
            restored += 1
            synced_any = (
                self._sync_shape_visibility_without_signal(shape, True) or synced_any
            )
        if restored:
            if not synced_any:
                self._refresh_shape_views()
            self._rebuild_unique_label_list()
            self.setDirty()
        return restored

    def inferCurrentLargeImagePagePolygons(self, _value=False) -> bool:
        inferer = getattr(self, "inferCurrentLargeImagePageAnnotations", None)
        if not callable(inferer):
            return False
        if not self.canInferCurrentLargeImagePagePolygons():
            return False
        try:
            return bool(inferer())
        except Exception:
            return False

    def startAdjoiningPolygonFromSelection(self, _value=False, edge_index=None) -> None:
        editor = self._active_shape_editor()
        tiled_editor = getattr(self, "large_image_view", None)

        try:
            self.toggleDrawMode(False, createMode="polygon")
        except Exception:
            pass

        starter = getattr(editor, "startAdjoiningPolygonFromSelection", None)
        if callable(starter):
            try:
                if starter(edge_index):
                    return
            except Exception:
                pass

        if tiled_editor is None or tiled_editor is editor:
            return

        starter = getattr(tiled_editor, "startAdjoiningPolygonFromSelection", None)
        if not callable(starter):
            return
        try:
            starter(edge_index)
        except Exception:
            return

    def startSharedBoundaryReshape(self, _value=False) -> None:
        editor = self._active_shape_editor()
        tiled_editor = getattr(self, "large_image_view", None)

        starter = getattr(editor, "startSharedBoundaryReshape", None)
        if callable(starter):
            try:
                if starter():
                    return
            except Exception:
                pass

        if tiled_editor is None or tiled_editor is editor:
            return

        starter = getattr(tiled_editor, "startSharedBoundaryReshape", None)
        if not callable(starter):
            return
        try:
            starter()
        except Exception:
            return

    def newShape(self):
        """Pop-up and give focus to the label editor."""
        editor = self._active_shape_editor()
        if self.canvas.createMode == "grounding_sam":
            self.labelList.clearSelection()
            shapes = [
                shape
                for shape in self.canvas.shapes
                if shape.description == "grounding_sam"
            ]
            shape = shapes.pop()
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            if self._try_apply_keypoint_sequence_labeling():
                return
            items = self.uniqLabelList.selectedItems()
            text = None
            if items:
                text = items[0].data(Qt.UserRole)
            flags = {}
            group_id = None
            description = ""
            zone_defaults = getattr(self, "_zone_authoring_defaults", None)
            if isinstance(zone_defaults, dict):
                text = zone_defaults.get("text") or text
                flags = dict(zone_defaults.get("flags") or {})
                group_id = zone_defaults.get("group_id", None)
                description = str(zone_defaults.get("description") or "")
            if str(text or "").strip().lower() == "chamber_1":
                text = ""
            show_popup = (
                bool(zone_defaults) or self._config["display_label_popup"] or not text
            )
            if show_popup:
                previous_text = self.labelDialog.edit.text()
                text, flags, group_id, description = self.labelDialog.popUp(
                    text,
                    flags=flags,
                    group_id=group_id,
                    description=description,
                    show_flags=not bool(zone_defaults),
                )
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
                shapes = editor.setLastLabel(text, flags)
                for shape in shapes:
                    shape.group_id = group_id
                    shape.description = description
                    self.addLabel(shape)
                self.actions.editMode.setEnabled(True)
                self.actions.undoLastPoint.setEnabled(False)
                self.actions.undo.setEnabled(True)
                self.setDirty()
            else:
                editor.undoLastLine()
                if hasattr(self.canvas, "shapesBackups") and self.canvas.shapesBackups:
                    self.canvas.shapesBackups.pop()

    def _try_apply_keypoint_sequence_labeling(self) -> bool:
        canvas = getattr(self, "canvas", None)
        if canvas is None or getattr(canvas, "createMode", None) != "point":
            return False

        sequencer = getattr(self, "keypoint_sequence_widget", None)
        if sequencer is None or not bool(
            getattr(sequencer, "is_sequence_enabled", lambda: False)()
        ):
            return False

        next_label = getattr(sequencer, "consume_next_label", lambda: None)()
        if not next_label:
            return False

        self.labelList.clearSelection()
        shapes = self._active_shape_editor().setLastLabel(next_label, {})
        for shape in shapes:
            self.addLabel(shape)
        self.actions.editMode.setEnabled(True)
        self.actions.undoLastPoint.setEnabled(False)
        self.actions.undo.setEnabled(True)
        self.setDirty()

        should_auto_save = bool(
            getattr(sequencer, "auto_save_on_click", lambda: False)()
        )
        if should_auto_save and getattr(self, "filename", None):
            try:
                self.saveFile()
            except Exception:
                pass
        return True
