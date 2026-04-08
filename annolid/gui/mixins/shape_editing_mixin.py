from __future__ import annotations

from qtpy.QtCore import Qt


class ShapeEditingMixin:
    """Shape list/canvas editing actions."""

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
