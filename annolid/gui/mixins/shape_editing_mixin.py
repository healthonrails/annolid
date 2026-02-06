from __future__ import annotations

from qtpy.QtCore import Qt


class ShapeEditingMixin:
    """Shape list/canvas editing actions."""

    def remLabels(self, shapes) -> None:
        super().remLabels(shapes)
        self._rebuild_unique_label_list()

    def deleteSelectedShapes(self, _value=False) -> None:
        # Prefer explicit selection from Label Instances dock when available.
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
                self.canvas.selectShapes(selected_shapes)
            except Exception:
                pass

        deleted = self.canvas.deleteSelected() or []
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

    def newShape(self):
        """Pop-up and give focus to the label editor."""
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
