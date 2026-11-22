from __future__ import annotations

from collections import Counter
import math

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt

from annolid.gui.polygon_tools import collapse_polygon_shape
from annolid.gui.polygon_tools import is_collapsed_polygon
from annolid.gui.polygon_tools import restore_polygon_shape
from annolid.utils.zone_shapes import is_zone_shape_payload


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

        if active_editor is not canvas and not selected_shapes:
            editor_selection = list(getattr(active_editor, "selectedShapes", []) or [])
            if editor_selection:
                try:
                    canvas.selectShapes(editor_selection)
                except Exception:
                    pass

        had_selection = bool(getattr(canvas, "selectedShapes", None))
        all_locked = had_selection and all(s.is_locked() for s in canvas.selectedShapes)
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
        elif all_locked:
            try:
                self.statusBar().showMessage(
                    self.tr(
                        "Selected shape(s) are locked. Unlock (Ctrl+Shift+L) before deleting."
                    ),
                    3000,
                )
            except Exception:
                pass

    def canMergeSelectedShapes(self) -> bool:
        canvas = self.canvas
        if (
            self._active_shape_editor() is not canvas
            or not canvas.editing()
            or len(canvas.selectedShapes) < 2
        ):
            return False
        selected = list(canvas.selectedShapes)
        if any(
            s.is_locked() or not bool(getattr(s, "visible", True)) for s in selected
        ):
            return False
        if all(
            s.shape_type == "mask" and getattr(s, "mask", None) is not None
            for s in selected
        ):
            return True
        return all(
            s.shape_type in {"polygon", "rectangle", "oriented_rectangle"}
            and getattr(s, "mask", None) is None
            for s in selected
        )

    def mergeSelectedShapes(self, _value=False) -> bool:
        from qtpy import QtWidgets
        from shapely.errors import ShapelyError
        from annolid.gui.shape_merge import merge_shapes

        if not self.canMergeSelectedShapes():
            return False
        selected_ids = {id(s) for s in self.canvas.selectedShapes}
        selected = [s for s in self.canvas.shapes if id(s) in selected_ids]
        try:
            merged = merge_shapes(selected)
        except (ValueError, ShapelyError) as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot Merge Shapes", str(exc))
            return False
        replacement = []
        for shape in self.canvas.shapes:
            if shape is selected[0]:
                replacement.append(merged)
            if id(shape) not in selected_ids:
                replacement.append(shape)
        if not self.canvas.shapesBackups:
            self.canvas.storeShapes()
        self.canvas.deSelectShape()
        self.loadShapes(replacement, replace=True)
        self.canvas.selectShapes([merged])
        self.setDirty()
        return True

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

    def _collect_selected_shapes_for_action(self) -> list:
        canvas = getattr(self, "canvas", None)
        selected = list(getattr(canvas, "selectedShapes", None) or [])
        if selected:
            return selected
        result = []
        label_list = getattr(self, "labelList", None)
        if label_list is not None:
            for item in label_list.selectedItems():
                shape = item.shape() if hasattr(item, "shape") else None
                if shape is not None and not any(s is shape for s in result):
                    result.append(shape)
        return result

    def copySelectedShapes(self, _value=False) -> list:
        selected = self._collect_selected_shapes_for_action()
        if not selected:
            return []
        copied = []
        for shape in selected:
            clone = shape.copy()
            clone.locked = False
            clone.selected = False
            clone.shared_vertex_ids = []
            clone.shared_edge_ids = []
            copied.append(clone)
        self._copied_shapes = copied
        paste_action = getattr(getattr(self, "actions", None), "pasteShapes", None)
        if paste_action is not None:
            paste_action.setEnabled(True)
        try:
            self.statusBar().showMessage(
                self.tr("Copied %d shape(s).") % len(copied), 2500
            )
        except Exception:
            pass
        return list(copied)

    @staticmethod
    def _shape_coord_signature(shape) -> tuple:
        pts = tuple(
            (round(float(p.x()), 2), round(float(p.y()), 2))
            for p in (getattr(shape, "points", None) or [])
        )
        return (str(getattr(shape, "shape_type", "") or ""), pts)

    def pasteSelectedShapes(self, _value=False) -> list:
        canvas = getattr(self, "canvas", None)
        clipboard = list(getattr(self, "_copied_shapes", None) or [])
        if canvas is None or not clipboard:
            return []
        pasted = []
        for template in clipboard:
            clone = template.copy()
            clone.locked = False
            clone.selected = False
            clone.shared_vertex_ids = []
            clone.shared_edge_ids = []
            pasted.append(clone)

        existing_sigs = {
            self._shape_coord_signature(s)
            for s in (getattr(canvas, "shapes", None) or [])
        }
        if any(self._shape_coord_signature(s) in existing_sigs for s in pasted):
            canvas.boundedShiftShapes(pasted)

        for shape in pasted:
            canvas.shapes.append(shape)
            canvas.visible[id(shape)] = bool(getattr(shape, "visible", True))
            self.addLabel(shape, rebuild_unique=False)
        if hasattr(canvas, "_shared_finalize_topology_edit"):
            canvas._shared_finalize_topology_edit()
        canvas.storeShapes()
        large_view = getattr(self, "large_image_view", None)
        if large_view is not None and hasattr(large_view, "set_shapes"):
            try:
                large_view.set_shapes(list(canvas.shapes))
            except Exception:
                pass
        self._rebuild_unique_label_list()
        canvas.selectShapes(pasted)
        canvas.update()
        self.setDirty()
        return pasted

    def selectAllShapes(self, _value=False) -> list:
        canvas = getattr(self, "canvas", None)
        if canvas is None or not hasattr(canvas, "selectAllVisibleShapes"):
            return []
        if not canvas.editing() and hasattr(self, "toggleDrawMode"):
            self.toggleDrawMode(True)
        return canvas.selectAllVisibleShapes()

    def setSelectedShapesLocked(self, locked: bool) -> int:
        selected = self._collect_selected_shapes_for_action()
        if not selected:
            return 0
        target = bool(locked)
        changed = []
        for shape in selected:
            if bool(getattr(shape, "locked", False)) != target:
                shape.locked = target
                changed.append(shape)
        if not changed:
            return 0
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.storeShapes()
        if hasattr(self, "_refresh_label_list_items_for_shapes"):
            self._refresh_label_list_items_for_shapes(changed)
        if hasattr(self, "_refresh_shape_views"):
            self._refresh_shape_views()
        elif canvas is not None:
            canvas.update()
        self.shapeSelectionChanged(selected)
        self.setDirty()
        try:
            verb = "Locked" if target else "Unlocked"
            self.statusBar().showMessage(
                self.tr("%s %d shape(s).") % (verb, len(changed)), 2500
            )
        except Exception:
            pass
        return len(changed)

    def toggleLockSelectedShapes(self, _value=False) -> bool:
        selected = self._collect_selected_shapes_for_action()
        if not selected:
            return False
        all_locked = all(s.is_locked() for s in selected)
        target = not all_locked
        self.setSelectedShapesLocked(target)
        return target

    def _set_all_shapes_visible(self, visible: bool) -> int:
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return 0
        shapes = list(getattr(canvas, "shapes", None) or [])
        if not shapes:
            return 0
        visible_flag = bool(visible)
        changed = 0
        for shape in shapes:
            if visible_flag and is_collapsed_polygon(shape):
                try:
                    restore_polygon_shape(shape)
                except Exception:
                    pass
            if bool(getattr(shape, "visible", True)) != visible_flag:
                changed += 1
            self._sync_shape_visibility_without_signal(shape, visible_flag)
        if not visible_flag and getattr(canvas, "selectedShapes", None):
            canvas.deSelectShape()
        if hasattr(self, "_refresh_shape_views"):
            self._refresh_shape_views()
        else:
            canvas.update()
        if hasattr(self, "_update_polygon_tool_action_state"):
            try:
                self._update_polygon_tool_action_state()
            except Exception:
                pass
        if changed:
            self.setDirty()
        return changed

    def showAllShapes(self, _value=False) -> int:
        return self._set_all_shapes_visible(True)

    def hideAllShapes(self, _value=False) -> int:
        return self._set_all_shapes_visible(False)

    def toggleAllShapes(self, _value=False) -> int:
        canvas = getattr(self, "canvas", None)
        shapes = (
            list(getattr(canvas, "shapes", None) or []) if canvas is not None else []
        )
        if not shapes:
            return 0
        any_visible = any(bool(getattr(s, "visible", True)) for s in shapes)
        return self._set_all_shapes_visible(not any_visible)

    def undoLastPointAction(self, _value=False) -> None:
        editor = self._active_shape_editor()
        if editor is not None and hasattr(editor, "undoLastPoint"):
            editor.undoLastPoint()

    def addPointToEdgeAction(self, _value=False) -> bool:
        canvas = getattr(self, "canvas", None)
        if canvas is None or not hasattr(canvas, "addPointToEdge"):
            return False
        if canvas.addPointToEdge():
            canvas.storeShapes()
            self.setDirty()
            return True
        return False

    def toggleShowLabelNames(self, enabled: bool) -> None:
        flag = bool(enabled)
        canvas_cfg = self._config.setdefault("canvas", {})
        if isinstance(canvas_cfg, dict):
            canvas_cfg["show_label_names"] = flag
        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "setShowLabelNames"):
            canvas.setShowLabelNames(flag)
        large_view = getattr(self, "large_image_view", None)
        if large_view is not None and hasattr(large_view, "setShowLabelNames"):
            large_view.setShowLabelNames(flag)

    def toggleAllowPointsOutsideImage(self, enabled: bool) -> None:
        flag = bool(enabled)
        canvas_cfg = self._config.setdefault("canvas", {})
        if isinstance(canvas_cfg, dict):
            canvas_cfg["allow_points_outside_image"] = flag
        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "setAllowPointsOutsideImage"):
            canvas.setAllowPointsOutsideImage(flag)
        large_view = getattr(self, "large_image_view", None)
        if large_view is not None and hasattr(large_view, "setAllowPointsOutsideImage"):
            large_view.setAllowPointsOutsideImage(flag)

    def saveFileAs(self, _value=False) -> bool:
        if not getattr(self, "filename", None):
            return False
        default_path = ""
        try:
            default_path = self.getLabelFile()
        except Exception:
            default_path = str(self.filename)
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Save Annotations As"),
            default_path,
            self.tr("Label files (*.json)"),
        )
        if not target:
            return False
        if not str(target).lower().endswith(".json"):
            target = f"{target}.json"
        if hasattr(self, "_saveFile"):
            self._saveFile(target)
            return True
        return False

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
            if shape.is_locked():
                continue
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
            canvas = getattr(self, "canvas", None)
            if canvas is not None and hasattr(canvas, "storeShapes"):
                canvas.storeShapes()
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
            if shape.is_locked():
                continue
            if not is_collapsed_polygon(shape):
                continue
            if not restore_polygon_shape(shape):
                continue
            restored += 1
            synced_any = (
                self._sync_shape_visibility_without_signal(shape, True) or synced_any
            )
        if restored:
            canvas = getattr(self, "canvas", None)
            if canvas is not None and hasattr(canvas, "storeShapes"):
                canvas.storeShapes()
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
            shape = (
                self.canvas.shapes[-1]
                if self.canvas.shapes
                and getattr(self.canvas.shapes[-1], "description", None)
                == "grounding_sam"
                else None
            )
            if shape is None:
                shapes = [
                    candidate
                    for candidate in self.canvas.shapes
                    if candidate.description == "grounding_sam"
                ]
                if not shapes:
                    return
                shape = shapes[-1]
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
                preselected_polygons = self._collect_preselected_polygons(editor)
                self.labelList.clearSelection()
                shapes = editor.setLastLabel(text, flags)
                for shape in shapes:
                    shape.group_id = group_id
                    shape.description = description
                    self._resolve_created_point_polygon_association(
                        shape,
                        editor=editor,
                        preselected_polygons=preselected_polygons,
                    )
                    self.addLabel(shape)
                if editor is self.canvas and getattr(
                    self.canvas, "shapesBackups", None
                ):
                    self.canvas.shapesBackups[-1] = [
                        s.copy() for s in self.canvas.shapes
                    ]
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

        editor = self._active_shape_editor()
        preselected_polygons = self._collect_preselected_polygons(editor)
        self.labelList.clearSelection()
        shapes = editor.setLastLabel(next_label, {})
        for shape in shapes:
            self._resolve_created_point_polygon_association(
                shape,
                editor=editor,
                preselected_polygons=preselected_polygons,
            )
            self.addLabel(shape)
        if editor is canvas and getattr(canvas, "shapesBackups", None):
            canvas.shapesBackups[-1] = [s.copy() for s in canvas.shapes]
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

    @staticmethod
    def _shape_point_xy(point: object) -> tuple[float, float] | None:
        if (
            hasattr(point, "x")
            and hasattr(point, "y")
            and callable(point.x)
            and callable(point.y)
        ):
            try:
                x = float(point.x())
                y = float(point.y())
                return (x, y) if (math.isfinite(x) and math.isfinite(y)) else None
            except Exception:
                return None
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                x = float(point[0])
                y = float(point[1])
                return (x, y) if (math.isfinite(x) and math.isfinite(y)) else None
            except Exception:
                return None
        return None

    @classmethod
    def _point_in_polygon(
        cls,
        x: float,
        y: float,
        vertices: list[tuple[float, float]],
    ) -> bool:
        n = len(vertices)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            dx = xj - xi
            dy = yj - yi
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq <= 1e-12:
                if math.hypot(x - xi, y - yi) <= 1e-6:
                    return True
            else:
                t = ((x - xi) * dx + (y - yi) * dy) / seg_len_sq
                if 0.0 <= t <= 1.0:
                    proj_x = xi + t * dx
                    proj_y = yi + t * dy
                    if math.hypot(x - proj_x, y - proj_y) <= 1e-4:
                        return True
            if (yi > y) != (yj > y):
                x_intersect = (xj - xi) * (y - yi) / ((yj - yi) + 1e-15) + xi
                if x <= x_intersect:
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _is_zone_shape_object(shape: object) -> bool:
        flags = getattr(shape, "flags", None)
        payload = {
            "label": getattr(shape, "label", ""),
            "description": getattr(shape, "description", ""),
            "flags": flags if isinstance(flags, dict) else {},
        }
        return is_zone_shape_payload(payload)

    def _collect_preselected_polygons(self, editor: object) -> list[object]:
        polygons: list[object] = []
        seen_ids: set[int] = set()

        def _consider(candidate: object) -> None:
            if candidate is None or id(candidate) in seen_ids:
                return
            stype = str(getattr(candidate, "shape_type", "") or "").strip()
            if stype != "polygon" or self._is_zone_shape_object(candidate):
                return
            seen_ids.add(id(candidate))
            polygons.append(candidate)

        label_list = getattr(self, "labelList", None)
        selected_items_fn = getattr(label_list, "selectedItems", None)
        if callable(selected_items_fn):
            try:
                for item in selected_items_fn():
                    shape_fn = getattr(item, "shape", None)
                    if callable(shape_fn):
                        _consider(shape_fn())
            except Exception:
                pass

        for surface in (editor, getattr(self, "canvas", None)):
            if surface is None:
                continue
            for cand in list(getattr(surface, "selectedShapes", None) or []):
                _consider(cand)

        return polygons

    def _prompt_overlapping_polygon_owner(
        self,
        shape: object,
        containing_entries: list[dict[str, object]],
    ) -> dict[str, object] | None:
        if getattr(self, "_suppress_overlap_owner_prompt", False):
            return None
        force_prompt = bool(getattr(self, "_force_overlap_owner_prompt", False))
        if not force_prompt:
            if QtWidgets.QApplication.instance() is None:
                return None
            if not isinstance(self, QtWidgets.QWidget) or not self.isVisible():
                return None

        options: list[str] = []
        for idx, entry in enumerate(containing_entries):
            raw_lbl = str(
                entry.get("raw_label") or entry.get("inst_label") or "polygon"
            )
            inst_lbl = str(entry.get("inst_label") or raw_lbl)
            gid = entry.get("group_id")
            if gid is not None:
                display = f"{raw_lbl} (group {gid})"
            elif inst_lbl != raw_lbl:
                display = f"{raw_lbl} [{inst_lbl}]"
            else:
                display = f"{raw_lbl} #{idx + 1}"
            options.append(display)

        kp_name = str(getattr(shape, "label", "") or "keypoint")
        parent = self if isinstance(self, QtWidgets.QWidget) else None
        choice, accepted = QtWidgets.QInputDialog.getItem(
            parent,
            "Overlapping Polygon Keypoint",
            f"Select the polygon instance that owns '{kp_name}':",
            options,
            0,
            False,
        )
        if not accepted or not choice:
            return None
        try:
            chosen_idx = options.index(choice)
        except ValueError:
            return None
        return containing_entries[chosen_idx]

    @staticmethod
    def _bind_point_to_polygon_entry(
        shape: object,
        entry: dict[str, object],
        *,
        set_instance_flag: bool,
    ) -> None:
        gid = entry.get("group_id")
        if gid is not None and getattr(shape, "group_id", None) is None:
            shape.group_id = gid
        if set_instance_flag:
            inst_label = str(entry.get("inst_label") or "").strip()
            if inst_label:
                flags = getattr(shape, "flags", None)
                updated_flags = dict(flags) if isinstance(flags, dict) else {}
                updated_flags["instance_label"] = inst_label
                raw_label = str(getattr(shape, "label", "") or "").strip()
                if raw_label and "display_label" not in updated_flags:
                    updated_flags["display_label"] = raw_label
                shape.flags = updated_flags

    def _resolve_created_point_polygon_association(
        self,
        shape: object,
        *,
        editor: object = None,
        preselected_polygons: list[object] | tuple[object, ...] = (),
    ) -> None:
        stype = str(getattr(shape, "shape_type", "") or "").strip() or "point"
        raw_pts = getattr(shape, "points", None)
        if (
            stype != "point"
            or not isinstance(raw_pts, (list, tuple))
            or len(raw_pts) != 1
        ):
            return
        xy = self._shape_point_xy(raw_pts[0])
        if xy is None:
            return
        px, py = xy

        if getattr(shape, "group_id", None) is not None:
            return
        raw_kp_label = str(getattr(shape, "label", "") or "").strip()
        if ":" in raw_kp_label:
            return
        raw_flags = getattr(shape, "flags", None)
        if (
            isinstance(raw_flags, dict)
            and str(raw_flags.get("instance_label") or "").strip()
        ):
            return

        raw_shapes = getattr(editor, "shapes", None)
        if not isinstance(raw_shapes, (list, tuple)):
            raw_shapes = getattr(getattr(self, "canvas", None), "shapes", None)
        if not isinstance(raw_shapes, (list, tuple)) or not raw_shapes:
            return

        base_counts: Counter[str] = Counter()
        poly_raw: list[
            tuple[object, str, str, object, bool, list[tuple[float, float]]]
        ] = []
        for cand in raw_shapes:
            if cand is shape or self._is_zone_shape_object(cand):
                continue
            if str(getattr(cand, "shape_type", "") or "").strip() != "polygon":
                continue
            cand_pts = getattr(cand, "points", None) or []
            verts = [
                pt_xy
                for pt in cand_pts
                if (pt_xy := self._shape_point_xy(pt)) is not None
            ]
            if len(verts) < 3:
                continue
            c_label = str(getattr(cand, "label", "") or "").strip()
            c_flags = getattr(cand, "flags", None)
            explicit_inst = (
                str(c_flags.get("instance_label") or "").strip()
                if isinstance(c_flags, dict)
                else ""
            )
            base_inst = explicit_inst or (
                c_label[:-5] if c_label.endswith("_mask") else c_label
            )
            if not base_inst:
                continue
            base_counts[base_inst] += 1
            poly_raw.append(
                (
                    cand,
                    c_label,
                    base_inst,
                    getattr(cand, "group_id", None),
                    bool(explicit_inst),
                    verts,
                )
            )

        if not poly_raw:
            return

        used_inst_labels: set[str] = set()
        dup_ordinals: Counter[str] = Counter()
        poly_entries: list[dict[str, object]] = []
        for cand, c_label, base_inst, c_gid, has_explicit, verts in poly_raw:
            inst_label = base_inst
            if base_counts[base_inst] > 1 and not has_explicit:
                gid_str = str(c_gid).strip() if c_gid is not None else ""
                if not gid_str or f"{base_inst}_{gid_str}" in used_inst_labels:
                    ordinal = dup_ordinals[base_inst]
                    while f"{base_inst}_{ordinal}" in used_inst_labels:
                        ordinal += 1
                    dup_ordinals[base_inst] = ordinal + 1
                    gid_str = str(ordinal)
                inst_label = f"{base_inst}_{gid_str}"
            used_inst_labels.add(inst_label)
            poly_entries.append(
                {
                    "shape": cand,
                    "raw_label": c_label or base_inst,
                    "inst_label": inst_label,
                    "group_id": c_gid,
                    "has_explicit_inst": has_explicit,
                    "vertices": verts,
                }
            )

        containing = [
            entry
            for entry in poly_entries
            if self._point_in_polygon(px, py, entry["vertices"])  # type: ignore[arg-type]
        ]
        if not containing:
            return

        multi_poly_scene = len(poly_entries) >= 2
        if len(containing) == 1:
            sole = containing[0]
            if (
                multi_poly_scene
                or sole.get("group_id") is not None
                or bool(sole.get("has_explicit_inst"))
            ):
                self._bind_point_to_polygon_entry(
                    shape,
                    sole,
                    set_instance_flag=(
                        multi_poly_scene or bool(sole.get("has_explicit_inst"))
                    ),
                )
            return

        # Overlapping (>= 2) polygons contain this point:
        preselected_matches = [
            entry
            for entry in containing
            if any(entry["shape"] is sel for sel in preselected_polygons)
        ]
        if len(preselected_matches) == 1:
            self._bind_point_to_polygon_entry(
                shape, preselected_matches[0], set_instance_flag=True
            )
            return

        # Prefix match against containing polygon labels/instance_labels:
        lower_kp = raw_kp_label.lower()
        prefix_matches: list[dict[str, object]] = []
        for entry in containing:
            for token in {
                str(entry.get("inst_label") or "").strip(),
                str(entry.get("raw_label") or "").strip(),
            }:
                if (
                    token
                    and len(raw_kp_label) > len(token)
                    and lower_kp.startswith(token.lower())
                    and raw_kp_label[len(token)] in "_-:| "
                ):
                    prefix_matches.append(entry)
                    break
        if len(prefix_matches) == 1:
            self._bind_point_to_polygon_entry(
                shape, prefix_matches[0], set_instance_flag=True
            )
            return

        # Unique missing landmark among containing polygons:
        norm_kp = raw_kp_label.lower()
        if norm_kp:
            owned_by_inst: dict[str, set[str]] = {
                str(entry["inst_label"]): set() for entry in containing
            }
            gid_to_inst: dict[str, str | None] = {}
            for entry in poly_entries:
                if entry.get("group_id") is None:
                    continue
                gkey = str(entry["group_id"]).strip()
                if gkey not in gid_to_inst:
                    gid_to_inst[gkey] = str(entry["inst_label"])
                elif gid_to_inst[gkey] != str(entry["inst_label"]):
                    gid_to_inst[gkey] = None

            for other in raw_shapes:
                if other is shape or self._is_zone_shape_object(other):
                    continue
                o_type = str(getattr(other, "shape_type", "") or "").strip() or "point"
                o_pts = getattr(other, "points", None)
                if (
                    o_type != "point"
                    or not isinstance(o_pts, (list, tuple))
                    or len(o_pts) != 1
                ):
                    continue
                o_xy = self._shape_point_xy(o_pts[0])
                if o_xy is None:
                    continue
                o_lbl = str(getattr(other, "label", "") or "").strip()
                o_flags = getattr(other, "flags", None)
                o_disp = (
                    str(o_flags.get("display_label") or "").strip()
                    if isinstance(o_flags, dict)
                    else ""
                )
                o_inst = (
                    str(o_flags.get("instance_label") or "").strip()
                    if isinstance(o_flags, dict)
                    else ""
                )
                o_kp = o_disp or (o_lbl.split(":", 1)[1] if ":" in o_lbl else o_lbl)
                if ":" in o_lbl and not o_inst:
                    o_inst = o_lbl.split(":", 1)[0]
                if not o_inst and getattr(other, "group_id", None) is not None:
                    mapped = gid_to_inst.get(str(getattr(other, "group_id")).strip())
                    if mapped:
                        o_inst = mapped
                if not o_inst:
                    o_cont = [
                        entry
                        for entry in poly_entries
                        if self._point_in_polygon(
                            o_xy[0],
                            o_xy[1],
                            entry["vertices"],  # type: ignore[arg-type]
                        )
                    ]
                    if len(o_cont) == 1:
                        o_inst = str(o_cont[0]["inst_label"])
                if o_inst in owned_by_inst and o_kp.strip():
                    owned_by_inst[o_inst].add(o_kp.strip().lower())

            missing_entries = [
                entry
                for entry in containing
                if norm_kp not in owned_by_inst.get(str(entry["inst_label"]), set())
            ]
            if len(missing_entries) == 1:
                self._bind_point_to_polygon_entry(
                    shape, missing_entries[0], set_instance_flag=True
                )
                return

        chosen = self._prompt_overlapping_polygon_owner(shape, containing)
        if chosen is not None:
            self._bind_point_to_polygon_entry(shape, chosen, set_instance_flag=True)
