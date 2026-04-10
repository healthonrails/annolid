from __future__ import annotations

import re
from numbers import Number
import os
from pathlib import Path
from typing import Optional

from qtpy import QtCore

from annolid.gui.brain_3d_model import (
    Brain3DConfig,
    apply_coronal_polygon_edit,
    brain_model_from_other_data,
    build_brain_3d_model,
    materialize_coronal_plane_shapes,
    reslice_brain_model,
    set_region_presence_on_plane,
    store_brain_model_in_other_data,
)
from annolid.gui.label_file import LabelFile, LabelFileError
from annolid.gui.shape import Shape
from annolid.gui.shared_vertices import rebuild_polygon_topology
from annolid.gui.polygon_tools import infer_polygon_shapes_between_pages
from annolid.gui.polygon_tools import is_inferable_polygon
from annolid.gui.polygon_tools import polygon_identity_key
from annolid.gui.status import post_window_status
from annolid.gui.widgets.brain_3d_dock import Brain3DSessionDockWidget
from annolid.infrastructure import AnnotationStore
from annolid.postprocessing.quality_control import pred_dict_to_labelme
from annolid.utils.logger import logger


class AnnotationLoadingMixin:
    """Annotation/label loading workflow for frames and images."""

    def _materialize_label_shapes(self, shapes):
        s = []
        for shape_data in shapes:
            label = shape_data["label"]
            points = shape_data["points"]
            shape_type = shape_data["shape_type"]
            flags = shape_data["flags"]
            group_id = shape_data["group_id"]
            description = shape_data.get("description", "")
            other_data = shape_data["other_data"]
            if "visible" in shape_data:
                visible = shape_data["visible"]
            else:
                visible = True

            if not points:
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                mask=shape_data["mask"],
                visible=visible,
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shared_ids = list(shape_data.get("shared_vertex_ids") or [])
            if shared_ids:
                normalized_ids = [str(value or "") for value in shared_ids]
                if len(normalized_ids) < len(shape.points):
                    normalized_ids.extend(
                        [""] * (len(shape.points) - len(normalized_ids))
                    )
                shape.shared_vertex_ids = normalized_ids[: len(shape.points)]
            shared_edge_ids = list(shape_data.get("shared_edge_ids") or [])
            if shared_edge_ids:
                normalized_edge_ids = [str(value or "") for value in shared_edge_ids]
                if len(normalized_edge_ids) < len(shape.points):
                    normalized_edge_ids.extend(
                        [""] * (len(shape.points) - len(normalized_edge_ids))
                    )
                shape.shared_edge_ids = normalized_edge_ids[: len(shape.points)]
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
        return s

    def loadLabels(self, shapes):
        s = self._materialize_label_shapes(shapes)
        self.loadShapes(s)
        return s

    def _large_image_page_baseline_other_data(self) -> dict:
        baseline: dict = {}
        current_other_data = getattr(self, "otherData", None)
        if isinstance(current_other_data, dict):
            if "large_image" in current_other_data:
                baseline["large_image"] = current_other_data["large_image"]
            if "svg_overlays" in current_other_data:
                baseline["svg_overlays"] = current_other_data["svg_overlays"]
            if "label_image_overlay" in current_other_data:
                baseline["label_image_overlay"] = current_other_data[
                    "label_image_overlay"
                ]
        return baseline

    def _large_image_page_label_file(
        self, page_index: int
    ) -> tuple[Path, LabelFile] | None:
        resolver = getattr(self, "_large_image_stack_label_path", None)
        if not callable(resolver):
            return None
        label_path_str = resolver(page_index=page_index)
        if not label_path_str:
            return None
        label_path = Path(label_path_str)
        if not label_path.exists():
            return None
        try:
            label_file = LabelFile(str(label_path), is_video_frame=True)
        except Exception as exc:
            logger.debug(
                "Unable to load TIFF page label file %s for inference: %s",
                label_path,
                exc,
            )
            return None
        return label_path, label_file

    def _large_image_page_candidate_payload(
        self, page_index: int
    ) -> dict[str, object] | None:
        page = self._large_image_page_label_file(page_index)
        if page is None:
            return None
        label_path, label_file = page
        shapes = self._materialize_label_shapes(
            list(getattr(label_file, "shapes", []) or [])
        )
        if not any(is_inferable_polygon(shape) for shape in shapes):
            return None
        return {
            "page_index": int(page_index),
            "label_path": label_path,
            "label_file": label_file,
            "shapes": shapes,
            "other_data": dict(getattr(label_file, "otherData", {}) or {}),
            "caption": label_file.get_caption(),
        }

    def _large_image_neighbor_page_payloads(self, page_index: int):
        backend = getattr(self, "large_image_backend", None)
        page_count = int(getattr(backend, "get_page_count", lambda: 1)() or 1)
        previous_payload = None
        next_payload = None
        for index in range(int(page_index) - 1, -1, -1):
            previous_payload = self._large_image_page_candidate_payload(index)
            if previous_payload is not None:
                break
        for index in range(int(page_index) + 1, page_count):
            next_payload = self._large_image_page_candidate_payload(index)
            if next_payload is not None:
                break
        return previous_payload, next_payload

    def _infer_large_image_page_annotations(
        self,
        page_index: int,
        *,
        fallback_other_data: Optional[dict] = None,
    ) -> bool:
        previous_payload, next_payload = self._large_image_neighbor_page_payloads(
            page_index
        )
        previous_shapes = (
            list(previous_payload.get("shapes") or []) if previous_payload else []
        )
        next_shapes = list(next_payload.get("shapes") or []) if next_payload else []
        if not previous_shapes and not next_shapes:
            return False

        previous_page = (
            int(previous_payload["page_index"]) if previous_payload else None
        )
        next_page = int(next_payload["page_index"]) if next_payload else None
        inferred_shapes = infer_polygon_shapes_between_pages(
            previous_shapes,
            next_shapes,
            target_page=int(page_index),
            previous_page=previous_page,
            next_page=next_page,
        )
        if not inferred_shapes:
            return False

        baseline_other_data = self._large_image_page_baseline_other_data()
        merged_other_data = dict(baseline_other_data)
        if isinstance(fallback_other_data, dict):
            merged_other_data.update(fallback_other_data)
        merged_other_data["large_image_page"] = {
            "page_index": int(page_index),
            "label_path": str(
                getattr(
                    self,
                    "_large_image_stack_label_path",
                    lambda **kwargs: "",
                )(page_index=page_index)
                or ""
            ),
            "inferred": True,
            "source_pages": [
                int(payload["page_index"])
                for payload in (previous_payload, next_payload)
                if payload is not None
            ],
            "source_label_paths": [
                str(payload["label_path"])
                for payload in (previous_payload, next_payload)
                if payload is not None
            ],
        }
        if previous_payload is not None:
            merged_other_data["large_image_page"]["previous_caption"] = str(
                previous_payload.get("caption") or ""
            )
        if next_payload is not None:
            merged_other_data["large_image_page"]["next_caption"] = str(
                next_payload.get("caption") or ""
            )
        self.labelFile = None
        self.otherData = merged_other_data
        if hasattr(self.canvas, "setBehaviorText"):
            self.canvas.setBehaviorText(None)
        self.loadShapes(inferred_shapes, replace=True)
        if getattr(self, "caption_widget", None) is not None:
            self.caption_widget.set_caption("")
            self.caption_widget.set_image_path(
                str(getattr(self, "imagePath", "") or "")
            )
        self._post_window_status(
            self.tr("Loaded inferred polygons for TIFF page %d") % (int(page_index) + 1)
        )
        return True

    def inferCurrentLargeImagePageAnnotations(self) -> bool:
        if not bool(getattr(self, "_has_large_image_page_navigation", lambda: False)()):
            return False
        if not bool(
            getattr(self, "canInferCurrentLargeImagePagePolygons", lambda: False)()
        ):
            return False
        page_index = int(getattr(self, "frame_number", 0) or 0)
        return self._infer_large_image_page_annotations(page_index)

    def _ensureBrain3DSessionDock(self) -> Brain3DSessionDockWidget:
        dock = getattr(self, "brain3d_session_dock", None)
        if isinstance(dock, Brain3DSessionDockWidget):
            return dock
        dock = Brain3DSessionDockWidget(self)
        dock.rebuildRequested.connect(self.buildBrain3DModelFromSagittalPages)
        dock.regenerateRequested.connect(self.regenerateBrain3DCoronalPlanes)
        dock.applyEditsRequested.connect(self.applyCurrentCoronalEditsToBrain3DModel)
        dock.planeSelectionChanged.connect(self._onBrain3DPlaneSelectionChanged)
        dock.regionStateRequested.connect(self._onBrain3DRegionStateRequested)
        self.brain3d_session_dock = dock
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        try:
            if getattr(self, "viewer_layer_dock", None) is not None:
                self.tabifyDockWidget(dock, self.viewer_layer_dock)
            elif getattr(self, "shape_dock", None) is not None:
                self.tabifyDockWidget(dock, self.shape_dock)
        except Exception:
            pass
        return dock

    def openBrain3DSession(self, _value=False) -> None:
        dock = self._ensureBrain3DSessionDock()
        dock.show()
        dock.raise_()
        self._refreshBrain3DSessionDock()

    def _refreshBrain3DSessionDock(self) -> None:
        dock = getattr(self, "brain3d_session_dock", None)
        if not isinstance(dock, Brain3DSessionDockWidget):
            return
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None:
            dock.set_summary(region_count=0, source_page_count=0, plane_count=0)
            dock.set_regions([])
            return
        planes = reslice_brain_model(
            model,
            orientation="coronal",
            spacing=model.config.coronal_spacing,
            plane_count=model.config.coronal_plane_count,
        )
        plane_count = len(planes)
        current_plane = int(getattr(self, "frame_number", 0) or 0)
        if plane_count > 0:
            current_plane = min(max(current_plane, 0), plane_count - 1)
        source_page_count = int(model.metadata.get("source_page_count", 0) or 0)
        dock.set_summary(
            region_count=len(model.regions),
            source_page_count=source_page_count,
            plane_count=plane_count,
        )
        dock.set_current_plane(current_plane)
        if plane_count <= 0:
            dock.set_regions([])
            return
        plane = planes[current_plane]
        dock.set_regions(
            [
                {
                    "region_id": str(region.region_id),
                    "label": str(region.label),
                    "state": str(region.state),
                    "source": str(region.source),
                    "points_count": len(region.points),
                }
                for region in list(plane.regions or [])
            ]
        )

    def _onBrain3DPlaneSelectionChanged(self, plane_index: int) -> None:
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None:
            return
        target = int(plane_index)
        if bool(getattr(self, "_has_large_image_page_navigation", lambda: False)()):
            self.setLargeImagePageNumber(target)
        else:
            planes = reslice_brain_model(
                model,
                orientation="coronal",
                spacing=model.config.coronal_spacing,
                plane_count=model.config.coronal_plane_count,
            )
            if 0 <= target < len(planes):
                self.loadShapes(
                    materialize_coronal_plane_shapes(
                        planes[target], include_hidden=False
                    ),
                    replace=True,
                )
        self._refreshBrain3DSessionDock()

    def _onBrain3DRegionStateRequested(
        self, plane_index: int, region_id: str, state: str
    ) -> None:
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None:
            return
        set_region_presence_on_plane(
            model, int(plane_index), str(region_id), str(state)
        )
        self.otherData = store_brain_model_in_other_data(self.otherData, model)
        self.setDirty()
        self.regenerateBrain3DCoronalPlanes()

    def _collect_brain3d_sagittal_pages(self) -> list[dict[str, object]]:
        pages: list[dict[str, object]] = []
        if bool(getattr(self, "_has_large_image_page_navigation", lambda: False)()):
            backend = getattr(self, "large_image_backend", None)
            page_count = int(getattr(backend, "get_page_count", lambda: 1)() or 1)
            current_page = int(getattr(self, "frame_number", 0) or 0)
            for page_index in range(page_count):
                page = self._large_image_page_label_file(page_index)
                if page is not None:
                    _, label_file = page
                    shapes = self._materialize_label_shapes(
                        list(getattr(label_file, "shapes", []) or [])
                    )
                elif page_index == current_page:
                    shapes = list(getattr(self.canvas, "shapes", []) or [])
                else:
                    continue
                if not shapes:
                    continue
                pages.append({"page_index": int(page_index), "shapes": list(shapes)})
            return pages
        shapes = list(getattr(self.canvas, "shapes", []) or [])
        if shapes:
            pages.append({"page_index": 0, "shapes": shapes})
        return pages

    def buildBrain3DModelFromSagittalPages(self, _value=False) -> bool:
        pages = self._collect_brain3d_sagittal_pages()
        if not pages:
            post_window_status(
                self,
                self.tr("No sagittal polygon pages found for 3D reconstruction."),
                5000,
            )
            return False

        point_count = int(
            (
                (self._config or {}).get("brain3d_point_count")
                if isinstance(getattr(self, "_config", None), dict)
                else 64
            )
            or 64
        )
        spacing = (
            (self._config or {}).get("brain3d_coronal_spacing")
            if isinstance(getattr(self, "_config", None), dict)
            else 1.0
        )
        config = Brain3DConfig(point_count=max(3, point_count), coronal_spacing=spacing)
        model = build_brain_3d_model(pages, config)
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        self.otherData = store_brain_model_in_other_data(self.otherData, model)
        self.setDirty()
        post_window_status(
            self,
            self.tr("Built Brain 3D model from %d sagittal page(s).") % len(pages),
            5000,
        )
        self._refreshBrain3DSessionDock()
        return True

    @staticmethod
    def _brain3d_shape_to_labelme(shape) -> dict[str, object]:
        other = dict(getattr(shape, "other_data", {}) or {})
        return {
            "label": str(getattr(shape, "label", "") or ""),
            "points": [(float(p.x()), float(p.y())) for p in (shape.points or [])],
            "group_id": getattr(shape, "group_id", None),
            "shape_type": "polygon",
            "flags": dict(getattr(shape, "flags", {}) or {}),
            "description": str(getattr(shape, "description", "") or ""),
            "mask": None,
            "visible": bool(getattr(shape, "visible", True)),
            "shared_vertex_ids": list(getattr(shape, "shared_vertex_ids", []) or []),
            "shared_edge_ids": list(getattr(shape, "shared_edge_ids", []) or []),
            **other,
        }

    def regenerateBrain3DCoronalPlanes(self, _value=False) -> int:
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None and not self.buildBrain3DModelFromSagittalPages():
            return 0
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None:
            return 0

        planes = reslice_brain_model(
            model,
            orientation="coronal",
            spacing=model.config.coronal_spacing,
            plane_count=model.config.coronal_plane_count,
        )
        resolver = getattr(self, "_large_image_stack_label_path", None)
        write_pages = callable(resolver)
        written = 0
        image_path = str(getattr(self, "imagePath", "") or "")
        image_name = os.path.basename(image_path) if image_path else "image.png"
        image = getattr(self, "image", None)
        image_width = int(image.width()) if image is not None else 0
        image_height = int(image.height()) if image is not None else 0
        if model.image_shape is not None:
            image_width = max(image_width, int(model.image_shape[0] or 0))
            image_height = max(image_height, int(model.image_shape[1] or 0))

        for plane in planes:
            shapes = materialize_coronal_plane_shapes(plane, include_hidden=False)
            shape_payload = [
                self._brain3d_shape_to_labelme(shape) for shape in list(shapes or [])
            ]
            if write_pages:
                label_path = Path(
                    str(resolver(page_index=int(plane.plane_index)) or "")
                )
                if label_path:
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    page_other_data = dict(getattr(self, "otherData", {}) or {})
                    page_other_data = store_brain_model_in_other_data(
                        page_other_data, model
                    )
                    page_other_data["large_image_page"] = {
                        "page_index": int(plane.plane_index),
                        "brain3d_generated": True,
                        "brain3d_orientation": "coronal",
                        "brain3d_plane_position": float(plane.plane_position),
                    }
                    LabelFile().save(
                        filename=str(label_path),
                        shapes=shape_payload,
                        imagePath=image_name,
                        imageData=None,
                        imageHeight=max(1, int(image_height or 1)),
                        imageWidth=max(1, int(image_width or 1)),
                        otherData=page_other_data,
                        flags={},
                        caption=None,
                    )
                    written += 1

            if int(plane.plane_index) == int(getattr(self, "frame_number", 0) or 0):
                self.loadShapes(list(shapes), replace=True)

        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        self.otherData = store_brain_model_in_other_data(self.otherData, model)
        self.setDirty()
        post_window_status(
            self,
            self.tr("Regenerated %d coronal plane annotation page(s).") % int(written),
            5000,
        )
        self._refreshBrain3DSessionDock()
        return int(written)

    def applyCurrentCoronalEditsToBrain3DModel(self, _value=False) -> bool:
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None:
            post_window_status(
                self,
                self.tr("No Brain 3D model found. Build the model first."),
                5000,
            )
            return False
        plane_index = int(getattr(self, "frame_number", 0) or 0)
        shapes = list(getattr(self.canvas, "shapes", []) or [])
        updated = 0
        for shape in shapes:
            if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
                continue
            other = dict(getattr(shape, "other_data", {}) or {})
            region_id = str(other.get("region_id", "") or "")
            if not region_id:
                label, group_id, description = polygon_identity_key(shape)
                region_id = f"{label}|{group_id}|{description}"
            apply_coronal_polygon_edit(model, plane_index, region_id, shape)
            updated += 1
        if updated <= 0:
            post_window_status(self, self.tr("No polygon edits to apply."), 3000)
            return False
        self.otherData = store_brain_model_in_other_data(self.otherData, model)
        self.setDirty()
        post_window_status(
            self,
            self.tr("Applied %d coronal polygon edit(s) to Brain 3D model.") % updated,
            5000,
        )
        self._refreshBrain3DSessionDock()
        return True

    def _loadLargeImagePageAnnotations(
        self,
        page_index: int,
        *,
        fallback_shapes=None,
        fallback_other_data: Optional[dict] = None,
    ) -> bool:
        resolver = getattr(self, "_large_image_stack_label_path", None)
        if not callable(resolver):
            return False

        label_path_str = resolver(page_index=page_index)
        if not label_path_str:
            return False

        label_path = Path(label_path_str)
        baseline_other_data = self._large_image_page_baseline_other_data()

        def _apply_loaded_shapes(shape_payload):
            if shape_payload and isinstance(shape_payload[0], dict):
                self.loadLabels(shape_payload)
            else:
                self.loadShapes(shape_payload or [], replace=True)

        if label_path.exists():
            try:
                label_file = LabelFile(str(label_path), is_video_frame=True)
            except LabelFileError as exc:
                logger.error(
                    "Failed to load TIFF page label file %s: %s",
                    label_path,
                    exc,
                )
            except Exception as exc:
                logger.error(
                    "Unexpected error loading TIFF page label file %s: %s",
                    label_path,
                    exc,
                )
            else:
                self.labelFile = label_file
                merged_other_data = dict(baseline_other_data)
                merged_other_data.update(
                    dict(getattr(label_file, "otherData", {}) or {})
                )
                merged_other_data["large_image_page"] = {
                    "page_index": int(page_index),
                    "label_path": str(label_path),
                }
                self.otherData = merged_other_data
                if hasattr(self.canvas, "setBehaviorText"):
                    self.canvas.setBehaviorText(None)
                _apply_loaded_shapes(label_file.shapes)
                self.update_flags_from_file(label_file)
                caption = label_file.get_caption()
                if getattr(self, "caption_widget", None) is not None:
                    if caption:
                        self.caption_widget.set_caption(caption)
                    else:
                        self.caption_widget.set_caption("")
                    self.caption_widget.set_image_path(
                        str(getattr(self, "imagePath", "") or "")
                    )
                return True

        if not fallback_shapes:
            if self._infer_large_image_page_annotations(
                page_index, fallback_other_data=fallback_other_data
            ):
                return True

        self.labelFile = None
        merged_other_data = dict(baseline_other_data)
        if isinstance(fallback_other_data, dict):
            merged_other_data.update(fallback_other_data)
        merged_other_data["large_image_page"] = {
            "page_index": int(page_index),
            "label_path": str(label_path),
        }
        self.otherData = merged_other_data
        if hasattr(self.canvas, "setBehaviorText"):
            self.canvas.setBehaviorText(None)
        _apply_loaded_shapes(fallback_shapes or [])
        if getattr(self, "caption_widget", None) is not None:
            self.caption_widget.set_caption("")
            self.caption_widget.set_image_path(
                str(getattr(self, "imagePath", "") or "")
            )
        return bool(fallback_shapes)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        if replace:
            self.labelList.clear()
        for shape in shapes:
            if not isinstance(shape.points[0], QtCore.QPointF):
                shape.points = [QtCore.QPointF(x, y) for x, y in shape.points]
            self.addLabel(shape, rebuild_unique=False)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        rebuild_polygon_topology(shapes)
        self.canvas.loadShapes(shapes, replace=replace)
        if getattr(self, "large_image_view", None) is not None:
            self.large_image_view.set_shapes(self.canvas.shapes)
        if hasattr(self, "_refreshVectorOverlayDock"):
            self._refreshVectorOverlayDock()
        if hasattr(self, "_update_polygon_tool_action_state"):
            try:
                self._update_polygon_tool_action_state()
            except Exception:
                pass
        self._rebuild_unique_label_list()
        try:
            caption = self.labelFile.get_caption() if self.labelFile else None
        except AttributeError:
            caption = None
        if caption is not None and len(caption) > 0:
            if self.caption_widget is None:
                self.openCaption()
            self.caption_widget.set_caption(caption)
            self.caption_widget.set_image_path(self.filename)

    def update_flags_from_file(self, label_file):
        """Update flags from label file with proper validation and error handling."""
        if not hasattr(label_file, "flags"):
            logger.warning("Label file has no flags attribute")
            return

        try:
            if isinstance(label_file.flags, dict):
                new_flags = label_file.flags.copy()
                flags_in_frame = ",".join(new_flags.keys())
                self.canvas.setBehaviorText(flags_in_frame)
                _existing_flags = self.flag_widget._get_existing_flag_names()
                for _flag in _existing_flags:
                    if _flag not in new_flags:
                        new_flags[_flag] = False
                self.flag_widget.loadFlags(new_flags)
            else:
                logger.error(f"Invalid flags format: {type(label_file.flags)}")
        except Exception as e:
            logger.error(f"Error updating flags: {e}")

    def _annotation_store_has_frame(self, label_json_file: str) -> bool:
        """Return True if the annotation store contains a record for the given label path."""
        try:
            path = Path(label_json_file)
            frame_number = AnnotationStore.frame_number_from_path(path)
            if frame_number is None:
                return False
            store = AnnotationStore.for_frame_path(path)
            if not store.store_path.exists():
                return False
            return store.get_frame(frame_number) is not None
        except Exception:
            return False

    def _annotation_store_frame_count(self) -> int:
        """Return the number of frames currently stored in the annotation store."""
        if not self.video_results_folder:
            return 0
        try:
            store = AnnotationStore.for_frame_path(
                self.video_results_folder
                / f"{self.video_results_folder.name}_000000000.json"
            )
            if not store.store_path.exists():
                return 0
            return len(list(store.iter_frames()))
        except Exception:
            return 0

    def _tracking_rows_for_frame(self, frame_number: int) -> list[dict]:
        """Return tracking CSV rows for a frame without scanning the whole table."""
        controller = getattr(self, "tracking_data_controller", None)
        if controller is not None and hasattr(controller, "tracking_rows_for_frame"):
            try:
                return controller.tracking_rows_for_frame(frame_number)
            except Exception:
                logger.debug(
                    "Tracking controller frame lookup failed for frame %s.",
                    frame_number,
                    exc_info=True,
                )

        df = getattr(self, "_df", None)
        if df is None or getattr(df, "empty", True):
            return []

        frame_index = getattr(self, "_tracking_frame_indices", None)
        try:
            frame_key = int(frame_number)
        except (TypeError, ValueError):
            return []

        if isinstance(frame_index, dict):
            indices = frame_index.get(frame_key)
            if indices:
                try:
                    return df.iloc[list(indices)].to_dict(orient="records")
                except Exception:
                    logger.debug(
                        "Cached tracking frame lookup failed for frame %s.",
                        frame_number,
                        exc_info=True,
                    )

        try:
            return df[df.frame_number == frame_key].to_dict(orient="records")
        except Exception:
            logger.debug(
                "Fallback tracking lookup failed for frame %s.",
                frame_number,
                exc_info=True,
            )
            return []

    def _iter_frame_label_candidates(
        self, frame_number: int, frame_path: Optional[Path]
    ) -> list[Path]:
        """Return possible annotation paths for a given frame."""
        candidates: list[Path] = []

        def _append_candidate(path: Optional[Path]) -> None:
            if path is None:
                return
            if path not in candidates:
                candidates.append(path)

        if hasattr(self, "_has_large_image_page_navigation") and bool(
            self._has_large_image_page_navigation()
        ):
            resolver = getattr(self, "_large_image_stack_label_path", None)
            if callable(resolver):
                stack_label_path = resolver(page_index=frame_number)
                if stack_label_path:
                    _append_candidate(Path(stack_label_path))

        if frame_path is not None:
            frame_path = Path(frame_path)
            if frame_path.suffix.lower() == ".json":
                _append_candidate(frame_path)
            else:
                _append_candidate(frame_path.with_suffix(".json"))

            stem = frame_path.stem
            if "_" in stem:
                alt_name = f"{stem.split('_')[-1]}.json"
                _append_candidate(frame_path.parent / alt_name)

        frame_tag = f"{int(frame_number):09}"
        base_dir: Optional[Path] = None
        if frame_path is not None:
            base_dir = frame_path.parent
        if self.video_results_folder:
            base_dir = self.video_results_folder

        if base_dir is not None:
            if self.video_results_folder:
                stem_name = self.video_results_folder.name
            elif frame_path is not None:
                stem_name = frame_path.stem.rsplit("_", 1)[0]
            else:
                stem_name = base_dir.name

            if stem_name:
                _append_candidate(base_dir / f"{stem_name}_{frame_tag}.json")
            _append_candidate(base_dir / f"{frame_tag}.json")

        if self.video_results_folder:
            pred_dir = self.video_results_folder / self._pred_res_folder_suffix
            if pred_dir.exists():
                stem_name = self.video_results_folder.name
                _append_candidate(pred_dir / f"{stem_name}_{frame_tag}.json")
                _append_candidate(pred_dir / f"{frame_tag}.json")

        if self.annotation_dir:
            annot_dir = Path(self.annotation_dir)
            stem_name = annot_dir.name
            _append_candidate(annot_dir / f"{stem_name}_{frame_tag}.json")
            _append_candidate(annot_dir / f"{frame_tag}.json")

        return candidates

    def loadPredictShapes(self, frame_number, filename):
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(filename)

        frame_path = Path(filename) if filename else None
        label_candidates = self._iter_frame_label_candidates(frame_number, frame_path)

        seen_candidates: set[Path] = set()
        label_loaded = False
        for candidate in label_candidates:
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)

            candidate_exists = candidate.exists()
            candidate_in_store = self._annotation_store_has_frame(candidate)
            if not candidate_exists and not candidate_in_store:
                continue

            try:
                label_file = LabelFile(
                    str(candidate),
                    is_video_frame=True,
                )
            except LabelFileError as exc:
                logger.error(
                    "Failed to load label file %s: %s",
                    candidate,
                    exc,
                )
                continue
            except Exception as exc:
                logger.error(
                    "Unexpected error loading label file %s: %s",
                    candidate,
                    exc,
                )
                continue

            self.labelFile = label_file
            self.canvas.setBehaviorText(None)
            self.loadLabels(label_file.shapes)
            self.update_flags_from_file(label_file)
            if (
                len(self.canvas.current_behavior_text) > 1
                and "other" not in self.canvas.current_behavior_text.lower()
            ):
                self.add_highlighted_mark(
                    self.frame_number, mark_type=self.canvas.current_behavior_text
                )
            caption = label_file.get_caption()
            if caption is not None and len(caption) > 0:
                if self.caption_widget is None:
                    self.openCaption()
                self.caption_widget.set_caption(caption)
            elif self.caption_widget is not None:
                applied = self._apply_timeline_caption_if_available(
                    frame_number, only_if_empty=False
                )
                if not applied:
                    self.caption_widget.set_caption("")
            label_loaded = True
            break

        if label_loaded:
            return

        if self._df is not None and (frame_path is None or not frame_path.exists()):
            frame_rows = self._tracking_rows_for_frame(frame_number)
            if frame_rows:
                frame_label_list = []
                for row in frame_rows:
                    row = dict(row)
                    if "x1" not in row:
                        row["x1"] = 2
                        row["y1"] = 2
                        row["x2"] = 4
                        row["y2"] = 4
                        row["class_score"] = 1
                        try:
                            instance_names = [
                                col
                                for col, value in row.items()
                                if col != "frame_number"
                                and isinstance(value, Number)
                                and value > 0
                            ]
                            row["instance_name"] = (
                                "_".join(instance_names)
                                if instance_names
                                else "unknown"
                            )
                        except Exception:
                            row["instance_name"] = "unknown"
                        row["segmentation"] = None
                    pred_label_list = pred_dict_to_labelme(row)
                    frame_label_list += pred_label_list

                self.loadShapes(frame_label_list)

        if not label_loaded and self.caption_widget is not None:
            applied = self._apply_timeline_caption_if_available(
                frame_number, only_if_empty=False
            )
            if not applied:
                self.caption_widget.set_caption("")
