from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from collections import OrderedDict, deque
from numbers import Number
import os
import tempfile
import time
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

from qtpy import QtCore, QtWidgets

from annolid.gui.brain_3d_model import (
    Brain3DConfig,
    apply_coronal_polygon_edit,
    brain_model_from_other_data,
    build_brain_3d_model,
    export_brain_model_mesh_obj,
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
from annolid.gui.widgets.zone_manager_utils import (
    is_zone_shape,
    zone_file_for_source,
    zone_payload_to_shape,
)
from annolid.infrastructure import AnnotationStore
from annolid.postprocessing.zone_schema import load_zone_shapes
from annolid.postprocessing.quality_control import pred_dict_to_labelme
from annolid.utils.logger import logger


class AnnotationLoadingMixin:
    """Annotation/label loading workflow for frames and images."""

    _BRAIN3D_HIGHLIGHT_STROKE_KEY = "brain3d_overlay_stroke"
    _BRAIN3D_HIGHLIGHT_FILL_KEY = "brain3d_overlay_fill"
    _BRAIN3D_HIGHLIGHT_FLAG_KEY = "brain3d_highlight"
    _BRAIN3D_HIGHLIGHT_MODE_KEY = "brain3d_highlight_mode"
    _BRAIN3D_PENDING_REGION_KEY = "_brain3d_pending_region_id"
    _BRAIN3D_INTERNAL_MUTATION_KEY = "_brain3d_internal_mutation_in_progress"
    _ZONE_OVERLAY_CACHE_PATH_KEY = "_zone_overlay_cache_path"
    _ZONE_OVERLAY_CACHE_MTIME_KEY = "_zone_overlay_cache_mtime_ns"
    _ZONE_OVERLAY_CACHE_PAYLOADS_KEY = "_zone_overlay_cache_payloads"
    _FRAME_LABEL_CACHE_KEY = "_frame_label_cache"
    _FRAME_LABEL_CACHE_MAX_ENTRIES = 48
    _FRAME_STORE_HAS_FRAME_CACHE_KEY = "_frame_store_has_frame_cache"
    _FRAME_LABEL_SOURCE_INDEX_CACHE_KEY = "_frame_label_source_index_cache"
    _FRAME_LABEL_PREFETCH_QUEUE_KEY = "_frame_label_prefetch_queue"
    _FRAME_LABEL_PREFETCH_ACTIVE_KEY = "_frame_label_prefetch_active"
    _FRAME_LABEL_PREFETCH_WINDOW = 4
    _FRAME_LABEL_PREFETCH_BUDGET = 2
    _TRACKING_SHAPE_CACHE_KEY = "_tracking_shape_cache"
    _TRACKING_SHAPE_CACHE_MAX_ENTRIES = 128
    _TRACKING_KEYPOINT_AREA_THRESHOLD = 16
    _PREFER_ANNOTATION_STORE_OVER_FRAME_JSON = True
    _ENABLE_TRACKING_CSV_SHAPE_FALLBACK = False
    # Correctness-first default: keep label/store probing during playback so
    # frame-specific predictions are not skipped. The tracking fast path remains
    # available as an explicit opt-in for constrained environments.
    _PREFER_TRACKING_OVER_LABEL_IO_DURING_PLAYBACK = False

    @staticmethod
    def _brain3d_polygon_signature(shapes) -> str:
        entries: list[tuple[str, str, str, tuple[tuple[float, float], ...]]] = []
        for shape in list(shapes or []):
            if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
                continue
            points = [
                (round(float(point.x()), 4), round(float(point.y()), 4))
                for point in list(getattr(shape, "points", []) or [])
            ]
            if len(points) < 3:
                continue
            entries.append(
                (
                    str(getattr(shape, "label", "") or ""),
                    str(
                        ""
                        if getattr(shape, "group_id", None) is None
                        else getattr(shape, "group_id")
                    ),
                    str(getattr(shape, "description", "") or ""),
                    tuple(points),
                )
            )
        entries.sort(key=lambda value: (value[0], value[1], value[2], len(value[3])))
        payload = repr(entries).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _zone_shape_signature(shape: Shape) -> tuple:
        points = []
        for point in list(getattr(shape, "points", []) or []):
            x = point.x() if hasattr(point, "x") else point[0]
            y = point.y() if hasattr(point, "y") else point[1]
            points.append((round(float(x), 3), round(float(y), 3)))
        return (
            str(getattr(shape, "label", "") or ""),
            str(getattr(shape, "shape_type", "") or ""),
            tuple(points),
        )

    @staticmethod
    def _clone_zone_shapes(shapes) -> list[Shape]:
        cloned: list[Shape] = []
        for shape in list(shapes or []):
            if not is_zone_shape(shape):
                continue
            payload = {
                "label": str(getattr(shape, "label", "") or ""),
                "description": str(getattr(shape, "description", "") or ""),
                "shape_type": str(getattr(shape, "shape_type", "polygon") or "polygon"),
                "points": [
                    [float(point.x()), float(point.y())]
                    for point in list(getattr(shape, "points", []) or [])
                ],
                "flags": dict(getattr(shape, "flags", {}) or {}),
                "group_id": getattr(shape, "group_id", None),
                "visible": bool(getattr(shape, "visible", True)),
            }
            cloned.append(zone_payload_to_shape(payload))
        return cloned

    def _display_zones_on_all_frames_enabled(self) -> bool:
        return bool(getattr(self, "_display_zones_on_all_frames", True))

    def _zone_overlay_candidate_files(self, frame_path: Optional[Path]) -> list[Path]:
        candidates: list[Path] = []

        def _append(path_value) -> None:
            if not path_value:
                return
            path = Path(path_value).expanduser()
            if path not in candidates:
                candidates.append(path)

        zone_path = getattr(self, "zone_path", None)
        if zone_path:
            _append(zone_path)
        _append(zone_file_for_source(getattr(self, "video_file", None)))
        _append(zone_file_for_source(getattr(self, "filename", None)))
        if frame_path is not None:
            _append(zone_file_for_source(frame_path))
        return [path for path in candidates if path.exists() and path.is_file()]

    def _load_zone_payloads_from_file(self, zone_file: Path) -> list[dict]:
        try:
            stat_result = zone_file.stat()
            mtime_ns = int(getattr(stat_result, "st_mtime_ns", 0))
        except OSError:
            return []

        cached_path = getattr(self, self._ZONE_OVERLAY_CACHE_PATH_KEY, None)
        cached_mtime = getattr(self, self._ZONE_OVERLAY_CACHE_MTIME_KEY, None)
        cached_payloads = getattr(self, self._ZONE_OVERLAY_CACHE_PAYLOADS_KEY, None)
        if (
            cached_path == str(zone_file)
            and cached_mtime == mtime_ns
            and isinstance(cached_payloads, list)
        ):
            return [dict(payload) for payload in cached_payloads]

        try:
            payload = json.loads(zone_file.read_text(encoding="utf-8"))
            specs = load_zone_shapes(payload)
            normalized = [spec.to_shape_dict() for spec in specs]
        except Exception:
            logger.debug(
                "Failed to load zone overlay file: %s", zone_file, exc_info=True
            )
            normalized = []

        setattr(self, self._ZONE_OVERLAY_CACHE_PATH_KEY, str(zone_file))
        setattr(self, self._ZONE_OVERLAY_CACHE_MTIME_KEY, mtime_ns)
        setattr(
            self,
            self._ZONE_OVERLAY_CACHE_PAYLOADS_KEY,
            [dict(item) for item in normalized],
        )
        return [dict(payload) for payload in normalized]

    def _persistent_zone_shapes_for_frame(
        self, frame_path: Optional[Path]
    ) -> list[Shape]:
        if not self._display_zones_on_all_frames_enabled():
            return []
        zone_shapes = self._clone_zone_shapes(getattr(self.canvas, "shapes", []) or [])
        for zone_file in self._zone_overlay_candidate_files(frame_path):
            for payload in self._load_zone_payloads_from_file(zone_file):
                try:
                    zone_shapes.append(zone_payload_to_shape(payload))
                except Exception:
                    logger.debug(
                        "Failed to materialize zone payload from %s",
                        zone_file,
                        exc_info=True,
                    )
        if not zone_shapes:
            return []
        deduped: list[Shape] = []
        seen: set[tuple] = set()
        for shape in zone_shapes:
            signature = self._zone_shape_signature(shape)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(shape)
        return deduped

    def _merge_persistent_zones_into_shapes(
        self,
        shapes,
        frame_path: Optional[Path],
        *,
        persistent_zone_shapes: Optional[list[Shape]] = None,
    ) -> list[Shape]:
        merged: list[Shape] = []
        for shape in list(shapes or []):
            if isinstance(shape, Shape):
                merged.append(shape)
            else:
                merged.extend(self._materialize_label_shapes([shape]))
        existing = {
            self._zone_shape_signature(shape)
            for shape in merged
            if is_zone_shape(shape)
        }
        zone_shapes = (
            list(persistent_zone_shapes or [])
            if persistent_zone_shapes is not None
            else self._persistent_zone_shapes_for_frame(frame_path)
        )
        for zone_shape in zone_shapes:
            signature = self._zone_shape_signature(zone_shape)
            if signature in existing:
                continue
            existing.add(signature)
            merged.append(zone_shape)
        return merged

    def _brain3d_current_page_index(self) -> int:
        return int(getattr(self, "frame_number", 0) or 0)

    def _brain3d_current_page_info(self) -> dict:
        if not isinstance(getattr(self, "otherData", None), dict):
            return {}
        return dict(self.otherData.get("large_image_page") or {})

    def _brain3d_is_current_page_generated_coronal(self) -> bool:
        page_info = self._brain3d_current_page_info()
        if bool(page_info.get("brain3d_generated", False)):
            return True
        if (
            str(page_info.get("brain3d_orientation", "") or "").strip().lower()
            == "coronal"
        ):
            return True
        return False

    def _set_dirty_with_brain3d_internal_guard(self) -> None:
        setattr(self, self._BRAIN3D_INTERNAL_MUTATION_KEY, True)
        try:
            self.setDirty()
        finally:
            setattr(self, self._BRAIN3D_INTERNAL_MUTATION_KEY, False)

    def _invalidate_brain3d_model(self, *, reason: str) -> bool:
        if not isinstance(getattr(self, "otherData", None), dict):
            return False
        if "brain_3d_model" not in self.otherData:
            return False
        updated = dict(self.otherData)
        updated.pop("brain_3d_model", None)
        sync = dict(updated.get("brain3d_sync") or {})
        sync.update(
            {
                "valid": False,
                "invalidated": True,
                "reason": str(reason or "source_update"),
                "page_index": int(self._brain3d_current_page_index()),
                "source_orientation": "sagittal",
                "updated_at_epoch_s": float(time.time()),
            }
        )
        updated["brain3d_sync"] = sync
        self.otherData = updated
        post_window_status(
            self,
            self.tr(
                "Brain 3D model invalidated by sagittal source edits. Rebuild or regenerate before continuing."
            ),
            5000,
        )
        return True

    def _onAnnotationDirty(self) -> None:
        if bool(getattr(self, self._BRAIN3D_INTERNAL_MUTATION_KEY, False)):
            return
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None:
            return
        if (
            str(getattr(model, "source_orientation", "") or "sagittal").lower()
            != "sagittal"
        ):
            return
        if self._brain3d_is_current_page_generated_coronal():
            return
        source_signatures = dict(
            getattr(model, "metadata", {}).get("source_page_signatures") or {}
        )
        source_indices = {
            int(value)
            for value in list(
                getattr(model, "metadata", {}).get("source_page_indices") or []
            )
            if isinstance(value, Number)
        }
        page_index = int(self._brain3d_current_page_index())
        if source_indices and page_index not in source_indices:
            return
        if not source_signatures and not source_indices:
            self._invalidate_brain3d_model(reason="source_page_changed")
            return
        expected = str(source_signatures.get(str(page_index), "") or "")
        if not expected:
            self._invalidate_brain3d_model(reason="source_page_changed")
            return
        current = self._brain3d_polygon_signature(
            getattr(self.canvas, "shapes", []) or []
        )
        if current != expected:
            self._invalidate_brain3d_model(reason="source_page_changed")

    def _brain3d_prepare_save(self, _filename: str) -> None:
        if bool(getattr(self, self._BRAIN3D_INTERNAL_MUTATION_KEY, False)):
            return
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None:
            return
        if not self._brain3d_is_current_page_generated_coronal():
            return
        setattr(self, self._BRAIN3D_INTERNAL_MUTATION_KEY, True)
        try:
            self.applyCurrentCoronalEditsToBrain3DModel()
        finally:
            setattr(self, self._BRAIN3D_INTERNAL_MUTATION_KEY, False)

    def _materialize_label_shapes(self, shapes):
        s = []
        for shape_data in shapes:
            label = shape_data["label"]
            points = shape_data["points"]
            shape_type = shape_data.get("shape_type", "polygon")
            flags = dict(shape_data.get("flags") or {})
            group_id = shape_data.get("group_id")
            description = shape_data.get("description", "")
            other_data = dict(shape_data.get("other_data") or {})
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
                mask=shape_data.get("mask"),
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
        dock.openPreviewRequested.connect(self.openBrain3DMeshPreview)
        dock.planeSelectionChanged.connect(self._onBrain3DPlaneSelectionChanged)
        dock.regionSelectionChanged.connect(self._onBrain3DRegionSelectionChanged)
        dock.highlightModeChanged.connect(self._onBrain3DHighlightModeChanged)
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
        if isinstance(getattr(self, "otherData", None), dict):
            mode = str(
                (self.otherData.get(self._BRAIN3D_HIGHLIGHT_MODE_KEY) or "region_only")
            )
            dock.set_highlight_mode(mode)
        self._refreshBrain3DSessionDock()

    def _refreshBrain3DSessionDock(self) -> None:
        dock = getattr(self, "brain3d_session_dock", None)
        if not isinstance(dock, Brain3DSessionDockWidget):
            return
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None:
            dock.set_summary(region_count=0, source_page_count=0, plane_count=0)
            dock.set_regions([])
            dock.set_highlight_summary(
                highlighted_count=0, total_polygons=0, mode="region_only"
            )
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
            dock.set_highlight_summary(
                highlighted_count=0, total_polygons=0, mode="region_only"
            )
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
        pending_region = str(getattr(self, self._BRAIN3D_PENDING_REGION_KEY, "") or "")
        if pending_region:
            if dock.select_region(pending_region, emit_signal=False):
                setattr(self, self._BRAIN3D_PENDING_REGION_KEY, "")
        self._refreshBrain3DHighlightSummary()

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
        self._set_dirty_with_brain3d_internal_guard()
        self.regenerateBrain3DCoronalPlanes(local_only=True)

    def _onBrain3DHighlightModeChanged(self, mode: str) -> None:
        normalized = str(mode or "").strip().lower()
        if normalized not in {"region_only", "label_group"}:
            normalized = "region_only"
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        self.otherData[self._BRAIN3D_HIGHLIGHT_MODE_KEY] = normalized
        dock = getattr(self, "brain3d_session_dock", None)
        if isinstance(dock, Brain3DSessionDockWidget):
            self._applyBrain3DRegionHighlight(dock.selected_region_id())
            self._refreshBrain3DHighlightSummary()

    def _onBrain3DRegionSelectionChanged(self, region_id: str) -> None:
        target = str(region_id or "")
        if not target:
            self._clearBrain3DRegionHighlight()
            return
        if not self._focusBrain3DRegionOnNearestPlane(target):
            self._clearBrain3DRegionHighlight()
            return
        canvas = getattr(self, "canvas", None)
        matched = []
        for shape in list(getattr(canvas, "shapes", []) or []):
            if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
                continue
            other = dict(getattr(shape, "other_data", {}) or {})
            region = str(other.get("region_id", "") or "")
            if not region:
                polygon_edit = dict(other.get("polygon_edit") or {})
                region = str(polygon_edit.get("region_id", "") or "")
            if region == target:
                matched.append(shape)
        self._applyBrain3DRegionHighlight(target)
        if not matched:
            return
        setattr(self, self._BRAIN3D_PENDING_REGION_KEY, "")
        try:
            canvas.selectShapes(matched)
        except Exception:
            try:
                canvas.selectedShapes = list(matched)
            except Exception:
                return
        if hasattr(self, "shapeSelectionChanged"):
            try:
                self.shapeSelectionChanged(matched)
            except Exception:
                pass

    def _brain3d_best_plane_for_region(self, model, region_id: str) -> int | None:
        target = str(region_id or "")
        if not target:
            return None
        planes = reslice_brain_model(
            model,
            orientation="coronal",
            spacing=model.config.coronal_spacing,
            plane_count=model.config.coronal_plane_count,
        )
        available_polygon: list[int] = []
        available_state_only: list[int] = []
        for plane in list(planes or []):
            for region in list(getattr(plane, "regions", []) or []):
                if str(getattr(region, "region_id", "") or "") != target:
                    continue
                state = str(getattr(region, "state", "present") or "present")
                if state in {"hidden", "zero_area"}:
                    continue
                plane_index = int(getattr(plane, "plane_index", -1))
                available_state_only.append(plane_index)
                if len(list(getattr(region, "points", []) or [])) >= 3:
                    available_polygon.append(plane_index)
                break
        available = available_polygon or available_state_only
        if not available:
            return None
        current = int(getattr(self, "frame_number", 0) or 0)
        return min(available, key=lambda idx: (abs(int(idx) - current), int(idx)))

    def _focusBrain3DRegionOnNearestPlane(self, region_id: str) -> bool:
        target = str(region_id or "")
        if not target:
            return False
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None:
            return False
        desired_plane = self._brain3d_best_plane_for_region(model, target)
        if desired_plane is None:
            return False
        current_plane = int(getattr(self, "frame_number", 0) or 0)
        if int(desired_plane) == int(current_plane):
            return True
        if bool(getattr(self, "_has_large_image_page_navigation", lambda: False)()):
            try:
                self.setLargeImagePageNumber(int(desired_plane))
                return True
            except Exception:
                return False
        try:
            planes = reslice_brain_model(
                model,
                orientation="coronal",
                spacing=model.config.coronal_spacing,
                plane_count=model.config.coronal_plane_count,
            )
            plane_by_index = {
                int(getattr(plane, "plane_index", -1)): plane for plane in list(planes)
            }
            plane = plane_by_index.get(int(desired_plane))
            if plane is None:
                return False
            self.loadShapes(
                materialize_coronal_plane_shapes(plane, include_hidden=False),
                replace=True,
            )
            self.frame_number = int(desired_plane)
            return True
        except Exception:
            return False

    @staticmethod
    def _brain3d_region_id_from_shape(shape) -> str:
        other = dict(getattr(shape, "other_data", {}) or {})
        region = str(other.get("region_id", "") or "")
        if region:
            return region
        polygon_edit = dict(other.get("polygon_edit") or {})
        return str(polygon_edit.get("region_id", "") or "")

    def _clearBrain3DRegionHighlight(self) -> None:
        canvas = getattr(self, "canvas", None)
        changed = False
        for shape in list(getattr(canvas, "shapes", []) or []):
            other = dict(getattr(shape, "other_data", {}) or {})
            if (
                self._BRAIN3D_HIGHLIGHT_FLAG_KEY in other
                or self._BRAIN3D_HIGHLIGHT_STROKE_KEY in other
                or self._BRAIN3D_HIGHLIGHT_FILL_KEY in other
            ):
                other.pop(self._BRAIN3D_HIGHLIGHT_FLAG_KEY, None)
                other.pop(self._BRAIN3D_HIGHLIGHT_STROKE_KEY, None)
                other.pop(self._BRAIN3D_HIGHLIGHT_FILL_KEY, None)
                shape.other_data = other
                changed = True
        if changed:
            try:
                if canvas is not None:
                    canvas.update()
            except Exception:
                pass
            large_view = getattr(self, "large_image_view", None)
            if large_view is not None:
                try:
                    large_view.viewport().update()
                except Exception:
                    pass
        self._refreshBrain3DHighlightSummary()

    def _applyBrain3DRegionHighlight(self, region_id: str) -> None:
        target = str(region_id or "")
        if not target:
            self._clearBrain3DRegionHighlight()
            return
        mode = "region_only"
        if isinstance(getattr(self, "otherData", None), dict):
            mode = (
                str(
                    self.otherData.get(self._BRAIN3D_HIGHLIGHT_MODE_KEY)
                    or "region_only"
                )
                .strip()
                .lower()
            )
        target_label = target.split("|", 1)[0] if "|" in target else target
        canvas = getattr(self, "canvas", None)
        changed = False
        for shape in list(getattr(canvas, "shapes", []) or []):
            other = dict(getattr(shape, "other_data", {}) or {})
            shape_region = self._brain3d_region_id_from_shape(shape)
            shape_label = (
                shape_region.split("|", 1)[0] if "|" in shape_region else shape_region
            )
            match = shape_region == target
            if mode == "label_group" and target_label:
                match = bool(shape_label and shape_label == target_label)
            if match:
                other[self._BRAIN3D_HIGHLIGHT_FLAG_KEY] = True
                other[self._BRAIN3D_HIGHLIGHT_STROKE_KEY] = "#ffb300"
                other[self._BRAIN3D_HIGHLIGHT_FILL_KEY] = "#ffb300"
            else:
                other.pop(self._BRAIN3D_HIGHLIGHT_FLAG_KEY, None)
                other.pop(self._BRAIN3D_HIGHLIGHT_STROKE_KEY, None)
                other.pop(self._BRAIN3D_HIGHLIGHT_FILL_KEY, None)
            if dict(getattr(shape, "other_data", {}) or {}) != other:
                shape.other_data = other
                changed = True
        if changed:
            try:
                if canvas is not None:
                    canvas.update()
            except Exception:
                pass
            large_view = getattr(self, "large_image_view", None)
            if large_view is not None:
                try:
                    large_view.viewport().update()
                except Exception:
                    pass
        self._refreshBrain3DHighlightSummary()

    def _refreshBrain3DHighlightSummary(self) -> None:
        dock = getattr(self, "brain3d_session_dock", None)
        if not isinstance(dock, Brain3DSessionDockWidget):
            return
        canvas = getattr(self, "canvas", None)
        shapes = [
            shape
            for shape in list(getattr(canvas, "shapes", []) or [])
            if str(getattr(shape, "shape_type", "") or "").lower() == "polygon"
        ]
        highlighted = 0
        for shape in shapes:
            other = dict(getattr(shape, "other_data", {}) or {})
            if bool(other.get(self._BRAIN3D_HIGHLIGHT_FLAG_KEY, False)):
                highlighted += 1
        mode = "region_only"
        if isinstance(getattr(self, "otherData", None), dict):
            mode = (
                str(
                    self.otherData.get(self._BRAIN3D_HIGHLIGHT_MODE_KEY)
                    or "region_only"
                )
                .strip()
                .lower()
            )
        dock.set_highlight_summary(
            highlighted_count=highlighted,
            total_polygons=len(shapes),
            mode=mode,
        )

    def _syncBrain3DSelectionFromShapes(self, selected_shapes) -> None:
        dock = getattr(self, "brain3d_session_dock", None)
        if not isinstance(dock, Brain3DSessionDockWidget):
            return
        for shape in list(selected_shapes or []):
            region = self._brain3d_region_id_from_shape(shape)
            if not region:
                continue
            self._applyBrain3DRegionHighlight(region)
            if dock.select_region(region, emit_signal=False):
                setattr(self, self._BRAIN3D_PENDING_REGION_KEY, "")
                return
            setattr(self, self._BRAIN3D_PENDING_REGION_KEY, region)
        self._clearBrain3DRegionHighlight()

    def openBrain3DMeshPreview(self, _value=False) -> bool:
        model = brain_model_from_other_data(getattr(self, "otherData", None))
        if model is None:
            post_window_status(
                self, self.tr("No Brain 3D model found. Build the model first."), 5000
            )
            return False
        manager = getattr(self, "threejs_manager", None)
        if manager is None:
            post_window_status(self, self.tr("Three.js viewer is not available."), 5000)
            return False
        try:
            export_dir = Path(tempfile.gettempdir()) / "annolid_brain3d_preview"
            export_dir.mkdir(parents=True, exist_ok=True)
            filename = f"brain3d_preview_{int(time.time() * 1000)}.obj"
            mesh_path, object_region_map = export_brain_model_mesh_obj(
                model,
                export_dir / filename,
            )
        except Exception as exc:
            post_window_status(
                self,
                self.tr("Failed to export Brain 3D mesh preview: %s") % str(exc),
                5000,
            )
            return False
        try:
            opened = bool(
                manager.show_model_in_viewer(
                    str(mesh_path),
                    pick_mode="brain3d_region",
                    object_region_map=object_region_map,
                )
            )
        except Exception:
            opened = False
        if not opened:
            post_window_status(
                self,
                self.tr("Unable to open Brain 3D mesh preview in Three.js viewer."),
                5000,
            )
            return False
        post_window_status(
            self,
            self.tr("Opened Brain 3D mesh preview: %s") % str(mesh_path.name),
            4000,
        )
        return True

    def _onBrain3DMeshRegionPicked(self, region_id: str) -> None:
        target = str(region_id or "")
        if not target:
            return
        setattr(self, self._BRAIN3D_PENDING_REGION_KEY, target)
        dock = getattr(self, "brain3d_session_dock", None)
        if isinstance(dock, Brain3DSessionDockWidget):
            if not dock.select_region(target, emit_signal=True):
                self._onBrain3DRegionSelectionChanged(target)
        else:
            self._onBrain3DRegionSelectionChanged(target)

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

    def _brain3d_config_from_overrides(
        self,
        overrides: dict[str, object] | None = None,
    ) -> Brain3DConfig:
        point_count = int(
            (
                (self._config or {}).get("brain3d_point_count")
                if isinstance(getattr(self, "_config", None), dict)
                else 64
            )
            or 64
        )
        interpolation_density = int(
            (
                (self._config or {}).get("brain3d_interpolation_density")
                if isinstance(getattr(self, "_config", None), dict)
                else 1
            )
            or 1
        )
        spacing = (
            (self._config or {}).get("brain3d_coronal_spacing")
            if isinstance(getattr(self, "_config", None), dict)
            else 1.0
        )
        plane_count = (
            (self._config or {}).get("brain3d_coronal_plane_count")
            if isinstance(getattr(self, "_config", None), dict)
            else None
        )
        smoothing_longitudinal = (
            (self._config or {}).get("brain3d_smoothing_longitudinal")
            if isinstance(getattr(self, "_config", None), dict)
            else 0.0
        )
        smoothing_inplane = (
            (self._config or {}).get("brain3d_smoothing_inplane")
            if isinstance(getattr(self, "_config", None), dict)
            else 0.0
        )
        snapping_enabled = bool(
            (self._config or {}).get("brain3d_snapping_enabled", False)
            if isinstance(getattr(self, "_config", None), dict)
            else False
        )
        snapping_strength = (
            (self._config or {}).get("brain3d_snapping_strength")
            if isinstance(getattr(self, "_config", None), dict)
            else 0.0
        )
        snapping_max_distance = (
            (self._config or {}).get("brain3d_snapping_max_distance")
            if isinstance(getattr(self, "_config", None), dict)
            else 8.0
        )
        if isinstance(overrides, dict):
            point_count = int(overrides.get("point_count", point_count) or point_count)
            interpolation_density = int(
                overrides.get("interpolation_density", interpolation_density)
                or interpolation_density
            )
            spacing = overrides.get("coronal_spacing", spacing)
            plane_count = overrides.get("coronal_plane_count", plane_count)
            smoothing_longitudinal = overrides.get(
                "smoothing_longitudinal", smoothing_longitudinal
            )
            smoothing_inplane = overrides.get("smoothing_inplane", smoothing_inplane)
            snapping_enabled = bool(overrides.get("snapping_enabled", snapping_enabled))
            snapping_strength = overrides.get("snapping_strength", snapping_strength)
            snapping_max_distance = overrides.get(
                "snapping_max_distance", snapping_max_distance
            )
        return Brain3DConfig(
            point_count=max(3, int(point_count)),
            interpolation_density=max(1, int(interpolation_density)),
            coronal_spacing=(None if spacing is None else float(spacing)),
            coronal_plane_count=(
                None if plane_count in (None, "") else int(plane_count)
            ),
            smoothing_longitudinal=float(smoothing_longitudinal or 0.0),
            smoothing_inplane=float(smoothing_inplane or 0.0),
            snapping_enabled=bool(snapping_enabled),
            snapping_strength=float(snapping_strength or 0.0),
            snapping_max_distance=max(0.1, float(snapping_max_distance or 8.0)),
        )

    def _promptBrain3DReconstructionConfig(self) -> dict[str, object] | None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("Build Brain 3D Reconstruction"))
        form = QtWidgets.QFormLayout(dialog)

        mode_combo = QtWidgets.QComboBox(dialog)
        mode_combo.addItem(self.tr("By spacing"), "spacing")
        mode_combo.addItem(self.tr("By plane count"), "count")

        spacing_spin = QtWidgets.QDoubleSpinBox(dialog)
        spacing_spin.setRange(0.1, 10000.0)
        spacing_spin.setDecimals(3)
        spacing_spin.setValue(
            float(
                ((self._config or {}).get("brain3d_coronal_spacing", 1.0))
                if isinstance(getattr(self, "_config", None), dict)
                else 1.0
            )
        )

        plane_count_spin = QtWidgets.QSpinBox(dialog)
        plane_count_spin.setRange(0, 20000)
        plane_count_spin.setValue(
            int(
                ((self._config or {}).get("brain3d_coronal_plane_count", 0))
                if isinstance(getattr(self, "_config", None), dict)
                else 0
            )
        )

        point_count_spin = QtWidgets.QSpinBox(dialog)
        point_count_spin.setRange(3, 4096)
        point_count_spin.setValue(
            int(
                ((self._config or {}).get("brain3d_point_count", 64))
                if isinstance(getattr(self, "_config", None), dict)
                else 64
            )
        )
        interpolation_density_spin = QtWidgets.QSpinBox(dialog)
        interpolation_density_spin.setRange(1, 32)
        interpolation_density_spin.setValue(
            int(
                ((self._config or {}).get("brain3d_interpolation_density", 1))
                if isinstance(getattr(self, "_config", None), dict)
                else 1
            )
        )

        smooth_long_spin = QtWidgets.QDoubleSpinBox(dialog)
        smooth_long_spin.setRange(0.0, 1.0)
        smooth_long_spin.setDecimals(3)
        smooth_long_spin.setSingleStep(0.05)
        smooth_long_spin.setValue(
            float(
                ((self._config or {}).get("brain3d_smoothing_longitudinal", 0.0))
                if isinstance(getattr(self, "_config", None), dict)
                else 0.0
            )
        )

        smooth_plane_spin = QtWidgets.QDoubleSpinBox(dialog)
        smooth_plane_spin.setRange(0.0, 1.0)
        smooth_plane_spin.setDecimals(3)
        smooth_plane_spin.setSingleStep(0.05)
        smooth_plane_spin.setValue(
            float(
                ((self._config or {}).get("brain3d_smoothing_inplane", 0.0))
                if isinstance(getattr(self, "_config", None), dict)
                else 0.0
            )
        )
        snapping_enabled_check = QtWidgets.QCheckBox(dialog)
        snapping_enabled_check.setChecked(
            bool(
                ((self._config or {}).get("brain3d_snapping_enabled", False))
                if isinstance(getattr(self, "_config", None), dict)
                else False
            )
        )
        snapping_strength_spin = QtWidgets.QDoubleSpinBox(dialog)
        snapping_strength_spin.setRange(0.0, 1.0)
        snapping_strength_spin.setDecimals(3)
        snapping_strength_spin.setSingleStep(0.05)
        snapping_strength_spin.setValue(
            float(
                ((self._config or {}).get("brain3d_snapping_strength", 0.0))
                if isinstance(getattr(self, "_config", None), dict)
                else 0.0
            )
        )
        snapping_distance_spin = QtWidgets.QDoubleSpinBox(dialog)
        snapping_distance_spin.setRange(0.1, 1000.0)
        snapping_distance_spin.setDecimals(2)
        snapping_distance_spin.setSingleStep(0.5)
        snapping_distance_spin.setValue(
            float(
                ((self._config or {}).get("brain3d_snapping_max_distance", 8.0))
                if isinstance(getattr(self, "_config", None), dict)
                else 8.0
            )
        )

        form.addRow(self.tr("Coronal output"), mode_combo)
        form.addRow(self.tr("Coronal spacing"), spacing_spin)
        form.addRow(self.tr("Coronal plane count"), plane_count_spin)
        form.addRow(self.tr("Contour point count"), point_count_spin)
        form.addRow(self.tr("Interpolation density"), interpolation_density_spin)
        form.addRow(self.tr("Longitudinal smoothing"), smooth_long_spin)
        form.addRow(self.tr("Coronal in-plane smoothing"), smooth_plane_spin)
        form.addRow(self.tr("Enable reference snapping"), snapping_enabled_check)
        form.addRow(self.tr("Snapping strength"), snapping_strength_spin)
        form.addRow(self.tr("Snapping max distance"), snapping_distance_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None

        mode = str(mode_combo.currentData() or "spacing")
        return {
            "point_count": int(point_count_spin.value()),
            "interpolation_density": int(interpolation_density_spin.value()),
            "coronal_spacing": float(spacing_spin.value())
            if mode == "spacing"
            else None,
            "coronal_plane_count": (
                int(plane_count_spin.value()) if mode == "count" else None
            ),
            "smoothing_longitudinal": float(smooth_long_spin.value()),
            "smoothing_inplane": float(smooth_plane_spin.value()),
            "snapping_enabled": bool(snapping_enabled_check.isChecked()),
            "snapping_strength": float(snapping_strength_spin.value()),
            "snapping_max_distance": float(snapping_distance_spin.value()),
        }

    def startBrain3DReconstructionWorkflow(self, _value=False) -> bool:
        config_overrides = self._promptBrain3DReconstructionConfig()
        if config_overrides is None:
            return False
        if not self.buildBrain3DModelFromSagittalPages(
            _value=False, config_overrides=config_overrides
        ):
            return False
        self.openBrain3DSession()
        self.regenerateBrain3DCoronalPlanes()
        self.openBrain3DMeshPreview()
        detect_source = getattr(self, "_detect_existing_3d_source", None)
        open_viewer = getattr(self, "_open_3d_volume_viewer", None)
        if callable(detect_source) and callable(open_viewer):
            try:
                source_path = detect_source()
            except Exception:
                source_path = None
            if source_path:
                try:
                    open_viewer(source_path)
                except Exception:
                    pass
        return True

    def buildBrain3DModelFromSagittalPages(
        self,
        _value=False,
        *,
        config_overrides: dict[str, object] | None = None,
    ) -> bool:
        pages = self._collect_brain3d_sagittal_pages()
        if not pages:
            post_window_status(
                self,
                self.tr("No sagittal polygon pages found for 3D reconstruction."),
                5000,
            )
            return False

        config = self._brain3d_config_from_overrides(config_overrides)
        model = build_brain_3d_model(pages, config)
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        self.otherData = store_brain_model_in_other_data(self.otherData, model)
        self._set_dirty_with_brain3d_internal_guard()
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

    def _brain3d_target_planes_for_local_regeneration(
        self,
        model,
    ) -> set[int]:
        requests = list(model.metadata.get("local_regeneration_requests") or [])
        target_indices: set[int] = set()
        for request in requests:
            if not isinstance(request, dict):
                continue
            plane_index = int(request.get("plane_index", -1) or -1)
            if plane_index < 0:
                continue
            radius = max(0, int(request.get("radius", 1) or 1))
            for idx in range(plane_index - radius, plane_index + radius + 1):
                if idx >= 0:
                    target_indices.add(int(idx))
        if not target_indices:
            fallback_plane = int(getattr(self, "frame_number", 0) or 0)
            target_indices.add(fallback_plane)
        model.metadata["local_regeneration_requests"] = []
        return target_indices

    def _is_threejs_view_active_for_brain3d(self) -> bool:
        try:
            manager = getattr(self, "threejs_manager", None)
            viewer_stack = getattr(self, "viewer_stack", None)
            if manager is None or viewer_stack is None:
                return False
            viewer = manager.viewer_widget()
            if viewer is None:
                return False
            return viewer_stack.currentWidget() is viewer
        except Exception:
            return False

    def regenerateBrain3DCoronalPlanes(self, _value=False, *, local_only=False) -> int:
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
        target_plane_indices = (
            self._brain3d_target_planes_for_local_regeneration(model)
            if bool(local_only)
            else None
        )

        for plane in planes:
            if (
                target_plane_indices is not None
                and int(plane.plane_index) not in target_plane_indices
            ):
                continue
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
        self._set_dirty_with_brain3d_internal_guard()
        post_window_status(
            self,
            self.tr("Regenerated %d coronal plane annotation page(s).") % int(written),
            5000,
        )
        if bool(local_only) and self._is_threejs_view_active_for_brain3d():
            try:
                self.openBrain3DMeshPreview()
            except Exception:
                pass
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
            snap_strength = (
                float(model.config.snapping_strength or 0.0)
                if bool(getattr(model.config, "snapping_enabled", False))
                else 0.0
            )
            apply_coronal_polygon_edit(
                model,
                plane_index,
                region_id,
                shape,
                snapping_strength=snap_strength,
                snapping_max_distance=float(
                    getattr(model.config, "snapping_max_distance", 8.0) or 8.0
                ),
            )
            updated += 1
        if updated <= 0:
            post_window_status(self, self.tr("No polygon edits to apply."), 3000)
            return False
        self.otherData = store_brain_model_in_other_data(self.otherData, model)
        self._set_dirty_with_brain3d_internal_guard()
        regenerated = self.regenerateBrain3DCoronalPlanes(local_only=True)
        post_window_status(
            self,
            self.tr(
                "Applied %d coronal polygon edit(s) to Brain 3D model; regenerated %d nearby plane(s)."
            )
            % (updated, int(regenerated)),
            5000,
        )
        self._refreshBrain3DSessionDock()
        return True

    def _applyBrain3DSelectedRegionState(self, state: str) -> bool:
        dock = getattr(self, "brain3d_session_dock", None)
        if not isinstance(dock, Brain3DSessionDockWidget):
            return False
        region_id = str(dock.selected_region_id() or "")
        if not region_id:
            return False
        plane_index = int(dock.plane_spin.value())
        self._onBrain3DRegionStateRequested(plane_index, region_id, str(state))
        return True

    def createBrain3DRegionOnPlane(self, _value=False) -> bool:
        return self._applyBrain3DSelectedRegionState("created")

    def hideBrain3DRegionOnPlane(self, _value=False) -> bool:
        return self._applyBrain3DSelectedRegionState("hidden")

    def restoreBrain3DRegionOnPlane(self, _value=False) -> bool:
        return self._applyBrain3DSelectedRegionState("present")

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
            try:
                stat_result = store.store_path.stat()
                store_signature = (
                    int(getattr(stat_result, "st_mtime_ns", 0)),
                    int(getattr(stat_result, "st_size", 0)),
                )
            except OSError:
                store_signature = (-1, -1)
            cache = getattr(self, self._FRAME_STORE_HAS_FRAME_CACHE_KEY, None)
            if not isinstance(cache, dict):
                cache = {}
                setattr(self, self._FRAME_STORE_HAS_FRAME_CACHE_KEY, cache)
            cache_key = (
                str(store.store_path),
                int(frame_number),
            )
            cached_value = cache.get(cache_key)
            if isinstance(cached_value, bool):
                # Backward-compatible migration for old cache entries.
                cached_value = {
                    "has_frame": bool(cached_value),
                    "signature": None,
                }
            if isinstance(cached_value, dict):
                cached_has_frame = cached_value.get("has_frame")
                cached_signature = cached_value.get("signature")
                if isinstance(cached_has_frame, bool):
                    if cached_signature == store_signature:
                        return cached_has_frame
                    # Store changed since cached result. Re-check once.
            has_frame = bool(store.get_frame_fast(frame_number))
            cache[cache_key] = {
                "has_frame": bool(has_frame),
                "signature": store_signature,
            }
            return has_frame
        except Exception:
            return False

    def _label_candidate_cache_token(self, candidate: Path) -> tuple | None:
        """Return a freshness token for on-demand frame label loading."""
        if (
            str(candidate.suffix or "").lower() == ".json"
            and candidate.exists()
            and candidate.is_file()
            and candidate.with_suffix(".png").exists()
        ):
            try:
                stat_result = candidate.stat()
                return (
                    "file_manual",
                    str(candidate),
                    int(getattr(stat_result, "st_mtime_ns", 0)),
                    int(getattr(stat_result, "st_size", 0)),
                )
            except Exception:
                logger.debug(
                    "Failed to stat manual label candidate %s.",
                    candidate,
                    exc_info=True,
                )

        prefer_store = bool(
            getattr(self, "_PREFER_ANNOTATION_STORE_OVER_FRAME_JSON", True)
        )
        if prefer_store and str(candidate.suffix or "").lower() == ".json":
            try:
                if self._annotation_store_has_frame(candidate):
                    store = AnnotationStore.for_frame_path(candidate)
                    store_path = Path(store.store_path)
                    if store_path.exists():
                        stat_result = store_path.stat()
                        return (
                            "store_preferred",
                            str(candidate),
                            str(store_path),
                            int(getattr(stat_result, "st_mtime_ns", 0)),
                            int(getattr(stat_result, "st_size", 0)),
                        )
            except Exception:
                logger.debug(
                    "Failed to resolve store-preferred token for %s.",
                    candidate,
                    exc_info=True,
                )

        try:
            if candidate.exists() and candidate.is_file():
                stat_result = candidate.stat()
                return (
                    "file",
                    str(candidate),
                    int(getattr(stat_result, "st_mtime_ns", 0)),
                    int(getattr(stat_result, "st_size", 0)),
                )
        except Exception:
            logger.debug("Failed to stat label candidate %s.", candidate, exc_info=True)

        try:
            if not self._annotation_store_has_frame(candidate):
                return None
            store = AnnotationStore.for_frame_path(candidate)
            store_path = Path(store.store_path)
            if store_path.exists():
                stat_result = store_path.stat()
                return (
                    "store",
                    str(candidate),
                    str(store_path),
                    int(getattr(stat_result, "st_mtime_ns", 0)),
                    int(getattr(stat_result, "st_size", 0)),
                )
            return ("store", str(candidate), "unknown")
        except Exception:
            logger.debug(
                "Failed to resolve annotation-store cache token for %s.",
                candidate,
                exc_info=True,
            )
        return None

    def _load_store_backed_label_file(self, candidate: Path):
        try:
            frame_number = AnnotationStore.frame_number_from_path(candidate)
            if frame_number is None:
                return None
            store = AnnotationStore.for_frame_path(candidate)
            record = store.get_frame_fast(frame_number)
            if record is None:
                record = store.get_frame(frame_number)
            if not isinstance(record, dict):
                return None
            shapes = list(record.get("shapes") or [])
            flags = dict(record.get("flags") or {})
            caption = record.get("caption")
            return SimpleNamespace(
                shapes=shapes,
                flags=flags,
                get_caption=(lambda value=caption: value),
            )
        except Exception:
            logger.debug(
                "Failed to load store-backed frame payload for %s.",
                candidate,
                exc_info=True,
            )
            return None

    def _frame_label_index_cache_get(self, cache_key: tuple, signature: tuple):
        cache = getattr(self, self._FRAME_LABEL_SOURCE_INDEX_CACHE_KEY, None)
        if not isinstance(cache, dict):
            return None
        cached = cache.get(cache_key)
        if not isinstance(cached, dict):
            return None
        if cached.get("signature") != signature:
            return None
        return cached.get("value")

    def _frame_label_index_cache_set(
        self,
        cache_key: tuple,
        signature: tuple,
        value,
    ) -> None:
        cache = getattr(self, self._FRAME_LABEL_SOURCE_INDEX_CACHE_KEY, None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, self._FRAME_LABEL_SOURCE_INDEX_CACHE_KEY, cache)
        cache[cache_key] = {
            "signature": signature,
            "value": value,
        }

    def _manual_label_frames_for_dir(self, directory: Path) -> dict[int, Path]:
        try:
            stat_result = directory.stat()
            signature = (
                int(getattr(stat_result, "st_mtime_ns", 0)),
                int(getattr(stat_result, "st_size", 0)),
            )
        except OSError:
            return {}

        cache_key = ("manual_dir", str(directory))
        cached = self._frame_label_index_cache_get(cache_key, signature)
        if isinstance(cached, dict):
            return dict(cached)

        frame_map: dict[int, Path] = {}
        try:
            for entry in directory.iterdir():
                if not entry.is_file() or str(entry.suffix).lower() != ".json":
                    continue
                if not entry.with_suffix(".png").exists():
                    continue
                frame_number = AnnotationStore.frame_number_from_path(entry)
                if frame_number is None:
                    continue
                frame_map[int(frame_number)] = entry
        except OSError:
            return {}

        self._frame_label_index_cache_set(cache_key, signature, dict(frame_map))
        return frame_map

    def _annotation_store_frames_for_candidate(self, candidate: Path) -> list[int]:
        try:
            store = AnnotationStore.for_frame_path(candidate)
            store_path = Path(store.store_path)
            if not store_path.exists():
                return []
            stat_result = store_path.stat()
            signature = (
                int(getattr(stat_result, "st_mtime_ns", 0)),
                int(getattr(stat_result, "st_size", 0)),
            )
        except OSError:
            return []

        cache_key = ("store_frames", str(store_path))
        cached = self._frame_label_index_cache_get(cache_key, signature)
        if isinstance(cached, list):
            return [int(frame) for frame in cached]

        try:
            frames = sorted(int(frame) for frame in store.iter_frames())
        except Exception:
            logger.debug(
                "Failed to index annotation store frames for %s.",
                store_path,
                exc_info=True,
            )
            return []

        self._frame_label_index_cache_set(cache_key, signature, list(frames))
        return frames

    @staticmethod
    def _nearest_frame_at_or_before(
        frame_number: int,
        candidate_frames: list[int],
    ) -> Optional[int]:
        if not candidate_frames:
            return None
        try:
            target = int(frame_number)
        except (TypeError, ValueError):
            return None
        idx = bisect_right(candidate_frames, target) - 1
        if idx < 0:
            return None
        return int(candidate_frames[idx])

    @staticmethod
    def _generated_store_candidate_path(directory: Path, frame_number: int) -> Path:
        return directory / f"{directory.name}_{int(frame_number):09}.json"

    def _resolve_sparse_frame_label_candidate(
        self,
        frame_number: int,
        frame_path: Optional[Path],
        label_candidates: list[Path],
    ) -> Optional[Path]:
        unique_candidates: list[Path] = []
        seen_candidates: set[Path] = set()
        for candidate in list(label_candidates or []):
            try:
                normalized = Path(candidate)
            except Exception:
                continue
            if normalized in seen_candidates:
                continue
            seen_candidates.add(normalized)
            unique_candidates.append(normalized)

        directories: list[Path] = []
        seen_dirs: set[Path] = set()
        for candidate in unique_candidates:
            directory = candidate.parent
            if directory in seen_dirs:
                continue
            seen_dirs.add(directory)
            directories.append(directory)
        if frame_path is not None:
            frame_dir = Path(frame_path).parent
            if frame_dir not in seen_dirs:
                directories.append(frame_dir)

        best_manual_frame: Optional[int] = None
        best_manual_candidate: Optional[Path] = None
        for directory in directories:
            frame_map = self._manual_label_frames_for_dir(directory)
            resolved_frame = self._nearest_frame_at_or_before(
                frame_number,
                sorted(frame_map.keys()),
            )
            if resolved_frame is None:
                continue
            if best_manual_frame is None or int(resolved_frame) > int(
                best_manual_frame
            ):
                best_manual_frame = int(resolved_frame)
                best_manual_candidate = frame_map.get(int(resolved_frame))
        if best_manual_candidate is not None:
            return best_manual_candidate

        best_store_frame: Optional[int] = None
        best_store_dir: Optional[Path] = None
        seen_store_dirs: set[Path] = set()
        for candidate in unique_candidates:
            directory = candidate.parent
            if directory in seen_store_dirs:
                continue
            seen_store_dirs.add(directory)
            resolved_frame = self._nearest_frame_at_or_before(
                frame_number,
                self._annotation_store_frames_for_candidate(candidate),
            )
            if resolved_frame is None:
                continue
            if best_store_frame is None or int(resolved_frame) > int(best_store_frame):
                best_store_frame = int(resolved_frame)
                best_store_dir = directory
        if best_store_dir is not None and best_store_frame is not None:
            return self._generated_store_candidate_path(
                best_store_dir,
                int(best_store_frame),
            )
        return None

    def _label_file_cache_get(self, candidate: Path, token: tuple):
        cache = getattr(self, self._FRAME_LABEL_CACHE_KEY, None)
        if not isinstance(cache, OrderedDict):
            return None
        key = str(candidate)
        cached = cache.get(key)
        if not isinstance(cached, dict):
            return None
        if cached.get("token") != token:
            return None
        cache.move_to_end(key)
        return cached.get("label_file")

    def _label_file_cache_set(self, candidate: Path, token: tuple, label_file) -> None:
        cache = getattr(self, self._FRAME_LABEL_CACHE_KEY, None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            setattr(self, self._FRAME_LABEL_CACHE_KEY, cache)
        key = str(candidate)
        cache[key] = {"token": token, "label_file": label_file}
        cache.move_to_end(key)
        while len(cache) > int(self._FRAME_LABEL_CACHE_MAX_ENTRIES):
            cache.popitem(last=False)

    def _enqueue_neighbor_label_prefetch(
        self, frame_number: int, frame_path: Optional[Path]
    ) -> None:
        if not getattr(self, "video_file", None):
            return
        total_frames = int(getattr(self, "num_frames", 0) or 0)
        if total_frames <= 0:
            return
        queue = getattr(self, self._FRAME_LABEL_PREFETCH_QUEUE_KEY, None)
        if not isinstance(queue, deque):
            queue = deque()
            setattr(self, self._FRAME_LABEL_PREFETCH_QUEUE_KEY, queue)
        window = int(self._FRAME_LABEL_PREFETCH_WINDOW)
        try:
            frame_idx = int(frame_number)
        except Exception:
            return
        for offset in range(1, max(window, 1) + 1):
            neighbor = frame_idx + offset
            if neighbor < 0 or neighbor >= total_frames:
                continue
            if neighbor not in queue:
                queue.append(neighbor)
        if not bool(getattr(self, self._FRAME_LABEL_PREFETCH_ACTIVE_KEY, False)):
            setattr(self, self._FRAME_LABEL_PREFETCH_ACTIVE_KEY, True)
            QtCore.QTimer.singleShot(
                0, lambda: self._drain_label_prefetch_queue(frame_path)
            )

    def _frame_path_for_index(
        self, frame_index: int, fallback_path: Optional[Path]
    ) -> Optional[Path]:
        if (
            frame_index == int(getattr(self, "frame_number", 0) or 0)
            and fallback_path is not None
        ):
            return fallback_path
        resolver = getattr(self, "_frame_image_path", None)
        if callable(resolver):
            try:
                return Path(resolver(int(frame_index)))
            except Exception:
                return fallback_path
        return fallback_path

    def _prefetch_label_for_frame(
        self, frame_index: int, fallback_path: Optional[Path]
    ) -> None:
        frame_path = self._frame_path_for_index(frame_index, fallback_path)
        candidates = self._iter_frame_label_candidates(int(frame_index), frame_path)
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            token = self._label_candidate_cache_token(candidate)
            if token is None:
                continue
            if self._label_file_cache_get(candidate, token) is not None:
                continue
            try:
                label_file = LabelFile(str(candidate), is_video_frame=True)
                self._label_file_cache_set(candidate, token, label_file)
            except Exception:
                continue
            break

    def _drain_label_prefetch_queue(self, fallback_path: Optional[Path]) -> None:
        queue = getattr(self, self._FRAME_LABEL_PREFETCH_QUEUE_KEY, None)
        if not isinstance(queue, deque):
            setattr(self, self._FRAME_LABEL_PREFETCH_ACTIVE_KEY, False)
            return
        budget = int(self._FRAME_LABEL_PREFETCH_BUDGET)
        processed = 0
        while queue and processed < max(1, budget):
            frame_index = queue.popleft()
            self._prefetch_label_for_frame(int(frame_index), fallback_path)
            processed += 1
        if queue:
            QtCore.QTimer.singleShot(
                0, lambda: self._drain_label_prefetch_queue(fallback_path)
            )
            return
        setattr(self, self._FRAME_LABEL_PREFETCH_ACTIVE_KEY, False)

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

    def _tracking_shape_cache_token(self):
        controller = getattr(self, "tracking_data_controller", None)
        csv_path = getattr(controller, "_tracking_csv_path", None)
        if csv_path:
            return str(csv_path)
        video_file = getattr(self, "video_file", None)
        if video_file:
            return f"video:{video_file}"
        df = getattr(self, "_df", None)
        if df is not None:
            return f"df:{id(df)}"
        return "none"

    def _tracking_shape_cache_get(
        self, frame_number: int, *, decode_segmentation: bool
    ) -> Optional[list[Shape]]:
        cache = getattr(self, self._TRACKING_SHAPE_CACHE_KEY, None)
        if not isinstance(cache, OrderedDict):
            return None
        key = (
            self._tracking_shape_cache_token(),
            int(frame_number),
            bool(decode_segmentation),
        )
        cached = cache.get(key)
        if not isinstance(cached, list):
            return None
        cache.move_to_end(key)
        return [
            shape.copy() if hasattr(shape, "copy") else shape for shape in list(cached)
        ]

    def _tracking_shape_cache_set(
        self, frame_number: int, *, decode_segmentation: bool, shapes: list[Shape]
    ) -> None:
        cache = getattr(self, self._TRACKING_SHAPE_CACHE_KEY, None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            setattr(self, self._TRACKING_SHAPE_CACHE_KEY, cache)
        key = (
            self._tracking_shape_cache_token(),
            int(frame_number),
            bool(decode_segmentation),
        )
        cache[key] = [
            shape.copy() if hasattr(shape, "copy") else shape
            for shape in list(shapes or [])
        ]
        cache.move_to_end(key)
        while len(cache) > int(self._TRACKING_SHAPE_CACHE_MAX_ENTRIES):
            cache.popitem(last=False)

    def _tracking_shapes_for_frame(
        self, frame_number: int, *, decode_segmentation: bool
    ) -> list[Shape]:
        cached = self._tracking_shape_cache_get(
            frame_number, decode_segmentation=decode_segmentation
        )
        if cached is not None:
            return cached

        frame_rows = self._tracking_rows_for_frame(frame_number)
        if not frame_rows:
            self._tracking_shape_cache_set(
                frame_number, decode_segmentation=decode_segmentation, shapes=[]
            )
            return []

        frame_label_list: list[Shape] = []
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
                        "_".join(instance_names) if instance_names else "unknown"
                    )
                except Exception:
                    row["instance_name"] = "unknown"
                row["segmentation"] = None
            if not decode_segmentation:
                # Fast playback path: avoid expensive mask decoding but still render
                # robust per-instance overlays from bbox coordinates.
                try:
                    x1 = float(row.get("x1"))
                    y1 = float(row.get("y1"))
                    x2 = float(row.get("x2"))
                    y2 = float(row.get("y2"))
                except (TypeError, ValueError):
                    x1 = y1 = x2 = y2 = float("nan")
                if all(value == value and value >= 0.0 for value in (x1, y1, x2, y2)):
                    label_name = str(row.get("instance_name") or "unknown")
                    tracking_id = row.get("tracking_id")
                    try:
                        tracking_id_int = int(tracking_id)
                    except (TypeError, ValueError):
                        tracking_id_int = -1
                    if tracking_id_int >= 0:
                        label_name = f"{label_name}_{tracking_id_int}"
                    rect_shape = Shape(
                        label=label_name,
                        shape_type="rectangle",
                        flags={},
                    )
                    rect_shape.addPoint((x1, y1))
                    rect_shape.addPoint((x2, y2))
                    frame_label_list.append(rect_shape)
                continue
            frame_label_list.extend(
                pred_dict_to_labelme(
                    row,
                    keypoint_area_threshold=int(
                        getattr(self, "_TRACKING_KEYPOINT_AREA_THRESHOLD", 16)
                    ),
                    decode_segmentation=decode_segmentation,
                )
            )
        self._tracking_shape_cache_set(
            frame_number,
            decode_segmentation=decode_segmentation,
            shapes=frame_label_list,
        )
        return frame_label_list

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
        persistent_zone_shapes = self._persistent_zone_shapes_for_frame(frame_path)
        label_candidates = self._iter_frame_label_candidates(frame_number, frame_path)
        sparse_candidate = self._resolve_sparse_frame_label_candidate(
            int(frame_number),
            frame_path,
            label_candidates,
        )
        if sparse_candidate is not None and sparse_candidate not in label_candidates:
            label_candidates = [*label_candidates, sparse_candidate]

        seen_candidates: set[Path] = set()
        label_loaded = False
        for candidate in label_candidates:
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)

            cache_token = self._label_candidate_cache_token(candidate)
            if cache_token is None:
                continue

            label_file = self._label_file_cache_get(candidate, cache_token)
            if label_file is None:
                try:
                    token_kind = str(cache_token[0]) if cache_token else ""
                    if token_kind == "store_preferred":
                        label_file = self._load_store_backed_label_file(candidate)
                        if label_file is None:
                            continue
                    else:
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
                self._label_file_cache_set(candidate, cache_token, label_file)

            try:
                self.labelFile = label_file
                self.canvas.setBehaviorText(None)
                frame_shapes = self._materialize_label_shapes(label_file.shapes)
                if any(is_zone_shape(shape) for shape in frame_shapes):
                    self.zone_path = str(candidate)
                if persistent_zone_shapes:
                    frame_shapes = self._merge_persistent_zones_into_shapes(
                        frame_shapes,
                        frame_path,
                        persistent_zone_shapes=persistent_zone_shapes,
                    )
            except Exception:
                logger.debug(
                    "Failed to prepare loaded label payload from %s.",
                    candidate,
                    exc_info=True,
                )
                continue

            self.loadShapes(frame_shapes)
            label_loaded = True

            try:
                self.update_flags_from_file(label_file)
            except Exception:
                logger.debug(
                    "Failed to update flags from loaded label payload %s.",
                    candidate,
                    exc_info=True,
                )

            try:
                if (
                    len(self.canvas.current_behavior_text) > 1
                    and "other" not in self.canvas.current_behavior_text.lower()
                ):
                    self.add_highlighted_mark(
                        self.frame_number, mark_type=self.canvas.current_behavior_text
                    )
            except Exception:
                logger.debug(
                    "Failed to add highlighted mark for loaded label payload %s.",
                    candidate,
                    exc_info=True,
                )

            try:
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
            except Exception:
                logger.debug(
                    "Failed to update caption from loaded label payload %s.",
                    candidate,
                    exc_info=True,
                )
            break

        if label_loaded:
            self._enqueue_neighbor_label_prefetch(frame_number, frame_path)
            return

        # Do not fall back to tracking CSV for shape overlays in the frame loader.
        fallback_shapes: list[Shape] = []
        if persistent_zone_shapes:
            fallback_shapes = self._merge_persistent_zones_into_shapes(
                fallback_shapes,
                frame_path,
                persistent_zone_shapes=persistent_zone_shapes,
            )
        self.loadShapes(fallback_shapes)

        if not label_loaded and self.caption_widget is not None:
            applied = self._apply_timeline_caption_if_available(
                frame_number, only_if_empty=False
            )
            if not applied:
                self.caption_widget.set_caption("")
