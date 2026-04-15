import copy
import hashlib
import gc
import os
import time
from pathlib import Path

import cv2
import numpy as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtWidgets import QLabel
from PIL import Image

from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    set_keypoint_visibility_on_shape_object,
)
from annolid.annotation.pose_schema import PoseSchema
from annolid.gui.shape import Shape, MaskShape, MultipoinstShape
from annolid.gui.shared_vertices import SharedTopologyRegistry
from annolid.gui.window_base import QT5
from annolid.segmentation.SAM.sam_hq import SamHQSegmenter
from annolid.utils.annotation_compat import AI_MODELS
from annolid.utils.annotation_compat import utils
from annolid.utils.devices import clear_device_cache
from annolid.utils.logger import logger
from annolid.utils.prompts import extract_number_and_remove_digits
from annolid.utils.qt2cv import convert_qt_image_to_rgb_cv_image
from annolid.gui.qt_compat import painter_render_hint
from annolid.gui.widgets.ai_polygon_helpers import (
    mask_bbox as _ai_mask_bbox,
    normalize_ai_polygon_points as _ai_normalize,
    polygon_from_refined_mask as _ai_polygon_from_mask,
    predict_ai_polygon_points as _ai_predict,
    simplify_ai_polygon_points as _ai_simplify,
)
from annolid.gui.mixins.shared_polygon_edit_mixin import SharedPolygonEditMixin
# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0


class Canvas(SharedPolygonEditMixin, QtWidgets.QWidget):
    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, object)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)
    overlayLandmarkPairSelected = QtCore.Signal(str)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(self.double_click)
            )
        self.caption_label = QtWidgets.QTextEdit()
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "grounding_sam": False,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_mask": False,
                "polygonSAM": False,
            },
        )
        self.sam_config = kwargs.pop(
            "sam",
            {
                "maxside": 2048,
                "approxpoly_epsilon": 0.5,
                "weights": "vit_h",
                "device": "cuda",
            },
        )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._adjoining_source_shape = None
        self._cursor = CURSOR_DEFAULT
        self.mouse_xy_text = ""
        self._icons_dir = Path(__file__).resolve().parents[1] / "icons"
        # Collect shapes that need prediction
        self.shapes_to_predict = []

        self.label = QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        # self.label.resize(120, 20)
        # self.label.setStyleSheet("background-color:white")

        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self._ai_model = None
        self._ai_model_pixmap_key = None
        self._ai_model_image_signature = None
        self._ai_model_signature_cache_token = None
        self._ai_model_signature_cache_value = None
        self._ai_model_rect = None
        self.sam_predictor = None
        self.sam_hq_model = None
        self.sam_mask = MaskShape()
        self._sam_predictor_missing_logged = False
        self._sam_last_load_error = None
        self._shared_topology_registry = SharedTopologyRegistry.from_shapes([])
        self._shared_boundary_reshape_mode = False
        self._shared_boundary_shape = None
        self._shared_boundary_edge_index = None
        self._dragging_shared_boundary = False
        self._shared_boundary_last_pos = None
        self.behavior_text_position = "top-left"  # Default position
        self.behavior_text_color = QtGui.QColor(255, 255, 255)
        # Semi-transparent background keeps overlay text readable on bright images
        self.behavior_text_background = QtGui.QColor(0, 0, 0, 180)
        self.current_behavior_text = None

        # Patch similarity helpers
        self._patch_similarity_active = False
        self._patch_similarity_callback = None

        # Keep internal selection state consistent with the selectionChanged signal.
        # This mirrors LabelMe Canvas behavior and is relied on by delete/duplicate/etc.
        try:
            self.selectionChanged.connect(self._on_selection_changed)
        except Exception:
            pass
        self._patch_similarity_pixmap = None
        self._pca_map_pixmap = None
        self._depth_preview_pixmap = None
        self._flow_preview_pixmap = None

        # Pose skeleton overlay (optional)
        self._pose_schema: PoseSchema | None = None
        self._show_pose_edges: bool = False
        self._show_pose_bboxes: bool = True
        self._pose_edge_color = QtGui.QColor(0, 255, 255, 190)
        self._pose_edge_shadow = QtGui.QColor(0, 0, 0, 160)
        self._selected_overlay_landmark_pair_id = None

    def _on_selection_changed(self, shapes):
        """Update internal selection state from a newly selected shapes list."""
        selected = list(shapes or [])
        selected_ids = {id(s) for s in selected}
        # Only keep shapes that are still present on the canvas, and use identity.
        self.selectedShapes = [s for s in self.shapes if id(s) in selected_ids]
        for s in self.shapes:
            try:
                s.selected = id(s) in selected_ids
            except Exception:
                pass
        if self.hShape not in self.selectedShapes:
            self.hShapeIsSelected = False
        self.update()

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    def _selected_polygon_source(self):
        for shape in self.selectedShapes or []:
            if (
                str(getattr(shape, "shape_type", "") or "").lower() == "polygon"
                and len(getattr(shape, "points", []) or []) >= 2
            ):
                return shape
        return None

    def _adjoining_boundary_source(self):
        source = getattr(self, "_adjoining_source_shape", None)
        if source is not None and source in (self.shapes or []):
            return source
        return self._selected_polygon_source()

    def _shared_boundary_source(self):
        shape = self.hShape if self.hEdge is not None else None
        edge_index = self.hEdge if self.hEdge is not None else None
        if shape is None or edge_index is None:
            return None, None
        if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
            return None, None
        edge_id = None
        try:
            edge_id = shape.shared_edge_id(int(edge_index))
        except Exception:
            edge_id = None
        if not edge_id:
            return None, None
        try:
            registry = getattr(self, "_shared_topology_registry", None)
            if isinstance(registry, SharedTopologyRegistry):
                if len(registry.edge_occurrences(edge_id)) < 2:
                    return None, None
        except Exception:
            return None, None
        return shape, int(edge_index)

    def _clear_adjoining_source(self):
        self._adjoining_source_shape = None

    def _sync_shared_vertex(self, shape, index, point=None):
        try:
            return self._shared_sync_vertex(shape, index, point=point)
        except Exception:
            logger.debug("Failed to synchronize shared vertex.", exc_info=True)
            return None

    def _snap_to_adjoining_boundary(self, pos):
        source = self._adjoining_boundary_source()
        if source is None or str(self.createMode or "").lower() != "polygon":
            return QtCore.QPointF(pos), None
        epsilon = max(1.0, float(self.epsilon) / max(self.scale, 1e-6))
        feature = getattr(source, "nearest_boundary_feature", None)
        if callable(feature):
            try:
                boundary = feature(QtCore.QPointF(pos), epsilon)
            except Exception:
                boundary = None
            if boundary is not None:
                return QtCore.QPointF(boundary["point"]), boundary
        return QtCore.QPointF(pos), None

    def setCaption(self, text):
        self.caption_label.setText(text)

    @staticmethod
    def _release_device_cache() -> None:
        """Best-effort cache release after unloading GPU-backed models."""
        gc.collect()
        clear_device_cache()

    def _release_model(self, model, *, context: str) -> None:
        """Best-effort close hook for optional AI model instances."""
        if model is None:
            return
        try:
            close_fn = getattr(model, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            logger.debug("Failed to close %s model cleanly.", context, exc_info=True)
        self._release_device_cache()

    def setBehaviorText(self, text):
        self.current_behavior_text = text

    def getCaption(self):
        return self.caption_label.toPlainText()

    # ------------------------------------------------------------------
    # Pose skeleton overlay
    # ------------------------------------------------------------------
    def setPoseSchema(self, schema: PoseSchema | None) -> None:
        self._pose_schema = schema
        self.update()

    def setShowPoseEdges(self, enabled: bool) -> None:
        self._show_pose_edges = bool(enabled)
        self.update()

    def showPoseEdges(self) -> bool:
        return bool(self._show_pose_edges)

    def setShowPoseBBoxes(self, enabled: bool) -> None:
        self._show_pose_bboxes = bool(enabled)
        self.update()

    def showPoseBBoxes(self) -> bool:
        return bool(self._show_pose_bboxes)

    @staticmethod
    def _is_pose_bbox_shape(shape) -> bool:
        if str(getattr(shape, "shape_type", "")).lower() != "rectangle":
            return False
        other = getattr(shape, "other_data", None)
        if not isinstance(other, dict):
            return False
        if other.get("instance_id") is None:
            return False
        return True

    def _infer_pose_instance_label(
        self, shape, candidate_labels, default_label: str = "object"
    ) -> str:
        flags = getattr(shape, "flags", None)
        if isinstance(flags, dict):
            instance_label = flags.get("instance_label")
            if isinstance(instance_label, str) and instance_label.strip():
                return instance_label.strip()
        other = getattr(shape, "other_data", None)
        if isinstance(other, dict):
            instance_label = other.get("instance_label")
            if isinstance(instance_label, str) and instance_label.strip():
                return instance_label.strip()

        group_id = getattr(shape, "group_id", None)
        if group_id not in (None, ""):
            return str(group_id)

        label = str(getattr(shape, "label", "") or "").strip()
        lower_label = label.lower()
        for candidate in sorted(candidate_labels, key=len, reverse=True):
            if not candidate:
                continue
            if lower_label.startswith(candidate.lower()):
                return candidate
        for delimiter in ("_", "-", ":", "|", " "):
            if delimiter in label:
                return label.split(delimiter, 1)[0].strip() or default_label
        return default_label

    def _infer_pose_keypoint_label(self, shape, instance_label: str) -> str:
        flags = getattr(shape, "flags", None)
        if isinstance(flags, dict):
            display_label = flags.get("display_label")
            if isinstance(display_label, str) and display_label.strip():
                return display_label.strip()
        other = getattr(shape, "other_data", None)
        if isinstance(other, dict):
            display_label = other.get("display_label")
            if isinstance(display_label, str) and display_label.strip():
                return display_label.strip()

        label = str(getattr(shape, "label", "") or "").strip()
        if instance_label:
            inst_len = len(instance_label)
            if label.lower().startswith(instance_label.lower()) and inst_len < len(
                label
            ):
                suffix = label[inst_len:].lstrip("_-:| ")
                if suffix:
                    return suffix
        for delimiter in ("_", "-", ":", "|"):
            if delimiter in label:
                suffix = label.split(delimiter, 1)[1].strip()
                if suffix:
                    return suffix
        return label

    def _draw_pose_edges(self, painter: QtGui.QPainter) -> None:
        if (
            not self._show_pose_edges
            or not self._pose_schema
            or not self._pose_schema.edges
        ):
            return

        candidate_labels = {
            str(getattr(s, "label", "") or "").strip()
            for s in self.shapes
            if str(getattr(s, "shape_type", "") or "").lower() != "point"
        }
        candidate_labels.discard("")

        instances = {}
        for shape in self.shapes:
            if str(getattr(shape, "shape_type", "") or "").lower() != "point":
                continue
            if not self.isVisible(shape):
                continue
            points = getattr(shape, "points", None) or []
            if not points:
                continue
            point = points[0]
            instance_label = None
            kp_label = None

            try:
                inst_from_schema, base_from_schema = (
                    self._pose_schema.strip_instance_prefix(
                        str(getattr(shape, "label", "") or "")
                    )
                )
                if inst_from_schema and base_from_schema:
                    instance_label = inst_from_schema
                    kp_label = base_from_schema
            except Exception:
                instance_label = None
                kp_label = None

            if not instance_label:
                instance_label = self._infer_pose_instance_label(
                    shape, candidate_labels, default_label="object"
                )
            if not kp_label:
                kp_label = self._infer_pose_keypoint_label(shape, instance_label)
            if not kp_label:
                continue
            instances.setdefault(instance_label, {})[kp_label] = point

        if not instances:
            return

        edge_width = max(1, int(round(3.0 / max(self.scale, 1e-6))))
        shadow_width = edge_width + max(1, int(round(2.0 / max(self.scale, 1e-6))))

        shadow_pen = QtGui.QPen(self._pose_edge_shadow)
        shadow_pen.setWidth(shadow_width)
        shadow_pen.setCapStyle(QtCore.Qt.RoundCap)
        shadow_pen.setJoinStyle(QtCore.Qt.RoundJoin)

        edge_pen = QtGui.QPen(self._pose_edge_color)
        edge_pen.setWidth(edge_width)
        edge_pen.setCapStyle(QtCore.Qt.RoundCap)
        edge_pen.setJoinStyle(QtCore.Qt.RoundJoin)

        painter.save()
        try:
            schema_instances = [
                inst for inst in (self._pose_schema.instances or []) if inst
            ]
            active_instances = list(instances.keys())
            if schema_instances:
                instance_order = [
                    inst for inst in schema_instances if inst in instances
                ] + [inst for inst in active_instances if inst not in schema_instances]
            else:
                instance_order = active_instances

            for edge in self._pose_schema.edges:
                a, b = edge
                inst_a, kp_a = self._pose_schema.strip_instance_prefix(a)
                inst_b, kp_b = self._pose_schema.strip_instance_prefix(b)
                if not self._pose_schema.instances:
                    sep = getattr(self._pose_schema, "instance_separator", "_") or "_"
                    if inst_a is None and isinstance(a, str) and sep in a:
                        candidate_inst, candidate_kp = a.split(sep, 1)
                        if candidate_inst in instances:
                            inst_a = candidate_inst
                            kp_a = candidate_kp.strip()
                    if inst_b is None and isinstance(b, str) and sep in b:
                        candidate_inst, candidate_kp = b.split(sep, 1)
                        if candidate_inst in instances:
                            inst_b = candidate_inst
                            kp_b = candidate_kp.strip()

                if inst_a or inst_b:
                    inst = inst_a or inst_b
                    if not inst:
                        continue
                    if inst_a and inst_b and inst_a != inst_b:
                        continue
                    kp_map = instances.get(inst, {})
                    p1 = kp_map.get(kp_a)
                    p2 = kp_map.get(kp_b)
                    if p1 is None or p2 is None:
                        continue
                    painter.setPen(shadow_pen)
                    painter.drawLine(p1, p2)
                    painter.setPen(edge_pen)
                    painter.drawLine(p1, p2)
                    continue

                # Base edges: replicate per instance.
                for inst in instance_order:
                    kp_map = instances.get(inst, {})
                    p1 = kp_map.get(kp_a)
                    p2 = kp_map.get(kp_b)
                    if p1 is None or p2 is None:
                        continue
                    painter.setPen(shadow_pen)
                    painter.drawLine(p1, p2)
                    painter.setPen(edge_pen)
                    painter.drawLine(p1, p2)
        finally:
            painter.restore()

    # ------------------------------------------------------------------
    # Patch similarity helpers (DINO visualization)
    # ------------------------------------------------------------------
    def enablePatchSimilarityMode(self, callback):
        """Enable click-to-query mode for DINO patch similarity."""
        self._patch_similarity_active = True
        self._patch_similarity_callback = callback
        self.overrideCursor(CURSOR_POINT)

    def disablePatchSimilarityMode(self):
        self._patch_similarity_active = False
        self._patch_similarity_callback = None
        self._patch_similarity_pixmap = None
        self.restoreCursor()
        self.update()

    def setPatchSimilarityOverlay(self, overlay_rgba):
        """Update or clear the rendered heatmap overlay."""
        if overlay_rgba is None:
            self._patch_similarity_pixmap = None
        else:
            h, w, _ = overlay_rgba.shape
            image = QtGui.QImage(
                overlay_rgba.data,
                w,
                h,
                overlay_rgba.strides[0],
                QtGui.QImage.Format_RGBA8888,
            )
            self._patch_similarity_pixmap = QtGui.QPixmap.fromImage(image.copy())
        self.update()

    def setPCAMapOverlay(self, overlay_rgba):
        """Update or clear the PCA coloring overlay."""
        if overlay_rgba is None:
            self._pca_map_pixmap = None
        else:
            h, w, _ = overlay_rgba.shape
            image = QtGui.QImage(
                overlay_rgba.data,
                w,
                h,
                overlay_rgba.strides[0],
                QtGui.QImage.Format_RGBA8888,
            )
            self._pca_map_pixmap = QtGui.QPixmap.fromImage(image.copy())
        self.update()

    def setDepthPreviewOverlay(self, overlay_rgba):
        """Update or clear the streaming depth preview overlay."""
        if overlay_rgba is None:
            self._depth_preview_pixmap = None
        else:
            h, w, _ = overlay_rgba.shape
            image = QtGui.QImage(
                overlay_rgba.data,
                w,
                h,
                overlay_rgba.strides[0],
                QtGui.QImage.Format_RGBA8888,
            )
            self._depth_preview_pixmap = QtGui.QPixmap.fromImage(image.copy())
        self.update()

    def setFlowPreviewOverlay(self, overlay_rgba):
        """Update or clear the optical-flow overlay."""
        if overlay_rgba is None:
            self._flow_preview_pixmap = None
        else:
            h, w, _ = overlay_rgba.shape
            image = QtGui.QImage(
                overlay_rgba.data,
                w,
                h,
                overlay_rgba.strides[0],
                QtGui.QImage.Format_RGBA8888,
            )
            self._flow_preview_pixmap = QtGui.QPixmap.fromImage(image.copy())
        self.update()

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "polygonSAM",
            "ai_polygon",
            "grounding_sam",
            "ai_mask",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value
        self.sam_mask = MaskShape()
        self.current = None

    def initializeAiModel(self, name, _custom_ai_models=None):
        init_start = time.perf_counter()
        model_class = self._resolve_ai_model_class(
            name=name,
            custom_model_names=_custom_ai_models,
        )
        if model_class is None:
            elapsed_ms = (time.perf_counter() - init_start) * 1000.0
            logger.info("AI model init skipped for '%s' (%.1fms).", name, elapsed_ms)
            return

        reused_model = (
            self._ai_model is not None and self._ai_model.name == model_class.name
        )
        if reused_model:
            logger.debug("AI model is already initialized: %r" % model_class.name)
        else:
            logger.debug("Initializing AI model: %r" % model_class.name)
            self._release_model(self._ai_model, context="previous AI")
            self._ai_model = None
            self._ai_model_pixmap_key = None
            self._ai_model = self._instantiate_ai_model(
                model_class,
                requested_name=name,
            )
            if self._ai_model is None:
                elapsed_ms = (time.perf_counter() - init_start) * 1000.0
                logger.warning(
                    "AI model init failed for '%s' in %.1fms.",
                    getattr(model_class, "name", name),
                    elapsed_ms,
                )
                return

        # Check if self.pixmap is None before calling isNull()
        if self.pixmap is None or self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return

        # Avoid rebuilding embeddings for the same frame/model pair.
        self._sync_ai_model_image(force=not reused_model)
        elapsed_ms = (time.perf_counter() - init_start) * 1000.0
        if reused_model:
            logger.info(
                "AI model '%s' reused and synced in %.1fms.",
                getattr(model_class, "name", name),
                elapsed_ms,
            )
        else:
            logger.info(
                "AI model '%s' initialized and synced in %.1fms.",
                getattr(model_class, "name", name),
                elapsed_ms,
            )

    def _resolve_ai_model_class(self, name: str, custom_model_names=None):
        model_class = None
        for candidate in AI_MODELS:
            if candidate.name == name:
                model_class = candidate
                break

        if model_class is None:
            if custom_model_names and name in custom_model_names:
                logger.debug(
                    "Custom model selected: %s, skipping SAM initialization", name
                )
                return None
            for candidate in AI_MODELS:
                if getattr(candidate, "name", None) == "EfficientSam (speed)":
                    model_class = candidate
                    break
            if model_class is None and AI_MODELS:
                model_class = AI_MODELS[0]
            if model_class is None:
                logger.warning(
                    "No AI segmentation models are available; skipping initialization."
                )
                return None
        return model_class

    def _instantiate_ai_model(self, model_class, *, requested_name: str):
        try:
            return model_class()
        except Exception as exc:
            self._ai_model = None
            QtWidgets.QMessageBox.warning(
                self,
                "AI Model Unavailable",
                f"Failed to initialize AI model '{getattr(model_class, 'name', requested_name)}'.\n\n{exc}",
            )
            return None

    def _sync_ai_model_image(self, *, force: bool = False) -> bool:
        if self._ai_model is None:
            return False
        if self.pixmap is None or self.pixmap.isNull():
            return False
        source_signature = self._ai_model_image_signature_value()
        if not force:
            # The semantic image/frame identity is the real cache key.
            # `QPixmap.cacheKey()` is only a fallback when signature generation is unavailable.
            if (
                source_signature is not None
                and self._ai_model_image_signature == source_signature
            ):
                return True
            if source_signature is None:
                try:
                    current_key = int(self.pixmap.cacheKey())
                except Exception:
                    current_key = None
                if (
                    current_key is not None
                    and self._ai_model_pixmap_key is not None
                    and int(self._ai_model_pixmap_key) == int(current_key)
                ):
                    return True
        try:
            self._ai_model.set_image(image=utils.img_qt_to_arr(self.pixmap.toImage()))
            self._ai_model_pixmap_key = int(self.pixmap.cacheKey())
            self._ai_model_image_signature = source_signature
            return True
        except Exception:
            logger.debug("Failed to sync AI model image.", exc_info=True)
            return False

    def _ai_model_image_signature_value(self):
        if self.pixmap is None or self.pixmap.isNull():
            return None
        source_hint = self._ai_model_source_hint()
        cache_key = None
        try:
            cache_key = int(self.pixmap.cacheKey())
        except Exception:
            cache_key = None
        signature_cache_token = (cache_key, source_hint)
        if (
            self._ai_model_signature_cache_token == signature_cache_token
            and self._ai_model_signature_cache_value is not None
        ):
            return self._ai_model_signature_cache_value
        try:
            qimage = self.pixmap.toImage()
            if qimage.isNull():
                return None
            image_data = utils.img_qt_to_arr(qimage)
            digest = hashlib.sha1(
                np.ascontiguousarray(image_data).tobytes()
            ).hexdigest()
            signature = (
                "image",
                digest,
                int(self.pixmap.width()),
                int(self.pixmap.height()),
                source_hint,
            )
            self._ai_model_signature_cache_token = signature_cache_token
            self._ai_model_signature_cache_value = signature
            return signature
        except Exception:
            signature = source_hint
            if signature is None:
                try:
                    signature = ("pixmap", int(self.pixmap.cacheKey()))
                except Exception:
                    signature = None
            self._ai_model_signature_cache_token = signature_cache_token
            self._ai_model_signature_cache_value = signature
            return signature

    def _ai_model_source_hint(self):
        candidates = []
        try:
            candidates.append(self.window())
        except Exception:
            pass
        candidates.append(self)
        for owner in candidates:
            if owner is None:
                continue
            source_path = str(
                getattr(owner, "filename", "") or getattr(owner, "imagePath", "") or ""
            ).strip()
            frame_number = getattr(owner, "frame_number", None)
            if source_path:
                if frame_number is not None:
                    try:
                        return (source_path, int(frame_number))
                    except Exception:
                        return (source_path, None)
                return (source_path, None)
        return None

    def _ensure_ai_model_initialized(self, *, force_sync: bool = False) -> bool:
        if self._ai_model is not None:
            return self._sync_ai_model_image(force=force_sync)
        default_name = None
        try:
            for m in AI_MODELS:
                if getattr(m, "name", None) == "EfficientSam (speed)":
                    default_name = m.name
                    break
            if default_name is None and AI_MODELS:
                default_name = AI_MODELS[0].name
        except Exception:
            default_name = None
        if default_name:
            self.initializeAiModel(default_name)
        return self._ai_model is not None and self._sync_ai_model_image(
            force=force_sync
        )

    def auto_mask_generator(
        self, image_data, label, points_per_side=32, is_polygon_output=True
    ):
        """
        Generate masks or polygons from segmentation annotations.

        Args:
            image_data (numpy.ndarray): The input image data for segmentation.
            label (str): The label to be assigned to generated shapes.
            points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
            is_polygon_output (bool, optional):
              Whether to output polygons (True) or masks (False).
                Defaults to True.

        Returns:
            None

        """
        # Segment everything in the image using the SamAutomaticMaskGenerator
        anns = self.sam_hq_model.segment_everything(
            image_data, points_per_side=points_per_side
        )

        # Iterate over each segmentation annotation
        for i, ann in enumerate(anns):
            # Check if the output format is polygons
            if is_polygon_output:
                # Convert segmentation to polygons
                self.current = MaskShape(
                    label=f"{label}_{i}", flags={}, description="grounding_sam"
                )
                self.current.mask = ann["segmentation"]
                self.current = self.current.toPolygons()[0]
            else:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = ann["bbox"]
                # Create shape object with refined shape
                self.current = Shape(
                    label=f"{label}_{i}", flags={}, description="grounding_sam"
                )
                self.current.setShapeRefined(
                    shape_type="mask",
                    points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                    point_labels=[1, 1],
                    mask=ann["segmentation"][int(y1) : int(y2), int(x1) : int(x2)],
                )
            # Add stability score as other data
            self.current.other_data["score"] = str(ann["stability_score"])

            # Finalize the process
            self.finalise()

    def predictAiRectangle(
        self, prompt, rectangle_shapes=None, is_polygon_output=True, use_countgd=False
    ):
        """
        Predict bounding boxes and then polygons based on the given prompt.

        Args:
            prompt (str): The prompt for prediction.
            is_polygon_output (bool, optional):
            Whether to output polygons (True).
                Defaults to True.

        Returns:
            None

        """
        # Check if the pixmap is set
        if self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return

        # Convert Qt image to RGB OpenCV image
        qt_image = self.pixmap.toImage()
        image_data = convert_qt_image_to_rgb_cv_image(qt_image)

        # Initialize SAM HQ model if not already initialized
        if self.sam_hq_model is None:
            try:
                self.sam_hq_model = SamHQSegmenter()
            except ModuleNotFoundError as exc:
                QtWidgets.QMessageBox.about(
                    self,
                    "Missing Segment Anything dependency",
                    str(exc),
                )
                return

        # If the prompt contains 'every', segment everything
        if prompt and "every" in prompt.lower():
            points_per_side, prompt = extract_number_and_remove_digits(prompt)
            if points_per_side < 1 or points_per_side > 100:
                points_per_side = 32

            label = prompt.replace("every", "").strip()
            logger.info(
                f"{points_per_side} points to be sampled along one side of the image"
            )
            self.auto_mask_generator(
                image_data,
                label,
                points_per_side=points_per_side,
                is_polygon_output=is_polygon_output,
            )
            return

        label = prompt
        if rectangle_shapes is not None:
            _bboxes = self._predict_similar_rectangles(
                rectangle_shapes=rectangle_shapes, prompt=prompt
            )
        else:
            rectangle_shapes = []
            _bboxes = []
            if use_countgd:
                _bboxes = self._predict_similar_rectangles(
                    rectangle_shapes=rectangle_shapes, prompt=prompt
                )
            # Initialize AI model if not already initialized
            if self._ai_model_rect is None:
                try:
                    from annolid.detector.grounding_dino import GroundingDINO

                    self._ai_model_rect = GroundingDINO()
                except ModuleNotFoundError as exc:
                    QtWidgets.QMessageBox.about(
                        self,
                        "Missing GroundingDINO dependency",
                        str(exc),
                    )
                    return

            # Predict bounding boxes using the GroundingDINO model
            bboxes = self._ai_model_rect.predict_bboxes(image_data, prompt)
            gd_bboxes = [list(box) for box, _ in bboxes]
            _bboxes.extend(gd_bboxes)

        if not _bboxes:
            logger.info("No candidate boxes found for prompt '%s'.", prompt)
            return

        # Segment objects using SAM HQ model with predicted bounding boxes
        masks, scores, _bboxes = self.sam_hq_model.segment_objects(image_data, _bboxes)

        # Iterate over each predicted bounding box
        for i, box in enumerate(_bboxes):
            # Check if the output format is polygons
            if is_polygon_output:
                # Convert segmentation mask to polygons
                self.current = MaskShape(
                    label=f"{label}_{i}", flags={}, description="grounding_sam"
                )
                self.current.mask = masks[i]
                self.current = self.current.toPolygons()[0]
            else:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box
                # Create shape object with refined shape
                self.current = Shape(
                    label=f"{label}_{i}", flags={}, description="grounding_sam"
                )
                self.current.setShapeRefined(
                    shape_type="mask",
                    points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                    point_labels=[1, 1],
                    mask=masks[i][int(y1) : int(y2), int(x1) : int(x2)],
                )
            # Add stability score as other data
            self.current.other_data["score"] = str(scores[i])

            # Finalize the process
            self.finalise()

    def loadSamPredictor(
        self,
    ):
        """
                The code requires Python version 3.8 or higher, along with PyTorch version 1.7 or higher,
                as well as TorchVision version 0.8 or higher. To install these dependencies,
                kindly refer to the instructions provided here. It's strongly recommended to install both PyTorch and TorchVision with CUDA support.
        To install Segment Anything, run the following command in your terminal:
        pip install git+https://github.com/facebookresearch/segment-anything.git

        """
        self._sam_last_load_error = None
        if self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            self._sam_last_load_error = "no_pixmap"
            return
        if not self.sam_predictor:
            try:
                import torch
                from pathlib import Path
                from segment_anything import sam_model_registry, SamPredictor
                from annolid.utils.devices import has_gpu
            except ImportError:
                self._sam_last_load_error = "missing_dependency"
                return
            if not has_gpu() and not torch.cuda.is_available():
                QtWidgets.QMessageBox.about(
                    self,
                    "GPU not available",
                    """For optimal performance, it is recommended to \
                                                use a GPU for running the Segment Anything model. \
                                                    Running the model on a CPU will result \
                                                        in significantly longer processing times.""",
                )
            here = (
                Path(os.path.abspath(__file__)).parent.parent.parent
                / "segmentation"
                / "SAM"
            )
            cachedir = str(here)
            os.makedirs(cachedir, exist_ok=True)
            weight_file = os.path.join(cachedir, self.sam_config["weights"] + ".pth")
            weight_urls = {
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            }
            if not os.path.isfile(weight_file):
                torch.hub.download_url_to_file(
                    weight_urls[self.sam_config["weights"]], weight_file
                )
            sam = sam_model_registry[self.sam_config["weights"]](checkpoint=weight_file)
            if self.sam_config["device"] == "cuda" and torch.cuda.is_available():
                sam.to(device="cuda")
            self.sam_predictor = SamPredictor(sam)
            self._sam_predictor_missing_logged = False
        self.samEmbedding()

    def samEmbedding(
        self,
    ):
        image = self.pixmap.toImage().copy()
        img_size = image.size()
        s = image.bits().asstring(
            img_size.height() * img_size.width() * image.depth() // 8
        )
        image = np.frombuffer(s, dtype=np.uint8).reshape(
            [img_size.height(), img_size.width(), image.depth() // 8]
        )
        image = image[:, :, :3].copy()
        h, w, _ = image.shape
        self.sam_image_scale = self.sam_config["maxside"] / max(h, w)
        self.sam_image_scale = min(1, self.sam_image_scale)
        image = cv2.resize(
            image,
            None,
            fx=self.sam_image_scale,
            fy=self.sam_image_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        self.sam_predictor.set_image(image)
        self.update()

    def samPrompt(self, points, labels):
        if self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return
        if not self.sam_predictor:
            if not self._sam_predictor_missing_logged:
                logger.warning(
                    "SAM predictor is not initialized; run 'Segment Anything' to load the model."
                )
                self._sam_predictor_missing_logged = True
            return
        if not hasattr(self, "sam_image_scale"):
            try:
                self.samEmbedding()
            except Exception:
                logger.exception("Failed to set SAM image embedding.")
                return
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=points * self.sam_image_scale,
            point_labels=labels,
            mask_input=self.sam_mask.logits[None, :, :]
            if self.sam_mask.logits is not None
            else None,
            multimask_output=False,
        )
        self.sam_mask.logits = logits[np.argmax(scores), :, :]
        mask = masks[np.argmax(scores), :, :]
        self.sam_mask.setScaleMask(self.sam_image_scale, mask)

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self._shared_topology_registry = SharedTopologyRegistry.from_shapes(self.shapes)
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self._clearSharedBoundaryReshape()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self._clearSharedBoundaryReshape()
        self.restoreCursor()

    def isVisible(self, shape):
        if not self._show_pose_bboxes and self._is_pose_bbox_shape(shape):
            return False
        if not getattr(shape, "visible", True):
            return False
        return self.visible.get(id(shape), True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.cancelCurrentDrawing(clear_sam_mask=True)
            self._clearSharedBoundaryReshape()
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE
            self.unHighlight()
            self._clearSharedBoundaryReshape()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        x, y = ev.x(), ev.y()
        self.mouse_xy_text = f"x:{pos.x():.1f},y:{pos.y():.1f}"
        self.label.setText(self.mouse_xy_text)
        self.label.adjustSize()
        label_width = self.label.width()
        label_height = self.label.height()
        self.label.move(int(x - label_width / 2), int(y + label_height))
        self.prevMovePoint = pos
        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier
        self.restoreCursor()
        # Polygon drawing.
        if self.drawing():
            if self.createMode in ["ai_polygon", "ai_mask", "grounding_sam"]:
                self.line.shape_type = "points"
            else:
                self.line.shape_type = (
                    self.createMode if "polygonSAM" != self.createMode else "polygon"
                )

            self.overrideCursor(CURSOR_DRAW)
            adjoining_pos, adjoining_feature = self._snap_to_adjoining_boundary(pos)
            if not self.current:
                if self.createMode == "polygonSAM":
                    points = np.array([[pos.x(), pos.y()]])
                    labels = np.array([1])
                    self.samPrompt(points, labels)
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(adjoining_pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], adjoining_pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(adjoining_pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            elif self.createMode == "polygon" and adjoining_feature is not None:
                pos = adjoining_pos
            elif self.createMode == "polygon":
                close_target = self._polygon_close_target(adjoining_pos)
                if close_target is not None:
                    pos = close_target
            if self.createMode in ["polygon", "linestrip"]:
                self.line.points = [self.current[-1], pos]
                self.line.point_labels = [1, 1]
            elif self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.points = [self.current.points[-1], pos]
                self.line.point_labels = [
                    self.current.point_labels[-1],
                    0 if is_shift_pressed else 1,
                ]
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.close()
            elif self.createMode == "polygonSAM":
                self.line.points = [pos, pos]
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self._dragging_shared_boundary and self._shared_boundary_last_pos:
                delta = QtCore.QPointF(
                    float(pos.x()) - float(self._shared_boundary_last_pos.x()),
                    float(pos.y()) - float(self._shared_boundary_last_pos.y()),
                )
                if self._reshapeSharedBoundaryBy(delta):
                    self._shared_boundary_last_pos = QtCore.QPointF(pos)
                    self.movingShape = True
                    self.repaint()
                return
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click & drag to move point"))
                self.label.setText(shape.label + "," + self.mouse_xy_text)
                self.label.adjustSize()
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click to create point"))
                self.label.setText(shape.label + "," + self.mouse_xy_text)
                self.label.adjustSize()
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.label.setText(shape.label + "," + self.mouse_xy_text)
                self.label.adjustSize()
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        self._shared_insert_vertex_on_edge(shape, index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.hShape if self.hShape is not None else self.prevhShape
        index = self.hVertex if self.hVertex is not None else self.prevhVertex
        if shape is None or index is None:
            return False
        if index < 0 or index >= len(shape.points):
            return False
        if not self._shared_remove_vertex(shape, index):
            return False
        shape.highlightClear()
        self.hShape = shape
        if shape.points:
            new_index = min(index, len(shape.points) - 1)
            self.hVertex = new_index
            self.prevhVertex = new_index
            shape.highlightVertex(new_index, shape.MOVE_VERTEX)
        else:
            self.hVertex = None
            self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = True  # Save changes
        self.vertexSelected.emit(self.hVertex is not None)
        self.update()
        return True

    def _boundary_polygon_source(self):
        if (
            self.hShape is not None
            and str(getattr(self.hShape, "shape_type", "") or "").lower() == "polygon"
            and len(getattr(self.hShape, "points", []) or []) >= 2
            and self.hEdge is not None
        ):
            return self.hShape, self.hEdge
        return None, None

    def _selected_polygon_source(self):
        for item in self.selectedShapes or []:
            if (
                str(getattr(item, "shape_type", "") or "").lower() == "polygon"
                and len(getattr(item, "points", []) or []) >= 2
            ):
                return item
        return None

    def _selected_shared_boundary_candidate(self):
        shape = self._selected_polygon_source()
        if shape is None:
            return None
        registry = getattr(self, "_shared_topology_registry", None)
        if not isinstance(registry, SharedTopologyRegistry):
            return None
        for edge_id in list(getattr(shape, "shared_edge_ids", []) or []):
            if not edge_id:
                continue
            try:
                if len(registry.edge_occurrences(edge_id)) >= 2:
                    return shape
            except Exception:
                continue
        return None

    def canStartAdjoiningPolygon(self) -> bool:
        return self._selected_polygon_source() is not None

    def canStartSharedBoundaryReshape(self) -> bool:
        shape, edge_index = self._shared_boundary_source()
        if shape is not None and edge_index is not None:
            return True
        return self._selected_shared_boundary_candidate() is not None

    def startSharedBoundaryReshape(self) -> bool:
        shape, edge_index = self._shared_boundary_source()
        if shape is None or edge_index is None:
            shape = self._selected_shared_boundary_candidate()
            edge_index = None
        if shape is None:
            return False
        self._shared_boundary_reshape_mode = True
        self._shared_boundary_shape = shape
        self._shared_boundary_edge_index = (
            int(edge_index) if edge_index is not None else None
        )
        self._dragging_shared_boundary = False
        self._shared_boundary_last_pos = None
        self.overrideCursor(CURSOR_MOVE)
        self.update()
        return True

    def _clearSharedBoundaryReshape(self) -> None:
        self._shared_boundary_reshape_mode = False
        self._shared_boundary_shape = None
        self._shared_boundary_edge_index = None
        self._dragging_shared_boundary = False
        self._shared_boundary_last_pos = None

    def _reshapeSharedBoundaryBy(self, delta) -> bool:
        shape = self._shared_boundary_shape
        edge_index = self._shared_boundary_edge_index
        if shape is None or edge_index is None:
            return False
        if not self._shared_reshape_boundary(shape, int(edge_index), delta):
            return False
        return True

    def beginAdjoiningPolygonFromSeed(self, seed_shape) -> bool:
        if seed_shape is None:
            return False
        seed = seed_shape
        self.mode = self.CREATE
        self.createMode = "polygon"
        self.current = seed
        self.current.setOpen()
        self.current.fill = self.fillDrawing()
        self.line.shape_type = "polygon"
        last_point = QtCore.QPointF(seed.points[-1])
        self.line.points = [QtCore.QPointF(last_point), QtCore.QPointF(last_point)]
        self.line.point_labels = [1, 1]
        self.setHiding()
        self.drawingPolygon.emit(True)
        self.update()
        return True

    def startAdjoiningPolygonFromSelection(self, edge_index=None) -> bool:
        shape = self._selected_polygon_source()
        if shape is None:
            return False
        seed = self._shared_adjoining_seed_for_shape(shape, edge_index=edge_index)
        if seed is None:
            return False
        self._adjoining_source_shape = shape
        return self.beginAdjoiningPolygonFromSeed(seed)

    def _icon(self, filename: str) -> QtGui.QIcon:
        path = self._icons_dir / filename
        if path.exists():
            return QtGui.QIcon(str(path))
        return QtGui.QIcon()

    def _ensure_action_icon(
        self,
        action: QtWidgets.QAction | None,
        *,
        icon_filename: str | None = None,
        fallback_standard: QtWidgets.QStyle.StandardPixmap | None = None,
    ) -> QtWidgets.QAction | None:
        if action is None:
            return None
        if action.icon().isNull() and icon_filename:
            icon = self._icon(icon_filename)
            if not icon.isNull():
                action.setIcon(icon)
        if action.icon().isNull() and fallback_standard is not None:
            action.setIcon(self.style().standardIcon(fallback_standard))
        return action

    def _build_context_menu(self, main_window) -> QtWidgets.QMenu:
        """Build a flat, icon-first context menu for drawing and shape actions."""
        menu = QtWidgets.QMenu(self)
        menu.setToolTipsVisible(True)

        def _add_existing_action(
            act: QtWidgets.QAction | None,
            *,
            icon_filename: str | None = None,
            fallback_standard: QtWidgets.QStyle.StandardPixmap | None = None,
        ) -> bool:
            if act is not None:
                self._ensure_action_icon(
                    act,
                    icon_filename=icon_filename,
                    fallback_standard=fallback_standard,
                )
                menu.addAction(act)
                return True
            return False

        def _add_draw_mode_action(
            mode: str,
            *,
            source_action: QtWidgets.QAction | None = None,
            text: str | None = None,
            edit: bool = False,
            icon_filename: str | None = None,
            fallback_standard: QtWidgets.QStyle.StandardPixmap | None = None,
        ) -> bool:
            if source_action is None and text is None:
                return False
            action = QtWidgets.QAction(text or source_action.text(), menu)
            if source_action is not None:
                action.setEnabled(source_action.isEnabled())
                action.setToolTip(source_action.toolTip())
                if not source_action.icon().isNull():
                    action.setIcon(source_action.icon())
            self._ensure_action_icon(
                action,
                icon_filename=icon_filename,
                fallback_standard=fallback_standard,
            )

            def _trigger():
                self.cancelCurrentDrawing(clear_sam_mask=True)
                if hasattr(main_window, "toggleDrawMode"):
                    main_window.toggleDrawMode(
                        bool(edit),
                        createMode="polygon" if edit else mode,
                    )

            action.triggered.connect(_trigger)
            menu.addAction(action)
            return True

        # Flat, user-friendly context menu: edit first, then draw modes, then AI.
        actions = getattr(main_window, "actions", None)
        if actions is not None:
            _add_draw_mode_action(
                "polygon",
                source_action=getattr(actions, "editMode", None),
                edit=True,
                icon_filename="edit_polygons.svg",
            )
            menu.addSeparator()
            _add_draw_mode_action(
                "polygon",
                source_action=getattr(actions, "createMode", None),
                icon_filename="create_polygons.svg",
            )
            _add_draw_mode_action(
                "rectangle",
                source_action=getattr(actions, "createRectangleMode", None),
                fallback_standard=QtWidgets.QStyle.SP_FileDialogDetailedView,
            )
            _add_draw_mode_action(
                "circle",
                source_action=getattr(actions, "createCircleMode", None),
                fallback_standard=QtWidgets.QStyle.SP_BrowserReload,
            )
            _add_draw_mode_action(
                "line",
                source_action=getattr(actions, "createLineMode", None),
                fallback_standard=QtWidgets.QStyle.SP_ArrowRight,
            )
            _add_draw_mode_action(
                "point",
                source_action=getattr(actions, "createPointMode", None),
                fallback_standard=QtWidgets.QStyle.SP_DialogApplyButton,
            )
            _add_draw_mode_action(
                "linestrip",
                source_action=getattr(actions, "createLineStripMode", None),
                fallback_standard=QtWidgets.QStyle.SP_MediaSeekForward,
            )
            menu.addSeparator()
            _add_draw_mode_action(
                "ai_polygon",
                source_action=getattr(actions, "createAiPolygonMode", None),
                text="AI Polygon",
                icon_filename="ai_polygons.svg",
            )
            _add_draw_mode_action(
                "ai_mask",
                source_action=getattr(actions, "createAiMaskMode", None),
                text="AI Mask",
                icon_filename="ai_polygons.svg",
            )
            _add_draw_mode_action(
                "grounding_sam",
                source_action=getattr(actions, "createGroundingSAMMode", None),
                text="Grounding SAM",
                fallback_standard=QtWidgets.QStyle.SP_ComputerIcon,
            )

        # ------------------------------------------------------------
        # SAM3 session operations
        # ------------------------------------------------------------
        if (
            hasattr(main_window, "reset_sam3_session")
            or hasattr(main_window, "remove_sam3_object")
            or hasattr(main_window, "close_sam3_session")
        ):
            menu.addSeparator()
            if hasattr(main_window, "reset_sam3_session"):
                sam3_reset_action = QtWidgets.QAction(
                    self._icon("undo.svg"),
                    "Reset SAM3 Session",
                    menu,
                )
                if sam3_reset_action.icon().isNull():
                    sam3_reset_action.setIcon(
                        self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
                    )
                sam3_reset_action.triggered.connect(
                    lambda: main_window.reset_sam3_session()
                )
                menu.addAction(sam3_reset_action)
            if hasattr(main_window, "close_sam3_session"):
                sam3_close_action = QtWidgets.QAction(
                    self._icon("close.svg"),
                    "Close SAM3 Session",
                    menu,
                )
                if sam3_close_action.icon().isNull():
                    sam3_close_action.setIcon(
                        self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton)
                    )
                sam3_close_action.triggered.connect(
                    lambda: main_window.close_sam3_session()
                )
                menu.addAction(sam3_close_action)
            if hasattr(main_window, "remove_sam3_object"):
                sam3_remove_action = QtWidgets.QAction(
                    self._icon("delete_polygons.svg"),
                    "Remove SAM3 Object...",
                    menu,
                )
                if sam3_remove_action.icon().isNull():
                    sam3_remove_action.setIcon(
                        self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon)
                    )
                sam3_remove_action.triggered.connect(
                    lambda: main_window.remove_sam3_object()
                )
                menu.addAction(sam3_remove_action)

        active_editor = getattr(main_window, "_active_shape_editor", None)
        editor = active_editor() if callable(active_editor) else None
        tiled_editor = getattr(main_window, "large_image_view", None)
        selected_shapes = list(self.selectedShapes or [])
        if not selected_shapes and tiled_editor is not None:
            selected_shapes = list(getattr(tiled_editor, "selectedShapes", []) or [])
        can_start_adjoining = bool(
            getattr(editor, "canStartAdjoiningPolygon", lambda: False)()
        ) or bool(getattr(tiled_editor, "canStartAdjoiningPolygon", lambda: False)())
        can_start_boundary_reshape = bool(
            getattr(editor, "canStartSharedBoundaryReshape", lambda: False)()
        ) or bool(
            getattr(tiled_editor, "canStartSharedBoundaryReshape", lambda: False)()
        )
        can_infer_page_polygons = bool(
            getattr(
                main_window, "canInferCurrentLargeImagePagePolygons", lambda: False
            )()
        )
        # ------------------------------------------------------------
        # Shape operations
        # ------------------------------------------------------------
        if (
            selected_shapes
            or can_start_adjoining
            or can_start_boundary_reshape
            or can_infer_page_polygons
        ):
            menu.addSeparator()
            if can_infer_page_polygons:
                infer_action = getattr(actions, "inferPagePolygons", None)
                _add_existing_action(
                    infer_action, icon_filename="duplicate_polygons.svg"
                )
            if selected_shapes:
                propagate_action = QtWidgets.QAction(
                    self._icon("next_frame.svg"),
                    "Propagate Selected Shape",
                    menu,
                )
                propagate_action.triggered.connect(
                    lambda: self.propagateSelectedShapeFromCanvas()
                )
                menu.addAction(propagate_action)

            if actions is not None:
                if selected_shapes:
                    duplicate_action = getattr(actions, "duplicateShapes", None)
                    _add_existing_action(
                        duplicate_action, icon_filename="duplicate_polygons.svg"
                    )
                adjoining_action = getattr(actions, "startAdjoiningPolygon", None)
                if adjoining_action is not None:
                    adjoining_action.setEnabled(can_start_adjoining)
                    _add_existing_action(
                        adjoining_action, icon_filename="duplicate_polygons.svg"
                    )
                collapse_action = getattr(actions, "collapsePolygons", None)
                if collapse_action is not None:
                    collapse_action.setEnabled(
                        bool(
                            getattr(
                                main_window,
                                "canCollapseSelectedPolygons",
                                lambda: False,
                            )()
                        )
                    )
                    _add_existing_action(
                        collapse_action, icon_filename="delete_polygons.svg"
                    )
                restore_action = getattr(actions, "restorePolygons", None)
                if restore_action is not None:
                    restore_action.setEnabled(
                        bool(
                            getattr(
                                main_window,
                                "canRestoreSelectedPolygons",
                                lambda: False,
                            )()
                        )
                    )
                    _add_existing_action(restore_action, icon_filename="undo.svg")
                if can_start_boundary_reshape:
                    boundary_reshape_action = QtWidgets.QAction(
                        self._icon("edit_polygons.svg"),
                        "Reshape Shared Boundary",
                        menu,
                    )
                    if boundary_reshape_action.icon().isNull():
                        boundary_reshape_action.setIcon(
                            self.style().standardIcon(
                                QtWidgets.QStyle.SP_DialogApplyButton
                            )
                        )

                    def _trigger_boundary_reshape():
                        starter = getattr(
                            main_window, "startSharedBoundaryReshape", None
                        )
                        if callable(starter):
                            starter()

                    boundary_reshape_action.triggered.connect(_trigger_boundary_reshape)
                    menu.addAction(boundary_reshape_action)
                if selected_shapes:
                    delete_action = getattr(actions, "deleteShapes", None)
                    _add_existing_action(
                        delete_action, icon_filename="delete_polygons.svg"
                    )

            if hasattr(main_window, "run_sam3d_reconstruction"):
                sam3d_action = QtWidgets.QAction(
                    self._icon("reconstruct_3d.svg"),
                    "Reconstruct 3D (SAM 3D)",
                    menu,
                )
                if sam3d_action.icon().isNull():
                    sam3d_action.setIcon(
                        self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
                    )
                sam3d_action.triggered.connect(
                    lambda: main_window.run_sam3d_reconstruction()
                )
                menu.addAction(sam3d_action)
        return menu

    def contextMenuEvent(self, event):
        main_window = self.window()
        menu = self._build_context_menu(main_window)
        menu.exec_(event.globalPos())

    def propagateSelectedShapeFromCanvas(self):
        # Since the canvas doesn't directly have frame data,
        # we call the main window's method.
        main_window = self.window()  # Assumes top-level window is AnnolidWindow
        if hasattr(main_window, "propagateSelectedShape"):
            main_window.propagateSelectedShape()

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        if self._patch_similarity_active and ev.button() == QtCore.Qt.LeftButton:
            if not self.outOfPixmap(pos) and self._patch_similarity_callback:
                self._patch_similarity_callback(int(pos.x()), int(pos.y()))
            ev.accept()
            return

        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                snapped_pos, adjoining_feature = self._snap_to_adjoining_boundary(pos)
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        close_target = self._polygon_close_target(snapped_pos)
                        if close_target is not None:
                            snapped_pos = close_target
                        self.current.addPoint(snapped_pos)
                        if close_target is None and adjoining_feature is not None:
                            self._shared_link_adjoining_point(
                                self.current,
                                len(self.current.points) - 1,
                                snapped_pos,
                                self._adjoining_boundary_source(),
                                adjoining_feature,
                            )
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if ev.modifiers() == QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask"]:
                        # Use the current click position to avoid stale preview-point
                        # drift when mouse-move and press events are interleaved.
                        click_label = 0 if is_shift_pressed else 1
                        self.current.addPoint(snapped_pos, label=click_label)
                        self.line.points[0] = self.current.points[-1]
                        self.line.point_labels[0] = self.current.point_labels[-1]
                        self.line.points[1] = snapped_pos
                        self.line.point_labels[1] = click_label
                        if ev.modifiers() & QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode == "polygonSAM":
                        self.current.addPoint(self.line[1], True)
                        points = [
                            [point.x(), point.y()] for point in self.current.points
                        ]
                        labels = [int(label) for label in self.current.labels]
                        self.samPrompt(np.array(points), np.array(labels))
                        if ev.modifiers() == QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    if self.createMode != "polygonSAM":
                        self.current = Shape(
                            shape_type="points"
                            if self.createMode
                            in ["ai_polygon", "ai_mask", "grounding_sam"]
                            else self.createMode
                        )
                        self.current.addPoint(
                            snapped_pos, label=0 if is_shift_pressed else 1
                        )
                        if self.createMode == "point":
                            if is_shift_pressed:
                                set_keypoint_visibility_on_shape_object(
                                    self.current, KeypointVisibility.OCCLUDED
                                )
                            self.finalise()
                        elif (
                            self.createMode
                            in ["ai_polygon", "ai_mask", "grounding_sam"]
                            and ev.modifiers() & QtCore.Qt.ControlModifier
                        ):
                            self.finalise()
                        else:
                            if self.createMode == "circle":
                                self.current.shape_type = "circle"
                            self.line.points = [snapped_pos, snapped_pos]
                            if (
                                self.createMode
                                in ["ai_polygon", "ai_mask", "grounding_sam"]
                                and is_shift_pressed
                            ):
                                self.line.point_labels = [0, 0]
                            else:
                                self.line.point_labels = [1, 1]
                            self.setHiding()
                            self.drawingPolygon.emit(True)
                            self.update()
                    else:
                        self.current = MultipoinstShape()
                        self.current.addPoint(pos, True)
            elif self.editing():
                if self._shared_boundary_reshape_mode:
                    shape, edge_index = self._shared_boundary_source()
                    if shape is not None and edge_index is not None:
                        self._shared_boundary_shape = shape
                        self._shared_boundary_edge_index = edge_index
                        self._dragging_shared_boundary = True
                        self._shared_boundary_last_pos = QtCore.QPointF(pos)
                        self.overrideCursor(CURSOR_MOVE)
                        self.repaint()
                        return
                    ev.accept()
                    return
                if self.selectedEdge():
                    self.addPointToEdge()
                elif (
                    self.selectedVertex() and ev.modifiers() == QtCore.Qt.ShiftModifier
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.removeSelectedPoint()

                group_mode = ev.modifiers() == QtCore.Qt.ControlModifier
                if (
                    not group_mode
                    and not self.selectedEdge()
                    and not self.selectedVertex()
                    and self.hShape is None
                ):
                    pair_id = self._nearest_explicit_landmark_pair(pos)
                    if pair_id:
                        self.overlayLandmarkPairSelected.emit(str(pair_id))
                        ev.accept()
                        return
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton:
            if self.drawing() and self.createMode == "polygonSAM":
                if self.current:
                    self.current.addPoint(self.line[1], False)
                    points = [[point.x(), point.y()] for point in self.current.points]
                    labels = [int(label) for label in self.current.labels]
                    self.samPrompt(np.array(points), np.array(labels))
                    if ev.modifiers() == QtCore.Qt.ControlModifier:
                        self.finalise()
                elif not self.outOfPixmap(pos):
                    self.current = MultipoinstShape()
                    self.current.addPoint(pos, False)
                    self.line.points = [pos, pos]
                    self.setHiding()
                    self.drawingPolygon.emit(True)
                    self.update()
            elif self.editing():
                group_mode = ev.modifiers() == QtCore.Qt.ControlModifier
                selected_ids = {id(s) for s in (self.selectedShapes or [])}
                if not self.selectedShapes or (
                    self.hShape is not None and id(self.hShape) not in selected_ids
                ):
                    self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                    self.repaint()
                self.prevPoint = pos

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.createMode != "polygonSAM":
                main_window = self.window()
                menu = self._build_context_menu(main_window)
                self.restoreCursor()
                if (
                    not menu.exec_(self.mapToGlobal(ev.pos()))
                    and self.selectedShapesCopy
                ):
                    # Cancel the move by deleting the shadow copy.
                    self.selectedShapesCopy = []
                    self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if self._dragging_shared_boundary:
                    self._dragging_shared_boundary = False
                    self._shared_boundary_last_pos = None
                    self._clearSharedBoundaryReshape()
                    self._shared_finalize_topology_edit()
                    if self.movingShape:
                        self.storeShapes()
                        self.shapeMoved.emit()
                    self.movingShape = False
                    self.repaint()
                    ev.accept()
                    return
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [
                            x
                            for x in self.selectedShapes
                            if self.hShape is None or id(x) != id(self.hShape)
                        ]
                    )

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if self.shapesBackups[-1][index].points != self.shapes[index].points:
                self._shared_finalize_topology_edit()
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self._shared_finalize_topology_edit()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        if not self.drawing() or self.current is None:
            return False
        if self.createMode == "polygonSAM":
            return len(self.current) > 0
        if self.createMode in [
            "polygon",
            "linestrip",
            "ai_polygon",
            "ai_mask",
            "grounding_sam",
        ]:
            return len(self.current) > 2
        return False

    def _trim_double_click_tail_point(self) -> None:
        """Remove redundant trailing point introduced by double-click press flow."""
        if self.current is None or len(self.current) <= 1:
            return
        # Historical polygon behavior removes the tail click point on close.
        if self.createMode == "polygon" and len(self.current) > 3:
            self.current.popPoint()
            return
        # For AI prompt modes and linestrip, only trim when the point is effectively
        # duplicated (same physical click landing nearly at the same spot).
        if self.createMode in ["linestrip", "ai_polygon", "ai_mask", "grounding_sam"]:
            last_point = self.current[-1]
            prev_point = self.current[-2]
            if self.closeEnough(last_point, prev_point):
                self.current.popPoint()

    def mouseDoubleClickEvent(self, ev):
        if self.double_click != "close":
            return
        if not self.drawing() or self.current is None:
            return
        self._trim_double_click_tail_point()
        if self.canCloseShape():
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            selected_ids = {id(s) for s in (self.selectedShapes or [])}
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if id(shape) not in selected_ids:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(self.selectedShapes + [shape])
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            if s.shape_type == "mask":
                continue
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        try:
            point = shape[index]
        except IndexError as e:
            logger.info(e)
            return
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)
        self._sync_shared_vertex(shape, index, shape[index])

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPoint(int(min(0, o1.x())), int(min(0, o1.y())))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPoint(
                int(min(0, self.pixmap.width() - o2.x())),
                int(min(0, self.pixmap.height() - o2.y())),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            registry = getattr(self, "_shared_topology_registry", None)
            if isinstance(registry, SharedTopologyRegistry):
                registry.translate_shapes(shapes, dp)
            else:
                self._shared_move_selected_shapes(shapes, dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                # Remove by identity (Shape implements fuzzy __eq__).
                for idx, s in enumerate(list(self.shapes)):
                    if s is shape:
                        del self.shapes[idx]
                        break
                deleted_shapes.append(shape)
            self._shared_finalize_topology_edit()
            self.storeShapes()
            self.selectedShapes = []
            # Keep the rest of the UI in sync (toolbar actions, label list).
            try:
                self.selectionChanged.emit([])
            except Exception:
                pass
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if self.selectedShapes:
            shape_id = id(shape)
            self.selectedShapes = [s for s in self.selectedShapes if id(s) != shape_id]
        if self.shapes:
            for idx, s in enumerate(list(self.shapes)):
                if s is shape:
                    del self.shapes[idx]
                    break
        self._shared_finalize_topology_edit()
        self.storeShapes()
        if not self.selectedShapes:
            try:
                self.selectionChanged.emit([])
            except Exception:
                pass
        self.update()

    def duplicateSelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def _paint_ai_polygon_preview(self, painter):
        if self.current is None:
            return
        drawing_shape = self.current.copy()
        if len(self.line.points) < 2:
            return
        drawing_shape.addPoint(
            point=self.line.points[1],
            label=self.line.point_labels[1],
        )
        try:
            if not self._ensure_ai_model_initialized():
                logger.error(
                    "AI polygon model is not initialized; skipping prediction."
                )
                return
            prompt_points = [[point.x(), point.y()] for point in drawing_shape.points]
            point_labels = list(drawing_shape.point_labels or [])
            normalized_points = self._predict_ai_polygon_points(
                prompt_points=prompt_points,
                point_labels=point_labels,
            )
            if len(normalized_points) > 2:
                drawing_shape.setShapeRefined(
                    shape_type="polygon",
                    points=normalized_points,
                    point_labels=[1] * len(normalized_points),
                )
                drawing_shape.fill = self.fillDrawing()
                drawing_shape.paint(painter)
        except Exception as e:
            # Downgrade to warning since this can happen frequently with invalid inputs
            # and is often a non-critical model error (e.g. Expand node errors).
            logger.warning(f"AI polygon prediction failed: {e}")

    def _paint_ai_mask_preview(self, painter):
        if self.current is None:
            return
        drawing_shape = self.current.copy()
        if len(self.line.points) < 2:
            return
        drawing_shape.addPoint(
            point=self.line.points[1],
            label=self.line.point_labels[1],
        )
        try:
            if not self._ensure_ai_model_initialized():
                logger.error("AI mask model is not initialized; skipping prediction.")
                return
            mask = self._ai_model.predict_mask_from_points(
                points=[[point.x(), point.y()] for point in drawing_shape.points],
                point_labels=drawing_shape.point_labels,
            )
            bbox = self._mask_bbox(mask)
            if bbox is None:
                return
            y1, x1, y2, x2 = bbox
            drawing_shape.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1:y2, x1:x2],
            )
            drawing_shape.selected = True
            drawing_shape.paint(painter)
        except Exception as e:
            # Downgrade to warning as this is often a recoverable model/input error.
            logger.warning(f"AI mask prediction failed: {e}")

    def _predict_ai_polygon_points(
        self, *, prompt_points: list[list[float]], point_labels: list[int]
    ) -> list[QtCore.QPointF]:
        """Predict AI polygon points from the SAM mask, with polygon fallback only if needed."""
        return _ai_predict(
            ai_model=getattr(self, "_ai_model", None),
            pixmap=self.pixmap,
            prompt_points=prompt_points,
            point_labels=point_labels,
        )

    @staticmethod
    def _mask_bbox(mask):
        return _ai_mask_bbox(mask)

    def _polygon_from_prompt_mask(
        self, mask, *, prompt_points: list[list[float]], point_labels: list[int]
    ) -> list[QtCore.QPointF]:
        """Convert the current refined mask into a stable polygon."""
        return _ai_polygon_from_mask(
            mask,
            pixmap=self.pixmap,
            prompt_points=prompt_points,
            point_labels=point_labels,
        )

    def _normalize_ai_polygon_points(self, points) -> list[QtCore.QPointF]:
        return _ai_normalize(points, self.pixmap)

    def _simplify_ai_polygon_points(
        self, points: list[QtCore.QPointF] | np.ndarray
    ) -> list[QtCore.QPointF]:
        return _ai_simplify(points, self.pixmap)

    def _build_polygon_preview_line(self):
        if self.current is None or len(self.current.points) == 0:
            return None
        if len(self.line.points) < 2:
            return None
        preview = Shape(
            shape_type="line", line_color=getattr(self.line, "line_color", None)
        )
        preview.points = [
            QtCore.QPointF(self.current[-1]),
            QtCore.QPointF(self.line.points[1]),
        ]
        preview.point_labels = [1, 1]
        return preview

    @staticmethod
    def _materialize_polygon_shape(source_shape, points):
        """Create a stable polygon shape detached from transient AI prompt state."""
        polygon = Shape(
            label=getattr(source_shape, "label", None),
            line_color=getattr(source_shape, "line_color", None),
            shape_type="polygon",
            flags=copy.deepcopy(getattr(source_shape, "flags", None)),
            group_id=getattr(source_shape, "group_id", None),
            description=getattr(source_shape, "description", None),
            visible=bool(getattr(source_shape, "visible", True)),
        )
        polygon.other_data = copy.deepcopy(
            dict(getattr(source_shape, "other_data", {}) or {})
        )
        polygon.fill = bool(getattr(source_shape, "fill", False))
        polygon.selected = bool(getattr(source_shape, "selected", False))
        polygon.points = [QtCore.QPointF(point) for point in points]
        polygon.point_labels = [1] * len(polygon.points)
        polygon.close()
        return polygon

    def _clear_preview_line(self):
        self.line = Shape()

    def cancelCurrentDrawing(self, *, clear_sam_mask: bool = False) -> None:
        self.current = None
        if clear_sam_mask:
            self.sam_mask = MaskShape()
        self._clear_preview_line()
        self._clear_adjoining_source()
        self._clearSharedBoundaryReshape()
        self.setHiding(False)
        self.drawingPolygon.emit(False)

    def _clear_ai_prompt_state(self) -> None:
        """Drop transient AI prompt state when the source image changes."""
        self.cancelCurrentDrawing(clear_sam_mask=True)
        try:
            if hasattr(self, "sam_image_scale"):
                delattr(self, "sam_image_scale")
        except Exception:
            pass

    @staticmethod
    def _collect_explicit_landmark_pairs_from_shapes(shapes):
        overlay_points = {}
        image_points = {}
        pair_visible = {}
        for shape in list(shapes or []):
            if str(getattr(shape, "shape_type", "") or "").lower() != "point":
                continue
            points = getattr(shape, "points", None) or []
            if not points:
                continue
            other = dict(getattr(shape, "other_data", {}) or {})
            pair_id = str(other.get("overlay_landmark_pair_id") or "")
            if not pair_id:
                continue
            point = points[0]
            coords = (float(point.x()), float(point.y()))
            if "overlay_id" in other:
                overlay_points[pair_id] = coords
                pair_visible[pair_id] = bool(
                    other.get("overlay_landmarks_visible", True)
                )
            else:
                image_points[pair_id] = coords
        pairs = []
        for pair_id, src in overlay_points.items():
            dst = image_points.get(pair_id)
            if dst is None or not bool(pair_visible.get(pair_id, True)):
                continue
            pairs.append((pair_id, src, dst))
        return pairs

    @staticmethod
    def _explicit_landmark_pair_pen(scale: float, *, selected: bool = False):
        color = (
            QtGui.QColor(255, 140, 0, 240)
            if selected
            else QtGui.QColor(255, 215, 0, 220)
        )
        pen = QtGui.QPen(color)
        base_width = 3.0 if selected else 2.0
        pen.setWidthF(max(1.5, base_width / max(scale, 0.01)))
        pen.setStyle(QtCore.Qt.DashLine)
        return pen

    def setSelectedOverlayLandmarkPair(self, pair_id: str | None) -> None:
        normalized = str(pair_id or "") or None
        if normalized == self._selected_overlay_landmark_pair_id:
            return
        self._selected_overlay_landmark_pair_id = normalized
        self.update()

    def _nearest_explicit_landmark_pair(self, pos, tolerance: float | None = None):
        threshold = float(
            tolerance if tolerance is not None else self.epsilon / max(self.scale, 0.01)
        )
        best_pair_id = None
        best_distance = None
        query = QtCore.QPointF(pos)
        for pair_id, src, dst in self._collect_explicit_landmark_pairs_from_shapes(
            self.shapes
        ):
            line = QtCore.QLineF(
                QtCore.QPointF(src[0], src[1]), QtCore.QPointF(dst[0], dst[1])
            )
            length = line.length()
            if length <= 0:
                continue
            dx = line.dx()
            dy = line.dy()
            t = (((query.x() - line.x1()) * dx) + ((query.y() - line.y1()) * dy)) / (
                length * length
            )
            t = max(0.0, min(1.0, t))
            closest = QtCore.QPointF(line.x1() + (dx * t), line.y1() + (dy * t))
            distance = QtCore.QLineF(query, closest).length()
            if distance > threshold:
                continue
            if best_distance is None or distance < best_distance:
                best_pair_id = pair_id
                best_distance = distance
        return best_pair_id

    def _draw_explicit_landmark_pairs(self, painter):
        pairs = self._collect_explicit_landmark_pairs_from_shapes(self.shapes)
        if not pairs:
            return
        painter.save()
        for pair_id, src, dst in pairs:
            painter.setPen(
                self._explicit_landmark_pair_pen(
                    self.scale,
                    selected=pair_id == self._selected_overlay_landmark_pair_id,
                )
            )
            painter.drawLine(
                QtCore.QPointF(src[0], src[1]),
                QtCore.QPointF(dst[0], dst[1]),
            )
        painter.restore()

    def _selected_explicit_landmark_pair_points(self):
        selected_pair_id = str(self._selected_overlay_landmark_pair_id or "")
        if not selected_pair_id:
            return []
        for pair_id, src, dst in self._collect_explicit_landmark_pairs_from_shapes(
            self.shapes
        ):
            if pair_id == selected_pair_id:
                return [src, dst]
        return []

    def _draw_selected_explicit_landmark_pair_points(self, painter):
        points = self._selected_explicit_landmark_pair_points()
        if not points:
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        outer_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220))
        outer_pen.setWidthF(max(1.5, 2.0 / max(self.scale, 0.01)))
        painter.setPen(outer_pen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 140, 0, 180)))
        outer_radius = max(6.0, 8.0 / max(self.scale, 0.01))
        inner_radius = outer_radius * 0.45
        for x, y in points:
            center = QtCore.QPointF(x, y)
            painter.drawEllipse(center, outer_radius, outer_radius)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 240)))
            painter.drawEllipse(center, inner_radius, inner_radius)
            painter.setPen(outer_pen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 140, 0, 180)))
        painter.restore()

    def _paint_live_preview(self, painter):
        if self.selectedShapesCopy:
            for shape in self.selectedShapesCopy:
                shape.paint(painter)

        if not self.drawing():
            return

        if self.createMode in ["polygon", "linestrip"] and self.current is not None:
            preview_line = self._build_polygon_preview_line()
            if preview_line is not None:
                preview_line.paint(painter)
        else:
            self.line.paint(painter)

        try:
            if (
                self.fillDrawing()
                and self.createMode == "polygon"
                and self.current is not None
                and len(self.current.points) >= 2
            ):
                drawing_shape = self.current.copy()
                if len(self.line.points) >= 2:
                    drawing_shape.addPoint(self.line[1])
                drawing_shape.fill = True
                drawing_shape.paint(painter)
            elif self.createMode == "ai_polygon" and self.current is not None:
                self._paint_ai_polygon_preview(painter)
            elif self.createMode == "ai_mask" and self.current is not None:
                self._paint_ai_mask_preview(painter)
            elif (
                self.createMode in ["linestrip", "line"]
                and self.drawing()
                and self.current
                and len(self.current.points) > 0
            ):
                if len(self.line.points) < 2:
                    return
                start_point = self.current.points[-1]
                end_point = self.line.points[1]

                dx = end_point.x() - start_point.x()
                dy = end_point.y() - start_point.y()
                distance = (dx**2 + dy**2) ** 0.5

                mid_point = QtCore.QPointF(
                    (start_point.x() + end_point.x()) / 2,
                    (start_point.y() + end_point.y()) / 2,
                )

                font_size = int(max(8, min(20, distance / 10)))

                font = QtGui.QFont()
                font.setPointSize(font_size)

                line_pen = painter.pen()

                painter.setFont(font)
                painter.setPen(line_pen)
                localized_text = QtCore.QCoreApplication.translate(
                    "Canvas", "{:.1f}px".format(distance)
                )
                painter.drawText(mid_point, localized_text)
                painter.drawText(mid_point, f"{distance:.1f}px")

        except Exception as e:
            logger.error(f"An error occurred in paintEvent: {e}")

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(painter_render_hint("Antialiasing"), True)
            try:
                p.setRenderHint(painter_render_hint("HighQualityAntialiasing"), True)
            except AttributeError:
                pass
            p.setRenderHint(painter_render_hint("SmoothPixmapTransform"), True)

            p.scale(self.scale, self.scale)
            p.translate(self.offsetToCenter())

            # Check if pixmap is valid
            if self.pixmap and not self.pixmap.isNull():
                p.drawPixmap(0, 0, self.pixmap)

            self.sam_mask.paint(p)

            if self._patch_similarity_pixmap is not None:
                p.drawPixmap(0, 0, self._patch_similarity_pixmap)

            if self._pca_map_pixmap is not None:
                p.drawPixmap(0, 0, self._pca_map_pixmap)

            if self._depth_preview_pixmap is not None:
                p.drawPixmap(0, 0, self._depth_preview_pixmap)

            if self._flow_preview_pixmap is not None:
                p.drawPixmap(0, 0, self._flow_preview_pixmap)

            if self.current_behavior_text and len(self.current_behavior_text) > 0:
                p.save()
                p.resetTransform()

                if self.pixmap and not self.pixmap.isNull():
                    base_size = min(self.pixmap.width(), self.pixmap.height())
                else:
                    base_size = min(max(self.width(), 1), max(self.height(), 1))
                font_size = max(20, base_size // 40)

                font = QtGui.QFont()
                font.setPointSize(font_size)
                font.setBold(True)
                p.setFont(font)

                metrics = QtGui.QFontMetrics(font)
                text = self.current_behavior_text
                text_bounds = metrics.boundingRect(text)

                margin = 12
                padding_x = 8
                padding_y = 6

                background_rect = QtCore.QRect(
                    margin,
                    margin,
                    text_bounds.width() + 2 * padding_x,
                    text_bounds.height() + 2 * padding_y,
                )

                bg_color = self.behavior_text_background
                if bg_color is None:
                    bg_color = QtGui.QColor(0, 0, 0, 180)
                else:
                    bg_color = QtGui.QColor(bg_color)

                if bg_color.alpha() > 0:
                    p.setPen(QtCore.Qt.NoPen)
                    p.setBrush(QtGui.QBrush(bg_color))
                    p.drawRoundedRect(background_rect, 6, 6)

                text_rect = QtCore.QRect(
                    background_rect.left() + padding_x,
                    background_rect.top() + padding_y,
                    text_bounds.width(),
                    text_bounds.height(),
                )

                # Drop shadow ensures contrast even if the background is light/transparent
                shadow_rect = text_rect.translated(1, 1)
                p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 200)))
                p.drawText(
                    shadow_rect,
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                    text,
                )

                p.setPen(QtGui.QPen(QtGui.QColor(self.behavior_text_color)))
                p.drawText(
                    text_rect,
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                    text,
                )

                p.restore()

            # draw crosshair
            if (not self.createMode == "grounding_sam") and (
                self._crosshair[self._createMode] and self.drawing()
            ):
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1, QtCore.Qt.DotLine))
                p.drawLine(
                    0,
                    int(self.prevMovePoint.y()),
                    self.width() - 1,
                    int(self.prevMovePoint.y()),
                )
                p.drawLine(
                    int(self.prevMovePoint.x()),
                    0,
                    int(self.prevMovePoint.x()),
                    self.height() - 1,
                )

            Shape.scale = self.scale
            MultipoinstShape.scale = self.scale
            if self.pixmap and not self.pixmap.isNull():
                image_width = self.pixmap.width()
                image_height = self.pixmap.height()
            else:
                image_width = None
                image_height = None

            # Draw pose edges behind the keypoints (but above the image).
            self._draw_pose_edges(p)
            self._draw_explicit_landmark_pairs(p)

            for shape in self.shapes:
                if (shape.selected or not self._hideBackround) and self.isVisible(
                    shape
                ):
                    shape.fill = shape.selected or shape == self.hShape
                    original_line = getattr(shape, "line_color", None)
                    original_fill = getattr(shape, "fill_color", None)
                    original_fill_flag = bool(getattr(shape, "fill", False))
                    try:
                        other = dict(getattr(shape, "other_data", {}) or {})
                        stroke = QtGui.QColor(
                            str(other.get("brain3d_overlay_stroke") or "")
                        )
                        fill = QtGui.QColor(
                            str(other.get("brain3d_overlay_fill") or "")
                        )
                        if stroke.isValid():
                            stroke.setAlpha(max(stroke.alpha(), 220))
                            shape.line_color = stroke
                        if fill.isValid():
                            fill.setAlpha(max(fill.alpha(), 80))
                            shape.fill_color = fill
                            shape.fill = True
                        shape.paint(p, image_width, image_height)
                    except SystemError as e:
                        print(e)
                    finally:
                        shape.line_color = original_line
                        shape.fill_color = original_fill
                        shape.fill = original_fill_flag
            self._draw_selected_explicit_landmark_pair_points(p)
            if self.current:
                self.current.paint(p)
            self._paint_live_preview(p)
        finally:
            if p.isActive():
                p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        aw, ah = area.width(), area.height()
        if self.pixmap is None:
            return QtCore.QPointF(0, 0)
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        if self.pixmap is None:
            return True
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        if self.current is None:
            logger.warning("finalise called without an active shape; ignoring.")
            return
        if self.createMode == "rectangle":
            self.current.close()
            rect_shape = self.current
            self.shapes.append(rect_shape)
            self.storeShapes()
            self.current = None
            self.setHiding(False)
            self.newShape.emit()
            self.update()

            # Get the label of the rectangle
            rect_label = rect_shape.label
            # Launch CountGD when requested either via description hints or the UI checkbox.
            description_text = rect_shape.description
            description_flagged = False
            if isinstance(description_text, str):
                lowered = description_text.lower()
                description_flagged = (
                    "exemplar" in lowered
                    or "count" in lowered
                    or description_text.startswith("0")
                )

            checkbox_flagged = False
            try:
                main_window = self.window()
                checkbox = getattr(
                    getattr(main_window, "aiRectangle", None),
                    "_useCountGDCheckbox",
                    None,
                )
                checkbox_flagged = bool(checkbox and checkbox.isChecked())
            except Exception:
                checkbox_flagged = False

            if description_flagged or checkbox_flagged:
                if description_text:
                    if not description_text.startswith("used_"):
                        rect_shape.description = f"used_{description_text}"
                else:
                    rect_shape.description = "used_countgd"
                self.createMode = "grounding_sam"
                # Call predictAiRectangle with the rectangle shape and its label
                self.predictAiRectangle(
                    rect_label,
                    [rect_shape],
                    is_polygon_output=True,
                    use_countgd=True,
                )
            return
        if self.createMode == "ai_polygon":
            # convert points to polygon by an AI model
            if self.current.shape_type != "points":
                return
            if not self._ensure_ai_model_initialized():
                logger.error(
                    "AI polygon model is not initialized; skipping finalisation."
                )
                return
            try:
                prompt_points = [
                    [point.x(), point.y()] for point in self.current.points
                ]
                point_labels = list(self.current.point_labels or [])
            except Exception as exc:
                logger.warning(
                    "AI polygon finalisation failed; keep editing prompts. Error: %s",
                    exc,
                )
                return
            normalized_points = self._predict_ai_polygon_points(
                prompt_points=prompt_points,
                point_labels=point_labels,
            )
            if len(normalized_points) < 3:
                logger.warning(
                    "AI polygon prediction returned an invalid polygon; keep editing prompts."
                )
                return
            self.current = self._materialize_polygon_shape(
                self.current,
                normalized_points,
            )
        elif self.createMode == "ai_mask":
            # convert points to mask by an AI model
            assert self.current.shape_type == "points"
            if not self._ensure_ai_model_initialized():
                logger.error("AI mask model is not initialized; skipping finalisation.")
                return
            try:
                mask = self._ai_model.predict_mask_from_points(
                    points=[[point.x(), point.y()] for point in self.current.points],
                    point_labels=self.current.point_labels,
                )
            except Exception as exc:
                logger.warning(
                    "AI mask finalisation failed; keep editing prompts. Error: %s",
                    exc,
                )
                return
            bbox = self._mask_bbox(mask)
            if bbox is None:
                logger.error("AI mask prediction returned an empty mask.")
                return
            y1, x1, y2, x2 = bbox
            self.current.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1:y2, x1:x2],
            )
        if self.createMode != "grounding_sam":
            self.current.close()
        if self.createMode == "polygonSAM":
            self.shapes.append(self.sam_mask)
        else:
            self.shapes.append(self.current)
        self._shared_finalize_topology_edit()
        self.storeShapes()
        self.sam_mask = MaskShape()
        self.current = None
        self._clear_adjoining_source()
        self._clear_preview_line()
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def _polygon_close_target(self, pos):
        if self.current is None or self.createMode != "polygon":
            return None
        if len(getattr(self.current, "points", []) or []) < 2:
            return None
        if pos is None:
            return None
        return self.current[0] if self.closeEnough(pos, self.current[0]) else None

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if mods == QtCore.Qt.ControlModifier:
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if mods == QtCore.Qt.ControlModifier:
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (mods == QtCore.Qt.ShiftModifier)
                        else QtCore.Qt.Vertical,
                    )
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(self.selectedShapes, self.prevPoint + offset)
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        handled = False
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.cancelCurrentDrawing(clear_sam_mask=True)
                self.update()
                handled = True
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
                handled = True
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
                handled = True
        elif self.editing():
            if key in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                if self.selectedVertex() and self.removeSelectedPoint():
                    self.storeShapes()
                    self.shapeMoved.emit()
                    self.movingShape = False
                    self.update()
                    handled = True
                elif self._trigger_delete_selected_shapes():
                    handled = True
            elif key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
                handled = True
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
                handled = True
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
                handled = True
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))
                handled = True
            elif key == QtCore.Qt.Key_Escape and self._shared_boundary_reshape_mode:
                self._clearSharedBoundaryReshape()
                self.restoreCursor()
                self.update()
                handled = True
        if handled:
            ev.accept()
            return
        super(Canvas, self).keyPressEvent(ev)

    def _trigger_delete_selected_shapes(self) -> bool:
        if not self.selectedShapes:
            fallback_shape = self.hShape if self.hShape is not None else self.prevhShape
            if fallback_shape is not None and any(
                shape is fallback_shape for shape in self.shapes
            ):
                try:
                    self.selectShapes([fallback_shape])
                except Exception:
                    pass
        if not self.selectedShapes:
            return False
        host = self.window()
        if host is None:
            return False
        try:
            actions = getattr(host, "actions", None)
            delete_action = (
                getattr(actions, "deleteShapes", None) if actions is not None else None
            )
            if delete_action is not None and delete_action.isEnabled():
                delete_action.trigger()
                return True
        except Exception:
            pass
        try:
            delete_fn = getattr(host, "deleteSelectedShapes", None)
            if callable(delete_fn):
                delete_fn()
                return True
        except Exception:
            return False
        return False

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if modifiers == QtCore.Qt.NoModifier:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                try:
                    index = self.shapes.index(self.selectedShapes[0])
                except ValueError:
                    return
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        polygons = None
        if isinstance(self.shapes[-1], MaskShape):
            mask_shape = self.shapes.pop()
            polygons = mask_shape.toPolygons(self.sam_config["approxpoly_epsilon"])
            self.shapes.extend(polygons)
        self.shapesBackups.pop()
        self.storeShapes()

        if isinstance(polygons, list):
            return polygons
        else:
            return self.shapes[-1:]

    def undoLastLine(self):
        assert self.shapes
        self._clearSharedBoundaryReshape()
        self.current = self.shapes.pop()
        if self.createMode != "polygonSAM":
            self.current.setOpen()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode in ["point", "polygonSAM"]:
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self._clearSharedBoundaryReshape()
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self._clear_adjoining_source()
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        previous_signature = self._ai_model_image_signature_value()
        self.pixmap = pixmap
        current_signature = self._ai_model_image_signature_value()
        if (
            previous_signature is not None
            and current_signature is not None
            and previous_signature != current_signature
        ):
            self._clear_ai_prompt_state()
        if self._ai_model is not None and self.pixmap is not None:
            self._sync_ai_model_image(force=False)
        if clear_shapes:
            self.shapes = []
            self.sam_mask = MaskShape()
            self.current = None
            self._clear_preview_line()

        if self.createMode == "polygonSAM" and self.pixmap and self.sam_predictor:
            self.samEmbedding()
        self.update()

    def closeEvent(self, event):
        self._release_model(self._ai_model, context="canvas AI")
        self._release_model(self.sam_hq_model, context="SAM HQ")
        self._release_model(self._ai_model_rect, context="rectangle detector")
        self._ai_model = None
        self.sam_hq_model = None
        self._ai_model_rect = None
        super().closeEvent(event)

    def _predict_similar_rectangles(
        self, rectangle_shapes=None, prompt=None, confidence_threshold=0.23
    ):
        """Predict more rectangle shapes without modifying self.shapes directly here."""
        detected_boxes = []
        if self.pixmap and not self.pixmap.isNull():
            try:
                from annolid.detector.countgd.predict import ObjectCounter

                qt_image = self.pixmap.toImage()
                image_data = convert_qt_image_to_rgb_cv_image(qt_image)
                pil_img = Image.fromarray(image_data)
                object_counter = ObjectCounter()
                exemplar_boxes = []
                for rectangle_shape in rectangle_shapes:
                    x1 = int(
                        min(
                            rectangle_shape.points[0].x(), rectangle_shape.points[1].x()
                        )
                    )
                    y1 = int(
                        min(
                            rectangle_shape.points[0].y(), rectangle_shape.points[1].y()
                        )
                    )
                    x2 = int(
                        max(
                            rectangle_shape.points[0].x(), rectangle_shape.points[1].x()
                        )
                    )
                    y2 = int(
                        max(
                            rectangle_shape.points[0].y(), rectangle_shape.points[1].y()
                        )
                    )
                    exemplar_boxes.append([x1, y1, x2, y2])

                detected_boxes = object_counter.count_objects(
                    pil_img,
                    text_prompt=prompt,
                    exemplar_image=pil_img,
                    exemplar_boxes=exemplar_boxes,
                    confidence_threshold=confidence_threshold,
                )

            except ImportError:
                logger.warning(
                    "CountGD is not available. Install optional dependency and "
                    "its requirements to enable CountGD-assisted prompting."
                )
            except Exception as e:
                logger.error(f"Error in CountGD: {e}")
        return detected_boxes

    def loadShapes(self, shapes, replace=True):
        self._clearSharedBoundaryReshape()
        if replace:
            self.shapes = list(shapes)
            # Visibility is tracked per-shape identity; replacing shapes means
            # prior entries are stale.
            self.visible = {}
        else:
            self.shapes.extend(shapes)
        self._shared_finalize_topology_edit()
        self.storeShapes()
        self.current = None
        self._clear_preview_line()
        self.sam_mask = MaskShape()
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setRealtimeShapes(self, shapes):
        """Replace shapes without touching the undo stack."""
        self._clearSharedBoundaryReshape()
        self.shapes = list(shapes or [])
        self._shared_finalize_topology_edit()
        self.current = None
        self._clear_preview_line()
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.selectedShapes = []
        self.selectedShapesCopy = []
        self.update()

    def setShapeVisible(self, shape, value, *, emit_selection=True):
        _ = emit_selection
        try:
            shape.visible = bool(value)
        except Exception:
            pass
        self.visible[id(shape)] = bool(value)
        self.update()

    def overrideCursor(self, cursor):
        self._cursor = cursor
        try:
            self.setCursor(cursor)
        except Exception:
            pass
        viewport = getattr(self, "viewport", None)
        if callable(viewport):
            try:
                vp = viewport()
                if vp is not None:
                    vp.setCursor(cursor)
            except Exception:
                pass

    def restoreCursor(self):
        try:
            self.unsetCursor()
        except Exception:
            pass
        viewport = getattr(self, "viewport", None)
        if callable(viewport):
            try:
                vp = viewport()
                if vp is not None:
                    vp.unsetCursor()
            except Exception:
                pass

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.shapes = []
        self.sam_mask = MaskShape()
        self.update()
