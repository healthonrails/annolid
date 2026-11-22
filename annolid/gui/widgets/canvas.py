"""Interactive 2D annotation canvas widget for Annolid."""

from __future__ import annotations

import copy
from pathlib import Path
import threading
from typing import Callable

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QLabel

from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    set_keypoint_visibility_on_shape_object,
)
from annolid.annotation.polygon_constraints import (
    DEFAULT_POLYGON_CLEARANCE_PX,
    make_instance_masks_exclusive,
    resolve_polygon_shape_conflicts,
)
from annolid.annotation.pose_schema import PoseSchema
from annolid.gui.cursor_utils import set_widget_busy_cursor
from annolid.gui.mixins.shared_polygon_edit_mixin import SharedPolygonEditMixin
from annolid.gui.qt_compat import (
    event_pos_f,
    event_pos_i,
    painter_render_hint,
)
from annolid.gui.shape import MaskShape, MultipoinstShape, Shape
from annolid.gui.shape_geometry import (
    circle_from_three_points,
    clamp_point_to_bounds,
    find_neighbor_shape,
    intersect_segment_with_bounds,
    move_oriented_rect_vertex,
    oriented_rect_from_three_points,
    rotate_oriented_rect_by_vertex,
    shape_bounds,
)
from annolid.gui.shared_vertices import SharedTopologyRegistry
from annolid.gui.widgets.ai_polygon_helpers import (
    mask_bbox as _ai_mask_bbox,
    normalize_ai_polygon_points as _ai_normalize,
    polygon_from_refined_mask as _ai_polygon_from_mask,
    predict_ai_polygon_points as _ai_predict,
    simplify_ai_polygon_points as _ai_simplify,
)
from annolid.gui.widgets.canvas_ai_prompts import (
    CanvasAiPromptMixin,
    TextPromptComputationResult,
    _ANCHORED_QT_WORKERS,
    _ANCHORED_QT_WORKERS_LOCK,
    _PROMPT_WORKER_STACK_SIZE,
    _anchor_qt_worker,
    _release_anchored_qt_worker,
)
from annolid.gui.widgets.canvas_context_menu import build_canvas_context_menu
from annolid.gui.widgets.canvas_overlays import CanvasOverlayMixin
from annolid.gui.workers import FlexibleWorker
from annolid.utils.annotation_compat import AI_MODELS, utils
from annolid.utils.logger import logger

__all__ = [
    "Canvas",
    "TextPromptComputationResult",
    "AI_MODELS",
    "make_instance_masks_exclusive",
    "resolve_polygon_shape_conflicts",
    "utils",
    "_ANCHORED_QT_WORKERS",
    "_ANCHORED_QT_WORKERS_LOCK",
    "_PROMPT_WORKER_STACK_SIZE",
    "_anchor_qt_worker",
    "_release_anchored_qt_worker",
]

CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0

_VALID_CREATE_MODES = frozenset(
    {
        "polygon",
        "rectangle",
        "oriented_rectangle",
        "circle",
        "circle_3p",
        "line",
        "point",
        "linestrip",
        "polygonSAM",
        "ai_polygon",
        "grounding_sam",
        "ai_mask",
    }
)


class Canvas(
    CanvasAiPromptMixin,
    CanvasOverlayMixin,
    SharedPolygonEditMixin,
    QtWidgets.QWidget,
):
    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, object)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)
    edgeSelected = QtCore.Signal(bool)
    overlayLandmarkPairSelected = QtCore.Signal(str)

    CREATE, EDIT = 0, 1

    _createMode = "polygon"
    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(self.double_click)
            )
        self.num_backups = max(1, int(kwargs.pop("num_backups", 10)))
        self._allow_points_outside_image = bool(
            kwargs.pop("allow_points_outside_image", False)
        )
        self._show_label_names = bool(kwargs.pop("show_label_names", True))
        default_crosshair = {
            "polygon": False,
            "rectangle": True,
            "oriented_rectangle": True,
            "grounding_sam": False,
            "circle": False,
            "circle_3p": False,
            "line": False,
            "point": False,
            "linestrip": False,
            "ai_polygon": False,
            "ai_mask": False,
            "polygonSAM": False,
        }
        provided_crosshair = kwargs.pop("crosshair", None)
        if isinstance(provided_crosshair, dict):
            default_crosshair.update(provided_crosshair)
        self._crosshair = default_crosshair
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

        # Local annotation state.
        self.mode = self.EDIT
        self.shapes: list[Shape] = []
        self.shapesBackups: list[list[Shape]] = [[]]
        self.current: Shape | None = None
        self.selectedShapes: list[Shape] = []
        self.selectedShapesCopy: list[Shape] = []
        self.line = Shape()
        self.prevPoint = QtCore.QPointF()
        self.prevMovePoint = QtCore.QPointF()
        self.offsets = QtCore.QPointF(), QtCore.QPointF()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible: dict[int, bool] = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape: Shape | None = None
        self.prevhShape: Shape | None = None
        self.hVertex: int | None = None
        self.prevhVertex: int | None = None
        self.hEdge: int | None = None
        self.prevhEdge: int | None = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._adjoining_source_shape = None
        self._cursor = CURSOR_DEFAULT
        self.mouse_xy_text = ""
        self._caption_text = ""
        self._icons_dir = Path(__file__).resolve().parents[1] / "icons"
        self._ai_polygon_preview_cache = None

        self.label = QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Compatibility placeholder for legacy callers referencing canvas.menus.
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())

        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

        # AI prompt & background worker state (used by CanvasAiPromptMixin).
        self._ai_model = None
        self._ai_model_pixmap_key = None
        self._ai_model_image_signature = None
        self._ai_model_signature_cache_token = None
        self._ai_model_signature_cache_value = None
        self._ai_model_image_dirty = True
        self._ai_model_rgb_digest: str | None = None
        self._ai_model_rect = None
        self._countgd_counter = None
        self._batch_finalising_prompt_shapes = False
        self._prompt_inference_lock = threading.RLock()
        self._prompt_request_seq = 0
        self._active_prompt_request_id: int | None = None
        self._inflight_prompt_workers: dict[
            int, tuple[QtCore.QThread, FlexibleWorker, threading.Event, int]
        ] = {}
        self._prompt_callbacks: dict[
            int,
            tuple[
                object,
                bool,
                float,
                Callable[[str, int, float, str | None], None] | None,
            ],
        ] = {}
        self._pending_close_model_release = False
        self._canvas_closing = False
        self.sam_predictor = None
        self.sam_hq_model = None
        self.sam_mask = MaskShape()
        self._sam_predictor_missing_logged = False
        self._sam_last_load_error = None

        # Shared polygon topology (large-TIFF mode opt-in).
        self._shared_topology_registry = SharedTopologyRegistry.from_shapes([])
        self._allow_shared_polygon_topology = False
        self._shared_boundary_reshape_mode = False
        self._shared_boundary_shape = None
        self._shared_boundary_edge_index = None
        self._dragging_shared_boundary = False
        self._shared_boundary_last_pos = None

        # Overlays & behavior banner (used by CanvasOverlayMixin).
        self.behavior_text_color = QtGui.QColor(255, 255, 255)
        self.behavior_text_background = QtGui.QColor(0, 0, 0, 180)
        self.current_behavior_text = None
        self._patch_similarity_active = False
        self._patch_similarity_callback = None
        self._patch_similarity_pixmap = None
        self._pca_map_pixmap = None
        self._depth_preview_pixmap = None
        self._flow_preview_pixmap = None
        self._pose_schema: PoseSchema | None = None
        self._show_pose_edges: bool = False
        self._show_pose_bboxes: bool = True
        self._pose_edge_color = QtGui.QColor(0, 255, 255, 190)
        self._pose_edge_shadow = QtGui.QColor(0, 0, 0, 160)
        self._selected_overlay_landmark_pair_id = None

        try:
            self.selectionChanged.connect(self._on_selection_changed)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Configuration & basic accessors
    # ------------------------------------------------------------------
    def _on_selection_changed(self, shapes):
        """Update internal selection state from a newly selected shapes list."""
        selected = list(shapes or [])
        selected_ids = {id(s) for s in selected}
        self.selectedShapes = [s for s in self.shapes if id(s) in selected_ids]
        for s in self.shapes:
            try:
                s.selected = id(s) in selected_ids
            except Exception:
                pass
        if self.hShape is None or id(self.hShape) not in selected_ids:
            self.hShapeIsSelected = False
        self.update()

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = bool(value)

    def allowPointsOutsideImage(self) -> bool:
        return bool(self._allow_points_outside_image)

    def setAllowPointsOutsideImage(self, enabled: bool) -> None:
        self._allow_points_outside_image = bool(enabled)
        self.update()

    def showLabelNames(self) -> bool:
        return bool(self._show_label_names)

    def setShowLabelNames(self, enabled: bool) -> None:
        self._show_label_names = bool(enabled)
        Shape.show_label_names = self._show_label_names
        self.update()

    def setCaption(self, text):
        self._caption_text = str(text or "")

    def getCaption(self):
        return self._caption_text

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in _VALID_CREATE_MODES:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value
        self.sam_mask = MaskShape()
        self.current = None
        self._ai_polygon_preview_cache = None

    # ------------------------------------------------------------------
    # Shared polygon topology & vertex separation helpers
    # ------------------------------------------------------------------
    def _selected_polygon_source(self):
        for shape in self.selectedShapes or []:
            if (
                str(getattr(shape, "shape_type", "") or "").lower() == "polygon"
                and len(getattr(shape, "points", []) or []) >= 2
            ):
                return shape
        return None

    def _adjoining_boundary_source(self):
        if not self._allow_shared_polygon_topology:
            return None
        source = getattr(self, "_adjoining_source_shape", None)
        if source is not None and any(s is source for s in (self.shapes or [])):
            return source
        return self._selected_polygon_source()

    def _shared_boundary_source(self):
        if not self._allow_shared_polygon_topology:
            return None, None
        shape = self.hShape if self.hEdge is not None else None
        edge_index = self.hEdge if self.hEdge is not None else None
        if shape is None or edge_index is None or shape.is_locked():
            return None, None
        if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
            return None, None
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

    @staticmethod
    def _detach_canvas_polygon_topology(shape) -> None:
        if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
            return
        point_count = len(getattr(shape, "points", []) or [])
        shape.shared_vertex_ids = [""] * point_count
        shape.shared_edge_ids = [""] * point_count

    def _shared_finalize_topology_edit(self):
        if not getattr(self, "_allow_shared_polygon_topology", False):
            self._shared_topology_registry = SharedTopologyRegistry.from_shapes([])
            return self._shared_topology_registry
        return super()._shared_finalize_topology_edit()

    def _separate_canvas_polygon_vertices(self, shape) -> bool:
        """Move exact cross-polygon vertex matches by a sub-pixel clearance."""
        if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
            return False
        points = list(getattr(shape, "points", []) or [])
        if not points:
            return False

        occupied: list[tuple[float, float]] = []
        for existing in self.shapes or []:
            if str(getattr(existing, "shape_type", "") or "").lower() != "polygon":
                continue
            occupied.extend(
                (float(point.x()), float(point.y()))
                for point in (getattr(existing, "points", []) or [])
            )

        clearance = float(DEFAULT_POLYGON_CLEARANCE_PX)
        equality_tolerance = 1e-9
        has_pixmap = self.pixmap is not None and not self.pixmap.isNull()
        pixmap_width = int(self.pixmap.width()) if has_pixmap else 0
        pixmap_height = int(self.pixmap.height()) if has_pixmap else 0

        def _is_available(x: float, y: float) -> bool:
            if not self._allow_points_outside_image:
                if pixmap_width > 0 and not (0.0 <= x <= float(pixmap_width - 1)):
                    return False
                if pixmap_height > 0 and not (0.0 <= y <= float(pixmap_height - 1)):
                    return False
            return all(
                abs(x - other_x) > equality_tolerance
                or abs(y - other_y) > equality_tolerance
                for other_x, other_y in occupied
            )

        changed = False
        offsets = (
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0),
        )
        for index, point in enumerate(points):
            x, y = float(point.x()), float(point.y())
            if _is_available(x, y):
                occupied.append((x, y))
                continue
            replacement = None
            for radius in range(1, 65):
                for dx, dy in offsets:
                    candidate_x = x + dx * clearance * radius
                    candidate_y = y + dy * clearance * radius
                    if _is_available(candidate_x, candidate_y):
                        replacement = (candidate_x, candidate_y)
                        break
                if replacement is not None:
                    break
            if replacement is None:
                logger.warning(
                    "Could not separate Canvas polygon vertex %s at (%s, %s).",
                    index,
                    x,
                    y,
                )
                occupied.append((x, y))
                continue
            shape.points[index] = QtCore.QPointF(*replacement)
            occupied.append(replacement)
            changed = True

        self._detach_canvas_polygon_topology(shape)
        return changed

    def _sync_shared_vertex(self, shape, index, point=None):
        if not self._allow_shared_polygon_topology:
            return None
        try:
            return self._shared_sync_vertex(shape, index, point=point)
        except Exception:
            logger.debug("Failed to synchronize shared vertex.", exc_info=True)
            return None

    def _snap_to_adjoining_boundary(self, pos):
        if not self._allow_shared_polygon_topology:
            return QtCore.QPointF(pos), None
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

    def _selected_shared_boundary_candidate(self):
        shape = self._selected_polygon_source()
        if shape is None or shape.is_locked():
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
        if not self._allow_shared_polygon_topology:
            return False
        return self._selected_polygon_source() is not None

    def canStartSharedBoundaryReshape(self) -> bool:
        if not self._allow_shared_polygon_topology:
            return False
        if not self.editing():
            return False
        return self._selected_shared_boundary_candidate() is not None

    def startSharedBoundaryReshape(self) -> bool:
        if not self.canStartSharedBoundaryReshape():
            return False
        shape = self._selected_shared_boundary_candidate()
        if shape is None:
            return False
        self._shared_boundary_reshape_mode = True
        self._shared_boundary_shape = shape
        self._shared_boundary_edge_index = None
        self._dragging_shared_boundary = False
        self._shared_boundary_last_pos = None
        self.selectShapes([shape])
        self.overrideCursor(CURSOR_POINT)
        self.setToolTip(self.tr("Drag a highlighted shared boundary edge to reshape"))
        self.setStatusTip(self.toolTip())
        self.update()
        return True

    def _clearSharedBoundaryReshape(self) -> None:
        self._shared_boundary_reshape_mode = False
        self._shared_boundary_shape = None
        self._shared_boundary_edge_index = None
        self._dragging_shared_boundary = False
        self._shared_boundary_last_pos = None

    def _reshapeSharedBoundaryBy(self, delta) -> bool:
        if (
            self._shared_boundary_shape is None
            or self._shared_boundary_edge_index is None
            or self._shared_boundary_shape.is_locked()
        ):
            return False
        return bool(
            self._shared_reshape_edge(
                self._shared_boundary_shape,
                int(self._shared_boundary_edge_index),
                delta,
            )
        )

    def beginAdjoiningPolygonFromSeed(self, seed_shape) -> bool:
        if not self._allow_shared_polygon_topology or seed_shape is None:
            return False
        points = list(getattr(seed_shape, "points", []) or [])
        if len(points) < 2:
            return False
        self.setEditing(False)
        self.createMode = "polygon"
        self.current = seed_shape
        self.line.points = [QtCore.QPointF(points[-1]), QtCore.QPointF(points[-1])]
        self.line.point_labels = [1, 1]
        self.setHiding()
        self.drawingPolygon.emit(True)
        self.update()
        return True

    def startAdjoiningPolygonFromSelection(self, edge_index=None) -> bool:
        if not self._allow_shared_polygon_topology:
            return False
        source = self._selected_polygon_source()
        if source is None:
            return False
        self._adjoining_source_shape = source
        self.setEditing(False)
        self.createMode = "polygon"
        self.current = None
        self._clear_preview_line()
        self.update()
        return True

    # ------------------------------------------------------------------
    # Undo / backup stack
    # ------------------------------------------------------------------
    def storeShapes(self):
        shapesBackup = [shape.copy() for shape in self.shapes]
        # Keep up to num_backups prior snapshots plus the new current snapshot.
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (with an initial baseline snapshot),
        # so an edit is undoable whenever >= 2 snapshots exist.
        return len(self.shapesBackups) >= 2

    def restoreShape(self):
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # discard current state
        # Caller (undoShapeEdit -> loadShapes) will push the restored state back.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self._shared_finalize_topology_edit()
        self.selectedShapes = []
        self.selectedShapesCopy = []
        self.visible = {id(s): bool(getattr(s, "visible", True)) for s in self.shapes}
        self.unHighlight()
        self.prevhShape = None
        self.prevhVertex = None
        self.prevhEdge = None
        for shape in self.shapes:
            shape.selected = False
        try:
            self.selectionChanged.emit([])
        except Exception:
            pass
        self.update()

    def _shape_index_by_identity(self, target) -> int | None:
        for idx, shape in enumerate(self.shapes):
            if shape is target:
                return idx
        return None

    def _has_shape_moved_since_backup(self, shape) -> bool:
        idx = self._shape_index_by_identity(shape)
        if idx is None:
            return False
        if not self.shapesBackups or idx >= len(self.shapesBackups[-1]):
            return True
        prev_pts = self.shapesBackups[-1][idx].points
        cur_pts = self.shapes[idx].points
        if len(prev_pts) != len(cur_pts):
            return True
        return any(
            abs(float(a.x()) - float(b.x())) > 1e-6
            or abs(float(a.y()) - float(b.y())) > 1e-6
            for a, b in zip(prev_pts, cur_pts)
        )

    # ------------------------------------------------------------------
    # Visibility, mode & highlight state
    # ------------------------------------------------------------------
    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self._clearSharedBoundaryReshape()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.snapping = True
        self._clearSharedBoundaryReshape()
        self.restoreCursor()

    def isVisible(self, shape):
        if not self._show_pose_bboxes and self._is_pose_bbox_shape(shape):
            return False
        if not bool(getattr(shape, "visible", True)):
            return False
        return self.visible.get(id(shape), True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            self.repaint()
        else:
            self.unHighlight()
            self.deSelectShape()
        self._clearSharedBoundaryReshape()

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

    def _shape_points_for_preview(self, pos: QtCore.QPointF) -> None:
        """Update `self.line` geometry for the active drawing mode."""
        is_shift_pressed = bool(
            QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier
        )
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
        elif self.createMode == "oriented_rectangle":
            if len(self.current.points) == 1:
                self.line.shape_type = "line"
                self.line.points = [self.current[0], pos]
            elif len(self.current.points) >= 2:
                corners = oriented_rect_from_three_points(
                    self.current.points[0], self.current.points[1], pos
                )
                if corners is not None:
                    self.line.shape_type = "oriented_rectangle"
                    self.line.points = [QtCore.QPointF(x, y) for x, y in corners]
                    self.line.close()
                else:
                    self.line.shape_type = "line"
                    self.line.points = [self.current.points[0], self.current.points[1]]
        elif self.createMode == "circle":
            self.line.points = [self.current[0], pos]
            self.line.shape_type = "circle"
        elif self.createMode == "circle_3p":
            if len(self.current.points) == 1:
                self.line.shape_type = "line"
                self.line.points = [self.current[0], pos]
            elif len(self.current.points) >= 2:
                fitted = circle_from_three_points(
                    self.current.points[0], self.current.points[1], pos
                )
                if fitted is not None:
                    (cx, cy), (ex, ey) = fitted
                    self.line.shape_type = "circle"
                    self.line.points = [
                        QtCore.QPointF(cx, cy),
                        QtCore.QPointF(ex, ey),
                    ]
                else:
                    self.line.shape_type = "linestrip"
                    self.line.points = [
                        self.current.points[0],
                        self.current.points[1],
                        pos,
                    ]
        elif self.createMode == "line":
            self.line.points = [self.current[0], pos]
            self.line.close()
        elif self.createMode == "point":
            self.line.points = [self.current[0]]
            self.line.close()
        elif self.createMode == "polygonSAM":
            self.line.points = [pos, pos]

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------
    def _clamp_drawing_pos(
        self, pos: QtCore.QPointF, anchor: QtCore.QPointF | None = None
    ) -> QtCore.QPointF:
        if (
            self._allow_points_outside_image
            or self.pixmap is None
            or self.pixmap.isNull()
        ):
            return QtCore.QPointF(pos)
        if not self.outOfPixmap(pos):
            return QtCore.QPointF(pos)
        if anchor is not None:
            return self.intersectionPoint(anchor, pos)
        cx, cy = clamp_point_to_bounds(
            pos.x(), pos.y(), self.pixmap.width(), self.pixmap.height()
        )
        return QtCore.QPointF(cx, cy)

    def mouseMoveEvent(self, ev):
        """Update live drawing preview, shape drag, or hover highlight."""
        try:
            raw_pos = event_pos_f(ev)
            pos = self.transformPos(raw_pos)
        except AttributeError:
            return

        x, y = raw_pos.x(), raw_pos.y()
        self.mouse_xy_text = f"x:{pos.x():.1f},y:{pos.y():.1f}"
        self.label.setText(self.mouse_xy_text)
        self.label.adjustSize()
        label_width = self.label.width()
        label_height = self.label.height()
        self.label.move(int(x - label_width / 2), int(y + label_height))
        self.prevMovePoint = pos
        self.restoreCursor()

        # Drawing mode.
        if self.drawing():
            if self.createMode in ["ai_polygon", "ai_mask", "grounding_sam"]:
                self.line.shape_type = "points"
            elif self.createMode == "polygonSAM":
                self.line.shape_type = "polygon"
            elif self.createMode == "circle_3p":
                self.line.shape_type = "circle"
            else:
                self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            adjoining_pos, adjoining_feature = self._snap_to_adjoining_boundary(pos)
            if not self.current:
                if self.createMode == "polygonSAM":
                    points = np.array([[pos.x(), pos.y()]])
                    labels = np.array([1])
                    self.samPrompt(points, labels)
                self.repaint()
                return

            if not self._allow_points_outside_image and self.outOfPixmap(adjoining_pos):
                pos = self.intersectionPoint(self.current[-1], adjoining_pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(adjoining_pos, self.current[0])
            ):
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            elif self.createMode == "polygon" and adjoining_feature is not None:
                pos = adjoining_pos
            elif self.createMode == "polygon":
                close_target = self._polygon_close_target(adjoining_pos)
                if close_target is not None:
                    pos = close_target
            self._shape_points_for_preview(pos)
            self.repaint()
            self.current.highlightClear()
            return

        # Right-button copy/move drag.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint is not None:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos, ignore_locked=True)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
                for copied in self.selectedShapesCopy:
                    copied.locked = False
                self.repaint()
            return

        # Left-button vertex / shape drag.
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
                if self.hShape is not None and not self.hShape.is_locked():
                    rotate_obb = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
                    self.boundedMoveVertex(pos, rotate_obb=rotate_obb)
                    self.repaint()
                    self.movingShape = True
            elif self.selectedShapes and self.prevPoint is not None:
                unlocked = [s for s in self.selectedShapes if not s.is_locked()]
                if unlocked:
                    self.overrideCursor(CURSOR_MOVE)
                    if self.boundedMoveShapes(unlocked, pos):
                        self.repaint()
                        self.movingShape = True
            return

        # Hover hit-testing.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            index = shape.nearestVertex(pos, self.epsilon / max(self.scale, 1e-6))
            index_edge = shape.nearestEdge(pos, self.epsilon / max(self.scale, 1e-6))
            is_locked = shape.is_locked()
            if index is not None and not is_locked:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                if shape.shape_type == "oriented_rectangle":
                    self.setToolTip(
                        self.tr("Drag corner to resize (Shift+Drag to rotate) '%s'")
                        % shape.label
                    )
                else:
                    self.setToolTip(self.tr("Click & drag to move point"))
                self.label.setText(f"{shape.label},{self.mouse_xy_text}")
                self.label.adjustSize()
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint() and not is_locked:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click to create point"))
                self.label.setText(f"{shape.label},{self.mouse_xy_text}")
                self.label.adjustSize()
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos) or (index is not None and is_locked):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                suffix = " [Locked]" if is_locked else ""
                self.label.setText(f"{shape.label}{suffix},{self.mouse_xy_text}")
                self.label.adjustSize()
                if is_locked:
                    self.setToolTip(
                        self.tr("Shape '%s' is locked (Ctrl+Shift+L to unlock)")
                        % shape.label
                    )
                    self.overrideCursor(CURSOR_DEFAULT)
                else:
                    self.setToolTip(
                        self.tr("Click & drag to move shape '%s'") % shape.label
                    )
                    self.overrideCursor(CURSOR_GRAB)
                self.setStatusTip(self.toolTip())
                self.update()
                break
        else:
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)
        self.edgeSelected.emit(self.hEdge is not None)

    def addPointToEdge(self):
        shape = self.hShape if self.hShape is not None else self.prevhShape
        index = self.hEdge if self.hEdge is not None else self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None or shape.is_locked():
            return False
        if not shape.canAddPoint():
            return False
        if self._allow_shared_polygon_topology:
            self._shared_insert_vertex_on_edge(shape, index, point)
        else:
            self._detach_canvas_polygon_topology(shape)
            shape.insertPoint(index, QtCore.QPointF(point))
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True
        self.update()
        return True

    def removeSelectedPoint(self):
        shape = self.hShape if self.hShape is not None else self.prevhShape
        index = self.hVertex if self.hVertex is not None else self.prevhVertex
        if shape is None or index is None or shape.is_locked():
            return False
        if index < 0 or index >= len(shape.points):
            return False
        if self._allow_shared_polygon_topology:
            if not self._shared_remove_vertex(shape, index):
                return False
        else:
            self._detach_canvas_polygon_topology(shape)
            previous_count = len(shape.points)
            shape.removePoint(index)
            if len(shape.points) == previous_count:
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
        self.movingShape = True
        self.vertexSelected.emit(self.hVertex is not None)
        self.edgeSelected.emit(self.hEdge is not None)
        self.update()
        return True

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------
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
        return build_canvas_context_menu(self, main_window)

    def completeRightDragMove(self, *, copy: bool) -> bool:
        """Finalize right-drag Copy Here / Move Here and sync host label list."""
        if not (self.selectedShapes and self.selectedShapesCopy):
            return False
        copied_shapes = list(self.selectedShapesCopy) if copy else []
        ok = self.endMove(copy=copy)
        if not ok:
            return False
        host = self.window()
        if copy and host is not None and hasattr(host, "addLabel"):
            for shape in copied_shapes:
                try:
                    host.addLabel(shape, rebuild_unique=False)
                except TypeError:
                    host.addLabel(shape)
            rebuild = getattr(host, "_rebuild_unique_label_list", None)
            if callable(rebuild):
                rebuild()
        self.shapeMoved.emit()
        try:
            self.selectionChanged.emit(list(self.selectedShapes))
        except Exception:
            pass
        return True

    def contextMenuEvent(self, event):
        # Right mouse button release already presents the context menu; avoid double popup.
        if event.reason() == QtGui.QContextMenuEvent.Mouse:
            event.accept()
            return
        main_window = self.window()
        menu = self._build_context_menu(main_window)
        menu.exec_(event.globalPos())

    def propagateSelectedShapeFromCanvas(self):
        main_window = self.window()
        if main_window is not None and hasattr(main_window, "propagateSelectedShape"):
            main_window.propagateSelectedShape()
        else:
            logger.warning("Main window does not have propagateSelectedShape method.")

    def switchSelectedShapeLabelsFromCanvas(self):
        main_window = self.window()
        if main_window is not None and hasattr(
            main_window, "switchSelectedShapeLabels"
        ):
            main_window.switchSelectedShapeLabels()

    def mousePressEvent(self, ev):
        pos = self.transformPos(event_pos_f(ev))
        if self._patch_similarity_active and ev.button() == QtCore.Qt.LeftButton:
            if not self.outOfPixmap(pos) and self._patch_similarity_callback:
                try:
                    self._patch_similarity_callback(float(pos.x()), float(pos.y()))
                except Exception as exc:
                    logger.error(f"Patch similarity callback failed: {exc}")
            return
        is_shift_pressed = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
        is_ctrl_pressed = bool(ev.modifiers() & QtCore.Qt.ControlModifier)

        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                snapped_pos, adjoining_feature = self._snap_to_adjoining_boundary(pos)
                if self.current:
                    clamped_pos = self._clamp_drawing_pos(
                        snapped_pos, anchor=self.current[-1]
                    )
                    if self.createMode == "polygon":
                        close_target = self._polygon_close_target(clamped_pos)
                        if close_target is not None:
                            clamped_pos = close_target
                        self.current.addPoint(clamped_pos)
                        if close_target is None and adjoining_feature is not None:
                            self._shared_link_adjoining_point(
                                self.current,
                                len(self.current.points) - 1,
                                clamped_pos,
                                self._adjoining_boundary_source(),
                                adjoining_feature,
                            )
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        if len(self.current.points) == 1:
                            end_pt = (
                                self.line.points[1]
                                if len(self.line.points) >= 2
                                else clamped_pos
                            )
                            self.current.points = [self.current.points[0], end_pt]
                            self.current.point_labels = [1, 1]
                            self.finalise()
                    elif self.createMode == "oriented_rectangle":
                        if len(self.current.points) == 1:
                            self.current.addPoint(clamped_pos)
                            self.line.points = [self.current[0], clamped_pos]
                            self.update()
                        elif len(self.current.points) >= 2:
                            corners = oriented_rect_from_three_points(
                                self.current.points[0],
                                self.current.points[1],
                                clamped_pos,
                            )
                            if corners is not None:
                                self.current.points = [
                                    QtCore.QPointF(x, y) for x, y in corners
                                ]
                                self.current.point_labels = [1] * 4
                                self.finalise()
                    elif self.createMode == "circle_3p":
                        if len(self.current.points) == 1:
                            self.current.addPoint(clamped_pos)
                            self.line.points = [self.current[0], clamped_pos]
                            self.update()
                        elif len(self.current.points) >= 2:
                            fitted = circle_from_three_points(
                                self.current.points[0],
                                self.current.points[1],
                                clamped_pos,
                            )
                            if fitted is not None:
                                (cx, cy), (ex, ey) = fitted
                                self.current.shape_type = "circle"
                                self.current.points = [
                                    QtCore.QPointF(cx, cy),
                                    QtCore.QPointF(ex, ey),
                                ]
                                self.current.point_labels = [1, 1]
                                self.finalise()
                            else:
                                logger.warning(
                                    "3-point circle points are collinear; pick a third non-collinear point."
                                )
                    elif self.createMode == "linestrip":
                        end_pt = (
                            self.line[1] if len(self.line.points) >= 2 else clamped_pos
                        )
                        self.current.addPoint(end_pt)
                        self.line[0] = self.current[-1]
                        if is_ctrl_pressed:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask"]:
                        click_label = 0 if is_shift_pressed else 1
                        self.current.addPoint(clamped_pos, label=click_label)
                        self._ai_polygon_preview_cache = None
                        self.line.points = [self.current.points[-1], clamped_pos]
                        self.line.point_labels = [
                            self.current.point_labels[-1],
                            click_label,
                        ]
                        if is_ctrl_pressed:
                            self.finalise()
                    elif self.createMode == "polygonSAM":
                        target_pt = (
                            self.line[1] if len(self.line.points) >= 2 else clamped_pos
                        )
                        self.current.addPoint(target_pt, True)
                        points = [
                            [point.x(), point.y()] for point in self.current.points
                        ]
                        labels = [int(label) for label in self.current.labels]
                        self.samPrompt(np.array(points), np.array(labels))
                        if is_ctrl_pressed:
                            self.finalise()
                elif self._allow_points_outside_image or not self.outOfPixmap(pos):
                    start_pos = self._clamp_drawing_pos(snapped_pos)
                    if self.createMode != "polygonSAM":
                        if self.createMode in [
                            "ai_polygon",
                            "ai_mask",
                            "grounding_sam",
                        ]:
                            initial_shape_type = "points"
                        elif self.createMode == "circle_3p":
                            initial_shape_type = "linestrip"
                        else:
                            initial_shape_type = self.createMode
                        self.current = Shape(shape_type=initial_shape_type)
                        self.current.addPoint(
                            start_pos, label=0 if is_shift_pressed else 1
                        )
                        self._ai_polygon_preview_cache = None
                        if self.createMode == "point":
                            if is_shift_pressed:
                                set_keypoint_visibility_on_shape_object(
                                    self.current, KeypointVisibility.OCCLUDED
                                )
                            self.finalise()
                        elif (
                            self.createMode
                            in ["ai_polygon", "ai_mask", "grounding_sam"]
                            and is_ctrl_pressed
                        ):
                            self.finalise()
                        else:
                            if self.createMode == "circle":
                                self.current.shape_type = "circle"
                            self.line.points = [start_pos, start_pos]
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
                        self.current.addPoint(start_pos, True)
                        self.line.points = [start_pos, start_pos]
                        self.samPrompt(
                            np.array([[start_pos.x(), start_pos.y()]]),
                            np.array([1]),
                        )
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
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
                elif self.selectedVertex() and is_shift_pressed:
                    self.removeSelectedPoint()

                group_mode = is_ctrl_pressed
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
                self.prevPoint = QtCore.QPointF(pos)
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton:
            if self.drawing() and self.createMode == "polygonSAM":
                if self.current:
                    target_pt = self.line[1] if len(self.line.points) >= 2 else pos
                    self.current.addPoint(target_pt, False)
                    points = [[point.x(), point.y()] for point in self.current.points]
                    labels = [int(label) for label in self.current.labels]
                    self.samPrompt(np.array(points), np.array(labels))
                    if is_ctrl_pressed:
                        self.finalise()
                elif self._allow_points_outside_image or not self.outOfPixmap(pos):
                    self.current = MultipoinstShape()
                    self.current.addPoint(pos, False)
                    self.line.points = [pos, pos]
                    self.setHiding()
                    self.drawingPolygon.emit(True)
                    self.update()
            elif self.editing():
                group_mode = is_ctrl_pressed
                selected_ids = {id(s) for s in (self.selectedShapes or [])}
                if not self.selectedShapes or (
                    self.hShape is not None and id(self.hShape) not in selected_ids
                ):
                    self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                    self.repaint()
                self.prevPoint = QtCore.QPointF(pos)

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.createMode != "polygonSAM":
                main_window = self.window()
                menu = self._build_context_menu(main_window)
                self.restoreCursor()
                menu.exec_(self.mapToGlobal(event_pos_i(ev)))
                if self.selectedShapesCopy:
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

        if self.movingShape:
            moved_target = self.hShape or (
                self.selectedShapes[0] if self.selectedShapes else None
            )
            if moved_target is not None and self._has_shape_moved_since_backup(
                moved_target
            ):
                self._shared_finalize_topology_edit()
                self.storeShapes()
                self.shapeMoved.emit()
            self.movingShape = False

    def endMove(self, copy):
        if not (self.selectedShapes and self.selectedShapesCopy):
            return False
        if len(self.selectedShapesCopy) != len(self.selectedShapes):
            self.selectedShapesCopy = []
            return False
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                shape.locked = False
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
                shape.selected = True
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                if self.selectedShapes[i].is_locked():
                    continue
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self._shared_finalize_topology_edit()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
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
        if self.createMode == "polygon" and len(self.current) > 3:
            self.current.popPoint()
            return
        if self.createMode in ["linestrip", "ai_polygon", "ai_mask", "grounding_sam"]:
            last_point = self.current[-1]
            prev_point = self.current[-2]
            if self.closeEnough(last_point, prev_point):
                self.current.popPoint()
                self._ai_polygon_preview_cache = None

    def mouseDoubleClickEvent(self, ev):
        if self.double_click != "close":
            return
        if not self.drawing() or self.current is None:
            return
        self._trim_double_click_tail_point()
        if self.canCloseShape():
            self.finalise()

    # ------------------------------------------------------------------
    # Selection, movement, deletion & duplication
    # ------------------------------------------------------------------
    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(list(shapes or []))
        self.update()

    def selectAllVisibleShapes(self) -> list[Shape]:
        visible = [s for s in self.shapes if self.isVisible(s)]
        self.selectShapes(visible)
        return visible

    def selectNeighborShape(self, direction: str) -> Shape | None:
        current = self.selectedShapes[0] if self.selectedShapes else None
        neighbor = find_neighbor_shape(
            self.shapes,
            current,
            direction,
            is_visible=self.isVisible,
        )
        if neighbor is None:
            return None
        self.selectShapes([neighbor])
        return neighbor

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the topmost visible shape containing ``point``."""
        if self.selectedVertex():
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

    def calculateOffsets(self, point, shapes=None):
        target_shapes = self.selectedShapes if shapes is None else list(shapes)
        has_pixmap = self.pixmap is not None and not self.pixmap.isNull()
        left = float(self.pixmap.width() - 1) if has_pixmap else float("inf")
        right = 0.0 if has_pixmap else float("-inf")
        top = float(self.pixmap.height() - 1) if has_pixmap else float("inf")
        bottom = 0.0 if has_pixmap else float("-inf")
        found = False
        for s in target_shapes:
            bounds = shape_bounds(s)
            if bounds is None:
                continue
            bx0, by0, bx1, by1 = bounds
            if not found:
                left, top, right, bottom = bx0, by0, bx1, by1
                found = True
            else:
                left = min(left, bx0)
                top = min(top, by0)
                right = max(right, bx1)
                bottom = max(bottom, by1)

        if not found:
            self.offsets = QtCore.QPointF(0.0, 0.0), QtCore.QPointF(0.0, 0.0)
            return

        x1 = left - float(point.x())
        y1 = top - float(point.y())
        x2 = right - float(point.x())
        y2 = bottom - float(point.y())
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos, *, rotate_obb: bool = False):
        index, shape = self.hVertex, self.hShape
        if shape is None or index is None or shape.is_locked():
            return
        try:
            point = shape[index]
        except IndexError as e:
            logger.info(e)
            return
        if not self._allow_points_outside_image and self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        if shape.shape_type == "oriented_rectangle" and len(shape.points) == 4:
            if rotate_obb:
                updated = rotate_oriented_rect_by_vertex(shape.points, index, pos)
            else:
                updated = move_oriented_rect_vertex(shape.points, index, pos)
            if (
                not self._allow_points_outside_image
                and self.pixmap
                and not self.pixmap.isNull()
            ):
                w, h = float(self.pixmap.width()), float(self.pixmap.height())
                if any(
                    not (0.0 <= ux <= w - 1.0 and 0.0 <= uy <= h - 1.0)
                    for ux, uy in updated
                ):
                    return
            shape.points = [QtCore.QPointF(ux, uy) for ux, uy in updated]
            return
        if not self._allow_shared_polygon_topology:
            self._detach_canvas_polygon_topology(shape)
        shape.moveVertexBy(index, pos - point)
        self._sync_shared_vertex(shape, index, shape[index])

    def boundedMoveShapes(self, shapes, pos, *, ignore_locked: bool = False):
        movable = [
            s
            for s in list(shapes or [])
            if s is not None and (ignore_locked or not s.is_locked())
        ]
        if not movable:
            return False
        pos_f = QtCore.QPointF(pos)
        if not self._allow_points_outside_image:
            if self.outOfPixmap(pos_f):
                return False
            o1 = pos_f + self.offsets[0]
            if self.outOfPixmap(o1):
                pos_f = QtCore.QPointF(
                    pos_f.x() - min(0.0, o1.x()),
                    pos_f.y() - min(0.0, o1.y()),
                )
            o2 = pos_f + self.offsets[1]
            if self.outOfPixmap(o2):
                max_x = float(max(0, self.pixmap.width() - 1))
                max_y = float(max(0, self.pixmap.height() - 1))
                pos_f = QtCore.QPointF(
                    pos_f.x() + min(0.0, max_x - o2.x()),
                    pos_f.y() + min(0.0, max_y - o2.y()),
                )
        dp = pos_f - QtCore.QPointF(self.prevPoint)
        if abs(dp.x()) > 1e-9 or abs(dp.y()) > 1e-9:
            if not self._allow_shared_polygon_topology:
                for shape in movable:
                    self._detach_canvas_polygon_topology(shape)
                    shape.moveBy(dp)
            else:
                registry = getattr(self, "_shared_topology_registry", None)
                if isinstance(registry, SharedTopologyRegistry):
                    registry.translate_shapes(movable, dp)
                else:
                    self._shared_move_selected_shapes(movable, dp)
            self.prevPoint = pos_f
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
            remaining_locked = [s for s in self.selectedShapes if s.is_locked()]
            deletable = [s for s in self.selectedShapes if not s.is_locked()]
            if not deletable:
                return []
            deleted_ids = {id(s) for s in deletable}
            if self.hShape is not None and id(self.hShape) in deleted_ids:
                self.unHighlight()
            if self.prevhShape is not None and id(self.prevhShape) in deleted_ids:
                self.prevhShape = self.prevhVertex = self.prevhEdge = None
            for shape in deletable:
                for idx, s in enumerate(list(self.shapes)):
                    if s is shape:
                        del self.shapes[idx]
                        break
                deleted_shapes.append(shape)
            self._shared_finalize_topology_edit()
            self.storeShapes()
            self.selectedShapes = remaining_locked
            self.hShapeIsSelected = False
            try:
                self.selectionChanged.emit(list(remaining_locked))
            except Exception:
                pass
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape is None or shape.is_locked():
            return False
        if self.hShape is shape:
            self.unHighlight()
        if self.prevhShape is shape:
            self.prevhShape = self.prevhVertex = self.prevhEdge = None
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
            self.hShapeIsSelected = False
            try:
                self.selectionChanged.emit([])
            except Exception:
                pass
        self.update()
        return True

    def duplicateSelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            for copied in self.selectedShapesCopy:
                copied.locked = False
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        valid = [s for s in list(shapes or []) if getattr(s, "points", None)]
        if not valid:
            return
        point = QtCore.QPointF(valid[0][0])
        offset = QtCore.QPointF(2.0, 2.0)
        self.calculateOffsets(point, shapes=valid)
        self.prevPoint = point
        if not self.boundedMoveShapes(valid, point - offset, ignore_locked=True):
            self.boundedMoveShapes(valid, point + offset, ignore_locked=True)

    # ------------------------------------------------------------------
    # Live preview & AI prompt preview helpers
    # ------------------------------------------------------------------
    def _paint_ai_polygon_preview(self, painter):
        if self.current is None:
            return
        drawing_shape = self.current.copy()
        try:
            if not self._ensure_ai_model_initialized():
                logger.error(
                    "AI polygon model is not initialized; skipping prediction."
                )
                return
            prompt_points = [[point.x(), point.y()] for point in drawing_shape.points]
            point_labels = list(drawing_shape.point_labels or [])
            try:
                pixmap_key = (
                    int(self.pixmap.cacheKey())
                    if self.pixmap is not None and not self.pixmap.isNull()
                    else None
                )
            except Exception:
                pixmap_key = None
            cache_key = (
                tuple(
                    (round(float(x), 3), round(float(y), 3)) for x, y in prompt_points
                ),
                tuple(int(v) for v in point_labels),
                id(getattr(self, "_ai_model", None)),
                pixmap_key,
            )
            cache = getattr(self, "_ai_polygon_preview_cache", None)
            if cache is not None and cache[0] == cache_key:
                normalized_points = cache[1]
            else:
                normalized_points = self._predict_ai_polygon_points(
                    prompt_points=prompt_points,
                    point_labels=point_labels,
                )
                self._ai_polygon_preview_cache = (cache_key, list(normalized_points))
            if len(normalized_points) > 2:
                drawing_shape.setShapeRefined(
                    shape_type="polygon",
                    points=normalized_points,
                    point_labels=[1] * len(normalized_points),
                )
                drawing_shape.fill = self.fillDrawing()
                drawing_shape.paint(painter)
        except Exception as e:
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
            logger.warning(f"AI mask prediction failed: {e}")

    def _predict_ai_polygon_points(
        self, *, prompt_points: list[list[float]], point_labels: list[int]
    ) -> list[QtCore.QPointF]:
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
        self._ai_polygon_preview_cache = None
        if clear_sam_mask:
            self.sam_mask = MaskShape()
        self._clear_preview_line()
        self._clear_adjoining_source()
        self._clearSharedBoundaryReshape()
        self.setHiding(False)
        self.drawingPolygon.emit(False)

    def _clear_ai_prompt_state(self) -> None:
        self.cancelCurrentDrawing(clear_sam_mask=True)
        try:
            if hasattr(self, "sam_image_scale"):
                delattr(self, "sam_image_scale")
        except Exception:
            pass

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
                    (start_point.x() + end_point.x()) / 2.0,
                    (start_point.y() + end_point.y()) / 2.0,
                )
                font_size = int(max(8, min(20, distance / 10)))
                font = QtGui.QFont()
                font.setPointSize(font_size)
                painter.setFont(font)
                painter.drawText(mid_point, f"{distance:.1f}px")
        except Exception as e:
            logger.error(f"An error occurred in paintEvent: {e}")

    # ------------------------------------------------------------------
    # Paint & coordinate transforms
    # ------------------------------------------------------------------
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

            has_pixmap = self.pixmap is not None and not self.pixmap.isNull()
            if has_pixmap:
                p.drawPixmap(0, 0, self.pixmap)

            self.sam_mask.paint(p)
            self._draw_raster_overlays(p)
            self._draw_behavior_text_banner(p)

            # Draw crosshair in image coordinates.
            if (
                self.createMode != "grounding_sam"
                and self._crosshair.get(self._createMode, False)
                and self.drawing()
            ):
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1, QtCore.Qt.DotLine))
                max_x = (
                    float(self.pixmap.width() - 1)
                    if has_pixmap
                    else float(self.width()) / max(self.scale, 1e-6)
                )
                max_y = (
                    float(self.pixmap.height() - 1)
                    if has_pixmap
                    else float(self.height()) / max(self.scale, 1e-6)
                )
                p.drawLine(
                    QtCore.QPointF(0.0, float(self.prevMovePoint.y())),
                    QtCore.QPointF(max_x, float(self.prevMovePoint.y())),
                )
                p.drawLine(
                    QtCore.QPointF(float(self.prevMovePoint.x()), 0.0),
                    QtCore.QPointF(float(self.prevMovePoint.x()), max_y),
                )

            Shape.scale = self.scale
            Shape.show_label_names = self._show_label_names
            MultipoinstShape.scale = self.scale
            image_width = self.pixmap.width() if has_pixmap else None
            image_height = self.pixmap.height() if has_pixmap else None

            self._draw_pose_edges(p)
            self._draw_explicit_landmark_pairs(p)

            for shape in self.shapes:
                if (shape.selected or not self._hideBackround) and self.isVisible(
                    shape
                ):
                    shape.fill = shape.selected or shape is self.hShape
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
                        logger.debug("Shape paint system error: %s", e)
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
        return QtCore.QPointF(point) / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        aw, ah = area.width(), area.height()
        if self.pixmap is None or self.pixmap.isNull():
            return QtCore.QPointF(0, 0)
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        if self.pixmap is None or self.pixmap.isNull():
            return True
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    # ------------------------------------------------------------------
    # Shape finalisation & boundary clamping
    # ------------------------------------------------------------------
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
            self._clear_adjoining_source()
            self._clear_preview_line()
            self.setHiding(False)
            self.newShape.emit()
            self.update()

            rect_label = rect_shape.label
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
                async_starter = getattr(
                    self.window(), "_start_exemplar_countgd_async", None
                )
                if callable(async_starter):
                    async_starter(rect_label, [rect_shape])
                else:
                    self.predictAiRectangle(
                        rect_label,
                        [rect_shape],
                        is_polygon_output=True,
                        use_countgd=True,
                    )
            return

        if self.createMode == "ai_polygon":
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
            if self.current.shape_type != "points":
                return
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
        elif getattr(self.current, "shape_type", None) == "rectangle":
            self.current.close()
        if self.createMode != "polygonSAM":
            self._separate_canvas_polygon_vertices(self.current)
        if self.createMode == "polygonSAM":
            self.shapes.append(self.sam_mask)
        else:
            self.shapes.append(self.current)
        if not getattr(self, "_batch_finalising_prompt_shapes", False):
            self._shared_finalize_topology_edit()
            self.storeShapes()
        self.sam_mask = MaskShape()
        self.current = None
        self._ai_polygon_preview_cache = None
        self._clear_adjoining_source()
        self._clear_preview_line()
        self.setHiding(False)
        self.newShape.emit()
        if not getattr(self, "_batch_finalising_prompt_shapes", False):
            self.update()

    def closeEnough(self, p1, p2):
        return utils.distance(p1 - p2) < (self.epsilon / max(self.scale, 1e-6))

    def _polygon_close_target(self, pos):
        if self.current is None or self.createMode != "polygon":
            return None
        if len(getattr(self.current, "points", []) or []) < 2:
            return None
        if pos is None:
            return None
        return self.current[0] if self.closeEnough(pos, self.current[0]) else None

    def intersectionPoint(self, p1, p2):
        if self.pixmap is None or self.pixmap.isNull():
            return QtCore.QPointF(p2)
        size = self.pixmap.size()
        x, y = intersect_segment_with_bounds(
            p1, p2, width=float(size.width()), height=float(size.height())
        )
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # ------------------------------------------------------------------
    # Sizing, wheel, keyboard navigation
    # ------------------------------------------------------------------
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap is not None and not self.pixmap.isNull():
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        mods = ev.modifiers()
        delta = ev.angleDelta()
        if mods & QtCore.Qt.ControlModifier:
            self.zoomRequest.emit(delta.y(), event_pos_i(ev))
        else:
            self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
            self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        ev.accept()

    def moveByKeyboard(self, offset):
        unlocked = [s for s in (self.selectedShapes or []) if not s.is_locked()]
        if not unlocked:
            return
        bounds = shape_bounds(unlocked[0])
        if bounds is not None:
            anchor = QtCore.QPointF(
                0.5 * (bounds[0] + bounds[2]), 0.5 * (bounds[1] + bounds[3])
            )
            self.prevPoint = anchor
            self.calculateOffsets(anchor, shapes=unlocked)
        if self.boundedMoveShapes(unlocked, self.prevPoint + offset):
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        handled = False
        ctrl_pressed = bool(modifiers & QtCore.Qt.ControlModifier)

        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.cancelCurrentDrawing(clear_sam_mask=True)
                self.update()
                handled = True
            elif (
                key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter)
                and self.canCloseShape()
            ):
                self.finalise()
                handled = True
            elif (
                key in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete) and self.current
            ):
                self.undoLastPoint()
                handled = True
            elif modifiers & QtCore.Qt.AltModifier:
                self.snapping = False
                handled = True
        elif self.editing():
            if ctrl_pressed and key == QtCore.Qt.Key_A:
                self.selectAllVisibleShapes()
                handled = True
            elif ctrl_pressed and key in (
                QtCore.Qt.Key_Up,
                QtCore.Qt.Key_Down,
                QtCore.Qt.Key_Left,
                QtCore.Qt.Key_Right,
            ):
                dir_map = {
                    QtCore.Qt.Key_Up: "up",
                    QtCore.Qt.Key_Down: "down",
                    QtCore.Qt.Key_Left: "left",
                    QtCore.Qt.Key_Right: "right",
                }
                self.selectNeighborShape(dir_map[key])
                handled = True
            elif key in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
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
        if all(s.is_locked() for s in self.selectedShapes):
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
            if not (modifiers & QtCore.Qt.AltModifier):
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                if self._has_shape_moved_since_backup(self.selectedShapes[0]):
                    self._shared_finalize_topology_edit()
                    self.storeShapes()
                    self.shapeMoved.emit()
                self.movingShape = False

    # ------------------------------------------------------------------
    # Label assignment, line undo, pixmap & shape loading, lifecycle
    # ------------------------------------------------------------------
    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        polygons = None
        if isinstance(self.shapes[-1], MaskShape):
            mask_shape = self.shapes.pop()
            polygons = mask_shape.toPolygons(self.sam_config["approxpoly_epsilon"])
            self.shapes.extend(polygons)
        if self.shapesBackups:
            self.shapesBackups.pop()
        self.storeShapes()

        if isinstance(polygons, list):
            return polygons
        return self.shapes[-1:]

    def undoLastLine(self):
        if not self.shapes:
            return
        self._clearSharedBoundaryReshape()
        self.current = self.shapes.pop()
        if self.createMode != "polygonSAM":
            self.current.setOpen()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode in ["oriented_rectangle", "circle_3p"]:
            self.current.points = self.current.points[:2]
        elif self.createMode in ["point", "polygonSAM", "ai_polygon", "ai_mask"]:
            self.current = None
        self.drawingPolygon.emit(self.current is not None)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self._clearSharedBoundaryReshape()
        self.current.popPoint()
        self._ai_polygon_preview_cache = None
        if len(self.current) > 0:
            if len(self.line.points) >= 1:
                self.line[0] = self.current[-1]
            else:
                self.line.points = [self.current[-1], self.current[-1]]
        else:
            self.current = None
            self._clear_adjoining_source()
            self._clear_preview_line()
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        previous_signature = self._ai_model_image_signature_value()
        try:
            previous_pixmap_key = (
                int(self.pixmap.cacheKey())
                if self.pixmap is not None and not self.pixmap.isNull()
                else None
            )
        except Exception:
            previous_pixmap_key = None
        self.pixmap = pixmap if pixmap is not None else QtGui.QPixmap()
        try:
            current_pixmap_key = (
                int(self.pixmap.cacheKey()) if not self.pixmap.isNull() else None
            )
        except Exception:
            current_pixmap_key = None
        self._invalidate_ai_model_image_signature_cache()
        self._ai_polygon_preview_cache = None
        self._ai_model_image_dirty = previous_pixmap_key != current_pixmap_key
        current_signature = self._ai_model_image_signature_value()
        if previous_signature != current_signature and self.is_text_prompt_running():
            self.cancel_async_text_prompt(wait_ms=0, reason="frame_changed")
        if (
            previous_signature is not None
            and current_signature is not None
            and previous_signature != current_signature
        ):
            self._clear_ai_prompt_state()
        if self._ai_model is not None and not self.pixmap.isNull():
            if self._inflight_prompt_workers:
                self._ai_model_image_dirty = True
            else:
                self._sync_ai_model_image(force=bool(self._ai_model_image_dirty))
        if clear_shapes:
            self.shapes = []
            # Seed empty baseline so the very first created shape is immediately undoable.
            self.shapesBackups = [[]]
            self.selectedShapes = []
            self.selectedShapesCopy = []
            self.visible = {}
            self.sam_mask = MaskShape()
            self.current = None
            self._clear_preview_line()

        if (
            self.createMode == "polygonSAM"
            and not self.pixmap.isNull()
            and self.sam_predictor
        ):
            self.samEmbedding()
        self.update()

    def closeEvent(self, event):
        self._canvas_closing = True
        self._pending_close_model_release = True
        self.cancel_async_text_prompt(
            wait_ms=250,
            reason="closing",
            notify_callback=False,
        )
        lock_acquired = self._prompt_inference_lock.acquire(timeout=0.5)
        try:
            if lock_acquired:
                self._release_all_prompt_models()
                if self._has_other_running_prompt_workers(-1):
                    self._pending_close_model_release = True
            else:
                logger.warning(
                    "Canvas closing while prompt inference lock is still busy; deferring model release."
                )
        finally:
            if lock_acquired:
                self._prompt_inference_lock.release()
        super().closeEvent(event)

    def loadShapes(self, shapes, replace=True):
        self._clearSharedBoundaryReshape()
        if replace:
            self.shapes = list(shapes)
            self.visible = {}
        else:
            self.shapes.extend(shapes)
        self._shared_finalize_topology_edit()
        self.storeShapes()
        self.current = None
        self._ai_polygon_preview_cache = None
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
        if self.is_text_prompt_running():
            set_widget_busy_cursor(self, True)
            return
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
        if self.is_text_prompt_running():
            self.cancel_async_text_prompt(wait_ms=0, reason="reset")
        self.restoreCursor()
        self.pixmap = QtGui.QPixmap()
        self.shapesBackups = [[]]
        self.shapes = []
        self.selectedShapes = []
        self.selectedShapesCopy = []
        self.visible = {}
        self.current = None
        self._ai_polygon_preview_cache = None
        self._clear_preview_line()
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.hShapeIsSelected = False
        self._clear_adjoining_source()
        self._clearSharedBoundaryReshape()
        self._patch_similarity_pixmap = None
        self._pca_map_pixmap = None
        self._depth_preview_pixmap = None
        self._flow_preview_pixmap = None
        self.sam_mask = MaskShape()
        self.update()
