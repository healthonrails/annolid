from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import numpy as np
import pytest
from PIL import Image
from qtpy import QtCore, QtGui, QtWidgets

from annolid.annotation import labelme2coco
from annolid.annotation.shape_io import (
    deserialize_shape_record,
    normalize_group_id,
    serialize_shape,
)
from annolid.gui.label_file import LabelFile
from annolid.gui.shape import Shape
from annolid.gui.shape_geometry import (
    circle_from_three_points,
    find_neighbor_shape,
    intersect_segment_with_bounds,
    move_oriented_rect_vertex,
    oriented_rect_from_three_points,
    rotate_oriented_rect_by_vertex,
)
from annolid.gui.shape_merge import merge_shapes
from annolid.gui.shared_vertices import SharedTopologyRegistry
from annolid.gui.widgets.ai_polygon_helpers import simplify_ai_polygon_points
from annolid.gui.widgets.canvas import Canvas
from annolid.gui.widgets.label_dialog import AnnolidLabelDialog
from annolid.utils.annotation_compat import shape_to_mask, shapes_to_label
from annolid.utils.annotation_store import AnnotationStore


def _ensure_qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_rect(label="mouse", x1=10.0, y1=10.0, x2=30.0, y2=30.0, **kwargs) -> Shape:
    shape = Shape(label=label, shape_type="rectangle", **kwargs)
    shape.points = [QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)]
    shape.point_labels = [1, 1]
    shape.close()
    return shape


def _make_mask_shape(label="mouse", x1=10, y1=10, arr=None, **kwargs) -> Shape:
    if arr is None:
        arr = np.zeros((10, 10), dtype=bool)
        arr[2:8, 2:8] = True
    h, w = arr.shape[:2]
    shape = Shape(label=label, shape_type="mask", mask=arr, **kwargs)
    shape.points = [QtCore.QPointF(x1, y1), QtCore.QPointF(x1 + w, y1 + h)]
    shape.point_labels = [1, 1]
    shape.close()
    return shape


# ---------------------------------------------------------------------------
# Pure geometry tests
# ---------------------------------------------------------------------------


def test_oriented_rectangle_geometry_build_resize_and_rotate():
    # Degenerate zero-length base edge returns None.
    assert oriented_rect_from_three_points((5, 5), (5, 5), (10, 10)) is None

    corners = oriented_rect_from_three_points((0, 0), (10, 0), (10, 6))
    assert corners is not None and len(corners) == 4
    # Edges should be orthogonal (dot product ~ 0) with lengths 10 and 6.
    v01 = (corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
    v12 = (corners[2][0] - corners[1][0], corners[2][1] - corners[1][1])
    assert abs(v01[0] * v12[0] + v01[1] * v12[1]) < 1e-6
    assert math.hypot(*v01) == pytest.approx(10.0, abs=1e-6)
    assert math.hypot(*v12) == pytest.approx(6.0, abs=1e-6)

    # Resizing corner 2 preserves 90-degree angles and keeps opposite corner 0 fixed.
    resized = move_oriented_rect_vertex(corners, 2, (20.0, 12.0))
    assert resized[0] == pytest.approx(corners[0], abs=1e-6)
    assert resized[2] == pytest.approx((20.0, 12.0), abs=1e-6)
    e1 = (resized[1][0] - resized[0][0], resized[1][1] - resized[0][1])
    e2 = (resized[3][0] - resized[0][0], resized[3][1] - resized[0][1])
    assert abs(e1[0] * e2[0] + e1[1] * e2[1]) < 1e-6

    # Rotating around center preserves side lengths and center.
    cx_before = sum(p[0] for p in corners) / 4.0
    cy_before = sum(p[1] for p in corners) / 4.0
    rotated = rotate_oriented_rect_by_vertex(corners, 1, (cx_before, cy_before + 10.0))
    cx_after = sum(p[0] for p in rotated) / 4.0
    cy_after = sum(p[1] for p in rotated) / 4.0
    assert cx_after == pytest.approx(cx_before, abs=1e-5)
    assert cy_after == pytest.approx(cy_before, abs=1e-5)
    assert math.hypot(
        rotated[1][0] - rotated[0][0], rotated[1][1] - rotated[0][1]
    ) == pytest.approx(10.0, abs=1e-5)


def test_circle_from_three_points():
    # Collinear points return None.
    assert circle_from_three_points((0, 0), (5, 5), (10, 10)) is None

    # Circle centered at (10, 20) with radius 5.
    res = circle_from_three_points((15, 20), (10, 25), (5, 20))
    assert res is not None
    center, edge = res
    assert center == pytest.approx((10.0, 20.0), abs=1e-5)
    assert math.hypot(edge[0] - center[0], edge[1] - center[1]) == pytest.approx(
        5.0, abs=1e-5
    )


def test_intersect_segment_with_bounds_edge_cases():
    # Normal exit to right boundary of 100x80 image (max x=99, max y=79).
    x, y = intersect_segment_with_bounds((50, 40), (150, 40), 100, 80)
    assert x == pytest.approx(99.0)
    assert y == pytest.approx(40.0)

    # Degenerate 1x1 image and coincident endpoints must never raise ValueError.
    assert intersect_segment_with_bounds((0, 0), (50, -20), 1, 1) == (0.0, 0.0)
    assert intersect_segment_with_bounds((10, 10), (10, 10), 50, 50) == (10.0, 10.0)


def test_find_neighbor_shape_skips_hidden_and_follows_cardinal_directions():
    center = _make_rect("center", 40, 40, 50, 50)
    right_hidden = _make_rect("right_hidden", 60, 40, 70, 50, visible=False)
    right_vis = _make_rect("right_vis", 80, 40, 90, 50)
    left_vis = _make_rect("left_vis", 10, 40, 20, 50)
    up_vis = _make_rect("up_vis", 40, 10, 50, 20)
    down_vis = _make_rect("down_vis", 40, 80, 50, 90)
    shapes = [center, right_hidden, right_vis, left_vis, up_vis, down_vis]

    assert find_neighbor_shape(shapes, center, "right") is right_vis
    assert find_neighbor_shape(shapes, center, "left") is left_vis
    assert find_neighbor_shape(shapes, center, "up") is up_vis
    assert find_neighbor_shape(shapes, center, "down") is down_vis


# ---------------------------------------------------------------------------
# Shape, mask, merge, AI polygon, shared topology regression tests
# ---------------------------------------------------------------------------


def test_mask_shape_contains_point_bounds_and_safe_deepcopy():
    _ensure_qapp()
    arr = np.zeros((10, 10), dtype=bool)
    arr[0:5, 0:5] = True  # foreground touches top-left edge (0, 0)
    mask_shape = _make_mask_shape("cell", x1=20, y1=30, arr=arr)

    # Inside foreground
    assert mask_shape.containsPoint(QtCore.QPointF(22, 32))
    # Inside bbox but background pixel
    assert not mask_shape.containsPoint(QtCore.QPointF(28, 38))
    # Outside bbox (previously buggy due to np.clip clamping negative offsets to 0!)
    assert not mask_shape.containsPoint(QtCore.QPointF(5, 5))
    assert not mask_shape.containsPoint(QtCore.QPointF(100, 100))

    # Paint onto a QImage to populate _mask_paint_cache, then deepcopy must succeed.
    img = QtGui.QImage(64, 64, QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(img)
    try:
        mask_shape.paint(painter)
    finally:
        painter.end()
    assert mask_shape._mask_paint_cache is not None

    cloned = copy.deepcopy(mask_shape)
    assert cloned is not mask_shape
    assert cloned.label == "cell"
    assert cloned.mask is not None and cloned.mask.shape == (10, 10)
    assert cloned._mask_paint_cache is None


def test_adjoining_polygon_seed_wraps_edge_zero():
    poly = Shape(label="region", shape_type="polygon")
    for x, y in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        poly.addPoint(QtCore.QPointF(x, y))
    poly.shared_vertex_ids = ["v0", "v1", "v2", "v3"]
    poly.close()

    seed = poly.adjoining_polygon_seed(edge_index=0)
    assert seed is not None
    assert seed.shared_vertex_ids == ["v3", "v0"]


def test_merge_shapes_masks_oriented_rectangles_and_locked_rejection():
    # 1. Merge 2 mask fragments into one tight-cropped mask.
    m1 = np.zeros((6, 6), dtype=bool)
    m1[1:5, 1:5] = True
    m2 = np.zeros((6, 6), dtype=bool)
    m2[1:5, 1:5] = True
    s1 = _make_mask_shape("organ", x1=10, y1=10, arr=m1)
    s2 = _make_mask_shape("organ", x1=13, y1=12, arr=m2)

    merged_mask = merge_shapes([s1, s2])
    assert merged_mask.shape_type == "mask"
    assert merged_mask.mask is not None
    # Bounding box spans x in [11, 18), y in [11, 17) -> shape (6, 7)
    assert merged_mask.points[0] == QtCore.QPointF(11.0, 11.0)
    assert merged_mask.points[1] == QtCore.QPointF(18.0, 17.0)
    assert merged_mask.mask.shape == (6, 7)

    # 2. Merge oriented_rectangle + polygon.
    obb = Shape(label="mouse", shape_type="oriented_rectangle")
    for x, y in oriented_rect_from_three_points((0, 0), (10, 0), (10, 10)):
        obb.addPoint(QtCore.QPointF(x, y))
    obb.close()
    rect = _make_rect("mouse", 5, 0, 15, 10)
    merged_poly = merge_shapes([obb, rect])
    assert merged_poly.shape_type == "polygon"

    # 3. Reject locked shapes and mixed mask+polygon.
    s1.locked = True
    with pytest.raises(ValueError, match="locked"):
        merge_shapes([s1, s2])
    s1.locked = False
    with pytest.raises(ValueError):
        merge_shapes([s1, rect])


def test_simplify_ai_polygon_points_accepts_qpointf_list():
    pts = [
        QtCore.QPointF(0.0, 0.0),
        QtCore.QPointF(5.0, 0.05),
        QtCore.QPointF(10.0, 0.0),
        QtCore.QPointF(10.0, 10.0),
        QtCore.QPointF(0.0, 10.0),
    ]
    simplified = simplify_ai_polygon_points(pts)
    assert len(simplified) >= 4
    assert all(isinstance(p, QtCore.QPointF) for p in simplified)


def test_shared_topology_translate_shapes_moves_non_polygon_shapes():
    rect = _make_rect("box", 10, 20, 30, 40)
    circle = Shape(label="ball", shape_type="circle")
    circle.points = [QtCore.QPointF(50, 50), QtCore.QPointF(60, 50)]
    reg = SharedTopologyRegistry.from_shapes([rect, circle])
    assert reg.translate_shapes([rect, circle], QtCore.QPointF(5.0, -3.0))
    assert rect.points[0] == QtCore.QPointF(15.0, 17.0)
    assert rect.points[1] == QtCore.QPointF(35.0, 37.0)
    assert circle.points[0] == QtCore.QPointF(55.0, 47.0)


# ---------------------------------------------------------------------------
# Serialization, LabelFile, AnnotationStore, rasterization & COCO export
# ---------------------------------------------------------------------------


def test_shape_io_and_label_file_roundtrip_and_validation(tmp_path: Path):
    assert normalize_group_id(True) is None
    assert normalize_group_id(False) is None
    assert normalize_group_id("42") == 42
    assert normalize_group_id(7) == 7

    obb = Shape(
        label="fly",
        shape_type="oriented_rectangle",
        group_id=3,
        locked=True,
        flags={"occluded": False},
    )
    for x, y in oriented_rect_from_three_points((10, 10), (30, 10), (30, 20)):
        obb.addPoint(QtCore.QPointF(x, y))
    obb.shared_vertex_ids = ["sv1", "", "", ""]
    obb.shared_edge_ids = ["se1", "", "", ""]
    obb.close()

    # Non-mask shape with stale mask attribute must NOT emit a mask payload.
    obb.mask = np.ones((4, 4), dtype=bool)
    dumped = serialize_shape(obb, include_manual_tag=True, include_null_mask=True)
    assert dumped["locked"] is True
    assert dumped["mask"] is None
    assert dumped["shared_vertex_ids"] == ["sv1", "", "", ""]
    assert dumped["shared_edge_ids"] == ["se1", "", "", ""]
    assert dumped["shape_type"] == "oriented_rectangle"
    # Input shape was never mutated.
    assert obb.shape_type == "oriented_rectangle"

    # Malformed JSON record: bool group_id + null flags + missing image dims.
    img_path = tmp_path / "frame.png"
    Image.new("RGB", (64, 48), color=(0, 0, 0)).save(img_path)
    json_path = tmp_path / "frame.json"
    bad_record = deserialize_shape_record(
        {
            "label": "fly",
            "points": [[10, 10], [30, 10], [30, 20], [10, 20]],
            "group_id": True,
            "shape_type": "oriented_rectangle",
            "flags": None,
            "locked": True,
        }
    )
    assert bad_record["group_id"] is None
    assert bad_record["flags"] == {}
    assert bad_record["locked"] is True

    lf = LabelFile()
    lf.save(
        filename=str(json_path),
        shapes=[dumped],
        imagePath=img_path.name,
        imageHeight=48,
        imageWidth=64,
        flags=None,
    )
    # Remove imageHeight/imageWidth from disk JSON to verify LabelFile fills them back.
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw["imageHeight"] = None
    raw["imageWidth"] = None
    raw["flags"] = None
    json_path.write_text(json.dumps(raw), encoding="utf-8")

    reloaded = LabelFile(filename=str(json_path))
    assert reloaded.imageHeight == 48
    assert reloaded.imageWidth == 64
    assert reloaded.flags == {}
    assert reloaded.shapes[0]["locked"] is True
    assert reloaded.shapes[0]["shape_type"] == "oriented_rectangle"
    assert reloaded.shapes[0]["shared_vertex_ids"] == ["sv1", "", "", ""]


def test_annotation_store_get_frames_fast_returns_latest_appended_record(
    tmp_path: Path,
):
    store = AnnotationStore(tmp_path / "annotations.jsonl")
    store.append_frame({"frame": 5, "shapes": [{"label": "old"}]})
    store.append_frame({"frame": 6, "shapes": [{"label": "other"}]})
    store.append_frame({"frame": 5, "shapes": [{"label": "updated"}]})

    frames = store.get_frames_fast({5})
    assert 5 in frames
    assert frames[5]["shapes"][0]["label"] == "updated"


def test_rasterization_and_coco_export_for_mask_and_oriented_rectangle(tmp_path: Path):
    arr = np.zeros((10, 10), dtype=bool)
    arr[2:8, 2:8] = True
    mask_shape = _make_mask_shape("cell", x1=10, y1=10, arr=arr)
    mask_dict = serialize_shape(mask_shape)

    obb_pts = oriented_rect_from_three_points((30, 10), (45, 10), (45, 25))
    obb_dict = {
        "label": "mouse",
        "shape_type": "oriented_rectangle",
        "points": obb_pts,
        "flags": {},
    }

    m_mask = shape_to_mask(
        (60, 60), mask_dict["points"], shape_type="mask", mask=mask_dict["mask"]
    )
    assert int(m_mask.sum()) == 36
    assert m_mask[14, 14]
    assert not m_mask[0, 0]

    lbl, _ = shapes_to_label(
        (60, 60),
        [mask_dict, obb_dict],
        {"_background_": 0, "cell": 1, "mouse": 2},
    )
    assert int((lbl == 1).sum()) == 36
    assert int((lbl == 2).sum()) > 150

    # End-to-end labelme2coco export with oriented_rectangle + mask.
    pytest.importorskip("pycocotools.mask")
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    for stem in ("img_0", "img_1"):
        img_path = in_dir / f"{stem}.png"
        Image.new("RGB", (60, 60), color=(10, 20, 30)).save(img_path)
        (in_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "version": "5.0.0",
                    "flags": {},
                    "shapes": [mask_dict, obb_dict],
                    "imagePath": img_path.name,
                    "imageData": None,
                    "imageHeight": 60,
                    "imageWidth": 60,
                }
            ),
            encoding="utf-8",
        )
    list(
        labelme2coco.convert(
            str(in_dir),
            str(out_dir),
            labels_file=None,
            train_valid_split=0.5,
        )
    )
    coco = json.loads(
        (out_dir / "train" / "annotations.json").read_text(encoding="utf-8")
    )
    assert len(coco["annotations"]) == 2
    assert all(ann["area"] > 0 for ann in coco["annotations"])


# ---------------------------------------------------------------------------
# Canvas & GUI workflow tests
# ---------------------------------------------------------------------------


def _build_test_window():
    _ensure_qapp()
    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    img = QtGui.QImage(120, 100, QtGui.QImage.Format_RGB32)
    img.fill(QtGui.QColor("black"))
    w.image_to_canvas(img, "frame_0.png", 0)
    return w


def test_first_shape_undo_on_empty_image(monkeypatch):
    w = _build_test_window()
    try:
        assert len(w.canvas.shapes) == 0
        assert not w.canvas.isShapeRestorable

        # Draw a rectangle via 2 clicks in create mode.
        monkeypatch.setattr(
            w.labelDialog,
            "popUp",
            lambda *args, **kwargs: ("mouse", {}, 1, "first"),
        )
        w.toggleDrawMode(False, createMode="rectangle")
        w.canvas.current = Shape(shape_type="rectangle")
        w.canvas.current.addPoint(QtCore.QPointF(10, 10))
        w.canvas.current.addPoint(QtCore.QPointF(40, 40))
        w.canvas.finalise()

        assert len(w.canvas.shapes) == 1
        assert w.canvas.shapes[0].label == "mouse"
        assert w.canvas.shapes[0].group_id == 1
        assert w.canvas.shapes[0].description == "first"
        assert w.canvas.isShapeRestorable
        assert w.actions.undo.isEnabled()

        # Immediate Ctrl+Z restores the empty canvas!
        w.undoShapeEdit()
        assert len(w.canvas.shapes) == 0
        assert w.labelList.count() == 0
        assert not w.canvas.isShapeRestorable
    finally:
        w.setClean()
        w.close()


def test_shape_locking_blocks_edit_move_delete_and_merge(monkeypatch):
    w = _build_test_window()
    try:
        a = _make_rect("mouse", 10, 10, 30, 30)
        b = _make_rect("mouse", 20, 10, 40, 30)
        w.loadShapes([a, b])
        w.canvas.setEditing(True)
        w.canvas.selectShapes([a])

        # Lock shape `a`
        assert w.toggleLockSelectedShapes() is True
        assert a.is_locked()
        assert "[Locked]" in w.labelList.item(0).text()
        assert not w.actions.deleteShapes.isEnabled()
        assert not w.actions.editLabel.isEnabled()

        # Move is blocked
        orig_pts = [(p.x(), p.y()) for p in a.points]
        w.canvas.moveByKeyboard(QtCore.QPointF(5.0, 5.0))
        assert [(p.x(), p.y()) for p in a.points] == orig_pts

        # Delete is blocked
        w.deleteSelectedShapes()
        assert len(w.canvas.shapes) == 2

        # editLabel is blocked (popUp should never be invoked)
        popup_called = []
        monkeypatch.setattr(
            w.labelDialog,
            "popUp",
            lambda *args, **kwargs: (
                popup_called.append(True) or ("renamed", {}, None, "")
            ),
        )
        w.editLabel(w.labelList.item(0))
        assert not popup_called
        assert a.label == "mouse"

        # Merge with a locked shape is disallowed
        w.canvas.selectShapes([a, b])
        assert not w.canMergeSelectedShapes()

        # Unlock restores full editing
        w.canvas.selectShapes([a])
        assert w.toggleLockSelectedShapes() is False
        assert not a.is_locked()
        assert "[Locked]" not in w.labelList.item(0).text()
        assert w.actions.deleteShapes.isEnabled()
        assert w.actions.editLabel.isEnabled()
    finally:
        w.setClean()
        w.close()


def test_copy_paste_within_and_across_frames():
    w = _build_test_window()
    try:
        a = _make_rect("mouse", 10, 10, 30, 30, locked=True)
        w.loadShapes([a])
        w.canvas.setEditing(True)
        w.canvas.selectShapes([a])

        copied = w.copySelectedShapes()
        assert len(copied) == 1
        assert not copied[0].locked
        assert w.actions.pasteShapes.isEnabled()

        # Paste in same frame shifts slightly so it doesn't land identically on top.
        pasted_same = w.pasteSelectedShapes()
        assert len(pasted_same) == 1
        assert len(w.canvas.shapes) == 2
        assert pasted_same[0].points[0] != a.points[0]

        # Switch to a fresh frame and paste -> exact original coordinates preserved.
        img2 = QtGui.QImage(120, 100, QtGui.QImage.Format_RGB32)
        img2.fill(QtGui.QColor("black"))
        w.image_to_canvas(img2, "frame_1.png", 1)
        assert len(w.canvas.shapes) == 0
        pasted_next = w.pasteSelectedShapes()
        assert len(pasted_next) == 1
        assert pasted_next[0].points[0] == QtCore.QPointF(10, 10)
        assert pasted_next[0].points[1] == QtCore.QPointF(30, 30)
    finally:
        w.setClean()
        w.close()


def test_canvas_oriented_rectangle_and_3point_circle_creation():
    _ensure_qapp()
    canvas = Canvas()
    pm = QtGui.QPixmap(120, 100)
    pm.fill(QtGui.QColor("black"))
    canvas.loadPixmap(pm)
    canvas.resize(120, 100)

    def _left_click(x, y):
        widget_pos = (QtCore.QPointF(x, y) + canvas.offsetToCenter()) * canvas.scale
        ev = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress,
            widget_pos,
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
        )
        canvas.mousePressEvent(ev)

    # 1. Oriented rectangle: 3 clicks -> 4 orthogonal corners.
    canvas.setEditing(False)
    canvas.createMode = "oriented_rectangle"
    _left_click(20, 20)
    _left_click(60, 20)
    _left_click(60, 40)
    assert len(canvas.shapes) == 1
    obb = canvas.shapes[0]
    assert obb.shape_type == "oriented_rectangle"
    assert len(obb.points) == 4

    # 2. 3-point circle: 3 clicks on circumference -> stored as standard "circle" [center, edge].
    canvas.createMode = "circle_3p"
    _left_click(90, 50)
    _left_click(80, 60)
    _left_click(70, 50)
    assert len(canvas.shapes) == 2
    circ = canvas.shapes[1]
    assert circ.shape_type == "circle"
    assert len(circ.points) == 2
    assert circ.points[0].x() == pytest.approx(80.0, abs=1e-2)
    assert circ.points[0].y() == pytest.approx(50.0, abs=1e-2)


def test_allow_points_outside_image_toggle():
    _ensure_qapp()
    canvas = Canvas(allow_points_outside_image=False)
    pm = QtGui.QPixmap(50, 50)
    pm.fill(QtGui.QColor("black"))
    canvas.loadPixmap(pm)

    assert canvas.outOfPixmap(QtCore.QPointF(-10, 20))
    clamped = canvas._clamp_drawing_pos(QtCore.QPointF(-10, 20))
    assert clamped.x() >= 0.0

    canvas.setAllowPointsOutsideImage(True)
    assert canvas.allowPointsOutsideImage() is True
    unclamped = canvas._clamp_drawing_pos(QtCore.QPointF(-10, 20))
    assert unclamped == QtCore.QPointF(-10, 20)

    # Moving a vertex outside image bounds works when enabled and is blocked when disabled.
    rect = _make_rect("box", 10, 10, 30, 30)
    canvas.loadShapes([rect])
    canvas.hShape = rect
    canvas.hVertex = 0
    canvas.boundedMoveVertex(QtCore.QPointF(-15, -5))
    assert rect.points[0] == QtCore.QPointF(-15, -5)

    canvas.setAllowPointsOutsideImage(False)
    canvas.boundedMoveVertex(QtCore.QPointF(-30, -30))
    assert rect.points[0].x() >= 0.0 and rect.points[0].y() >= 0.0


def test_neighbor_navigation_and_fit_selection():
    w = _build_test_window()
    try:
        left = _make_rect("left", 10, 20, 25, 35)
        right = _make_rect("right", 70, 20, 90, 35)
        w.loadShapes([left, right])
        w.canvas.setEditing(True)
        w.canvas.selectShapes([left])

        moved = w.canvas.selectNeighborShape("right")
        assert moved is right
        assert w.canvas.selectedShapes == [right]

        assert w.fitSelection() is True
        assert w.zoomMode == w.MANUAL_ZOOM
    finally:
        w.setClean()
        w.close()


def test_edit_label_preserves_non_bool_flags_and_is_undoable(monkeypatch):
    w = _build_test_window()
    try:
        pt = Shape(
            label="nose",
            shape_type="point",
            flags={
                "instance_label": "mouse_0",
                "display_label": "nose",
                "kp_visibility": 2,
                "reviewed": True,
            },
        )
        pt.points = [QtCore.QPointF(25, 25)]
        pt.close()
        w.loadShapes([pt])
        w.canvas.setEditing(True)
        w.canvas.selectShapes([pt])

        monkeypatch.setattr(
            w.labelDialog,
            "popUp",
            lambda *args, **kwargs: ("snout", {"reviewed": False}, 4, "edited"),
        )
        backups_before = len(w.canvas.shapesBackups)
        w.editLabel(w.labelList.item(0))

        assert pt.label == "snout"
        assert pt.group_id == 4
        assert pt.flags["instance_label"] == "mouse_0"
        assert pt.flags["display_label"] == "nose"
        assert pt.flags["kp_visibility"] == 2
        assert pt.flags["reviewed"] is False
        assert len(w.canvas.shapesBackups) == backups_before + 1

        # Undo restores previous label and flags.
        w.undoShapeEdit()
        assert w.canvas.shapes[0].label == "nose"
        assert w.canvas.shapes[0].flags["reviewed"] is True
    finally:
        w.setClean()
        w.close()


def test_label_dialog_history_filtering_and_label_flags():
    _ensure_qapp()
    dlg = AnnolidLabelDialog(
        parent=None,
        config={
            "labels": ["mouse", "rat", "zone_arena"],
            "label_flags": {"^mouse": ["grooming", "rearing"]},
        },
    )
    assert dlg.labelList.count() == 3
    dlg.addLabelHistory("water_spout")
    assert dlg.labelList.count() == 4

    # Typing filters the history list and populates regex-matched label_flags.
    dlg._show_flags_active = True
    dlg._base_flags_in = {}
    dlg.edit.setText("mouse")
    visible_items = [
        dlg.labelList.item(i).text()
        for i in range(dlg.labelList.count())
        if not dlg.labelList.item(i).isHidden()
    ]
    assert visible_items == ["mouse"]
    assert set(dlg._flag_checks.keys()) == {"grooming", "rearing"}
    assert not dlg._flags_box.isHidden()

    # Non-matching label hides empty flags box.
    dlg.edit.setText("rat")
    assert dlg._flag_checks == {}
    assert dlg._flags_box.isHidden()


def test_bulk_visibility_and_show_hide_toggle_all_and_keep_prev():
    w = _build_test_window()
    try:
        a = _make_rect("mouse", 10, 10, 20, 20)
        b = _make_rect("rat", 30, 10, 40, 20)
        c = _make_rect("zone_a", 50, 10, 60, 20)
        w.loadShapes([a, b, c])

        # Select items 0 and 1, uncheck item 0 -> bulk uncheck applies to item 1 too!
        w.labelList.item(0).setSelected(True)
        w.labelList.item(1).setSelected(True)
        w.labelList.item(0).setCheckState(QtCore.Qt.Unchecked)
        QtWidgets.QApplication.processEvents()
        assert w.labelList.item(1).checkState() == QtCore.Qt.Unchecked
        assert not a.visible and not b.visible and c.visible

        # Show All / Toggle All / Hide All
        w.showAllShapes()
        assert a.visible and b.visible and c.visible
        w.toggleAllShapes()
        assert not a.visible and not b.visible and not c.visible
        w.toggleAllShapes()
        assert a.visible and b.visible and c.visible

        # keep_prev preserves shapes across image_to_canvas even after resetState().
        w.toggleKeepPrevMode(True)
        img2 = QtGui.QImage(120, 100, QtGui.QImage.Format_RGB32)
        img2.fill(QtGui.QColor("black"))
        w.image_to_canvas(img2, "frame_2.png", 2)
        assert len(w.canvas.shapes) == 3
        assert [s.label for s in w.canvas.shapes] == ["mouse", "rat", "zone_a"]
    finally:
        w.setClean()
        w.close()


def test_shortcuts_do_not_conflict_between_auto_save_and_open_prev():
    w = _build_test_window()
    try:
        auto_seqs = {s.toString() for s in w.actions.saveAuto.shortcuts()}
        prev_seqs = {s.toString() for s in w.actions.openPrevImg.shortcuts()}
        assert auto_seqs
        assert prev_seqs
        assert auto_seqs.isdisjoint(prev_seqs)
    finally:
        w.setClean()
        w.close()


def test_legacy_autosave_shortcut_conflict_is_migrated_in_config():
    from annolid.configs import get_config

    # Legacy ~/.labelmerc payload where both open_prev and toggle_auto_save used Ctrl+Shift+A.
    legacy_yaml = (
        "shortcuts:\n  open_prev: [A, Ctrl+Shift+A]\n  toggle_auto_save: Ctrl+Shift+A\n"
    )
    cfg = get_config(config_file_or_yaml=legacy_yaml)
    assert cfg["shortcuts"]["toggle_auto_save"] == "Ctrl+Alt+S"

    # Non-conflicting user shortcut is preserved untouched.
    custom_yaml = "shortcuts:\n  toggle_auto_save: Ctrl+Shift+U\n"
    cfg_custom = get_config(config_file_or_yaml=custom_yaml)
    assert cfg_custom["shortcuts"]["toggle_auto_save"] == "Ctrl+Shift+U"

    # Explicit config_from_args override wins even if user asks for a specific binding.
    cfg_override = get_config(
        config_file_or_yaml=legacy_yaml,
        config_from_args={"shortcuts": {"toggle_auto_save": "Ctrl+Shift+A"}},
    )
    assert cfg_override["shortcuts"]["toggle_auto_save"] == "Ctrl+Shift+A"


def test_tiled_image_view_native_obb_circle_3p_locking_and_navigation():
    from annolid.gui.widgets.tiled_image_view import TiledImageView

    _ensure_qapp()
    view = TiledImageView()
    try:
        view.resize(400, 300)
        view._content_size = (400, 300)
        view.setSceneRect(QtCore.QRectF(0, 0, 400, 300))
        view.set_zoom_percent(100)

        def _click(x, y, mods=QtCore.Qt.NoModifier):
            vp = view.mapFromScene(QtCore.QPointF(x, y))
            ev = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonPress,
                QtCore.QPointF(vp),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                mods,
            )
            view.mousePressEvent(ev)

        def _drag(x1, y1, x2, y2, mods=QtCore.Qt.NoModifier):
            _click(x1, y1, mods=mods)
            vp2 = view.mapFromScene(QtCore.QPointF(x2, y2))
            move_ev = QtGui.QMouseEvent(
                QtCore.QEvent.MouseMove,
                QtCore.QPointF(vp2),
                QtCore.Qt.NoButton,
                QtCore.Qt.LeftButton,
                mods,
            )
            view.mouseMoveEvent(move_ev)
            rel_ev = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonRelease,
                QtCore.QPointF(vp2),
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoButton,
                mods,
            )
            view.mouseReleaseEvent(rel_ev)

        # 1. Native 3-click oriented_rectangle creation + preview.
        view.setEditing(False)
        view.createMode = "oriented_rectangle"
        _click(40, 40)
        _click(100, 40)
        view._update_drawing_preview(QtCore.QPointF(100, 80))
        assert not view._preview_item.path().isEmpty()
        _click(100, 80)
        assert len(view._shapes) == 1
        obb = view._shapes[0]
        obb.label = "obb_tiled"
        assert obb.shape_type == "oriented_rectangle"
        assert len(obb.points) == 4

        # 2. Native 3-click circle_3p creation saved as standard "circle".
        view.createMode = "circle_3p"
        _click(260, 150)
        _click(240, 170)
        view._update_drawing_preview(QtCore.QPointF(220, 150))
        assert not view._preview_item.path().isEmpty()
        _click(220, 150)
        assert len(view._shapes) == 2
        circ = view._shapes[1]
        circ.label = "circ_tiled"
        assert circ.shape_type == "circle"
        assert len(circ.points) == 2
        assert circ.points[0].x() == pytest.approx(240.0, abs=2.0)
        assert circ.points[0].y() == pytest.approx(150.0, abs=2.0)

        # 3. OBB orthogonal corner resize and Shift+Drag rotation in tiled edit mode.
        view.setEditing(True)
        corner2 = QtCore.QPointF(obb.points[2])
        _drag(corner2.x(), corner2.y(), corner2.x() + 20.0, corner2.y() + 10.0)
        assert obb.points[2].x() == pytest.approx(corner2.x() + 20.0, abs=2.0)
        assert obb.points[2].y() == pytest.approx(corner2.y() + 10.0, abs=2.0)

        angle_before = math.atan2(
            obb.points[1].y() - obb.points[0].y(),
            obb.points[1].x() - obb.points[0].x(),
        )
        c1 = QtCore.QPointF(obb.points[1])
        _drag(
            c1.x(),
            c1.y(),
            c1.x(),
            c1.y() + 25.0,
            mods=QtCore.Qt.ShiftModifier,
        )
        angle_after = math.atan2(
            obb.points[1].y() - obb.points[0].y(),
            obb.points[1].x() - obb.points[0].x(),
        )
        assert abs(angle_after - angle_before) > 0.1

        # 4. Shape locking in tiled view: selectable, but drag & vertex delete are blocked.
        obb.locked = True
        locked_pts = [(p.x(), p.y()) for p in obb.points]
        cx = sum(p[0] for p in locked_pts) / 4.0
        cy = sum(p[1] for p in locked_pts) / 4.0
        _drag(cx, cy, cx + 30.0, cy + 30.0)
        assert view.selectedShapes == [obb]
        assert [(p.x(), p.y()) for p in obb.points] == locked_pts
        view._selected_vertex_shape = obb
        view._selected_vertex_index = 0
        assert view.removeSelectedPoint() is False

        # 5. Neighbor navigation, Select All, Fit to Selection, and allow_points_outside_image.
        assert view.selectNeighborShape("right") is circ
        assert view.selectedShapes == [circ]
        assert len(view.selectAllVisibleShapes()) == 2
        assert view.fit_to_selection([circ]) is True

        view.setAllowPointsOutsideImage(True)
        assert view.allowPointsOutsideImage() is True
        assert view._scene_contains(QtCore.QPointF(-20.0, -20.0)) is True
        assert view._clamp_scene_point(QtCore.QPointF(-20.0, -20.0)) == QtCore.QPointF(
            -20.0, -20.0
        )
    finally:
        view.close()


def test_canvas_obb_shift_drag_rotation_and_cursor_anchored_zoom():
    w = _build_test_window()
    try:
        # Use a large image so zooming activates scrollbars and tests true cursor anchoring.
        img = QtGui.QImage(600, 500, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor("black"))
        w.image_to_canvas(img, "zoom_test.png", 0)
        w.resize(600, 500)
        w.show()
        QtWidgets.QApplication.processEvents()
        w.setZoom(200)
        w.paintCanvas()
        w._sync_scroll_area_layout()

        h_bar = w.scrollBars[QtCore.Qt.Horizontal]
        v_bar = w.scrollBars[QtCore.Qt.Vertical]
        assert h_bar.maximum() > 100
        assert v_bar.maximum() > 50
        h_bar.setValue(80)
        v_bar.setValue(40)

        # Pick a cursor point inside the viewport and compute the image pixel under it.
        cursor_canvas_pos = QtCore.QPoint(220, 180)
        vp_x = cursor_canvas_pos.x() - h_bar.value()
        vp_y = cursor_canvas_pos.y() - v_bar.value()
        img_before = w.canvas.transformPos(QtCore.QPointF(cursor_canvas_pos))

        # Zoom in via wheel zoomRequest: the exact same image pixel stays under (vp_x, vp_y).
        w.zoomRequest(120, cursor_canvas_pos)
        new_canvas_cursor = QtCore.QPointF(vp_x + h_bar.value(), vp_y + v_bar.value())
        img_after = w.canvas.transformPos(new_canvas_cursor)
        assert img_after.x() == pytest.approx(img_before.x(), abs=1.0)
        assert img_after.y() == pytest.approx(img_before.y(), abs=1.0)

        # Zoom back out via wheel zoomRequest: pixel stays anchored in reverse too.
        w.zoomRequest(
            -120, QtCore.QPoint(int(new_canvas_cursor.x()), int(new_canvas_cursor.y()))
        )
        back_canvas_cursor = QtCore.QPointF(vp_x + h_bar.value(), vp_y + v_bar.value())
        img_back = w.canvas.transformPos(back_canvas_cursor)
        assert img_back.x() == pytest.approx(img_before.x(), abs=1.5)
        assert img_back.y() == pytest.approx(img_before.y(), abs=1.5)

        # Canvas Shift+Drag OBB rotation via real mouse events.
        obb = Shape(label="rot_obb", shape_type="oriented_rectangle")
        for x, y in oriented_rect_from_three_points((100, 100), (160, 100), (160, 140)):
            obb.addPoint(QtCore.QPointF(x, y))
        obb.close()
        w.loadShapes([obb])
        w.canvas.setEditing(True)
        w.canvas.hShape = obb
        w.canvas.hVertex = 1
        angle_before = math.atan2(
            obb.points[1].y() - obb.points[0].y(),
            obb.points[1].x() - obb.points[0].x(),
        )
        w.canvas.boundedMoveVertex(QtCore.QPointF(160.0, 130.0), rotate_obb=True)
        angle_after = math.atan2(
            obb.points[1].y() - obb.points[0].y(),
            obb.points[1].x() - obb.points[0].x(),
        )
        assert abs(angle_after - angle_before) > 0.1
    finally:
        w.setClean()
        w.close()


@pytest.mark.parametrize("dpr", [1.0, 2.0])
@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_hidpi_and_multi_zoom_shape_and_label_pill_rendering(dpr: float, scale: float):
    _ensure_qapp()
    logical_w, logical_h = 320, 240
    img = QtGui.QImage(
        int(logical_w * dpr),
        int(logical_h * dpr),
        QtGui.QImage.Format_ARGB32,
    )
    img.setDevicePixelRatio(dpr)
    img.fill(QtGui.QColor(12, 16, 22))

    # Cover normal, locked OBB, top-edge clamp, right-edge clamp, and mask shapes.
    obb_locked = Shape(
        label="mouse_obb",
        shape_type="oriented_rectangle",
        group_id=2,
        locked=True,
    )
    for x, y in oriented_rect_from_three_points((30, 30), (90, 45), (80, 85)):
        obb_locked.addPoint(QtCore.QPointF(x, y))
    obb_locked.close()

    top_right_rect = _make_rect("edge_box", 240, 2, 315, 55, group_id=9)
    mask_arr = np.zeros((24, 24), dtype=bool)
    mask_arr[4:20, 4:20] = True
    mask_shape = _make_mask_shape("mask_obj", x1=120, y1=80, arr=mask_arr)

    painter = QtGui.QPainter(img)
    try:
        painter.scale(scale, scale)
        Shape.scale = scale
        Shape.show_label_names = True
        for shape in (obb_locked, top_right_rect, mask_shape):
            shape.scale = scale
            shape.selected = shape is top_right_rect
            shape.paint(painter, logical_w, logical_h)
    finally:
        painter.end()

    # Verify non-background pixels were painted cleanly at every DPR and zoom scale.
    raw = img.constBits()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(
        img.height(), img.bytesPerLine() // 4, 4
    )
    bg = np.array([22, 16, 12, 255], dtype=np.uint8)  # BGRA for RGB(12, 16, 22)
    changed_pixels = int(np.any(arr[:, : img.width(), :] != bg, axis=-1).sum())
    assert changed_pixels > 100
