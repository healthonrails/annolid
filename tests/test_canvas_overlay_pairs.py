from __future__ import annotations

import os

from qtpy import QtCore, QtWidgets

from annolid.gui.shape import Shape
from annolid.gui.widgets.canvas import Canvas


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def _make_point(x: float, y: float, *, overlay: bool, pair_id: str) -> Shape:
    shape = Shape("pt", shape_type="point")
    shape.addPoint(QtCore.QPointF(x, y))
    if overlay:
        shape.other_data = {
            "overlay_id": "overlay_a",
            "overlay_landmark_pair_id": pair_id,
        }
    else:
        shape.other_data = {"overlay_landmark_pair_id": pair_id}
    return shape


def test_canvas_collects_explicit_landmark_pairs_from_shapes() -> None:
    _ensure_qapp()

    overlay_point = _make_point(1.0, 2.0, overlay=True, pair_id="pair_a")
    image_point = _make_point(5.0, 7.0, overlay=False, pair_id="pair_a")
    orphan_overlay = _make_point(9.0, 10.0, overlay=True, pair_id="pair_b")

    pairs = Canvas._collect_explicit_landmark_pairs_from_shapes(
        [overlay_point, image_point, orphan_overlay]
    )

    assert pairs == [("pair_a", (1.0, 2.0), (5.0, 7.0))]


def test_canvas_nearest_explicit_landmark_pair_detects_line_hit() -> None:
    _ensure_qapp()

    canvas = Canvas()
    try:
        overlay_point = _make_point(1.0, 2.0, overlay=True, pair_id="pair_a")
        image_point = _make_point(5.0, 6.0, overlay=False, pair_id="pair_a")
        canvas.shapes = [overlay_point, image_point]

        pair_id = canvas._nearest_explicit_landmark_pair(QtCore.QPointF(3.0, 4.0))

        assert pair_id == "pair_a"

        canvas.setSelectedOverlayLandmarkPair("pair_a")

        assert canvas._selected_overlay_landmark_pair_id == "pair_a"
        assert (
            canvas._explicit_landmark_pair_pen(1.0, selected=True).widthF()
            > canvas._explicit_landmark_pair_pen(1.0, selected=False).widthF()
        )
        assert canvas._selected_explicit_landmark_pair_points() == [
            (1.0, 2.0),
            (5.0, 6.0),
        ]
    finally:
        canvas.close()
