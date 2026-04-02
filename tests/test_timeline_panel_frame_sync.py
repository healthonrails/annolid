from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.widgets.timeline_panel import (
    TimelineEvent,
    TimelineGraphicsView,
    TimelineModel,
    TimelineTrack,
)


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _send_mouse_drag(
    view: QtWidgets.QGraphicsView, start: QtCore.QPoint, end: QtCore.QPoint
) -> None:
    viewport = view.viewport()
    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(start),
        QtCore.QPointF(viewport.mapToGlobal(start)),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    move = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(end),
        QtCore.QPointF(viewport.mapToGlobal(end)),
        QtCore.Qt.NoButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    release = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease,
        QtCore.QPointF(end),
        QtCore.QPointF(viewport.mapToGlobal(end)),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    view.mousePressEvent(press)
    view.mouseMoveEvent(move)
    view.mouseReleaseEvent(release)


def test_timeline_draw_drag_emits_frame_selected_updates() -> None:
    _ensure_qapp()
    view = TimelineGraphicsView()
    view.resize(640, 220)
    view.show()
    _ensure_qapp().processEvents()

    model = TimelineModel()
    model.set_tracks(
        [
            TimelineTrack(
                track_id="grooming",
                label="grooming",
                behaviors=("grooming",),
            )
        ]
    )
    view.set_edit_mode(True)
    view.set_edit_callback(lambda *_args: None)
    view.set_model(model)
    view.set_time_range(0, 99)
    view.rebuild_scene()
    _ensure_qapp().processEvents()

    frames: list[int] = []
    view.frameSelected.connect(lambda frame: frames.append(int(frame)))

    start = view.mapFromScene(QtCore.QPointF(8.0, 26.0))
    end = view.mapFromScene(QtCore.QPointF(120.0, 26.0))
    _send_mouse_drag(view, start, end)

    assert len(frames) >= 3
    assert frames[0] <= frames[-1]
    assert frames[-1] > frames[0]

    view.close()


def test_timeline_edit_drag_existing_segment_emits_frame_updates() -> None:
    _ensure_qapp()
    view = TimelineGraphicsView()
    view.resize(640, 220)
    view.show()
    _ensure_qapp().processEvents()

    model = TimelineModel()
    model.set_tracks(
        [
            TimelineTrack(
                track_id="grooming",
                label="grooming",
                behaviors=("grooming",),
            )
        ]
    )
    model.set_events(
        [
            TimelineEvent(
                track_id="grooming",
                start_frame=12,
                end_frame=24,
                label="grooming",
                behavior="grooming",
                kind="behavior",
            )
        ]
    )

    view.set_model(model)
    view.set_time_range(0, 99)
    view.set_edit_mode(True)
    view.set_edit_callback(lambda *_args: None)
    _ensure_qapp().processEvents()

    pixels_per_frame = view._base_pixels_per_frame * view._zoom_factor
    inside_frame = 16
    inside_x = view._frame_to_x(inside_frame, pixels_per_frame)
    inside_y = view._row_y(0) + (view._row_height / 2.0)
    start = view.mapFromScene(QtCore.QPointF(inside_x, inside_y))
    end = start + QtCore.QPoint(80, 0)

    frames: list[int] = []
    view.frameSelected.connect(lambda frame: frames.append(int(frame)))
    _send_mouse_drag(view, start, end)

    assert len(frames) >= 2
    assert frames[-1] > frames[0]

    view.close()


def test_timeline_set_current_frame_scrolls_view_to_follow_playhead() -> None:
    _ensure_qapp()
    view = TimelineGraphicsView()
    view.resize(320, 180)
    view.show()
    _ensure_qapp().processEvents()

    model = TimelineModel()
    model.set_tracks(
        [
            TimelineTrack(
                track_id="grooming",
                label="grooming",
                behaviors=("grooming",),
            )
        ]
    )
    view.set_model(model)
    view.set_time_range(0, 999)
    view.rebuild_scene()
    _ensure_qapp().processEvents()

    start_scroll = view.horizontalScrollBar().value()
    view.set_current_frame(980)
    _ensure_qapp().processEvents()
    end_scroll = view.horizontalScrollBar().value()

    assert end_scroll > start_scroll
    view.close()


def test_timeline_drag_playhead_emits_frame_updates() -> None:
    _ensure_qapp()
    view = TimelineGraphicsView()
    view.resize(640, 220)
    view.show()
    _ensure_qapp().processEvents()

    model = TimelineModel()
    model.set_tracks(
        [
            TimelineTrack(
                track_id="grooming",
                label="grooming",
                behaviors=("grooming",),
            )
        ]
    )
    view.set_model(model)
    view.set_time_range(0, 99)
    view.set_current_frame(25)
    _ensure_qapp().processEvents()

    pixels_per_frame = view._base_pixels_per_frame * view._zoom_factor
    start_x = view._frame_to_x(25, pixels_per_frame)
    end_x = view._frame_to_x(60, pixels_per_frame)
    y = max(2.0, (view._header_height / 2.0))

    frames: list[int] = []
    view.frameSelected.connect(lambda frame: frames.append(int(frame)))

    start = view.mapFromScene(QtCore.QPointF(start_x, y))
    end = view.mapFromScene(QtCore.QPointF(end_x, y))
    _send_mouse_drag(view, start, end)

    assert len(frames) >= 3
    assert frames[0] <= 26
    assert frames[-1] >= 59
    view.close()
