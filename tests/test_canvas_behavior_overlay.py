from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy import QtCore, QtGui, QtWidgets

from annolid.domain import BehaviorDefinition, ProjectSchema, SubjectDefinition
from annolid.gui.behavior_controller import BehaviorController
from annolid.gui.controllers.flags import FlagsController
from annolid.gui.mixins.annotation_loading_mixin import AnnotationLoadingMixin
from annolid.gui.mixins.flags_overlay_mixin import FlagsOverlayMixin
from annolid.gui.mixins.shape_editing_mixin import ShapeEditingMixin
from annolid.gui.shape import Shape
from annolid.gui.widgets.canvas import Canvas
from annolid.gui.widgets.canvas_overlays import BehaviorOverlayItem
from annolid.gui.widgets.flags import FlagTableWidget
from annolid.gui.widgets.tiled_image_view import TiledImageView


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _render_canvas(canvas: Canvas) -> QtGui.QImage:
    image = QtGui.QImage(
        max(1, canvas.width()),
        max(1, canvas.height()),
        QtGui.QImage.Format_ARGB32,
    )
    image.fill(QtGui.QColor(255, 255, 255))
    canvas.render(image)
    return image


def _rect_has_pixel(
    image: QtGui.QImage,
    rect: QtCore.QRectF,
    predicate,
) -> bool:
    x0 = max(0, int(rect.left()))
    y0 = max(0, int(rect.top()))
    x1 = min(image.width() - 1, int(rect.right()))
    y1 = min(image.height() - 1, int(rect.bottom()))
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if predicate(image.pixelColor(x, y)):
                return True
    return False


def test_canvas_behavior_overlay_renders_color_coded_pills_above_shapes() -> None:
    _ensure_qapp()
    canvas = Canvas()
    canvas.resize(640, 480)
    pixmap = QtGui.QPixmap(640, 480)
    pixmap.fill(QtGui.QColor(245, 245, 245))
    canvas.loadPixmap(pixmap)

    # Cover top-left corner with a solid bright-yellow shape to verify HUD z-order.
    shape = Shape(label="arena", shape_type="rectangle")
    shape.points = [
        QtCore.QPointF(0, 0),
        QtCore.QPointF(320, 0),
        QtCore.QPointF(320, 160),
        QtCore.QPointF(0, 160),
    ]
    shape.line_color = QtGui.QColor(255, 230, 0, 255)
    shape.fill_color = QtGui.QColor(255, 230, 0, 255)
    shape.fill = True
    shape.selected = True
    canvas.loadShapes([shape])

    palette = {
        "grooming": (235, 64, 52),
        "rearing": (46, 204, 113),
    }
    canvas.setBehaviorColorProvider(lambda name: palette.get(name))
    canvas.setBehaviorText("grooming,rearing")

    rendered = _render_canvas(canvas)
    rects = list(canvas._last_behavior_overlay_rects)
    assert len(rects) == 2

    # Non-overlapping pills placed cleanly inside the canvas.
    assert not rects[0].intersects(rects[1])
    for rect in rects:
        assert rect.left() >= 0.0
        assert rect.top() >= 0.0
        assert rect.right() <= 640.0
        assert rect.bottom() <= 480.0

    # Behavior pills must paint ON TOP of the yellow shape (dark glass fill present).
    assert _rect_has_pixel(
        rendered,
        rects[0],
        lambda c: c.red() < 60 and c.green() < 60 and c.blue() < 60,
    )
    # Red accent dot/border for grooming and green accent dot/border for rearing.
    assert _rect_has_pixel(
        rendered,
        rects[0],
        lambda c: c.red() > 190 and c.green() < 110 and c.blue() < 110,
    )
    assert _rect_has_pixel(
        rendered,
        rects[1],
        lambda c: c.green() > 160 and c.red() < 110,
    )


def test_canvas_behavior_overlay_wraps_elides_and_summarizes_overflow() -> None:
    _ensure_qapp()
    canvas = Canvas()
    canvas.resize(260, 220)
    pixmap = QtGui.QPixmap(260, 220)
    pixmap.fill(QtGui.QColor(40, 40, 40))
    canvas.loadPixmap(pixmap)

    labels = [
        "very_long_continuous_behavior_name_exceeding_viewport_width_12345",
        "grooming",
        "rearing",
        "sniffing",
        "digging",
        "resting",
        "chasing",
        "freezing",
        "jumping",
        "attack",
    ]
    anchor = QtCore.QRectF(0.0, 0.0, 260.0, 220.0)
    font = canvas._behavior_overlay_font(anchor)
    items = tuple(BehaviorOverlayItem(label=name) for name in labels)
    pills = canvas._layout_behavior_pills(items, anchor, font)

    assert pills
    # Every pill must remain strictly inside the anchor bounds.
    for pill in pills:
        assert pill.rect.left() >= anchor.left()
        assert pill.rect.right() <= anchor.right() + 0.5
        assert pill.rect.top() >= anchor.top()
        assert pill.rect.bottom() <= anchor.bottom() + 0.5

    # Extremely long label is elided instead of overflowing.
    assert pills[0].text != labels[0]
    assert pills[0].text.endswith(("…", "..."))
    # Remaining labels beyond max rows collapse into a '+N more' summary chip.
    assert pills[-1].is_overflow is True
    assert pills[-1].text.startswith("+") and pills[-1].text.endswith(" more")


def test_canvas_behavior_overlay_compact_sizing_on_4k_image_in_normal_window() -> None:
    _ensure_qapp()
    canvas = Canvas()
    canvas.resize(960, 540)
    pixmap = QtGui.QPixmap(3840, 2160)
    pixmap.fill(QtGui.QColor(30, 30, 30))
    canvas.loadPixmap(pixmap)
    canvas.scale = 0.25
    canvas.setBehaviorText("grooming")

    _render_canvas(canvas)
    rects = list(canvas._last_behavior_overlay_rects)
    assert len(rects) == 1
    # Previously scaled with raw 2160px image height -> ~54pt (~80px tall banner).
    assert 18.0 <= rects[0].height() <= 36.0


def test_canvas_behavior_overlay_pins_to_visible_scroll_viewport() -> None:
    _ensure_qapp()
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.resize(400, 300)
    stack = QtWidgets.QStackedWidget()
    canvas = Canvas()
    pixmap = QtGui.QPixmap(1600, 1200)
    pixmap.fill(QtGui.QColor(60, 60, 60))
    canvas.loadPixmap(pixmap)
    canvas.setFixedSize(1600, 1200)
    stack.addWidget(canvas)
    stack.setFixedSize(1600, 1200)
    scroll_area.setWidget(stack)
    scroll_area.show()
    QtWidgets.QApplication.processEvents()

    scroll_area.horizontalScrollBar().setValue(500)
    scroll_area.verticalScrollBar().setValue(400)
    QtWidgets.QApplication.processEvents()

    canvas.setBehaviorText("sniffing")
    painter = QtGui.QPainter(canvas)
    try:
        anchor = canvas._behavior_overlay_anchor_rect(painter)
    finally:
        painter.end()

    # Anchor must follow the scrolled viewport instead of staying off-screen at (0, 0).
    assert anchor.left() >= 450.0
    assert anchor.top() >= 350.0
    assert anchor.width() <= 405.0
    assert anchor.height() <= 305.0

    scroll_area.close()
    scroll_area.deleteLater()


def test_canvas_behavior_overlay_visibility_and_corner_positions() -> None:
    _ensure_qapp()
    canvas = Canvas()
    canvas.resize(640, 480)
    pixmap = QtGui.QPixmap(640, 480)
    pixmap.fill(QtGui.QColor(255, 255, 255))
    canvas.loadPixmap(pixmap)
    canvas.setBehaviorText("grooming")

    assert canvas.hasBehaviorOverlay() is True
    canvas.setShowBehaviorOverlay(False)
    assert canvas.hasBehaviorOverlay() is False
    _render_canvas(canvas)
    assert canvas._last_behavior_overlay_rects == []

    canvas.setShowBehaviorOverlay(True)
    for pos, check in (
        ("top_left", lambda r: r.left() < 100 and r.top() < 100),
        ("top_right", lambda r: r.right() > 540 and r.top() < 100),
        ("bottom_left", lambda r: r.left() < 100 and r.bottom() > 380),
        ("bottom_right", lambda r: r.right() > 540 and r.bottom() > 380),
    ):
        canvas.setBehaviorOverlayPosition(pos)
        _render_canvas(canvas)
        assert len(canvas._last_behavior_overlay_rects) == 1
        assert check(canvas._last_behavior_overlay_rects[0])

    with pytest.raises(ValueError):
        canvas.setBehaviorOverlayPosition("center")


def test_canvas_behavior_overlay_items_render_and_reset_on_plain_text_update() -> None:
    _ensure_qapp()
    canvas = Canvas()
    canvas.resize(640, 480)
    pixmap = QtGui.QPixmap(640, 480)
    pixmap.fill(QtGui.QColor(220, 220, 220))
    canvas.loadPixmap(pixmap)

    items = [
        BehaviorOverlayItem(label="attack", kind="state", subject="mouse_1"),
        BehaviorOverlayItem(label="bite", kind="point", subject="mouse_1"),
        BehaviorOverlayItem(label="background", kind="skipped"),
    ]
    canvas.setBehaviorText("attack,background,bite")
    canvas.setBehaviorOverlayItems(items)
    _render_canvas(canvas)
    assert len(canvas._last_behavior_overlay_rects) == 3

    # Plain setBehaviorText update clears stale structured items.
    canvas.setBehaviorText("resting")
    resolved = canvas._resolved_behavior_overlay_items()
    assert resolved == (BehaviorOverlayItem(label="resting", kind="state"),)


def test_flags_controller_start_flag_preserves_concurrent_behaviors(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    canvas = Canvas()
    flag_widget = FlagTableWidget()

    window = SimpleNamespace(
        here=tmp_path,
        canvas=canvas,
        seekbar=None,
        event_type=None,
        project_schema=None,
        _update_modifier_controls_for_behavior=lambda _name: None,
        statusBar=lambda: SimpleNamespace(showMessage=lambda *_a, **_kw: None),
    )
    controller = FlagsController(
        window=window,
        widget=flag_widget,
        config_path=tmp_path / "behaviors.yaml",
    )

    controller.start_flag("grooming", record_event=False)
    assert canvas.current_behavior_text == "grooming"

    # Starting a second concurrent behavior must keep the first one visible on canvas.
    controller.start_flag("sniffing", record_event=False)
    assert canvas.current_behavior_text == "grooming,sniffing"

    controller.end_flag("grooming", record_event=False)
    assert canvas.current_behavior_text == "sniffing"

    controller.end_flag("sniffing", record_event=False)
    assert canvas.current_behavior_text is None


class _OverlayIntegrationHost(FlagsOverlayMixin, AnnotationLoadingMixin):
    def __init__(self, tmp_path: Path, schema: ProjectSchema | None = None) -> None:
        self.here = tmp_path
        self.frame_number = 0
        self.labelFile = None
        self.project_schema = schema
        self.seekbar = None
        self.event_type = None
        self.canvas = Canvas()
        self.flag_widget = FlagTableWidget()
        self.behavior_controller = BehaviorController(lambda _b: (64, 180, 240))
        if schema is not None:
            self.behavior_controller.configure_from_schema(schema)
        self.flags_controller = FlagsController(
            window=self,
            widget=self.flag_widget,
            config_path=tmp_path / "behaviors.yaml",
        )
        self.behavior_controller.add_change_listener(
            self._on_behavior_controller_changed
        )

    def _update_modifier_controls_for_behavior(self, _name) -> None:
        return

    def statusBar(self):
        return SimpleNamespace(showMessage=lambda *_a, **_kw: None)


def test_refresh_behavior_overlay_preserves_skipped_label_with_real_flags_controller(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    host = _OverlayIntegrationHost(tmp_path)
    # Seed a timeline behavior inactive on frame 15 so loadFlags() runs.
    host.behavior_controller.record_event("contact", "start", 1, persist=False)
    host.behavior_controller.record_event("contact", "end", 5, persist=False)
    host._behavior_label_skipped_overlay_records = [
        {
            "start_frame": 10,
            "end_frame": 20,
            "label": "no_behavior",
            "classification": "no_behavior",
        }
    ]

    host._refresh_behavior_overlay(frame_number=15)

    assert host.canvas.current_behavior_text == "no_behavior"
    assert host.canvas._resolved_behavior_overlay_items() == (
        BehaviorOverlayItem(label="no_behavior", kind="skipped"),
    )
    assert host.flags_controller.pinned_flags.get("contact") is False


def test_refresh_behavior_overlay_shows_point_events_and_multi_subject_labels(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    schema = ProjectSchema(
        subjects=[
            SubjectDefinition(id="m1", name="mouse_1"),
            SubjectDefinition(id="m2", name="mouse_2"),
        ],
        behaviors=[
            BehaviorDefinition(code="attack", name="Attack", is_state=True),
            BehaviorDefinition(code="groom", name="Groom", is_state=True),
            BehaviorDefinition(code="bite", name="Bite", is_state=False),
        ],
    )
    host = _OverlayIntegrationHost(tmp_path, schema=schema)
    host.frame_number = 20

    host.behavior_controller.record_event(
        "attack", "start", 10, subject="mouse_1", persist=False
    )
    host.behavior_controller.record_event(
        "attack", "end", 30, subject="mouse_1", persist=False
    )
    host.behavior_controller.record_event(
        "groom", "start", 10, subject="mouse_2", persist=False
    )
    host.behavior_controller.record_event(
        "groom", "end", 30, subject="mouse_2", persist=False
    )
    host.behavior_controller.record_event(
        "bite", "point", 20, subject="mouse_1", persist=False
    )

    # Change listener already synchronized frame 20 automatically.
    assert host.canvas.current_behavior_text == "attack,bite,groom"
    items_f20 = host.canvas._resolved_behavior_overlay_items()
    assert (
        BehaviorOverlayItem(label="attack", kind="state", subject="mouse_1")
        in items_f20
    )
    assert (
        BehaviorOverlayItem(label="groom", kind="state", subject="mouse_2") in items_f20
    )
    assert (
        BehaviorOverlayItem(label="bite", kind="point", subject="mouse_1") in items_f20
    )
    # Point event must NOT latch as True in pinned_flags.
    assert host.flags_controller.pinned_flags.get("bite", False) is False

    # Advancing to frame 21 clears instantaneous point event while preserving states.
    host.frame_number = 21
    host._refresh_behavior_overlay(frame_number=21)
    assert host.canvas.current_behavior_text == "attack,groom"
    items_f21 = host.canvas._resolved_behavior_overlay_items()
    assert all(item.label != "bite" for item in items_f21)


def test_update_flags_from_file_filters_false_flags_and_survives_refresh(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    host = _OverlayIntegrationHost(tmp_path)
    label_file = SimpleNamespace(flags={"grooming": True, "resting": False})
    host.labelFile = label_file

    host.update_flags_from_file(label_file)
    assert host.canvas.current_behavior_text == "grooming"
    assert host.flags_controller.pinned_flags == {"grooming": True, "resting": False}

    # Visual overlay enrichment on the same frame must NOT wipe frame-level JSON flags.
    host._refresh_behavior_overlay(frame_number=0)
    assert host.canvas.current_behavior_text == "grooming"

    # Stepping to a frame without a label file clears the frame-level JSON flag.
    host.labelFile = None
    host._refresh_behavior_overlay(frame_number=1)
    assert host.canvas.current_behavior_text is None


def test_single_subject_session_omits_redundant_subject_suffix(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    host = _OverlayIntegrationHost(tmp_path)
    host.frame_number = 15
    host.behavior_controller.record_event(
        "grooming", "start", 10, subject="Subject 1", persist=False
    )
    host.behavior_controller.record_event(
        "grooming", "end", 20, subject="Subject 1", persist=False
    )

    assert host.canvas.current_behavior_text == "grooming"
    items = host.canvas._resolved_behavior_overlay_items()
    assert items == (BehaviorOverlayItem(label="grooming", kind="state", subject=None),)


def test_behavior_controller_change_listener_updates_canvas_on_interval_edits(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    host = _OverlayIntegrationHost(tmp_path)
    host.frame_number = 12

    host.behavior_controller.create_interval(
        behavior="rearing",
        start_frame=10,
        end_frame=15,
        subject="Subject 1",
    )
    assert host.canvas.current_behavior_text == "rearing"

    host.behavior_controller.delete_interval(
        behavior="rearing",
        start_frame=10,
        end_frame=15,
        subject="Subject 1",
    )
    assert host.canvas.current_behavior_text is None


@pytest.mark.parametrize("dpr", [1.0, 2.0])
@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_canvas_behavior_overlay_hidpi_and_multi_zoom_rendering(
    dpr: float, scale: float
) -> None:
    _ensure_qapp()
    logical_w, logical_h = 640, 480
    canvas = Canvas()
    canvas.resize(logical_w, logical_h)
    pixmap = QtGui.QPixmap(logical_w, logical_h)
    pixmap.fill(QtGui.QColor(240, 240, 240))
    canvas.loadPixmap(pixmap)
    canvas.scale = scale
    canvas.setBehaviorColorProvider(
        lambda label: {"grooming": (235, 64, 52), "rearing": (46, 204, 113)}.get(label)
    )
    canvas.setBehaviorOverlayPosition("bottom_right")
    canvas.setBehaviorText("grooming,rearing")

    img = QtGui.QImage(
        int(logical_w * dpr),
        int(logical_h * dpr),
        QtGui.QImage.Format_ARGB32,
    )
    img.setDevicePixelRatio(dpr)
    img.fill(QtGui.QColor(240, 240, 240))
    canvas.render(img)

    rects = list(canvas._last_behavior_overlay_rects)
    assert len(rects) == 2
    for rect in rects:
        assert rect.left() >= 0.0
        assert rect.top() >= 0.0
        assert rect.right() <= logical_w + 0.5
        assert rect.bottom() <= logical_h + 0.5
        # Verify physical pixels at (logical * dpr) actually contain dark pill fill.
        phys_rect = QtCore.QRectF(
            rect.left() * dpr,
            rect.top() * dpr,
            rect.width() * dpr,
            rect.height() * dpr,
        )
        assert _rect_has_pixel(
            img,
            phys_rect,
            lambda c: c.red() < 60 and c.green() < 60 and c.blue() < 60,
        )


@pytest.mark.parametrize("dpr", [1.0, 2.0])
def test_tiled_image_view_behavior_overlay_renders_and_avoids_status_hud(
    dpr: float,
) -> None:
    _ensure_qapp()
    view = TiledImageView()
    try:
        view.resize(640, 480)
        view.show()
        QtWidgets.QApplication.processEvents()

        view.setBehaviorColorProvider(
            lambda label: {"grooming": (52, 211, 153)}.get(label)
        )
        view.setBehaviorText("grooming")

        # Simulate visible LargeImageStatusOverlay at top-left.
        view._status_overlay.set_status(
            surface_text="Tiled Viewer | Editing",
            details_text="backend=tifffile page=1/10 level=0 zoom=100%",
            visible=True,
        )
        view._position_status_overlay()
        QtWidgets.QApplication.processEvents()

        vp = view.viewport()
        vp_w = max(1, vp.width())
        vp_h = max(1, vp.height())
        img = QtGui.QImage(
            int(vp_w * dpr),
            int(vp_h * dpr),
            QtGui.QImage.Format_ARGB32,
        )
        img.setDevicePixelRatio(dpr)
        img.fill(QtGui.QColor(245, 245, 245))

        painter = QtGui.QPainter(img)
        try:
            view.drawForeground(painter, QtCore.QRectF(0, 0, vp_w, vp_h))
        finally:
            painter.end()

        rects = list(view._last_behavior_overlay_rects)
        assert len(rects) == 1
        pill_rect = rects[0]
        # Behavior pill must sit cleanly below the visible status HUD (zero overlap).
        assert pill_rect.top() > view._status_overlay.geometry().bottom()
        assert pill_rect.right() <= vp_w + 0.5

        phys_rect = QtCore.QRectF(
            pill_rect.left() * dpr,
            pill_rect.top() * dpr,
            pill_rect.width() * dpr,
            pill_rect.height() * dpr,
        )
        assert _rect_has_pixel(
            img,
            phys_rect,
            lambda c: c.green() > 160 and c.red() < 110,
        )

        # clear() removes transient behavior text while preserving user display settings.
        view.setBehaviorOverlayPosition("bottom_left")
        view.setShowBehaviorOverlay(False)
        view.clear()
        assert view.current_behavior_text is None
        assert view.showBehaviorOverlay() is False
        assert view.behaviorOverlayPosition() == "bottom_left"
    finally:
        view.close()
        view.deleteLater()


def test_canvas_behavior_overlay_syncs_to_tiled_image_view() -> None:
    _ensure_qapp()

    class _SyncHost(ShapeEditingMixin):
        def __init__(self) -> None:
            self._config = {"canvas": {"show_behavior_overlay": True}}
            self.canvas = Canvas()
            self.large_image_view = TiledImageView()
            self.canvas.behaviorOverlayChanged.connect(
                self._sync_behavior_overlay_to_tiled_view
            )

    host = _SyncHost()
    try:
        items = [BehaviorOverlayItem(label="attack", kind="state", subject="mouse_1")]
        host.canvas.setBehaviorText("attack")
        host.canvas.setBehaviorOverlayItems(items)
        assert host.large_image_view.current_behavior_text == "attack"
        assert host.large_image_view._resolved_behavior_overlay_items() == tuple(items)

        host.toggleShowBehaviorOverlay(False)
        assert host.canvas.showBehaviorOverlay() is False
        assert host.large_image_view.showBehaviorOverlay() is False

        host.canvas.setBehaviorOverlayPosition("top_right")
        assert host.large_image_view.behaviorOverlayPosition() == "top_right"

        host.canvas.setBehaviorText(None)
        assert host.large_image_view.current_behavior_text is None
    finally:
        host.large_image_view.close()
        host.large_image_view.deleteLater()
