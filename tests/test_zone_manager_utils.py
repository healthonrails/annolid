from __future__ import annotations

from qtpy import QtCore, QtGui

from annolid.gui.label_file import LabelFile
from annolid.gui.shape import Shape
from annolid.gui.widgets.zone_manager_utils import (
    available_arena_layout_presets,
    build_zone_popup_defaults,
    generate_arena_layout_preset,
    shape_to_zone_payload,
    write_zone_json,
    zone_file_for_source,
    zone_payload_to_shape,
)


def test_shape_to_zone_payload_adds_explicit_zone_metadata():
    shape = Shape(
        label="north_chamber",
        shape_type="rectangle",
        flags={"tags": ["phase_1", "corner"]},
        description="north corner chamber",
    )
    shape.points = [QtCore.QPointF(10, 10), QtCore.QPointF(30, 30)]

    payload = shape_to_zone_payload(
        shape,
        zone_kind="chamber",
        phase="phase_1",
        occupant_role="stim",
        access_state="open",
    )

    assert payload["flags"]["semantic_type"] == "zone"
    assert payload["flags"]["shape_category"] == "zone"
    assert payload["flags"]["zone_kind"] == "chamber"
    assert payload["flags"]["phase"] == "phase_1"
    assert payload["flags"]["occupant_role"] == "stim"
    assert payload["flags"]["access_state"] == "open"
    assert payload["flags"]["tags"] == ["phase_1", "corner"]


def test_zone_payload_round_trip_writes_labelme_json(tmp_path):
    image_path = tmp_path / "session_frame.png"
    qimage = QtGui.QImage(32, 24, QtGui.QImage.Format_RGB32)
    qimage.fill(QtGui.QColor("white"))
    assert qimage.save(str(image_path))

    payload = shape_to_zone_payload(
        {
            "label": "doorway_1",
            "shape_type": "polygon",
            "points": [[0, 0], [20, 0], [20, 10], [0, 10]],
            "description": "barrier opening",
            "flags": {},
        },
        zone_kind="doorway",
        phase="phase_2",
        occupant_role="rover",
        access_state="open",
    )
    zone_path = tmp_path / "session_zones.json"

    write_zone_json(
        zone_path,
        shapes=[payload],
        image_path=str(image_path),
        image_width=32,
        image_height=24,
    )

    loaded = LabelFile(str(zone_path))
    assert loaded.shapes[0]["flags"]["semantic_type"] == "zone"
    assert loaded.shapes[0]["flags"]["zone_kind"] == "doorway"
    assert loaded.shapes[0]["flags"]["phase"] == "phase_2"
    assert loaded.shapes[0]["flags"]["occupant_role"] == "rover"

    shape = zone_payload_to_shape(loaded.shapes[0])
    assert shape.label == "doorway_1"
    assert len(shape.points) == 4
    assert shape.flags["semantic_type"] == "zone"


def test_zone_file_for_source_uses_source_stem():
    assert str(zone_file_for_source("/tmp/session.mp4")).endswith("session_zones.json")


def test_zone_file_for_source_ignores_blank_or_placeholder_sources():
    assert zone_file_for_source("") is None
    assert zone_file_for_source(".") is None


def test_zone_popup_defaults_include_explicit_zone_metadata():
    defaults = build_zone_popup_defaults(
        label="north_chamber",
        zone_kind="chamber",
        phase="phase_1",
        occupant_role="stim",
        access_state="open",
        tags="phase_1,corner",
        description="north corner chamber",
    )

    assert defaults["text"] == "north_chamber"
    assert defaults["flags"]["semantic_type"] == "zone"
    assert defaults["flags"]["zone_kind"] == "chamber"
    assert defaults["flags"]["phase"] == "phase_1"
    assert defaults["flags"]["occupant_role"] == "stim"
    assert defaults["flags"]["access_state"] == "open"
    assert defaults["flags"]["tags"] == ["phase_1", "corner"]


def test_3x3_chamber_preset_generates_nine_editable_zone_shapes():
    presets = available_arena_layout_presets()
    assert any(preset["key"] == "3x3_chamber" for preset in presets)
    assert any(preset["key"] == "3x3_social_doors" for preset in presets)

    shapes = generate_arena_layout_preset("3x3_chamber", 900, 900)

    assert len(shapes) == 9
    labels = [shape.label for shape in shapes]
    assert len(set(labels)) == 9
    assert labels[0] == "north_west_chamber"
    assert labels[4] == "center_chamber"
    assert labels[8] == "south_east_chamber"
    for shape in shapes:
        assert shape.shape_type == "rectangle"
        assert shape.flags["semantic_type"] == "zone"
        assert shape.flags["zone_kind"] == "chamber"
        assert shape.flags["layout_tag"] == "3x3_chamber"
        assert shape.flags["layout_rows"] == 3
        assert shape.flags["layout_cols"] == 3
        assert shape.flags["tags"][0] == "3x3_chamber"
        xs = [point.x() for point in shape.points]
        ys = [point.y() for point in shape.points]
        assert min(xs) >= 0
        assert min(ys) >= 0
        assert max(xs) <= 900
        assert max(ys) <= 900


def test_3x3_social_door_preset_generates_eight_social_zones():
    shapes = generate_arena_layout_preset("3x3_social_doors", 900, 900)

    assert len(shapes) == 8
    labels = [shape.label for shape in shapes]
    assert len(set(labels)) == 8
    assert all("social_zone" in (shape.flags.get("tags") or []) for shape in shapes)
    assert all(shape.flags["zone_kind"] == "interaction_zone" for shape in shapes)
    assert all(shape.flags["occupant_role"] == "rover" for shape in shapes)
