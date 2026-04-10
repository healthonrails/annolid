from __future__ import annotations

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


def _poly(x0: float, y0: float, w: float, h: float) -> list[list[float]]:
    return [
        [x0, y0],
        [x0 + w, y0],
        [x0 + w, y0 + h],
        [x0, y0 + h],
    ]


def _shape_dict(label: str, points: list[list[float]]) -> dict:
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
        "description": "",
        "visible": True,
    }


def test_build_brain_3d_model_interpolates_missing_sagittal_page() -> None:
    pages = [
        {"page_index": 0, "shapes": [_shape_dict("region_a", _poly(0, 0, 10, 10))]},
        {"page_index": 2, "shapes": [_shape_dict("region_a", _poly(2, 0, 10, 10))]},
    ]
    model = build_brain_3d_model(
        pages,
        Brain3DConfig(point_count=12, source_orientation="sagittal"),
    )
    assert model.section_indices == [0, 1, 2]
    assert len(model.regions) == 1
    track = next(iter(model.regions.values()))
    assert track.presence_interval == (0, 2)
    assert 0 in track.reconstructed_sections
    assert 2 in track.reconstructed_sections
    # Missing source page is interpolated.
    assert 1 in track.reconstructed_sections
    assert len(track.reconstructed_sections[1]) == 12


def test_reslice_edit_and_presence_round_trip() -> None:
    pages = [
        {"page_index": 0, "shapes": [_shape_dict("region_b", _poly(0, 0, 20, 20))]},
        {"page_index": 1, "shapes": [_shape_dict("region_b", _poly(2, 2, 20, 20))]},
        {"page_index": 2, "shapes": [_shape_dict("region_b", _poly(4, 4, 20, 20))]},
    ]
    model = build_brain_3d_model(pages, Brain3DConfig(point_count=16))
    region_id = next(iter(model.regions.keys()))

    planes = reslice_brain_model(model, plane_count=5)
    assert len(planes) == 5
    assert len(model.generated_coronal_planes) == 5
    mid_plane = planes[2]
    mid_entry = next(r for r in mid_plane.regions if r.region_id == region_id)
    assert mid_entry.state == "present"
    assert len(mid_entry.points) >= 3

    edited_points = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
    apply_coronal_polygon_edit(model, mid_plane.plane_index, region_id, edited_points)
    planes_after = reslice_brain_model(model, plane_count=5)
    mid_after = next(r for r in planes_after[2].regions if r.region_id == region_id)
    assert mid_after.source == "override"
    assert mid_after.points == edited_points
    assert model.metadata.get("local_regeneration_requests")

    set_region_presence_on_plane(model, mid_plane.plane_index, region_id, "hidden")
    planes_hidden = reslice_brain_model(model, plane_count=5)
    mid_hidden = next(r for r in planes_hidden[2].regions if r.region_id == region_id)
    assert mid_hidden.state == "hidden"
    assert mid_hidden.points == []


def test_brain_model_other_data_roundtrip_and_materialized_shapes() -> None:
    pages = [
        {"page_index": 0, "shapes": [_shape_dict("region_c", _poly(0, 5, 12, 6))]},
        {"page_index": 1, "shapes": [_shape_dict("region_c", _poly(1, 5, 12, 6))]},
    ]
    model = build_brain_3d_model(pages, Brain3DConfig(point_count=8))
    merged = store_brain_model_in_other_data({"foo": 1}, model)
    restored = brain_model_from_other_data(merged)
    assert restored is not None
    assert restored.to_dict() == model.to_dict()

    planes = reslice_brain_model(restored, plane_count=3)
    shapes = materialize_coronal_plane_shapes(planes[1], include_hidden=False)
    assert shapes
    for shape in shapes:
        assert str(shape.shape_type or "").lower() == "polygon"
        edit = dict(shape.other_data.get("polygon_edit") or {})
        assert edit.get("source_orientation") == "coronal"
        assert "region_id" in edit
