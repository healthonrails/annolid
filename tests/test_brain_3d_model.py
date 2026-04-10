from __future__ import annotations

from annolid.gui.brain_3d_model import (
    Brain3DConfig,
    apply_coronal_polygon_edit,
    brain_model_from_other_data,
    build_brain_3d_model,
    export_brain_model_mesh_ply,
    export_brain_model_mesh_obj,
    export_brain_model_mesh,
    load_brain_model_sidecar,
    materialize_coronal_plane_shapes,
    reslice_brain_model,
    save_brain_model_sidecar,
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
        Brain3DConfig(
            point_count=12,
            source_orientation="sagittal",
            section_positions=[0.0, 1.0],
        ),
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
    assert track.contour_nodes_3d
    assert model.version >= 2
    assert model.metadata.get("source_orientation") == "sagittal"
    assert "region_presence_intervals" in model.metadata
    assert model.metadata.get("workflow_mode") == "additive"
    source_indices = list(model.metadata.get("source_page_indices") or [])
    assert source_indices == [0, 2]
    source_axis = list(model.metadata.get("source_section_axis") or [])
    assert source_axis == [
        {"page_index": 0, "section_index": 0, "position": 0.0},
        {"page_index": 2, "section_index": 2, "position": 1.0},
    ]
    source_signatures = dict(model.metadata.get("source_page_signatures") or {})
    assert set(source_signatures.keys()) == {"0", "2"}
    assert all(str(value) for value in source_signatures.values())


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


def test_export_brain_model_mesh_ply_writes_mesh_file(tmp_path) -> None:
    pages = [
        {"page_index": 0, "shapes": [_shape_dict("region_d", _poly(0, 0, 8, 8))]},
        {"page_index": 1, "shapes": [_shape_dict("region_d", _poly(1, 0, 8, 8))]},
    ]
    model = build_brain_3d_model(pages, Brain3DConfig(point_count=8))
    out_path = export_brain_model_mesh_ply(model, tmp_path / "brain_preview.ply")
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "element vertex" in content
    assert "element face" in content
    assert model.mesh_cache_metadata.get("format") == "ply"
    assert model.mesh_cache_metadata.get("path") == str(out_path)


def test_export_brain_model_mesh_obj_writes_group_map(tmp_path) -> None:
    pages = [
        {
            "page_index": 0,
            "shapes": [
                _shape_dict("region_d", _poly(0, 0, 8, 8)),
                _shape_dict("region_e", _poly(20, 0, 8, 8)),
            ],
        },
        {
            "page_index": 1,
            "shapes": [
                _shape_dict("region_d", _poly(1, 0, 8, 8)),
                _shape_dict("region_e", _poly(21, 0, 8, 8)),
            ],
        },
    ]
    model = build_brain_3d_model(pages, Brain3DConfig(point_count=8))
    out_path, object_region_map = export_brain_model_mesh_obj(
        model, tmp_path / "brain_preview.obj"
    )
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "\no brain3d_region_" in f"\n{content}"
    assert "\ng brain3d_region_" in f"\n{content}"
    assert all(str(region_id) for region_id in object_region_map.values())
    assert model.mesh_cache_metadata.get("format") == "obj"
    assert model.mesh_cache_metadata.get("path") == str(out_path)


def test_export_brain_model_mesh_filters_regions() -> None:
    pages = [
        {
            "page_index": 0,
            "shapes": [
                _shape_dict("region_e", _poly(0, 0, 8, 8)),
                _shape_dict("region_f", _poly(20, 0, 8, 8)),
            ],
        },
        {
            "page_index": 1,
            "shapes": [
                _shape_dict("region_e", _poly(1, 0, 8, 8)),
                _shape_dict("region_f", _poly(21, 0, 8, 8)),
            ],
        },
    ]
    model = build_brain_3d_model(pages, Brain3DConfig(point_count=8))
    all_mesh = export_brain_model_mesh(model)
    assert len(all_mesh.regions) == 2
    only_region_e = export_brain_model_mesh(model, region_ids={"region_e||"})
    assert set(only_region_e.regions.keys()) == {"region_e||"}
    assert model.mesh_cache_metadata.get("kind") == "tri_mesh"


def test_brain_model_sidecar_roundtrip(tmp_path) -> None:
    pages = [
        {"page_index": 0, "shapes": [_shape_dict("region_g", _poly(0, 0, 8, 8))]},
        {"page_index": 1, "shapes": [_shape_dict("region_g", _poly(2, 0, 8, 8))]},
    ]
    model = build_brain_3d_model(pages, Brain3DConfig(point_count=8))
    sidecar_path = save_brain_model_sidecar(model, tmp_path / "brain3d.sidecar.json")
    assert sidecar_path.exists()
    restored = load_brain_model_sidecar(sidecar_path)
    assert restored.to_dict() == model.to_dict()


def test_reslice_supports_zero_area_presence_state() -> None:
    pages = [
        {"page_index": 0, "shapes": [_shape_dict("region_h", _poly(0, 0, 10, 10))]},
        {"page_index": 1, "shapes": [_shape_dict("region_h", _poly(1, 0, 10, 10))]},
    ]
    model = build_brain_3d_model(pages, Brain3DConfig(point_count=8))
    region_id = next(iter(model.regions.keys()))
    set_region_presence_on_plane(model, 0, region_id, "zero_area")
    planes = reslice_brain_model(model, plane_count=2)
    entry = next(r for r in planes[0].regions if r.region_id == region_id)
    assert entry.state == "zero_area"
    assert entry.points == []


def test_apply_coronal_edit_guided_snapping_is_assistive_and_reversible_metadata() -> (
    None
):
    pages = [
        {"page_index": 0, "shapes": [_shape_dict("region_i", _poly(0, 0, 10, 10))]},
        {"page_index": 1, "shapes": [_shape_dict("region_i", _poly(1, 0, 10, 10))]},
    ]
    model = build_brain_3d_model(
        pages,
        Brain3DConfig(point_count=8, snapping_strength=0.5),
    )
    region_id = next(iter(model.regions.keys()))
    original = [(0.0, 0.0), (4.0, 0.0), (2.0, 2.0)]
    guides = [(10.0, 10.0), (4.0, 10.0), (2.0, 10.0)]
    apply_coronal_polygon_edit(
        model,
        0,
        region_id,
        original,
        guide_points=guides,
        snapping_strength=0.5,
        snapping_max_distance=20.0,
    )
    snapped = model.coronal_overrides[0][region_id]
    assert snapped[0] != original[0]
    raw = (
        model.metadata.get("coronal_overrides_raw", {}).get("0", {}).get(region_id, [])
    )
    assert raw == [[0.0, 0.0], [4.0, 0.0], [2.0, 2.0]]


def test_longitudinal_and_inplane_smoothing_pipeline() -> None:
    # Non-linear section drift to force longitudinal smoothing effect.
    pages = [
        {
            "page_index": 0,
            "shapes": [
                _shape_dict(
                    "region_j",
                    [[0, 0], [8, 0], [8, 8], [0, 8]],
                )
            ],
        },
        {
            "page_index": 1,
            "shapes": [
                _shape_dict(
                    "region_j",
                    [[5, 1], [13, 1], [13, 9], [5, 9]],
                )
            ],
        },
        {
            "page_index": 2,
            "shapes": [
                _shape_dict(
                    "region_j",
                    [[2, 2], [10, 2], [10, 10], [2, 10]],
                )
            ],
        },
    ]
    model_raw = build_brain_3d_model(
        pages,
        Brain3DConfig(
            point_count=12, smoothing_longitudinal=0.0, smoothing_inplane=0.0
        ),
    )
    model_smooth = build_brain_3d_model(
        pages,
        Brain3DConfig(
            point_count=12, smoothing_longitudinal=0.6, smoothing_inplane=0.6
        ),
    )
    region_id = next(iter(model_raw.regions.keys()))
    mid_raw = model_raw.regions[region_id].reconstructed_sections[1]
    mid_smooth = model_smooth.regions[region_id].reconstructed_sections[1]
    assert mid_raw != mid_smooth
    planes_raw = reslice_brain_model(model_raw, plane_count=3)
    planes_smooth = reslice_brain_model(model_smooth, plane_count=3)
    raw_points = next(
        r for r in planes_raw[1].regions if r.region_id == region_id
    ).points
    smooth_points = next(
        r for r in planes_smooth[1].regions if r.region_id == region_id
    ).points
    assert raw_points != smooth_points


def test_interpolation_density_adds_virtual_sections_between_sagittal_pages() -> None:
    pages = [
        {"page_index": 0, "shapes": [_shape_dict("region_k", _poly(0, 0, 10, 10))]},
        {"page_index": 1, "shapes": [_shape_dict("region_k", _poly(2, 0, 10, 10))]},
    ]
    model = build_brain_3d_model(
        pages,
        Brain3DConfig(point_count=8, interpolation_density=3),
    )
    assert model.section_indices == [0, 1, 2, 3]
    assert model.metadata.get("source_index_scale") == 3
    positions = list(model.section_positions or [])
    assert len(positions) == 4
    assert positions[0] == 0.0
    assert abs(positions[-1] - 1.0) < 1e-6
