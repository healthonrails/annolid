from __future__ import annotations

import json
from pathlib import Path

from annolid.annotation.keypoints import merge_shapes
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.dino_kpseg_annotations import DinoKPSEGAnnotationParser


def test_merge_shapes_respects_group_id_and_shape_type():
    existing = [
        {"label": "nose", "group_id": 0, "shape_type": "point", "points": [[1, 1]]},
        {"label": "nose", "group_id": 1, "shape_type": "point", "points": [[2, 2]]},
    ]
    new = [
        {"label": "nose", "group_id": 0, "shape_type": "point", "points": [[3, 3]]},
    ]
    merged = merge_shapes(new, existing)
    assert len(merged) == 2
    pts_by_gid = {shape["group_id"]: shape["points"][0] for shape in merged}
    assert pts_by_gid[0] == [3, 3]
    assert pts_by_gid[1] == [2, 2]


def test_dino_kpseg_annotation_parser_assigns_points_to_instance_masks(tmp_path: Path):
    payload = {
        "shapes": [
            {
                "label": "mouse",
                "shape_type": "polygon",
                "group_id": 0,
                "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
            },
            {
                "label": "mouse",
                "shape_type": "polygon",
                "group_id": 1,
                "points": [[30, 30], [40, 30], [40, 40], [30, 40]],
            },
            {
                "label": "nose",
                "shape_type": "point",
                "group_id": 0,
                "points": [[15, 15]],
            },
            {
                "label": "tail",
                "shape_type": "point",
                "group_id": 1,
                "points": [[35, 35]],
            },
            {
                "label": "nose",
                "shape_type": "point",
                "points": [[35, 35]],
            },
        ]
    }
    json_path = tmp_path / "frame_000000000.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    adapter = AnnotationAdapter(image_height=64, image_width=64, persist_json=False)
    parser = DinoKPSEGAnnotationParser(
        image_height=64,
        image_width=64,
        adapter=adapter,
    )

    manual = parser.read_manual_annotation(0, json_path)
    assert set(manual.registry.instances.keys()) == {"0", "1"}
    assert manual.display_labels["0"] == "mouse"
    assert manual.display_labels["1"] == "mouse"

    # group_id-less "nose" point inside the second polygon is assigned to instance 1
    assert manual.keypoints_by_instance["1"]["nose"] == (35.0, 35.0)


def test_dino_kpseg_annotation_parser_reuses_group_ids_by_polygon_label(tmp_path: Path):
    adapter = AnnotationAdapter(image_height=32, image_width=32, persist_json=False)
    parser = DinoKPSEGAnnotationParser(
        image_height=32,
        image_width=32,
        adapter=adapter,
    )

    payload_a = {
        "shapes": [
            {
                "label": "mouse1",
                "shape_type": "polygon",
                "points": [[5, 5], [10, 5], [10, 10], [5, 10]],
            }
        ]
    }
    payload_b = {
        "shapes": [
            {
                "label": "mouse1",
                "shape_type": "polygon",
                "points": [[6, 6], [11, 6], [11, 11], [6, 11]],
            }
        ]
    }

    json_a = tmp_path / "a.json"
    json_b = tmp_path / "b.json"
    json_a.write_text(json.dumps(payload_a), encoding="utf-8")
    json_b.write_text(json.dumps(payload_b), encoding="utf-8")

    parsed_a = parser.read_manual_annotation(0, json_a)
    parsed_b = parser.read_manual_annotation(1, json_b)

    assert set(parsed_a.registry.instances.keys()) == {"0"}
    assert set(parsed_b.registry.instances.keys()) == {"0"}
