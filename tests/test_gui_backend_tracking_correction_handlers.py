from __future__ import annotations

import json
from pathlib import Path

from annolid.core.agent.gui_backend.tool_handlers_tracking_correction import (
    correct_tracking_ndjson_tool,
)


def _write_ndjson(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_correct_tracking_ndjson_replaces_only_empty_frames(tmp_path: Path) -> None:
    target = tmp_path / "mouse_annotations.ndjson"
    source = tmp_path / "sam3_annotations.ndjson"
    _write_ndjson(
        target,
        [
            {"frame": 1, "shapes": []},
            {"frame": 2, "shapes": [{"label": "mouse", "points": [[0, 0]]}]},
        ],
    )
    _write_ndjson(
        source,
        [
            {"frame": 1, "shapes": [{"label": "mouse", "points": [[1, 1]]}]},
            {"frame": 2, "shapes": [{"label": "mouse", "points": [[2, 2]]}]},
            {"frame": 3, "shapes": [{"label": "mouse", "points": [[3, 3]]}]},
        ],
    )

    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        source_ndjson_path=str(source),
        replace_only_empty_shapes=True,
        allow_append_new_frames=False,
        allowed_dir=tmp_path,
        allowed_read_roots=[tmp_path],
    )
    assert payload["ok"] is True
    assert payload["replaced_frames"] == 1
    assert payload["skipped_non_empty_frames"] == 1
    assert payload["appended_frames"] == 0

    lines = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines[0]["frame"] == 1
    assert lines[0]["shapes"][0]["points"] == [[1, 1]]
    assert lines[1]["frame"] == 2
    assert lines[1]["shapes"][0]["points"] == [[0, 0]]


def test_correct_tracking_ndjson_merges_propagated_shapes_by_frame_number(
    tmp_path: Path,
) -> None:
    target = tmp_path / "mouse_annotations.ndjson"
    source = tmp_path / "sam3_predictions.ndjson"
    _write_ndjson(
        target,
        [
            {
                "frame": 10,
                "version": "annolid",
                "imagePath": "mouse_000000010.jpg",
                "flags": {"reviewed": True},
                "shapes": [
                    {
                        "label": "manual_note",
                        "shape_type": "point",
                        "points": [[5, 5]],
                        "description": "manual",
                    },
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [[0, 0], [1, 0], [1, 1]],
                        "description": "sam3",
                    },
                ],
            }
        ],
    )
    _write_ndjson(
        source,
        [
            {
                "frame_number": 10,
                "imageHeight": 100,
                "imageWidth": 120,
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [[10, 10], [20, 10], [20, 20]],
                        "description": "sam3",
                    }
                ],
            }
        ],
    )

    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        source_ndjson_path=str(source),
        replace_only_empty_shapes=False,
        allowed_dir=tmp_path,
        allowed_read_roots=[tmp_path],
    )
    assert payload["ok"] is True
    assert payload["replaced_frames"] == 1

    row = json.loads(target.read_text(encoding="utf-8").strip())
    assert row["frame"] == 10
    assert row["version"] == "annolid"
    assert row["flags"] == {"reviewed": True}
    assert [shape["label"] for shape in row["shapes"]] == ["manual_note", "mouse"]
    assert row["shapes"][1]["points"] == [[10, 10], [20, 10], [20, 20]]


def test_correct_tracking_ndjson_replace_all_shapes_drops_manual_shapes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "mouse_annotations.ndjson"
    source = tmp_path / "sam3_predictions.ndjson"
    _write_ndjson(
        target,
        [
            {
                "frame": 1,
                "shapes": [
                    {"label": "manual_note", "shape_type": "point", "points": [[5, 5]]}
                ],
            }
        ],
    )
    _write_ndjson(
        source,
        [
            {
                "frame_number": 1,
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [[1, 1], [2, 1], [2, 2]],
                    }
                ],
            }
        ],
    )

    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        source_ndjson_path=str(source),
        replace_only_empty_shapes=False,
        replace_all_shapes=True,
        allowed_dir=tmp_path,
        allowed_read_roots=[tmp_path],
    )
    assert payload["ok"] is True

    row = json.loads(target.read_text(encoding="utf-8").strip())
    assert [shape["label"] for shape in row["shapes"]] == ["mouse"]


def test_correct_tracking_ndjson_temporal_repair_fills_occlusion_gap(
    tmp_path: Path,
) -> None:
    target = tmp_path / "mouse_annotations.ndjson"
    _write_ndjson(
        target,
        [
            {
                "frame": 0,
                "shapes": [
                    {
                        "label": "mouse_a",
                        "group_id": 1,
                        "shape_type": "polygon",
                        "points": [[0, 0], [2, 0], [2, 2]],
                    },
                    {
                        "label": "mouse_b",
                        "group_id": 2,
                        "shape_type": "polygon",
                        "points": [[100, 0], [102, 0], [102, 2]],
                    },
                ],
            },
            {
                "frame": 1,
                "shapes": [
                    {
                        "label": "mouse_a",
                        "group_id": 1,
                        "shape_type": "polygon",
                        "points": [[10, 0], [12, 0], [12, 2]],
                    },
                    {
                        "label": "manual_note",
                        "shape_type": "point",
                        "points": [[50, 50]],
                        "description": "manual",
                    },
                ],
            },
            {
                "frame": 2,
                "shapes": [
                    {
                        "label": "mouse_a",
                        "group_id": 1,
                        "shape_type": "polygon",
                        "points": [[20, 0], [22, 0], [22, 2]],
                    },
                    {
                        "label": "mouse_b",
                        "group_id": 2,
                        "shape_type": "polygon",
                        "points": [[120, 0], [122, 0], [122, 2]],
                    },
                ],
            },
        ],
    )

    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        temporal_repair=True,
        expected_instance_count=2,
        max_gap_frames=3,
        max_match_distance=40.0,
        allowed_dir=tmp_path,
        allowed_read_roots=[tmp_path],
    )
    assert payload["ok"] is True
    assert payload["missing_shapes_filled"] == 1

    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame_one = rows[1]
    assert [shape["label"] for shape in frame_one["shapes"]] == [
        "mouse_a",
        "manual_note",
        "mouse_b",
    ]
    assert [shape.get("group_id") for shape in frame_one["shapes"]] == [1, None, 2]
    filled = frame_one["shapes"][2]
    assert filled["label"] == "mouse_b"
    assert filled["flags"]["occlusion_fill"] is True
    assert filled["flags"]["annolid_correction"] == "occlusion_gap_interpolated"


def test_correct_tracking_ndjson_temporal_repair_corrects_id_switch(
    tmp_path: Path,
) -> None:
    target = tmp_path / "mouse_annotations.ndjson"
    _write_ndjson(
        target,
        [
            {
                "frame": 0,
                "shapes": [
                    {
                        "label": "mouse_a",
                        "group_id": 1,
                        "shape_type": "polygon",
                        "points": [[0, 0], [2, 0], [2, 2]],
                    },
                    {
                        "label": "mouse_b",
                        "group_id": 2,
                        "shape_type": "polygon",
                        "points": [[100, 0], [102, 0], [102, 2]],
                    },
                ],
            },
            {
                "frame": 1,
                "shapes": [
                    {
                        "label": "mouse_a",
                        "group_id": 1,
                        "shape_type": "polygon",
                        "points": [[100, 0], [102, 0], [102, 2]],
                    },
                    {
                        "label": "mouse_b",
                        "group_id": 2,
                        "shape_type": "polygon",
                        "points": [[0, 0], [2, 0], [2, 2]],
                    },
                ],
            },
        ],
    )

    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        temporal_repair=True,
        expected_instance_count=2,
        max_gap_frames=2,
        max_match_distance=10.0,
        allowed_dir=tmp_path,
        allowed_read_roots=[tmp_path],
    )
    assert payload["ok"] is True
    assert payload["id_switches_corrected"] == 2

    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    switched_frame = rows[1]
    assert [shape["group_id"] for shape in switched_frame["shapes"]] == [2, 1]
    assert [shape["label"] for shape in switched_frame["shapes"]] == [
        "mouse_b",
        "mouse_a",
    ]
    assert all(
        shape["flags"]["annolid_correction"] == "id_switch_corrected"
        for shape in switched_frame["shapes"]
    )


def test_correct_tracking_ndjson_temporal_repair_handles_duplicate_labels_without_ids(
    tmp_path: Path,
) -> None:
    target = tmp_path / "mouse_annotations.ndjson"
    _write_ndjson(
        target,
        [
            {
                "frame": 0,
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [[0, 0], [2, 0], [2, 2]],
                    },
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [[100, 0], [102, 0], [102, 2]],
                    },
                ],
            },
            {
                "frame": 1,
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [[10, 0], [12, 0], [12, 2]],
                    }
                ],
            },
        ],
    )

    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        temporal_repair=True,
        expected_instance_count=2,
        max_gap_frames=2,
        max_match_distance=20.0,
        allowed_dir=tmp_path,
        allowed_read_roots=[tmp_path],
    )
    assert payload["ok"] is True
    assert payload["temporal_reference_instances"] == 2
    assert payload["missing_shapes_filled"] == 1

    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [shape["flags"]["track_id"] for shape in rows[0]["shapes"]] == [
        "track_1",
        "track_2",
    ]
    assert [shape["flags"]["track_id"] for shape in rows[1]["shapes"]] == [
        "track_1",
        "track_2",
    ]


def test_correct_tracking_ndjson_blocks_source_outside_allowed_roots(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    blocked = tmp_path / "blocked"
    allowed.mkdir(parents=True)
    blocked.mkdir(parents=True)
    target = allowed / "mouse_annotations.ndjson"
    source = blocked / "sam3_annotations.ndjson"
    _write_ndjson(target, [{"frame": 1, "shapes": []}])
    _write_ndjson(source, [{"frame": 1, "shapes": [{"label": "mouse"}]}])

    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        source_ndjson_path=str(source),
        allowed_dir=allowed,
        allowed_read_roots=[allowed],
    )
    assert payload["ok"] is False
    assert "outside allowed read roots" in str(payload.get("error") or "")


def test_correct_tracking_ndjson_preserves_malformed_target_lines(
    tmp_path: Path,
) -> None:
    target = tmp_path / "mouse_annotations.ndjson"
    source = tmp_path / "sam3_annotations.ndjson"
    target.write_text(
        '{"frame":1,"shapes":[]}\nnot-json\n\n{"frame":2,"shapes":[]}\n',
        encoding="utf-8",
    )
    _write_ndjson(
        source,
        [
            {"frame": 1, "shapes": [{"label": "mouse", "points": [[1, 1]]}]},
            {"frame": 2, "shapes": [{"label": "mouse", "points": [[2, 2]]}]},
        ],
    )

    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        source_ndjson_path=str(source),
        allowed_dir=tmp_path,
        allowed_read_roots=[tmp_path],
    )
    assert payload["ok"] is True
    assert payload["target_invalid_lines"] == 1

    lines = target.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["shapes"][0]["points"] == [[1, 1]]
    assert lines[1] == "not-json"
    assert lines[2] == ""
    assert json.loads(lines[3])["shapes"][0]["points"] == [[2, 2]]


def test_correct_tracking_ndjson_requires_source_frame_metadata(
    tmp_path: Path,
) -> None:
    target = tmp_path / "mouse_annotations.ndjson"
    source = tmp_path / "sam3_annotations.ndjson"
    _write_ndjson(target, [{"frame": 1, "shapes": []}])
    _write_ndjson(source, [{"shapes": [{"label": "mouse"}]}])

    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        source_ndjson_path=str(source),
        allowed_dir=tmp_path,
        allowed_read_roots=[tmp_path],
    )
    assert payload["ok"] is False
    assert "no records with frame metadata" in str(payload.get("error") or "")

    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [{"frame": 1, "shapes": []}]


def test_correct_tracking_ndjson_blocks_in_place_write_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    data_root = tmp_path / "readable"
    workspace.mkdir()
    data_root.mkdir()
    target = data_root / "mouse_annotations.ndjson"
    source = data_root / "sam3_annotations.ndjson"
    _write_ndjson(target, [{"frame": 1, "shapes": []}])
    _write_ndjson(source, [{"frame": 1, "shapes": [{"label": "mouse"}]}])

    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        source_ndjson_path=str(source),
        allowed_dir=workspace,
        allowed_read_roots=[data_root],
    )
    assert payload["ok"] is False
    assert "outside allowed directory" in str(payload.get("error") or "")

    out_path = workspace / "corrected.ndjson"
    payload = correct_tracking_ndjson_tool(
        ndjson_path=str(target),
        source_ndjson_path=str(source),
        output_ndjson_path=str(out_path),
        allowed_dir=workspace,
        allowed_read_roots=[data_root],
    )
    assert payload["ok"] is True
    assert out_path.exists()
