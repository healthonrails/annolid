from __future__ import annotations

import json
from pathlib import Path

from annolid.core.agent.gui_backend.tool_handlers_shape_files import (
    delete_shapes_in_annotation_tool,
    list_shapes_in_annotation_tool,
    relabel_shapes_in_annotation_tool,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_list_and_relabel_shapes_in_labelme_json(tmp_path: Path) -> None:
    ann_path = tmp_path / "00000.json"
    _write_json(
        ann_path,
        {
            "version": "5.0",
            "imagePath": "00000.png",
            "imageHeight": 100,
            "imageWidth": 100,
            "shapes": [
                {"label": "nose", "shape_type": "point", "points": [[1, 2]]},
                {"label": "tail", "shape_type": "point", "points": [[3, 4]]},
            ],
        },
    )

    listed = list_shapes_in_annotation_tool(
        path=str(ann_path),
        exact_label="nose",
        workspace=tmp_path,
    )
    assert listed["ok"] is True
    assert listed["returned_count"] == 1
    assert listed["shapes"][0]["label"] == "nose"

    relabeled = relabel_shapes_in_annotation_tool(
        path=str(ann_path),
        old_label="nose",
        new_label="snout",
        workspace=tmp_path,
    )
    assert relabeled["ok"] is True
    assert relabeled["changed_shapes"] == 1

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    labels = [shape["label"] for shape in payload["shapes"]]
    assert labels == ["snout", "tail"]


def test_store_stub_operations_update_store_record(tmp_path: Path) -> None:
    frame_json = tmp_path / "video_00012.json"
    store_path = tmp_path / f"{tmp_path.name}_annotations.ndjson"
    stub = {
        "annotation_store": store_path.name,
        "frame": 12,
        "version": 1,
        "imagePath": "video_00012.png",
    }
    _write_json(frame_json, stub)
    store_lines = [
        {
            "frame": 12,
            "imagePath": "video_00012.png",
            "shapes": [
                {"label": "nose", "shape_type": "point", "points": [[10, 20]]},
                {"label": "tail", "shape_type": "point", "points": [[30, 40]]},
            ],
        }
    ]
    store_path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in store_lines) + "\n",
        encoding="utf-8",
    )

    listed = list_shapes_in_annotation_tool(
        path=str(frame_json),
        workspace=tmp_path,
    )
    assert listed["ok"] is True
    assert listed["source"] == "annotation_store_stub"
    assert listed["frame"] == 12
    assert listed["total_shapes"] == 2

    deleted = delete_shapes_in_annotation_tool(
        path=str(frame_json),
        exact_label="tail",
        workspace=tmp_path,
    )
    assert deleted["ok"] is True
    assert deleted["deleted_shapes"] == 1

    updated_line = store_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    updated_payload = json.loads(updated_line)
    labels = [shape["label"] for shape in updated_payload["shapes"]]
    assert labels == ["nose"]


def test_delete_tool_requires_filter_unless_delete_all(tmp_path: Path) -> None:
    ann_path = tmp_path / "a.json"
    _write_json(
        ann_path,
        {
            "version": "5.0",
            "imagePath": "a.png",
            "imageHeight": 10,
            "imageWidth": 10,
            "shapes": [{"label": "x", "shape_type": "point", "points": [[1, 1]]}],
        },
    )

    blocked = delete_shapes_in_annotation_tool(
        path=str(ann_path),
        workspace=tmp_path,
    )
    assert blocked["ok"] is False
    assert "Refusing to delete all shapes" in str(blocked["error"])

    allowed = delete_shapes_in_annotation_tool(
        path=str(ann_path),
        delete_all=True,
        workspace=tmp_path,
    )
    assert allowed["ok"] is True
    assert allowed["deleted_shapes"] == 1


def test_list_shapes_from_ndjson_in_allowed_directory(tmp_path: Path) -> None:
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir(parents=True)
    store_path = allowed_dir / "session_annotations.ndjson"
    store_path.write_text(
        json.dumps(
            {
                "frame": 5,
                "imagePath": "frame_00005.png",
                "shapes": [
                    {"label": "nose", "shape_type": "point", "points": [[1, 1]]},
                    {"label": "tail", "shape_type": "point", "points": [[2, 2]]},
                ],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    payload = list_shapes_in_annotation_tool(
        path=str(store_path),
        workspace=allowed_dir,
        allowed_roots=[str(allowed_dir)],
    )
    assert payload["ok"] is True
    assert payload["source"] == "annotation_store"
    assert payload["total_shapes"] == 2


def test_ndjson_path_outside_allowed_directory_is_blocked(tmp_path: Path) -> None:
    allowed_dir = tmp_path / "allowed"
    blocked_dir = tmp_path / "blocked"
    allowed_dir.mkdir(parents=True)
    blocked_dir.mkdir(parents=True)
    blocked_store = blocked_dir / "session_annotations.ndjson"
    blocked_store.write_text(
        json.dumps({"frame": 1, "shapes": []}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    payload = list_shapes_in_annotation_tool(
        path=str(blocked_store),
        workspace=allowed_dir,
        allowed_roots=[str(allowed_dir)],
    )
    assert payload["ok"] is False
    assert "outside allowed directories" in str(payload.get("error") or "")
