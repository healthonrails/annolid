from __future__ import annotations

import json
from pathlib import Path

from annolid.datasets.label_index_stats import (
    build_stats_from_label_index,
    label_stats_snapshot_path,
    update_label_stats_snapshot,
)


def _write_labelme(path: Path, *, label: str, shape_type: str = "polygon") -> None:
    payload = {
        "version": "5.0.0",
        "flags": {"annotator": "alice"},
        "shapes": [
            {
                "label": label,
                "points": [[1, 1], [5, 1], [5, 5]],
                "group_id": None,
                "shape_type": shape_type,
                "flags": {},
            }
        ],
        "imagePath": path.with_suffix(".png").name,
        "imageData": None,
        "imageHeight": 10,
        "imageWidth": 10,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _append_record(
    index_file: Path, *, json_path: Path, image_path: Path, indexed_at: str
) -> None:
    rec = {
        "record_version": 1,
        "indexed_at": indexed_at,
        "source": "annolid_gui",
        "image_path": str(image_path.resolve()),
        "json_path": str(json_path.resolve()),
        "shapes_count": 1,
        "labels": ["mouse"],
    }
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with index_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec))
        fh.write("\n")


def test_build_stats_from_label_index_uses_latest_records(tmp_path: Path) -> None:
    ann = tmp_path / "ann"
    ann.mkdir()
    j0 = ann / "a.json"
    j1 = ann / "b.json"
    _write_labelme(j0, label="mouse")
    _write_labelme(j1, label="mouse")
    (ann / "a.png").write_bytes(b"PNG")
    (ann / "b.png").write_bytes(b"PNG")

    index_file = tmp_path / "logs" / "label_index" / "annolid_dataset.jsonl"
    _append_record(
        index_file,
        json_path=j0,
        image_path=ann / "a.png",
        indexed_at="2026-02-26T10:00:00Z",
    )
    _append_record(
        index_file,
        json_path=j0,
        image_path=ann / "a.png",
        indexed_at="2026-02-27T10:00:00Z",
    )
    _append_record(
        index_file,
        json_path=j1,
        image_path=ann / "b.png",
        indexed_at="2026-02-27T10:01:00Z",
    )

    stats = build_stats_from_label_index(index_file=index_file, project_root=tmp_path)
    assert stats["records_total"] == 3
    assert stats["total_annotation_files"] == 2
    assert stats["edited_files"] == 1
    assert stats["created_files"] == 2
    assert stats["labeled_files"] == 2
    assert stats["total_shapes"] == 2


def test_update_label_stats_snapshot_writes_json_for_agent_access(
    tmp_path: Path,
) -> None:
    ann = tmp_path / "ann"
    ann.mkdir()
    j0 = ann / "a.json"
    _write_labelme(j0, label="mouse")
    img = ann / "a.png"
    img.write_bytes(b"PNG")

    index_file = tmp_path / "logs" / "label_index" / "annolid_dataset.jsonl"
    _append_record(
        index_file,
        json_path=j0,
        image_path=img,
        indexed_at="2026-02-27T11:00:00Z",
    )

    stats = update_label_stats_snapshot(index_file=index_file, project_root=tmp_path)
    out_file = label_stats_snapshot_path(index_file)
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["index_file"] == str(index_file.resolve())
    assert loaded["labeled_files"] == stats["labeled_files"]
