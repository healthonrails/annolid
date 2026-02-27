from __future__ import annotations

import json
from pathlib import Path

from annolid.annotation.batch_relabel import run_batch_relabel


def _write_labelme(path: Path, label: str) -> None:
    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [[1, 1], [2, 1], [2, 2]],
                "shape_type": "polygon",
                "flags": {},
            }
        ],
        "imagePath": path.with_suffix(".png").name,
        "imageData": None,
        "imageHeight": 4,
        "imageWidth": 4,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_batch_relabel_updates_json_and_store(tmp_path: Path) -> None:
    ann = tmp_path / "ann"
    ann.mkdir()
    j1 = ann / "f1.json"
    j2 = ann / "f2.json"
    _write_labelme(j1, "superanimal")
    _write_labelme(j2, "mouse")

    store = ann / f"{ann.name}_annotations.ndjson"
    rec1 = {
        "frame": 1,
        "shapes": [{"label": "superanimal", "points": [[0, 0]], "shape_type": "point"}],
    }
    rec2 = {
        "frame": 2,
        "shapes": [{"label": "mouse", "points": [[0, 0]], "shape_type": "point"}],
    }
    store.write_text(
        "\n".join((json.dumps(rec1), json.dumps(rec2))) + "\n", encoding="utf-8"
    )

    preview = run_batch_relabel(
        root=tmp_path,
        old_label="superanimal",
        new_label="mouse",
        include_json_files=True,
        include_annotation_stores=True,
        dry_run=True,
    )
    assert preview.shapes_renamed == 2
    assert preview.json_files_updated == 1
    assert preview.store_files_updated == 1

    applied = run_batch_relabel(
        root=tmp_path,
        old_label="superanimal",
        new_label="mouse",
        include_json_files=True,
        include_annotation_stores=True,
        dry_run=False,
    )
    assert applied.shapes_renamed == 2

    j1_payload = json.loads(j1.read_text(encoding="utf-8"))
    assert j1_payload["shapes"][0]["label"] == "mouse"

    store_lines = [
        json.loads(line)
        for line in store.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert store_lines[0]["shapes"][0]["label"] == "mouse"


def test_run_batch_relabel_skips_store_stub_json(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    stub = root / "000000001.json"
    stub.write_text(
        json.dumps({"annotation_store": "proj_annotations.ndjson", "frame": 1}),
        encoding="utf-8",
    )

    result = run_batch_relabel(
        root=root,
        old_label="superanimal",
        new_label="mouse",
        include_json_files=True,
        include_annotation_stores=False,
        dry_run=False,
    )
    assert result.shapes_renamed == 0
    loaded = json.loads(stub.read_text(encoding="utf-8"))
    assert loaded.get("annotation_store") == "proj_annotations.ndjson"


def test_collect_label_counts_includes_json_and_store(tmp_path: Path) -> None:
    from annolid.annotation.batch_relabel import collect_label_counts

    ann = tmp_path / "ann"
    ann.mkdir()
    _write_labelme(ann / "a.json", "superanimal")

    store = ann / f"{ann.name}_annotations.ndjson"
    store.write_text(
        json.dumps(
            {
                "frame": 1,
                "shapes": [
                    {"label": "mouse", "points": [[0, 0]], "shape_type": "point"}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    counts = collect_label_counts(
        root=tmp_path,
        include_json_files=True,
        include_annotation_stores=True,
    )
    assert counts.get("superanimal", 0) == 1
    assert counts.get("mouse", 0) == 1
