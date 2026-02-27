from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from annolid.gui.widgets.labeling_progress_dashboard import analyze_labeling_project


def _write_labelme_json(
    path: Path,
    *,
    label: str,
    shape_type: str = "polygon",
    annotator: str | None = None,
) -> None:
    payload = {
        "version": "5.0.0",
        "flags": ({"annotator": annotator} if annotator else {}),
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


def test_analyze_labeling_project_counts_shapes_and_coverage(tmp_path: Path) -> None:
    images = tmp_path / "images"
    ann = tmp_path / "annotations"
    images.mkdir()
    ann.mkdir()

    for i in range(4):
        (images / f"img_{i:04d}.png").write_bytes(b"PNG")

    _write_labelme_json(ann / "img_0000.json", label="mouse")
    _write_labelme_json(ann / "img_0001.json", label="mouse")
    # annotation JSON with no shapes should still count as annotation file
    (ann / "img_0002.json").write_text(
        json.dumps({"shapes": [], "imagePath": "img_0002.png"}),
        encoding="utf-8",
    )

    stats = analyze_labeling_project(project_root=tmp_path, annotation_root=ann)
    assert stats.total_images == 4
    assert stats.total_annotation_files == 3
    assert stats.labeled_files == 2
    assert stats.total_shapes == 2
    assert abs(stats.coverage_percent - 50.0) < 1e-6
    assert stats.top_labels and stats.top_labels[0][0] == "mouse"
    assert isinstance(stats.annotator_file_counts, list)
    assert isinstance(stats.annotator_shape_counts, list)


def test_analyze_labeling_project_tracks_streak_and_achievements(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    ann = tmp_path / "annotations"
    images.mkdir()
    ann.mkdir()
    for i in range(3):
        (images / f"img_{i:04d}.png").write_bytes(b"PNG")

    for i in range(3):
        _write_labelme_json(ann / f"img_{i:04d}.json", label="mouse")

    now = datetime(2026, 2, 27, 12, 0, 0)
    # touch files to build a 3-day streak: 27, 26, 25
    for idx, day in enumerate((27, 26, 25)):
        ts = datetime(2026, 2, day, 10, 0, 0).timestamp()
        p = ann / f"img_{idx:04d}.json"
        p.touch()
        p.chmod(0o644)
        os.utime(p, (ts, ts))

    stats = analyze_labeling_project(
        project_root=tmp_path, annotation_root=ann, now=now
    )
    assert stats.gamification.streak_days == 3
    names = {a.name: a.unlocked for a in stats.achievements}
    assert names["First Label"] is True
    assert names["Consistency"] is True


def test_analyze_labeling_project_annotator_split(tmp_path: Path) -> None:
    images = tmp_path / "images"
    ann = tmp_path / "annotations"
    images.mkdir()
    ann.mkdir()
    for i in range(2):
        (images / f"img_{i:04d}.png").write_bytes(b"PNG")

    _write_labelme_json(ann / "img_0000.json", label="mouse", annotator="alice")
    _write_labelme_json(ann / "img_0001.json", label="mouse", annotator="bob")

    stats = analyze_labeling_project(project_root=tmp_path, annotation_root=ann)
    file_map = dict(stats.annotator_file_counts)
    shape_map = dict(stats.annotator_shape_counts)
    assert file_map.get("alice", 0) == 1
    assert file_map.get("bob", 0) == 1
    assert shape_map.get("alice", 0) == 1
    assert shape_map.get("bob", 0) == 1
