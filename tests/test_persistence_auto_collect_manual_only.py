from __future__ import annotations

import json
from pathlib import Path

from annolid.gui.mixins.persistence_lifecycle_mixin import PersistenceLifecycleMixin


class _DummyWindow(PersistenceLifecycleMixin):
    def __init__(self, index_file: Path) -> None:
        self._index_file = str(index_file)

    def _resolve_label_index_file(self) -> str:
        return self._index_file


def _write_json(path: Path, description: str) -> None:
    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "animal",
                "points": [[10, 10], [12, 12]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "description": description,
            }
        ],
        "imagePath": path.with_suffix(".png").name,
        "imageData": None,
        "imageHeight": 64,
        "imageWidth": 64,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_auto_collect_skips_prediction_json_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    index_file = tmp_path / "index.jsonl"
    window = _DummyWindow(index_file)
    json_path = tmp_path / "frame_000000001.json"
    image_path = tmp_path / "frame_000000001.png"
    image_path.write_bytes(b"png")
    _write_json(json_path, "motion_index: 0.32")

    called = {"count": 0}

    def _fake_index_labelme_pair(**kwargs):
        _ = kwargs
        called["count"] += 1

    monkeypatch.setattr(
        "annolid.datasets.labelme_collection.index_labelme_pair",
        _fake_index_labelme_pair,
    )

    result = window._auto_collect_labelme_pair(str(json_path), str(image_path))
    assert result is None
    assert called["count"] == 0


def test_auto_collect_accepts_manual_json_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    index_file = tmp_path / "index.jsonl"
    window = _DummyWindow(index_file)
    json_path = tmp_path / "frame_000000002.json"
    image_path = tmp_path / "frame_000000002.png"
    image_path.write_bytes(b"png")
    _write_json(json_path, "manul")

    called = {"count": 0}

    def _fake_index_labelme_pair(**kwargs):
        _ = kwargs
        called["count"] += 1

    monkeypatch.setattr(
        "annolid.datasets.labelme_collection.index_labelme_pair",
        _fake_index_labelme_pair,
    )

    result = window._auto_collect_labelme_pair(str(json_path), str(image_path))
    assert result == str(index_file)
    assert called["count"] == 1


def test_auto_collect_can_include_prediction_json_with_override(
    monkeypatch, tmp_path: Path
) -> None:
    index_file = tmp_path / "index.jsonl"
    window = _DummyWindow(index_file)
    json_path = tmp_path / "frame_000000003.json"
    image_path = tmp_path / "frame_000000003.png"
    image_path.write_bytes(b"png")
    _write_json(json_path, "motion_index: 0.19")

    called = {"count": 0}

    def _fake_index_labelme_pair(**kwargs):
        _ = kwargs
        called["count"] += 1

    monkeypatch.setattr(
        "annolid.datasets.labelme_collection.index_labelme_pair",
        _fake_index_labelme_pair,
    )
    monkeypatch.setenv("ANNOLID_LABEL_INDEX_MANUAL_ONLY", "0")

    result = window._auto_collect_labelme_pair(str(json_path), str(image_path))
    assert result == str(index_file)
    assert called["count"] == 1
