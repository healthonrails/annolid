from __future__ import annotations

import json
from pathlib import Path

from annolid.utils.files import (
    has_frame_annotation,
    has_manual_labeled_frame,
    should_start_predictions_from_frame0,
)


def test_has_manual_labeled_frame_detects_png_json_pair(tmp_path: Path) -> None:
    folder = tmp_path / "video"
    folder.mkdir()

    (folder / "video_000000000.png").write_bytes(b"")
    (folder / "video_000000000.json").write_text("{}", encoding="utf-8")

    assert has_manual_labeled_frame(folder, 0) is True
    assert has_manual_labeled_frame(folder, 1) is False
    assert should_start_predictions_from_frame0(folder) is False


def test_has_manual_labeled_frame_false_when_only_json_exists(tmp_path: Path) -> None:
    folder = tmp_path / "video"
    folder.mkdir()

    (folder / "video_000000000.json").write_text("{}", encoding="utf-8")

    assert has_manual_labeled_frame(folder, 0) is False
    assert has_frame_annotation(folder, 0) is True
    assert should_start_predictions_from_frame0(folder) is False


def test_has_manual_labeled_frame_annotation_store_fallback(tmp_path: Path) -> None:
    folder = tmp_path / "video"
    folder.mkdir()

    store_path = folder / "video_annotations.ndjson"
    store_path.write_text(json.dumps({"frame": 0}) + "\n", encoding="utf-8")

    assert has_manual_labeled_frame(folder, 0) is True
    assert has_frame_annotation(folder, 0) is True
    assert should_start_predictions_from_frame0(folder) is False


def test_should_start_predictions_from_frame0_when_no_seed_or_output(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "video"
    folder.mkdir()

    assert has_manual_labeled_frame(folder, 0) is False
    assert has_frame_annotation(folder, 0) is False
    assert should_start_predictions_from_frame0(folder) is True
