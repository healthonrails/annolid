from __future__ import annotations

import json
import os
import types
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from annolid.core.behavior.spec import (
    DEFAULT_SCHEMA_FILENAME,
    load_behavior_spec,
    save_behavior_spec,
)
from annolid.gui.controllers.flags import FlagsController
from annolid.gui.label_file import LabelFile, LabelFileError
from annolid.gui.services.project_service import ProjectService
from annolid.infrastructure import atomic_write_json, atomic_write_text


def test_atomic_write_json_preserves_existing_file_on_serialization_error(
    tmp_path: Path,
) -> None:
    target = tmp_path / "frame_000000001.json"
    original = {"version": "5.0.1", "shapes": [{"label": "mouse_1"}]}
    atomic_write_json(target, original)
    before_bytes = target.read_bytes()

    with pytest.raises(TypeError):
        atomic_write_json(target, {"bad": object()})

    assert target.read_bytes() == before_bytes
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_serializes_numpy_scalars_and_preserves_mode(
    tmp_path: Path,
) -> None:
    target = tmp_path / "metrics.json"
    atomic_write_text(target, "{}")
    os.chmod(target, 0o600)

    atomic_write_json(
        target,
        {
            "frame": np.int64(42),
            "score": np.float32(0.875),
            "visible": np.bool_(True),
        },
    )

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == {"frame": 42, "score": 0.875, "visible": True}
    assert (target.stat().st_mode & 0o777) == 0o600


def test_label_file_missing_external_image_does_not_crash_and_preserves_pil_limit(
    tmp_path: Path,
) -> None:
    sentinel_limit = 123_456_789
    prev_limit = PIL.Image.MAX_IMAGE_PIXELS
    PIL.Image.MAX_IMAGE_PIXELS = sentinel_limit
    try:
        json_path = tmp_path / "standalone.json"
        payload = {
            "version": "5.0.1",
            "flags": {"sniffing": True},
            "shapes": [
                {
                    "label": "mouse_1",
                    "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]],
                    "shape_type": "polygon",
                    "flags": {},
                }
            ],
            "imagePath": "missing_image.png",
            "imageData": None,
            "imageHeight": 64,
            "imageWidth": 64,
        }
        json_path.write_text(json.dumps(payload), encoding="utf-8")

        lf = LabelFile(str(json_path))
        assert lf.imageData is None
        assert lf.imagePath == "missing_image.png"
        assert len(lf.shapes) == 1
        assert lf.flags == {"sniffing": True}
        assert PIL.Image.MAX_IMAGE_PIXELS == sentinel_limit
    finally:
        PIL.Image.MAX_IMAGE_PIXELS = prev_limit


def test_label_file_save_failure_does_not_truncate_existing_annotation(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "frame_000000010.json"
    lf = LabelFile()
    valid_shapes = [
        {
            "label": "mouse_1",
            "points": [[1.0, 2.0], [5.0, 2.0], [5.0, 6.0]],
            "shape_type": "polygon",
            "flags": {},
        }
    ]
    lf.save(
        filename=str(json_path),
        shapes=valid_shapes,
        imagePath="frame_000000010.png",
        imageHeight=100,
        imageWidth=100,
    )
    saved_before = json_path.read_text(encoding="utf-8")

    with pytest.raises(LabelFileError):
        lf.save(
            filename=str(json_path),
            shapes=valid_shapes,
            imagePath="frame_000000010.png",
            imageHeight=100,
            imageWidth=100,
            otherData={"unserializable": object()},
        )

    assert json_path.read_text(encoding="utf-8") == saved_before


def test_project_annolid_json_preserves_workspace_and_behavior_schema_both_ways(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    ok, msg = service.create_project(tmp_path, "cohort_study")
    assert ok is True, msg

    project_dir = tmp_path / "cohort_study"
    config_path = project_dir / DEFAULT_SCHEMA_FILENAME
    video_path = project_dir / "trial_01.mp4"
    video_path.write_bytes(b"")

    discovered_schema, discovered_path = load_behavior_spec(video_path=video_path)
    assert discovered_path == config_path
    assert len(discovered_schema.behaviors) >= 1
    assert len(discovered_schema.subjects) >= 1

    discovered_schema.behaviors[0].code = "grooming"
    discovered_schema.behaviors[0].name = "Grooming"
    save_behavior_spec(discovered_schema, config_path)

    reloaded_service = ProjectService()
    ok, load_msg = reloaded_service.load_project(project_dir)
    assert ok is True, load_msg

    discovered_schema.behaviors[0].description = "Updated ethogram entry"
    save_behavior_spec(discovered_schema, config_path)

    ok, class_msg = reloaded_service.add_project_class("mouse")
    assert ok is True, class_msg

    raw_after = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw_after["name"] == "cohort_study"
    assert raw_after["version"] == "1.0.0"
    assert raw_after["schema_version"] == discovered_schema.version
    assert raw_after["classes"] == [{"name": "mouse"}]
    assert raw_after["behaviors"][0]["code"] == "grooming"
    assert raw_after["behaviors"][0]["description"] == "Updated ethogram entry"


def test_flags_controller_refuses_to_overwrite_corrupted_yaml_config(
    tmp_path: Path,
) -> None:
    bad_yaml = tmp_path / "default_config.yaml"
    original_content = "auto_save: true\nbroken_yaml: [unterminated\n"
    bad_yaml.write_text(original_content, encoding="utf-8")

    dummy = types.SimpleNamespace(_config_path=bad_yaml)
    FlagsController._persist_flags(dummy, ["grooming", "rearing"])

    assert bad_yaml.read_text(encoding="utf-8") == original_content
