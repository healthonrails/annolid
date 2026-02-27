from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


_SCRIPT = Path(
    "annolid/core/agent/skills/dataset-convert/scripts/convert_dataset.py"
).resolve()


def _run_converter(*args: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_dataset_convert_script_coco_spec_to_yolo(tmp_path: Path) -> None:
    fixture_spec = Path("tests/fixtures/dino_kpseg_coco_tiny/coco_spec.yaml").resolve()
    out_dir = tmp_path / "yolo_out"
    payload = _run_converter(
        "coco-spec-to-yolo",
        "--spec-yaml",
        str(fixture_spec),
        "--output-dir",
        str(out_dir),
    )
    assert payload["ok"] is True

    data_yaml = Path(payload["data_yaml"])
    assert data_yaml.exists()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert cfg["nc"] == 1
    assert cfg["names"] == ["mouse"]
    assert cfg["kpt_shape"] == [3, 3]


def test_dataset_convert_script_coco_to_labelme(tmp_path: Path) -> None:
    out_dir = tmp_path / "labelme_out"
    payload = _run_converter(
        "coco-to-labelme",
        "--annotations-dir",
        str(Path("tests/fixtures/dino_kpseg_coco_tiny/annotations").resolve()),
        "--images-dir",
        str(Path("tests/fixtures/dino_kpseg_coco_tiny/images").resolve()),
        "--output-dir",
        str(out_dir),
        "--recursive",
    )
    assert payload["ok"] is True
    summary = payload["summary"]
    assert int(summary["converted_images"]) >= 1
    # sidecar LabelMe JSON files should be present
    assert any(out_dir.rglob("*.json"))


def test_dataset_convert_script_labelme_to_coco_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")

    labelme_out = tmp_path / "labelme_out"
    _run_converter(
        "coco-to-labelme",
        "--annotations-dir",
        str(Path("tests/fixtures/dino_kpseg_coco_tiny/annotations").resolve()),
        "--images-dir",
        str(Path("tests/fixtures/dino_kpseg_coco_tiny/images").resolve()),
        "--output-dir",
        str(labelme_out),
        "--recursive",
    )

    coco_out = tmp_path / "coco_out"
    payload = _run_converter(
        "labelme-to-coco",
        "--input-dir",
        str(labelme_out),
        "--output-dir",
        str(coco_out),
        "--mode",
        "segmentation",
        "--labels-file",
        "",
    )
    assert payload["ok"] is True
    assert payload["generated_annotations"]
