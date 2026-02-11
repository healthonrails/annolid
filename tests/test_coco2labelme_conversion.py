from __future__ import annotations

import json
from pathlib import Path

from annolid.annotation.coco2labelme import (
    convert_coco_annotations_dir_to_labelme_dataset,
    convert_coco_json_to_labelme,
)


def test_convert_coco_json_to_labelme_from_fixture(tmp_path: Path) -> None:
    fixture_root = Path("tests/fixtures/dino_kpseg_coco_tiny").resolve()
    coco_json = fixture_root / "annotations" / "train.json"
    out_dir = tmp_path / "labelme_out"

    summary = convert_coco_json_to_labelme(
        coco_json,
        output_dir=out_dir,
        images_dir=fixture_root,
    )

    assert summary["images_total"] == 2
    assert summary["converted_images"] == 2
    assert summary["missing_images"] == 0

    outputs = sorted(out_dir.glob("*.json"))
    assert len(outputs) == 2

    sample = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert "shapes" in sample
    assert isinstance(sample["shapes"], list)
    assert sample["imagePath"]
    assert sample["imageWidth"] > 0
    assert sample["imageHeight"] > 0

    # Expect keypoint points and a bbox rectangle from this fixture.
    shape_types = {str(shape.get("shape_type")) for shape in sample["shapes"]}
    assert "point" in shape_types
    assert "rectangle" in shape_types


def test_convert_coco_annotations_dir_to_labelme_dataset_from_fixture(
    tmp_path: Path,
) -> None:
    fixture_root = Path("tests/fixtures/dino_kpseg_coco_tiny").resolve()
    annotations_dir = fixture_root / "annotations"
    out_dir = tmp_path / "labelme_dataset"

    summary = convert_coco_annotations_dir_to_labelme_dataset(
        annotations_dir,
        output_dir=out_dir,
        images_dir=fixture_root,
    )

    assert summary["json_files_total"] == 2
    assert summary["images_total"] == 3
    assert summary["converted_images"] == 3
    assert summary["copied_images"] == 3
    assert summary["missing_images"] == 0

    png_files = sorted(out_dir.rglob("*.png"))
    json_files = sorted(out_dir.rglob("*.json"))
    assert len(png_files) == 3
    assert len(json_files) == 3

    # Every output image should have a sidecar LabelMe JSON.
    for img in png_files:
        sidecar = img.with_suffix(".json")
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["imagePath"] == img.name


def test_convert_coco_annotations_dir_auto_resolves_sibling_images(
    tmp_path: Path,
) -> None:
    fixture_root = Path("tests/fixtures/dino_kpseg_coco_tiny").resolve()
    annotations_dir = fixture_root / "annotations"
    out_dir = tmp_path / "labelme_dataset_auto"

    # No images_dir passed: should still resolve ../images/* paths from COCO.
    summary = convert_coco_annotations_dir_to_labelme_dataset(
        annotations_dir,
        output_dir=out_dir,
        images_dir=None,
    )

    assert summary["images_total"] == 3
    assert summary["converted_images"] == 3
    assert summary["missing_images"] == 0
