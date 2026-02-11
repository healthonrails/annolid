from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from annolid.annotation import labelme2coco


def _write_labelme_sample(root: Path, stem: str, offset: int) -> None:
    image_path = root / f"{stem}.png"
    Image.new("RGB", (64, 48), color=(0, 0, 0)).save(image_path)
    payload = {
        "version": "5.5.0",
        "imagePath": image_path.name,
        "imageHeight": 48,
        "imageWidth": 64,
        "shapes": [
            {
                "label": "mouse",
                "shape_type": "polygon",
                "points": [
                    [10 + offset, 10],
                    [26 + offset, 10],
                    [26 + offset, 26],
                    [10 + offset, 26],
                ],
                "group_id": 1,
            }
        ],
    }
    (root / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_labelme_pose_sample(root: Path, stem: str, offset: int) -> None:
    image_path = root / f"{stem}.png"
    Image.new("RGB", (96, 80), color=(0, 0, 0)).save(image_path)
    payload = {
        "version": "5.5.0",
        "imagePath": image_path.name,
        "imageHeight": 80,
        "imageWidth": 96,
        "shapes": [
            {
                "label": "superanimal",
                "shape_type": "rectangle",
                "points": [[10 + offset, 8], [60 + offset, 48]],
                "group_id": 7,
            },
            {
                "label": "nose",
                "shape_type": "point",
                "points": [[20 + offset, 18]],
                "group_id": 7,
            },
            {
                "label": "tail_base",
                "shape_type": "point",
                "points": [[50 + offset, 40]],
                "group_id": 7,
            },
        ],
    }
    (root / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_labelme2coco_emits_standard_fields(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "labelme"
    out = tmp_path / "coco_out"
    src.mkdir()

    _write_labelme_sample(src, "frame_0001", offset=0)
    _write_labelme_sample(src, "frame_0002", offset=4)

    progress = list(
        labelme2coco.convert(
            str(src),
            str(out),
            labels_file=None,
            train_valid_split=0.5,
        )
    )

    assert progress
    assert progress[-1][0] == 100

    train_json = out / "train" / "annotations.json"
    valid_json = out / "valid" / "annotations.json"
    assert train_json.exists()
    assert valid_json.exists()
    assert (out / "annotations_train.json").exists()
    assert (out / "annotations_valid.json").exists()
    assert (out / "data.yaml").exists()

    train = json.loads(train_json.read_text(encoding="utf-8"))
    valid = json.loads(valid_json.read_text(encoding="utf-8"))

    for payload in (train, valid):
        assert "images" in payload
        assert "annotations" in payload
        assert "categories" in payload
        assert payload["categories"][0]["id"] == 1
        assert payload["categories"][0]["name"] == "mouse"

    assert len(train["images"]) == 1
    assert len(valid["images"]) == 1

    for ann in train["annotations"] + valid["annotations"]:
        assert isinstance(ann["id"], int)
        assert isinstance(ann["image_id"], int)
        assert ann["category_id"] == 1
        assert ann["iscrowd"] == 0
        assert isinstance(ann["segmentation"], list)
        assert ann["segmentation"]
        assert isinstance(ann["bbox"], list)
        assert len(ann["bbox"]) == 4
        assert ann["area"] > 0


def test_labelme2coco_keypoints_mode_emits_pose_schema(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "labelme_pose"
    out = tmp_path / "coco_pose_out"
    src.mkdir()

    _write_labelme_pose_sample(src, "frame_pose_0001", offset=0)
    _write_labelme_pose_sample(src, "frame_pose_0002", offset=6)

    list(
        labelme2coco.convert(
            str(src),
            str(out),
            labels_file=None,
            train_valid_split=0.5,
            output_mode="keypoints",
        )
    )

    train = json.loads((out / "train" / "annotations.json").read_text(encoding="utf-8"))
    valid = json.loads((out / "valid" / "annotations.json").read_text(encoding="utf-8"))

    for payload in (train, valid):
        cats = payload["categories"]
        assert len(cats) == 1
        assert cats[0]["name"] == "superanimal"
        assert cats[0]["keypoints"] == ["nose", "tail_base"]

        anns = payload["annotations"]
        assert len(anns) == 1
        ann = anns[0]
        assert ann["category_id"] == 1
        assert ann["num_keypoints"] == 2
        assert len(ann["keypoints"]) == 6
        # v flags are the 3rd component in each triplet.
        assert ann["keypoints"][2] == 2.0
        assert ann["keypoints"][5] == 2.0
        assert isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4
        assert ann["area"] > 0
