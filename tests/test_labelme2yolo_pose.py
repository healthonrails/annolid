import json
from pathlib import Path

import pytest
from PIL import Image

from annolid.annotation.labelme2yolo import Labelme2YOLO


def _write_sample_annotation(tmp_path: Path) -> Path:
    image_width = 100
    image_height = 80
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (image_width, image_height),
              color=(0, 0, 0)).save(image_path)

    annotation = {
        "imagePath": str(image_path),
        "imageHeight": image_height,
        "imageWidth": image_width,
        "shapes": [
            {
                "label": "rat",
                "points": [[10, 20], [60, 20], [60, 50], [10, 50]],
                "shape_type": "polygon",
                "flags": {"instance_label": "rat"},
            },
            {
                "label": "rat_head",
                "points": [[20, 30]],
                "shape_type": "point",
                "flags": {},
                "visible": True,
            },
            {
                "label": "rat_tail",
                "points": [[55, 45]],
                "shape_type": "point",
                "flags": {
                    "display_label": "tail",
                    "instance_label": "rat",
                },
                "visible": False,
                "description": 1,
            },
        ],
    }

    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps(annotation))
    return json_path


def _write_points_only_annotation(tmp_path: Path) -> Path:
    image_width = 120
    image_height = 90
    image_path = tmp_path / "mouse.png"
    Image.new("RGB", (image_width, image_height),
              color=(0, 0, 0)).save(image_path)

    annotation = {
        "imagePath": str(image_path),
        "imageHeight": image_height,
        "imageWidth": image_width,
        "shapes": [
            {
                "label": "ear",
                "points": [[30, 40]],
                "shape_type": "point",
            },
            {
                "label": "tailbase",
                "points": [[80, 60]],
                "shape_type": "point",
            },
        ],
    }
    json_path = tmp_path / "mouse_000000000.json"
    json_path.write_text(json.dumps(annotation))
    return json_path


def test_labelme2yolo_pose_conversion(tmp_path):
    _write_sample_annotation(tmp_path)
    converter = Labelme2YOLO(str(tmp_path))

    assert converter.label_to_id_dict == {"rat": 0}
    assert converter.keypoint_labels_order == ["head", "tail"]

    converter.create_yolo_dataset_dirs()
    converter.json_to_text("train/", "sample.json")

    label_path = tmp_path / "YOLO_dataset" / "labels" / "train" / "sample.txt"
    content = label_path.read_text().strip().split()
    assert content[0] == "0"
    # class + 4 bbox values + 4 keypoint coordinates
    assert len(content) == 1 + 4 + 2 * 2

    # Verify bbox and keypoint coordinates
    floats = list(map(float, content[1:]))
    assert floats[:4] == pytest.approx([0.35, 0.4375, 0.5, 0.375])
    assert floats[4:6] == pytest.approx([0.2, 0.375])
    assert floats[6:8] == pytest.approx([0.55, 0.5625])

    assert converter.annotation_type == "pose"
    assert converter.kpt_shape == [2, 2]

    converter.save_data_yaml()
    yaml_path = tmp_path / "YOLO_dataset" / "data.yaml"
    yaml_text = yaml_path.read_text()
    assert "kpt_shape: [2, 2]" in yaml_text
    assert "kpt_labels" in yaml_text
    assert "0: head" in yaml_text
    assert "1: tail" in yaml_text


def test_labelme2yolo_pose_with_visibility(tmp_path):
    _write_sample_annotation(tmp_path)
    converter = Labelme2YOLO(
        str(tmp_path), yolo_dataset_name="YOLO_pose_vis", include_visibility=True
    )

    converter.create_yolo_dataset_dirs()
    converter.json_to_text("train/", "sample.json")

    label_path = (
        tmp_path / "YOLO_pose_vis" / "labels" / "train" / "sample.txt"
    )
    content = label_path.read_text().strip().split()
    assert content[0] == "0"
    # class + 4 bbox values + 2 keypoints * (x, y, v)
    assert len(content) == 1 + 4 + 2 * 3

    floats = list(map(float, content[1:]))
    assert floats[:4] == pytest.approx([0.35, 0.4375, 0.5, 0.375])
    # Head visible => v=2, tail occluded (description=1) => v=1
    assert floats[4:7] == pytest.approx([0.2, 0.375, 2.0])
    assert floats[7:10] == pytest.approx([0.55, 0.5625, 1.0])

    assert converter.kpt_shape == [2, 3]


def test_labelme2yolo_does_not_expand_single_instance_pose_schema(tmp_path: Path):
    image_width = 100
    image_height = 80
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (image_width, image_height),
              color=(0, 0, 0)).save(image_path)

    (tmp_path / "pose_schema.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "keypoints": ["head", "tail"],
                "edges": [],
                "symmetry_pairs": [],
                # Single-instance schemas should not force prefix expansion in YOLO export.
                "instances": ["mouse"],
                "instance_separator": "_",
            }
        ),
        encoding="utf-8",
    )

    annotation = {
        "imagePath": str(image_path),
        "imageHeight": image_height,
        "imageWidth": image_width,
        "shapes": [
            {
                "label": "mouse",
                "points": [[10, 20], [60, 20], [60, 50], [10, 50]],
                "shape_type": "polygon",
                "flags": {"instance_label": "mouse"},
            },
            {
                "label": "mouse_head",
                "points": [[20, 30]],
                "shape_type": "point",
                "flags": {"instance_label": "mouse", "display_label": "head"},
                "visible": True,
            },
            {
                "label": "mouse_tail",
                "points": [[55, 45]],
                "shape_type": "point",
                "flags": {"instance_label": "mouse", "display_label": "tail"},
                "visible": True,
            },
        ],
    }
    (tmp_path / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    converter = Labelme2YOLO(str(tmp_path))
    assert converter.pose_schema is not None
    assert converter.pose_schema.instances == ["mouse"]
    assert converter.keypoint_labels_order == ["head", "tail"]


def test_labelme2yolo_pose_points_only(tmp_path):
    _write_points_only_annotation(tmp_path)
    converter = Labelme2YOLO(str(tmp_path))

    assert converter.label_to_id_dict == {"mouse": 0}
    assert converter.keypoint_labels_order == ["ear", "tailbase"]

    converter.create_yolo_dataset_dirs()
    converter.json_to_text("train/", "mouse_000000000.json")

    label_path = tmp_path / \
        "YOLO_dataset" / "labels" / \
        "train" / "mouse_000000000.txt"
    content = label_path.read_text().strip().split()

    assert content[0] == "0"
    floats = list(map(float, content[1:]))
    # Ensure bbox has non-zero width/height
    assert floats[2] > 0.0 and floats[3] > 0.0
    # Keypoints should match the normalized coordinates
    assert floats[4:6] == pytest.approx([30 / 120, 40 / 90])
    assert floats[6:8] == pytest.approx([80 / 120, 60 / 90])


def test_labelme2yolo_does_not_split_concatenated_keypoint_labels(tmp_path):
    image_width = 120
    image_height = 90
    image_path = tmp_path / "mouse.png"
    Image.new("RGB", (image_width, image_height),
              color=(0, 0, 0)).save(image_path)

    annotation = {
        "imagePath": str(image_path),
        "imageHeight": image_height,
        "imageWidth": image_width,
        "shapes": [
            {
                "label": "tail",
                "points": [[10, 20], [30, 20], [30, 40], [10, 40]],
                "shape_type": "polygon",
            },
            {
                # This should remain "tailbase" rather than being split into instance "tail" + kp "base".
                "label": "tailbase",
                "points": [[80, 60]],
                "shape_type": "point",
            },
        ],
    }
    json_path = tmp_path / "mouse_000000000.json"
    json_path.write_text(json.dumps(annotation))

    converter = Labelme2YOLO(str(tmp_path))
    assert "tailbase" in converter.keypoint_labels_order
    assert "base" not in converter.keypoint_labels_order
