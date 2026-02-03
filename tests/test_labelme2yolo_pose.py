import json
from pathlib import Path

import pytest
from PIL import Image

from annolid.annotation.labelme2yolo import Labelme2YOLO


def _write_sample_annotation(tmp_path: Path) -> Path:
    image_width = 100
    image_height = 80
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (image_width, image_height), color=(0, 0, 0)).save(image_path)

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
    Image.new("RGB", (image_width, image_height), color=(0, 0, 0)).save(image_path)

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

    label_path = tmp_path / "YOLO_pose_vis" / "labels" / "train" / "sample.txt"
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


def test_labelme2yolo_pose_visibility_from_flags(tmp_path: Path):
    image_width = 100
    image_height = 80
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (image_width, image_height), color=(0, 0, 0)).save(image_path)

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
            },
            {
                "label": "rat_tail",
                "points": [[55, 45]],
                "shape_type": "point",
                "flags": {
                    "display_label": "tail",
                    "instance_label": "rat",
                    "kp_visible": False,
                },
            },
        ],
    }
    (tmp_path / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    converter = Labelme2YOLO(
        str(tmp_path), yolo_dataset_name="YOLO_pose_vis", include_visibility=True
    )
    converter.create_yolo_dataset_dirs()
    converter.json_to_text("train/", "sample.json")

    label_path = tmp_path / "YOLO_pose_vis" / "labels" / "train" / "sample.txt"
    content = label_path.read_text().strip().split()
    floats = list(map(float, content[1:]))
    # Head defaults to visible => v=2, tail uses flags => v=1
    assert floats[4:7] == pytest.approx([0.2, 0.375, 2.0])
    assert floats[7:10] == pytest.approx([0.55, 0.5625, 1.0])


def test_labelme2yolo_pose_reads_metadata_outside_flags(tmp_path: Path) -> None:
    image_width = 100
    image_height = 80
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (image_width, image_height), color=(0, 0, 0)).save(image_path)

    annotation = {
        "imagePath": str(image_path),
        "imageHeight": image_height,
        "imageWidth": image_width,
        "shapes": [
            {
                "label": "rat",
                "points": [[10, 20], [60, 20], [60, 50], [10, 50]],
                "shape_type": "polygon",
                "flags": {},
                "instance_label": "rat",
            },
            {
                "label": "rat_head",
                "points": [[20, 30]],
                "shape_type": "point",
                "flags": {},
                "display_label": "head",
                "instance_label": "rat",
                "kp_visibility": 2,
            },
            {
                "label": "rat_tail",
                "points": [[55, 45]],
                "shape_type": "point",
                "flags": {},
                "display_label": "tail",
                "instance_label": "rat",
                "kp_visibility": 1,
            },
        ],
    }
    (tmp_path / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    converter = Labelme2YOLO(
        str(tmp_path), yolo_dataset_name="YOLO_pose_vis", include_visibility=True
    )
    converter.create_yolo_dataset_dirs()
    converter.json_to_text("train/", "sample.json")

    label_path = tmp_path / "YOLO_pose_vis" / "labels" / "train" / "sample.txt"
    parts = label_path.read_text().strip().split()
    floats = list(map(float, parts[1:]))
    assert floats[4:7] == pytest.approx([0.2, 0.375, 2.0])
    assert floats[7:10] == pytest.approx([0.55, 0.5625, 1.0])


def test_labelme2yolo_pose_visibility_ignores_confidence_description(
    tmp_path: Path,
) -> None:
    """LabelMe `description` may store a confidence float, not YOLO visibility."""
    image_width = 100
    image_height = 80
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (image_width, image_height), color=(0, 0, 0)).save(image_path)

    annotation = {
        "imagePath": str(image_path),
        "imageHeight": image_height,
        "imageWidth": image_width,
        "shapes": [
            {
                "label": "rat",
                "points": [[10, 20], [60, 20], [60, 50], [10, 50]],
                "shape_type": "polygon",
            },
            {
                "label": "rat_head",
                "points": [[20, 30]],
                "shape_type": "point",
                # A float stored as text should not be treated as visibility=0.
                "description": "0.5452",
            },
        ],
    }
    (tmp_path / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    converter = Labelme2YOLO(
        str(tmp_path), yolo_dataset_name="YOLO_pose_vis", include_visibility=True
    )
    converter.create_yolo_dataset_dirs()
    converter.json_to_text("train/", "sample.json")

    label_path = tmp_path / "YOLO_pose_vis" / "labels" / "train" / "sample.txt"
    parts = label_path.read_text().strip().split()
    floats = list(map(float, parts[1:]))
    # Head defaults to visible => v=2.
    assert floats[4:7] == pytest.approx([0.2, 0.375, 2.0])


def test_labelme2yolo_does_not_expand_single_instance_pose_schema(tmp_path: Path):
    image_width = 100
    image_height = 80
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (image_width, image_height), color=(0, 0, 0)).save(image_path)

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

    label_path = tmp_path / "YOLO_dataset" / "labels" / "train" / "mouse_000000000.txt"
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
    Image.new("RGB", (image_width, image_height), color=(0, 0, 0)).save(image_path)

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


def test_labelme2yolo_pose_assigns_points_to_polygons_without_group_id(
    tmp_path: Path,
) -> None:
    image_width = 100
    image_height = 60
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (image_width, image_height), color=(0, 0, 0)).save(image_path)

    annotation = {
        "imagePath": str(image_path),
        "imageHeight": image_height,
        "imageWidth": image_width,
        "shapes": [
            {
                "label": "resident",
                "points": [[5, 5], [45, 5], [45, 45], [5, 45]],
                "shape_type": "polygon",
            },
            {"label": "nose", "points": [[20, 20]], "shape_type": "point"},
            {"label": "tail_base", "points": [[30, 35]], "shape_type": "point"},
            {
                "label": "intruder",
                "points": [[55, 5], [95, 5], [95, 45], [55, 45]],
                "shape_type": "polygon",
            },
            {"label": "nose", "points": [[70, 20]], "shape_type": "point"},
            {"label": "tail_base", "points": [[80, 35]], "shape_type": "point"},
        ],
    }
    (tmp_path / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    converter = Labelme2YOLO(str(tmp_path))
    assert converter.label_to_id_dict == {"resident": 0, "intruder": 1}
    assert converter.keypoint_labels_order == ["nose", "tail_base"]

    converter.create_yolo_dataset_dirs()
    converter.json_to_text("train/", "sample.json")

    label_path = tmp_path / "YOLO_dataset" / "labels" / "train" / "sample.txt"
    lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2

    parsed = {}
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        floats = list(map(float, parts[1:]))
        parsed[class_id] = floats

    # resident bbox + keypoints
    resident = parsed[0]
    assert resident[:4] == pytest.approx([0.25, 25 / 60, 0.4, 40 / 60])
    assert resident[4:6] == pytest.approx([20 / 100, 20 / 60])
    assert resident[6:8] == pytest.approx([30 / 100, 35 / 60])

    # intruder bbox + keypoints
    intruder = parsed[1]
    assert intruder[:4] == pytest.approx([0.75, 25 / 60, 0.4, 40 / 60])
    assert intruder[4:6] == pytest.approx([70 / 100, 20 / 60])
    assert intruder[6:8] == pytest.approx([80 / 100, 35 / 60])


def test_labelme2yolo_pose_strips_trailing_instance_separators(tmp_path: Path) -> None:
    image_width = 100
    image_height = 60
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (image_width, image_height), color=(0, 0, 0)).save(image_path)

    annotation = {
        "imagePath": str(image_path),
        "imageHeight": image_height,
        "imageWidth": image_width,
        "shapes": [
            {
                "label": "resident_",
                "points": [[5, 5], [45, 5], [45, 45], [5, 45]],
                "shape_type": "polygon",
            },
            {"label": "resident_nose", "points": [[20, 20]], "shape_type": "point"},
            {
                "label": "resident_tail_base",
                "points": [[30, 35]],
                "shape_type": "point",
            },
            {
                "label": "intruder_",
                "points": [[55, 5], [95, 5], [95, 45], [55, 45]],
                "shape_type": "polygon",
            },
            {"label": "intruder_nose", "points": [[70, 20]], "shape_type": "point"},
            {
                "label": "intruder_tail_base",
                "points": [[80, 35]],
                "shape_type": "point",
            },
        ],
    }
    (tmp_path / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    converter = Labelme2YOLO(str(tmp_path))
    assert converter.label_to_id_dict == {"resident": 0, "intruder": 1}
    assert converter.keypoint_labels_order == ["nose", "tail_base"]

    converter.create_yolo_dataset_dirs()
    converter.json_to_text("train/", "sample.json")

    label_path = tmp_path / "YOLO_dataset" / "labels" / "train" / "sample.txt"
    lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2

    parsed = {}
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        floats = list(map(float, parts[1:]))
        parsed[class_id] = floats

    # resident bbox + keypoints
    resident = parsed[0]
    assert resident[:4] == pytest.approx([0.25, 25 / 60, 0.4, 40 / 60])
    assert resident[4:6] == pytest.approx([20 / 100, 20 / 60])
    assert resident[6:8] == pytest.approx([30 / 100, 35 / 60])

    # intruder bbox + keypoints
    intruder = parsed[1]
    assert intruder[:4] == pytest.approx([0.75, 25 / 60, 0.4, 40 / 60])
    assert intruder[4:6] == pytest.approx([70 / 100, 20 / 60])
    assert intruder[6:8] == pytest.approx([80 / 100, 35 / 60])


def test_labelme2yolo_normalizes_prefixed_pose_schema_keypoints(tmp_path: Path) -> None:
    image_width = 100
    image_height = 80
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (image_width, image_height), color=(0, 0, 0)).save(image_path)

    (tmp_path / "pose_schema.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "keypoints": [
                    "intruder_left",
                    "intruder_right",
                    "resident_left",
                    "resident_right",
                ],
                "edges": [],
                "symmetry_pairs": [],
                "flip_idx": [1, 0, 3, 2],
                "instances": [],
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
                "label": "intruder",
                "points": [[10, 20], [60, 20], [60, 50], [10, 50]],
                "shape_type": "polygon",
            },
            {
                "label": "intruder_left",
                "points": [[20, 30]],
                "shape_type": "point",
            },
            {
                "label": "intruder_right",
                "points": [[55, 45]],
                "shape_type": "point",
            },
        ],
    }
    (tmp_path / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    converter = Labelme2YOLO(str(tmp_path))
    assert converter.pose_schema is not None
    assert converter.pose_schema.instances == ["intruder", "resident"]
    assert converter.pose_schema.keypoints == ["left", "right"]
    assert converter.keypoint_labels_order == ["left", "right"]
    assert converter.pose_schema.compute_flip_idx(["left", "right"]) == [1, 0]

    converter.create_yolo_dataset_dirs()
    converter.json_to_text("train/", "sample.json")
    converter.save_data_yaml()

    yaml_path = tmp_path / "YOLO_dataset" / "data.yaml"
    yaml_text = yaml_path.read_text()
    assert "kpt_shape: [2, 2]" in yaml_text
    assert "flip_idx: [1, 0]" in yaml_text


def test_labelme2yolo_uses_prefixed_split_folders_without_random_split(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    split_dirs = {
        "train": root / "train_session01",
        "val": root / "validation_session01",
        "test": root / "test_session01",
    }

    for split_name, folder in split_dirs.items():
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{split_name}_sample"
        image_path = folder / f"{stem}.png"
        Image.new("RGB", (64, 64), color=(0, 0, 0)).save(image_path)
        annotation = {
            "imagePath": image_path.name,
            "imageHeight": 64,
            "imageWidth": 64,
            "shapes": [
                {
                    "label": "animal",
                    "points": [[10, 10], [40, 10], [40, 40], [10, 40]],
                    "shape_type": "polygon",
                }
            ],
        }
        (folder / f"{stem}.json").write_text(json.dumps(annotation), encoding="utf-8")

    converter = Labelme2YOLO(str(root))
    converter.convert(val_size=0.9, test_size=0.9)

    yolo_root = root / "YOLO_dataset"
    assert len(list((yolo_root / "labels" / "train").glob("*.txt"))) == 1
    assert len(list((yolo_root / "labels" / "val").glob("*.txt"))) == 1
    assert len(list((yolo_root / "labels" / "test").glob("*.txt"))) == 1
