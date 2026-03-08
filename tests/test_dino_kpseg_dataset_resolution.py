from __future__ import annotations

from pathlib import Path

from PIL import Image

from annolid.segmentation.dino_kpseg.dataset_resolution import resolve_pose_dataset


def test_resolve_pose_dataset_auto_yolo_with_kpt_names(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    train_dir = root / "images" / "train"
    val_dir = root / "images" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    train_img = train_dir / "frame_0001.jpg"
    val_img = val_dir / "frame_0001.jpg"
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(train_img)
    Image.new("RGB", (32, 32), color=(100, 100, 100)).save(val_img)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [2, 3]",
                "kpt_names:",
                "  0: [nose, tail]",
            ]
        ),
        encoding="utf-8",
    )

    spec = resolve_pose_dataset(data_yaml=data_yaml, data_format="auto")
    assert spec.data_format == "yolo"
    assert spec.label_format == "yolo"
    assert len(spec.train_images) == 1
    assert len(spec.val_images) == 1
    assert spec.train_images[0].resolve() == train_img.resolve()
    assert spec.val_images[0].resolve() == val_img.resolve()
    assert spec.raw_train_entry == "images/train"
    assert spec.raw_val_entry == "images/val"


def test_resolve_pose_dataset_split_helpers(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    img_dir = root / "images" / "train"
    img_dir.mkdir(parents=True)
    img = img_dir / "frame_0001.jpg"
    Image.new("RGB", (24, 24), color=(80, 80, 80)).save(img)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/train",
                "kpt_shape: [1, 3]",
            ]
        ),
        encoding="utf-8",
    )
    spec = resolve_pose_dataset(data_yaml=data_yaml, data_format="auto")

    assert len(spec.split_images("train")) == 1
    assert len(spec.split_images("val")) == 1
    assert spec.split_labels("train") is None
    assert spec.split_labels("val") is None


def test_resolve_pose_dataset_rejects_coco_detection_spec(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    Image.new("RGB", (32, 24), color=(100, 100, 100)).save(images_dir / "train.png")
    (annotations_dir / "instances_train.json").write_text(
        """
{
  "images": [{"id": 1, "file_name": "train.png", "width": 32, "height": 24}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 4, 8, 8], "area": 64, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "mouse"}]
}
""".strip(),
        encoding="utf-8",
    )

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "format: coco",
                f"path: {root.as_posix()}",
                "image_root: images",
                "train: annotations/instances_train.json",
            ]
        ),
        encoding="utf-8",
    )

    try:
        resolve_pose_dataset(data_yaml=data_yaml, data_format="coco")
    except ValueError as exc:
        assert "Expected a COCO pose dataset" in str(exc)
    else:
        raise AssertionError(
            "Expected resolve_pose_dataset to reject COCO detection specs."
        )
