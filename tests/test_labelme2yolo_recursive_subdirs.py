import json
from pathlib import Path

from PIL import Image

from annolid.annotation.labelme2yolo import Labelme2YOLO


def _write_nested_annotation(root: Path, *, subdir: str, name: str) -> None:
    folder = root / subdir
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / f"{name}.png"
    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(image_path)

    annotation = {
        "version": "5.5.0",
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": 48,
        "imageWidth": 64,
        "flags": {},
        "instance_label": "mouse",
        "shapes": [
            {
                "label": "mouse",
                "points": [[10, 10], [30, 10], [30, 30], [10, 30]],
                "shape_type": "polygon",
                "flags": {"instance_label": "mouse"},
            }
        ],
    }
    (folder / f"{name}.json").write_text(json.dumps(annotation), encoding="utf-8")


def test_labelme2yolo_recurses_subdirs_and_avoids_name_collisions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "labeled-data"
    _write_nested_annotation(root, subdir="seg-1", name="img00001")
    _write_nested_annotation(root, subdir="seg-2", name="img00001")

    converter = Labelme2YOLO(str(root), recursive=True)
    converter.convert(val_size=0.0, test_size=0.0)

    labels_train = sorted((root / "YOLO_dataset" / "labels" / "train").glob("*.txt"))
    images_train = sorted((root / "YOLO_dataset" / "images" / "train").glob("*.png"))

    assert len(labels_train) == 2
    assert len(images_train) == 2
    assert labels_train[0].stem != labels_train[1].stem
    assert images_train[0].stem != images_train[1].stem
