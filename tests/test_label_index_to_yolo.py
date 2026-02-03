import json
from pathlib import Path

from annolid.datasets.builders.label_index_yolo import build_yolo_from_label_index


def _write_labelme_pair(folder: Path, *, stem: str) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / f"{stem}.png"
    json_path = folder / f"{stem}.json"

    image_path.write_bytes(b"not-a-real-png")
    payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [
            {
                "label": "animal",
                "points": [[0, 0], [10, 0], [10, 10]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        ],
        "imagePath": image_path.name,
        "imageHeight": 20,
        "imageWidth": 20,
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return json_path, image_path


def test_build_yolo_from_index_skips_missing_json(tmp_path):
    src_json, src_img = _write_labelme_pair(tmp_path / "src", stem="clip_000000000")
    missing_json = tmp_path / "src" / "missing_000000000.json"

    index_file = tmp_path / "annolid_dataset.jsonl"
    records = [
        {
            "record_version": 1,
            "image_path": str(src_img.resolve()),
            "json_path": str(src_json.resolve()),
        },
        {
            "record_version": 1,
            "image_path": str(src_img.resolve()),
            "json_path": str(missing_json.resolve()),
        },
    ]
    index_file.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )

    out_dir = tmp_path / "out"
    summary = build_yolo_from_label_index(
        index_file=index_file,
        output_dir=out_dir,
        dataset_name="ds",
        val_size=0.0,
        test_size=0.0,
        link_mode="copy",
        overwrite=True,
        keep_staging=False,
    )

    assert summary["status"] == "ok"
    assert summary["pairs_ok"] == 1
    assert summary["skipped_missing_json"] == 1

    dataset_dir = Path(summary["dataset_dir"])
    assert dataset_dir.exists()
    assert (dataset_dir / "data.yaml").exists()
    assert any((dataset_dir / "images" / "train").glob("*.png"))
    assert any((dataset_dir / "labels" / "train").glob("*.txt"))


def test_build_yolo_from_index_mixed_shapes_defaults_to_segmentation(tmp_path):
    folder = tmp_path / "src"
    json_path, img_path = _write_labelme_pair(folder, stem="clip_000000000")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["shapes"].append(
        {
            "label": "animal_nose",
            "points": [[5, 5]],
            "group_id": None,
            "shape_type": "point",
            "flags": {},
        }
    )
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    index_file = tmp_path / "annolid_dataset.jsonl"
    index_file.write_text(
        json.dumps(
            {
                "record_version": 1,
                "image_path": str(img_path.resolve()),
                "json_path": str(json_path.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    summary = build_yolo_from_label_index(
        index_file=index_file,
        output_dir=out_dir,
        dataset_name="ds",
        val_size=0.0,
        test_size=0.0,
        link_mode="copy",
        task="auto",
        overwrite=True,
        keep_staging=False,
    )
    assert summary["status"] == "ok"
    dataset_dir = Path(summary["dataset_dir"])
    data_yaml = (dataset_dir / "data.yaml").read_text(encoding="utf-8")
    assert "kpt_shape" in data_yaml


def test_build_yolo_from_index_mixed_shapes_can_force_pose(tmp_path):
    folder = tmp_path / "src"
    json_path, img_path = _write_labelme_pair(folder, stem="clip_000000000")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["shapes"].append(
        {
            "label": "animal_nose",
            "points": [[5, 5]],
            "group_id": None,
            "shape_type": "point",
            "flags": {},
        }
    )
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    index_file = tmp_path / "annolid_dataset.jsonl"
    index_file.write_text(
        json.dumps(
            {
                "record_version": 1,
                "image_path": str(img_path.resolve()),
                "json_path": str(json_path.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    summary = build_yolo_from_label_index(
        index_file=index_file,
        output_dir=out_dir,
        dataset_name="ds",
        val_size=0.0,
        test_size=0.0,
        link_mode="copy",
        task="segmentation",
        overwrite=True,
        keep_staging=False,
    )
    assert summary["status"] == "ok"
    dataset_dir = Path(summary["dataset_dir"])
    data_yaml = (dataset_dir / "data.yaml").read_text(encoding="utf-8")
    assert "kpt_shape" not in data_yaml


def test_build_yolo_from_index_uses_prefixed_split_folders(tmp_path):
    src_root = tmp_path / "src"
    train_json, train_img = _write_labelme_pair(
        src_root / "train_session01", stem="train_000000000"
    )
    val_json, val_img = _write_labelme_pair(
        src_root / "validation_session01", stem="val_000000000"
    )
    test_json, test_img = _write_labelme_pair(
        src_root / "test_session01", stem="test_000000000"
    )

    index_file = tmp_path / "annolid_dataset.jsonl"
    records = [
        {
            "record_version": 1,
            "image_path": str(train_img.resolve()),
            "json_path": str(train_json.resolve()),
        },
        {
            "record_version": 1,
            "image_path": str(val_img.resolve()),
            "json_path": str(val_json.resolve()),
        },
        {
            "record_version": 1,
            "image_path": str(test_img.resolve()),
            "json_path": str(test_json.resolve()),
        },
    ]
    index_file.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    summary = build_yolo_from_label_index(
        index_file=index_file,
        output_dir=tmp_path / "out",
        dataset_name="ds",
        val_size=0.0,
        test_size=0.0,
        link_mode="copy",
        overwrite=True,
        keep_staging=False,
    )

    assert summary["status"] == "ok"
    dataset_dir = Path(summary["dataset_dir"])
    assert len(list((dataset_dir / "labels" / "train").glob("*.txt"))) == 1
    assert len(list((dataset_dir / "labels" / "val").glob("*.txt"))) == 1
    assert len(list((dataset_dir / "labels" / "test").glob("*.txt"))) == 1
