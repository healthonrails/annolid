import json
from pathlib import Path

from PIL import Image

from annolid.datasets.importers.deeplabcut_training_data import (
    DeepLabCutTrainingImportConfig,
    import_deeplabcut_training_data,
)


def _write_collected_data_csv(csv_path: Path, *, folder: str, image_name: str) -> None:
    # DeepLabCut training CSVs are typically 3 header rows: scorer/bodyparts/coords.
    rows = [
        "scorer,,,hyn,hyn",
        "bodyparts,,,nose,nose",
        "coords,,,x,y",
        f"labeled-data,{folder},{image_name},10.0,20.0",
    ]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_import_deeplabcut_training_data_writes_labelme_json(tmp_path: Path) -> None:
    dataset = tmp_path / "dlc_dataset"
    labeled = dataset / "labeled-data"
    seg = labeled / "seg-1"
    seg.mkdir(parents=True)

    image = seg / "img00001.png"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(image)

    _write_collected_data_csv(
        seg / "CollectedData_test.csv", folder="seg-1", image_name="img00001.png"
    )

    summary = import_deeplabcut_training_data(
        DeepLabCutTrainingImportConfig(
            source_dir=dataset,
            labeled_data_root=Path("labeled-data"),
            instance_label="mouse",
            overwrite=True,
            recursive=True,
        ),
        write_pose_schema=True,
        pose_schema_out=Path("labeled-data/pose_schema.json"),
    )
    assert summary["json_written"] == 1

    json_path = image.with_suffix(".json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["imagePath"] == "img00001.png"
    assert payload["imageWidth"] == 100
    assert payload["imageHeight"] == 80
    assert payload["flags"] == {}
    assert payload["instance_label"] == "mouse"
    assert len(payload["shapes"]) == 1
    assert payload["shapes"][0]["shape_type"] == "point"
    assert payload["shapes"][0]["label"] == "mouse_nose"

    schema_path = labeled / "pose_schema.json"
    assert schema_path.exists()


def test_import_deeplabcut_training_data_can_write_pose_schema_without_overwrite(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dlc_dataset"
    labeled = dataset / "labeled-data"
    seg = labeled / "seg-1"
    seg.mkdir(parents=True)

    image = seg / "img00001.png"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(image)

    # Existing JSON should prevent overwriting, but schema should still be derived.
    image.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": "5.5.0",
                "flags": {},
                "shapes": [],
                "imagePath": image.name,
                "imageData": None,
                "imageHeight": 80,
                "imageWidth": 100,
            }
        ),
        encoding="utf-8",
    )

    _write_collected_data_csv(
        seg / "CollectedData_test.csv", folder="seg-1", image_name="img00001.png"
    )

    summary = import_deeplabcut_training_data(
        DeepLabCutTrainingImportConfig(
            source_dir=dataset,
            labeled_data_root=Path("labeled-data"),
            instance_label="mouse",
            overwrite=False,
            recursive=True,
        ),
        write_pose_schema=True,
        pose_schema_out=Path("labeled-data/pose_schema.json"),
    )
    assert summary["json_written"] == 0
    schema_path = labeled / "pose_schema.json"
    assert schema_path.exists()
