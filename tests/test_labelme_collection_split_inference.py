import json
from pathlib import Path

from annolid.datasets.labelme_collection import generate_labelme_spec_and_splits


def _write_labelme_pair(folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / f"{stem}.png"
    json_path = folder / f"{stem}.json"
    image_path.write_bytes(b"fake-png")
    json_path.write_text(
        json.dumps(
            {
                "version": "5.5.0",
                "flags": {},
                "shapes": [
                    {
                        "label": "mouse",
                        "points": [[0, 0], [1, 0], [1, 1]],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": image_path.name,
                "imageHeight": 10,
                "imageWidth": 10,
            }
        ),
        encoding="utf-8",
    )


def test_generate_labelme_spec_uses_train_val_test_prefixed_folders(tmp_path: Path):
    source_root = tmp_path / "labeled-data"
    _write_labelme_pair(source_root / "train_session01", "train_a")
    _write_labelme_pair(source_root / "val_session01", "val_a")
    _write_labelme_pair(source_root / "test_session01", "test_a")

    result = generate_labelme_spec_and_splits(
        sources=[source_root],
        dataset_root=tmp_path / "dataset",
        val_size=0.0,  # Should be ignored when split folders are inferred.
        test_size=0.0,  # Should be ignored when split folders are inferred.
        keypoint_names=["nose"],
        infer_flip_idx=False,
        source="test",
    )

    assert result["split_counts"]["train"] == 1
    assert result["split_counts"]["val"] == 1
    assert result["split_counts"]["test"] == 1
    assert result["val_index"] is not None
    assert result["test_index"] is not None
