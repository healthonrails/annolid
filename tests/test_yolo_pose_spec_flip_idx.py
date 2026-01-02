from pathlib import Path

from annolid.segmentation.dino_kpseg.data import load_yolo_pose_spec


def test_load_yolo_pose_spec_infers_flip_idx_for_left_right(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [4, 2]",
                "kpt_labels:",
                "  0: leftear",
                "  1: nose",
                "  2: rightear",
                "  3: tailbase",
                "",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_yolo_pose_spec(data_yaml)
    assert spec.keypoint_names == ["leftear", "nose", "rightear", "tailbase"]
    assert spec.flip_idx == [2, 1, 0, 3]
