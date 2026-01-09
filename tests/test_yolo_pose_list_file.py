from pathlib import Path

from PIL import Image

from annolid.segmentation.dino_kpseg.data import load_yolo_pose_spec


def test_load_yolo_pose_spec_supports_list_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    img_dir = root / "images" / "train"
    img_dir.mkdir(parents=True)

    img_path = img_dir / "frame_0001.jpg"
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(img_path)

    train_list = root / "train.txt"
    train_list.write_text("images/train/frame_0001.jpg\n", encoding="utf-8")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                f"train: {train_list.name}",
                "val: images/train",
                "kpt_shape: [2, 3]",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_yolo_pose_spec(data_yaml)
    assert len(spec.train_images) == 1
    assert spec.train_images[0].resolve() == img_path.resolve()
