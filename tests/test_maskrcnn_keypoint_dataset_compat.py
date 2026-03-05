import json

import torch
from PIL import Image

from annolid.segmentation.maskrcnn.coco_dataset import AnnolidCocoKeypointDataset


def test_keypoint_dataset_returns_n_k_3_tensor(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    Image.new("RGB", (32, 24), color=(0, 0, 0)).save(train_dir / "frame_0001.png")

    ann = {
        "images": [{"id": 1, "file_name": "frame_0001.png", "width": 32, "height": 24}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4.0, 5.0, 10.0, 8.0],
                "area": 80.0,
                "iscrowd": 0,
                "keypoints": [
                    8.0,
                    10.0,
                    2.0,
                    9.0,
                    11.0,
                    2.0,
                    10.0,
                    12.0,
                    1.0,
                    11.0,
                    13.0,
                    0.0,
                ],
                "num_keypoints": 3,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "mouse",
                "keypoints": ["nose", "neck", "tail_base", "tail_tip"],
                "skeleton": [[1, 2], [2, 3], [3, 4]],
            }
        ],
    }
    ann_path = train_dir / "annotations.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")

    ds = AnnolidCocoKeypointDataset(
        root=train_dir,
        annotations_json=ann_path,
        transforms=None,
    )

    image, target = ds[0]

    assert isinstance(image, Image.Image)
    assert target["keypoints"].shape == (1, 4, 3)
    assert target["keypoints"].dtype == torch.float32
    assert target["keypoints"][0, 0].tolist() == [8.0, 10.0, 2.0]
