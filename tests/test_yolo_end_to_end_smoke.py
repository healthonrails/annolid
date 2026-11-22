from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from annolid.engine.models.yolo_labelme import YOLOLabelMePlugin
from annolid.engine.models.yolo_ultralytics import YOLOUltralyticsPlugin
from annolid.yolo import discover_yolo_checkpoints


def _build_coco_seg_dataset(root: Path) -> Path:
    images_dir = root / "images"
    ann_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    for idx, name in enumerate(("train_0.png", "train_1.png", "val_0.png")):
        arr = np.full((64, 64, 3), 40 + idx * 30, dtype=np.uint8)
        arr[16:48, 16:48] = (200, 180, 120)
        Image.fromarray(arr).save(images_dir / name)

    train_coco = {
        "images": [
            {"id": 1, "file_name": "train_0.png", "width": 64, "height": 64},
            {"id": 2, "file_name": "train_1.png", "width": 64, "height": 64},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [16, 16, 32, 32],
                "area": 1024,
                "iscrowd": 0,
                "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [16, 16, 32, 32],
                "area": 1024,
                "iscrowd": 0,
                "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],
            },
        ],
        "categories": [{"id": 1, "name": "mouse"}],
    }
    val_coco = {
        "images": [
            {"id": 3, "file_name": "val_0.png", "width": 64, "height": 64},
        ],
        "annotations": [
            {
                "id": 3,
                "image_id": 3,
                "category_id": 1,
                "bbox": [16, 16, 32, 32],
                "area": 1024,
                "iscrowd": 0,
                "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],
            }
        ],
        "categories": [{"id": 1, "name": "mouse"}],
    }
    (ann_dir / "instances_train.json").write_text(
        json.dumps(train_coco), encoding="utf-8"
    )
    (ann_dir / "instances_val.json").write_text(json.dumps(val_coco), encoding="utf-8")
    return root


def _build_synthetic_video(path: Path, frames: int = 4) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 64),
    )
    for i in range(frames):
        frame = np.full((64, 64, 3), 50 + i * 20, dtype=np.uint8)
        frame[16:48, 16:48] = (210, 180, 120)
        writer.write(frame)
    writer.release()
    return path


def test_offline_yolo_seg_coco_train_discover_and_infer_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real offline Ultralytics smoke test: COCO seg -> yolo train -> checkpoint discovery -> predict."""
    pytest.importorskip("torch")
    ultralytics = pytest.importorskip("ultralytics")

    cfg_dir = tmp_path / "ultralytics_cfg"
    cfg_dir.mkdir()
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("ANNOLID_RUNS_ROOT", str(tmp_path / "runs"))

    # Create a tiny local segmentation checkpoint from Ultralytics YAML (no network download).
    seed_weights = tmp_path / "seed_seg.pt"
    ultralytics.YOLO("yolo11n-seg.yaml").save(str(seed_weights))
    assert seed_weights.is_file()

    coco_dir = _build_coco_seg_dataset(tmp_path / "coco_seg_ds")

    # 1. Real `yolo train` CLI execution via YOLOUltralyticsPlugin on COCO segmentation folder
    train_plugin = YOLOUltralyticsPlugin()
    train_parser = argparse.ArgumentParser()
    train_plugin.add_train_args(train_parser)
    train_args = train_parser.parse_args(
        [
            "--data",
            str(coco_dir),
            "--weights",
            str(seed_weights),
            "--epochs",
            "1",
            "--imgsz",
            "64",
            "--batch",
            "2",
            "--device",
            "cpu",
            "--workers",
            "0",
            "--override",
            "val=False",
            "--override",
            "amp=False",
        ]
    )
    assert train_plugin.train(train_args) == 0

    # 2. Newly trained checkpoint is automatically discovered under ANNOLID_RUNS_ROOT
    discovered = discover_yolo_checkpoints(
        runs_root=tmp_path / "runs", include_last=True
    )
    assert discovered, "Expected trained YOLO checkpoint to be discovered on disk."
    trained_ckpt = discovered[0].path
    assert trained_ckpt.is_file()

    # 3. Real streaming Ultralytics prediction on a synthetic video
    video_path = _build_synthetic_video(tmp_path / "clip.mp4", frames=4)
    pred_parser = argparse.ArgumentParser()
    train_plugin.add_predict_args(pred_parser)
    pred_args = pred_parser.parse_args(
        [
            "--weights",
            str(trained_ckpt),
            "--source",
            str(video_path),
            "--device",
            "cpu",
            "--imgsz",
            "64",
            "--conf",
            "0.25",
        ]
    )
    assert train_plugin.predict(pred_args) == 0

    # 4. Real batched LabelMe + CSV inference via YOLOLabelMePlugin
    labelme_out = tmp_path / "clip_labelme"
    preview_mp4 = tmp_path / "clip_preview.mp4"
    lm_plugin = YOLOLabelMePlugin()
    lm_parser = argparse.ArgumentParser()
    lm_plugin.add_predict_args(lm_parser)
    lm_args = lm_parser.parse_args(
        [
            "--weights",
            str(trained_ckpt),
            "--source",
            str(video_path),
            "--output-dir",
            str(labelme_out),
            "--device",
            "cpu",
            "--imgsz",
            "64",
            "--conf",
            "0.25",
            "--no-tracking",
            "--batch",
            "2",
            "--save-video",
            str(preview_mp4),
            "--save-csv",
        ]
    )
    assert lm_plugin.predict(lm_args) == 0
    assert len(list(labelme_out.glob("*.json"))) == 4
    assert preview_mp4.is_file()
    assert (tmp_path / "clip_labelme_tracking.csv").is_file()
