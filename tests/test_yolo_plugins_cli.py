from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml
from PIL import Image

from annolid.engine.models.yolo_labelme import YOLOLabelMePlugin
from annolid.engine.models.yolo_ultralytics import YOLOUltralyticsPlugin


def _create_mini_yolo_dataset(root: Path) -> Path:
    train_imgs = root / "images" / "train"
    val_imgs = root / "images" / "val"
    train_imgs.mkdir(parents=True)
    val_imgs.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(train_imgs / "a.jpg")
    Image.new("RGB", (32, 32)).save(val_imgs / "b.jpg")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "names": ["mouse"],
            }
        ),
        encoding="utf-8",
    )
    return yaml_path


def test_yolo_ultralytics_plugin_train_applies_preset_and_cli_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    ds_root = tmp_path / "dataset"
    _create_mini_yolo_dataset(ds_root)
    out_project = tmp_path / "runs"

    captured_cmd: list[str] = []

    def _fake_run_yolo_cli(cmd, output_sink=None, **_kwargs):
        captured_cmd.extend(cmd)
        # Simulate Ultralytics creating weights/best.pt
        ckpt = out_project / "exp_small" / "weights" / "best.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"pt")
        return SimpleNamespace(returncode=0, command=cmd, output_tail=[])

    monkeypatch.setattr(
        "annolid.yolo.ultralytics_cli.locate_yolo_cli", lambda: ["yolo"]
    )
    monkeypatch.setattr("annolid.yolo.ultralytics_cli.run_yolo_cli", _fake_run_yolo_cli)

    plugin = YOLOUltralyticsPlugin()
    parser = argparse.ArgumentParser()
    plugin.add_train_args(parser)
    args = parser.parse_args(
        [
            "--data",
            str(ds_root),
            "--weights",
            "yolo11s-seg.pt",
            "--preset",
            "small_dataset",
            "--epochs",
            "45",
            "--project",
            str(out_project),
            "--name",
            "exp_small",
            "--resume",
        ]
    )

    rc = plugin.train(args)
    assert rc == 0
    assert "epochs=45" in captured_cmd
    # From small_dataset preset:
    assert "freeze=10" in captured_cmd
    assert "patience=40" in captured_cmd
    assert "optimizer=AdamW" in captured_cmd
    assert "lr0=0.002" in captured_cmd
    assert "cos_lr=True" in captured_cmd
    assert "resume=True" in captured_cmd
    assert f"project={out_project.resolve()}" in captured_cmd
    assert "name=exp_small" in captured_cmd


def test_yolo_ultralytics_plugin_predict_streams_video_frames(
    tmp_path: Path, monkeypatch
) -> None:
    calls = {}

    class _FakeResult:
        def __init__(self, n_boxes: int, save_dir: str):
            self.boxes = list(range(n_boxes))
            self.save_dir = save_dir

    class _FakeYOLO:
        def __init__(self, weight_path: str):
            calls["weights"] = weight_path

        def track(self, **kwargs):
            calls["method"] = "track"
            calls["kwargs"] = kwargs
            yield _FakeResult(2, str(tmp_path / "out"))
            yield _FakeResult(3, str(tmp_path / "out"))

        def predict(self, **kwargs):
            calls["method"] = "predict"
            calls["kwargs"] = kwargs
            yield _FakeResult(1, str(tmp_path / "out"))

    monkeypatch.setattr("annolid.yolo.configure_ultralytics_cache", lambda: None)
    monkeypatch.setattr(
        "annolid.yolo.import_ultralytics_symbol", lambda name, purpose="": _FakeYOLO
    )
    monkeypatch.setattr("annolid.yolo.resolve_weight_path", lambda w: Path(w))

    plugin = YOLOUltralyticsPlugin()
    parser = argparse.ArgumentParser()
    plugin.add_predict_args(parser)
    args = parser.parse_args(
        [
            "--weights",
            "/tmp/best.pt",
            "--source",
            "/tmp/video.mp4",
            "--track",
            "--tracker",
            "botsort.yaml",
            "--conf",
            "0.35",
            "--iou",
            "0.6",
            "--imgsz",
            "960",
            "--max-det",
            "50",
            "--vid-stride",
            "2",
            "--classes",
            "0,2",
            "--device",
            "cpu",
        ]
    )

    rc = plugin.predict(args)
    assert rc == 0
    assert calls["method"] == "track"
    assert calls["kwargs"]["stream"] is True
    assert calls["kwargs"]["persist"] is True
    assert calls["kwargs"]["tracker"] == "botsort.yaml"
    assert calls["kwargs"]["conf"] == 0.35
    assert calls["kwargs"]["iou"] == 0.6
    assert calls["kwargs"]["imgsz"] == 960
    assert calls["kwargs"]["max_det"] == 50
    assert calls["kwargs"]["vid_stride"] == 2
    assert calls["kwargs"]["classes"] == [0, 2]
    assert calls["kwargs"]["device"] == "cpu"


def test_yolo_labelme_plugin_forwards_inference_flags_and_exports_csv(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    class _FakeProcessor:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def run_inference(self, **kwargs):
            captured["run"] = kwargs
            return "yolo#5"

    csv_calls = []

    def _fake_convert_json_to_csv(
        json_folder, csv_file=None, tracked_csv_file=None, **_kwargs
    ):
        csv_calls.append((json_folder, csv_file, tracked_csv_file))

    monkeypatch.setattr("annolid.segmentation.yolos.InferenceProcessor", _FakeProcessor)
    monkeypatch.setattr(
        "annolid.annotation.labelme2csv.convert_json_to_csv", _fake_convert_json_to_csv
    )

    out_dir = tmp_path / "video_out"
    preview_mp4 = tmp_path / "preview.mp4"

    plugin = YOLOLabelMePlugin()
    parser = argparse.ArgumentParser()
    plugin.add_predict_args(parser)
    args = parser.parse_args(
        [
            "--weights",
            "yolo11s-seg.pt",
            "--source",
            str(tmp_path / "video.mp4"),
            "--output-dir",
            str(out_dir),
            "--conf",
            "0.4",
            "--iou",
            "0.55",
            "--imgsz",
            "800",
            "--max-det",
            "25",
            "--device",
            "cpu",
            "--half",
            "--no-tracking",
            "--batch",
            "4",
            "--skip-existing",
            "--save-video",
            str(preview_mp4),
            "--save-csv",
        ]
    )

    rc = plugin.predict(args)
    assert rc == 0
    assert captured["init"]["conf"] == 0.4
    assert captured["init"]["iou"] == 0.55
    assert captured["init"]["imgsz"] == 800
    assert captured["init"]["max_det"] == 25
    assert captured["init"]["device"] == "cpu"
    assert captured["init"]["half"] is True

    assert captured["run"]["enable_tracking"] is False
    assert captured["run"]["batch_size"] == 4
    assert captured["run"]["skip_existing"] is True
    assert captured["run"]["save_video_path"] == preview_mp4.resolve()

    assert len(csv_calls) == 1
    assert csv_calls[0][0] == str(out_dir.resolve())
    assert csv_calls[0][1].endswith("_tracking.csv")
    assert csv_calls[0][2].endswith("_tracked.csv")
