import json
from pathlib import Path

from annolid.behavior import polygon_classifier_cli
from annolid.behavior.polygon_classifier_workflow import PolygonTrainingOutcome
from annolid.engine.cli import main as annolid_run


def test_annolid_run_trains_polygon_classifier_from_existing_csvs_with_tcn_defaults(
    tmp_path: Path, monkeypatch, capsys
):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    train_csv.write_text(
        "video,frame,frame_number,label,fly2_features\n", encoding="utf-8"
    )
    test_csv.write_text(
        "video,frame,frame_number,label,fly2_features\n", encoding="utf-8"
    )
    calls = {}

    def fake_train_polygon_classifier(**kwargs):
        calls.update(kwargs)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        checkpoint = run_dir / "polygon_tcn_classifier_best.pt"
        metrics = run_dir / "metrics.json"
        checkpoint.write_text("checkpoint", encoding="utf-8")
        metrics.write_text("{}", encoding="utf-8")
        return PolygonTrainingOutcome(
            run_dir=str(run_dir),
            checkpoint_path=str(checkpoint),
            metrics_path=str(metrics),
            labels=("background",),
            model_type=kwargs["model_type"],
        )

    monkeypatch.setattr(
        polygon_classifier_cli,
        "train_polygon_classifier",
        fake_train_polygon_classifier,
    )

    rc = annolid_run(
        [
            "train",
            "polygon_classifier",
            "--train-csv",
            str(train_csv),
            "--test-csv",
            str(test_csv),
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert rc == 0
    assert calls["model_type"] == "tcn"
    assert calls["num_epochs"] == 500
    assert calls["learning_rate"] == 1e-4
    assert calls["batch_size"] == 8
    assert calls["window_size"] == 1000
    assert calls["hidden_dim"] == 32
    assert calls["num_residual_blocks"] == 2
    assert calls["kernel_size"] == 9
    assert calls["dropout"] == 0.1
    payload = json.loads(capsys.readouterr().out)
    assert payload["parameters"]["model_type"] == "tcn"
    assert payload["training"]["checkpoint_path"].endswith(
        "polygon_tcn_classifier_best.pt"
    )


def test_annolid_run_polygon_classifier_generates_csvs_from_video_assignments(
    tmp_path: Path, monkeypatch, capsys
):
    train_video = tmp_path / "train_video"
    test_video = tmp_path / "test_video"
    train_video.mkdir()
    test_video.mkdir()
    label_payload = (
        '{"shapes":[{"label":"fly2","shape_type":"polygon",'
        '"points":[[0,0],[1,0],[1,1],[0,1]]}],"imagePath":"frame_000000000.png"}'
    )
    (train_video / "frame_000000000.json").write_text(label_payload, encoding="utf-8")
    (test_video / "frame_000000000.json").write_text(label_payload, encoding="utf-8")
    train_labels = tmp_path / "train_labels.csv"
    test_labels = tmp_path / "test_labels.csv"
    train_labels.write_text("frame,background\n0,1\n", encoding="utf-8")
    test_labels.write_text("frame,background\n0,1\n", encoding="utf-8")

    def fake_train_polygon_classifier(**kwargs):
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        return PolygonTrainingOutcome(
            run_dir=str(run_dir),
            checkpoint_path=str(run_dir / "checkpoint.pt"),
            metrics_path=str(run_dir / "metrics.json"),
            labels=("background",),
            model_type=kwargs["model_type"],
        )

    monkeypatch.setattr(
        polygon_classifier_cli,
        "train_polygon_classifier",
        fake_train_polygon_classifier,
    )

    rc = annolid_run(
        [
            "train",
            "polygon_classifier",
            "--train-video",
            str(train_video),
            str(train_labels),
            "--test-video",
            str(test_video),
            str(test_labels),
            "--csv-output-dir",
            str(tmp_path / "csvs"),
            "--output-dir",
            str(tmp_path / "runs"),
            "--num-epochs",
            "1",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"]["train"]["rows"] == 1
    assert payload["dataset"]["test"]["rows"] == 1
    assert Path(payload["dataset"]["train"]["csv"]).is_file()
