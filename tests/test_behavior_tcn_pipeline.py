from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from annolid.behavior.tcn import (
    BehaviorTCN,
    TCNFeatureConfig,
    TCNModelConfig,
    TCNSequenceDataset,
    TCNTrainingConfig,
    TCNRunConfig,
    TCNSession,
    add_velocity_features,
    evaluate_tcn,
    fit_normalization,
    load_tcn_checkpoint,
    read_feature_csv,
    save_tcn_checkpoint,
    train_tcn,
)
from annolid.engine.registry import get_model


def _write_feature_csv(path: Path, n_frames: int = 40) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y"])
        for idx in range(n_frames):
            writer.writerow([idx, float(idx), float(idx % 5)])


def _write_label_csv(path: Path, n_frames: int = 40) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "background", "walk", "groom"])
        for idx in range(n_frames):
            if idx < 10:
                row = [1, 0, 0]
            elif idx < 25:
                row = [0, 1, 0]
            else:
                row = [0, 0, 1]
            writer.writerow([idx, *row])


def _session(tmp_path: Path, name: str, split: str = "train") -> TCNSession:
    features = tmp_path / f"{name}_features.csv"
    labels = tmp_path / f"{name}_labels.csv"
    _write_feature_csv(features)
    _write_label_csv(labels)
    return TCNSession(name, features=features, labels=labels, split=split)


def test_feature_loader_and_velocity(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    _write_feature_csv(features, n_frames=5)

    values = read_feature_csv(features, input_type="features")
    assert values.shape == (5, 2)

    values_with_velocity = add_velocity_features(values)
    assert values_with_velocity.shape == (5, 4)
    np.testing.assert_allclose(values_with_velocity[0, 2:], 0.0)
    np.testing.assert_allclose(values_with_velocity[1, 2:], values[1] - values[0])


def test_tcn_training_evaluation_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    train_session = _session(tmp_path, "train", split="train")
    test_session = _session(tmp_path, "test", split="test")
    feature_config = TCNFeatureConfig(add_velocity=True)
    normalization = fit_normalization([train_session], feature_config=feature_config)
    train_dataset = TCNSequenceDataset(
        [train_session],
        feature_config=feature_config,
        label_names=["background", "walk", "groom"],
        sequence_length=20,
        normalization=normalization,
    )
    test_dataset = TCNSequenceDataset(
        [test_session],
        feature_config=feature_config,
        label_names=train_dataset.label_names,
        sequence_length=20,
        normalization=normalization,
    )
    model_config = TCNModelConfig(
        hidden_dim=8, num_blocks=1, kernel_size=3, dropout=0.0
    )
    model = BehaviorTCN(
        input_dim=train_dataset.input_dim,
        num_classes=len(train_dataset.label_names),
        config=model_config,
    )
    history = train_tcn(
        model,
        train_dataset,
        config=TCNTrainingConfig(
            epochs=2,
            batch_size=2,
            sequence_length=20,
            device="cpu",
        ),
    )
    assert len(history) == 2
    metrics = evaluate_tcn(model, test_dataset, device="cpu")
    assert metrics["n_labeled_frames"] == 30
    assert set(metrics["per_class"]) == {"walk", "groom"}

    run_config = TCNRunConfig(
        sessions=[train_session, test_session],
        labels=train_dataset.label_names,
        feature=feature_config,
        model=model_config,
        training=TCNTrainingConfig(epochs=2, sequence_length=20, device="cpu"),
    )
    checkpoint = tmp_path / "checkpoint.pt"
    save_tcn_checkpoint(
        checkpoint,
        model=model,
        run_config=run_config,
        normalization=normalization,
        input_dim=train_dataset.input_dim,
        label_names=train_dataset.label_names,
    )
    loaded, payload = load_tcn_checkpoint(checkpoint, device="cpu")
    assert isinstance(loaded, BehaviorTCN)
    assert payload["label_names"] == ["background", "walk", "groom"]


def test_tcn_training_reenables_grad_mode(tmp_path: Path) -> None:
    train_session = _session(tmp_path, "train", split="train")
    feature_config = TCNFeatureConfig(add_velocity=True)
    normalization = fit_normalization([train_session], feature_config=feature_config)
    train_dataset = TCNSequenceDataset(
        [train_session],
        feature_config=feature_config,
        label_names=["background", "walk", "groom"],
        sequence_length=20,
        normalization=normalization,
    )
    model = BehaviorTCN(
        input_dim=train_dataset.input_dim,
        num_classes=len(train_dataset.label_names),
        config=TCNModelConfig(hidden_dim=8, num_blocks=1, kernel_size=3, dropout=0.0),
    )

    torch.set_grad_enabled(False)
    try:
        history = train_tcn(
            model,
            train_dataset,
            config=TCNTrainingConfig(
                epochs=1,
                batch_size=2,
                sequence_length=20,
                device="cpu",
            ),
        )
    finally:
        torch.set_grad_enabled(True)

    assert len(history) == 1
    assert np.isfinite(history[0]["loss"])


def test_tcn_behavior_engine_plugin_train_predict(tmp_path: Path) -> None:
    train_session = _session(tmp_path, "train", split="train")
    test_session = _session(tmp_path, "test", split="test")
    config_path = tmp_path / "tcn_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "labels": ["background", "walk", "groom"],
                "sessions": [
                    {
                        "id": "train",
                        "features": str(train_session.features),
                        "labels": str(train_session.labels),
                        "split": "train",
                    },
                    {
                        "id": "test",
                        "features": str(test_session.features),
                        "labels": str(test_session.labels),
                        "split": "test",
                    },
                ],
                "feature": {"add_velocity": True},
                "model": {
                    "hidden_dim": 8,
                    "num_blocks": 1,
                    "kernel_size": 3,
                    "dropout": 0.0,
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "sequence_length": 20,
                    "device": "cpu",
                },
            }
        ),
        encoding="utf-8",
    )

    plugin = get_model("tcn_behavior")
    out_dir = tmp_path / "run"
    assert (
        plugin.train(
            type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "output_dir": str(out_dir),
                    "checkpoint_name": "model.pt",
                    "device": "cpu",
                    "epochs": 1,
                },
            )()
        )
        == 0
    )
    checkpoint = out_dir / "model.pt"
    assert checkpoint.exists()
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "test_metrics" in payload

    predictions = tmp_path / "predictions.csv"
    assert (
        plugin.predict(
            type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "checkpoint_path": str(checkpoint),
                    "output_csv": str(predictions),
                    "metrics_json": str(tmp_path / "predict_metrics.json"),
                    "split": "test",
                    "device": "cpu",
                },
            )()
        )
        == 0
    )
    assert predictions.exists()
    assert (
        predictions.read_text(encoding="utf-8")
        .splitlines()[0]
        .startswith("session_id,frame,predicted_index")
    )
