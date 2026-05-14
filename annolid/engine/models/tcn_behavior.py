from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from annolid.engine.registry import ModelPluginBase, register_model


def _read_config(path: str | Path):
    import yaml

    from annolid.behavior.tcn import TCNRunConfig

    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return TCNRunConfig.from_mapping(payload)


def _split_sessions(config: Any, split: str):
    wanted = str(split).strip().lower()
    return [s for s in config.sessions if str(s.split).strip().lower() == wanted]


def _normalization_from_payload(payload: dict[str, Any] | None) -> Any:
    if not payload:
        return None
    import numpy as np

    from annolid.behavior.tcn import TCNNormalization

    return TCNNormalization(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def _write_prediction_csv(
    path: Path, predictions: dict[str, Any], label_names: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "frame", "predicted_index", "predicted_label"])
        for session_id, values in predictions.items():
            for frame_idx, class_idx in enumerate(values.tolist()):
                writer.writerow(
                    [
                        session_id,
                        int(frame_idx),
                        int(class_idx),
                        label_names[int(class_idx)],
                    ]
                )


@register_model
class TCNBehaviorPlugin(ModelPluginBase):
    name = "tcn_behavior"
    description = (
        "Frame-level TCN behavior classifier for pose or feature CSV time series."
    )
    train_help_sections = (
        ("Required inputs", ("--config",)),
        ("Outputs and run location", ("--output-dir", "--checkpoint-name")),
        ("Runtime", ("--device", "--epochs")),
    )
    predict_help_sections = (
        ("Required inputs", ("--config", "--checkpoint-path")),
        ("Outputs", ("--output-csv", "--metrics-json")),
        ("Runtime", ("--device",)),
    )

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--config",
            required=True,
            help="YAML config with sessions, labels, feature options, model, and training settings.",
        )
        parser.add_argument("--output-dir", default="runs/tcn_behavior")
        parser.add_argument("--checkpoint-name", default="best_model.pt")
        parser.add_argument(
            "--device",
            default=None,
            help="Override config training.device. Use cuda, mps, cpu, or auto.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="Override config training.epochs.",
        )

    def train(self, args: argparse.Namespace) -> int:
        from annolid.behavior.tcn import (
            BehaviorTCN,
            TCNSequenceDataset,
            evaluate_tcn,
            fit_normalization,
            save_tcn_checkpoint,
            train_tcn,
        )

        config = _read_config(args.config)
        train_sessions = _split_sessions(config, "train")
        test_sessions = _split_sessions(config, "test")
        if not train_sessions:
            raise ValueError(
                "TCN behavior config needs at least one session with split: train"
            )

        training = config.training
        if args.device:
            training = type(training)(
                **{**training.__dict__, "device": str(args.device)}
            )
        if args.epochs is not None:
            training = type(training)(
                **{**training.__dict__, "epochs": int(args.epochs)}
            )
        config.training = training

        normalization = (
            fit_normalization(train_sessions, feature_config=config.feature)
            if config.feature.zscore
            else None
        )
        train_dataset = TCNSequenceDataset(
            train_sessions,
            feature_config=config.feature,
            label_names=config.labels,
            sequence_length=config.training.sequence_length,
            normalization=normalization,
        )
        if not config.labels:
            config.labels = list(train_dataset.label_names)
        model = BehaviorTCN(
            input_dim=train_dataset.input_dim,
            num_classes=len(train_dataset.label_names),
            config=config.model,
        )
        history = train_tcn(model, train_dataset, config=config.training)

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / str(args.checkpoint_name)
        save_tcn_checkpoint(
            checkpoint_path,
            model=model,
            run_config=config,
            normalization=normalization,
            input_dim=train_dataset.input_dim,
            label_names=train_dataset.label_names,
        )

        payload: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "history": history,
            "labels": train_dataset.label_names,
        }
        if test_sessions:
            test_dataset = TCNSequenceDataset(
                test_sessions,
                feature_config=config.feature,
                label_names=train_dataset.label_names,
                sequence_length=config.training.sequence_length,
                normalization=normalization,
            )
            payload["test_metrics"] = evaluate_tcn(
                model,
                test_dataset,
                device=config.training.device,
            )
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(str(checkpoint_path))
        return 0

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--config",
            required=True,
            help="YAML config with sessions to predict. Uses split: test by default.",
        )
        parser.add_argument("--checkpoint-path", required=True)
        parser.add_argument("--output-csv", default="tcn_behavior_predictions.csv")
        parser.add_argument("--metrics-json", default=None)
        parser.add_argument("--split", default="test")
        parser.add_argument("--device", default="auto")

    def predict(self, args: argparse.Namespace) -> int:
        from annolid.behavior.tcn import (
            TCNRunConfig,
            TCNSequenceDataset,
            evaluate_tcn,
            load_tcn_checkpoint,
            predict_tcn,
        )

        config = _read_config(args.config)
        sessions = _split_sessions(config, str(args.split))
        if not sessions:
            sessions = list(config.sessions)
        model, payload = load_tcn_checkpoint(args.checkpoint_path, device=args.device)
        label_names = [str(v) for v in payload["label_names"]]
        normalization = _normalization_from_payload(payload.get("normalization"))
        run_feature = TCNRunConfig.from_mapping(payload["run_config"]).feature
        run_training = TCNRunConfig.from_mapping(payload["run_config"]).training

        dataset = TCNSequenceDataset(
            sessions,
            feature_config=run_feature,
            label_names=label_names,
            sequence_length=run_training.sequence_length,
            normalization=normalization,
            require_labels=False,
        )
        result = predict_tcn(model, dataset, device=args.device)
        output_csv = Path(args.output_csv).expanduser().resolve()
        _write_prediction_csv(output_csv, result["predictions"], label_names)

        if args.metrics_json:
            if any(session.labels is not None for session in sessions):
                metrics = evaluate_tcn(model, dataset, device=args.device)
            else:
                metrics = {
                    "macro_f1": 0.0,
                    "accuracy_non_background": 0.0,
                    "n_labeled_frames": 0,
                    "per_class": {},
                }
            Path(args.metrics_json).expanduser().resolve().write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
        print(str(output_csv))
        return 0
