"""Run Annolid TCN reproduction experiments on the local DAART paper datasets."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from annolid.behavior.tcn import (
    BehaviorTCN,
    TCNFeatureConfig,
    TCNModelConfig,
    TCNRunConfig,
    TCNSequenceDataset,
    TCNTrainingConfig,
    TCNSession,
    evaluate_tcn,
    fit_normalization,
    save_tcn_checkpoint,
    train_tcn,
)


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "fly_markers": {
        "root": "/path/to/head_fixed_fly",
        "feature_dir": "markers",
        "label_dir": "labels-hand-paper-matched",
        "input_type": "markers",
        "labels": [
            "background",
            "still",
            "walk",
            "front_groom",
            "back_groom",
            "abdomen-move",
        ],
        "train_ids": [
            "2019_08_07_fly2",
            "2019_08_08_fly1",
            "2019_08_20_fly2",
            "2019_10_10_fly3",
            "2019_10_14_fly3",
        ],
        "test_ids": [
            "2019_06_26_fly2",
            "2019_08_14_fly1",
            "2019_08_20_fly3",
            "2019_10_14_fly2",
            "2019_10_21_fly1",
        ],
    },
    "fly_posvel": {
        "root": "/path/to/head_fixed_fly",
        "feature_dir": "features-posvel",
        "label_dir": "labels-hand-paper-matched",
        "input_type": "features",
        "labels": [
            "background",
            "still",
            "walk",
            "front_groom",
            "back_groom",
            "abdomen-move",
        ],
        "train_ids": [
            "2019_08_07_fly2",
            "2019_08_08_fly1",
            "2019_08_20_fly2",
            "2019_10_10_fly3",
            "2019_10_14_fly3",
        ],
        "test_ids": [
            "2019_06_26_fly2",
            "2019_08_14_fly1",
            "2019_08_20_fly3",
            "2019_10_14_fly2",
            "2019_10_21_fly1",
        ],
    },
    "mouse_features": {
        "root": "/path/to/freely_moving_mouse",
        "feature_dir": "features-sturman",
        "label_dir": "labels-hand",
        "input_type": "features",
        "labels": ["Background", "Supported", "Unsupported", "Grooming"],
        "train_ids": [
            "OFT_5",
            "OFT_6",
            "OFT_11",
            "OFT_12",
            "OFT_14",
            "OFT_15",
            "OFT_16",
            "OFT_23",
            "OFT_24",
            "OFT_38",
        ],
        "test_ids": [
            "OFT_39",
            "OFT_41",
            "OFT_43",
            "OFT_44",
            "OFT_49",
            "OFT_50",
            "OFT_51",
            "OFT_52",
            "OFT_54",
            "OFT_58",
        ],
    },
    "mouse_posvel": {
        "root": "/path/to/freely_moving_mouse",
        "feature_dir": "features-sturman-posvel",
        "label_dir": "labels-hand",
        "input_type": "features",
        "labels": ["Background", "Supported", "Unsupported", "Grooming"],
        "train_ids": [
            "OFT_5",
            "OFT_6",
            "OFT_11",
            "OFT_12",
            "OFT_14",
            "OFT_15",
            "OFT_16",
            "OFT_23",
            "OFT_24",
            "OFT_38",
        ],
        "test_ids": [
            "OFT_39",
            "OFT_41",
            "OFT_43",
            "OFT_44",
            "OFT_49",
            "OFT_50",
            "OFT_51",
            "OFT_52",
            "OFT_54",
            "OFT_58",
        ],
    },
}


def build_sessions(spec: dict[str, Any]) -> list[TCNSession]:
    root = Path(spec["root"]).expanduser()
    sessions: list[TCNSession] = []
    for split, ids in (("train", spec["train_ids"]), ("test", spec["test_ids"])):
        for session_id in ids:
            sessions.append(
                TCNSession(
                    session_id=session_id,
                    features=root / spec["feature_dir"] / f"{session_id}_labeled.csv",
                    labels=root / spec["label_dir"] / f"{session_id}_labels.csv",
                    split=split,
                )
            )
    return sessions


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "macro_f1",
                "accuracy_non_background",
                "n_labeled_eval_frames",
            ]
        )
        for result in results:
            metrics = result["metrics"]
            writer.writerow(
                [
                    result["experiment"],
                    metrics["macro_f1"],
                    metrics["accuracy_non_background"],
                    metrics["n_labeled_frames"],
                ]
            )


def run_one(
    name: str, spec: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    print(f"\n=== {name} ===", flush=True)
    sessions = build_sessions(spec)
    train_sessions = [session for session in sessions if session.split == "train"]
    test_sessions = [session for session in sessions if session.split == "test"]
    feature_config = TCNFeatureConfig(
        input_type=spec["input_type"],
        add_velocity=False,
        zscore=True,
    )
    model_config = TCNModelConfig(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )
    training_config = TCNTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    run_config = TCNRunConfig(
        sessions=sessions,
        labels=list(spec["labels"]),
        feature=feature_config,
        model=model_config,
        training=training_config,
    )
    normalization = fit_normalization(train_sessions, feature_config=feature_config)
    train_dataset = TCNSequenceDataset(
        train_sessions,
        feature_config=feature_config,
        label_names=run_config.labels,
        sequence_length=training_config.sequence_length,
        normalization=normalization,
    )
    test_dataset = TCNSequenceDataset(
        test_sessions,
        feature_config=feature_config,
        label_names=train_dataset.label_names,
        sequence_length=training_config.sequence_length,
        normalization=normalization,
    )
    model = BehaviorTCN(
        input_dim=train_dataset.input_dim,
        num_classes=len(train_dataset.label_names),
        config=model_config,
    )
    history = train_tcn(model, train_dataset, config=training_config)
    metrics = evaluate_tcn(model, test_dataset, device=training_config.device)
    result = {
        "experiment": name,
        "train_ids": list(spec["train_ids"]),
        "test_ids": list(spec["test_ids"]),
        "feature_dir": spec["feature_dir"],
        "label_dir": spec["label_dir"],
        "input_type": spec["input_type"],
        "labels": train_dataset.label_names,
        "feature_config": asdict(feature_config),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "input_dim": train_dataset.input_dim,
        "metrics": metrics,
        "history": history,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_tcn_checkpoint(
        args.out_dir / f"{name}_model.pt",
        model=model,
        run_config=run_config,
        normalization=normalization,
        input_dim=train_dataset.input_dim,
        label_names=train_dataset.label_names,
    )
    with (args.out_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        f"{name}: macro_f1={metrics['macro_f1']:.4f}, "
        f"accuracy_non_background={metrics['accuracy_non_background']:.4f}, "
        f"n={metrics['n_labeled_frames']}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reproduction_results/annolid_tcn"),
    )
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    args.out_dir = args.out_dir.expanduser().resolve()
    print(f"device={args.device}", flush=True)

    results = []
    for name in args.experiments:
        results.append(run_one(name, EXPERIMENTS[name], args))
        write_summary(results, args.out_dir)

    print("\nSummary")
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['experiment']}: macro_f1={metrics['macro_f1']:.4f}, "
            f"accuracy_non_background={metrics['accuracy_non_background']:.4f}, "
            f"n={metrics['n_labeled_frames']}"
        )


if __name__ == "__main__":
    main()
