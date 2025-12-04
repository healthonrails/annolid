#!/usr/bin/env python
"""Training script for polygon frame classifier.

This mirrors the experiment_20250328 setup:
  - Sliding window 1D Conv residual model with optional channel attention.
  - Balanced sampling, weighted CE loss, LR scheduler, early stopping.
  - Deterministic seeding and simple test evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from sklearn import metrics

from annolid.behavior.models.polygon_frame_classifier import (
    ModelConfig,
    PolygonFeatureConfig,
    PolygonFrameDataset,
    TrainingConfig,
    train_polygon_frame_classifier,
)
from annolid.behavior.models.polygon_frame_classifier import ImprovedFrameLabelConvNet
from annolid.behavior.models.polygon_frame_classifier import _collate_fn  # type: ignore
from annolid.utils.logger import logger


@dataclass
class RunConfig:
    train_csv: Path
    test_csv: Path
    output_dir: Path
    log_dir: Path
    tb_log_dir: Path
    seed: int = 42
    log_level: str = "INFO"


def _parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Train polygon frame classifier from CSV features.")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML/JSON config to seed parameters (CLI overrides).")
    parser.add_argument("--train_csv", required=False,
                        default=None, help="Path to training CSV.")
    parser.add_argument("--test_csv", required=False,
                        default=None, help="Path to test CSV.")
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save models and metrics.")
    parser.add_argument("--log_dir", default="logs",
                        help="Directory for log files.")
    parser.add_argument("--tb_log_dir", default="runs",
                        help="Directory for TensorBoard logs (unused placeholder).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")

    parser.add_argument("--window_size", type=int, default=11)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--num_residual_blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_attention", action="store_true", default=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=4e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_patience", type=int, default=15)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sampling_strategy", default="balanced_sampler",
                        choices=["balanced_sampler", "random"])
    parser.add_argument("--log_every", type=int, default=50)

    parser.add_argument("--frame_width", type=int, default=1024)
    parser.add_argument("--frame_height", type=int, default=570)
    parser.add_argument("--polygon_pad_len", type=int, default=None)

    args = parser.parse_args()
    return args, parser


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        if cfg_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh) or {}
        return json.load(fh)


def _collect_cli_overrides(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, value in vars(args).items():
        default = parser.get_default(key)
        if value is None:
            continue
        if value != default:
            overrides[key] = value
    return overrides


def _merge_section(defaults: Dict[str, Any], config_values: Dict[str, Any], cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, val in (config_values or {}).items():
        if key in merged:
            merged[key] = val
    for key, val in cli_overrides.items():
        if key in merged:
            merged[key] = val
    return merged


def _evaluate(
    model: ImprovedFrameLabelConvNet,
    dataset: PolygonFrameDataset,
    device: torch.device,
) -> Dict[str, Any]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate_fn,
    )
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_probs: list[List[float]] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    label_names = [dataset.index_to_label[idx]
                   for idx in range(len(dataset.index_to_label))]
    accuracy = metrics.accuracy_score(all_targets, all_preds)
    macro_f1 = metrics.f1_score(
        all_targets, all_preds, average="macro", zero_division=0)
    per_class = metrics.classification_report(
        all_targets,
        all_preds,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    confusion = metrics.confusion_matrix(
        all_targets, all_preds, labels=list(range(len(label_names)))).tolist()
    # Per-class AP and mAP using probability scores.
    per_class_ap: Dict[str, float] = {}
    try:
        prob_array = np.asarray(all_probs, dtype=float)
        true_array = np.asarray(all_targets, dtype=int)
        ap_scores = []
        for idx, name in enumerate(label_names):
            binary_true = (true_array == idx).astype(int)
            ap = metrics.average_precision_score(
                binary_true, prob_array[:, idx]) if prob_array.size else 0.0
            if np.isnan(ap):
                ap = 0.0
            per_class_ap[name] = float(ap)
            ap_scores.append(ap)
        macro_map = float(np.mean(ap_scores)) if ap_scores else 0.0
    except Exception:
        macro_map = 0.0
        per_class_ap = {}
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_map": macro_map,
        "per_class_ap": per_class_ap,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "labels": label_names,
    }


def _save_checkpoint(state: Dict[str, Any], output_dir: Path, label: str = "best") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = output_dir / f"polygon_frame_classifier_{label}_{ts}.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)
    return ckpt_path


def _dump_run_artifacts(output_dir: Path, run_payload: Dict[str, Any], suffix: str) -> Path:
    path = output_dir / f"run_{suffix}.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_to_builtin(run_payload), fh, sort_keys=False)
    return path


def _log_run_artifacts(run_payload: Dict[str, Any], label: str) -> None:
    logger.info(
        f"{label} configuration/details:\n{json.dumps(_to_builtin(run_payload), indent=2)}")


def _configure_run_logger(log_dir: Path, command: str) -> Path:
    """Add a file handler for this run and log the invoked command."""
    _ensure_dir(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Logging initiated. Log file: {log_file}")
    logger.info(f"Command: {command}")
    return log_file


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.generic,)):  # numpy scalar
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    args, parser = _parse_args()
    cfg_file = _load_config_file(args.config)
    cli_overrides = _collect_cli_overrides(args, parser)

    run_defaults = {
        "train_csv": None,
        "test_csv": None,
        "output_dir": "results",
        "log_dir": "logs",
        "tb_log_dir": "runs",
        "seed": 42,
        "log_level": "INFO",
    }
    feature_defaults = asdict(PolygonFeatureConfig())
    model_defaults = asdict(ModelConfig())
    training_defaults = asdict(TrainingConfig())

    run_params = _merge_section(
        run_defaults, cfg_file.get("run", {}), cli_overrides)
    feature_params = _merge_section(
        feature_defaults, cfg_file.get("feature", {}), cli_overrides)
    model_params = _merge_section(
        model_defaults, cfg_file.get("model", {}), cli_overrides)
    training_params = _merge_section(
        training_defaults, cfg_file.get("training", {}), cli_overrides)

    if not run_params["train_csv"] or not run_params["test_csv"]:
        raise ValueError(
            "train_csv and test_csv must be provided via config or CLI.")

    run_cfg = RunConfig(
        train_csv=Path(run_params["train_csv"]),
        test_csv=Path(run_params["test_csv"]),
        output_dir=Path(run_params["output_dir"]),
        log_dir=Path(run_params["log_dir"]),
        tb_log_dir=Path(run_params["tb_log_dir"]),
        seed=int(run_params.get("seed", 42)),
        log_level=str(run_params.get("log_level", "INFO")).upper(),
    )

    _ensure_dir(run_cfg.output_dir)
    _ensure_dir(run_cfg.log_dir)
    _ensure_dir(run_cfg.tb_log_dir)

    logger.setLevel(run_cfg.log_level)
    _set_seeds(run_cfg.seed)

    feature_config = PolygonFeatureConfig(**feature_params)
    model_config = ModelConfig(**model_params)
    training_config = TrainingConfig(**training_params)

    run_dict = asdict(run_cfg)
    for key in ("train_csv", "test_csv", "output_dir", "log_dir", "tb_log_dir"):
        run_dict[key] = str(run_dict[key])
    cli_command = "python -m annolid.behavior.training.polygon_frame_training"
    if args.config:
        cli_command += f" --config {shlex.quote(str(args.config))}"
    log_file = _configure_run_logger(run_cfg.log_dir, cli_command)
    logger.info(
        "Training configuration: %s",
        json.dumps(
            {
                "feature": asdict(feature_config),
                "model": asdict(model_config),
                "training": asdict(training_config),
                "run": run_dict,
            },
            indent=2,
        ),
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    best_state = train_polygon_frame_classifier(
        run_cfg.train_csv,
        run_cfg.test_csv,
        feature_config=feature_config,
        model_config=model_config,
        training_config=training_config,
        device=device,
        checkpoint_dir=run_cfg.output_dir,
    )

    ckpt_best = _save_checkpoint(best_state, run_cfg.output_dir, label="best")
    logger.info(f"Saved best checkpoint to {ckpt_best}")

    if "latest_model_state" in best_state:
        latest_state = dict(best_state)
        latest_state["model_state"] = latest_state.pop("latest_model_state")
        ckpt_latest = _save_checkpoint(
            latest_state, run_cfg.output_dir, label="latest")
        logger.info(f"Saved latest checkpoint to {ckpt_latest}")
    _dump_run_artifacts(
        run_cfg.output_dir,
        {
            "feature": asdict(feature_config),
            "model": asdict(model_config),
            "training": asdict(training_config),
            "run": {**run_dict, "log_file": str(log_file)},
        },
        suffix="config",
    )
    _log_run_artifacts(
        {
            "feature": asdict(feature_config),
            "model": asdict(model_config),
            "training": asdict(training_config),
            "run": {**run_dict, "log_file": str(log_file)},
        },
        label="Run",
    )

    # Evaluate on test set with the learned label mapping and polygon lengths.
    test_feature_config = replace(
        feature_config, polygon_pad_len=best_state["polygon_lengths"])
    normalization = best_state.get("normalization")
    test_dataset = PolygonFrameDataset(
        run_cfg.test_csv,
        test_feature_config,
        window_size=model_config.window_size,
        label_to_index=best_state["label_to_index"],
        normalization=normalization,
    )
    model = ImprovedFrameLabelConvNet(
        feature_dim=best_state["feature_dim"],
        num_classes=len(best_state["label_to_index"]),
        window_size=model_config.window_size,
        hidden_dim=model_config.hidden_dim,
        kernel_size=model_config.kernel_size,
        num_residual_blocks=model_config.num_residual_blocks,
        dropout=model_config.dropout,
        use_attention=model_config.use_attention,
    ).to(device)
    logger.info(f"Evaluation model structure:\n{model}")
    model.load_state_dict(best_state["model_state"])
    metrics = _evaluate(model, test_dataset, device)
    metrics_path = run_cfg.output_dir / "metrics.json"
    metrics_payload = _to_builtin({
        "test_metrics": metrics,
        "label_to_index": best_state["label_to_index"],
    })
    metrics_path.write_text(json.dumps(
        metrics_payload, indent=2), encoding="utf-8")
    logger.info(
        "Test metrics | loss: %.4f | acc: %.4f | macro_f1: %.4f | macro_mAP: %.4f | labels: %s",
        metrics.get("loss", 0.0),
        metrics.get("accuracy", 0.0),
        metrics.get("macro_f1", 0.0),
        metrics.get("macro_map", 0.0),
        metrics.get("labels", []),
    )
    per_class = metrics.get("per_class", {})
    for label, vals in per_class.items():
        if isinstance(vals, dict):
            logger.info(
                "Class %s | precision: %.4f | recall: %.4f | f1: %.4f",
                label,
                vals.get("precision", 0.0),
                vals.get("recall", 0.0),
                vals.get("f1-score", 0.0),
            )
    per_class_ap = metrics.get("per_class_ap", {})
    for label, ap in per_class_ap.items():
        logger.info("Class %s | AP: %.4f", label, ap)
    logger.info(f"Metrics saved to {metrics_path}")
    _dump_run_artifacts(run_cfg.output_dir, metrics_payload, suffix="metrics")
    _log_run_artifacts(metrics_payload, label="Run metrics")


if __name__ == "__main__":
    main()
