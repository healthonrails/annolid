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
from typing import Any, Dict, Optional, List

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
    ImprovedFrameLabelConvNet,
    _collate_fn,  # type: ignore
    _seed_worker,  # type: ignore
)
from annolid.utils.logger import logger

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional
    plt = None


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
        description="Train polygon frame classifier from CSV features."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML/JSON config to seed parameters (CLI overrides).",
    )
    parser.add_argument(
        "--train_csv", required=False, default=None, help="Path to training CSV."
    )
    parser.add_argument(
        "--test_csv", required=False, default=None, help="Path to test CSV."
    )
    parser.add_argument(
        "--output_dir", default="results", help="Directory to save models and metrics."
    )
    parser.add_argument("--log_dir", default="logs", help="Directory for log files.")
    parser.add_argument(
        "--tb_log_dir",
        default="runs",
        help="Directory for TensorBoard logs (unused placeholder).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")

    parser.add_argument("--window_size", type=int, default=11)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--num_residual_blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--use_attention",
        dest="use_attention",
        action="store_true",
        default=None,
        help="Enable channel attention (default: enabled).",
    )
    parser.add_argument(
        "--no_attention",
        dest="use_attention",
        action="store_false",
        help="Disable channel attention.",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=4e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_patience", type=int, default=15)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--sampling_strategy",
        default="balanced_sampler",
        choices=["balanced_sampler", "random"],
    )
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=None,
        help="Label smoothing factor (default from config; set 0 to disable).",
    )
    parser.add_argument(
        "--no_label_smoothing",
        dest="label_smoothing",
        action="store_const",
        const=0.0,
        help="Disable label smoothing (sets label_smoothing=0).",
    )

    parser.add_argument("--frame_width", type=int, default=1024)
    parser.add_argument("--frame_height", type=int, default=570)
    parser.add_argument("--polygon_pad_len", type=int, default=None)
    parser.add_argument(
        "--compute_dynamic_features",
        dest="compute_dynamic_features",
        action="store_true",
        default=None,
        help="Compute dynamic features (inter_animal_distance, relative_velocity, facing_angle) if missing in CSV.",
    )
    parser.add_argument(
        "--no_dynamic_features",
        dest="compute_dynamic_features",
        action="store_false",
        help="Do not compute dynamic features when missing.",
    )
    parser.add_argument(
        "--compute_motion_index",
        dest="compute_motion_index",
        action="store_true",
        default=None,
        help="Use motion index features if present.",
    )
    parser.add_argument(
        "--no_motion_index",
        dest="compute_motion_index",
        action="store_false",
        help="Disable motion index features even if present.",
    )

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


def _increment_path(base: Path) -> Path:
    """YOLO-style run directory incrementer: exp, exp2, exp3, ..."""
    if not base.exists():
        return base
    parent = base.parent
    name = base.name
    existing = [p.name for p in parent.glob(f"{name}*") if p.is_dir()]
    indices: List[int] = []
    for n in existing:
        suffix = n[len(name) :]
        if suffix == "":
            indices.append(1)
        elif suffix.isdigit():
            indices.append(int(suffix))
    next_idx = (max(indices) + 1) if indices else 1
    return parent / (name if next_idx == 1 else f"{name}{next_idx}")


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


def _collect_cli_overrides(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, value in vars(args).items():
        default = parser.get_default(key)
        if value is None:
            continue
        if value != default:
            overrides[key] = value
    return overrides


def _merge_section(
    defaults: Dict[str, Any],
    config_values: Dict[str, Any],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, val in (config_values or {}).items():
        if key in merged:
            merged[key] = val
    for key, val in cli_overrides.items():
        if key in merged:
            merged[key] = val
    return merged


def _apply_rolling_median_probs(
    probs: np.ndarray,
    video_ids: np.ndarray,
    window: int,
) -> np.ndarray:
    if window <= 1 or probs.size == 0:
        return probs
    smoothed = probs.copy()
    half = window // 2
    unique_videos = np.unique(video_ids)
    for vid in unique_videos:
        idxs = np.where(video_ids == vid)[0]
        if idxs.size == 0:
            continue
        sub = probs[idxs]  # (T, C)
        t_len, n_classes = sub.shape
        for c in range(n_classes):
            series = sub[:, c]
            pad_left = np.repeat(series[0], half)
            pad_right = np.repeat(series[-1], max(0, window - half - 1))
            padded = np.concatenate([pad_left, series, pad_right])
            out = np.empty_like(series)
            for t in range(t_len):
                out[t] = np.median(padded[t : t + window])
            smoothed[idxs, c] = out
    return smoothed


def _evaluate(
    model: ImprovedFrameLabelConvNet,
    dataset: PolygonFrameDataset,
    device: torch.device,
    plot_dir: Optional[Path] = None,
    smooth_predictions: bool = True,
    smooth_window: int = 7,
) -> Dict[str, Any]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate_fn,
        worker_init_fn=_seed_worker,
    )
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []
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
    label_names = [
        dataset.index_to_label[idx] for idx in range(len(dataset.index_to_label))
    ]

    prob_array = np.asarray(all_probs, dtype=float)
    true_array = np.asarray(all_targets, dtype=int)
    pred_array = np.asarray(all_preds, dtype=int)

    if smooth_predictions and prob_array.size and hasattr(dataset, "indices"):
        video_ids = np.asarray([vid for vid, _ in dataset.indices])
        if video_ids.shape[0] == prob_array.shape[0]:
            prob_array = _apply_rolling_median_probs(
                prob_array, video_ids, smooth_window
            )
            pred_array = prob_array.argmax(axis=1)

    accuracy = metrics.accuracy_score(true_array, pred_array)
    macro_f1 = metrics.f1_score(
        true_array, pred_array, average="macro", zero_division=0
    )
    per_class = metrics.classification_report(
        true_array,
        pred_array,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    confusion = metrics.confusion_matrix(
        true_array, pred_array, labels=list(range(len(label_names)))
    ).tolist()

    # Per-class AP and mAP using smoothed probability scores.
    per_class_ap: Dict[str, float] = {}
    try:
        ap_scores = []
        for idx, name in enumerate(label_names):
            binary_true = (true_array == idx).astype(int)
            ap = (
                metrics.average_precision_score(binary_true, prob_array[:, idx])
                if prob_array.size
                else 0.0
            )
            if np.isnan(ap):
                ap = 0.0
            per_class_ap[name] = float(ap)
            ap_scores.append(ap)
        macro_map = float(np.mean(ap_scores)) if ap_scores else 0.0
    except Exception:
        macro_map = 0.0
        per_class_ap = {}

    if plot_dir is not None:
        try:
            _plot_eval_curves_and_confusion(
                prob_array.tolist(),
                true_array.tolist(),
                pred_array.tolist(),
                label_names,
                Path(plot_dir),
            )
        except Exception as exc:  # pragma: no cover - plotting is optional
            logger.warning(f"Failed to plot evaluation metrics: {exc}")
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


def _plot_eval_curves_and_confusion(
    probs: List[List[float]],
    targets: List[int],
    preds: List[int],
    label_names: List[str],
    plot_dir: Path,
) -> None:
    if plt is None:
        return
    if not probs:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_score = np.asarray(probs, dtype=float)
    y_true_idx = np.asarray(targets, dtype=int)
    y_pred_idx = np.asarray(preds, dtype=int)
    num_classes = len(label_names)

    # Confusion matrix heatmap
    cm = metrics.confusion_matrix(
        y_true_idx, y_pred_idx, labels=list(range(num_classes))
    )
    fig_cm, ax_cm = plt.subplots(1, 1, figsize=(6, 5))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    fig_cm.colorbar(im, ax=ax_cm)
    ax_cm.set_xticks(range(num_classes))
    ax_cm.set_yticks(range(num_classes))
    ax_cm.set_xticklabels(label_names, rotation=45, ha="right")
    ax_cm.set_yticklabels(label_names)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    for i in range(num_classes):
        for j in range(num_classes):
            ax_cm.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig_cm.tight_layout()
    cm_path = plot_dir / "confusion_matrix.png"
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)
    logger.info(f"Saved confusion matrix plot to {cm_path}")

    # Precision-recall curves per class
    y_true = np.zeros((len(y_true_idx), num_classes), dtype=int)
    for i, t in enumerate(y_true_idx):
        if 0 <= t < num_classes:
            y_true[i, t] = 1

    fig_pr, ax_pr = plt.subplots(1, 1, figsize=(6, 5))
    for idx, name in enumerate(label_names):
        if y_true[:, idx].sum() == 0:
            continue
        precision, recall, _ = metrics.precision_recall_curve(
            y_true[:, idx], y_score[:, idx]
        )
        ap = metrics.average_precision_score(y_true[:, idx], y_score[:, idx])
        ax_pr.step(
            recall,
            precision,
            where="post",
            label=f"{name} (AP={ap:.3f})",
        )
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(fontsize=8)
    fig_pr.tight_layout()
    pr_path = plot_dir / "pr_curves.png"
    fig_pr.savefig(pr_path)
    plt.close(fig_pr)
    logger.info(f"Saved PR curves plot to {pr_path}")


def _save_checkpoint(
    state: Dict[str, Any], output_dir: Path, label: str = "best"
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = output_dir / f"polygon_frame_classifier_{label}_{ts}.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)
    return ckpt_path


def _dump_run_artifacts(
    output_dir: Path, run_payload: Dict[str, Any], suffix: str
) -> Path:
    path = output_dir / f"run_{suffix}.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_to_builtin(run_payload), fh, sort_keys=False)
    return path


def _log_run_artifacts(run_payload: Dict[str, Any], label: str) -> None:
    logger.info(
        f"{label} configuration/details:\n{json.dumps(_to_builtin(run_payload), indent=2)}"
    )


def _configure_run_logger(log_dir: Path, command: str) -> Path:
    """Add a file handler for this run and log the invoked command."""
    _ensure_dir(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
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
        "output_dir": "results",  # project directory, e.g. runs/polygon_frame
        "log_dir": "logs",  # subdirectory inside each run directory
        "tb_log_dir": "tb",  # subdirectory inside each run directory
        "seed": 42,
        "log_level": "INFO",
        "run_name": "exp",
    }
    feature_defaults = asdict(PolygonFeatureConfig())
    model_defaults = asdict(ModelConfig())
    training_defaults = asdict(TrainingConfig())

    run_params = _merge_section(run_defaults, cfg_file.get("run", {}), cli_overrides)
    feature_params = _merge_section(
        feature_defaults, cfg_file.get("feature", {}), cli_overrides
    )
    model_params = _merge_section(
        model_defaults, cfg_file.get("model", {}), cli_overrides
    )
    training_params = _merge_section(
        training_defaults, cfg_file.get("training", {}), cli_overrides
    )

    if not run_params["train_csv"] or not run_params["test_csv"]:
        raise ValueError("train_csv and test_csv must be provided via config or CLI.")

    project_dir = Path(run_params["output_dir"])
    _ensure_dir(project_dir)
    base_run_name = str(run_params.get("run_name", "exp"))
    run_dir = _increment_path(project_dir / base_run_name)
    log_subdir = Path(run_params["log_dir"]).name
    tb_subdir = Path(run_params["tb_log_dir"]).name

    run_cfg = RunConfig(
        train_csv=Path(run_params["train_csv"]),
        test_csv=Path(run_params["test_csv"]),
        output_dir=run_dir,
        log_dir=run_dir / log_subdir,
        tb_log_dir=run_dir / tb_subdir,
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
    run_dict["project_dir"] = str(project_dir)
    run_dict["run_name"] = base_run_name
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
        ckpt_latest = _save_checkpoint(latest_state, run_cfg.output_dir, label="latest")
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
        feature_config, polygon_pad_len=best_state["polygon_lengths"]
    )
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
    eval_state = best_state
    try:
        loaded_state = torch.load(ckpt_best, map_location=device)
        if isinstance(loaded_state, dict) and "model_state" in loaded_state:
            eval_state = loaded_state
            logger.info(f"Loaded best checkpoint from {ckpt_best} for evaluation.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            f"Failed to load best checkpoint from {ckpt_best}, using in-memory state: {exc}"
        )
    model.load_state_dict(eval_state["model_state"])
    metrics = _evaluate(
        model,
        test_dataset,
        device,
        plot_dir=run_cfg.output_dir,
        smooth_predictions=training_config.apply_rolling_median,
        smooth_window=training_config.rolling_window,
    )
    metrics_path = run_cfg.output_dir / "metrics.json"
    metrics_payload = _to_builtin(
        {
            "test_metrics": metrics,
            "label_to_index": best_state["label_to_index"],
        }
    )
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info(
        f"Test metrics | loss: {metrics.get('loss', 0.0):.4f} | "
        f"acc: {metrics.get('accuracy', 0.0):.4f} | "
        f"macro_f1: {metrics.get('macro_f1', 0.0):.4f} | "
        f"macro_mAP: {metrics.get('macro_map', 0.0):.4f} | "
        f"labels: {metrics.get('labels', [])}"
    )
    per_class = metrics.get("per_class", {})
    for label, vals in per_class.items():
        if isinstance(vals, dict):
            logger.info(
                f"Class {label} | precision: {vals.get('precision', 0.0):.4f} | "
                f"recall: {vals.get('recall', 0.0):.4f} | f1: {vals.get('f1-score', 0.0):.4f}"
            )
    per_class_ap = metrics.get("per_class_ap", {})
    for label, ap in per_class_ap.items():
        logger.info(f"Class {label} | AP: {ap:.4f}")
    logger.info(f"Metrics saved to {metrics_path}")
    _dump_run_artifacts(run_cfg.output_dir, metrics_payload, suffix="metrics")
    _log_run_artifacts(metrics_payload, label="Run metrics")


if __name__ == "__main__":
    main()
