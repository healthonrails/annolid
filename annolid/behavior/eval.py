from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from annolid.behavior.reporting import plot_behavior_eval_artifacts
from annolid.behavior.data_loading.datasets import BehaviorDataset
from annolid.behavior.pipeline import (
    BACKBONE_CHOICES,
    DEFAULT_DINOV3_MODEL,
    build_transform,
    load_classifier,
)

logger = logging.getLogger(__name__)


def _build_dataset(
    *,
    video_folder: str,
    split: str,
    val_ratio: float,
    random_seed: int,
    transform: Any,
):
    split_norm = str(split or "all").strip().lower()
    if split_norm in {"train", "val"}:
        return BehaviorDataset(
            video_folder,
            transform=transform,
            split=split_norm,
            val_ratio=float(val_ratio),
            random_seed=int(random_seed),
        )
    if split_norm == "all":
        dataset = BehaviorDataset(
            video_folder,
            transform=transform,
            split="train",
            val_ratio=float(val_ratio),
            random_seed=int(random_seed),
        )
        total = sum(len(df) for df in dataset.all_annotations.values())
        dataset.indices = list(range(int(total)))
        return dataset
    raise ValueError("split must be one of: all, train, val")


def evaluate_behavior_classifier(
    *,
    video_folder: str,
    checkpoint_path: str,
    batch_size: int = 1,
    feature_backbone: str = "clip",
    dinov3_model_name: str = DEFAULT_DINOV3_MODEL,
    feature_dim: int | None = None,
    transformer_dim: int = 768,
    device: str | None = None,
    split: str = "all",
    val_ratio: float = 0.2,
    random_seed: int = 42,
    num_workers: int = 0,
    plot_dir: str | None = None,
) -> dict[str, Any]:
    if device:
        device_t = torch.device(str(device))
    else:
        device_t = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"  # type: ignore[attr-defined]
        )

    transform = build_transform(feature_backbone)
    dataset = _build_dataset(
        video_folder=str(video_folder),
        split=str(split),
        val_ratio=float(val_ratio),
        random_seed=int(random_seed),
        transform=transform,
    )
    num_classes = int(dataset.get_num_classes())
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
    )
    model = load_classifier(
        checkpoint_path=str(Path(checkpoint_path).expanduser().resolve()),
        num_classes=num_classes,
        backbone=str(feature_backbone),
        device=device_t,
        dinov3_model=str(dinov3_model_name),
        feature_dim=feature_dim,
        transformer_dim=int(transformer_dim),
    )

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    all_probs: list[list[float]] = []
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_video_names: list[str] = []

    with torch.no_grad():
        for inputs, targets, video_names in loader:
            inputs = inputs.to(device_t)
            targets = targets.to(device_t)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item())
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.extend(probs.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_video_names.extend([str(item) for item in video_names])

    avg_loss = total_loss / max(1, len(loader))
    index_to_label = {idx: label for label, idx in dataset.label_mapping.items()}
    label_names = [index_to_label[idx] for idx in sorted(index_to_label.keys())]
    true_array = np.asarray(all_targets, dtype=int)
    pred_array = np.asarray(all_preds, dtype=int)
    prob_array = np.asarray(all_probs, dtype=float)

    accuracy = (
        metrics.accuracy_score(true_array, pred_array) if len(true_array) else 0.0
    )
    macro_f1 = (
        metrics.f1_score(true_array, pred_array, average="macro", zero_division=0)
        if len(true_array)
        else 0.0
    )
    per_class = (
        metrics.classification_report(
            true_array,
            pred_array,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )
        if len(true_array)
        else {}
    )
    confusion = (
        metrics.confusion_matrix(
            true_array, pred_array, labels=list(range(len(label_names)))
        ).tolist()
        if len(true_array)
        else []
    )

    per_class_ap: dict[str, float] = {}
    ap_scores: list[float] = []
    if prob_array.size and len(true_array):
        for idx, name in enumerate(label_names):
            binary_true = (true_array == idx).astype(int)
            ap = metrics.average_precision_score(binary_true, prob_array[:, idx])
            if np.isnan(ap):
                ap = 0.0
            per_class_ap[name] = float(ap)
            ap_scores.append(float(ap))
    macro_map = float(np.mean(ap_scores)) if ap_scores else 0.0

    artifacts: dict[str, str] = {}
    if str(plot_dir or "").strip():
        try:
            artifacts = plot_behavior_eval_artifacts(
                probs=all_probs,
                targets=all_targets,
                preds=all_preds,
                label_names=label_names,
                plot_dir=Path(str(plot_dir)).expanduser().resolve(),
            )
        except Exception as exc:  # pragma: no cover - optional plotting
            logger.warning("Failed to generate behavior eval plots: %s", exc)

    return {
        "test_metrics": {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "macro_map": float(macro_map),
            "per_class_ap": per_class_ap,
            "per_class": per_class,
            "confusion_matrix": confusion,
            "labels": label_names,
        },
        "predictions": [
            {
                "video_name": name,
                "target_index": int(target),
                "predicted_index": int(pred),
                "class_probabilities": [float(v) for v in probs],
            }
            for name, target, pred, probs in zip(
                all_video_names, all_targets, all_preds, all_probs
            )
        ],
        "label_mapping": dataset.label_mapping,
        "split": str(split),
        "val_ratio": float(val_ratio),
        "random_seed": int(random_seed),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a behavior classifier checkpoint on labeled behavior videos."
    )
    parser.add_argument(
        "--video-folder", required=True, help="Folder of labeled videos"
    )
    parser.add_argument(
        "--checkpoint-path", required=True, help="Path to trained weights (.pth)"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--feature-backbone", choices=BACKBONE_CHOICES, default="clip")
    parser.add_argument("--dinov3-model-name", type=str, default=DEFAULT_DINOV3_MODEL)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--transformer-dim", type=int, default=768)
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory to write confusion-matrix and PR-curve plots.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    payload = evaluate_behavior_classifier(
        video_folder=str(args.video_folder),
        checkpoint_path=str(args.checkpoint_path),
        batch_size=int(args.batch_size),
        feature_backbone=str(args.feature_backbone),
        dinov3_model_name=str(args.dinov3_model_name),
        feature_dim=args.feature_dim,
        transformer_dim=int(args.transformer_dim),
        device=(str(args.device).strip() if args.device else None),
        split=str(args.split),
        val_ratio=float(args.val_ratio),
        random_seed=int(args.random_seed),
        num_workers=int(args.num_workers),
        plot_dir=(str(args.plot_dir).strip() if args.plot_dir else None),
    )
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
