#!/usr/bin/env python
"""Frame-level polygon interaction classifier with 1D Conv residual blocks.

This implements the model:
  - Sliding temporal window over per-frame polygon features (default 11 frames)
  - 1D Conv backbone with residual blocks and optional channel attention
  - Cross-entropy training with class balancing and early stopping helpers
"""

from __future__ import annotations

import ast
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from annolid.utils.logger import logger

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional
    plt = None

__all__ = [
    "PolygonFeatureConfig",
    "ModelConfig",
    "TrainingConfig",
    "PolygonFrameDataset",
    "ResidualBlock",
    "ChannelAttention",
    "ImprovedFrameLabelConvNet",
    "train_polygon_frame_classifier",
]


@dataclass
class PolygonFeatureConfig:
    polygon_cols: Tuple[str, str] = ("intruder_features", "resident_features")
    extra_cols: Tuple[str, ...] = (
        "intruder_area",
        "resident_area",
        "intruder_centroid",
        "resident_centroid",
        "intruder_perimeter",
        "resident_perimeter",
        "intruder_motion_index",
        "resident_motion_index",
        "inter_animal_distance",
        "relative_velocity",
        "facing_angle",
    )
    polygon_pad_len: Optional[Union[int, Dict[str, int]]] = None
    frame_width: int = 1024
    frame_height: int = 570
    compute_dynamic_features: bool = True
    compute_motion_index: bool = True
    normalize_features: bool = True
    normalization_eps: float = 1e-6
    rescale_coordinates: bool = False  # if True, divide x/y by frame dims

    def __post_init__(self) -> None:
        try:
            self.normalization_eps = float(self.normalization_eps)
        except Exception:
            self.normalization_eps = 1e-6


@dataclass
class ModelConfig:
    window_size: int = 11
    hidden_dim: int = 128
    kernel_size: int = 3
    num_residual_blocks: int = 6
    dropout: float = 0.3
    use_attention: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 4e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 8
    early_stopping_patience: int = 12
    # Which validation metric to monitor for early stopping: "map" or "loss".
    early_stopping_monitor: str = "map"
    val_split_ratio: float = 0.1
    num_workers: int = 2
    sampling_strategy: str = "balanced_sampler"  # or "random"
    log_every: int = 50
    loss_type: str = "ce"  # ce or focal
    focal_gamma: float = 2.0
    add_noise_std: float = 0.0
    label_smoothing: float = 0.0
    apply_rolling_median: bool = True
    rolling_window: int = 7


def _parse_array(value) -> List[float]:
    """Parse list-like values stored in CSV."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(map(float, value))
    if isinstance(value, str):
        try:
            return list(map(float, json.loads(value)))
        except Exception:
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    return list(map(float, parsed))
            except Exception:
                return []
    try:
        return [float(value)]
    except Exception:
        return []


def _pad_or_truncate(arr: List[float], length: int) -> List[float]:
    if length <= 0:
        return []
    if len(arr) >= length:
        return arr[:length]
    return arr + [0.0] * (length - len(arr))


def _frame_sort_key(name: str):
    stem = Path(str(name)).stem
    parts = stem.split("_")
    try:
        return int(parts[-1])
    except Exception:
        return stem


class PolygonFrameDataset(Dataset):
    """Windowed frame dataset built from polygon feature CSV."""

    def __init__(
        self,
        csv_path: Path,
        feature_config: PolygonFeatureConfig,
        window_size: int,
        label_to_index: Optional[Dict[str, int]] = None,
        normalization: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.feature_config = feature_config
        self.window_size = window_size
        self._normalization = normalization

        self.dataframe = pd.read_csv(self.csv_path)
        if self.dataframe.empty:
            raise ValueError(f"No data found in {self.csv_path}")

        # Optionally enrich the dataframe with dynamic geometric features
        # computed per-video from centroids and polygon coordinates when
        # they are not already present in the CSV.
        self._augment_dynamic_features()

        self.label_to_index = label_to_index or {
            label: idx for idx, label in enumerate(sorted(self.dataframe["label"].unique()))
        }
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

        self._polygon_lengths = self._infer_polygon_lengths()
        self.feature_dim = None

        self.video_features: Dict[str, np.ndarray] = {}
        self.video_labels: Dict[str, np.ndarray] = {}
        self.indices: List[Tuple[str, int]] = []

        self._build_video_cache()
        if self.feature_config.normalize_features:
            if self._normalization is None:
                self._normalization = self._compute_normalization()
            if self._normalization:
                logger.info(
                    f"Applied feature normalization with mean[0]={float(self._normalization['mean'][0]):.4f} "
                    f"std[0]={float(self._normalization['std'][0]):.4f}"
                )

    def _augment_dynamic_features(self) -> None:
        if not self.feature_config.compute_dynamic_features:
            return
        required_centroids = {"intruder_centroid", "resident_centroid"}
        if not required_centroids.issubset(self.dataframe.columns):
            return
        # If all three dynamic columns already exist, respect the dataset
        # and avoid recomputing.
        dynamic_cols = {
            "inter_animal_distance",
            "relative_velocity",
            "facing_angle",
        }
        if dynamic_cols.issubset(self.dataframe.columns):
            return

        inter_distances: Dict[int, float] = {}
        rel_velocities: Dict[int, float] = {}
        facing_angles: Dict[int, float] = {}

        for video, df in self.dataframe.groupby("video"):
            if "frame" in df:
                df_sorted = df.sort_values(
                    by="frame", key=lambda s: s.map(_frame_sort_key))
            else:
                df_sorted = df.sort_index()

            prev_distance: Optional[float] = None
            prev_frame_number: Optional[int] = None

            for idx, row in df_sorted.iterrows():
                centroid_i = _parse_array(row.get("intruder_centroid", []))
                centroid_r = _parse_array(row.get("resident_centroid", []))
                centroid_i = centroid_i if len(centroid_i) >= 2 else [0.0, 0.0]
                centroid_r = centroid_r if len(centroid_r) >= 2 else [0.0, 0.0]

                dx = centroid_i[0] - centroid_r[0]
                dy = centroid_i[1] - centroid_r[1]
                distance = float(math.hypot(dx, dy))
                inter_distances[idx] = distance

                frame_number: Optional[int] = None
                if "frame" in row:
                    try:
                        frame_number = int(_frame_sort_key(row["frame"]))
                    except Exception:
                        frame_number = None

                dt = 1
                if frame_number is not None and prev_frame_number is not None:
                    dt = max(1, frame_number - prev_frame_number)

                if prev_distance is not None:
                    rel_velocities[idx] = float(
                        prev_distance - distance) / float(dt)
                else:
                    rel_velocities[idx] = 0.0
                prev_distance = distance
                prev_frame_number = frame_number

                # Facing angle derived from intruder polygon coordinates:
                # use vector from centroid to the furthest intruder point
                # and compare to vector from centroid to resident.
                intruder_coords = _parse_array(
                    row.get("intruder_features", []))
                if intruder_coords and distance > 0.0:
                    arr = np.asarray(
                        intruder_coords, dtype=float).reshape(-1, 2)
                    centroid_arr = np.asarray(centroid_i[:2], dtype=float)
                    offsets = arr - centroid_arr
                    dists = np.linalg.norm(offsets, axis=1)
                    head_idx = int(np.argmax(dists))
                    head_vec = offsets[head_idx]
                    to_resident = np.asarray(
                        [centroid_r[0] - centroid_i[0],
                         centroid_r[1] - centroid_i[1]],
                        dtype=float,
                    )
                    head_norm = float(np.linalg.norm(head_vec))
                    to_res_norm = float(np.linalg.norm(to_resident))
                    if head_norm > 0.0 and to_res_norm > 0.0:
                        cos_angle = float(
                            np.clip(
                                np.dot(head_vec, to_resident)
                                / (head_norm * to_res_norm),
                                -1.0,
                                1.0,
                            )
                        )
                        facing_angles[idx] = float(math.acos(cos_angle))
                    else:
                        facing_angles[idx] = 0.0
                else:
                    facing_angles[idx] = 0.0

        if "inter_animal_distance" not in self.dataframe.columns:
            self.dataframe["inter_animal_distance"] = self.dataframe.index.map(
                lambda i: inter_distances.get(i, 0.0)
            )
        if "relative_velocity" not in self.dataframe.columns:
            self.dataframe["relative_velocity"] = self.dataframe.index.map(
                lambda i: rel_velocities.get(i, 0.0)
            )
        if "facing_angle" not in self.dataframe.columns:
            self.dataframe["facing_angle"] = self.dataframe.index.map(
                lambda i: facing_angles.get(i, 0.0)
            )

    def _infer_polygon_lengths(self) -> Dict[str, int]:
        lengths: Dict[str, int] = {}
        pad_len = self.feature_config.polygon_pad_len
        for col in self.feature_config.polygon_cols:
            if isinstance(pad_len, dict) and col in pad_len:
                lengths[col] = int(pad_len[col])
                continue
            if isinstance(pad_len, int):
                lengths[col] = pad_len
                continue
            lengths[col] = int(
                self.dataframe[col]
                .dropna()
                .apply(lambda x: len(_parse_array(x)))
                .max()
            )
        return lengths

    def _relative_features(self, row: pd.Series) -> List[float]:
        area_i = float(row.get("intruder_area", 0) or 0)
        area_r = float(row.get("resident_area", 0) or 0)
        peri_i = float(row.get("intruder_perimeter", 0) or 0)
        peri_r = float(row.get("resident_perimeter", 0) or 0)
        mi = float(row.get("intruder_motion_index", 0) or 0)
        mr = float(row.get("resident_motion_index", 0) or 0)

        centroid_i = _parse_array(row.get("intruder_centroid", []))
        centroid_r = _parse_array(row.get("resident_centroid", []))
        centroid_i = centroid_i if len(centroid_i) >= 2 else [0.0, 0.0]
        centroid_r = centroid_r if len(centroid_r) >= 2 else [0.0, 0.0]

        dx = centroid_i[0] - centroid_r[0]
        dy = centroid_i[1] - centroid_r[1]
        diag = math.hypot(self.feature_config.frame_width,
                          self.feature_config.frame_height) or 1.0
        centroid_distance = math.hypot(dx, dy) / diag

        eps = 1e-6
        return [
            area_i - area_r,
            area_i / (area_r + eps),
            peri_i / (peri_r + eps),
            centroid_distance,
            dx / self.feature_config.frame_width,
            dy / self.feature_config.frame_height,
            mi - mr,
            mi / (mr + eps),
        ]

    def _row_to_feature(self, row: pd.Series) -> List[float]:
        values: List[float] = []
        for col in self.feature_config.polygon_cols:
            parsed = _parse_array(row[col])
            padded = _pad_or_truncate(parsed, self._polygon_lengths[col])
            if self.feature_config.rescale_coordinates and len(padded) >= 2:
                # treat pairs as x,y; clip to [0,1] after scaling
                arr = np.asarray(padded, dtype=float).reshape(-1, 2)
                arr[:, 0] = np.clip(
                    arr[:, 0] / max(self.feature_config.frame_width, 1), 0.0, 1.0)
                arr[:, 1] = np.clip(
                    arr[:, 1] / max(self.feature_config.frame_height, 1), 0.0, 1.0)
                padded = arr.flatten().tolist()
            values.extend(padded)

        for col in self.feature_config.extra_cols:
            if (not self.feature_config.compute_motion_index) and ("motion_index" in col):
                continue
            parsed = _parse_array(row.get(col, 0))
            values.extend(parsed)

        values.extend(self._relative_features(row))
        return values

    def _build_video_cache(self) -> None:
        grouped = self.dataframe.groupby("video")
        for video, df in grouped:
            if "frame" in df:
                df_sorted = df.sort_values(
                    by="frame", key=lambda s: s.map(_frame_sort_key))
            else:
                df_sorted = df.sort_index()
            features = []
            labels = []
            for _, row in df_sorted.iterrows():
                features.append(self._row_to_feature(row))
                labels.append(self.label_to_index[row["label"]])
            feature_array = np.asarray(features, dtype=np.float32)
            if self.feature_dim is None:
                self.feature_dim = feature_array.shape[1]
                logger.info(
                    f"Determined feature dimension: {self.feature_dim}")
            self.video_features[video] = feature_array
            self.video_labels[video] = np.asarray(labels, dtype=np.int64)
            for idx in range(len(labels)):
                self.indices.append((video, idx))

    @property
    def polygon_lengths(self) -> Dict[str, int]:
        return dict(self._polygon_lengths)

    def __len__(self) -> int:
        return len(self.indices)

    def _window_slice(self, video: str, center: int) -> np.ndarray:
        half = self.window_size // 2
        feats = self.video_features[video]
        start = max(center - half, 0)
        end = min(center + half + 1, feats.shape[0])
        window = feats[start:end]
        if len(window) < self.window_size:
            pad_top = max(0, half - center)
            pad_bottom = self.window_size - len(window) - pad_top
            if pad_top:
                window = np.vstack(
                    (np.repeat(window[:1], pad_top, axis=0), window))
            if pad_bottom:
                window = np.vstack(
                    (window, np.repeat(window[-1:], pad_bottom, axis=0)))
        return window

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video, center = self.indices[idx]
        window = self._window_slice(video, center)
        if self.feature_config.normalize_features and self._normalization:
            window = (
                window - self._normalization["mean"]) / self._normalization["std"]
        label = int(self.video_labels[video][center])
        return torch.from_numpy(window), label

    def _compute_normalization(self) -> Optional[Dict[str, np.ndarray]]:
        try:
            stacked = np.vstack(list(self.video_features.values()))
        except ValueError:
            return None
        mean = stacked.astype(np.float32).mean(axis=0)
        std = stacked.astype(np.float32).std(axis=0) + \
            float(self.feature_config.normalization_eps)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.dp1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        self.dp2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dp1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dp2(out)
        return out + residual


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        weights = self.fc(y).view(b, c, 1)
        return x * weights


class ImprovedFrameLabelConvNet(nn.Module):
    """1D Conv model with residual blocks and optional channel attention."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        window_size: int = 11,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        num_residual_blocks: int = 6,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.initial_conv = nn.Conv1d(
            feature_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)
        self.initial_act = nn.LeakyReLU(0.01, inplace=True)
        self.initial_dropout = nn.Dropout(dropout)

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden_dim, kernel_size=kernel_size, dropout=dropout)
                for _ in range(num_residual_blocks)
            ]
        )

        self.attention = ChannelAttention(
            hidden_dim) if use_attention else None
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.window_size = window_size
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, window, feature_dim) -> transpose to (B, C, L)
        x = x.transpose(1, 2)
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = self.initial_act(out)
        out = self.initial_dropout(out)

        out = self.res_blocks(out)
        if self.attention:
            out = self.attention(out)

        out = self.pool(out).squeeze(-1)
        return self.fc(out)


def _make_sampler(labels: np.ndarray, strategy: str) -> Optional[WeightedRandomSampler]:
    if strategy != "balanced_sampler":
        return None
    class_sample_count = np.bincount(labels)
    class_sample_count = np.where(
        class_sample_count == 0, 1, class_sample_count)
    weights = 1.0 / class_sample_count
    sample_weights = weights[labels]
    return WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.float), len(sample_weights))


def _collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    windows, labels = zip(*batch)
    return torch.stack(windows, dim=0), torch.as_tensor(labels, dtype=torch.long)


def _seed_worker(worker_id: int) -> None:
    """Set numpy/Python RNG seeds for deterministic DataLoader workers."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _mean_average_precision(probs: List[np.ndarray], targets: List[int], num_classes: int) -> float:
    if not probs:
        return 0.0
    y_true = np.zeros((len(targets), num_classes), dtype=float)
    for i, t in enumerate(targets):
        if 0 <= t < num_classes:
            y_true[i, t] = 1.0
    y_score = np.vstack(probs)
    try:
        return float(metrics.average_precision_score(y_true, y_score, average="macro"))
    except Exception:
        return 0.0


def _plot_training_curves(history: Dict[str, List[float]], output_dir: Path, prefix: str) -> None:
    if plt is None:
        return
    if not history.get("epoch"):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes

    ax1.plot(epochs, history.get("train_loss", []),
             label="train_loss", marker="o")
    ax1.plot(epochs, history.get("val_loss", []),
             label="val_loss", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.get("val_map", []),
             label="val mAP", marker="o")
    ax2.plot(epochs, history.get("val_macro_f1", []),
             label="val macro F1", marker="o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"{prefix}_training_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved training curves plot to {out_path}")


def _save_training_history(history: Dict[str, List[float]], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_history.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    try:
        _plot_training_curves(history, output_dir, prefix)
    except Exception as exc:  # pragma: no cover - plotting is optional
        logger.warning(f"Failed to plot training history: {exc}")


def _make_loss(
    loss_type: str,
    class_weights: torch.Tensor,
    focal_gamma: float,
    label_smoothing: float,
) -> nn.Module:
    if loss_type == "focal":
        class FocalLoss(nn.Module):
            def __init__(self, weights: torch.Tensor, gamma: float):
                super().__init__()
                self.weights = weights
                self.gamma = gamma

            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                probs = torch.softmax(logits, dim=1)
                targets_one_hot = torch.nn.functional.one_hot(
                    targets, num_classes=probs.shape[1]).float()
                pt = (probs * targets_one_hot).sum(dim=1).clamp(min=1e-6)
                log_pt = pt.log()
                weights = self.weights[targets]
                loss = -weights * ((1 - pt) ** self.gamma) * log_pt
                return loss.mean()

        return FocalLoss(class_weights, focal_gamma)

    smoothing = float(label_smoothing)
    if smoothing <= 0.0:
        return nn.CrossEntropyLoss(weight=class_weights)

    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, weights: torch.Tensor, smoothing: float):
            super().__init__()
            self.weights = weights
            self.smoothing = smoothing

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            num_classes = logits.size(1)
            log_probs = torch.log_softmax(logits, dim=1)

            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1),
                                   1.0 - self.smoothing)

            nll_loss = -(true_dist * log_probs).sum(dim=1)
            weights = self.weights[targets]
            loss = (nll_loss * weights).sum() / weights.sum()
            return loss

    return LabelSmoothingCrossEntropy(class_weights, smoothing)


def train_polygon_frame_classifier(
    train_csv: Path,
    feature_config: Optional[PolygonFeatureConfig] = None,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_prefix: str = "polygon_frame_classifier",
) -> Dict[str, torch.Tensor]:
    """End-to-end training pipeline for the ImprovedFrameLabelConvNet."""
    feature_config = feature_config or PolygonFeatureConfig()
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # type: ignore[attr-defined]
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    full_dataset = PolygonFrameDataset(
        train_csv,
        feature_config,
        window_size=model_config.window_size,
    )
    labels_ordered = np.asarray(
        [full_dataset.video_labels[vid][idx] for vid, idx in full_dataset.indices])
    label_names = [full_dataset.index_to_label[idx]
                   for idx in range(len(full_dataset.index_to_label))]
    unique_labels, label_counts = np.unique(labels_ordered, return_counts=True)
    distribution = {label_names[int(lbl)]: int(cnt)
                    for lbl, cnt in zip(unique_labels, label_counts)}
    logger.info(f"Label distribution (full dataset): {distribution}")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=training_config.val_split_ratio, random_state=42
    )
    groups = np.asarray([vid for vid, _ in full_dataset.indices])
    dummy = np.zeros(len(full_dataset))
    train_indices, val_indices = next(
        splitter.split(dummy, dummy, groups=groups))
    if len(val_indices) == 0:
        if len(train_indices) <= 1:
            logger.warning(
                "Validation split is empty and dataset is too small to reassign samples. "
                "Training will proceed without validation; consider increasing dataset size or val_split_ratio."
            )
        else:
            val_indices = train_indices[-1:]
            train_indices = train_indices[:-1]
            logger.warning(
                "Validation split was empty; moved 1 sample from train to validation. "
                "Consider increasing val_split_ratio or number of videos.")
    logger.info(
        f"Split dataset into train={len(train_indices)}, val={len(val_indices)} "
        f"(val_split={training_config.val_split_ratio:.2f})",
    )

    train_labels = labels_ordered[train_indices]
    sampler = _make_sampler(train_labels, training_config.sampling_strategy)

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=training_config.batch_size,
        sampler=sampler if sampler else RandomSampler(train_subset),
        num_workers=training_config.num_workers,
        collate_fn=_collate_fn,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=_collate_fn,
        worker_init_fn=_seed_worker,
    )

    model = ImprovedFrameLabelConvNet(
        feature_dim=full_dataset.feature_dim,
        num_classes=len(full_dataset.label_to_index),
        window_size=model_config.window_size,
        hidden_dim=model_config.hidden_dim,
        kernel_size=model_config.kernel_size,
        num_residual_blocks=model_config.num_residual_blocks,
        dropout=model_config.dropout,
        use_attention=model_config.use_attention,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model structure:\n{model}")
    logger.info(f"Total trainable parameters: {param_count:,}")

    # Weighted CE/Focal using inverse class frequency; normalize weights to mean=1 to avoid tiny losses.
    class_counts = np.bincount(
        labels_ordered, minlength=len(full_dataset.label_to_index))
    class_counts = np.where(class_counts == 0, 1, class_counts)
    raw_weights = 1.0 / class_counts
    normalized_weights = raw_weights / np.mean(raw_weights)
    class_weights = torch.tensor(
        normalized_weights, dtype=torch.float, device=device)
    criterion = _make_loss(training_config.loss_type,
                           class_weights, training_config.focal_gamma, training_config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=training_config.scheduler_patience, factor=0.5, verbose=False
    )

    def _maybe_save_checkpoint(state: Dict[str, object], label: str) -> Optional[Path]:
        if checkpoint_dir is None:
            return None
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{checkpoint_prefix}_{label}.pt"
        torch.save(state, path)
        logger.info(f"Checkpoint saved ({label}) to {path}")
        return path

    best_val_loss = float("inf")
    best_val_map = float("-inf")
    best_state = {
        "model_state": model.state_dict(),
        "label_to_index": full_dataset.label_to_index,
        "feature_config": feature_config,
        "model_config": model_config,
        "polygon_lengths": full_dataset.polygon_lengths,
        "feature_dim": full_dataset.feature_dim,
        "normalization": getattr(full_dataset, "_normalization", None),
        "best_val_map": best_val_map,
        "best_val_loss": best_val_loss,
    }
    latest_state: Dict[str, object] = {}
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_map": [],
        "val_macro_f1": [],
    }
    epochs_without_improve = 0
    for epoch in range(training_config.num_epochs):
        model.train()
        total_loss = 0.0
        nan_loss_detected = False
        for step, (inputs, targets) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            if training_config.add_noise_std > 0:
                noise = torch.randn_like(
                    inputs) * training_config.add_noise_std
                inputs = inputs + noise
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss):
                logger.error(
                    f"Non-finite loss detected at epoch {epoch + 1} step {step}: {loss.item()}"
                )
                nan_loss_detected = True
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % training_config.log_every == 0:
                logger.info(
                    f"Epoch {epoch + 1} Step {step} - loss {loss.item():.4f}")

        if nan_loss_detected:
            logger.info(
                "Stopping training early due to non-finite loss.")
            break

        avg_train_loss = total_loss / max(1, len(train_loader))
        model.eval()
        val_loss = 0.0
        val_probs: List[np.ndarray] = []
        val_targets: List[int] = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(targets.cpu().numpy().tolist())
        val_loss /= max(1, len(val_loader))
        has_val = len(val_targets) > 0
        if has_val:
            val_map = _mean_average_precision(
                val_probs, val_targets, num_classes=len(full_dataset.label_to_index))
            scheduler.step(val_loss)
            val_macro_f1 = metrics.f1_score(
                val_targets, np.argmax(val_probs, axis=1), average="macro", zero_division=0)
        else:
            logger.warning(
                "Validation set is empty; skipping validation metrics. "
                "Increase val_split_ratio or ensure multiple videos are available.")
            val_map = 0.0
            val_macro_f1 = 0.0
            scheduler.step(val_loss)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_map"].append(float(val_map))
        history["val_macro_f1"].append(float(val_macro_f1))

        logger.info(
            f"Epoch {epoch + 1}/{training_config.num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val mAP: {val_map:.4f} | Val F1(macro): {val_macro_f1:.4f}"
        )
        improved_map = has_val and (
            val_map > best_val_map or (
                val_map == best_val_map and val_loss < best_val_loss
            )
        )
        improved_loss = has_val and val_loss < best_val_loss
        if training_config.early_stopping_monitor == "loss":
            improved = improved_loss
        else:
            improved = improved_map

        # Always keep the latest checkpoint.
        latest_state = {
            "model_state": model.state_dict(),
            "latest_model_state": model.state_dict(),
            "latest_val_map": val_map,
            "latest_val_loss": val_loss,
            "label_to_index": full_dataset.label_to_index,
            "feature_config": feature_config,
            "model_config": model_config,
            "polygon_lengths": full_dataset.polygon_lengths,
            "feature_dim": full_dataset.feature_dim,
            "normalization": getattr(full_dataset, "_normalization", None),
        }
        _maybe_save_checkpoint(latest_state, label="latest")

        if not has_val:
            # Without validation data, keep the latest model as the current best.
            best_state = {
                **latest_state,
                "best_val_map": val_map,
                "best_val_loss": val_loss,
            }
            best_val_map = val_map
            best_val_loss = val_loss
            continue

        if improved:
            best_val_loss = val_loss
            best_val_map = val_map
            epochs_without_improve = 0
            best_state = {
                "model_state": model.state_dict(),
                "label_to_index": full_dataset.label_to_index,
                "feature_config": feature_config,
                "model_config": model_config,
                "polygon_lengths": full_dataset.polygon_lengths,
                "feature_dim": full_dataset.feature_dim,
                "best_val_map": best_val_map,
                "best_val_loss": best_val_loss,
                "normalization": getattr(full_dataset, "_normalization", None),
            }
            _maybe_save_checkpoint(best_state, label="best")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= training_config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epochs_without_improve} epochs without improvement.")
                break

    # Merge references so downstream can access both best and latest.
    if latest_state:
        best_state["latest_model_state"] = latest_state.get("model_state")
        best_state["latest_val_map"] = latest_state.get("latest_val_map")
        best_state["latest_val_loss"] = latest_state.get("latest_val_loss")
    if checkpoint_dir is not None:
        _save_training_history(history, checkpoint_dir, checkpoint_prefix)
    return best_state
