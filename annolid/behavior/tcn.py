"""Temporal convolutional behavior classification from pose or feature CSVs.

This module provides a small, modernized implementation of the DAART-style
supervised TCN pipeline used in Blau et al. 2024:

* load per-frame marker/features CSVs plus one-hot behavior labels
* optionally concatenate per-frame velocity features
* train a dilated temporal convolutional network with ignored background labels
* evaluate frame-level precision/recall/F1 on held-out sessions

The code is intentionally independent from the old DAART package so it can run
with Annolid's current PyTorch/NumPy dependency stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TCNSession:
    """A single labeled behavior session."""

    session_id: str
    features: Path
    labels: Path | None = None
    split: str = "train"


@dataclass(frozen=True)
class TCNFeatureConfig:
    """Feature loading and normalization options."""

    input_type: str = "features"
    add_velocity: bool = False
    zscore: bool = True
    background_label: str | int = 0
    drop_likelihood: bool = True


@dataclass(frozen=True)
class TCNModelConfig:
    """Model architecture options."""

    hidden_dim: int = 32
    num_blocks: int = 2
    kernel_size: int = 9
    dropout: float = 0.10


@dataclass(frozen=True)
class TCNTrainingConfig:
    """Training loop options."""

    epochs: int = 500
    batch_size: int = 8
    sequence_length: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    class_weighting: str = "inverse_frequency"


@dataclass
class TCNRunConfig:
    """Top-level run configuration."""

    sessions: list[TCNSession]
    labels: list[str] = field(default_factory=list)
    feature: TCNFeatureConfig = field(default_factory=TCNFeatureConfig)
    model: TCNModelConfig = field(default_factory=TCNModelConfig)
    training: TCNTrainingConfig = field(default_factory=TCNTrainingConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TCNRunConfig":
        sessions = [
            TCNSession(
                session_id=str(item.get("id") or item.get("session_id")),
                features=Path(str(item["features"])).expanduser(),
                labels=(
                    Path(str(item["labels"])).expanduser()
                    if item.get("labels") is not None
                    else None
                ),
                split=str(item.get("split", "train")),
            )
            for item in payload.get("sessions", [])
        ]
        if not sessions:
            raise ValueError("TCN config requires at least one session")
        return cls(
            sessions=sessions,
            labels=[str(v) for v in payload.get("labels", [])],
            feature=TCNFeatureConfig(**dict(payload.get("feature", {}) or {})),
            model=TCNModelConfig(**dict(payload.get("model", {}) or {})),
            training=TCNTrainingConfig(**dict(payload.get("training", {}) or {})),
        )


@dataclass(frozen=True)
class TCNNormalization:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def resolve_device(name: str | None = "auto") -> torch.device:
    label = str(name or "auto").strip().lower()
    if label != "auto":
        return torch.device(label)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_feature_csv(
    path: str | Path, *, input_type: str, drop_likelihood: bool = True
) -> np.ndarray:
    """Read feature or DLC-style marker CSV into a ``T x C`` float array."""

    path = Path(path).expanduser()
    if input_type == "markers":
        df = pd.read_csv(path, header=[0, 1, 2])
        if "scorer" in df.columns.get_level_values(0):
            df = df.drop(["scorer"], axis=1, level=0)
        values = df.to_numpy(dtype=np.float32)
        if drop_likelihood:
            xs = values[:, 0::3]
            ys = values[:, 1::3]
            return np.concatenate([xs, ys], axis=1).astype(np.float32)
        return values.astype(np.float32)

    df = pd.read_csv(path)
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    index_columns = [
        col
        for col in df.columns
        if str(col).strip().lower() in {"frame", "frames", "frame_index", "index"}
    ]
    if index_columns:
        df = df.drop(columns=index_columns)
    return df.to_numpy(dtype=np.float32)


def add_velocity_features(values: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        velocity[1:] = values[1:] - values[:-1]
    return np.concatenate([values, velocity], axis=1).astype(np.float32)


def read_label_csv(
    path: str | Path, labels: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Read one-hot behavior labels into class ids."""

    df = pd.read_csv(Path(path).expanduser())
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    if labels:
        missing = [label for label in labels if label not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns in {path}: {missing}")
        df = df[labels]
    else:
        index_columns = [
            col
            for col in df.columns
            if str(col).strip().lower() in {"frame", "frames", "frame_index", "index"}
        ]
        if index_columns:
            df = df.drop(columns=index_columns)
    label_names = [str(col) for col in df.columns]
    values = df.to_numpy(dtype=np.float32)
    class_ids = values.argmax(axis=1).astype(np.int64)
    return class_ids, label_names


def _sequence_starts(
    length: int, sequence_length: int, *, stride: int | None = None
) -> list[int]:
    if length <= 0:
        return []
    stride = int(stride or sequence_length)
    if length <= sequence_length:
        return [0]
    starts = list(range(0, length - sequence_length + 1, stride))
    final = length - sequence_length
    if starts[-1] != final:
        starts.append(final)
    return starts


class TCNSequenceDataset(Dataset):
    """Windowed per-frame dataset for TCN training/evaluation."""

    def __init__(
        self,
        sessions: Iterable[TCNSession],
        *,
        feature_config: TCNFeatureConfig,
        label_names: list[str] | None,
        sequence_length: int,
        normalization: TCNNormalization | None = None,
        require_labels: bool = True,
    ) -> None:
        self.sessions = list(sessions)
        self.feature_config = feature_config
        self.label_names = list(label_names or [])
        self.sequence_length = int(sequence_length)
        self.normalization = normalization
        self.features_by_session: dict[str, np.ndarray] = {}
        self.labels_by_session: dict[str, np.ndarray | None] = {}
        self.index: list[tuple[str, int]] = []

        for session in self.sessions:
            features = read_feature_csv(
                session.features,
                input_type=self.feature_config.input_type,
                drop_likelihood=self.feature_config.drop_likelihood,
            )
            if self.feature_config.add_velocity:
                features = add_velocity_features(features)
            if self.normalization is not None:
                features = self.normalization.apply(features).astype(np.float32)

            labels: np.ndarray | None = None
            if session.labels is not None:
                labels, names = read_label_csv(session.labels, self.label_names or None)
                if not self.label_names:
                    self.label_names = names
                if len(labels) != len(features):
                    raise ValueError(
                        f"{session.session_id}: feature/label length mismatch "
                        f"({len(features)} != {len(labels)})"
                    )
            elif require_labels:
                raise ValueError(f"{session.session_id}: labels are required")

            self.features_by_session[session.session_id] = features.astype(np.float32)
            self.labels_by_session[session.session_id] = labels
            for start in _sequence_starts(len(features), self.sequence_length):
                self.index.append((session.session_id, start))

        if not self.index:
            raise ValueError("No sequences available for TCN dataset")
        self.input_dim = next(iter(self.features_by_session.values())).shape[1]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        session_id, start = self.index[int(idx)]
        features = self.features_by_session[session_id]
        labels = self.labels_by_session[session_id]
        x = np.zeros((self.sequence_length, features.shape[1]), dtype=np.float32)
        y = np.zeros((self.sequence_length,), dtype=np.int64)
        valid_len = min(self.sequence_length, max(0, len(features) - start))
        x[:valid_len] = features[start : start + valid_len]
        if labels is not None:
            y[:valid_len] = labels[start : start + valid_len]
        return {
            "features": torch.from_numpy(x),
            "labels": torch.from_numpy(y),
            "valid_length": torch.tensor(valid_len, dtype=torch.long),
            "session_id": session_id,
            "start": int(start),
        }


def fit_normalization(
    sessions: Iterable[TCNSession],
    *,
    feature_config: TCNFeatureConfig,
) -> TCNNormalization:
    chunks: list[np.ndarray] = []
    for session in sessions:
        values = read_feature_csv(
            session.features,
            input_type=feature_config.input_type,
            drop_likelihood=feature_config.drop_likelihood,
        )
        if feature_config.add_velocity:
            values = add_velocity_features(values)
        chunks.append(values)
    data = np.concatenate(chunks, axis=0).astype(np.float32)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return TCNNormalization(mean=mean.astype(np.float32), std=std.astype(np.float32))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size]


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.LeakyReLU(inplace=True),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.LeakyReLU(inplace=True),
            nn.Dropout1d(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.downsample(x))


class BehaviorTCN(nn.Module):
    """Dilated TCN returning per-frame behavior logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        config: TCNModelConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or TCNModelConfig()
        layers: list[nn.Module] = []
        in_channels = int(input_dim)
        for block_idx in range(int(config.num_blocks)):
            dilation = 2**block_idx
            layers.append(
                TCNBlock(
                    in_channels,
                    int(config.hidden_dim),
                    kernel_size=int(config.kernel_size),
                    dilation=dilation,
                    dropout=float(config.dropout),
                )
            )
            in_channels = int(config.hidden_dim)
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(in_channels, int(num_classes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch, time, channels
        z = x.transpose(1, 2)
        z = self.encoder(z)
        logits = self.classifier(z)
        return logits.transpose(1, 2)


def class_weights_from_dataset(
    dataset: TCNSequenceDataset,
    *,
    num_classes: int,
    background_index: int = 0,
) -> torch.Tensor:
    counts = np.zeros(int(num_classes), dtype=np.float64)
    for labels in dataset.labels_by_session.values():
        if labels is None:
            continue
        values, curr = np.unique(labels, return_counts=True)
        counts[values.astype(int)] += curr
    weights = np.ones(int(num_classes), dtype=np.float32)
    classes = [idx for idx in range(int(num_classes)) if idx != int(background_index)]
    inv = 1.0 / np.maximum(counts[classes], 1.0)
    weights[classes] = inv / inv.mean()
    weights[int(background_index)] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def train_tcn(
    model: BehaviorTCN,
    dataset: TCNSequenceDataset,
    *,
    config: TCNTrainingConfig,
    ignore_index: int = 0,
) -> list[dict[str, float]]:
    device = resolve_device(config.device)
    model.to(device)
    loader = DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=max(0, int(config.num_workers)),
    )
    weights = None
    if config.class_weighting == "inverse_frequency":
        weights = class_weights_from_dataset(
            dataset, num_classes=len(dataset.label_names), background_index=ignore_index
        ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=int(ignore_index))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        amsgrad=True,
    )
    history: list[dict[str, float]] = []
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    with torch.enable_grad():
        for epoch in range(1, int(config.epochs) + 1):
            model.train()
            losses: list[float] = []
            skipped = 0
            for batch in loader:
                x = batch["features"].to(device)
                y = batch["labels"].to(device)
                if not torch.any(y != int(ignore_index)):
                    skipped += 1
                    continue
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(np.mean(losses)) if losses else float("nan"),
                    "skipped_batches": float(skipped),
                }
            )
    return history


def predict_tcn(
    model: BehaviorTCN,
    dataset: TCNSequenceDataset,
    *,
    device: str | torch.device = "auto",
) -> dict[str, np.ndarray]:
    device_t = (
        resolve_device(str(device)) if not isinstance(device, torch.device) else device
    )
    model.to(device_t)
    model.eval()
    pred_by_session: dict[str, np.ndarray] = {
        session_id: np.zeros(len(features), dtype=np.int64)
        for session_id, features in dataset.features_by_session.items()
    }
    score_by_session: dict[str, np.ndarray] = {
        session_id: np.zeros(
            (len(features), len(dataset.label_names)), dtype=np.float32
        )
        for session_id, features in dataset.features_by_session.items()
    }
    counts_by_session: dict[str, np.ndarray] = {
        session_id: np.zeros(len(features), dtype=np.float32)
        for session_id, features in dataset.features_by_session.items()
    }
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            session_id = str(batch["session_id"][0])
            start = int(batch["start"][0])
            valid_len = int(batch["valid_length"][0])
            logits = model(batch["features"].to(device_t))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0, :valid_len]
            stop = start + valid_len
            score_by_session[session_id][start:stop] += probs
            counts_by_session[session_id][start:stop] += 1.0
    for session_id, scores in score_by_session.items():
        counts = np.maximum(counts_by_session[session_id][:, None], 1.0)
        scores = scores / counts
        score_by_session[session_id] = scores
        pred_by_session[session_id] = scores.argmax(axis=1).astype(np.int64)
    return {"predictions": pred_by_session, "scores": score_by_session}


def evaluate_tcn_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: list[str],
    background_index: int = 0,
) -> dict[str, Any]:
    mask = y_true != int(background_index)
    labels = [idx for idx in range(len(label_names)) if idx != int(background_index)]
    if not np.any(mask):
        return {
            "macro_f1": 0.0,
            "accuracy_non_background": 0.0,
            "n_labeled_frames": 0,
            "per_class": {},
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true[mask],
        y_pred[mask],
        labels=labels,
        zero_division=0,
    )
    per_class = {
        label_names[label]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    return {
        "macro_f1": float(np.mean(f1)),
        "accuracy_non_background": float(accuracy_score(y_true[mask], y_pred[mask])),
        "n_labeled_frames": int(mask.sum()),
        "per_class": per_class,
    }


def evaluate_tcn(
    model: BehaviorTCN,
    dataset: TCNSequenceDataset,
    *,
    background_index: int = 0,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    pred = predict_tcn(model, dataset, device=device)["predictions"]
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for session_id, labels in dataset.labels_by_session.items():
        if labels is None:
            continue
        y_true.append(labels)
        y_pred.append(pred[session_id])
    if not y_true:
        return {
            "macro_f1": 0.0,
            "accuracy_non_background": 0.0,
            "n_labeled_frames": 0,
            "per_class": {},
        }
    return evaluate_tcn_predictions(
        np.concatenate(y_true),
        np.concatenate(y_pred),
        label_names=dataset.label_names,
        background_index=background_index,
    )


def save_tcn_checkpoint(
    path: str | Path,
    *,
    model: BehaviorTCN,
    run_config: TCNRunConfig,
    normalization: TCNNormalization | None,
    input_dim: int,
    label_names: list[str],
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "run_config": {
            **asdict(run_config),
            "sessions": [
                {
                    **asdict(session),
                    "features": str(session.features),
                    "labels": str(session.labels) if session.labels else None,
                }
                for session in run_config.sessions
            ],
        },
        "normalization": (
            {
                "mean": normalization.mean.tolist(),
                "std": normalization.std.tolist(),
            }
            if normalization is not None
            else None
        ),
        "input_dim": int(input_dim),
        "label_names": list(label_names),
    }
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_tcn_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[BehaviorTCN, dict[str, Any]]:
    device_t = (
        resolve_device(str(device)) if not isinstance(device, torch.device) else device
    )
    payload = torch.load(Path(path).expanduser(), map_location=device_t)
    model_cfg = TCNModelConfig(**payload["run_config"]["model"])
    model = BehaviorTCN(
        input_dim=int(payload["input_dim"]),
        num_classes=len(payload["label_names"]),
        config=model_cfg,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device_t)
    model.eval()
    return model, payload
