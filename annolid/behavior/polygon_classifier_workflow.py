"""Workflow helpers for polygon-frame behavior classifiers."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn import metrics

from annolid.behavior.models.polygon_frame_classifier import (
    ImprovedFrameLabelConvNet,
    ModelConfig,
    PolygonFeatureConfig,
    PolygonFrameDataset,
    TrainingConfig,
    _collate_fn,  # type: ignore
    _frame_sort_key,  # type: ignore
    train_polygon_frame_classifier,
)
from annolid.datasets.polygon_features import create_dataset
from annolid.datasets.polygon_utils import (
    frame_number_from_filename,
    load_annotation,
    polygon_area,
    polygon_centroid,
    polygon_perimeter,
    resample_polygon,
)

if TYPE_CHECKING:
    from annolid.behavior.tcn import TCNSession


@dataclass(frozen=True)
class PolygonDatasetOutcome:
    train_csv: str
    test_csv: str
    train_rows: int
    test_rows: int
    labels: tuple[str, ...]


@dataclass(frozen=True)
class PolygonPointsCSVOutcome:
    output_csv: str
    rows: int
    labels: tuple[str, ...]
    polygon_columns: tuple[str, ...]
    skipped_frames: int


@dataclass(frozen=True)
class PolygonVideoAssignment:
    annotation_dir: str
    label_csv: str


@dataclass(frozen=True)
class PolygonSplitCSVOutcome:
    csv: str
    rows: int
    labels: tuple[str, ...]
    polygon_columns: tuple[str, ...]
    videos: tuple[PolygonPointsCSVOutcome, ...]


@dataclass(frozen=True)
class PolygonTrainTestCSVOutcome:
    train: PolygonSplitCSVOutcome
    test: PolygonSplitCSVOutcome


@dataclass(frozen=True)
class PolygonTrainingOutcome:
    run_dir: str
    checkpoint_path: str
    metrics_path: str
    labels: tuple[str, ...]
    model_type: str = "convnet"


@dataclass(frozen=True)
class PolygonInferenceOutcome:
    output_csv: str
    rows: int
    labels: tuple[str, ...]
    model_type: str = "convnet"


def _slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "polygon"


def _truthy_label_columns(df: pd.DataFrame) -> list[str]:
    ignored = {
        "frame",
        "frame_number",
        "index",
        "unnamed_0",
        "unnamed: 0",
        "video",
        "timestamp",
        "time",
    }
    labels: list[str] = []
    for col in df.columns:
        if _slug(col) in ignored or str(col).startswith("Unnamed"):
            continue
        labels.append(str(col))
    return labels


def _manual_labels_by_frame(label_csv: str | Path) -> Dict[int, str]:
    path = Path(label_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manual label CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Manual label CSV is empty: {path}")

    frame_col = None
    for candidate in ("frame_number", "frame", "Unnamed: 0"):
        if candidate in df.columns:
            frame_col = candidate
            break
    if frame_col is None:
        frame_col = df.columns[0]

    label_cols = _truthy_label_columns(df)
    if not label_cols:
        raise ValueError(
            f"Manual label CSV has no behavior label columns after frame column: {path}"
        )

    labels: Dict[int, str] = {}
    for idx, row in df.iterrows():
        try:
            frame = int(row.get(frame_col, idx))
        except (TypeError, ValueError):
            frame = int(idx)
        active = []
        for col in label_cols:
            value = row.get(col, 0)
            try:
                is_active = float(value) > 0
            except (TypeError, ValueError):
                is_active = bool(value)
            if is_active:
                active.append(str(col))
        if active:
            labels[frame] = active[0]
    return labels


def _iter_labelme_jsons(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.glob("*.json")):
        name = path.name.lower()
        if name == "project.annolid.json" or name.endswith("_stats.json"):
            continue
        yield path


def _shape_points(shape: Mapping[str, Any]) -> list[list[float]]:
    points = shape.get("points") or []
    if not isinstance(points, list) or len(points) < 3:
        return []
    normalized: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return []
        try:
            normalized.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return []
    return normalized


def _is_polygon_shape(shape: Mapping[str, Any]) -> bool:
    shape_type = str(shape.get("shape_type") or "").strip().lower()
    if shape_type and shape_type != "polygon":
        return False
    return bool(_shape_points(shape))


def _record_has_polygon(record: Mapping[str, Any]) -> bool:
    return any(
        isinstance(shape, Mapping) and _is_polygon_shape(shape)
        for shape in (record.get("shapes") or [])
    )


def _frame_number_from_record(record: Mapping[str, Any]) -> int | None:
    for value in (record.get("frame"), record.get("frame_number")):
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    image_path = record.get("imagePath")
    if image_path:
        return frame_number_from_filename(Path(str(image_path)))
    return None


def _iter_ndjson_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for row_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid NDJSON record at {path}:{row_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                continue
            frame = _frame_number_from_record(record)
            if frame is None:
                continue
            yield int(frame), record


def _iter_annotation_payloads(
    folder: Path,
) -> Iterable[tuple[int, str, dict[str, Any]]]:
    """Yield one polygon annotation payload per frame from JSON plus NDJSON stores.

    Per-frame JSON remains authoritative when it contains polygon shapes. NDJSON
    records fill in predicted polygon frames that do not have usable JSON
    sidecars, which is the common output layout for long tracking runs.
    """
    records_by_frame: dict[int, tuple[str, dict[str, Any]]] = {}

    for json_path in _iter_labelme_jsons(folder):
        frame = frame_number_from_filename(json_path)
        if frame is None:
            continue
        data = load_annotation(json_path)
        if data is None or not _record_has_polygon(data):
            continue
        records_by_frame[int(frame)] = (json_path.name, data)

    for ndjson_path in sorted(folder.glob("*.ndjson")):
        for frame, record in _iter_ndjson_records(ndjson_path):
            if frame in records_by_frame or not _record_has_polygon(record):
                continue
            frame_name = record.get("imagePath") or f"{folder.name}_{frame:09d}.json"
            records_by_frame[int(frame)] = (str(frame_name), record)

    for frame in sorted(records_by_frame):
        frame_name, record = records_by_frame[frame]
        yield frame, frame_name, record


def generate_polygon_points_csv(
    *,
    annotation_dir: str | Path,
    label_csv: str | Path,
    output_csv: str | Path,
    num_points: int = 50,
    include_unlabeled: bool = False,
) -> PolygonPointsCSVOutcome:
    """Merge predicted polygon annotations with manual labels into a feature CSV."""
    ann_dir = Path(annotation_dir).expanduser().resolve()
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation folder not found: {ann_dir}")
    if int(num_points) <= 0:
        raise ValueError("num_points must be greater than 0.")

    labels_by_frame = _manual_labels_by_frame(label_csv)
    records: list[dict[str, object]] = []
    polygon_columns: set[str] = set()
    skipped = 0

    for frame, frame_name, data in _iter_annotation_payloads(ann_dir):
        behavior_label = labels_by_frame.get(frame)
        if not behavior_label and not include_unlabeled:
            skipped += 1
            continue

        record: dict[str, object] = {
            "video": ann_dir.name,
            "frame": frame_name,
            "frame_number": int(frame),
            "label": behavior_label or "",
        }
        found_polygon = False
        for shape in data.get("shapes", []) or []:
            if not isinstance(shape, Mapping) or not _is_polygon_shape(shape):
                continue
            points = _shape_points(shape)
            if len(points) < 3:
                continue
            shape_key = _slug(shape.get("label") or "polygon")
            resampled = resample_polygon(points, int(num_points))
            flat = [coord for point in resampled for coord in point]
            features_col = f"{shape_key}_features"
            polygon_columns.add(features_col)
            record[features_col] = flat
            record[f"{shape_key}_area"] = polygon_area(points)
            record[f"{shape_key}_centroid"] = polygon_centroid(points)
            record[f"{shape_key}_perimeter"] = polygon_perimeter(points)
            if shape.get("motion_index") is not None:
                try:
                    record[f"{shape_key}_motion_index"] = float(shape["motion_index"])
                except (TypeError, ValueError):
                    record[f"{shape_key}_motion_index"] = 0.0
            found_polygon = True

        if found_polygon:
            records.append(record)
        else:
            skipped += 1

    if not records:
        raise ValueError(
            "No polygon rows were created. Check the annotation folder, shape types, "
            "and manual label frame numbers."
        )

    out_path = Path(output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).sort_values(["video", "frame_number"]).to_csv(
        out_path, index=False
    )

    labels = tuple(sorted({str(row["label"]) for row in records if row.get("label")}))
    return PolygonPointsCSVOutcome(
        output_csv=str(out_path),
        rows=len(records),
        labels=labels,
        polygon_columns=tuple(sorted(polygon_columns)),
        skipped_frames=int(skipped),
    )


def _normalize_video_assignments(
    assignments: Iterable[PolygonVideoAssignment | tuple[str | Path, str | Path]],
) -> list[PolygonVideoAssignment]:
    normalized: list[PolygonVideoAssignment] = []
    for item in assignments:
        if isinstance(item, PolygonVideoAssignment):
            video_dir = item.annotation_dir
            label_csv = item.label_csv
        else:
            video_dir, label_csv = item
        normalized.append(
            PolygonVideoAssignment(
                annotation_dir=str(Path(video_dir).expanduser()),
                label_csv=str(Path(label_csv).expanduser()),
            )
        )
    return normalized


def generate_polygon_train_test_csvs(
    *,
    train_assignments: Iterable[PolygonVideoAssignment | tuple[str | Path, str | Path]],
    test_assignments: Iterable[PolygonVideoAssignment | tuple[str | Path, str | Path]],
    output_dir: str | Path,
    num_points: int = 50,
    include_unlabeled: bool = False,
) -> PolygonTrainTestCSVOutcome:
    """Create combined train/test polygon point CSVs from video annotation sets."""

    def feature_columns(columns: Iterable[str]) -> set[str]:
        return {str(col) for col in columns if str(col).endswith("_features")}

    def video_output_path(video_folder: str, split: str, index: int) -> Path:
        video_name = Path(video_folder).expanduser().name or f"{split}_{index}"
        return out_dir / f"{index:02d}_{video_name}_{split}_polygon_points.csv"

    def build_split(
        assignments: list[PolygonVideoAssignment], split: str
    ) -> PolygonSplitCSVOutcome:
        if not assignments:
            raise ValueError(f"No {split} videos were assigned.")

        outcomes: list[PolygonPointsCSVOutcome] = []
        frames: list[pd.DataFrame] = []
        expected_features: set[str] | None = None
        for index, assignment in enumerate(assignments, start=1):
            outcome = generate_polygon_points_csv(
                annotation_dir=assignment.annotation_dir,
                label_csv=assignment.label_csv,
                output_csv=video_output_path(assignment.annotation_dir, split, index),
                num_points=int(num_points),
                include_unlabeled=bool(include_unlabeled),
            )
            current_features = feature_columns(outcome.polygon_columns)
            if expected_features is None:
                expected_features = current_features
            elif current_features != expected_features:
                raise ValueError(
                    f"{split} video {assignment.annotation_dir} has polygon columns "
                    f"{sorted(current_features)}, expected {sorted(expected_features)}. "
                    "Use the same body-part polygon labels for every video in a split."
                )
            outcomes.append(outcome)
            frames.append(pd.read_csv(outcome.output_csv))

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined_csv = out_dir / f"{split}_polygon_points.csv"
        combined.to_csv(combined_csv, index=False)
        labels = tuple(
            sorted({label for outcome in outcomes for label in outcome.labels})
        )
        return PolygonSplitCSVOutcome(
            csv=str(combined_csv),
            rows=int(len(combined)),
            labels=labels,
            polygon_columns=tuple(sorted(expected_features or set())),
            videos=tuple(outcomes),
        )

    out_dir = Path(output_dir).expanduser().resolve()
    if int(num_points) <= 0:
        raise ValueError("num_points must be greater than 0.")
    out_dir.mkdir(parents=True, exist_ok=True)

    train = build_split(_normalize_video_assignments(train_assignments), "train")
    test = build_split(_normalize_video_assignments(test_assignments), "test")
    if set(train.polygon_columns) != set(test.polygon_columns):
        raise ValueError(
            "Train/test polygon feature columns do not match. "
            f"Train: {list(train.polygon_columns)}; test: {list(test.polygon_columns)}. "
            "Use compatible body-part polygon labels before training."
        )
    return PolygonTrainTestCSVOutcome(train=train, test=test)


def build_polygon_feature_dataset(
    *,
    train_folder: str | Path,
    test_folder: str | Path,
    output_folder: str | Path,
    num_points: int = 10,
    normalize: bool = False,
) -> PolygonDatasetOutcome:
    """Create train/test polygon feature CSV files from LabelMe frame folders."""
    train_path = Path(train_folder).expanduser().resolve()
    test_path = Path(test_folder).expanduser().resolve()
    out_dir = Path(output_folder).expanduser().resolve()

    if not train_path.is_dir():
        raise FileNotFoundError(f"Training folder not found: {train_path}")
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test folder not found: {test_path}")
    if int(num_points) <= 0:
        raise ValueError("num_points must be greater than 0.")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = create_dataset(train_path, int(num_points), normalize=bool(normalize))
    test_df = create_dataset(test_path, int(num_points), normalize=bool(normalize))

    if train_df.empty:
        raise ValueError(
            "No training rows were created. Check that LabelMe JSON files contain "
            "behavior flags and intruder/resident polygon labels."
        )
    if test_df.empty:
        raise ValueError(
            "No test rows were created. Check that LabelMe JSON files contain "
            "behavior flags and intruder/resident polygon labels."
        )

    train_csv = out_dir / "train_dataset.csv"
    test_csv = out_dir / "test_dataset.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    labels = tuple(
        sorted(
            {
                *map(str, train_df["label"].unique()),
                *map(str, test_df["label"].unique()),
            }
        )
    )
    return PolygonDatasetOutcome(
        train_csv=str(train_csv),
        test_csv=str(test_csv),
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        labels=labels,
    )


def _select_device(device: str | None = None) -> torch.device:
    token = str(device or "").strip().lower()
    if token:
        return torch.device(token)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict) or "model_state" not in state:
        raise ValueError(f"Invalid polygon classifier checkpoint: {checkpoint_path}")
    return state


def _checkpoint_model_config(state: Dict[str, Any]) -> ModelConfig:
    cfg = state.get("model_config")
    if isinstance(cfg, ModelConfig):
        return cfg
    if isinstance(cfg, dict):
        return ModelConfig(**cfg)
    return ModelConfig()


def _checkpoint_feature_config(state: Dict[str, Any]) -> PolygonFeatureConfig:
    cfg = state.get("feature_config")
    if isinstance(cfg, PolygonFeatureConfig):
        return cfg
    if isinstance(cfg, dict):
        return PolygonFeatureConfig(**cfg)
    return PolygonFeatureConfig()


def _checkpoint_labels(state: Dict[str, Any]) -> tuple[str, ...]:
    label_to_index = dict(state.get("label_to_index") or {})
    return tuple(
        label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])
    )


def _feature_config_from_columns(columns: Iterable[str]) -> PolygonFeatureConfig:
    column_set = set(map(str, columns))
    polygon_cols = tuple(sorted(col for col in column_set if col.endswith("_features")))
    if not polygon_cols:
        polygon_cols = PolygonFeatureConfig().polygon_cols
    known_core = {"video", "frame", "frame_number", "label", *polygon_cols}
    extra_cols = tuple(
        sorted(
            col
            for col in column_set
            if col not in known_core
            and (
                col.endswith("_area")
                or col.endswith("_centroid")
                or col.endswith("_perimeter")
                or col.endswith("_motion_index")
            )
        )
    )
    return PolygonFeatureConfig(
        polygon_cols=polygon_cols,
        extra_cols=extra_cols,
        normalize_features=False,
    )


def _feature_config_for_csv(path: Path) -> PolygonFeatureConfig:
    return _feature_config_from_columns(pd.read_csv(path, nrows=0).columns)


def _ordered_labels(*paths: Path) -> list[str]:
    labels: set[str] = set()
    for path in paths:
        df = pd.read_csv(path, usecols=["label"])
        labels.update(str(value) for value in df["label"].dropna().unique())
    ordered = sorted(labels)
    for background in ("background", "none", "other"):
        if background in ordered:
            ordered.remove(background)
            ordered.insert(0, background)
            break
    if not ordered:
        raise ValueError("Training CSVs must include at least one label.")
    return ordered


def _write_tcn_inputs_from_polygon_csv(
    *,
    csv_path: Path,
    output_dir: Path,
    feature_config: PolygonFeatureConfig,
    label_names: list[str],
    require_labels: bool,
) -> tuple[list["TCNSession"], dict[str, list[dict[str, Any]]]]:
    from annolid.behavior.tcn import TCNSession

    source_df = pd.read_csv(csv_path)
    if source_df.empty:
        raise ValueError(f"No data found in {csv_path}")

    dataset_csv_path = csv_path
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    if "label" not in source_df.columns or not set(source_df["label"]).issubset(
        set(label_to_index)
    ):
        if require_labels:
            raise ValueError(
                f"Feature CSV has labels outside the training set: {csv_path}"
            )
        temp_dir = tempfile.TemporaryDirectory(prefix="annolid_polygon_tcn_")
        dataset_csv_path = Path(temp_dir.name) / "features_with_dummy_labels.csv"
        dataset_df = source_df.copy()
        dataset_df["label"] = label_names[0]
        dataset_df.to_csv(dataset_csv_path, index=False)

    try:
        polygon_dataset = PolygonFrameDataset(
            dataset_csv_path,
            feature_config,
            window_size=1,
            label_to_index=label_to_index,
            normalization=None,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        sessions: list[TCNSession] = []
        rows_by_session: dict[str, list[dict[str, Any]]] = {}
        index_to_label = {idx: label for label, idx in label_to_index.items()}
        session_counts: dict[str, int] = {}

        for video, features in polygon_dataset.video_features.items():
            session_base = _slug(video)
            session_counts[session_base] = session_counts.get(session_base, 0) + 1
            session_id = (
                session_base
                if session_counts[session_base] == 1
                else f"{session_base}_{session_counts[session_base]}"
            )
            feature_path = output_dir / f"{session_id}_features.csv"
            label_path = output_dir / f"{session_id}_labels.csv"
            pd.DataFrame(
                features,
                columns=[f"feature_{idx}" for idx in range(features.shape[1])],
            ).to_csv(feature_path, index=False)

            labels = polygon_dataset.video_labels[video]
            label_rows = np.zeros((len(labels), len(label_names)), dtype=np.int64)
            label_rows[np.arange(len(labels)), labels] = 1
            pd.DataFrame(label_rows, columns=label_names).to_csv(
                label_path, index=False
            )

            source_rows = polygon_dataset.dataframe[
                polygon_dataset.dataframe["video"] == video
            ]
            if "frame" in source_rows.columns:
                source_rows = source_rows.sort_values(
                    by="frame", key=lambda series: series.map(_frame_sort_key)
                )
            elif "frame_number" in source_rows.columns:
                source_rows = source_rows.sort_values("frame_number")
            else:
                source_rows = source_rows.sort_index()
            rows_by_session[session_id] = [
                {
                    "video": row.get("video", video),
                    "frame": row.get("frame", ""),
                    "frame_number": int(row.get("frame_number", idx) or idx),
                    "label": index_to_label[int(labels[idx])]
                    if require_labels
                    else row.get("label", ""),
                }
                for idx, (_, row) in enumerate(source_rows.iterrows())
            ]
            sessions.append(
                TCNSession(
                    session_id=session_id,
                    features=feature_path,
                    labels=label_path if require_labels else None,
                    split="train",
                )
            )
        return sessions, rows_by_session
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _train_polygon_tcn_classifier(
    *,
    train_path: Path,
    test_path: Path,
    run_dir: Path,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    window_size: int,
    hidden_dim: int,
    num_residual_blocks: int,
    kernel_size: int | None,
    dropout: float,
    device: str | None,
) -> PolygonTrainingOutcome:
    from annolid.behavior.tcn import (
        BehaviorTCN,
        TCNFeatureConfig,
        TCNModelConfig,
        TCNRunConfig,
        TCNSequenceDataset,
        TCNTrainingConfig,
        evaluate_tcn,
        fit_normalization,
        save_tcn_checkpoint,
        train_tcn,
    )

    labels = _ordered_labels(train_path, test_path)
    feature_config = _feature_config_for_csv(train_path)
    train_sessions, _ = _write_tcn_inputs_from_polygon_csv(
        csv_path=train_path,
        output_dir=run_dir / "tcn_inputs" / "train",
        feature_config=feature_config,
        label_names=labels,
        require_labels=True,
    )
    test_sessions, _ = _write_tcn_inputs_from_polygon_csv(
        csv_path=test_path,
        output_dir=run_dir / "tcn_inputs" / "test",
        feature_config=feature_config,
        label_names=labels,
        require_labels=True,
    )
    train_sessions = [
        type(session)(session.session_id, session.features, session.labels, "train")
        for session in train_sessions
    ]
    test_sessions = [
        type(session)(session.session_id, session.features, session.labels, "test")
        for session in test_sessions
    ]
    training_config = TCNTrainingConfig(
        epochs=int(num_epochs),
        batch_size=int(batch_size),
        sequence_length=max(1, int(window_size)),
        learning_rate=float(learning_rate),
        device=str(_select_device(device)),
        num_workers=0,
    )
    model_config = TCNModelConfig(
        hidden_dim=int(hidden_dim),
        num_blocks=max(1, int(num_residual_blocks)),
        kernel_size=max(1, int(kernel_size or TCNModelConfig().kernel_size)),
        dropout=float(dropout),
    )
    run_config = TCNRunConfig(
        sessions=[*train_sessions, *test_sessions],
        labels=labels,
        feature=TCNFeatureConfig(input_type="features", zscore=True),
        model=model_config,
        training=training_config,
    )
    normalization = fit_normalization(
        train_sessions,
        feature_config=run_config.feature,
    )
    train_dataset = TCNSequenceDataset(
        train_sessions,
        feature_config=run_config.feature,
        label_names=labels,
        sequence_length=training_config.sequence_length,
        normalization=normalization,
    )
    background_index = labels.index("background") if "background" in labels else -100
    model = BehaviorTCN(
        input_dim=train_dataset.input_dim,
        num_classes=len(labels),
        config=model_config,
    )
    history = train_tcn(
        model,
        train_dataset,
        config=training_config,
        ignore_index=background_index,
    )
    checkpoint_path = run_dir / "polygon_tcn_classifier_best.pt"
    save_tcn_checkpoint(
        checkpoint_path,
        model=model,
        run_config=run_config,
        normalization=normalization,
        input_dim=train_dataset.input_dim,
        label_names=labels,
        metadata={
            "annolid_model_type": "polygon_tcn_classifier",
            "polygon_feature_config": asdict(feature_config),
        },
    )

    test_dataset = TCNSequenceDataset(
        test_sessions,
        feature_config=run_config.feature,
        label_names=labels,
        sequence_length=training_config.sequence_length,
        normalization=normalization,
    )
    test_metrics = evaluate_tcn(
        model,
        test_dataset,
        background_index=background_index,
        device=training_config.device,
    )
    inference = predict_polygon_classifier_csv(
        feature_csv=test_path,
        checkpoint_path=checkpoint_path,
        output_csv=run_dir / "test_predictions.csv",
        device=training_config.device,
    )
    metrics_payload: Dict[str, Any] = {
        "model_type": "tcn",
        "labels": list(inference.labels),
        "checkpoint_path": str(checkpoint_path),
        "train_csv": str(train_path),
        "test_csv": str(test_path),
        "run_dir": str(run_dir),
        "history": history,
        "test_metrics": test_metrics,
        "config": {
            "feature": asdict(feature_config),
            "model": asdict(model_config),
            "training": asdict(training_config),
        },
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return PolygonTrainingOutcome(
        run_dir=str(run_dir),
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        labels=tuple(labels),
        model_type="tcn",
    )


def _normalization_from_tcn_payload(payload: Mapping[str, Any]) -> Any:
    normalization_payload = payload.get("normalization")
    if not normalization_payload:
        return None
    from annolid.behavior.tcn import TCNNormalization

    return TCNNormalization(
        mean=np.asarray(normalization_payload["mean"], dtype=np.float32),
        std=np.asarray(normalization_payload["std"], dtype=np.float32),
    )


def _predict_polygon_tcn_classifier_csv(
    *,
    feature_csv: Path,
    checkpoint_path: Path,
    output_csv: str | Path,
    device: str,
    payload: Mapping[str, Any],
) -> PolygonInferenceOutcome:
    from annolid.behavior.tcn import (
        TCNFeatureConfig,
        TCNRunConfig,
        TCNSequenceDataset,
        load_tcn_checkpoint,
        predict_tcn,
    )

    label_names = [str(label) for label in payload.get("label_names", [])]
    if not label_names:
        raise ValueError("TCN checkpoint is missing label_names.")
    metadata = payload.get("metadata", {}) or {}
    polygon_cfg_payload = metadata.get("polygon_feature_config") or {}
    feature_config = (
        PolygonFeatureConfig(**polygon_cfg_payload)
        if polygon_cfg_payload
        else _feature_config_for_csv(feature_csv)
    )
    feature_config.normalize_features = False

    with tempfile.TemporaryDirectory(prefix="annolid_polygon_tcn_predict_") as tmp:
        sessions, rows_by_session = _write_tcn_inputs_from_polygon_csv(
            csv_path=feature_csv,
            output_dir=Path(tmp),
            feature_config=feature_config,
            label_names=label_names,
            require_labels=False,
        )
        run_config = TCNRunConfig.from_mapping(payload["run_config"])
        run_feature = run_config.feature
        if not isinstance(run_feature, TCNFeatureConfig):
            run_feature = TCNFeatureConfig(input_type="features", zscore=True)
        dataset = TCNSequenceDataset(
            sessions,
            feature_config=run_feature,
            label_names=label_names,
            sequence_length=run_config.training.sequence_length,
            normalization=_normalization_from_tcn_payload(payload),
            require_labels=False,
        )
        model, _payload = load_tcn_checkpoint(checkpoint_path, device=device)
        result = predict_tcn(model, dataset, device=device)
        predictions = result["predictions"]
        scores = result["scores"]

        records: list[dict[str, Any]] = []
        for session in sessions:
            session_id = session.session_id
            session_rows = rows_by_session.get(session_id, [])
            pred_values = predictions[session_id]
            score_values = scores[session_id]
            if len(session_rows) != len(pred_values):
                raise RuntimeError(
                    f"Prediction row count mismatch for {session_id}: "
                    f"{len(session_rows)} rows, {len(pred_values)} predictions."
                )
            for row, pred_idx, probs in zip(
                session_rows, pred_values.tolist(), score_values.tolist()
            ):
                pred_idx = int(pred_idx)
                record = {
                    "video": row.get("video", ""),
                    "frame": row.get("frame", ""),
                    "frame_number": int(row.get("frame_number", -1) or -1),
                    "predicted_label": label_names[pred_idx],
                    "confidence": float(probs[pred_idx]),
                }
                if row.get("label"):
                    record["label"] = row.get("label", "")
                for idx, label in enumerate(label_names):
                    record[f"prob_{label}"] = float(probs[idx])
                records.append(record)

        out_path = Path(output_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(records).to_csv(out_path, index=False)
        return PolygonInferenceOutcome(
            output_csv=str(out_path),
            rows=len(records),
            labels=tuple(label_names),
            model_type="tcn",
        )


def train_polygon_classifier(
    *,
    train_csv: str | Path,
    test_csv: str | Path,
    output_dir: str | Path,
    model_type: str = "convnet",
    run_name: str = "exp",
    num_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 4e-3,
    window_size: int = 11,
    hidden_dim: int = 128,
    num_residual_blocks: int = 6,
    kernel_size: int | None = None,
    dropout: float = 0.3,
    device: str | None = None,
) -> PolygonTrainingOutcome:
    """Train a polygon classifier and save checkpoint plus test metrics."""
    train_path = Path(train_csv).expanduser().resolve()
    test_path = Path(test_csv).expanduser().resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")
    model_kind = _slug(model_type)
    if model_kind in {"tcn_behavior", "temporal_convolutional_network"}:
        model_kind = "tcn"
    if model_kind not in {"convnet", "tcn"}:
        raise ValueError("model_type must be 'convnet' or 'tcn'.")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{run_name or 'exp'}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    if model_kind == "tcn":
        if int(num_epochs) == 30:
            num_epochs = 500
        if int(batch_size) == 64:
            batch_size = 8
        if float(learning_rate) == 4e-3:
            learning_rate = 1e-4
        if int(window_size) == 11:
            window_size = 1000
        if int(hidden_dim) == 128:
            hidden_dim = 32
        if int(num_residual_blocks) == 6:
            num_residual_blocks = 2
        if kernel_size is None:
            kernel_size = 9
        if float(dropout) == 0.3:
            dropout = 0.1
        return _train_polygon_tcn_classifier(
            train_path=train_path,
            test_path=test_path,
            run_dir=run_dir,
            num_epochs=int(num_epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            window_size=int(window_size),
            hidden_dim=int(hidden_dim),
            num_residual_blocks=int(num_residual_blocks),
            kernel_size=kernel_size,
            dropout=float(dropout),
            device=device,
        )

    model_config = ModelConfig(
        window_size=int(window_size),
        hidden_dim=int(hidden_dim),
        num_residual_blocks=int(num_residual_blocks),
        dropout=float(dropout),
    )
    training_config = TrainingConfig(
        num_epochs=int(num_epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        num_workers=0,
    )
    feature_config = _feature_config_for_csv(train_path)
    feature_config.normalize_features = True
    torch_device = _select_device(device)

    best_state = train_polygon_frame_classifier(
        train_path,
        feature_config=feature_config,
        model_config=model_config,
        training_config=training_config,
        device=torch_device,
        checkpoint_dir=run_dir,
    )

    checkpoint_path = run_dir / "polygon_frame_classifier_best.pt"
    torch.save(best_state, checkpoint_path)

    inference = predict_polygon_classifier_csv(
        feature_csv=test_path,
        checkpoint_path=checkpoint_path,
        output_csv=run_dir / "test_predictions.csv",
        device=str(torch_device),
    )
    predictions = pd.read_csv(inference.output_csv)
    truth = pd.read_csv(test_path)
    metrics_payload: Dict[str, Any] = {
        "labels": list(inference.labels),
        "checkpoint_path": str(checkpoint_path),
        "train_csv": str(train_path),
        "test_csv": str(test_path),
        "run_dir": str(run_dir),
        "config": {
            "feature": asdict(feature_config),
            "model": asdict(model_config),
            "training": asdict(training_config),
        },
    }
    if "label" in truth.columns and len(truth) == len(predictions):
        metrics_payload["accuracy"] = float(
            metrics.accuracy_score(truth["label"], predictions["predicted_label"])
        )
        metrics_payload["macro_f1"] = float(
            metrics.f1_score(
                truth["label"],
                predictions["predicted_label"],
                average="macro",
                zero_division=0,
            )
        )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return PolygonTrainingOutcome(
        run_dir=str(run_dir),
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        labels=inference.labels,
        model_type="convnet",
    )


def predict_polygon_classifier_csv(
    *,
    feature_csv: str | Path,
    checkpoint_path: str | Path,
    output_csv: str | Path,
    device: str | None = None,
) -> PolygonInferenceOutcome:
    """Run frame-level polygon classifier inference for a feature CSV."""
    csv_path = Path(feature_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    torch_device = _select_device(device)
    checkpoint_resolved = Path(checkpoint_path).expanduser().resolve()
    try:
        raw_state = torch.load(
            checkpoint_resolved, map_location=torch_device, weights_only=False
        )
    except TypeError:
        raw_state = torch.load(checkpoint_resolved, map_location=torch_device)
    if (
        isinstance(raw_state, dict)
        and raw_state.get("metadata", {}).get("annolid_model_type")
        == "polygon_tcn_classifier"
    ):
        return _predict_polygon_tcn_classifier_csv(
            feature_csv=csv_path,
            checkpoint_path=checkpoint_resolved,
            output_csv=output_csv,
            device=str(torch_device),
            payload=raw_state,
        )

    state = _load_checkpoint(checkpoint_resolved, torch_device)
    model_config = _checkpoint_model_config(state)
    feature_config = _checkpoint_feature_config(state)
    if state.get("polygon_lengths"):
        feature_config.polygon_pad_len = state["polygon_lengths"]

    label_to_index = dict(state.get("label_to_index") or {})
    if not label_to_index:
        raise ValueError("Checkpoint is missing label_to_index.")
    labels = _checkpoint_labels(state)

    source_df = pd.read_csv(csv_path)
    dataset_csv_path = csv_path
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if "label" not in source_df.columns or not set(source_df["label"]).issubset(
        set(label_to_index)
    ):
        if not labels:
            raise ValueError("Checkpoint does not define labels for inference.")
        temp_dir = tempfile.TemporaryDirectory(prefix="annolid_polygon_infer_")
        dataset_csv_path = Path(temp_dir.name) / "features_with_dummy_labels.csv"
        dataset_df = source_df.copy()
        dataset_df["label"] = labels[0]
        dataset_df.to_csv(dataset_csv_path, index=False)

    try:
        dataset = PolygonFrameDataset(
            dataset_csv_path,
            feature_config,
            window_size=model_config.window_size,
            label_to_index=label_to_index,
            normalization=state.get("normalization"),
        )
        model = ImprovedFrameLabelConvNet(
            feature_dim=int(state["feature_dim"]),
            num_classes=len(label_to_index),
            window_size=model_config.window_size,
            hidden_dim=model_config.hidden_dim,
            kernel_size=model_config.kernel_size,
            num_residual_blocks=model_config.num_residual_blocks,
            dropout=model_config.dropout,
            use_attention=model_config.use_attention,
        ).to(torch_device)
        model.load_state_dict(state["model_state"])
        model.eval()

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_fn,
        )
        all_probs: list[list[float]] = []
        with torch.no_grad():
            for inputs, _targets in loader:
                logits = model(inputs.to(torch_device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs.tolist())

        ordered_rows = []
        for _video, df in source_df.groupby("video", sort=False):
            if "frame" in df.columns:
                df_sorted = df.sort_values(
                    by="frame", key=lambda series: series.map(_frame_sort_key)
                )
            elif "frame_number" in df.columns:
                df_sorted = df.sort_values("frame_number")
            else:
                df_sorted = df.sort_index()
            for _, row in df_sorted.iterrows():
                ordered_rows.append(row)

        if len(ordered_rows) != len(all_probs):
            raise RuntimeError(
                f"Prediction row count mismatch: {len(ordered_rows)} rows, {len(all_probs)} predictions."
            )

        records = []
        for row, probs in zip(ordered_rows, all_probs):
            probs_arr = np.asarray(probs, dtype=float)
            pred_idx = int(probs_arr.argmax())
            record = {
                "video": row.get("video", ""),
                "frame": row.get("frame", ""),
                "frame_number": int(row.get("frame_number", -1) or -1),
                "predicted_label": labels[pred_idx],
                "confidence": float(probs_arr[pred_idx]),
            }
            if "label" in row:
                record["label"] = row.get("label", "")
            for idx, label in enumerate(labels):
                record[f"prob_{label}"] = float(probs_arr[idx])
            records.append(record)

        out_path = Path(output_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(records).to_csv(out_path, index=False)
        return PolygonInferenceOutcome(
            output_csv=str(out_path),
            rows=len(records),
            labels=labels,
            model_type="convnet",
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
