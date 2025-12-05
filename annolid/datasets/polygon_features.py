#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from annolid.datasets.polygon_utils import (
    behavior_label,
    flatten_points,
    frame_number_from_filename,
    load_annotation,
    normalize_polygon,
    polygons_by_label,
    polygon_area,
    polygon_centroid,
    polygon_perimeter,
    resample_polygon,
)
from annolid.utils.logger import logger

EXPECTED_LABELS: Tuple[str, str] = ("intruder", "resident")

__all__ = [
    "create_dataset",
    "load_frame_polygons",
    "resolve_motion_indices",
    "main",
]


def _load_tracking_csv(csv_path: Path, video_name: str) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        logger.warning(
            "Tracked CSV file '%s' missing for video '%s'. Motion indices will default to 0.",
            csv_path,
            video_name,
        )
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error loading tracking CSV '%s': %s", csv_path, exc)
        return None


def resolve_motion_indices(
    df_track: Optional[pd.DataFrame],
    frame_number: Optional[int],
    video_name: str,
) -> Tuple[float, float]:
    """Look up motion indices for intruder and resident for a given frame."""
    if df_track is None or frame_number is None:
        return 0.0, 0.0

    required_columns = {"frame_number", "instance_name", "motion_index"}
    if not required_columns.issubset(df_track.columns):
        logger.warning(
            "Tracking CSV for '%s' is missing expected columns %s", video_name, required_columns
        )
        return 0.0, 0.0

    rows = df_track[df_track["frame_number"] == frame_number]

    def _lookup(instance: str) -> float:
        instance_rows = rows[rows["instance_name"] == instance]
        if instance_rows.empty:
            logger.warning(
                "Motion index for '%s' not found for frame %s in video '%s'",
                instance,
                frame_number,
                video_name,
            )
            return 0.0
        return float(instance_rows.iloc[0].get("motion_index", 0) or 0)

    return _lookup("intruder"), _lookup("resident")


def load_frame_polygons(
    json_path: Path,
    *,
    normalize: bool = False,
    expected_labels: Sequence[str] = EXPECTED_LABELS,
) -> Optional[Tuple[str, List[List[float]], List[List[float]]]]:
    """Load behavior label and polygons for intruder/resident from a frame."""
    data = load_annotation(json_path)
    if data is None:
        return None

    label = behavior_label(data.get("flags", {}), context=str(json_path))
    if label is None:
        return None

    polygons = polygons_by_label(
        data.get("shapes", []),
        labels=expected_labels,
        context=str(json_path),
    )
    intruder_points = polygons.get("intruder", [])
    resident_points = polygons.get("resident", [])

    if normalize:
        intruder_points = normalize_polygon(intruder_points)
        resident_points = normalize_polygon(resident_points)

    return label, intruder_points, resident_points


def create_dataset(data_folder: Path, num_points: int, normalize: bool = False) -> pd.DataFrame:
    """Build a feature dataframe from polygon annotations and tracked motion indices."""
    records: List[Dict[str, object]] = []
    if not data_folder.is_dir():
        logger.error(
            "Data folder '%s' does not exist or is not a directory.", data_folder)
        return pd.DataFrame(records)

    for video_folder in sorted([p for p in data_folder.iterdir() if p.is_dir()]):
        logger.info("Processing video folder: '%s'", video_folder.name)
        tracked_csv_path = data_folder / f"{video_folder.name}_tracked.csv"
        df_track = _load_tracking_csv(tracked_csv_path, video_folder.name)

        prev_distance: Optional[float] = None
        prev_frame_number: Optional[int] = None

        for json_path in sorted(video_folder.glob("*.json")):
            frame_data = load_frame_polygons(json_path, normalize=normalize)
            if frame_data is None:
                continue
            label, intruder_points, resident_points = frame_data

            intruder_area = polygon_area(intruder_points)
            resident_area = polygon_area(resident_points)
            intruder_centroid = polygon_centroid(intruder_points)
            resident_centroid = polygon_centroid(resident_points)
            intruder_perimeter = polygon_perimeter(intruder_points)
            resident_perimeter = polygon_perimeter(resident_points)

            intruder_resampled = resample_polygon(
                intruder_points, num_points=num_points)
            resident_resampled = resample_polygon(
                resident_points, num_points=num_points)

            intruder_features = flatten_points(intruder_resampled)
            resident_features = flatten_points(resident_resampled)

            frame_number = frame_number_from_filename(json_path)
            intruder_motion_index, resident_motion_index = resolve_motion_indices(
                df_track, frame_number, video_folder.name
            )

            # Inter-animal distance (pixel space) between centroids.
            dx = intruder_centroid[0] - resident_centroid[0]
            dy = intruder_centroid[1] - resident_centroid[1]
            inter_animal_distance = float(math.hypot(dx, dy))

            # Relative velocity: positive when animals are moving towards each other.
            if prev_distance is not None and frame_number is not None and prev_frame_number is not None:
                dt = max(1, frame_number - prev_frame_number)
                relative_velocity = float(
                    (prev_distance - inter_animal_distance) / dt)
            else:
                relative_velocity = 0.0
            prev_distance = inter_animal_distance
            prev_frame_number = frame_number

            # Facing angle: angle between intruder "orientation" and vector to resident.
            # Use vector from intruder centroid to the furthest intruder point as an orientation proxy.
            if intruder_points and inter_animal_distance > 0:
                pts = np.asarray(intruder_points, dtype=float)
                centroid_arr = np.asarray(intruder_centroid, dtype=float)
                offsets = pts - centroid_arr
                dists = np.linalg.norm(offsets, axis=1)
                head_idx = int(np.argmax(dists))
                head_vec = offsets[head_idx]
                to_resident = np.asarray(
                    [resident_centroid[0] - intruder_centroid[0],
                     resident_centroid[1] - intruder_centroid[1]],
                    dtype=float,
                )
                head_norm = float(np.linalg.norm(head_vec))
                to_res_norm = float(np.linalg.norm(to_resident))
                if head_norm > 0.0 and to_res_norm > 0.0:
                    cos_angle = float(
                        np.clip(
                            np.dot(head_vec, to_resident) /
                            (head_norm * to_res_norm),
                            -1.0,
                            1.0,
                        )
                    )
                    facing_angle = float(math.acos(cos_angle))
                else:
                    facing_angle = 0.0
            else:
                facing_angle = 0.0

            record = {
                "video": video_folder.name,
                "frame": json_path.name,
                "label": label,
                "intruder_features": intruder_features,
                "resident_features": resident_features,
                "intruder_area": intruder_area,
                "resident_area": resident_area,
                "intruder_centroid": intruder_centroid,
                "resident_centroid": resident_centroid,
                "intruder_perimeter": intruder_perimeter,
                "resident_perimeter": resident_perimeter,
                "intruder_motion_index": intruder_motion_index,
                "resident_motion_index": resident_motion_index,
                "inter_animal_distance": inter_animal_distance,
                "relative_velocity": relative_velocity,
                "facing_angle": facing_angle,
                "frame_number": frame_number if frame_number is not None else -1,
            }
            records.append(record)

    return pd.DataFrame(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train and test datasets for behavior classification with fixed-length polygon features "
            "and additional geometric + motion features."
        )
    )
    parser.add_argument("--train_folder", type=str,
                        required=True, help="Path to the train folder")
    parser.add_argument("--test_folder", type=str,
                        required=True, help="Path to the test folder")
    parser.add_argument("--output_folder", type=str, default=".",
                        help="Folder to save the output CSV files")
    parser.add_argument("--num_points", type=int, default=10,
                        help="Number of points to resample each polygon to")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize polygon points (center them at the origin)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    logger.info("Starting dataset creation with enhanced polygon features.")

    train_df = create_dataset(Path(args.train_folder),
                              args.num_points, normalize=args.normalize)
    test_df = create_dataset(Path(args.test_folder),
                             args.num_points, normalize=args.normalize)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    train_csv = output_folder / "train_dataset.csv"
    test_csv = output_folder / "test_dataset.csv"

    try:
        train_df.to_csv(train_csv, index=False)
        logger.info("Train dataset saved to '%s'.", train_csv)
    except Exception as exc:
        logger.error("Error saving train dataset to '%s': %s", train_csv, exc)

    try:
        test_df.to_csv(test_csv, index=False)
        logger.info("Test dataset saved to '%s'.", test_csv)
    except Exception as exc:
        logger.error("Error saving test dataset to '%s': %s", test_csv, exc)

    logger.info("Dataset creation completed.")


if __name__ == "__main__":
    main()
