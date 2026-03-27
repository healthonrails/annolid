from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _frame_number_from_name(path: Path) -> int | None:
    match = re.search(r"(\d+)(?=\.json$)", path.name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _extract_point(shape: Mapping[str, Any]) -> tuple[float, float] | None:
    points = shape.get("points") or []
    if not isinstance(points, list) or not points:
        return None
    first = points[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        return None
    try:
        return float(first[0]), float(first[1])
    except Exception:
        return None


def _extract_instance_name(shape: Mapping[str, Any]) -> str:
    for key in ("instance_label", "instance_name"):
        value = _normalize_text(shape.get(key))
        if value:
            return value
    for key in ("flags", "other_data", "otherData"):
        payload = shape.get(key)
        if not isinstance(payload, Mapping):
            continue
        value = _normalize_text(
            payload.get("instance_label") or payload.get("instance_name")
        )
        if value:
            return value
    label = _normalize_text(shape.get("label"))
    if ":" in label:
        return label.split(":", 1)[0].strip()
    return label


def _extract_keypoint_name(shape: Mapping[str, Any]) -> str:
    display = _normalize_text(shape.get("display_label"))
    if display:
        return display
    for key in ("flags", "other_data", "otherData"):
        payload = shape.get(key)
        if not isinstance(payload, Mapping):
            continue
        value = _normalize_text(
            payload.get("display_label") or payload.get("keypoint_label")
        )
        if value:
            return value
    label = _normalize_text(shape.get("label"))
    if ":" in label:
        return label.split(":", 1)[-1].strip()
    return label


def _is_zone_shape(shape: Mapping[str, Any]) -> bool:
    flags = shape.get("flags") or {}
    if isinstance(flags, Mapping):
        semantic_type = _normalize_text(flags.get("semantic_type"))
        if semantic_type == "zone":
            return True
    label = _normalize_text(shape.get("label"))
    description = _normalize_text(shape.get("description"))
    return "zone" in f"{label} {description}".lower()


def load_pose_keypoint_observations(
    video_path: str | Path,
    *,
    pose_folder: str | Path | None = None,
) -> pd.DataFrame:
    """Load per-frame point shapes and pivot them into a keypoint observation table."""

    video_path = Path(video_path)
    folder = (
        Path(pose_folder)
        if pose_folder is not None
        else video_path.parent / video_path.stem
    )
    if not folder.exists() or not folder.is_dir():
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for json_path in sorted(folder.glob("*.json")):
        frame_number = _frame_number_from_name(json_path)
        if frame_number is None:
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        shapes = payload.get("shapes") or []
        if not isinstance(shapes, list):
            continue

        per_instance: dict[str, dict[str, Any]] = {}
        for shape in shapes:
            if not isinstance(shape, Mapping):
                continue
            if str(shape.get("shape_type") or "").strip().lower() != "point":
                continue
            if _is_zone_shape(shape):
                continue
            point = _extract_point(shape)
            instance_name = _extract_instance_name(shape)
            keypoint_name = _extract_keypoint_name(shape)
            if point is None or not instance_name or not keypoint_name:
                continue
            row = per_instance.setdefault(
                instance_name,
                {"frame_number": frame_number, "instance_name": instance_name},
            )
            row[f"{keypoint_name}_x"] = point[0]
            row[f"{keypoint_name}_y"] = point[1]

        records.extend(per_instance.values())

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _centroid_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("cx_tracked", "cy_tracked"),
        ("cx_tracking", "cy_tracking"),
        ("cx", "cy"),
        ("x", "y"),
    ]
    for x_col, y_col in candidates:
        if x_col in dataframe.columns and y_col in dataframe.columns:
            return x_col, y_col
    raise KeyError(
        "Could not resolve centroid columns. Expected one of "
        "cx_tracked/cy_tracked, cx_tracking/cy_tracking, cx/cy, or x/y."
    )


def build_anchor_dataframe(
    tracking_df: pd.DataFrame,
    keypoint_df: pd.DataFrame | None = None,
    *,
    anchor_priority: Sequence[str] = ("nose", "snout", "muzzle", "head", "body_center"),
) -> pd.DataFrame:
    """Return a frame/instance table with an anchor point for social-zone analysis."""

    if tracking_df is None or tracking_df.empty:
        return pd.DataFrame()
    if (
        "frame_number" not in tracking_df.columns
        or "instance_name" not in tracking_df.columns
    ):
        return tracking_df.iloc[0:0].copy()

    df = tracking_df.copy()
    centroid_x, centroid_y = _centroid_columns(df)
    df["anchor_x"] = pd.to_numeric(df[centroid_x], errors="coerce")
    df["anchor_y"] = pd.to_numeric(df[centroid_y], errors="coerce")
    df["anchor_source"] = "centroid"

    if keypoint_df is not None and not keypoint_df.empty:
        kp_df = keypoint_df.copy()
        df = df.merge(kp_df, on=["frame_number", "instance_name"], how="left")

        for keypoint_name in anchor_priority:
            x_col = f"{keypoint_name}_x"
            y_col = f"{keypoint_name}_y"
            if x_col not in df.columns or y_col not in df.columns:
                continue
            valid = df[x_col].notna() & df[y_col].notna()
            df.loc[valid, "anchor_x"] = pd.to_numeric(
                df.loc[valid, x_col], errors="coerce"
            )
            df.loc[valid, "anchor_y"] = pd.to_numeric(
                df.loc[valid, y_col], errors="coerce"
            )
            df.loc[valid, "anchor_source"] = keypoint_name

        all_keypoint_pairs: list[tuple[str, str]] = []
        for column in df.columns:
            if not column.endswith("_x"):
                continue
            base = column[:-2]
            y_col = f"{base}_y"
            if y_col in df.columns:
                all_keypoint_pairs.append((column, y_col))

        for x_col, y_col in all_keypoint_pairs:
            missing = df["anchor_x"].isna() | df["anchor_y"].isna()
            if not missing.any():
                break
            valid = missing & df[x_col].notna() & df[y_col].notna()
            if not valid.any():
                continue
            df.loc[valid, "anchor_x"] = pd.to_numeric(
                df.loc[valid, x_col], errors="coerce"
            )
            df.loc[valid, "anchor_y"] = pd.to_numeric(
                df.loc[valid, y_col], errors="coerce"
            )
            df.loc[valid, "anchor_source"] = x_col[:-2]

    return df


def compute_pairwise_centroid_summary(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-instance nearest-neighbor summary and detailed per-frame distances."""

    if (
        dataframe.empty
        or "frame_number" not in dataframe.columns
        or "instance_name" not in dataframe.columns
    ):
        return pd.DataFrame(), pd.DataFrame()

    x_col, y_col = _centroid_columns(dataframe)
    pair_rows: list[dict[str, Any]] = []
    nearest_stats: dict[str, list[float]] = {}
    nearest_labels: dict[str, list[str]] = {}

    for frame_number, frame_group in dataframe.groupby("frame_number", sort=True):
        labels = frame_group["instance_name"].dropna().astype(str).tolist()
        if len(labels) < 2:
            continue
        coords = {
            str(row["instance_name"]): (float(row[x_col]), float(row[y_col]))
            for _, row in frame_group.iterrows()
            if pd.notna(row.get(x_col)) and pd.notna(row.get(y_col))
        }
        for left, right in combinations(sorted(coords.keys()), 2):
            x1, y1 = coords[left]
            x2, y2 = coords[right]
            distance = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            proximity_score = float(1.0 / (1.0 + distance))
            pair_rows.append(
                {
                    "frame_number": int(frame_number),
                    "instance_name_1": left,
                    "instance_name_2": right,
                    "distance_px": distance,
                    "proximity_score": proximity_score,
                }
            )
            nearest_stats.setdefault(left, []).append(distance)
            nearest_stats.setdefault(right, []).append(distance)
            nearest_labels.setdefault(left, []).append(right)
            nearest_labels.setdefault(right, []).append(left)

    summary_rows: list[dict[str, Any]] = []
    for instance_name, distances in nearest_stats.items():
        if not distances:
            continue
        summary_rows.append(
            {
                "instance_name": instance_name,
                "mean_nearest_neighbor_distance_px": float(
                    sum(distances) / len(distances)
                ),
                "min_nearest_neighbor_distance_px": float(min(distances)),
                "mean_nearest_neighbor_proximity_score": float(
                    sum(1.0 / (1.0 + distance) for distance in distances)
                    / len(distances)
                ),
                "closest_neighbor_label": max(
                    set(nearest_labels.get(instance_name, [])),
                    key=lambda label: nearest_labels.get(instance_name, []).count(
                        label
                    ),
                    default="",
                ),
                "neighbor_observations": len(distances),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.set_index("instance_name")
    return summary_df, pd.DataFrame(pair_rows)
