from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from annolid.utils.body_part_shapes import (
    is_anatomical_body_part_label,
    split_body_part_csv_instance_name,
)


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
    work = dataframe[["frame_number", "instance_name", x_col, y_col]].dropna(
        subset=["frame_number", "instance_name"]
    )
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    frames_num = pd.to_numeric(work["frame_number"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    xs_num = pd.to_numeric(work[x_col], errors="coerce").to_numpy(dtype=np.float64)
    ys_num = pd.to_numeric(work[y_col], errors="coerce").to_numpy(dtype=np.float64)
    inst_str = work["instance_name"].astype(str).str.strip()
    valid_mask = (
        np.isfinite(frames_num)
        & np.isfinite(xs_num)
        & np.isfinite(ys_num)
        & (inst_str != "").to_numpy(dtype=bool)
    )
    if not np.any(valid_mask):
        return pd.DataFrame(), pd.DataFrame()

    clean = pd.DataFrame(
        {
            "frame_number": frames_num[valid_mask].astype(np.int64),
            "instance_name": inst_str.iloc[np.flatnonzero(valid_mask)].to_numpy(),
            "x": xs_num[valid_mask],
            "y": ys_num[valid_mask],
        }
    )

    all_names = [str(n) for n in clean["instance_name"].unique().tolist()]
    base_parents = {n for n in all_names if ":" not in n}
    known_parents = base_parents if base_parents else None
    whole_names: set[str] = set()
    part_to_parent: dict[str, str] = {}
    for name in all_names:
        parent, part = split_body_part_csv_instance_name(
            name, known_parents=known_parents
        )
        if not part:
            if ":" not in name and is_anatomical_body_part_label(name):
                continue
            whole_names.add(name)
        elif parent:
            part_to_parent[name] = parent

    if not whole_names and not part_to_parent:
        return pd.DataFrame(), pd.DataFrame()

    whole_clean = (
        clean[clean["instance_name"].isin(whole_names)].drop_duplicates(
            subset=["frame_number", "instance_name"], keep="last"
        )
        if whole_names
        else clean.iloc[0:0].copy()
    )
    if part_to_parent:
        parts_clean = clean[clean["instance_name"].isin(part_to_parent)].copy()
        parts_clean["instance_name"] = parts_clean["instance_name"].map(part_to_parent)
        agg_parts = parts_clean.groupby(
            ["frame_number", "instance_name"], as_index=False, sort=False
        )[["x", "y"]].mean()
        if not whole_clean.empty:
            covered_keys = pd.MultiIndex.from_frame(
                whole_clean[["frame_number", "instance_name"]]
            )
            part_keys = pd.MultiIndex.from_frame(
                agg_parts[["frame_number", "instance_name"]]
            )
            agg_parts = agg_parts[~part_keys.isin(covered_keys)]
        clean = (
            pd.concat([whole_clean, agg_parts], ignore_index=True)
            if not agg_parts.empty
            else whole_clean
        )
    else:
        clean = whole_clean

    if clean.empty:
        return pd.DataFrame(), pd.DataFrame()

    clean = clean.drop_duplicates(
        subset=["frame_number", "instance_name"], keep="last"
    ).sort_values(["frame_number", "instance_name"], kind="mergesort")

    unique_names = sorted(clean["instance_name"].unique().tolist())
    n_inst = len(unique_names)
    if n_inst < 2:
        return pd.DataFrame(), pd.DataFrame()

    frame_vals = clean["frame_number"].to_numpy(dtype=np.int64)
    x_vals = clean["x"].to_numpy(dtype=np.float64)
    y_vals = clean["y"].to_numpy(dtype=np.float64)
    name_to_idx = {name: idx for idx, name in enumerate(unique_names)}
    inst_idx = np.fromiter(
        (name_to_idx[name] for name in clean["instance_name"]),
        dtype=np.int64,
        count=len(clean),
    )

    unique_frames, frame_inv, frame_counts = np.unique(
        frame_vals, return_inverse=True, return_counts=True
    )
    multi_frame_mask = frame_counts >= 2
    if not np.any(multi_frame_mask):
        return pd.DataFrame(), pd.DataFrame()

    if n_inst <= 64:
        valid_frames = unique_frames[multi_frame_mask]
        remap = np.full(len(unique_frames), -1, dtype=np.int64)
        remap[multi_frame_mask] = np.arange(len(valid_frames), dtype=np.int64)
        row_f = remap[frame_inv]
        keep = row_f >= 0
        row_f = row_f[keep]
        col_i = inst_idx[keep]

        n_f = len(valid_frames)
        x_grid = np.full((n_f, n_inst), np.nan, dtype=np.float64)
        y_grid = np.full((n_f, n_inst), np.nan, dtype=np.float64)
        x_grid[row_f, col_i] = x_vals[keep]
        y_grid[row_f, col_i] = y_vals[keep]

        iu, ju = np.triu_indices(n_inst, k=1)
        pair_dists = np.hypot(
            x_grid[:, ju] - x_grid[:, iu], y_grid[:, ju] - y_grid[:, iu]
        )
        f_pos, p_pos = np.nonzero(np.isfinite(pair_dists))
        if len(f_pos) == 0:
            return pd.DataFrame(), pd.DataFrame()

        d_flat = pair_dists[f_pos, p_pos]
        names_arr = np.asarray(unique_names, dtype=object)
        detail_df = pd.DataFrame(
            {
                "frame_number": valid_frames[f_pos],
                "instance_name_1": names_arr[iu[p_pos]],
                "instance_name_2": names_arr[ju[p_pos]],
                "distance_px": d_flat,
                "proximity_score": 1.0 / (1.0 + d_flat),
            }
        )

        full_dists = np.full((n_f, n_inst, n_inst), np.inf, dtype=np.float64)
        finite_pairs = np.where(np.isfinite(pair_dists), pair_dists, np.inf)
        full_dists[:, iu, ju] = finite_pairs
        full_dists[:, ju, iu] = finite_pairs

        summary_rows: list[dict[str, Any]] = []
        for idx, instance_name in enumerate(unique_names):
            inst_row_dists = full_dists[:, idx, :]
            min_per_frame = inst_row_dists.min(axis=1)
            has_nbr = np.isfinite(min_per_frame)
            if not np.any(has_nbr):
                continue
            d_inst = min_per_frame[has_nbr]
            nbr_indices = inst_row_dists[has_nbr, :].argmin(axis=1)
            counts = np.bincount(nbr_indices, minlength=n_inst)
            closest_idx = int(np.argmax(counts))
            all_pair_d = inst_row_dists[np.isfinite(inst_row_dists)]
            summary_rows.append(
                {
                    "instance_name": instance_name,
                    "mean_nearest_neighbor_distance_px": float(np.mean(d_inst)),
                    "min_nearest_neighbor_distance_px": float(np.min(d_inst)),
                    "mean_nearest_neighbor_proximity_score": float(
                        np.mean(1.0 / (1.0 + d_inst))
                    ),
                    "closest_neighbor_label": unique_names[closest_idx],
                    "neighbor_observations": int(len(d_inst)),
                    "mean_pairwise_distance_px": float(np.mean(all_pair_d)),
                    "min_pairwise_distance_px": float(np.min(all_pair_d)),
                    "max_pairwise_distance_px": float(np.max(all_pair_d)),
                    "mean_pairwise_proximity_score": float(
                        np.mean(1.0 / (1.0 + all_pair_d))
                    ),
                    "pairwise_observations": int(len(all_pair_d)),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.set_index("instance_name")
        return summary_df, detail_df

    # Fallback for large instance cardinality (n_inst > 64): slice contiguous frame blocks
    pair_rows: list[dict[str, Any]] = []
    nearest_stats: dict[str, list[float]] = {}
    all_pair_stats: dict[str, list[float]] = {}
    nearest_counts: dict[str, Counter[str]] = {}
    boundaries = np.concatenate(([0], np.cumsum(frame_counts)))
    names_col = clean["instance_name"].to_numpy()
    for grp_idx, count in enumerate(frame_counts.tolist()):
        if count < 2:
            continue
        start = int(boundaries[grp_idx])
        end = int(boundaries[grp_idx + 1])
        f_num = int(unique_frames[grp_idx])
        f_names = names_col[start:end]
        fx = x_vals[start:end]
        fy = y_vals[start:end]
        d_mat = np.hypot(fx[:, None] - fx[None, :], fy[:, None] - fy[None, :])
        for i_local, j_local in combinations(range(count), 2):
            dist = float(d_mat[i_local, j_local])
            left_name = str(f_names[i_local])
            right_name = str(f_names[j_local])
            pair_rows.append(
                {
                    "frame_number": f_num,
                    "instance_name_1": left_name,
                    "instance_name_2": right_name,
                    "distance_px": dist,
                    "proximity_score": float(1.0 / (1.0 + dist)),
                }
            )
            all_pair_stats.setdefault(left_name, []).append(dist)
            all_pair_stats.setdefault(right_name, []).append(dist)
        np.fill_diagonal(d_mat, np.inf)
        min_j = d_mat.argmin(axis=1)
        min_d = d_mat[np.arange(count), min_j]
        for i_local in range(count):
            src_name = str(f_names[i_local])
            nbr_name = str(f_names[int(min_j[i_local])])
            nearest_stats.setdefault(src_name, []).append(float(min_d[i_local]))
            nearest_counts.setdefault(src_name, Counter())[nbr_name] += 1

    summary_rows = []
    for instance_name in unique_names:
        distances = nearest_stats.get(instance_name)
        if not distances:
            continue
        d_arr = np.asarray(distances, dtype=np.float64)
        all_d_arr = np.asarray(
            all_pair_stats.get(instance_name, distances), dtype=np.float64
        )
        nbr_counter = nearest_counts[instance_name]
        closest_label = min(nbr_counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        summary_rows.append(
            {
                "instance_name": instance_name,
                "mean_nearest_neighbor_distance_px": float(np.mean(d_arr)),
                "min_nearest_neighbor_distance_px": float(np.min(d_arr)),
                "mean_nearest_neighbor_proximity_score": float(
                    np.mean(1.0 / (1.0 + d_arr))
                ),
                "closest_neighbor_label": closest_label,
                "neighbor_observations": int(len(d_arr)),
                "mean_pairwise_distance_px": float(np.mean(all_d_arr)),
                "min_pairwise_distance_px": float(np.min(all_d_arr)),
                "max_pairwise_distance_px": float(np.max(all_d_arr)),
                "mean_pairwise_proximity_score": float(
                    np.mean(1.0 / (1.0 + all_d_arr))
                ),
                "pairwise_observations": int(len(all_d_arr)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.set_index("instance_name")
    return summary_df, pd.DataFrame(pair_rows)
