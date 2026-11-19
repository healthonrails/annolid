"""Calibrated trajectory kinematics, activity-state budgets, freezing bouts, and dyadic proximity."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import html
import io
from itertools import combinations
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter as _savgol_filter
except Exception:  # pragma: no cover - optional scientific dependency
    _savgol_filter = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None  # type: ignore[assignment]

from annolid.postprocessing.spatial_calibration import (
    CalibrationError,
    PlanarCalibration,
    resolve_planar_calibration,
)
from annolid.utils.annotation_store import (
    AnnotationStore,
    AnnotationStoreError,
    load_labelme_json,
)
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text
from annolid.utils.body_part_shapes import (
    is_anatomical_body_part_label,
    is_body_part_shape_payload,
    split_body_part_csv_instance_name,
)
from annolid.utils.zone_shapes import is_zone_shape_payload

__all__ = [
    "CalibrationError",
    "PlanarCalibration",
    "KinematicsComputationError",
    "InstanceKinematicsRow",
    "KinematicsBout",
    "PairwiseKinematicsRow",
    "KinematicsReport",
    "analyze_kinematics",
    "format_kinematics_report_table",
    "write_kinematics_report_bundle",
    "write_kinematics_events_csv",
]


class KinematicsComputationError(Exception):
    """Raised when kinematics input data cannot be loaded or parsed."""


@dataclass(frozen=True)
class InstanceKinematicsRow:
    """Per-instance calibrated kinematic, spatial, and activity-state summary."""

    instance: str
    observed_frames: int
    duration_s: float
    coverage_ratio: float
    total_distance_px: float
    total_distance_cm: Optional[float]
    net_displacement_px: float
    net_displacement_cm: Optional[float]
    straightness_index: float
    tortuosity: float
    mean_speed_px_s: float
    median_speed_px_s: float
    p95_speed_px_s: float
    max_speed_px_s: float
    mean_speed_cm_s: Optional[float]
    max_speed_cm_s: Optional[float]
    mean_accel_px_s2: float
    max_accel_px_s2: float
    mean_angular_velocity_deg_s: float
    median_area_px: float
    median_area_cm2: Optional[float]
    freezing_seconds: float
    freezing_percent: float
    freezing_bout_count: int
    resting_percent: float
    moderate_activity_percent: float
    high_activity_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance": self.instance,
            "observed_frames": int(self.observed_frames),
            "duration_s": round(float(self.duration_s), 6),
            "coverage_ratio": round(float(self.coverage_ratio), 6),
            "total_distance_px": round(float(self.total_distance_px), 6),
            "total_distance_cm": (
                round(float(self.total_distance_cm), 6)
                if self.total_distance_cm is not None
                else None
            ),
            "net_displacement_px": round(float(self.net_displacement_px), 6),
            "net_displacement_cm": (
                round(float(self.net_displacement_cm), 6)
                if self.net_displacement_cm is not None
                else None
            ),
            "straightness_index": round(float(self.straightness_index), 6),
            "tortuosity": round(float(self.tortuosity), 6),
            "mean_speed_px_s": round(float(self.mean_speed_px_s), 6),
            "median_speed_px_s": round(float(self.median_speed_px_s), 6),
            "p95_speed_px_s": round(float(self.p95_speed_px_s), 6),
            "max_speed_px_s": round(float(self.max_speed_px_s), 6),
            "mean_speed_cm_s": (
                round(float(self.mean_speed_cm_s), 6)
                if self.mean_speed_cm_s is not None
                else None
            ),
            "max_speed_cm_s": (
                round(float(self.max_speed_cm_s), 6)
                if self.max_speed_cm_s is not None
                else None
            ),
            "mean_accel_px_s2": round(float(self.mean_accel_px_s2), 6),
            "max_accel_px_s2": round(float(self.max_accel_px_s2), 6),
            "mean_angular_velocity_deg_s": round(
                float(self.mean_angular_velocity_deg_s), 6
            ),
            "median_area_px": round(float(self.median_area_px), 6),
            "median_area_cm2": (
                round(float(self.median_area_cm2), 6)
                if self.median_area_cm2 is not None
                else None
            ),
            "freezing_seconds": round(float(self.freezing_seconds), 6),
            "freezing_percent": round(float(self.freezing_percent), 6),
            "freezing_bout_count": int(self.freezing_bout_count),
            "resting_percent": round(float(self.resting_percent), 6),
            "moderate_activity_percent": round(
                float(self.moderate_activity_percent), 6
            ),
            "high_activity_percent": round(float(self.high_activity_percent), 6),
        }


@dataclass(frozen=True)
class KinematicsBout:
    """Contiguous kinematic or dyadic event bout (freezing, high_activity, close_contact)."""

    instance: str
    bout_type: str
    start_frame: int
    end_frame: int
    anchor_frame: int
    start_s: float
    end_s: float
    duration_s: float
    peak_value: float
    mean_value: float
    partner: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance": self.instance,
            "partner": self.partner,
            "bout_type": self.bout_type,
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "anchor_frame": int(self.anchor_frame),
            "start_s": round(float(self.start_s), 6),
            "end_s": round(float(self.end_s), 6),
            "duration_s": round(float(self.duration_s), 6),
            "peak_value": round(float(self.peak_value), 6),
            "mean_value": round(float(self.mean_value), 6),
        }


@dataclass(frozen=True)
class PairwiseKinematicsRow:
    """Dyadic inter-animal distance, close-contact, and velocity-coupling summary."""

    instance_a: str
    instance_b: str
    shared_frames: int
    min_distance_px: float
    mean_distance_px: float
    median_distance_px: float
    min_distance_cm: Optional[float]
    mean_distance_cm: Optional[float]
    close_contact_frames: int
    close_contact_seconds: float
    close_contact_bouts: int
    velocity_correlation: Optional[float]
    close_contact_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_a": self.instance_a,
            "instance_b": self.instance_b,
            "shared_frames": int(self.shared_frames),
            "min_distance_px": round(float(self.min_distance_px), 6),
            "mean_distance_px": round(float(self.mean_distance_px), 6),
            "median_distance_px": round(float(self.median_distance_px), 6),
            "min_distance_cm": (
                round(float(self.min_distance_cm), 6)
                if self.min_distance_cm is not None
                else None
            ),
            "mean_distance_cm": (
                round(float(self.mean_distance_cm), 6)
                if self.mean_distance_cm is not None
                else None
            ),
            "close_contact_frames": int(self.close_contact_frames),
            "close_contact_seconds": round(float(self.close_contact_seconds), 6),
            "close_contact_percent": round(float(self.close_contact_percent), 6),
            "close_contact_bouts": int(self.close_contact_bouts),
            "velocity_correlation": (
                round(float(self.velocity_correlation), 6)
                if self.velocity_correlation is not None
                else None
            ),
        }


@dataclass
class KinematicsReport:
    """Complete trajectory kinematics, freezing, and dyadic summary report."""

    source_path: str
    source_kind: str
    fps: float
    px_per_cm: Optional[float]
    frame_min: Optional[int]
    frame_max: Optional[int]
    total_frame_span: int
    instances: Tuple[InstanceKinematicsRow, ...]
    bouts: Tuple[KinematicsBout, ...]
    pairwise: Tuple[PairwiseKinematicsRow, ...]
    trajectories: Dict[str, List[Tuple[int, float, float, float]]] = field(
        default_factory=dict, repr=False
    )
    calibration: Optional[PlanarCalibration] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "kinematics_report/v1",
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "fps": round(float(self.fps), 6),
            "px_per_cm": (
                round(float(self.px_per_cm), 6) if self.px_per_cm is not None else None
            ),
            "calibration": (
                self.calibration.to_dict() if self.calibration is not None else None
            ),
            "frame_min": self.frame_min,
            "frame_max": self.frame_max,
            "total_frame_span": int(self.total_frame_span),
            "instances": [row.to_dict() for row in self.instances],
            "bouts": [bout.to_dict() for bout in self.bouts],
            "pairwise": [pair.to_dict() for pair in self.pairwise],
        }


def _polygon_centroid_and_area(
    points: Sequence[Sequence[Any]], shape_type: str
) -> Optional[Tuple[float, float, float]]:
    if not points:
        return None
    coords = []
    for pt in points:
        if not isinstance(pt, Sequence) or len(pt) < 2:
            continue
        try:
            x_v, y_v = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_v) and math.isfinite(y_v):
            coords.append((x_v, y_v))
    if not coords:
        return None
    stype = str(shape_type or "polygon").strip().lower()
    if stype == "rectangle" and len(coords) >= 2:
        (x1, y1), (x2, y2) = coords[0], coords[1]
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2), abs(x2 - x1) * abs(y2 - y1)
    if stype == "circle" and len(coords) >= 2:
        (cx, cy), (px, py) = coords[0], coords[1]
        radius = math.hypot(px - cx, py - cy)
        return float(cx), float(cy), float(math.pi * radius * radius)
    if len(coords) < 3:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys)), 0.0
    area2 = 0.0
    cx_acc = 0.0
    cy_acc = 0.0
    n = len(coords)
    for idx in range(n):
        x0, y0 = coords[idx]
        x1, y1 = coords[(idx + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx_acc += (x0 + x1) * cross
        cy_acc += (y0 + y1) * cross
    area = abs(area2) * 0.5
    if area <= 1e-6:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return float(sum(xs) / n), float(sum(ys) / n), 0.0
    cx = cx_acc / (3.0 * area2)
    cy = cy_acc / (3.0 * area2)
    return float(cx), float(cy), float(area)


def _probe_video_fps_and_frames(
    video_path: Path,
) -> Tuple[Optional[float], Optional[int]]:
    if not video_path.is_file():
        return None, None
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None
        fps_val = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fc_val = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return (fps_val if fps_val > 0 else None), (fc_val if fc_val > 0 else None)
    except Exception:
        return None, None


def _resolve_kinematics_source(
    source_path: Path,
) -> Tuple[Path, str, Optional[Path]]:
    """Resolve a video file, CSV file, or LabelMe directory to its tracking data source."""
    if not source_path.exists():
        raise FileNotFoundError(f"Kinematics input not found: {source_path}")
    if source_path.is_dir():
        # Check for sibling video file
        vid_cand = None
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            maybe_v = source_path.with_suffix(ext)
            if maybe_v.is_file():
                vid_cand = maybe_v
                break
        return source_path, "labelme_dir", vid_cand

    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        stem = source_path.stem
        for tag in ("_tracked", "_tracking"):
            if stem.endswith(tag):
                base = stem[: -len(tag)]
                for ext in (".mp4", ".avi", ".mov", ".mkv"):
                    cand = source_path.with_name(base + ext)
                    if cand.is_file():
                        return source_path, "tracking_csv", cand
        return source_path, "tracking_csv", None

    # Assume video file: prefer *_tracked.csv, then *_tracking.csv, then <stem>/ dir
    for tag in ("_tracked.csv", "_tracking.csv"):
        csv_cand = source_path.with_name(f"{source_path.stem}{tag}")
        if csv_cand.is_file():
            return csv_cand, "tracking_csv", source_path
    dir_cand = source_path.with_suffix("")
    if dir_cand.is_dir():
        return dir_cand, "labelme_dir", source_path
    raise KinematicsComputationError(
        f"No tracking CSV or annotation directory found for video: {source_path}"
    )


def _load_observations_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "frame_number" not in df.columns and "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "frame_number"})
    if "frame_number" not in df.columns or "instance_name" not in df.columns:
        raise KinematicsComputationError(
            f"Tracking CSV '{csv_path}' must contain 'frame_number' and 'instance_name' columns."
        )

    x_col = y_col = None
    for xc, yc in (("cx_tracked", "cy_tracked"), ("cx", "cy"), ("x", "y")):
        if xc in df.columns and yc in df.columns:
            x_col, y_col = xc, yc
            break
    if x_col is None and {"x1", "y1", "x2", "y2"}.issubset(df.columns):
        df = df.copy()
        df["_cx"] = 0.5 * (
            pd.to_numeric(df["x1"], errors="coerce")
            + pd.to_numeric(df["x2"], errors="coerce")
        )
        df["_cy"] = 0.5 * (
            pd.to_numeric(df["y1"], errors="coerce")
            + pd.to_numeric(df["y2"], errors="coerce")
        )
        x_col, y_col = "_cx", "_cy"
    if x_col is None or y_col is None:
        raise KinematicsComputationError(
            f"Tracking CSV '{csv_path}' has no centroid or bounding-box columns."
        )

    work = pd.DataFrame(
        {
            "frame": pd.to_numeric(df["frame_number"], errors="coerce"),
            "instance": df["instance_name"].fillna("").astype(str).str.strip(),
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
        }
    )
    if "mask_area" in df.columns:
        work["area"] = pd.to_numeric(df["mask_area"], errors="coerce").fillna(0.0)
    elif {"x1", "y1", "x2", "y2"}.issubset(df.columns):
        w = (
            pd.to_numeric(df["x2"], errors="coerce")
            - pd.to_numeric(df["x1"], errors="coerce")
        ).abs()
        h = (
            pd.to_numeric(df["y2"], errors="coerce")
            - pd.to_numeric(df["y1"], errors="coerce")
        ).abs()
        work["area"] = (w * h).fillna(0.0)
        # Exclude zero-size keypoint point rows (x1==x2 and y1==y2) when parent boxes exist
        is_point_row = (w <= 1e-6) & (h <= 1e-6)
        if (~is_point_row).any():
            work = work.loc[~is_point_row].copy()
    else:
        work["area"] = 0.0

    if "motion_index" in df.columns:
        work["motion_index"] = pd.to_numeric(df["motion_index"], errors="coerce")
    else:
        work["motion_index"] = np.nan

    work = work.dropna(subset=["frame", "x", "y"])
    work = work[work["instance"] != ""].copy()
    if work.empty:
        return work
    work["frame"] = work["frame"].astype(np.int64)

    # Collapse '<parent>:<body_part>' rows only if parent itself is missing on that frame
    all_names = work["instance"].unique().tolist()
    base_parents = {
        n for n in all_names if ":" not in n and not is_anatomical_body_part_label(n)
    }
    known_parents = base_parents if base_parents else None
    whole_rows = []
    part_map: Dict[str, str] = {}
    for name in all_names:
        parent, part = split_body_part_csv_instance_name(
            name, known_parents=known_parents
        )
        if part and parent:
            part_map[name] = parent
        elif not is_anatomical_body_part_label(name):
            whole_rows.append(name)

    if part_map:
        w_df = work[work["instance"].isin(whole_rows)].copy()
        p_df = work[work["instance"].isin(part_map)].copy()
        p_df["instance"] = p_df["instance"].map(part_map)
        agg_p = p_df.groupby(["frame", "instance"], as_index=False).agg(
            x=("x", "mean"),
            y=("y", "mean"),
            area=("area", "sum"),
            motion_index=("motion_index", "mean"),
        )
        if not w_df.empty:
            covered = pd.MultiIndex.from_frame(w_df[["frame", "instance"]])
            cand = pd.MultiIndex.from_frame(agg_p[["frame", "instance"]])
            agg_p = agg_p[~cand.isin(covered)]
        work = pd.concat([w_df, agg_p], ignore_index=True)

    work = work.sort_values(["instance", "frame"], kind="mergesort")
    work = work.drop_duplicates(subset=["instance", "frame"], keep="last")
    return work.reset_index(drop=True)


def _load_observations_from_labelme_dir(folder: Path) -> pd.DataFrame:
    frame_paths: Dict[int, Path] = {}
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".json"):
            continue
        if name.endswith("_tracking_stats.json") or name.endswith("_zones.json"):
            continue
        m = re.search(r"(\d+)(?=\.json$)", name)
        if m is not None:
            frame_paths[int(m.group(1))] = folder / name

    store = AnnotationStore.for_frame_path(folder / f"{folder.name}_000000000.json")
    if store.store_path.is_file():
        for fno in store.iter_frames():
            frame_paths.setdefault(
                int(fno), folder / f"{folder.name}_{int(fno):09d}.json"
            )

    rows: List[Dict[str, Any]] = []
    for fnum, fpath in sorted(frame_paths.items()):
        try:
            payload = load_labelme_json(fpath)
        except (FileNotFoundError, AnnotationStoreError, ValueError):
            continue
        if not isinstance(payload, Mapping):
            continue
        shapes = payload.get("shapes") or []
        if not isinstance(shapes, Sequence):
            continue
        for sh in shapes:
            if not isinstance(sh, Mapping) or is_zone_shape_payload(sh):
                continue
            stype = str(sh.get("shape_type") or "polygon").strip().lower()
            if stype == "point" or is_body_part_shape_payload(sh):
                continue
            inst = str(sh.get("instance_label") or sh.get("label") or "").strip()
            if not inst or is_anatomical_body_part_label(inst):
                continue
            geom = _polygon_centroid_and_area(sh.get("points") or [], stype)
            if geom is None:
                continue
            cx, cy, area = geom
            rows.append(
                {
                    "frame": int(fnum),
                    "instance": inst,
                    "x": cx,
                    "y": cy,
                    "area": area,
                    "motion_index": np.nan,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["frame", "instance", "x", "y", "area", "motion_index"]
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["instance", "frame", "area"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["instance", "frame"], keep="first")
    return df.sort_values(["instance", "frame"]).reset_index(drop=True)


def _extract_contiguous_bouts(
    frames: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    *,
    instance: str,
    bout_type: str,
    fps: float,
    min_frames: int = 1,
    max_gap_frames: int = 1,
    partner: Optional[str] = None,
    prefer_min_for_anchor: bool = False,
) -> List[KinematicsBout]:
    idx_active = np.flatnonzero(mask)
    if len(idx_active) == 0:
        return []
    bouts: List[KinematicsBout] = []
    run_start = int(idx_active[0])
    run_prev = run_start

    def _emit(i0: int, i1: int) -> None:
        f0 = int(frames[i0])
        f1 = int(frames[i1])
        span_frames = f1 - f0 + 1
        if span_frames < min_frames:
            return
        seg_vals = values[i0 : i1 + 1]
        finite = seg_vals[np.isfinite(seg_vals)]
        if len(finite) == 0:
            peak_v = 0.0
            mean_v = 0.0
            anchor_f = f0
        else:
            mean_v = float(np.mean(finite))
            if prefer_min_for_anchor:
                peak_v = float(np.min(finite))
                rel = int(np.nanargmin(seg_vals))
            else:
                peak_v = float(np.max(finite))
                rel = int(np.nanargmax(seg_vals))
            anchor_f = int(frames[i0 + rel])
        bouts.append(
            KinematicsBout(
                instance=instance,
                partner=partner,
                bout_type=bout_type,
                start_frame=f0,
                end_frame=f1,
                anchor_frame=anchor_f,
                start_s=float(f0) / fps,
                end_s=float(f1 + 1) / fps,
                duration_s=float(span_frames) / fps,
                peak_value=peak_v,
                mean_value=mean_v,
            )
        )

    for idx_cur in idx_active[1:]:
        cur = int(idx_cur)
        if int(frames[cur]) - int(frames[run_prev]) <= max(1, int(max_gap_frames)):
            run_prev = cur
        else:
            _emit(run_start, run_prev)
            run_start = cur
            run_prev = cur
    _emit(run_start, run_prev)
    return bouts


def _savgol_smooth_1d(arr: np.ndarray, win: int, polyorder: int) -> np.ndarray:
    """Apply Savitzky-Golay 1D smoothing with polynomial boundary fitting (SciPy-optional)."""
    if _savgol_filter is not None:
        return np.asarray(
            _savgol_filter(arr, window_length=win, polyorder=polyorder),
            dtype=np.float64,
        )
    half = win // 2
    t_local = np.arange(-half, half + 1, dtype=np.float64)
    vander = np.vander(t_local, N=polyorder + 1, increasing=True)
    kernel = np.linalg.pinv(vander)[0][::-1]
    out = np.convolve(arr, kernel, mode="same").astype(np.float64)
    # Match scipy mode='interp' at edges via least-squares polynomial fit
    t_edge = np.arange(win, dtype=np.float64)
    left_poly = np.polyfit(t_edge, arr[:win], deg=polyorder)
    out[:half] = np.polyval(left_poly, t_edge[:half])
    right_poly = np.polyfit(t_edge, arr[-win:], deg=polyorder)
    out[-half:] = np.polyval(right_poly, t_edge[-half:])
    return out


def _savgol_smooth_contiguous(
    frames: np.ndarray, arr: np.ndarray, win: int, polyorder: int
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing strictly within contiguous 1-frame runs."""
    n = len(arr)
    if win < 5 or n < win:
        return arr
    out = np.asarray(arr, dtype=np.float64).copy()
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))
    for s_idx, e_idx in zip(starts.tolist(), ends.tolist()):
        if e_idx - s_idx >= win:
            out[s_idx:e_idx] = _savgol_smooth_1d(out[s_idx:e_idx], win, polyorder)
    return out


def analyze_kinematics(
    source_path: Path | str,
    *,
    fps: Optional[float] = None,
    px_per_cm: Optional[float] = None,
    calibration: PlanarCalibration | Path | str | Mapping[str, Any] | None = None,
    smooth_window: int = 5,
    freeze_speed_threshold: Optional[float] = None,
    min_freeze_duration_s: float = 0.5,
    close_contact_distance_px: float = 80.0,
) -> KinematicsReport:
    """Compute calibrated kinematics, activity budgets, freezing bouts, and pairwise proximity."""
    raw_source = Path(source_path).expanduser().resolve()
    resolved_data, source_kind, video_path = _resolve_kinematics_source(raw_source)

    probed_fps, probed_total_frames = (
        _probe_video_fps_and_frames(video_path)
        if video_path is not None
        else (None, None)
    )
    eff_fps = (
        float(fps) if (fps is not None and float(fps) > 0) else (probed_fps or 30.0)
    )
    plan_cal = resolve_planar_calibration(calibration=calibration, px_per_cm=px_per_cm)
    cal = plan_cal.px_per_cm_nominal if plan_cal is not None else None

    if source_kind == "tracking_csv":
        obs_df = _load_observations_from_csv(resolved_data)
    else:
        obs_df = _load_observations_from_labelme_dir(resolved_data)

    if obs_df.empty:
        return KinematicsReport(
            source_path=str(resolved_data),
            source_kind=source_kind,
            fps=eff_fps,
            px_per_cm=cal,
            frame_min=None,
            frame_max=None,
            total_frame_span=probed_total_frames or 0,
            instances=(),
            bouts=(),
            pairwise=(),
            trajectories={},
            calibration=plan_cal,
        )

    frame_min = int(obs_df["frame"].min())
    frame_max = int(obs_df["frame"].max())
    observed_span = max(1, frame_max - frame_min + 1)
    total_span = max(observed_span, int(probed_total_frames or 0))

    # Default freezing threshold: 1.0 cm/s when calibrated, else 15.0 px/s
    if plan_cal is not None:
        eff_freeze_cm_s = (
            float(freeze_speed_threshold)
            if (
                freeze_speed_threshold is not None
                and float(freeze_speed_threshold) >= 0
            )
            else 1.0
        )
        eff_freeze_px_s = eff_freeze_cm_s * plan_cal.px_per_cm_nominal
    else:
        eff_freeze_cm_s = None
        eff_freeze_px_s = (
            float(freeze_speed_threshold)
            if (
                freeze_speed_threshold is not None
                and float(freeze_speed_threshold) >= 0
            )
            else 15.0
        )
    resting_px_s = eff_freeze_px_s * 2.0
    min_freeze_frames = max(
        1, int(round(max(0.0, float(min_freeze_duration_s)) * eff_fps))
    )

    instance_rows: List[InstanceKinematicsRow] = []
    all_bouts: List[KinematicsBout] = []
    trajectories: Dict[str, List[Tuple[int, float, float, float]]] = {}
    speed_series_by_inst: Dict[str, pd.Series] = {}

    for inst_name, grp in obs_df.groupby("instance", sort=True):
        frames = grp["frame"].to_numpy(dtype=np.int64)
        xs = grp["x"].to_numpy(dtype=np.float64)
        ys = grp["y"].to_numpy(dtype=np.float64)
        areas = grp["area"].to_numpy(dtype=np.float64)
        n_obs = len(frames)

        # Smooth coordinates slightly on contiguous runs when requested
        win = int(smooth_window)
        if win >= 3 and win % 2 == 0:
            win += 1
        if win >= 5 and n_obs >= win:
            xs_eval = _savgol_smooth_contiguous(frames, xs, win, min(2, win - 1))
            ys_eval = _savgol_smooth_contiguous(frames, ys, win, min(2, win - 1))
        else:
            xs_eval = xs
            ys_eval = ys

        speeds_px_s = np.full(n_obs, np.nan, dtype=np.float64)
        accels_px_s2 = np.full(n_obs, np.nan, dtype=np.float64)
        ang_vel_deg_s = np.full(n_obs, np.nan, dtype=np.float64)
        step_dists_px = np.zeros(n_obs, dtype=np.float64)

        speeds_cm_s: Optional[np.ndarray] = None
        step_dists_cm: Optional[np.ndarray] = None
        world_x_raw: Optional[np.ndarray] = None
        world_y_raw: Optional[np.ndarray] = None
        if plan_cal is not None:
            world_x_raw, world_y_raw = plan_cal.project_points(xs, ys)
            world_x_eval, world_y_eval = plan_cal.project_points(xs_eval, ys_eval)
            speeds_cm_s = np.full(n_obs, np.nan, dtype=np.float64)
            step_dists_cm = np.zeros(n_obs, dtype=np.float64)
        else:
            world_x_eval = None
            world_y_eval = None

        if n_obs >= 2:
            df_steps = np.diff(frames).astype(np.float64)
            dt = np.maximum(df_steps / eff_fps, 1.0 / eff_fps)
            dx = np.diff(xs_eval)
            dy = np.diff(ys_eval)
            dists = np.hypot(dx, dy)
            # Only accumulate path distance & speed across reasonable frame gaps
            valid_step = df_steps <= max(5.0, eff_fps)
            step_dists_px[1:] = np.where(valid_step, dists, 0.0)
            step_speeds = np.where(valid_step, dists / dt, np.nan)
            speeds_px_s[1:] = step_speeds
            speeds_px_s[0] = step_speeds[0]

            if (
                plan_cal is not None
                and world_x_eval is not None
                and world_y_eval is not None
                and speeds_cm_s is not None
                and step_dists_cm is not None
            ):
                dists_w = np.hypot(np.diff(world_x_eval), np.diff(world_y_eval))
                step_dists_cm[1:] = np.where(valid_step, dists_w, 0.0)
                step_speeds_w = np.where(valid_step, dists_w / dt, np.nan)
                speeds_cm_s[1:] = step_speeds_w
                speeds_cm_s[0] = step_speeds_w[0]

            if n_obs >= 3:
                both_valid = valid_step[:-1] & valid_step[1:]
                accels_px_s2[2:] = np.where(
                    both_valid, np.abs(np.diff(step_speeds)) / dt[1:], np.nan
                )
                headings_dx = (
                    np.diff(world_x_eval)
                    if (plan_cal is not None and world_x_eval is not None)
                    else dx
                )
                headings_dy = (
                    np.diff(world_y_eval)
                    if (plan_cal is not None and world_y_eval is not None)
                    else dy
                )
                headings = np.degrees(np.arctan2(headings_dy, headings_dx))
                d_head = (np.diff(headings) + 180.0) % 360.0 - 180.0
                ang_vel_deg_s[2:] = np.where(
                    both_valid, np.abs(d_head) / dt[1:], np.nan
                )

        tot_dist_px = float(np.sum(step_dists_px))
        net_disp_px = (
            float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) if n_obs >= 2 else 0.0
        )
        tot_dist_cm: Optional[float] = (
            float(np.sum(step_dists_cm)) if step_dists_cm is not None else None
        )
        net_disp_cm: Optional[float] = (
            (
                float(
                    math.hypot(
                        world_x_raw[-1] - world_x_raw[0],
                        world_y_raw[-1] - world_y_raw[0],
                    )
                )
                if n_obs >= 2
                else 0.0
            )
            if (world_x_raw is not None and world_y_raw is not None)
            else None
        )

        # Use physical arena trajectory when calibrated so perspective tilt never biases straightness
        ref_tot = tot_dist_cm if tot_dist_cm is not None else tot_dist_px
        ref_net = net_disp_cm if net_disp_cm is not None else net_disp_px
        straightness = min(1.0, ref_net / ref_tot) if ref_tot > 1e-6 else 0.0
        tortuosity = (
            ref_tot / max(ref_net, 1e-6 if tot_dist_cm is not None else 1.0)
            if ref_tot > 1e-6
            else 1.0
        )

        finite_spd_px_mask = np.isfinite(speeds_px_s)
        finite_spd_px = speeds_px_s[finite_spd_px_mask]
        p95_spd = (
            float(np.percentile(finite_spd_px, 95)) if len(finite_spd_px) > 0 else 0.0
        )
        if speeds_cm_s is not None and eff_freeze_cm_s is not None:
            finite_spd_cm_mask = np.isfinite(speeds_cm_s)
            finite_spd_cm = speeds_cm_s[finite_spd_cm_mask]
            resting_cm_s = eff_freeze_cm_s * 2.0
            p95_cm_s = (
                float(np.percentile(finite_spd_cm, 95))
                if len(finite_spd_cm) > 0
                else 0.0
            )
            high_cm_s = max(resting_cm_s * 2.0, p95_cm_s)
            is_freezing_raw = finite_spd_cm_mask & (speeds_cm_s <= eff_freeze_cm_s)
            valid_spd_mask = finite_spd_cm_mask
        else:
            finite_spd_cm = None
            resting_cm_s = None
            high_cm_s = None
            high_thresh = max(resting_px_s * 2.0, p95_spd)
            is_freezing_raw = finite_spd_px_mask & (speeds_px_s <= eff_freeze_px_s)
            valid_spd_mask = finite_spd_px_mask

        freeze_bouts = _extract_contiguous_bouts(
            frames,
            speeds_px_s,
            is_freezing_raw,
            instance=str(inst_name),
            bout_type="freezing",
            fps=eff_fps,
            min_frames=min_freeze_frames,
            max_gap_frames=2,
            prefer_min_for_anchor=True,
        )
        freeze_frame_set: set[int] = set()
        for fb in freeze_bouts:
            freeze_frame_set.update(range(fb.start_frame, fb.end_frame + 1))
        freeze_mask = (
            np.isin(frames, list(freeze_frame_set))
            if freeze_frame_set
            else np.zeros(n_obs, dtype=bool)
        )
        dur_s = float(n_obs) / eff_fps
        freeze_s = min(dur_s, float(np.sum(freeze_mask)) / eff_fps)

        eval_mask = valid_spd_mask | freeze_mask
        n_eval = int(np.sum(eval_mask))

        if (
            speeds_cm_s is not None
            and resting_cm_s is not None
            and high_cm_s is not None
        ):
            high_mask = (
                eval_mask
                & (~freeze_mask)
                & valid_spd_mask
                & (speeds_cm_s >= high_cm_s)
                & (speeds_cm_s > resting_cm_s)
            )
            rest_mask = (
                eval_mask
                & (~freeze_mask)
                & valid_spd_mask
                & (speeds_cm_s <= resting_cm_s)
            )
        else:
            high_mask = (
                eval_mask
                & (~freeze_mask)
                & valid_spd_mask
                & (speeds_px_s >= high_thresh)
                & (speeds_px_s > resting_px_s)
            )
            rest_mask = (
                eval_mask
                & (~freeze_mask)
                & valid_spd_mask
                & (speeds_px_s <= resting_px_s)
            )

        high_bouts = _extract_contiguous_bouts(
            frames,
            speeds_px_s,
            high_mask,
            instance=str(inst_name),
            bout_type="high_activity",
            fps=eff_fps,
            min_frames=max(1, int(round(0.2 * eff_fps))),
            max_gap_frames=2,
        )

        mod_mask = eval_mask & (~freeze_mask) & (~rest_mask) & (~high_mask)

        if n_eval > 0:
            denom_eval = float(n_eval)
            freeze_pct = float(np.sum(freeze_mask)) * 100.0 / denom_eval
            rest_pct = float(np.sum(rest_mask)) * 100.0 / denom_eval
            mod_pct = float(np.sum(mod_mask)) * 100.0 / denom_eval
            high_pct = float(np.sum(high_mask)) * 100.0 / denom_eval
        else:
            freeze_pct = 0.0
            rest_pct = 100.0 if n_obs > 0 else 0.0
            mod_pct = 0.0
            high_pct = 0.0

        med_area_px = float(np.median(areas)) if n_obs > 0 else 0.0
        med_area_cm2: Optional[float] = (
            float(np.median(plan_cal.transform_areas(xs, ys, areas)))
            if (plan_cal is not None and n_obs > 0)
            else (0.0 if plan_cal is not None else None)
        )
        mean_spd_px = float(np.mean(finite_spd_px)) if len(finite_spd_px) > 0 else 0.0
        med_spd_px = float(np.median(finite_spd_px)) if len(finite_spd_px) > 0 else 0.0
        max_spd_px = float(np.max(finite_spd_px)) if len(finite_spd_px) > 0 else 0.0
        mean_spd_cm: Optional[float] = (
            float(np.mean(finite_spd_cm))
            if (finite_spd_cm is not None and len(finite_spd_cm) > 0)
            else (0.0 if speeds_cm_s is not None else None)
        )
        max_spd_cm: Optional[float] = (
            float(np.max(finite_spd_cm))
            if (finite_spd_cm is not None and len(finite_spd_cm) > 0)
            else (0.0 if speeds_cm_s is not None else None)
        )
        finite_accels = accels_px_s2[np.isfinite(accels_px_s2)]
        finite_ang_vel = ang_vel_deg_s[np.isfinite(ang_vel_deg_s)]

        instance_rows.append(
            InstanceKinematicsRow(
                instance=str(inst_name),
                observed_frames=n_obs,
                duration_s=dur_s,
                coverage_ratio=min(1.0, float(n_obs) / float(total_span)),
                total_distance_px=tot_dist_px,
                total_distance_cm=tot_dist_cm,
                net_displacement_px=net_disp_px,
                net_displacement_cm=net_disp_cm,
                straightness_index=straightness,
                tortuosity=tortuosity,
                mean_speed_px_s=mean_spd_px,
                median_speed_px_s=med_spd_px,
                p95_speed_px_s=p95_spd,
                max_speed_px_s=max_spd_px,
                mean_speed_cm_s=mean_spd_cm,
                max_speed_cm_s=max_spd_cm,
                mean_accel_px_s2=(
                    float(np.mean(finite_accels)) if len(finite_accels) > 0 else 0.0
                ),
                max_accel_px_s2=(
                    float(np.max(finite_accels)) if len(finite_accels) > 0 else 0.0
                ),
                mean_angular_velocity_deg_s=(
                    float(np.mean(finite_ang_vel)) if len(finite_ang_vel) > 0 else 0.0
                ),
                median_area_px=med_area_px,
                median_area_cm2=med_area_cm2,
                freezing_seconds=freeze_s,
                freezing_percent=freeze_pct,
                freezing_bout_count=len(freeze_bouts),
                resting_percent=rest_pct,
                moderate_activity_percent=mod_pct,
                high_activity_percent=high_pct,
            )
        )
        all_bouts.extend(freeze_bouts)
        all_bouts.extend(high_bouts)

        # Downsample trajectory points to at most 1500 points per animal for lightweight SVG/plots
        stride = max(1, n_obs // 1500)
        trajectories[str(inst_name)] = [
            (
                int(frames[idx]),
                float(xs[idx]),
                float(ys[idx]),
                float(speeds_px_s[idx]) if np.isfinite(speeds_px_s[idx]) else 0.0,
            )
            for idx in range(0, n_obs, stride)
        ]
        speed_series_by_inst[str(inst_name)] = pd.Series(
            speeds_cm_s if speeds_cm_s is not None else speeds_px_s,
            index=frames,
            dtype=np.float64,
        ).dropna()

    # Pairwise inter-animal kinematics
    pairwise_rows: List[PairwiseKinematicsRow] = []
    by_inst_df = {
        str(name): g[["frame", "x", "y"]].reset_index(drop=True)
        for name, g in obs_df.groupby("instance", sort=True)
    }
    inst_names = sorted(by_inst_df.keys())
    contact_px = max(0.0, float(close_contact_distance_px))

    for inst_a, inst_b in combinations(inst_names, 2):
        merged = pd.merge(
            by_inst_df[inst_a],
            by_inst_df[inst_b],
            on="frame",
            suffixes=("_a", "_b"),
        )
        if merged.empty:
            continue
        p_frames = merged["frame"].to_numpy(dtype=np.int64)
        xa = merged["x_a"].to_numpy(dtype=np.float64)
        ya = merged["y_a"].to_numpy(dtype=np.float64)
        xb = merged["x_b"].to_numpy(dtype=np.float64)
        yb = merged["y_b"].to_numpy(dtype=np.float64)
        dists = np.hypot(xb - xa, yb - ya)
        if plan_cal is not None:
            world_xa, world_ya = plan_cal.project_points(xa, ya)
            world_xb, world_yb = plan_cal.project_points(xb, yb)
            dists_cm_arr = np.hypot(world_xb - world_xa, world_yb - world_ya)
            contact_cm = contact_px / plan_cal.px_per_cm_nominal
            close_mask = dists_cm_arr <= contact_cm
            min_d_cm: Optional[float] = float(np.min(dists_cm_arr))
            mean_d_cm: Optional[float] = float(np.mean(dists_cm_arr))
        else:
            close_mask = dists <= contact_px
            min_d_cm = None
            mean_d_cm = None
        contact_bouts = _extract_contiguous_bouts(
            p_frames,
            dists,
            close_mask,
            instance=inst_a,
            partner=inst_b,
            bout_type="close_contact",
            fps=eff_fps,
            min_frames=max(1, int(round(0.1 * eff_fps))),
            max_gap_frames=2,
            prefer_min_for_anchor=True,
        )
        all_bouts.extend(contact_bouts)
        close_frames = int(np.sum(close_mask))

        vel_corr: Optional[float] = None
        sa = speed_series_by_inst.get(inst_a)
        sb = speed_series_by_inst.get(inst_b)
        if sa is not None and sb is not None:
            aligned_a, aligned_b = sa.align(sb, join="inner")
            if len(aligned_a) >= 3:
                va = aligned_a.to_numpy(dtype=np.float64)
                vb = aligned_b.to_numpy(dtype=np.float64)
                if np.std(va) > 1e-9 and np.std(vb) > 1e-9:
                    c_val = float(np.corrcoef(va, vb)[0, 1])
                    if math.isfinite(c_val):
                        vel_corr = c_val

        min_d = float(np.min(dists))
        mean_d = float(np.mean(dists))
        med_d = float(np.median(dists))
        pairwise_rows.append(
            PairwiseKinematicsRow(
                instance_a=inst_a,
                instance_b=inst_b,
                shared_frames=len(p_frames),
                min_distance_px=min_d,
                mean_distance_px=mean_d,
                median_distance_px=med_d,
                min_distance_cm=min_d_cm,
                mean_distance_cm=mean_d_cm,
                close_contact_frames=close_frames,
                close_contact_seconds=float(close_frames) / eff_fps,
                close_contact_bouts=len(contact_bouts),
                velocity_correlation=vel_corr,
                close_contact_percent=(
                    100.0 * float(close_frames) / float(len(p_frames))
                    if len(p_frames) > 0
                    else 0.0
                ),
            )
        )

    all_bouts.sort(key=lambda b: (b.start_frame, b.end_frame, b.instance, b.bout_type))

    return KinematicsReport(
        source_path=str(resolved_data),
        source_kind=source_kind,
        fps=eff_fps,
        px_per_cm=cal,
        frame_min=frame_min,
        frame_max=frame_max,
        total_frame_span=total_span,
        instances=tuple(instance_rows),
        bouts=tuple(all_bouts),
        pairwise=tuple(pairwise_rows),
        trajectories=trajectories,
        calibration=plan_cal,
    )


def format_kinematics_report_table(report: KinematicsReport) -> str:
    """Render a human-readable plain-text kinematics summary."""
    if report.calibration is not None:
        cal_str = report.calibration.summary_label()
    elif report.px_per_cm is not None:
        cal_str = f"{report.px_per_cm:.2f} px/cm"
    else:
        cal_str = "uncalibrated (px)"
    lines = [
        f"Source: {report.source_path} ({report.source_kind})",
        f"FPS: {report.fps:.2f} | Calibration: {cal_str} | Span: {report.total_frame_span} frame(s)",
    ]
    if not report.instances:
        lines.append("No tracked animal trajectories found.")
        return "\n".join(lines)

    dist_hdr = "Dist(cm)" if report.px_per_cm else "Dist(px)"
    spd_hdr = "MeanSpd(cm/s)" if report.px_per_cm else "MeanSpd(px/s)"
    lines.append("")
    lines.append(
        f"{'Instance':<16} {'Frames':>7} {'Cov%':>6} {dist_hdr:>11} "
        f"{spd_hdr:>14} {'Freeze%':>8} {'HighAct%':>9}"
    )
    lines.append("-" * 78)
    for row in report.instances:
        dist_val = (
            row.total_distance_cm
            if row.total_distance_cm is not None
            else row.total_distance_px
        )
        spd_val = (
            row.mean_speed_cm_s
            if row.mean_speed_cm_s is not None
            else row.mean_speed_px_s
        )
        lines.append(
            f"{row.instance:<16} {row.observed_frames:>7d} {row.coverage_ratio * 100.0:>5.1f}% "
            f"{dist_val:>11.2f} {spd_val:>14.2f} {row.freezing_percent:>7.1f}% "
            f"{row.high_activity_percent:>8.1f}%"
        )

    if report.pairwise:
        lines.append("")
        lines.append("Pairwise Inter-Animal Proximity:")
        for p in report.pairwise:
            corr_txt = (
                f", vel_r={p.velocity_correlation:+.2f}"
                if p.velocity_correlation is not None
                else ""
            )
            cm_txt = (
                f" ({p.mean_distance_cm:.2f}cm)"
                if p.mean_distance_cm is not None
                else ""
            )
            lines.append(
                f"  {p.instance_a} <-> {p.instance_b}: mean={p.mean_distance_px:.1f}px{cm_txt}, "
                f"min={p.min_distance_px:.1f}px, close_contact={p.close_contact_seconds:.2f}s "
                f"({p.close_contact_bouts} bout(s)){corr_txt}"
            )
    return "\n".join(lines)


def write_kinematics_events_csv(
    report: KinematicsReport, output_path: Path | str
) -> Path:
    """Export detected kinematic bouts as a canonical Annolid Behavior Event CSV."""
    dst = Path(output_path)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Recording time", "Subject", "Behavior", "Event", "Modifiers"])
    events: List[Tuple[float, str, str, str, str]] = []
    for bout in report.bouts:
        mod = f"partner={bout.partner}" if bout.partner else ""
        events.append((bout.start_s, bout.instance, bout.bout_type, "start", mod))
        events.append((bout.end_s, bout.instance, bout.bout_type, "stop", mod))
    events.sort(key=lambda item: (item[0], 0 if item[3] == "stop" else 1, item[1]))
    for rec_t, subj, beh, ev_kind, mod in events:
        writer.writerow([f"{rec_t:.4f}", subj, beh, ev_kind, mod])
    atomic_write_text(dst, buf.getvalue(), encoding="utf-8")
    return dst


def _render_kinematics_svg_plots(
    report: KinematicsReport, output_dir: Path
) -> Dict[str, str]:
    """Render deterministic standalone vector SVG plots without Matplotlib."""
    plots: Dict[str, str] = {}
    palette = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2")

    if report.trajectories:
        all_pts = [pt for pts in report.trajectories.values() for pt in pts]
        if all_pts:
            w, h, pad = 680, 520, 48
            min_x = min(p[1] for p in all_pts)
            max_x = max(p[1] for p in all_pts)
            min_y = min(p[2] for p in all_pts)
            max_y = max(p[2] for p in all_pts)
            span_x = max(1.0, max_x - min_x)
            span_y = max(1.0, max_y - min_y)
            scale = min((w - 2 * pad) / span_x, (h - 2 * pad - 24) / span_y)
            svg_lines = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                f'<text x="{pad}" y="28" font-family="sans-serif" font-size="15" font-weight="600" fill="#1e293b">Tracked Trajectories</text>',
                f'<rect x="{pad}" y="40" width="{w - 2 * pad}" height="{h - 2 * pad - 16}" fill="#f8fafc" stroke="#cbd5e1"/>',
            ]
            for idx, (inst_name, pts) in enumerate(sorted(report.trajectories.items())):
                if not pts:
                    continue
                color = palette[idx % len(palette)]
                coords = " ".join(
                    f"{pad + (p[1] - min_x) * scale:.1f},{40 + (p[2] - min_y) * scale:.1f}"
                    for p in pts
                )
                sx = pad + (pts[0][1] - min_x) * scale
                sy = 40 + (pts[0][2] - min_y) * scale
                svg_lines.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="1.8" opacity="0.88" points="{coords}"/>'
                )
                svg_lines.append(
                    f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="{color}"/>'
                )
                lx = pad + (idx % 4) * 145
                ly = h - 14
                svg_lines.append(
                    f'<text x="{lx}" y="{ly}" font-family="sans-serif" font-size="11" fill="{color}">● {html.escape(inst_name)}</text>'
                )
            svg_lines.append("</svg>")
            traj_svg = output_dir / "kinematics_trajectories.svg"
            atomic_write_text(traj_svg, "\n".join(svg_lines) + "\n", encoding="utf-8")
            plots["trajectory_plot"] = str(traj_svg)

    w2 = 680
    row_h = 34
    h2 = max(160, 76 + len(report.instances) * row_h)
    bar_left = 150
    bar_w = w2 - bar_left - 36
    b_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w2}" height="{h2}" viewBox="0 0 {w2} {h2}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="15" font-weight="600" fill="#1e293b">Activity &amp; Freezing Budget by Instance</text>',
        '<text x="150" y="48" font-family="sans-serif" font-size="11" fill="#4c6ef5">■ Freezing</text>',
        '<text x="240" y="48" font-family="sans-serif" font-size="11" fill="#38d9a9">■ Resting</text>',
        '<text x="325" y="48" font-family="sans-serif" font-size="11" fill="#ffa94d">■ Moderate</text>',
        '<text x="420" y="48" font-family="sans-serif" font-size="11" fill="#ff6b6b">■ High Activity</text>',
    ]
    for r_idx, row in enumerate(report.instances):
        y0 = 62 + r_idx * row_h
        b_lines.append(
            f'<text x="20" y="{y0 + 16}" font-family="sans-serif" font-size="12" fill="#334155">{html.escape(row.instance[:18])}</text>'
        )
        cur_x = float(bar_left)
        for pct, col in (
            (row.freezing_percent, "#4c6ef5"),
            (row.resting_percent, "#38d9a9"),
            (row.moderate_activity_percent, "#ffa94d"),
            (row.high_activity_percent, "#ff6b6b"),
        ):
            seg_w = max(0.0, min(bar_w, (pct / 100.0) * bar_w))
            if seg_w > 0.2:
                b_lines.append(
                    f'<rect x="{cur_x:.1f}" y="{y0}" width="{seg_w:.1f}" height="22" fill="{col}"/>'
                )
                cur_x += seg_w
    b_lines.append("</svg>")
    budget_svg = output_dir / "kinematics_activity_budget.svg"
    atomic_write_text(budget_svg, "\n".join(b_lines) + "\n", encoding="utf-8")
    plots["activity_budget_plot"] = str(budget_svg)
    return plots


def _render_kinematics_plots(
    report: KinematicsReport, output_dir: Path
) -> Dict[str, str]:
    if not report.instances:
        return {}
    if plt is None:
        return _render_kinematics_svg_plots(report, output_dir)
    plots: Dict[str, str] = {}

    # 1. Trajectory map
    if report.trajectories:
        fig, ax = plt.subplots(figsize=(7, 6))
        for inst_name, pts in sorted(report.trajectories.items()):
            if not pts:
                continue
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            ax.plot(xs, ys, linewidth=1.3, alpha=0.85, label=inst_name)
            ax.scatter([xs[0]], [ys[0]], s=28, marker="o")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("Tracked Trajectories")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        traj_path = output_dir / "kinematics_trajectories.png"
        fig.savefig(traj_path, dpi=160)
        plt.close(fig)
        plots["trajectory_plot"] = str(traj_path)

    # 2. Activity state stacked bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    names = [r.instance for r in report.instances]
    f_vals = np.asarray([r.freezing_percent for r in report.instances])
    r_vals = np.asarray([r.resting_percent for r in report.instances])
    m_vals = np.asarray([r.moderate_activity_percent for r in report.instances])
    h_vals = np.asarray([r.high_activity_percent for r in report.instances])
    ax2.bar(names, f_vals, label="Freezing", color="#4c6ef5")
    ax2.bar(names, r_vals, bottom=f_vals, label="Resting", color="#38d9a9")
    ax2.bar(names, m_vals, bottom=f_vals + r_vals, label="Moderate", color="#ffa94d")
    ax2.bar(
        names,
        h_vals,
        bottom=f_vals + r_vals + m_vals,
        label="High Activity",
        color="#ff6b6b",
    )
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("% of Observed Time")
    ax2.set_title("Activity & Freezing Budget by Instance")
    ax2.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    budget_path = output_dir / "kinematics_activity_budget.png"
    fig2.savefig(budget_path, dpi=160)
    plt.close(fig2)
    plots["activity_budget_plot"] = str(budget_path)
    return plots


def write_kinematics_report_bundle(
    report: KinematicsReport,
    output_dir: Path | str,
    *,
    include_plots: bool = True,
    export_events_csv: bool = True,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Write JSON, summary CSV, bouts CSV, pairwise CSV, events CSV, and plots."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = report.to_dict()
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    json_path = out_dir / "kinematics_report.json"
    atomic_write_json(json_path, payload, indent=2, ensure_ascii=False)

    summary_csv = out_dir / "kinematics_summary.csv"
    pd.DataFrame([r.to_dict() for r in report.instances]).to_csv(
        summary_csv, index=False
    )

    bouts_csv = out_dir / "kinematics_bouts.csv"
    pd.DataFrame([b.to_dict() for b in report.bouts]).to_csv(bouts_csv, index=False)

    pairwise_csv = out_dir / "kinematics_pairwise.csv"
    pd.DataFrame([p.to_dict() for p in report.pairwise]).to_csv(
        pairwise_csv, index=False
    )

    artifacts: Dict[str, str] = {
        "output_dir": str(out_dir),
        "report_json": str(json_path),
        "summary_csv": str(summary_csv),
        "bouts_csv": str(bouts_csv),
        "pairwise_csv": str(pairwise_csv),
    }
    if export_events_csv:
        events_csv = out_dir / "kinematics_events.csv"
        write_kinematics_events_csv(report, events_csv)
        artifacts["events_csv"] = str(events_csv)

    if include_plots:
        artifacts.update(_render_kinematics_plots(report, out_dir))
    return artifacts
