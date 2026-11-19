"""Planar spatial calibration (uniform scale and perspective homography).

Supports converting pixel coordinates ``(x_px, y_px)`` and pixel areas ``area_px``
into physical arena coordinates ``(X_cm, Y_cm)`` and physical areas ``area_cm2``
under either:

1. Uniform orthographic scale (``px_per_cm``), or
2. Full 3x3 planar perspective homography estimated via Hartley-normalized
   Direct Linear Transform (DLT) from >= 4 arena reference points, a physical
   rectangle specification, a LabelMe/Annolid zone JSON file with physical
   dimensions in ``flags``, or an explicit 3x3 homography matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CalibrationError",
    "PlanarCalibration",
    "resolve_planar_calibration",
]


class CalibrationError(ValueError):
    """Raised when spatial calibration parameters or point correspondences are invalid."""


def _hartley_normalize_2d(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(normalized_pts, T_3x3)`` with zero centroid and mean radius sqrt(2)."""
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    dists = np.hypot(shifted[:, 0], shifted[:, 1])
    mean_dist = float(np.mean(dists))
    if not math.isfinite(mean_dist) or mean_dist <= 1e-12:
        raise CalibrationError(
            "Calibration points are degenerate (zero spatial spread)."
        )
    scale = math.sqrt(2.0) / mean_dist
    t_mat = np.array(
        [
            [scale, 0.0, -scale * float(centroid[0])],
            [0.0, scale, -scale * float(centroid[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    norm_pts = (t_mat @ hom.T).T[:, :2]
    return norm_pts, t_mat


def _check_non_collinear(pts: np.ndarray, label: str) -> None:
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        raise CalibrationError(
            f"{label} must be an (N, 2) array with at least 4 points."
        )
    if not np.all(np.isfinite(pts)):
        raise CalibrationError(f"{label} contains non-finite coordinates.")
    centered = pts - np.mean(pts, axis=0)
    svals = np.linalg.svd(centered, compute_uv=False)
    if len(svals) < 2 or float(svals[1]) <= 1e-9 * max(1.0, float(svals[0])):
        raise CalibrationError(
            f"{label} points are collinear or degenerate; at least 4 non-collinear points are required."
        )


def _estimate_dlt_homography(src_px: np.ndarray, dst_cm: np.ndarray) -> np.ndarray:
    """Estimate 3x3 planar homography ``H`` mapping ``src_px -> dst_cm`` via normalized DLT."""
    _check_non_collinear(src_px, "src_points_px")
    _check_non_collinear(dst_cm, "dst_points_cm")
    if len(src_px) != len(dst_cm):
        raise CalibrationError(
            f"Point count mismatch: {len(src_px)} image points vs {len(dst_cm)} world points."
        )

    src_norm, t_src = _hartley_normalize_2d(src_px)
    dst_norm, t_dst = _hartley_normalize_2d(dst_cm)

    n_pts = len(src_px)
    a_mat = np.zeros((2 * n_pts, 9), dtype=np.float64)
    for idx in range(n_pts):
        x, y = float(src_norm[idx, 0]), float(src_norm[idx, 1])
        u, v = float(dst_norm[idx, 0]), float(dst_norm[idx, 1])
        a_mat[2 * idx] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        a_mat[2 * idx + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    _, _, vt = np.linalg.svd(a_mat)
    h_tilde = vt[-1].reshape(3, 3)
    h_mat = np.linalg.inv(t_dst) @ h_tilde @ t_src

    # Orient sign so homogeneous denominator w > 0 at calibration points
    w_vals = h_mat[2, 0] * src_px[:, 0] + h_mat[2, 1] * src_px[:, 1] + h_mat[2, 2]
    if np.all(w_vals < -1e-9):
        h_mat = -h_mat
        w_vals = -w_vals
    if np.any(w_vals <= 1e-9):
        raise CalibrationError(
            "Degenerate perspective homography: calibration polygon is self-intersecting "
            "or crosses the perspective horizon (check corner ordering)."
        )

    if abs(float(h_mat[2, 2])) > 1e-12:
        h_mat = h_mat / float(h_mat[2, 2])
    else:
        norm_h = float(np.linalg.norm(h_mat))
        if norm_h <= 1e-15:
            raise CalibrationError("Estimated homography matrix is zero.")
        h_mat = h_mat / norm_h

    det_h = float(np.linalg.det(h_mat))
    if not math.isfinite(det_h) or abs(det_h) <= 1e-18:
        raise CalibrationError("Estimated homography matrix is singular.")
    return h_mat


def _expand_shape_corners_px(shape: Mapping[str, Any]) -> Optional[np.ndarray]:
    shape_type = str(shape.get("shape_type") or "polygon").strip().lower()
    raw_pts = shape.get("points")
    if not isinstance(raw_pts, Sequence):
        return None
    pts: List[Tuple[float, float]] = []
    for pt in raw_pts:
        if isinstance(pt, Sequence) and len(pt) >= 2:
            try:
                pts.append((float(pt[0]), float(pt[1])))
            except (TypeError, ValueError):
                return None
    if shape_type == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        return np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float64,
        )
    if len(pts) >= 4:
        # Drop duplicate closing vertex if polygon repeats first vertex at end
        if (
            len(pts) == 5
            and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 1e-6
        ):
            pts = pts[:4]
        return np.asarray(pts, dtype=np.float64)
    return None


@dataclass(frozen=True)
class PlanarCalibration:
    """Immutable planar spatial calibration model (uniform or perspective homography)."""

    mode: str
    homography: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]
    px_per_cm_nominal: float
    unit: str = "cm"
    reprojection_rmse_cm: Optional[float] = None
    source: Optional[str] = None

    @classmethod
    def from_px_per_cm(
        cls,
        px_per_cm: float,
        *,
        unit: str = "cm",
        source: Optional[str] = None,
    ) -> "PlanarCalibration":
        val = float(px_per_cm)
        if not math.isfinite(val) or val <= 0.0:
            raise CalibrationError(
                f"px_per_cm must be a positive finite number, got {px_per_cm!r}."
            )
        inv = 1.0 / val
        return cls(
            mode="uniform",
            homography=(
                (inv, 0.0, 0.0),
                (0.0, inv, 0.0),
                (0.0, 0.0, 1.0),
            ),
            px_per_cm_nominal=val,
            unit=str(unit or "cm"),
            reprojection_rmse_cm=0.0,
            source=source,
        )

    @classmethod
    def from_homography(
        cls,
        matrix: Sequence[Sequence[float]],
        *,
        reference_point_px: Tuple[float, float] = (0.0, 0.0),
        unit: str = "cm",
        reprojection_rmse_cm: Optional[float] = None,
        source: Optional[str] = None,
    ) -> "PlanarCalibration":
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
            raise CalibrationError("Homography matrix must be a finite 3x3 array.")
        det_h = float(np.linalg.det(arr))
        if not math.isfinite(det_h) or abs(det_h) <= 1e-18:
            raise CalibrationError("Homography matrix must be non-singular.")
        rx, ry = float(reference_point_px[0]), float(reference_point_px[1])
        w_ref = float(arr[2, 0] * rx + arr[2, 1] * ry + arr[2, 2])
        if abs(w_ref) <= 1e-12:
            w_ref = float(arr[2, 2]) if abs(float(arr[2, 2])) > 1e-12 else 1.0
        if w_ref < 0.0:
            arr = -arr
            det_h = float(np.linalg.det(arr))
            w_ref = -w_ref
        jac_det = abs(det_h) / (abs(w_ref) ** 3)
        px_per_cm_eff = (1.0 / math.sqrt(jac_det)) if jac_det > 1e-24 else 1.0
        is_uniform = (
            abs(arr[0, 1]) <= 1e-12
            and abs(arr[1, 0]) <= 1e-12
            and abs(arr[2, 0]) <= 1e-12
            and abs(arr[2, 1]) <= 1e-12
            and abs(arr[0, 2]) <= 1e-12
            and abs(arr[1, 2]) <= 1e-12
            and arr[2, 2] > 1e-12
            and arr[0, 0] > 0.0
            and arr[1, 1] > 0.0
            and abs(arr[0, 0] - arr[1, 1]) <= 1e-12 * max(1.0, abs(arr[0, 0]))
        )
        tup = (
            (float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2])),
            (float(arr[1, 0]), float(arr[1, 1]), float(arr[1, 2])),
            (float(arr[2, 0]), float(arr[2, 1]), float(arr[2, 2])),
        )
        return cls(
            mode="uniform" if is_uniform else "homography",
            homography=tup,
            px_per_cm_nominal=px_per_cm_eff,
            unit=str(unit or "cm"),
            reprojection_rmse_cm=reprojection_rmse_cm,
            source=source,
        )

    @classmethod
    def from_correspondences(
        cls,
        src_points_px: Sequence[Sequence[float]],
        dst_points_cm: Sequence[Sequence[float]],
        *,
        unit: str = "cm",
        source: Optional[str] = None,
    ) -> "PlanarCalibration":
        src = np.asarray(src_points_px, dtype=np.float64)
        dst = np.asarray(dst_points_cm, dtype=np.float64)
        h_mat = _estimate_dlt_homography(src, dst)
        ref_px = (float(np.mean(src[:, 0])), float(np.mean(src[:, 1])))
        prelim = cls.from_homography(
            h_mat,
            reference_point_px=ref_px,
            unit=unit,
            source=source,
        )
        proj_x, proj_y = prelim.project_points(src[:, 0], src[:, 1])
        err = np.hypot(proj_x - dst[:, 0], proj_y - dst[:, 1])
        rmse = float(np.sqrt(np.mean(err * err)))
        return cls(
            mode="homography",
            homography=prelim.homography,
            px_per_cm_nominal=prelim.px_per_cm_nominal,
            unit=prelim.unit,
            reprojection_rmse_cm=rmse,
            source=source,
        )

    @classmethod
    def from_rectangle_corners(
        cls,
        src_points_px: Sequence[Sequence[float]],
        width_cm: float,
        height_cm: float,
        *,
        origin_cm: Tuple[float, float] = (0.0, 0.0),
        unit: str = "cm",
        source: Optional[str] = None,
    ) -> "PlanarCalibration":
        w_cm = float(width_cm)
        h_cm = float(height_cm)
        if (
            not math.isfinite(w_cm)
            or w_cm <= 0.0
            or not math.isfinite(h_cm)
            or h_cm <= 0.0
        ):
            raise CalibrationError(
                f"Rectangle physical dimensions must be positive finite numbers, got width_cm={width_cm!r}, height_cm={height_cm!r}."
            )
        src = np.asarray(src_points_px, dtype=np.float64)
        if src.shape != (4, 2):
            raise CalibrationError(
                "Rectangle corner calibration requires exactly 4 corner points in [TL, TR, BR, BL] order."
            )
        ox, oy = float(origin_cm[0]), float(origin_cm[1])
        dst = np.array(
            [
                [ox, oy],
                [ox + w_cm, oy],
                [ox + w_cm, oy + h_cm],
                [ox, oy + h_cm],
            ],
            dtype=np.float64,
        )
        return cls.from_correspondences(src, dst, unit=unit, source=source)

    @classmethod
    def load(
        cls,
        spec: Path | str | Mapping[str, Any],
    ) -> "PlanarCalibration":
        """Load calibration from a dict, calibration JSON file, or LabelMe/Zone JSON file."""
        source_label: Optional[str] = None
        if isinstance(spec, (str, Path)):
            p = Path(spec).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Calibration file not found: {p}")
            source_label = str(p)
            data = json.loads(p.read_text(encoding="utf-8"))
        elif isinstance(spec, Mapping):
            data = dict(spec)
            source_label = str(data.get("source") or "") or None
        else:
            raise CalibrationError(
                f"Unsupported calibration specification type: {type(spec).__name__}"
            )

        if not isinstance(data, Mapping):
            raise CalibrationError("Calibration JSON payload must be an object.")

        unit = str(data.get("unit") or "cm")

        if data.get("homography") is not None:
            ref_pt = data.get("reference_point_px") or (0.0, 0.0)
            ref_tuple = (
                (float(ref_pt[0]), float(ref_pt[1]))
                if isinstance(ref_pt, Sequence) and len(ref_pt) >= 2
                else (0.0, 0.0)
            )
            cal = cls.from_homography(
                data["homography"],
                reference_point_px=ref_tuple,
                unit=unit,
                reprojection_rmse_cm=(
                    float(data["reprojection_rmse_cm"])
                    if data.get("reprojection_rmse_cm") is not None
                    else None
                ),
                source=source_label,
            )
            if (
                str(data.get("mode") or "").lower() == "homography"
                and cal.mode != "homography"
            ):
                return cls(
                    mode="homography",
                    homography=cal.homography,
                    px_per_cm_nominal=cal.px_per_cm_nominal,
                    unit=cal.unit,
                    reprojection_rmse_cm=cal.reprojection_rmse_cm,
                    source=source_label,
                )
            return cal

        src_pts = data.get("src_points_px") or data.get("image_points_px")
        dst_pts = data.get("dst_points_cm") or data.get("world_points_cm")
        if src_pts is not None and dst_pts is not None:
            return cls.from_correspondences(
                src_pts, dst_pts, unit=unit, source=source_label
            )

        width_cm = data.get("width_cm") or data.get("arena_width_cm")
        height_cm = data.get("height_cm") or data.get("arena_height_cm")
        if src_pts is not None and width_cm is not None and height_cm is not None:
            return cls.from_rectangle_corners(
                src_pts,
                float(width_cm),
                float(height_cm),
                unit=unit,
                source=source_label,
            )

        if data.get("px_per_cm") is not None:
            return cls.from_px_per_cm(
                float(data["px_per_cm"]), unit=unit, source=source_label
            )

        # Support LabelMe / Annolid Zone JSON files containing calibrated shapes
        shapes = data.get("shapes")
        if isinstance(shapes, Sequence):
            for shp in shapes:
                if not isinstance(shp, Mapping):
                    continue
                flags = (
                    shp.get("flags") if isinstance(shp.get("flags"), Mapping) else {}
                )
                shp_dst = flags.get("world_points_cm") or flags.get("dst_points_cm")
                corners = _expand_shape_corners_px(shp)
                if corners is not None and shp_dst is not None:
                    return cls.from_correspondences(
                        corners, shp_dst, unit=unit, source=source_label
                    )
                w_flag = (
                    flags.get("width_cm")
                    or flags.get("arena_width_cm")
                    or shp.get("width_cm")
                )
                h_flag = (
                    flags.get("height_cm")
                    or flags.get("arena_height_cm")
                    or shp.get("height_cm")
                )
                if (
                    corners is not None
                    and len(corners) == 4
                    and w_flag is not None
                    and h_flag is not None
                ):
                    return cls.from_rectangle_corners(
                        corners,
                        float(w_flag),
                        float(h_flag),
                        unit=unit,
                        source=source_label,
                    )
                ppc_flag = flags.get("px_per_cm") or shp.get("px_per_cm")
                if ppc_flag is not None:
                    return cls.from_px_per_cm(
                        float(ppc_flag), unit=unit, source=source_label
                    )

        raise CalibrationError(
            "Calibration JSON must define 'homography', ('src_points_px' + 'dst_points_cm'), "
            "('src_points_px' + 'width_cm' + 'height_cm'), 'px_per_cm', or a LabelMe shape "
            "with physical dimensions in 'flags'."
        )

    def project_points(
        self, x_px: Sequence[float] | np.ndarray, y_px: Sequence[float] | np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project pixel coordinates ``(x_px, y_px)`` into physical world coordinates ``(X_cm, Y_cm)``."""
        xs = np.asarray(x_px, dtype=np.float64)
        ys = np.asarray(y_px, dtype=np.float64)
        if xs.shape != ys.shape:
            raise CalibrationError("x_px and y_px must have matching shapes.")
        if xs.size == 0:
            return xs.copy(), ys.copy()
        if self.mode == "uniform":
            inv = 1.0 / self.px_per_cm_nominal
            return xs * inv, ys * inv

        h = np.asarray(self.homography, dtype=np.float64)
        u = h[0, 0] * xs + h[0, 1] * ys + h[0, 2]
        v = h[1, 0] * xs + h[1, 1] * ys + h[1, 2]
        w = h[2, 0] * xs + h[2, 1] * ys + h[2, 2]
        finite_mask = np.isfinite(xs) & np.isfinite(ys)
        if np.any(finite_mask & (np.abs(w) <= 1e-9)):
            raise CalibrationError(
                "Trajectory point lies on the perspective horizon (homogeneous w == 0)."
            )
        safe_w = np.where(np.abs(w) > 1e-9, w, np.nan)
        return u / safe_w, v / safe_w

    def transform_areas(
        self,
        x_px: Sequence[float] | np.ndarray,
        y_px: Sequence[float] | np.ndarray,
        area_px: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Convert pixel areas ``area_px`` at ``(x_px, y_px)`` to physical ``cm^2`` via local Jacobian determinant."""
        areas = np.asarray(area_px, dtype=np.float64)
        if areas.size == 0:
            return areas.copy()
        if self.mode == "uniform":
            return areas / (self.px_per_cm_nominal * self.px_per_cm_nominal)

        xs = np.asarray(x_px, dtype=np.float64)
        ys = np.asarray(y_px, dtype=np.float64)
        h = np.asarray(self.homography, dtype=np.float64)
        det_h = abs(float(np.linalg.det(h)))
        w = h[2, 0] * xs + h[2, 1] * ys + h[2, 2]
        safe_w3 = np.where(np.abs(w) > 1e-9, np.abs(w) ** 3, np.nan)
        jac_det = det_h / safe_w3
        return areas * jac_det

    def summary_label(self) -> str:
        if self.mode == "homography":
            rmse_txt = (
                f", rmse={self.reprojection_rmse_cm:.3f} {self.unit}"
                if self.reprojection_rmse_cm is not None
                else ""
            )
            return f"perspective homography (~{self.px_per_cm_nominal:.2f} px/{self.unit}{rmse_txt})"
        return f"{self.px_per_cm_nominal:.2f} px/{self.unit}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "unit": self.unit,
            "px_per_cm_nominal": round(float(self.px_per_cm_nominal), 6),
            "homography": [
                [round(float(v), 10) for v in row] for row in self.homography
            ],
            "reprojection_rmse_cm": (
                round(float(self.reprojection_rmse_cm), 6)
                if self.reprojection_rmse_cm is not None
                else None
            ),
            "source": self.source,
        }


def resolve_planar_calibration(
    *,
    calibration: PlanarCalibration | Path | str | Mapping[str, Any] | None = None,
    px_per_cm: Optional[float] = None,
) -> Optional[PlanarCalibration]:
    """Resolve optional homography/JSON calibration or fallback scalar ``px_per_cm``."""
    if isinstance(calibration, PlanarCalibration):
        return calibration
    if calibration is not None and str(calibration).strip() != "":
        return PlanarCalibration.load(calibration)
    if px_per_cm is not None:
        return PlanarCalibration.from_px_per_cm(float(px_per_cm))
    return None
