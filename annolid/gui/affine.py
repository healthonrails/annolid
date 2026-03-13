from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:
    from qtpy import QtCore
except Exception:  # pragma: no cover - affine math is still testable without Qt
    QtCore = None


def solve_affine_from_landmarks(
    source_points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Solve a 2D affine transform from paired landmark coordinates."""
    src = np.asarray(source_points, dtype=float)
    dst = np.asarray(target_points, dtype=float)
    if src.shape != dst.shape:
        raise ValueError("source_points and target_points must have the same shape")
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("Expected Nx2 landmark coordinates")
    if src.shape[0] < 3:
        raise ValueError("At least 3 landmark pairs are required for affine solve")

    design = np.concatenate([src, np.ones((src.shape[0], 1), dtype=float)], axis=1)
    solution, _, rank, _ = np.linalg.lstsq(design, dst, rcond=None)
    if rank < 3:
        raise ValueError("Landmark configuration is degenerate for affine solving")

    matrix = np.eye(3, dtype=float)
    matrix[:2, :] = solution.T
    return matrix


def apply_affine_to_points(
    points: Iterable[Sequence[float]], matrix: np.ndarray
) -> list[tuple[float, float]]:
    """Apply a 3x3 affine matrix to 2D points."""
    affine = np.asarray(matrix, dtype=float)
    if affine.shape != (3, 3):
        raise ValueError("Expected a 3x3 affine matrix")
    pts = np.asarray(list(points), dtype=float)
    if pts.size == 0:
        return []
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Expected Nx2 points")
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=float)], axis=1)
    transformed = hom @ affine.T
    return [(float(x), float(y)) for x, y in transformed[:, :2]]


def apply_affine_to_shape_points(shape, matrix: np.ndarray) -> None:
    """Mutate a Shape-like object's points in place using a 3x3 affine matrix."""
    transformed = apply_affine_to_points(
        [
            (float(point.x()), float(point.y()))
            for point in getattr(shape, "points", [])
        ],
        matrix,
    )
    if QtCore is None:
        shape.points = transformed
        return
    shape.points = [QtCore.QPointF(x, y) for x, y in transformed]


def compose_affine(*matrices: np.ndarray) -> np.ndarray:
    """Compose affine transforms in left-to-right application order."""
    result = np.eye(3, dtype=float)
    for matrix in matrices:
        affine = np.asarray(matrix, dtype=float)
        if affine.shape != (3, 3):
            raise ValueError("Expected a 3x3 affine matrix")
        result = affine @ result
    return result
