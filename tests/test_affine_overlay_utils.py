from __future__ import annotations

import numpy as np

from annolid.gui.affine import apply_affine_to_points, solve_affine_from_landmarks


def test_solve_affine_from_landmarks_translation() -> None:
    src = [(0, 0), (10, 0), (0, 10)]
    dst = [(5, 7), (15, 7), (5, 17)]

    matrix = solve_affine_from_landmarks(src, dst)
    transformed = apply_affine_to_points([(2, 3)], matrix)

    assert np.allclose(matrix, [[1, 0, 5], [0, 1, 7], [0, 0, 1]])
    assert np.allclose(transformed, [(7, 10)])


def test_solve_affine_requires_non_degenerate_landmarks() -> None:
    src = [(0, 0), (1, 1), (2, 2)]
    dst = [(0, 0), (2, 2), (4, 4)]

    try:
        solve_affine_from_landmarks(src, dst)
    except ValueError as exc:
        assert "degenerate" in str(exc).lower()
    else:
        raise AssertionError("Expected degenerate affine solve to fail")
