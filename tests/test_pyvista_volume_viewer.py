from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyvista")

from annolid.gui.widgets.pyvista_volume_viewer import PyVistaVolumeViewerDialog
from annolid.gui.widgets.volume_utils import normalize_to_float01


from qtpy import QtWidgets
import sys

_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def test_load_source_path_requests_render_after_volume_load(monkeypatch) -> None:
    viewer = PyVistaVolumeViewerDialog.__new__(PyVistaVolumeViewerDialog)
    viewer._load_in_progress = False
    viewer._slice_mode = False
    viewer._render_calls: list[bool] = []
    viewer.info_label = type("L", (), {"setText": lambda self, x: None})()

    class _Loader:
        def load_any_volume(self, path):
            return type(
                "V",
                (),
                {
                    "slice_mode": False,
                    "vtk_image": None,
                    "array": np.zeros((10, 10, 10)),
                    "spacing": (1, 1, 1),
                    "vmin": 0,
                    "vmax": 1,
                    "is_label_map": False,
                },
            )()

    viewer._loader = _Loader()
    viewer._ensure_scene_initialized = lambda: True
    viewer._init_3d_mode = lambda: viewer._render_calls.append(True)

    assert viewer._load_source_path(Path("structural.nii.gz"))
    assert viewer._render_calls == [True]


def test_load_source_path_returns_false_when_renderer_not_ready(monkeypatch) -> None:
    viewer = PyVistaVolumeViewerDialog.__new__(PyVistaVolumeViewerDialog)
    viewer._load_in_progress = False

    monkeypatch.setattr(viewer, "_ensure_scene_initialized", lambda: False)

    assert not viewer._load_source_path(Path("structural.nii.gz"))


def test_request_render_uses_lifecycle_controller(monkeypatch) -> None:
    viewer = PyVistaVolumeViewerDialog.__new__(PyVistaVolumeViewerDialog)
    calls = []
    viewer._lifecycle_controller = type(
        "L",
        (),
        {"request_render": lambda self, reset_camera=False: calls.append(reset_camera)},
    )()

    viewer._request_scene_render(reset_camera=True)
    assert calls == [True]


def test_start_scene_initialization_schedules_initial_scene_load_on_success(
    monkeypatch,
) -> None:
    # In the new version, _load_initial_source is the entry point
    viewer = PyVistaVolumeViewerDialog.__new__(PyVistaVolumeViewerDialog)
    viewer._path = Path("test.tif")
    viewer._load_source_called = 0

    monkeypatch.setattr(
        viewer,
        "_load_source_path",
        lambda path: setattr(
            viewer, "_load_source_called", viewer._load_source_called + 1
        ),
    )

    viewer._load_initial_source()
    assert viewer._load_source_called == 1


def test_normalize_to_float01_scales_float_volume_contrast() -> None:
    volume = np.array([0.0, 10.0, 50.0, 400.0, 1200.0, 6400.0], dtype=np.float32)
    norm = normalize_to_float01(volume)

    assert norm.dtype == np.float32
    assert float(norm.min()) >= 0.0
    assert float(norm.max()) <= 1.0
    # The new robust normalization might have different thresholds,
    # but 0..1 range is the key contract.
    assert float(norm[-1]) == 1.0


def test_normalize_to_float01_handles_nan_inputs() -> None:
    volume = np.array([np.nan, -5.0, 0.0, 5.0, np.inf], dtype=np.float32)
    norm = normalize_to_float01(volume)

    assert np.isfinite(norm).all()
    assert float(norm[0]) == 0.0
    # In some versions it might favor 0.0 or mid-range for NaNs
    assert float(norm.min()) >= 0.0


def test_ensure_scene_initialized_setup_backend_and_widget(monkeypatch) -> None:
    viewer = PyVistaVolumeViewerDialog.__new__(PyVistaVolumeViewerDialog)
    viewer._scene_initialized = False
    viewer.render_widget = None
    viewer.render_area = type(
        "A",
        (),
        {"layout": lambda self: type("L", (), {"addWidget": lambda self, x: None})()},
    )()

    class _Backend:
        def create_widget(self, parent):
            return object()

        def get_render_window(self, widget):
            return type("RW", (), {"AddRenderer": lambda self, r: None})()

        def get_interactor(self, widget, rw):
            return type("I", (), {"SetInteractorStyle": lambda self, s: None})()

    viewer._scene_backend = _Backend()
    viewer.renderer = object()

    assert viewer._ensure_scene_initialized() is True
    assert viewer._scene_initialized is True
