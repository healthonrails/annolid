from __future__ import annotations

from pathlib import Path

import numpy as np

from annolid.gui.widgets.vtk_volume_viewer import VTKVolumeViewerDialog
from annolid.gui.widgets.vtk_volume_utils import normalize_to_float01


def test_load_source_path_requests_render_after_volume_load(monkeypatch) -> None:
    viewer = VTKVolumeViewerDialog.__new__(VTKVolumeViewerDialog)
    viewer._load_in_progress = False
    viewer._slice_mode = False
    viewer._render_calls: list[bool] = []

    class _Readers:
        def is_volume_candidate(self, path):
            return True

        def is_point_cloud_candidate(self, path):
            return False

        def is_mesh_candidate(self, path):
            return False

        def resolve_initial_source(self, path):
            return path

    monkeypatch.setattr(viewer, "_volume_readers", lambda: _Readers())
    monkeypatch.setattr(viewer, "_load_volume", lambda: True)
    monkeypatch.setattr(viewer, "_ensure_vtk_initialized", lambda: True)
    monkeypatch.setattr(viewer, "_set_load_busy", lambda busy: None)
    monkeypatch.setattr(viewer, "_set_volume_controls_enabled", lambda enabled: None)
    monkeypatch.setattr(viewer, "_update_mesh_status_label", lambda: None)
    monkeypatch.setattr(viewer, "_refresh_status_summary", lambda: None)
    monkeypatch.setattr(
        viewer,
        "_request_render",
        lambda reset_camera=False: viewer._render_calls.append(reset_camera),
    )

    assert viewer._load_source_path(Path("structural.nii.gz"), show_errors=False)
    assert viewer._render_calls == [True]


def test_load_source_path_returns_false_when_vtk_not_ready(monkeypatch) -> None:
    viewer = VTKVolumeViewerDialog.__new__(VTKVolumeViewerDialog)
    viewer._load_in_progress = False

    monkeypatch.setattr(viewer, "_ensure_vtk_initialized", lambda: False)

    assert not viewer._load_source_path(Path("structural.nii.gz"), show_errors=False)


def test_request_render_triggers_single_init_when_vtk_not_initialized(
    monkeypatch,
) -> None:
    viewer = VTKVolumeViewerDialog.__new__(VTKVolumeViewerDialog)
    viewer._vtk_initialized = False
    viewer._vtk_init_scheduled = False
    viewer._render_pending = False
    viewer._render_reset_camera_pending = False
    calls: list[tuple[int, object]] = []

    class _Timer:
        @staticmethod
        def singleShot(delay, callback):
            calls.append((delay, callback))

    monkeypatch.setattr("annolid.gui.widgets.vtk_volume_viewer.QtCore.QTimer", _Timer)
    monkeypatch.setattr(viewer, "_start_vtk_initialization", lambda: None)

    viewer._request_render(reset_camera=True)
    viewer._request_render(reset_camera=False)

    assert viewer._vtk_init_scheduled is True
    assert len(calls) == 1
    assert calls[0][0] == 0


def test_request_render_does_not_retry_while_hidden(monkeypatch) -> None:
    viewer = VTKVolumeViewerDialog.__new__(VTKVolumeViewerDialog)
    viewer._vtk_initialized = True
    viewer._vtk_init_scheduled = False
    viewer._render_pending = False
    viewer._render_reset_camera_pending = False
    viewer.renderer = type("R", (), {"ResetCamera": lambda self: None})()

    class _RW:
        def __init__(self):
            self.render_calls = 0

        def Render(self):
            self.render_calls += 1

    rw = _RW()
    monkeypatch.setattr(viewer, "_get_render_window", lambda: rw)
    monkeypatch.setattr(viewer, "isVisible", lambda: False)

    callbacks: list[tuple[int, object]] = []

    class _Timer:
        @staticmethod
        def singleShot(delay, callback):
            callbacks.append((delay, callback))

    monkeypatch.setattr("annolid.gui.widgets.vtk_volume_viewer.QtCore.QTimer", _Timer)

    viewer._request_render(reset_camera=True)
    assert len(callbacks) == 1
    # Run queued render callback once; hidden windows should not re-queue.
    callbacks[0][1]()
    assert len(callbacks) == 1
    assert viewer._render_pending is False
    assert rw.render_calls == 0


def test_start_vtk_initialization_retries_when_init_fails(monkeypatch) -> None:
    viewer = VTKVolumeViewerDialog.__new__(VTKVolumeViewerDialog)
    viewer._vtk_init_scheduled = True
    viewer._vtk_initialized = False
    viewer._vtk_init_failures = 0
    viewer._vtk_init_last_error = "init-failure"
    viewer._initial_source_load_pending = True
    viewer._initial_source_load_scheduled = False
    refresh_calls: list[bool] = []
    busy_calls: list[bool] = []
    timer_calls: list[int] = []

    monkeypatch.setattr(viewer, "_ensure_vtk_initialized", lambda: False)
    monkeypatch.setattr(
        viewer, "_refresh_status_summary", lambda: refresh_calls.append(True)
    )
    monkeypatch.setattr(viewer, "_set_load_busy", lambda busy: busy_calls.append(busy))
    monkeypatch.setattr(viewer, "isVisible", lambda: True)

    class _Timer:
        @staticmethod
        def singleShot(delay, callback):
            timer_calls.append(delay)

    monkeypatch.setattr("annolid.gui.widgets.vtk_volume_viewer.QtCore.QTimer", _Timer)

    viewer._start_vtk_initialization()

    assert viewer._vtk_init_failures == 1
    assert viewer._vtk_init_scheduled is True
    assert timer_calls == [150]
    assert busy_calls == [False]
    assert refresh_calls == [True]


def test_normalize_to_float01_scales_float_volume_contrast() -> None:
    volume = np.array([0.0, 10.0, 50.0, 400.0, 1200.0, 6400.0], dtype=np.float32)
    norm = normalize_to_float01(volume)

    assert norm.dtype == np.float32
    assert float(norm.min()) >= 0.0
    assert float(norm.max()) <= 1.0
    # Mid-range intensity should remain well below saturation.
    assert float(norm[2]) < 0.2
    # High tail stays bright but clipped safely.
    assert float(norm[-1]) == 1.0


def test_normalize_to_float01_handles_nan_inputs() -> None:
    volume = np.array([np.nan, -5.0, 0.0, 5.0, np.inf], dtype=np.float32)
    norm = normalize_to_float01(volume)

    assert np.isfinite(norm).all()
    assert float(norm[0]) == 0.0
    assert float(norm[-1]) == 0.0


def test_make_slice_volume_data_builds_slice_mode_metadata() -> None:
    viewer = VTKVolumeViewerDialog.__new__(VTKVolumeViewerDialog)

    class _Loader:
        def shape(self):
            return (12, 34, 56)

        def dtype(self):
            return np.dtype(np.uint16)

    data = viewer._make_slice_volume_data(
        _Loader(),
        spacing=(0.4, 0.5, 0.6),
        value_range=(10.0, 2000.0),
        is_grayscale=True,
        is_label_map=False,
    )

    assert data.slice_mode is True
    assert data.slice_loader is not None
    assert data.is_out_of_core is True
    assert data.array is None
    assert data.spacing == (0.4, 0.5, 0.6)
    assert data.vmin == 10.0
    assert data.vmax == 2000.0
    assert data.volume_shape == (12, 34, 56)


def test_volume_readers_make_simple_volume_data_uses_finite_range() -> None:
    viewer = VTKVolumeViewerDialog.__new__(VTKVolumeViewerDialog)
    readers = viewer._volume_readers()
    arr = np.array([np.nan, -10.0, 100.0, np.inf], dtype=np.float32)
    out = readers.make_simple_volume_data(arr, (1.0, 1.0, 1.0))

    assert out.vmin == -10.0
    assert out.vmax == 100.0


def test_initial_window_range_uses_robust_percentiles() -> None:
    viewer = VTKVolumeViewerDialog.__new__(VTKVolumeViewerDialog)
    viewer._vmin = 0.0
    viewer._vmax = 6427.0
    volume = np.array([0.0] * 500 + [49.0] * 300 + [1014.0] * 150 + [6427.0] * 5)

    low, high = viewer._initial_window_range(volume)

    assert 0.0 <= low <= 49.0
    assert 900.0 <= high <= 1500.0
    assert high < viewer._vmax
