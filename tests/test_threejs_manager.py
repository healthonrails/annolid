from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy import QtCore, QtWidgets
import numpy as np

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None
try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

from annolid.gui.widgets.threejs_manager import ThreeJsManager


requires_zarr = pytest.mark.skipif(zarr is None, reason="zarr is not installed")
requires_tifffile = pytest.mark.skipif(
    tifffile is None, reason="tifffile is not installed"
)


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


class _DummyStatusBar:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def showMessage(self, text: str, timeout: int = 0) -> None:
        self.messages.append((text, timeout))


class _DummyWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._status = _DummyStatusBar()
        self.actions = SimpleNamespace(close=None)
        self.picked_regions: list[str] = []

    def statusBar(self):
        return self._status

    def set_unrelated_docks_visible(self, _visible: bool) -> None:
        return None

    def _set_active_view(self, _view: str) -> None:
        return None

    def _onBrain3DMeshRegionPicked(self, region_id: str) -> None:
        self.picked_regions.append(str(region_id or ""))


class _DummyViewer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str | None]] = []

    def load_simulation_payload(
        self, path: str | Path, title: str | None = None
    ) -> None:
        self.calls.append((Path(path), title))


class _DummyModelViewer:
    def __init__(self) -> None:
        self.model_calls: list[Path] = []
        self.sim_calls: list[tuple[Path, str | None]] = []

    def load_model(
        self,
        path: str | Path,
        *,
        pick_mode: str = "",
        object_region_map: dict[str, str] | None = None,
    ) -> None:
        _ = pick_mode
        _ = object_region_map
        self.model_calls.append(Path(path))

    def load_simulation_payload(
        self, path: str | Path, title: str | None = None
    ) -> None:
        self.sim_calls.append((Path(path), title))


class _SignalViewer(QtWidgets.QWidget):
    status_changed = QtCore.Signal(str)
    flybody_command_requested = QtCore.Signal(str, str)
    region_picked = QtCore.Signal(str)


def test_show_simulation_in_viewer_accepts_prebuilt_payload(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))
    viewer = _DummyViewer()
    monkeypatch.setattr(manager, "ensure_threejs_viewer", lambda: viewer)

    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps({"kind": "annolid-simulation-v1", "frames": []}),
        encoding="utf-8",
    )

    assert manager.show_simulation_in_viewer(payload_path) is True
    assert viewer.calls == [(payload_path, "payload")]


def test_show_simulation_in_viewer_exports_non_payload_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))
    viewer = _DummyViewer()
    monkeypatch.setattr(manager, "ensure_threejs_viewer", lambda: viewer)

    source_path = tmp_path / "simulation.ndjson"
    source_path.write_text("{}", encoding="utf-8")
    payload_path = tmp_path / "exported.json"
    payload_path.write_text(
        json.dumps({"kind": "annolid-simulation-v1", "frames": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.threejs_manager.export_simulation_view_payload",
        lambda path: payload_path,
    )

    assert manager.show_simulation_in_viewer(source_path) is True
    assert viewer.calls == [(payload_path, "simulation")]


def test_threejs_manager_region_pick_forwards_to_window_handler() -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    manager._handle_region_picked("region_a||")
    assert window.picked_regions == ["region_a||"]


def test_threejs_manager_connects_viewer_region_picked_signal(monkeypatch) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))
    monkeypatch.setattr(
        "annolid.gui.widgets.threejs_manager.ThreeJsViewerWidget",
        _SignalViewer,
    )

    viewer = manager.ensure_threejs_viewer()
    viewer.region_picked.emit("region_b||")
    assert window.picked_regions == ["region_b||"]


@requires_zarr
def test_show_model_in_viewer_routes_zarr_through_simulation_payload(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))
    viewer = _DummyModelViewer()
    monkeypatch.setattr(manager, "ensure_threejs_viewer", lambda: viewer)

    zarr_path = tmp_path / "atlas_interleaved_30um_image.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    payload_path = tmp_path / "atlas_payload.json"
    payload_path.write_text(
        json.dumps({"kind": "annolid-simulation-v1", "frames": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        manager,
        "_resolve_zarr_simulation_payload",
        lambda path: payload_path,
    )

    assert manager.show_model_in_viewer(zarr_path) is True
    assert viewer.sim_calls == [(payload_path, "atlas_interleaved_30um_image")]
    assert viewer.model_calls == []


@requires_tifffile
def test_show_model_in_viewer_routes_tiff_through_simulation_payload(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))
    viewer = _DummyModelViewer()
    monkeypatch.setattr(manager, "ensure_threejs_viewer", lambda: viewer)

    tif_path = tmp_path / "atlas_interleaved_30um_image.tif"
    tifffile.imwrite(str(tif_path), np.zeros((8, 8, 8), dtype=np.uint16))
    payload_path = tmp_path / "atlas_payload.json"
    payload_path.write_text(
        json.dumps({"kind": "annolid-simulation-v1", "frames": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        manager,
        "_resolve_tiff_simulation_payload",
        lambda path: payload_path,
    )

    assert manager.show_model_in_viewer(tif_path) is True
    assert viewer.sim_calls == [(payload_path, "atlas_interleaved_30um_image")]
    assert viewer.model_calls == []


@requires_zarr
def test_build_zarr_simulation_payload_extracts_sparse_volume(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    zarr_path = tmp_path / "atlas_interleaved_30um_image.zarr"
    arr = zarr.open(str(zarr_path), mode="w", shape=(8, 8, 8), dtype="f4")
    data = np.zeros((8, 8, 8), dtype=np.float32)
    data[2:6, 2:6, 2:6] = 10.0
    data[4, 4, 4] = 20.0
    arr[:] = data

    payload = manager._build_zarr_simulation_payload(zarr_path)

    assert payload["kind"] == "annolid-simulation-v1"
    assert payload["adapter"] == "zarr-volume"
    assert payload["metadata"]["shape"] == [8, 8, 8]
    assert payload["metadata"]["render_mode"] == "gaussian_splatting"
    assert payload["metadata"]["point_count"] > 0
    assert (
        payload["metadata"]["confidence_range"][1]
        >= payload["metadata"]["confidence_range"][0]
    )
    assert payload["metadata"]["volume_render_defaults"]["preset"] == "section_stack"
    assert payload["metadata"]["volume_render_defaults"]["background_theme"] == "light"
    assert payload["metadata"]["interleaved_detected"] is True
    assert payload["metadata"]["section_axis"] == "z"
    assert payload["metadata"]["section_step_world"] >= 1.0
    assert payload["metadata"]["volume_grid_shape"] == [8, 8, 8]
    assert payload["metadata"]["volume_grid_base64"]
    assert len(payload["metadata"]["volume_histogram"]["counts"]) == 32
    assert (
        payload["metadata"]["volume_bounds"]["x"][1]
        >= payload["metadata"]["volume_bounds"]["x"][0]
    )
    assert (
        payload["metadata"]["volume_render_defaults"]["tf_low"]
        < payload["metadata"]["volume_render_defaults"]["tf_high"]
    )
    assert len(payload["frames"]) == 1
    assert len(payload["frames"][0]["points"]) > 0


@requires_tifffile
def test_build_tiff_simulation_payload_extracts_sparse_volume(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    tif_path = tmp_path / "mouse_nissl_sections.tif"
    data = np.zeros((16, 16, 16), dtype=np.float32)
    data[4:12, 4:12, 4:12] = 10.0
    data[7:9, 7:9, 7:9] = 20.0
    tifffile.imwrite(str(tif_path), data)

    payload = manager._build_tiff_simulation_payload(tif_path)

    assert payload["kind"] == "annolid-simulation-v1"
    assert payload["adapter"] == "tiff-volume"
    assert payload["metadata"]["shape"] == [16, 16, 16]
    assert payload["metadata"]["render_mode"] == "gaussian_splatting"
    assert payload["metadata"]["point_count"] > 0
    assert payload["metadata"]["volume_grid_shape"] == [16, 16, 16]
    assert payload["metadata"]["volume_grid_base64"]
    assert payload["metadata"]["section_step_world"] >= 1.0
    assert payload["metadata"]["voxel_spacing_zyx"] == [1.0, 1.0, 1.0]
    assert payload["metadata"]["voxel_spacing_xyz"] == [1.0, 1.0, 1.0]
    assert payload["metadata"]["volume_render_defaults"]["render_style"] == "raymarch"
    assert payload["metadata"]["volume_render_defaults"]["raymarch_steps"] >= 64
    assert len(payload["frames"]) == 1
    assert len(payload["frames"][0]["points"]) > 0


@requires_zarr
def test_build_zarr_simulation_payload_prefers_multiscale_level_for_large_data(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    zarr_path = tmp_path / "atlas_interleaved_30um_image.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [4.0, 2.0, 1.0]}
                    ],
                },
                {
                    "path": "1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [8.0, 4.0, 2.0]}
                    ],
                },
            ],
            "axes": ["z", "y", "x"],
        }
    ]
    root.create_dataset("0", shape=(400, 400, 400), dtype="f4", fill_value=0)
    lvl1 = root.create_dataset("1", shape=(40, 40, 40), dtype="f4", fill_value=0)
    lvl1[10:30, 10:30, 10:30] = 8.0

    payload = manager._build_zarr_simulation_payload(zarr_path)

    assert payload["metadata"]["source_dataset_path"] == "1"
    assert payload["metadata"]["shape"] == [40, 40, 40]
    assert payload["metadata"]["render_mode"] == "gaussian_splatting"
    assert payload["metadata"]["volume_render_defaults"]["blend_mode"] == "normal"
    assert payload["metadata"]["voxel_spacing_zyx"] == [8.0, 4.0, 2.0]
    assert payload["metadata"]["voxel_spacing_xyz"] == [2.0, 4.0, 8.0]
    assert payload["metadata"]["point_count"] > 0


@requires_zarr
def test_build_zarr_simulation_payload_respects_axes_and_channel_selection(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    zarr_path = tmp_path / "atlas_interleaved_30um_image.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "datasets": [{"path": "0"}],
            "axes": ["t", "c", "z", "y", "x"],
        }
    ]
    arr = root.create_dataset("0", shape=(2, 3, 10, 10, 10), dtype="f4", fill_value=0)
    # Channel 1 carries the meaningful signal; channel 0 is flat.
    arr[0, 1, 2:8, 2:8, 2:8] = 10.0
    arr[0, 2, 2:8, 2:8, 2:8] = 2.0

    payload = manager._build_zarr_simulation_payload(zarr_path)

    assert payload["metadata"]["axes"] == ["t", "c", "z", "y", "x"]
    assert payload["metadata"]["channel_index"] == 1
    assert payload["metadata"]["render_mode"] == "gaussian_splatting"
    assert payload["metadata"]["point_count"] > 0


@requires_zarr
def test_build_zarr_simulation_payload_inverts_bright_background_histology(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    zarr_path = tmp_path / "mouse_nissl_sections.zarr"
    arr = zarr.open(str(zarr_path), mode="w", shape=(12, 12, 12), dtype="f4")
    data = np.ones((12, 12, 12), dtype=np.float32)
    data[2:10, 2:10, 2:10] = 0.2
    data[4:8, 4:8, 4:8] = 0.05
    arr[:] = data

    payload = manager._build_zarr_simulation_payload(zarr_path)

    assert payload["metadata"]["intensity_inverted"] is True
    assert payload["metadata"]["signal_polarity"] == "dark_on_light"
    assert payload["metadata"]["volume_render_defaults"]["preset"] == "nissl_sections"
