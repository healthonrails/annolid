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
        self,
        path: str | Path,
        title: str | None = None,
        asset_roots: dict[str, str | Path] | None = None,
    ) -> None:
        _ = asset_roots
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
        self,
        path: str | Path,
        title: str | None = None,
        asset_roots: dict[str, str | Path] | None = None,
    ) -> None:
        _ = asset_roots
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


@requires_tifffile
def test_build_tiff_simulation_payload_detects_label_ids_volume(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    tif_path = tmp_path / "allen_annotation_labels.tif"
    data = np.zeros((12, 12, 12), dtype=np.uint16)
    data[2:8, 2:8, 2:8] = 315
    data[5:10, 5:10, 5:10] = 671
    data[1:3, 1:3, 1:3] = 997
    tifffile.imwrite(str(tif_path), data)

    payload = manager._build_tiff_simulation_payload(tif_path)

    assert payload["kind"] == "annolid-simulation-v1"
    assert payload["adapter"] == "tiff-volume"
    assert payload["metadata"]["render_mode"] == "label_ids"
    assert payload["metadata"]["label_volume"] is True
    assert payload["metadata"]["label_id_count"] >= 3
    assert payload["metadata"]["volume_label_id_lut"]
    assert isinstance(payload["metadata"]["volume_label_color_seed"], int)
    assert payload["metadata"]["volume_label_color_seed"] >= 0
    assert payload["metadata"]["volume_label_colors"] == {}
    assert (
        payload["metadata"]["volume_render_defaults"]["label_color_seed"]
        == payload["metadata"]["volume_label_color_seed"]
    )
    assert payload["metadata"]["volume_render_defaults"]["palette"] == "allen_labels"
    assert payload["metadata"]["volume_grid_shape"] == [12, 12, 12]
    assert payload["metadata"]["volume_grid_base64"]
    assert len(payload["frames"]) == 1
    assert len(payload["frames"][0]["points"]) > 0
    first = payload["frames"][0]["points"][0]
    assert "label_id" in first
    assert "label_index" in first


@requires_tifffile
def test_build_tiff_overlay_payload_uses_reference_and_annotation_layers(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    reference_path = tmp_path / "reference.tif"
    annotation_path = tmp_path / "annotation.tif"
    metadata_path = tmp_path / "metadata.json"
    reference = np.zeros((10, 12, 14), dtype=np.uint16)
    reference[2:8, 3:10, 4:12] = 1200
    annotation = np.zeros((10, 12, 14), dtype=np.uint16)
    annotation[2:7, 3:9, 4:10] = 151
    annotation[6:9, 7:11, 8:13] = 188
    tifffile.imwrite(str(reference_path), reference)
    tifffile.imwrite(str(annotation_path), annotation)
    metadata_path.write_text(
        json.dumps(
            {
                "name": "kim_mouse_25um_olfactory_bulb",
                "resolution_um": [25.0, 25.0, 25.0],
                "crop_bbox_zyx": [2, 12, 3, 15, 4, 18],
                "included_ids": [151, 188],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "structures.json").write_text(
        json.dumps(
            [
                {
                    "id": 151,
                    "name": "Area X",
                    "acronym": "Area X",
                    "rgb_triplet": [12, 34, 220],
                    "structure_id_path": [999, 2, 151],
                },
                {
                    "id": 188,
                    "name": "Hippocampus",
                    "acronym": "HP",
                    "rgb_triplet": [176, 108, 108],
                    "structure_id_path": [999, 1, 188],
                },
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "structures.csv").write_text(
        "\n".join(
            [
                "id,name,acronym,structure_id_path,parent_structure_id",
                "151,Area X,Area X,/999/2/151/,2",
                "188,Hippocampus,HP,/999/1/188/,1",
            ]
        ),
        encoding="utf-8",
    )
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "151.obj").write_text(
        "o AreaX\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )

    payload = manager._build_tiff_overlay_simulation_payload(
        reference_path=reference_path,
        annotation_path=annotation_path,
        metadata_path=metadata_path,
    )

    metadata = payload["metadata"]
    assert payload["adapter"] == "tiff-volume"
    assert metadata["volume_overlay"] is True
    assert metadata["layer_role"] == "reference"
    assert metadata["volume_layer_count"] == 2
    assert metadata["atlas_overlay_metadata"]["included_ids"] == [151, 188]
    assert metadata["volume_orientation"] == "zyx"
    assert metadata["atlas_crop_bbox_zyx"] == [2, 12, 3, 15, 4, 18]
    assert metadata["atlas_resolution_zyx_um"] == [25.0, 25.0, 25.0]
    assert metadata["atlas_mesh_coordinate_space"] == "atlas_zyx_um"
    assert metadata["voxel_spacing_zyx"] == [25.0, 25.0, 25.0]
    assert metadata["voxel_spacing_xyz"] == [25.0, 25.0, 25.0]
    assert metadata["volume_bounds"]["x"] == [0.0, 325.0]
    assert metadata["volume_bounds"]["y"] == [-275.0, 0.0]
    assert metadata["volume_bounds"]["z"] == [-225.0, 0.0]
    assert metadata["overlay_validation"]["orientation"] == "zyx"
    assert metadata["volume_grid_shape"] == [10, 12, 14]
    assert metadata["atlas_region_count"] == 2
    assert metadata["atlas_mesh_alias"] == "atlas_meshes"
    assert metadata["atlas_region_layers"][0]["id"] == 151
    assert metadata["atlas_region_layers"][0]["name"] == "Area X"
    assert metadata["atlas_region_layers"][0]["color"] == [12, 34, 220]
    assert metadata["atlas_region_layers"][0]["has_mesh"] is True
    assert metadata["atlas_region_layers"][0]["mesh_path"] == "atlas_meshes/151.obj"
    reference_defaults = metadata["volume_render_defaults"]
    assert reference_defaults["preset"] == "napari_reference"
    assert reference_defaults["render_style"] == "raymarch"
    assert reference_defaults["palette"] == "grayscale"
    assert reference_defaults["background_theme"] == "dark"
    assert reference_defaults["threshold"] < 0.05
    assert reference_defaults["density"] >= 0.7
    assert reference_defaults["opacity"] >= 0.8
    assert reference_defaults["tf_mid"] >= 0.5
    assert reference_defaults["gradient_opacity"] is True
    assert len(metadata["volume_layers"]) == 1
    layer = metadata["volume_layers"][0]
    assert layer["layer_id"] == "annotation"
    assert layer["layer_role"] == "annotation"
    assert layer["volume_orientation"] == "zyx"
    assert layer["atlas_crop_bbox_zyx"] == [2, 12, 3, 15, 4, 18]
    assert layer["atlas_resolution_zyx_um"] == [25.0, 25.0, 25.0]
    assert layer["atlas_mesh_coordinate_space"] == "atlas_zyx_um"
    assert layer["voxel_spacing_zyx"] == [25.0, 25.0, 25.0]
    assert layer["volume_bounds"] == metadata["volume_bounds"]
    assert layer["label_volume"] is True
    assert layer["render_mode"] == "label_ids"
    assert layer["volume_grid_shape"] == [10, 12, 14]
    assert layer["atlas_region_layers"][0]["id"] == 151
    assert layer["volume_label_id_lut"]
    assert layer["volume_render_defaults"]["blend_mode"] == "normal"
    assert layer["volume_render_defaults"]["render_style"] == "raymarch"
    assert layer["volume_render_defaults"]["opacity"] >= 0.64
    assert layer["volume_render_defaults"]["density"] >= 0.9
    assert layer["volume_render_defaults"]["saturation"] >= 1.18
    assert layer["volume_render_defaults"]["gradient_opacity"] is True
    assert layer["volume_render_defaults"]["use_shading"] is True


def test_normalize_atlas_orientation_accepts_kim_asr_metadata() -> None:
    assert ThreeJsManager._normalize_atlas_orientation({"orientation": "asr"}) == "asr"
    assert ThreeJsManager._normalize_atlas_orientation({"orientation": "bad"}) == "zyx"


def test_normalize_atlas_crop_bbox_accepts_min_and_max_corners() -> None:
    assert ThreeJsManager._normalize_atlas_crop_bbox_zyx(
        {"crop_bbox_zyx": [25, 91, 133, 125, 262, 323]}
    ) == [25, 125, 91, 262, 133, 323]


def test_atlas_asset_roots_prefer_cropped_smooth_meshes(tmp_path: Path) -> None:
    source = tmp_path / "reference.tif"
    source.write_bytes(b"")
    mesh_dir = tmp_path / "meshes"
    smooth_mesh_dir = tmp_path / "meshes_cropped_smooth"
    mesh_dir.mkdir()
    smooth_mesh_dir.mkdir()

    assert ThreeJsManager._atlas_mesh_dir_for_source(tmp_path) == smooth_mesh_dir
    assert ThreeJsManager._atlas_asset_roots_for_tiff(source) == {
        "atlas_meshes": smooth_mesh_dir
    }


def test_load_atlas_region_catalog_falls_back_to_brainglobe_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    cache_dir = home / ".brainglobe" / "demo_mouse_25um_v1.0"
    cache_dir.mkdir(parents=True)
    (cache_dir / "structures.csv").write_text(
        "\n".join(
            [
                "id,name,acronym,rgb_triplet,structure_id_path,parent_structure_id",
                '151,Accessory olfactory bulb,AOB,"[20, 180, 240]",/997/151/,997',
                '188,Main olfactory bulb,MOB,"[240, 110, 40]",/997/188/,997',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    catalog = ThreeJsManager._load_atlas_region_catalog(
        source_dir=tmp_path / "atlas_crop",
        annotation_metadata={"volume_label_id_lut": [151, 188]},
        atlas_metadata={"source_atlas": "demo_mouse_25um"},
    )

    assert [entry["acronym"] for entry in catalog] == ["AOB", "MOB"]
    assert catalog[0]["name"] == "Accessory olfactory bulb"
    assert catalog[1]["color"] == [240, 110, 40]


@requires_tifffile
def test_resolve_tiff_payload_detects_reference_annotation_siblings(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    reference_path = tmp_path / "reference.tif"
    annotation_path = tmp_path / "annotation.tif"
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 10
    labels = np.zeros((4, 4, 4), dtype=np.uint16)
    labels[1:3, 1:3, 1:3] = 151
    labels[2:4, 2:4, 2:4] = 188
    tifffile.imwrite(str(reference_path), data)
    tifffile.imwrite(str(annotation_path), labels)

    payload_path = manager._resolve_tiff_simulation_payload(reference_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    assert payload["metadata"]["volume_overlay"] is True
    assert payload["metadata"]["volume_layers"][0]["layer_role"] == "annotation"


@requires_tifffile
def test_resolve_tiff_payload_does_not_replace_unrelated_tiff_with_atlas_siblings(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    reference_path = tmp_path / "reference.tif"
    annotation_path = tmp_path / "annotation.tif"
    image_path = tmp_path / "sub-BC08_res-25um_channel-green.tif"
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 10
    labels = np.zeros((4, 4, 4), dtype=np.uint16)
    labels[1:3, 1:3, 1:3] = 151
    standalone = np.zeros((4, 4, 4), dtype=np.uint16)
    standalone[2:4, 2:4, 2:4] = 255
    tifffile.imwrite(str(reference_path), data)
    tifffile.imwrite(str(annotation_path), labels)
    tifffile.imwrite(str(image_path), standalone)

    assert manager._resolve_tiff_overlay_paths(image_path) is None

    payload_path = manager._resolve_tiff_simulation_payload(image_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    metadata = payload["metadata"]
    assert metadata["source_path"] == str(image_path)
    assert "volume_overlay" not in metadata
    assert "volume_layers" not in metadata


@requires_tifffile
def test_build_tiff_payload_uses_filename_resolution_for_isotropic_stack(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    tif_path = tmp_path / "sub-BC08_res-25um_channel-green.tif"
    data = np.zeros((12, 10, 8), dtype=np.uint16)
    data[3:9, 2:8, 2:6] = 300
    tifffile.imwrite(
        str(tif_path),
        data,
        resolution=(72, 72),
        resolutionunit="INCH",
    )

    payload = manager._build_tiff_simulation_payload(tif_path)

    assert payload["metadata"]["shape"] == [12, 10, 8]
    assert payload["metadata"]["voxel_spacing_zyx"] == [25.0, 25.0, 25.0]
    assert payload["metadata"]["voxel_spacing_xyz"] == [25.0, 25.0, 25.0]
    assert payload["metadata"]["section_step_world"] == 25.0


@requires_tifffile
def test_build_tiff_payload_ignores_display_dpi_without_volume_spacing(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    tif_path = tmp_path / "plain_channel-green.tif"
    data = np.zeros((12, 10, 8), dtype=np.uint16)
    data[3:9, 2:8, 2:6] = 300
    tifffile.imwrite(
        str(tif_path),
        data,
        resolution=(72, 72),
        resolutionunit="INCH",
    )

    payload = manager._build_tiff_simulation_payload(tif_path)

    assert payload["metadata"]["shape"] == [12, 10, 8]
    assert payload["metadata"]["voxel_spacing_zyx"] == [1.0, 1.0, 1.0]
    assert payload["metadata"]["voxel_spacing_xyz"] == [1.0, 1.0, 1.0]
    assert payload["metadata"]["section_step_world"] == 1.0


@requires_tifffile
def test_build_tiff_simulation_payload_loads_allen_ontology_colors(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    tif_path = tmp_path / "allen_annotation_labels.tif"
    data = np.zeros((10, 10, 10), dtype=np.uint16)
    data[1:6, 1:6, 1:6] = 315
    data[4:9, 4:9, 4:9] = 671
    tifffile.imwrite(str(tif_path), data)

    ontology_csv = tmp_path / "structure_tree_safe_2017.csv"
    ontology_csv.write_text(
        "\n".join(
            [
                "id,acronym,name,color_hex_triplet",
                "315,VISp,Primary visual area,FF3366",
                "671,CP,Caudoputamen,33CC55",
                "997,root,root,FFFFFF",
            ]
        ),
        encoding="utf-8",
    )

    payload = manager._build_tiff_simulation_payload(tif_path)
    colors = payload["metadata"]["volume_label_colors"]

    assert payload["metadata"]["label_volume"] is True
    assert payload["metadata"]["render_mode"] == "label_ids"
    assert colors["315"] == [255, 51, 102, 255]
    assert colors["671"] == [51, 204, 85, 255]
    # Unused IDs are not included in the exported override map.
    assert "997" not in colors


@requires_tifffile
def test_build_tiff_simulation_payload_loads_allen_ontology_colors_from_json(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    tif_path = tmp_path / "allen_annotation_labels.tif"
    data = np.zeros((10, 10, 10), dtype=np.uint16)
    data[1:6, 1:6, 1:6] = 315
    data[4:9, 4:9, 4:9] = 671
    tifffile.imwrite(str(tif_path), data)

    ontology_json = tmp_path / "structure_tree_safe_2017.json"
    ontology_json.write_text(
        json.dumps(
            {
                "msg": [
                    {
                        "id": 315,
                        "acronym": "VISp",
                        "name": "Primary visual area",
                        "color_hex_triplet": "FF3366",
                    },
                    {
                        "id": 671,
                        "acronym": "CP",
                        "name": "Caudoputamen",
                        "color_hex_triplet": "33CC55",
                    },
                    {"id": 997, "acronym": "root", "color_hex_triplet": "FFFFFF"},
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = manager._build_tiff_simulation_payload(tif_path)
    colors = payload["metadata"]["volume_label_colors"]

    assert payload["metadata"]["label_volume"] is True
    assert payload["metadata"]["render_mode"] == "label_ids"
    assert colors["315"] == [255, 51, 102, 255]
    assert colors["671"] == [51, 204, 85, 255]
    assert "997" not in colors


@requires_tifffile
def test_build_tiff_simulation_payload_keeps_continuous_uint16_as_intensity(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    tif_path = tmp_path / "reference_uint16_volume.tif"
    z, y, x = np.indices((24, 24, 24), dtype=np.float32)
    center = np.array([11.5, 11.5, 11.5], dtype=np.float32)
    radius = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    signal = np.clip(4095.0 - radius * 220.0, 0.0, 4095.0)
    data = signal.astype(np.uint16)
    tifffile.imwrite(str(tif_path), data)

    payload = manager._build_tiff_simulation_payload(tif_path)

    assert payload["metadata"]["label_volume"] is False
    assert payload["metadata"]["render_mode"] == "gaussian_splatting"
    assert payload["metadata"]["label_id_count"] == 0
    assert not payload["metadata"]["volume_label_id_lut"]


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


@requires_zarr
def test_build_zarr_simulation_payload_uses_flat_array_axes_and_voxel_um(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    zarr_path = tmp_path / "atlas_nissl_30um_image_registered.zarr"
    arr = zarr.open(str(zarr_path), mode="w", shape=(12, 16, 20), dtype="u1")
    arr.attrs["axes"] = ["ML", "DV", "AP"]
    arr.attrs["voxel_um"] = 30.0
    data = np.zeros((12, 16, 20), dtype=np.uint8)
    data[:, 2:14, 2:18] = 240
    data[3:9, 4:12, 5:15] = 90
    data[5:7, 6:10, 8:12] = 45
    arr[:] = data

    payload = manager._build_zarr_simulation_payload(zarr_path)

    assert payload["metadata"]["axes"] == ["ml", "dv", "ap"]
    assert payload["metadata"]["voxel_spacing_zyx"] == [30.0, 30.0, 30.0]
    assert payload["metadata"]["voxel_spacing_xyz"] == [30.0, 30.0, 30.0]
    assert payload["metadata"]["section_step_world"] == 30.0
    assert payload["metadata"]["render_profile"] == "nissl_sections"
    assert payload["metadata"]["intensity_inverted"] is True
    assert payload["metadata"]["signal_polarity"] == "dark_on_light"
    assert payload["metadata"]["volume_crop_origin_zyx"] != [0, 0, 0]
    assert payload["metadata"]["volume_grid_shape"][1] < 16
    assert payload["metadata"]["point_count"] > 0


@requires_zarr
def test_build_zarr_simulation_payload_falls_back_when_candidate_read_fails(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    manager = ThreeJsManager(window, QtWidgets.QStackedWidget(window))

    zarr_path = tmp_path / "atlas_nissl_30um_image.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)

    fake_source = object()
    broken_array = object()
    good_array = object()
    monkeypatch.setattr(
        zarr,
        "open",
        lambda *_args, **_kwargs: fake_source,
    )
    monkeypatch.setattr(
        manager,
        "_resolve_zarr_array_candidates_for_viewer",
        lambda _source: [
            (broken_array, "0", ["z", "y", "x"]),
            (good_array, "1", ["z", "y", "x"]),
        ],
    )

    call_order: list[str] = []

    def _build_from_array(
        *,
        source,
        source_path,
        array,
        source_dataset_path,
        axis_names,
        np_module,
    ):
        _ = source
        _ = source_path
        _ = axis_names
        _ = np_module
        call_order.append(source_dataset_path)
        if array is broken_array:
            raise RuntimeError("error during blosc decompression: -1")
        return {"kind": "annolid-simulation-v1", "metadata": {"ok": True}}

    monkeypatch.setattr(manager, "_build_zarr_payload_from_array", _build_from_array)

    payload = manager._build_zarr_simulation_payload(zarr_path)

    assert payload["metadata"]["ok"] is True
    assert call_order == ["0", "1"]
