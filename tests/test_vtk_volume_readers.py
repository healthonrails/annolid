from __future__ import annotations

import numpy as np

from annolid.gui.widgets.vtk_volume_readers import VolumeReaderConfig, VolumeReaders


def _make_volume_readers() -> VolumeReaders:
    cfg = VolumeReaderConfig(
        dicom_exts=(".dcm",),
        tiff_suffixes=(".tif", ".tiff"),
        ome_tiff_suffixes=(".ome.tif", ".ome.tiff"),
        auto_out_of_core_mb=0.0,
        max_volume_voxels=1_000_000_000,
        slice_mode_bytes=float("inf"),
    )
    return VolumeReaders(
        config=cfg,
        make_volume_data=lambda **kwargs: kwargs,
        make_slice_volume_data=lambda *args, **kwargs: kwargs,
        find_companion_file=lambda _path, _suffix: None,
        memmap_slice_loader_cls=object,
        tiff_slice_loader_cls=object,
        zarr_slice_loader_cls=object,
        zarr_v3_array_cls=np.ndarray,
    )


def test_resolve_initial_source_prefers_volume_over_point_cloud(tmp_path) -> None:
    (tmp_path / "annotations.csv").write_text("x,y,z\n1,2,3\n", encoding="utf-8")
    (tmp_path / "structural.nii.gz").write_bytes(b"")

    readers = _make_volume_readers()
    selected = readers.resolve_initial_source(tmp_path)

    assert selected.name == "structural.nii.gz"


def test_resolve_initial_source_falls_back_to_point_cloud_when_no_volume(
    tmp_path,
) -> None:
    (tmp_path / "annotations.csv").write_text("x,y,z\n1,2,3\n", encoding="utf-8")

    readers = _make_volume_readers()
    selected = readers.resolve_initial_source(tmp_path)

    assert selected.name == "annotations.csv"
