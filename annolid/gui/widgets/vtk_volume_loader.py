from __future__ import annotations

from pathlib import Path
from typing import Any

from annolid.utils.logger import logger


class VolumeSourceLoader:
    """Route a source path through the dedicated volume reader service."""

    def __init__(self, *, readers: Any) -> None:
        self._readers = readers

    def read_volume_any(self, path: Path) -> Any:
        """Read a 3D volume from TIFF/NIfTI/DICOM/Zarr."""
        try:
            readers = self._readers

            if path.is_dir():
                if readers.is_zarr_candidate(path):
                    return readers.read_zarr(path)
                volume, spacing = readers.read_dicom_series(path)
                return readers.make_simple_volume_data(volume, spacing)

            suffix = path.suffix.lower()
            name_lower = path.name.lower()
            if name_lower.endswith(".nii") or name_lower.endswith(".nii.gz"):
                try:
                    from vtkmodules.vtkIOImage import vtkNIFTIImageReader
                except Exception as exc:
                    raise RuntimeError(
                        "VTK NIFTI reader is not available in this build."
                    ) from exc
                reader = vtkNIFTIImageReader()
                reader.SetFileName(str(path))
                reader.Update()
                vtk_img = reader.GetOutput()
                vol = readers.vtk_image_to_numpy(vtk_img)
                s = vtk_img.GetSpacing()
                spacing = (s[0], s[1], s[2])
                return readers.make_simple_volume_data(vol, spacing)

            if suffix in (".hdr", ".img"):
                return readers.read_analyze_volume(path)

            if suffix in (".dcm", ".ima", ".dicom"):
                volume, spacing = readers.read_dicom_series(path.parent)
                return readers.make_simple_volume_data(volume, spacing)

            if readers.is_zarr_candidate(path):
                return readers.read_zarr(path)

            if readers.is_tiff_candidate(path):
                if readers.should_use_out_of_core_tiff(path):
                    return readers.read_tiff_out_of_core(path)
                try:
                    return readers.read_tiff_eager(path)
                except MemoryError as exc:
                    logger.warning(
                        "Standard TIFF loading failed (%s). Retrying with out-of-core caching.",
                        exc,
                    )
                    return readers.read_tiff_out_of_core(path)

            raise RuntimeError(f"Unsupported volume format: {path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to read volume from '{path}': {exc}")
