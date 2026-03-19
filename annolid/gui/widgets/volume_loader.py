from __future__ import annotations

from pathlib import Path
from typing import Any

from annolid.utils.logger import logger

try:
    import pyvista as pv
except Exception:
    pv = None  # type: ignore[assignment]


class VolumeSourceLoader:
    """Route a source path through the dedicated volume reader service."""

    def __init__(self, *, readers: Any) -> None:
        self._readers = readers

    def read_volume_any(self, path: Path) -> Any:
        """Read a 3D volume from TIFF/NIfTI/DICOM/Zarr."""
        try:
            readers = self._readers
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            logger.info(
                "3D viewer load: read_volume_any start path='%s' resolved='%s' exists=%s is_dir=%s",
                path,
                resolved,
                path.exists(),
                path.is_dir(),
            )

            if path.is_dir():
                logger.info("3D viewer load: source is a directory: '%s'.", path)
                if readers.is_zarr_candidate(path):
                    logger.info("3D viewer load: detected Zarr directory '%s'.", path)
                    return readers.read_zarr(path)
                logger.info(
                    "3D viewer load: treating directory as DICOM series '%s'.", path
                )
                volume, spacing = readers.read_dicom_series(path)
                logger.info(
                    "3D viewer load: DICOM directory loaded shape=%s dtype=%s spacing=%s",
                    getattr(volume, "shape", None),
                    getattr(volume, "dtype", None),
                    spacing,
                )
                return readers.make_simple_volume_data(volume, spacing)

            suffix = path.suffix.lower()
            name_lower = path.name.lower()
            if name_lower.endswith(".nii") or name_lower.endswith(".nii.gz"):
                logger.info(
                    "3D viewer load: using PyVista NIfTI reader for '%s'.", path
                )
                if pv is None:
                    raise RuntimeError(
                        "PyVista is unavailable; cannot read NIfTI volume."
                    )
                reader = pv.get_reader(str(path))
                logger.info(
                    "3D viewer load: PyVista selected reader '%s' for '%s'.",
                    type(reader).__name__,
                    path,
                )
                image = reader.read()
                if image is None:
                    raise RuntimeError(
                        "PyVista returned no image data for NIfTI source."
                    )
                vol = readers.vtk_image_to_numpy(image)
                s = getattr(image, "spacing", (1.0, 1.0, 1.0))
                spacing = (float(s[0]), float(s[1]), float(s[2]))
                logger.info(
                    "3D viewer load: NIfTI loaded shape=%s dtype=%s spacing=%s",
                    getattr(vol, "shape", None),
                    getattr(vol, "dtype", None),
                    spacing,
                )
                return readers.make_simple_volume_data(vol, spacing)

            if suffix in (".hdr", ".img"):
                logger.info("3D viewer load: using Analyze reader for '%s'.", path)
                return readers.read_analyze_volume(path)

            if suffix in (".dcm", ".ima", ".dicom"):
                logger.info(
                    "3D viewer load: using DICOM-series reader via parent '%s'.",
                    path.parent,
                )
                volume, spacing = readers.read_dicom_series(path.parent)
                logger.info(
                    "3D viewer load: DICOM file loaded shape=%s dtype=%s spacing=%s",
                    getattr(volume, "shape", None),
                    getattr(volume, "dtype", None),
                    spacing,
                )
                return readers.make_simple_volume_data(volume, spacing)

            if readers.is_zarr_candidate(path):
                logger.info("3D viewer load: detected Zarr source '%s'.", path)
                return readers.read_zarr(path)

            if readers.is_tiff_candidate(path):
                logger.info("3D viewer load: detected TIFF source '%s'.", path)
                if readers.should_use_out_of_core_tiff(path):
                    logger.info(
                        "3D viewer load: TIFF out-of-core mode selected for '%s'.",
                        path,
                    )
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
            logger.exception(
                "3D viewer load: read_volume_any failed for path='%s'.",
                path,
            )
            raise RuntimeError(f"Failed to read volume from '{path}': {exc}")
