from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import LargeImageBackend, LargeImageLoadResult, LargeImageMetadata
from .common import array_to_qimage, is_large_tiff_path


def _extract_physical_pixel_metadata(
    tif,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    unit = None
    size_x = None
    size_y = None
    try:
        ome_meta = getattr(tif, "ome_metadata", None)
        if isinstance(ome_meta, str) and "PhysicalSizeX" in ome_meta:
            import re

            match_x = re.search(r'PhysicalSizeX="([^"]+)"', ome_meta)
            match_y = re.search(r'PhysicalSizeY="([^"]+)"', ome_meta)
            match_unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', ome_meta)
            if match_x:
                size_x = float(match_x.group(1))
            if match_y:
                size_y = float(match_y.group(1))
            if match_unit:
                unit = match_unit.group(1)
    except Exception:
        pass
    return size_x, size_y, unit


class TiffFileBackend(LargeImageBackend):
    name = "tifffile"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._metadata_cache: Optional[LargeImageMetadata] = None
        self._level_shapes_cache: dict[int, tuple[int, int]] = {}
        self._level_memmap_cache: dict[int, np.ndarray] = {}

    def can_handle(self, path: Path) -> bool:
        if not is_large_tiff_path(path):
            return False
        try:
            import tifffile  # noqa: F401

            return True
        except Exception:
            return False

    def open(self, path: Path) -> "TiffFileBackend":
        return TiffFileBackend(path)

    def _require_path(self) -> Path:
        if self.path is None:
            raise ValueError("Backend is not opened")
        return self.path

    @staticmethod
    def _series_level(series: Any, level: int = 0) -> Any:
        levels = list(getattr(series, "levels", []) or [])
        if not levels:
            if level != 0:
                raise IndexError(f"Invalid TIFF pyramid level: {level}")
            return series
        if level < 0 or level >= len(levels):
            raise IndexError(f"Invalid TIFF pyramid level: {level}")
        return levels[level]

    @classmethod
    def _normalized_shape(cls, series: Any, level: int = 0) -> tuple[int, ...]:
        target = cls._series_level(series, level=level)
        shape = tuple(int(v) for v in getattr(target, "shape", ()) or ())
        while len(shape) > 3:
            shape = shape[1:]
        if len(shape) == 3:
            if shape[0] in (3, 4) and shape[-1] not in (3, 4):
                shape = (shape[1], shape[2], shape[0])
        return shape

    @classmethod
    def _normalized_level_shape(cls, series: Any, level: int = 0) -> tuple[int, int]:
        shape = cls._normalized_shape(series, level=level)
        width = int(shape[-1]) if len(shape) == 1 else int(shape[1])
        height = int(shape[0]) if shape else 0
        return width, height

    def _memmap_level_array(self, level: int = 0) -> np.ndarray | None:
        if level in self._level_memmap_cache:
            return self._level_memmap_cache[level]
        path = self._require_path()
        try:
            arr = np.asarray(__import__("tifffile").memmap(path, series=0, level=level))
        except Exception:
            return None
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        self._level_memmap_cache[level] = arr
        return arr

    def probe(self, path: str | Path | None = None) -> Optional[LargeImageMetadata]:
        import tifffile

        resolved = Path(path) if path is not None else self.path
        if resolved is None:
            return None
        if path is None and self._metadata_cache is not None:
            return self._metadata_cache
        with tifffile.TiffFile(resolved) as tif:
            series = tif.series[0] if tif.series else None
            axes = str(getattr(series, "axes", "") or "")
            width, height = self._normalized_level_shape(series, level=0)
            shape = self._normalized_shape(series, level=0)
            channels = None
            if len(shape) >= 3:
                if shape[-1] in (3, 4):
                    channels = int(shape[-1])
                elif axes:
                    upper_axes = axes.upper()
                    for idx, axis in enumerate(upper_axes[-len(shape) :]):
                        if axis in {"C", "S"}:
                            channels = int(shape[idx])
                            break
            levels = len(getattr(series, "levels", []) or [])
            page_count = len(tif.pages)
            dtype = None
            if series is not None and getattr(series, "dtype", None) is not None:
                dtype = str(series.dtype)
            size_x, size_y, unit = _extract_physical_pixel_metadata(tif)
            metadata = LargeImageMetadata(
                backend_name=self.name,
                width=width,
                height=height,
                channels=channels,
                dtype=dtype,
                levels=levels if levels > 0 else 1,
                page_count=page_count,
                axes=axes or None,
                is_ome=bool(getattr(tif, "is_ome", False)),
                physical_pixel_size_x=size_x,
                physical_pixel_size_y=size_y,
                physical_pixel_unit=unit,
            )
            if path is None:
                self._metadata_cache = metadata
            return metadata

    @staticmethod
    def _select_series_array(series: Any, level: int = 0) -> np.ndarray:
        target = series
        levels = list(getattr(series, "levels", []) or [])
        if levels:
            if level < 0 or level >= len(levels):
                raise IndexError(f"Invalid TIFF pyramid level: {level}")
            target = levels[level]
        arr = target.asarray()
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        return arr

    def get_level_count(self) -> int:
        import tifffile

        path = self._require_path()
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            levels = list(getattr(series, "levels", []) or [])
            return len(levels) if levels else 1

    def get_level_shape(self, level: int) -> tuple[int, int]:
        import tifffile

        if level in self._level_shapes_cache:
            return self._level_shapes_cache[level]
        path = self._require_path()
        with tifffile.TiffFile(path) as tif:
            size = self._normalized_level_shape(tif.series[0], level=level)
        self._level_shapes_cache[level] = size
        return size

    def read_region(self, x: int, y: int, w: int, h: int, level: int = 0):
        import tifffile

        image = self._memmap_level_array(level=level)
        if image is None:
            path = self._require_path()
            with tifffile.TiffFile(path) as tif:
                image = self._select_series_array(tif.series[0], level=level)
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(int(x + w), image.shape[1])
        y1 = min(int(y + h), image.shape[0])
        return image[y0:y1, x0:x1]

    def get_thumbnail(self, max_size: int = 2048):
        import tifffile

        path = self._require_path()
        level_count = self.get_level_count()
        level = max(0, level_count - 1)
        image = self._memmap_level_array(level=level)
        if image is None:
            with tifffile.TiffFile(path) as tif:
                image = self._select_series_array(tif.series[0], level=level)
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            step = max(1, int(np.ceil(max(height, width) / float(max_size))))
            image = image[::step, ::step]
        pil_image = Image.fromarray(
            image.astype(np.uint8) if image.dtype == np.bool_ else image
        )
        if pil_image.mode not in {"L", "RGB", "RGBA"}:
            pil_image = pil_image.convert("RGBA")
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return np.asarray(pil_image)

    def load(self, path: str | Path | None = None) -> LargeImageLoadResult:
        resolved = Path(path) if path is not None else self.path
        if resolved is None:
            raise ValueError("Backend is not opened")
        if self.path is None:
            self.path = resolved
        metadata = self.probe(resolved)
        thumbnail = self.get_thumbnail()
        return LargeImageLoadResult(
            qimage=array_to_qimage(thumbnail),
            metadata=metadata,
        )


def probe_tiff_metadata(path: str | Path) -> LargeImageMetadata:
    metadata = TiffFileBackend(path).probe()
    if metadata is None:
        raise ValueError(f"Could not probe TIFF metadata for {path}")
    return metadata


def load_tiff_with_tifffile(
    path: str | Path, *, max_preview_size: int = 4096
) -> LargeImageLoadResult:
    backend = TiffFileBackend(path)
    thumbnail = backend.get_thumbnail(max_preview_size)
    return LargeImageLoadResult(
        qimage=array_to_qimage(thumbnail),
        metadata=backend.probe(),
    )
