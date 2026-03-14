from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import (
    LargeImageBackend,
    LargeImageBackendCapabilities,
    LargeImageLoadResult,
    LargeImageMetadata,
)
from .common import array_to_qimage, is_large_tiff_path


class VipsBackend(LargeImageBackend):
    name = "pyvips"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._image = None

    @staticmethod
    def _import_pyvips():
        import pyvips

        return pyvips

    def can_handle(self, path: Path) -> bool:
        if not is_large_tiff_path(path):
            return False
        try:
            self._import_pyvips()
            return True
        except Exception:
            return False

    def open(self, path: Path) -> "VipsBackend":
        backend = VipsBackend(path)
        pyvips = backend._import_pyvips()
        backend._image = pyvips.Image.new_from_file(str(path), access="random")
        return backend

    def _opened_image(self):
        if self._image is None:
            if self.path is None:
                raise ValueError("Backend is not opened")
            self._image = self.open(self.path)._image
        return self._image

    def probe(self, path: str | Path | None = None):
        resolved = Path(path) if path is not None else self.path
        if resolved is None:
            return None
        image = self._opened_image() if path is None else self.open(resolved)._image
        if image is None:
            return None
        return LargeImageMetadata(
            backend_name=self.name,
            width=int(image.width),
            height=int(image.height),
            channels=int(image.bands),
            dtype=str(image.format),
            levels=1,
        )

    def get_level_count(self) -> int:
        return 1

    def get_level_shape(self, level: int) -> tuple[int, int]:
        if level != 0:
            raise IndexError("VipsBackend currently exposes a single logical level")
        image = self._opened_image()
        return int(image.width), int(image.height)

    def read_region(self, x: int, y: int, w: int, h: int, level: int = 0):
        if level != 0:
            raise IndexError("VipsBackend currently exposes a single logical level")
        image = self._opened_image()
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        width = max(0, min(int(w), int(image.width) - x0))
        height = max(0, min(int(h), int(image.height) - y0))
        if width <= 0 or height <= 0:
            return np.zeros((0, 0, int(image.bands)), dtype=np.uint8)
        region = image.crop(x0, y0, width, height)
        return region.numpy()

    def get_thumbnail(self, max_size: int = 2048):
        image = self._opened_image()
        if max(int(image.width), int(image.height)) <= int(max_size):
            return image.numpy()
        thumb = image.thumbnail_image(int(max_size), height=int(max_size), size="down")
        return thumb.numpy()

    def load(self, path: str | Path | None = None) -> LargeImageLoadResult:
        if path is not None:
            opened = self.open(Path(path))
            metadata = opened.probe()
            thumbnail = opened.get_thumbnail()
        else:
            metadata = self.probe()
            thumbnail = self.get_thumbnail()
        return LargeImageLoadResult(
            qimage=array_to_qimage(np.asarray(thumbnail)),
            metadata=metadata,
        )

    def capabilities(self) -> LargeImageBackendCapabilities:
        return LargeImageBackendCapabilities(
            supports_pages=False,
            supports_pyramids=False,
            supports_region_reads=True,
            supports_label_stack=False,
            supports_metadata_axes=False,
            supports_cache_optimization=True,
        )
