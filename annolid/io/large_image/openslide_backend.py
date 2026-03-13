from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import LargeImageBackend, LargeImageLoadResult, LargeImageMetadata
from .common import array_to_qimage, is_large_tiff_path


class OpenSlideBackend(LargeImageBackend):
    name = "openslide"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._slide = None

    @staticmethod
    def _import_openslide():
        import openslide

        return openslide

    def can_handle(self, path: Path) -> bool:
        if not is_large_tiff_path(path):
            return False
        try:
            openslide = self._import_openslide()
            return bool(openslide.OpenSlide.detect_format(str(path)))
        except Exception:
            return False

    def open(self, path: Path) -> "OpenSlideBackend":
        openslide = self._import_openslide()
        backend = OpenSlideBackend(path)
        backend._slide = openslide.OpenSlide(str(path))
        return backend

    def _opened_slide(self):
        if self._slide is None:
            if self.path is None:
                raise ValueError("Backend is not opened")
            self._slide = self.open(self.path)._slide
        return self._slide

    def probe(self, path: str | Path | None = None):
        resolved = Path(path) if path is not None else self.path
        if resolved is None:
            return None
        slide = self._opened_slide() if path is None else self.open(resolved)._slide
        if slide is None:
            return None
        return LargeImageMetadata(
            backend_name=self.name,
            width=int(slide.dimensions[0]),
            height=int(slide.dimensions[1]),
            levels=int(slide.level_count),
        )

    def get_level_count(self) -> int:
        slide = self._opened_slide()
        return int(slide.level_count)

    def get_level_shape(self, level: int) -> tuple[int, int]:
        slide = self._opened_slide()
        dims = slide.level_dimensions[int(level)]
        return int(dims[0]), int(dims[1])

    def read_region(self, x: int, y: int, w: int, h: int, level: int = 0):
        slide = self._opened_slide()
        region = slide.read_region((int(x), int(y)), int(level), (int(w), int(h)))
        return np.asarray(region.convert("RGBA"))

    def get_thumbnail(self, max_size: int = 2048):
        slide = self._opened_slide()
        thumb = slide.get_thumbnail((int(max_size), int(max_size)))
        return np.asarray(thumb.convert("RGBA"))

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
