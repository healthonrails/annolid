from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from qtpy import QtGui


@dataclass(frozen=True)
class LargeImageMetadata:
    backend_name: str
    width: int
    height: int
    channels: Optional[int] = None
    dtype: Optional[str] = None
    levels: Optional[int] = None
    page_count: Optional[int] = None
    axes: Optional[str] = None
    is_ome: bool = False
    physical_pixel_size_x: Optional[float] = None
    physical_pixel_size_y: Optional[float] = None
    physical_pixel_unit: Optional[str] = None
    recommended_backend: Optional[str] = None
    performance_hint: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": self.dtype,
            "levels": self.levels,
            "page_count": self.page_count,
            "axes": self.axes,
            "is_ome": self.is_ome,
            "physical_pixel_size_x": self.physical_pixel_size_x,
            "physical_pixel_size_y": self.physical_pixel_size_y,
            "physical_pixel_unit": self.physical_pixel_unit,
            "recommended_backend": self.recommended_backend,
            "performance_hint": self.performance_hint,
        }


@dataclass(frozen=True)
class LargeImageLoadResult:
    qimage: QtGui.QImage
    metadata: Optional[LargeImageMetadata]


@dataclass(frozen=True)
class LargeImageBackendCapabilities:
    supports_pages: bool = False
    supports_pyramids: bool = False
    supports_region_reads: bool = True
    supports_label_stack: bool = False
    supports_metadata_axes: bool = False
    supports_cache_optimization: bool = False


class LargeImageBackend(ABC):
    name: str

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def open(self, path: Path) -> "LargeImageBackend":
        raise NotImplementedError

    @abstractmethod
    def probe(self, path: str | Path | None = None) -> Optional[LargeImageMetadata]:
        raise NotImplementedError

    @abstractmethod
    def get_level_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_level_shape(self, level: int) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def read_region(self, x: int, y: int, w: int, h: int, level: int = 0) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_thumbnail(self, max_size: int = 2048) -> Any:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path | None = None) -> LargeImageLoadResult:
        raise NotImplementedError

    def capabilities(self) -> LargeImageBackendCapabilities:
        return LargeImageBackendCapabilities(
            supports_pages=self.get_page_count() > 1,
            supports_pyramids=self.get_level_count() > 1,
            supports_region_reads=True,
            supports_label_stack=self.get_page_count() > 1,
            supports_metadata_axes=False,
            supports_cache_optimization=False,
        )

    def get_page_count(self) -> int:
        return 1

    def get_current_page(self) -> int:
        return 0

    def set_page(self, page_index: int) -> None:
        if int(page_index) != 0:
            raise IndexError("This backend only exposes page 0")
