from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from qtpy import QtCore, QtGui

from .base import LargeImageBackend, LargeImageLoadResult, LargeImageMetadata
from .common import is_large_tiff_path, load_with_pillow
from .openslide_backend import OpenSlideBackend
from .tifffile_backend import TiffFileBackend
from .vips_backend import VipsBackend


def _backend_usability_report(path: Path) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for backend in (OpenSlideBackend(), VipsBackend(), TiffFileBackend()):
        try:
            usable = bool(backend.can_handle(path))
            reason = "available" if usable else "unavailable"
        except Exception as exc:
            usable = False
            reason = str(exc)
        report[backend.name] = {
            "usable": usable,
            "reason": reason,
        }
    return report


def _recommended_large_image_backend_name(
    info: dict[str, object], support: dict[str, dict[str, object]]
) -> str:
    if bool(support.get("openslide", {}).get("usable")) and bool(
        info.get("openslide_compatible")
    ):
        return "openslide"
    if bool(support.get("pyvips", {}).get("usable")) and (
        bool(info.get("is_pyramidal")) or bool(info.get("huge_2d"))
    ):
        return "pyvips"
    if bool(support.get("tifffile", {}).get("usable")):
        return "tifffile"
    return "qt"


def _performance_hint_for_backend(
    info: dict[str, object],
    support: dict[str, dict[str, object]],
    selected_backend: str,
) -> str | None:
    if selected_backend in {"openslide", "pyvips"}:
        return None
    if not bool(info.get("is_tiff_family")):
        return None
    if not (bool(info.get("huge_2d")) or bool(info.get("is_pyramidal"))):
        return None
    if not bool(support.get("pyvips", {}).get("usable")):
        return (
            "For faster pan/zoom on very large TIFF files, install the libvips runtime "
            "and enable the optional large_image backend."
        )
    if bool(info.get("openslide_compatible")) and not bool(
        support.get("openslide", {}).get("usable")
    ):
        return (
            "This image can use OpenSlide-style navigation, but the OpenSlide runtime "
            "is not available in the current environment."
        )
    return None


class QtImageBackend(LargeImageBackend):
    name = "qt"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None

    def can_handle(self, path: Path) -> bool:
        qimage = QtGui.QImage(str(path))
        return not qimage.isNull()

    def open(self, path: Path) -> "QtImageBackend":
        return QtImageBackend(path)

    def probe(self, path: str | Path | None = None):
        resolved = Path(path) if path is not None else self.path
        if resolved is None:
            return None
        qimage = QtGui.QImage(str(resolved))
        if qimage.isNull():
            return None
        return LargeImageMetadata(
            backend_name=self.name,
            width=qimage.width(),
            height=qimage.height(),
        )

    def get_level_count(self) -> int:
        return 1

    def get_level_shape(self, level: int) -> tuple[int, int]:
        if level != 0:
            raise IndexError("QtImageBackend only exposes level 0")
        metadata = self.probe()
        if metadata is None:
            raise ValueError("Backend is not opened")
        return metadata.width, metadata.height

    def read_region(self, x: int, y: int, w: int, h: int, level: int = 0):
        if level != 0:
            raise IndexError("QtImageBackend only exposes level 0")
        if self.path is None:
            raise ValueError("Backend is not opened")
        qimage = QtGui.QImage(str(self.path))
        return qimage.copy(int(x), int(y), int(w), int(h))

    def get_thumbnail(self, max_size: int = 2048):
        if self.path is None:
            raise ValueError("Backend is not opened")
        qimage = QtGui.QImage(str(self.path))
        return qimage.scaled(
            max_size,
            max_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

    def load(self, path: str | Path | None = None) -> LargeImageLoadResult:
        resolved = Path(path) if path is not None else self.path
        if resolved is None:
            raise ValueError("Backend is not opened")
        qimage = QtGui.QImage(str(resolved))
        if qimage.isNull():
            raise ValueError(f"Qt could not load image: {resolved}")
        return LargeImageLoadResult(qimage=qimage, metadata=self.probe(resolved))


class PillowImageBackend(LargeImageBackend):
    name = "pillow"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None

    def can_handle(self, path: Path) -> bool:
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            return True
        except Exception:
            return False

    def open(self, path: Path) -> "PillowImageBackend":
        return PillowImageBackend(path)

    def probe(self, path: str | Path | None = None):
        resolved = Path(path) if path is not None else self.path
        if resolved is None:
            return None
        return load_with_pillow(resolved).metadata

    def get_level_count(self) -> int:
        return 1

    def get_level_shape(self, level: int) -> tuple[int, int]:
        if level != 0:
            raise IndexError("PillowImageBackend only exposes level 0")
        metadata = self.probe()
        if metadata is None:
            raise ValueError("Backend is not opened")
        return metadata.width, metadata.height

    def read_region(self, x: int, y: int, w: int, h: int, level: int = 0):
        if level != 0:
            raise IndexError("PillowImageBackend only exposes level 0")
        from PIL import Image

        if self.path is None:
            raise ValueError("Backend is not opened")
        with Image.open(self.path) as image:
            return image.crop((int(x), int(y), int(x + w), int(y + h)))

    def get_thumbnail(self, max_size: int = 2048):
        from PIL import Image

        if self.path is None:
            raise ValueError("Backend is not opened")
        with Image.open(self.path) as image:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            return image.copy()

    def load(self, path: str | Path | None = None) -> LargeImageLoadResult:
        resolved = Path(path) if path is not None else self.path
        if resolved is None:
            raise ValueError("Backend is not opened")
        return load_with_pillow(resolved)


def available_large_image_backends() -> list[LargeImageBackend]:
    return [
        QtImageBackend(),
        TiffFileBackend(),
        VipsBackend(),
        OpenSlideBackend(),
        PillowImageBackend(),
    ]


def sniff_large_image(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    info: dict[str, object] = {
        "path": str(resolved),
        "is_tiff_family": is_large_tiff_path(resolved),
        "is_pyramidal": False,
        "is_ome": False,
        "huge_2d": False,
        "openslide_compatible": False,
    }
    if not is_large_tiff_path(resolved):
        return info
    support = _backend_usability_report(resolved)
    probe = None
    try:
        if bool(support.get("tifffile", {}).get("usable")):
            probe = TiffFileBackend(resolved).probe()
    except Exception:
        probe = None
    if probe is not None:
        info["width"] = probe.width
        info["height"] = probe.height
        info["channels"] = probe.channels
        info["dtype"] = probe.dtype
        info["levels"] = probe.levels
        info["is_ome"] = probe.is_ome
        info["is_pyramidal"] = (probe.levels or 1) > 1
        info["huge_2d"] = max(probe.width, probe.height) >= 8192
        info["physical_pixel_size_x"] = probe.physical_pixel_size_x
        info["physical_pixel_size_y"] = probe.physical_pixel_size_y
        info["physical_pixel_unit"] = probe.physical_pixel_unit
    try:
        info["openslide_compatible"] = bool(support.get("openslide", {}).get("usable"))
    except Exception:
        info["openslide_compatible"] = False
    info["backend_support"] = support
    info["recommended_backend"] = _recommended_large_image_backend_name(info, support)
    info["performance_hint"] = _performance_hint_for_backend(
        info, support, str(info["recommended_backend"])
    )
    return info


def _probe_large_tiff_with_available_backend(
    path: Path, info: dict[str, object] | None = None
) -> LargeImageMetadata | None:
    sniffed = info if info is not None else sniff_large_image(path)
    support = dict(sniffed.get("backend_support") or {})
    if bool(support.get("tifffile", {}).get("usable")):
        try:
            return TiffFileBackend(path).probe()
        except Exception:
            pass
    for backend in (
        OpenSlideBackend(),
        VipsBackend(),
        QtImageBackend(),
        PillowImageBackend(),
    ):
        try:
            if not backend.can_handle(path):
                continue
            return backend.open(path).probe()
        except Exception:
            continue
    return None


def open_large_image(path: str | Path) -> LargeImageBackend:
    resolved = Path(path)
    info = sniff_large_image(resolved)
    support = dict(info.get("backend_support") or {})
    candidates: list[LargeImageBackend] = []
    if bool(info.get("openslide_compatible")) and bool(
        support.get("openslide", {}).get("usable")
    ):
        candidates.append(OpenSlideBackend())
    if (bool(info.get("is_pyramidal")) or bool(info.get("huge_2d"))) and bool(
        support.get("pyvips", {}).get("usable")
    ):
        candidates.append(VipsBackend())
    if bool(support.get("tifffile", {}).get("usable")):
        candidates.append(TiffFileBackend())
    candidates.extend([QtImageBackend(), PillowImageBackend()])
    for backend in candidates:
        try:
            if not backend.can_handle(resolved):
                continue
            return backend.open(resolved)
        except Exception:
            continue
    return PillowImageBackend().open(resolved)


def _ordered_backends_for_path(path: Path) -> list[LargeImageBackend]:
    backends = available_large_image_backends()
    if not is_large_tiff_path(path):
        return backends
    preferred = ("openslide", "pyvips", "tifffile", "qt", "pillow")
    ordered: list[LargeImageBackend] = []
    for name in preferred:
        ordered.extend([backend for backend in backends if backend.name == name])
    return ordered


def load_image_with_backends(path: str | Path) -> LargeImageLoadResult:
    resolved = Path(path)
    if is_large_tiff_path(resolved):
        backend = open_large_image(resolved)
        result = backend.load()
        info = sniff_large_image(resolved)
        if result.metadata is not None:
            metadata = replace(
                result.metadata,
                recommended_backend=str(
                    info.get("recommended_backend")
                    or result.metadata.recommended_backend
                    or ""
                )
                or None,
                performance_hint=str(
                    info.get("performance_hint")
                    or result.metadata.performance_hint
                    or ""
                )
                or None,
            )
            return LargeImageLoadResult(qimage=result.qimage, metadata=metadata)
        return result
    for backend in _ordered_backends_for_path(resolved):
        if not backend.can_handle(resolved):
            continue
        try:
            return backend.open(resolved).load()
        except Exception:
            continue
    return load_with_pillow(resolved)


def probe_large_image(path: str | Path):
    resolved = Path(path)
    if is_large_tiff_path(resolved):
        info = sniff_large_image(resolved)
        metadata = _probe_large_tiff_with_available_backend(resolved, info=info)
        if metadata is not None:
            metadata = replace(
                metadata,
                recommended_backend=str(
                    info.get("recommended_backend")
                    or metadata.recommended_backend
                    or ""
                )
                or None,
                performance_hint=str(
                    info.get("performance_hint") or metadata.performance_hint or ""
                )
                or None,
            )
        return metadata
    for backend in _ordered_backends_for_path(resolved):
        if not backend.can_handle(resolved):
            continue
        try:
            return backend.open(resolved).probe()
        except Exception:
            continue
    return None
