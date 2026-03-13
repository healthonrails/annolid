from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from qtpy import QtGui

from .base import LargeImageLoadResult, LargeImageMetadata


TIFF_SUFFIXES = {".tif", ".tiff", ".ome.tif", ".ome.tiff", ".btf", ".tf8"}


def normalize_suffixes(path: Path) -> set[str]:
    suffixes = [s.lower() for s in path.suffixes]
    normalized = set(suffixes)
    if len(suffixes) >= 2:
        normalized.add("".join(suffixes[-2:]))
    return normalized


def is_large_tiff_path(path: str | Path) -> bool:
    return bool(normalize_suffixes(Path(path)) & TIFF_SUFFIXES)


def array_to_qimage(array: np.ndarray) -> QtGui.QImage:
    arr = np.asarray(array)
    if arr.ndim == 2:
        mode = "L"
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    else:
        raise ValueError(f"Unsupported image array shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            arr = np.zeros(arr.shape, dtype=np.uint8)
        else:
            low = float(finite.min())
            high = float(finite.max())
            if high <= low:
                arr = np.zeros(arr.shape, dtype=np.uint8)
            else:
                arr = np.clip((arr - low) * 255.0 / (high - low), 0, 255).astype(
                    np.uint8
                )

    image = Image.fromarray(arr, mode=mode)
    buffer = image.tobytes("raw", mode)
    if mode == "L":
        qimage = QtGui.QImage(
            buffer,
            image.width,
            image.height,
            image.width,
            QtGui.QImage.Format_Grayscale8,
        )
    elif mode == "RGB":
        qimage = QtGui.QImage(
            buffer,
            image.width,
            image.height,
            image.width * 3,
            QtGui.QImage.Format_RGB888,
        )
    else:
        qimage = QtGui.QImage(
            buffer,
            image.width,
            image.height,
            image.width * 4,
            QtGui.QImage.Format_RGBA8888,
        )
    return qimage.copy()


def load_with_pillow(path: Path) -> LargeImageLoadResult:
    with Image.open(path) as image:
        image.load()
        if image.mode not in {"L", "RGB", "RGBA"}:
            image = image.convert("RGBA")
        qimage = array_to_qimage(np.asarray(image))
        meta = LargeImageMetadata(
            backend_name="pillow",
            width=image.width,
            height=image.height,
            channels=len(image.getbands()),
            dtype="uint8",
        )
        return LargeImageLoadResult(qimage=qimage, metadata=meta)
