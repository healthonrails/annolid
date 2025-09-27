"""Qt-friendly wrapper around Dinov3 patch similarity.

This module provides a reusable service object that keeps a cached instance of
``Dinov3FeatureExtractor`` and executes the heavy lifting on a dedicated
``QThread``.  The service exposes Qt signals so GUI widgets can trigger the
computation, stay responsive, and receive either the similarity outputs or a
human readable error message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from PIL import Image
from qtpy import QtCore

from annolid.features import (
    Dinov3Config,
    Dinov3FeatureExtractor,
    Dinov3PCAMapper,
)
from annolid.features.dinov3_patch_similarity import DinoPatchSimilarity


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EXTRACTOR_CACHE: Dict[Tuple[str, int, Optional[str]],
                       Dinov3FeatureExtractor] = {}


def _get_or_create_extractor(model_name: str,
                             short_side: int,
                             device: Optional[str]) -> Dinov3FeatureExtractor:
    """Load (and cache) a Dinov3FeatureExtractor for the given configuration."""

    key = (model_name, short_side, device)
    extractor = _EXTRACTOR_CACHE.get(key)
    if extractor is not None:
        return extractor

    cfg = Dinov3Config(model_name=model_name,
                       short_side=short_side,
                       device=device)
    extractor = Dinov3FeatureExtractor(cfg)
    _EXTRACTOR_CACHE[key] = extractor
    return extractor


@dataclass
class DinoPatchRequest:
    image: Image.Image
    click_xy: Tuple[int, int]
    model_name: str
    short_side: int = 768
    device: Optional[str] = None
    alpha: float = 0.55


class _DinoPatchWorker(QtCore.QObject):
    """Worker object that performs the similarity computation on a QThread."""

    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, request: DinoPatchRequest) -> None:
        super().__init__()
        self._request = request

    @QtCore.Slot()
    def run(self) -> None:
        try:
            extractor = _get_or_create_extractor(
                self._request.model_name,
                self._request.short_side,
                self._request.device,
            )
            engine = DinoPatchSimilarity(extractor)
            result = engine.similarity(
                self._request.image,
                click_xy=self._request.click_xy,
                alpha=self._request.alpha,
                return_overlay=True,
            )

            overlay_rgba = None
            if result.overlay is not None:
                overlay_rgba = np.array(result.overlay.convert("RGBA"))

            payload = {
                "heatmap": result.heat01,
                "box": result.box_xyxy,
                "overlay_rgba": overlay_rgba,
                "model": self._request.model_name,
                "click_xy": self._request.click_xy,
                "image_size": (
                    self._request.image.width,
                    self._request.image.height,
                ),
            }
            self.finished.emit(payload)
        except Exception as exc:  # pragma: no cover - GUI surface
            self.error.emit(str(exc))


class DinoPatchSimilarityService(QtCore.QObject):
    """Facade used by the GUI to request patch-similarity overlays."""

    started = QtCore.Signal()
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_DinoPatchWorker] = None

    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def request(self, request: DinoPatchRequest) -> bool:
        """Queue a new similarity computation.

        Returns ``True`` if the job was accepted. When ``False`` the caller
        should refrain from submitting additional requests (e.g. wait until the
        current job completes).
        """

        if self.is_busy():
            return False

        self._thread = QtCore.QThread()
        self._worker = _DinoPatchWorker(request)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._handle_finished)
        self._worker.error.connect(self._handle_error)
        self._worker.finished.connect(self._cleanup)
        self._worker.error.connect(self._cleanup)
        self._thread.start()
        self.started.emit()
        return True

    @QtCore.Slot(dict)
    def _handle_finished(self, payload: dict) -> None:
        self.finished.emit(payload)

    @QtCore.Slot(str)
    def _handle_error(self, message: str) -> None:
        self.error.emit(message)

    def _cleanup(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None


@dataclass
class DinoPCARequest:
    image: Image.Image
    model_name: str
    short_side: int = 768
    device: Optional[str] = None
    output_size: Literal["input", "resized", "feature"] = "input"
    components: int = 3
    clip_percentile: Optional[float] = 1.0
    alpha: float = 0.65
    normalize: bool = True
    mask: Optional[np.ndarray] = None


class _DinoPCAWorker(QtCore.QObject):
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, request: DinoPCARequest) -> None:
        super().__init__()
        self._request = request

    @QtCore.Slot()
    def run(self) -> None:
        try:
            extractor = _get_or_create_extractor(
                self._request.model_name,
                self._request.short_side,
                self._request.device,
            )
            mapper = Dinov3PCAMapper(
                extractor,
                num_components=self._request.components,
                clip_percentile=self._request.clip_percentile,
            )
            result = mapper.map_image(
                self._request.image,
                output_size=self._request.output_size,
                return_type="array",
                normalize_features=self._request.normalize,
                mask=self._request.mask,
            )

            rgb = np.clip(result.output_rgb, 0.0, 1.0)
            rgb_uint8 = (rgb * 255.0).astype(np.uint8)
            alpha_val = int(np.clip(self._request.alpha * 255.0, 0, 255))
            overlay_rgba = np.dstack(
                [rgb_uint8, np.full(rgb_uint8.shape[:2],
                                    alpha_val, dtype=np.uint8)]
            )

            if self._request.mask is not None:
                mask = self._request.mask.astype(bool)
                if mask.shape != overlay_rgba.shape[:2]:
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask_img = mask_img.resize(
                        (overlay_rgba.shape[1], overlay_rgba.shape[0]),
                        resample=Image.NEAREST,
                    )
                    mask = np.array(mask_img) > 0
                overlay_rgba[..., :3][~mask] = 0
                overlay_rgba[..., 3][~mask] = 0

            payload = {
                "overlay_rgba": overlay_rgba,
                "model": self._request.model_name,
                "image_size": (
                    self._request.image.width,
                    self._request.image.height,
                ),
                "components": self._request.components,
            }
            self.finished.emit(payload)
        except Exception as exc:  # pragma: no cover - GUI surface
            self.error.emit(str(exc))


class DinoPCAMapService(QtCore.QObject):
    started = QtCore.Signal()
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_DinoPCAWorker] = None

    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def request(self, request: DinoPCARequest) -> bool:
        if self.is_busy():
            return False

        self._thread = QtCore.QThread()
        self._worker = _DinoPCAWorker(request)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._handle_finished)
        self._worker.error.connect(self._handle_error)
        self._worker.finished.connect(self._cleanup)
        self._worker.error.connect(self._cleanup)
        self._thread.start()
        self.started.emit()
        return True

    @QtCore.Slot(dict)
    def _handle_finished(self, payload: dict) -> None:
        self.finished.emit(payload)

    @QtCore.Slot(str)
    def _handle_error(self, message: str) -> None:
        self.error.emit(message)

    def _cleanup(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None
