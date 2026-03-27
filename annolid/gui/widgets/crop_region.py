from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QRectF


@dataclass(frozen=True)
class CropRegion:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_values(cls, x: int, y: int, width: int, height: int) -> CropRegion | None:
        if width <= 0 or height <= 0:
            return None
        return cls(int(x), int(y), int(width), int(height))

    @classmethod
    def from_qrectf(
        cls, rect: QRectF | None, bounds: QRectF | None = None
    ) -> CropRegion | None:
        if rect is None:
            return None

        clipped = QRectF(rect).normalized()
        if bounds is not None:
            clipped = clipped.intersected(bounds)

        x = int(round(clipped.x()))
        y = int(round(clipped.y()))
        width = int(round(clipped.width()))
        height = int(round(clipped.height()))
        if width <= 0 or height <= 0:
            return None

        if bounds is not None:
            max_width = int(round(bounds.width()))
            max_height = int(round(bounds.height()))
            width = min(width, max(0, max_width - x))
            height = min(height, max(0, max_height - y))
            if width <= 0 or height <= 0:
                return None

        return cls(x, y, width, height)

    def as_qrectf(self) -> QRectF:
        return QRectF(
            float(self.x), float(self.y), float(self.width), float(self.height)
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height
