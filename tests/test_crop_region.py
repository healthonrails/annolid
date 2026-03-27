from __future__ import annotations

from qtpy.QtCore import QRectF

from annolid.gui.widgets.crop_region import CropRegion


def test_crop_region_clamps_to_bounds_and_rounds() -> None:
    region = CropRegion.from_qrectf(
        QRectF(-10.2, -4.9, 50.4, 30.2),
        bounds=QRectF(0, 0, 200, 120),
    )

    assert region is not None
    assert region.as_tuple() == (0, 0, 40, 25)


def test_crop_region_rejects_invalid_size() -> None:
    assert CropRegion.from_values(1, 2, 0, 5) is None
    assert CropRegion.from_values(1, 2, 5, 0) is None
    assert CropRegion.from_qrectf(None) is None
