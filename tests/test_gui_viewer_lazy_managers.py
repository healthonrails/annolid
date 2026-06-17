from __future__ import annotations

import sys

from annolid.gui.features import viewers


def test_lazy_sam_predicates_do_not_import_manager_modules() -> None:
    sys.modules.pop("annolid.gui.widgets.sam2_manager", None)
    sys.modules.pop("annolid.gui.widgets.sam3_manager", None)

    sam2_manager = viewers._LazySam2Manager(object())
    sam3_manager = viewers._LazySam3Manager(object())

    assert sam2_manager.is_sam2_model("sam2_hiera_s", "")
    assert sam3_manager.is_sam3_model("SAM3", "")
    assert "annolid.gui.widgets.sam2_manager" not in sys.modules
    assert "annolid.gui.widgets.sam3_manager" not in sys.modules


def test_lazy_sam3_close_session_noops_before_resolution() -> None:
    manager = viewers._LazySam3Manager(object())

    assert manager.sam3_session is None
    assert manager.close_session() is None
