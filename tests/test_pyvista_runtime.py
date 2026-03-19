from __future__ import annotations

from annolid.gui.widgets import pyvista_runtime as pv_rt


def test_pyvista_runtime_exports_backend_capability_flags() -> None:
    assert isinstance(pv_rt.HAS_VTK, bool)
    assert isinstance(pv_rt.HAS_GDCM, bool)
    assert isinstance(pv_rt.HAS_PLANE_WIDGET, bool)


def test_pyvista_runtime_exports_core_symbols() -> None:
    assert hasattr(pv_rt, "QVTKRenderWindowInteractor")
    assert hasattr(pv_rt, "vtkRenderer")
    assert hasattr(pv_rt, "vtkImageData")
