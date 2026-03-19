from __future__ import annotations
from typing import Any

VTK_IMPORT_ERROR: Exception | None = None
HAS_PYVISTA = False
HAS_VTK = False

# Type hints for common VTK symbols to aid IDEs and static analysis.
QVTKRenderWindowInteractor: Any = None
vtkRenderer: Any = None
vtkVolume: Any = None
vtkVolumeProperty: Any = None
vtkWindowToImageFilter: Any = None
vtkTexture: Any = None
vtkPointPicker: Any = None
vtkColorTransferFunction: Any = None
vtkPolyDataMapper: Any = None
vtkActor: Any = None
vtkProperty: Any = None
vtkImageActor: Any = None
vtkLight: Any = None
vtkGlyph3DMapper: Any = None
vtkSmartVolumeMapper: Any = None
vtkImageData: Any = None
vtkPolyData: Any = None
vtkCellArray: Any = None
vtkPiecewiseFunction: Any = None
vtkPlane: Any = None
vtkPoints: Any = None
vtkStringArray: Any = None
vtkInteractorStyleTrackballCamera: Any = None
vtkInteractorStyleUser: Any = None
numpy_to_vtk: Any = None
vtk_to_numpy: Any = None
get_vtk_array_type: Any = None
vtkPNGWriter: Any = None
vtkImageReader2Factory: Any = None
vtkSphereSource: Any = None
vtkOpenGLPolyDataMapper: Any = None
vtkGDCMImageReader: Any = None
vtkImagePlaneWidget: Any = None
vtkPLYReader: Any = None
vtkSTLReader: Any = None
vtkOBJReader: Any = None
vtkNIFTIImageReader: Any = None
vtkAnalyzeReader: Any = None
vtkDICOMImageReader: Any = None


def _resolve_vtk_namespace(pv: Any) -> Any:
    """Safely locate the VTK namespace within PyVista."""
    for attr in ("_vtk", "plotting._vtk", "core._vtk_core"):
        try:
            parts = attr.split(".")
            target = pv
            for part in parts:
                target = getattr(target, part)
            if target is not None:
                return target
        except (AttributeError, ImportError):
            continue
    return None


try:
    import pyvista as pv  # type: ignore

    HAS_PYVISTA = True
except Exception as exc:
    VTK_IMPORT_ERROR = exc
    pv = None

if HAS_PYVISTA and pv is not None:
    try:
        from pyvistaqt import QtInteractor as QVTKRenderWindowInteractor  # type: ignore
    except Exception:
        QVTKRenderWindowInteractor = None

    vtk_ns = _resolve_vtk_namespace(pv)
    if vtk_ns is not None:
        # Resolve standard VTK classes
        symbols = [
            "vtkRenderer",
            "vtkVolume",
            "vtkVolumeProperty",
            "vtkWindowToImageFilter",
            "vtkTexture",
            "vtkPointPicker",
            "vtkColorTransferFunction",
            "vtkPolyDataMapper",
            "vtkActor",
            "vtkProperty",
            "vtkImageActor",
            "vtkLight",
            "vtkGlyph3DMapper",
            "vtkSmartVolumeMapper",
            "vtkImageData",
            "vtkPolyData",
            "vtkCellArray",
            "vtkPiecewiseFunction",
            "vtkPlane",
            "vtkPoints",
            "vtkStringArray",
            "vtkInteractorStyleTrackballCamera",
            "vtkInteractorStyleUser",
            "vtkPNGWriter",
            "vtkImageReader2Factory",
            "vtkSphereSource",
            "vtkOpenGLPolyDataMapper",
            "vtkGDCMImageReader",
            "vtkImagePlaneWidget",
            "vtkPLYReader",
            "vtkSTLReader",
            "vtkOBJReader",
            "vtkNIFTIImageReader",
            "vtkAnalyzeReader",
            "vtkDICOMImageReader",
        ]

        g = globals()
        for sym in symbols:
            g[sym] = getattr(vtk_ns, sym, None)

        try:
            from vtk.util.numpy_support import (  # type: ignore
                get_vtk_array_type,
                numpy_to_vtk,
                vtk_to_numpy,
            )
        except Exception as exc:
            VTK_IMPORT_ERROR = VTK_IMPORT_ERROR or exc
            get_vtk_array_type = None
            numpy_to_vtk = None
            vtk_to_numpy = None

        HAS_VTK = all(
            [
                vtkRenderer is not None,
                vtkVolume is not None,
                vtkVolumeProperty is not None,
                vtkImageData is not None,
                numpy_to_vtk is not None,
                vtk_to_numpy is not None,
            ]
        )

HAS_GDCM = vtkGDCMImageReader is not None
HAS_PLANE_WIDGET = vtkImagePlaneWidget is not None


def is_runtime_available() -> bool:
    """Check if the minimum required VTK/PyVista runtime is available."""
    return HAS_VTK
