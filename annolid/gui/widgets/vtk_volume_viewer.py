from __future__ import annotations
import os
import re
import tempfile
import json
import types
import zlib
from collections import OrderedDict
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple
from functools import partial

from annolid.utils.logger import logger

import numpy as np
import pandas as pd
from PIL import Image
from qtpy import QtCore, QtWidgets, QtGui

# VTK imports (modular) — if these fail, the caller should fall back
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
    vtkTexture,
    vtkPointPicker,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkActor,
    vtkProperty,
    vtkImageActor,
    vtkLight,
    vtkGlyph3DMapper,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkCommonDataModel import (
    vtkImageData,
    vtkPolyData,
    vtkCellArray,
    vtkPiecewiseFunction,
    vtkPlane,
)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonCore import vtkStringArray
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleUser,
)
from vtkmodules.util.numpy_support import numpy_to_vtk, get_vtk_array_type
from vtkmodules.vtkIOImage import vtkPNGWriter, vtkImageReader2Factory
from vtkmodules.vtkFiltersSources import vtkSphereSource

try:
    from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLPolyDataMapper
except Exception:  # pragma: no cover - optional renderer
    vtkOpenGLPolyDataMapper = None
try:  # Prefer GDCM when available (more robust series handling)
    from vtkmodules.vtkIOImage import vtkGDCMImageReader  # type: ignore

    _HAS_GDCM = True
except Exception:  # pragma: no cover
    _HAS_GDCM = False
try:
    from vtkmodules.vtkInteractionWidgets import vtkImagePlaneWidget
except Exception:
    try:
        from vtkmodules.vtkInteractionImage import vtkImagePlaneWidget
    except Exception:  # pragma: no cover
        vtkImagePlaneWidget = None
_HAS_PLANE_WIDGET = vtkImagePlaneWidget is not None


POINT_CLOUD_EXTS = (".ply", ".csv", ".xyz")
MESH_EXTS = (".stl", ".obj")
VOLUME_FILE_EXTS = (
    ".tif",
    ".tiff",
    ".nii",
    ".nii.gz",
    ".zarr",
    ".zarr.json",
    ".zgroup",
)
DICOM_EXTS = (".dcm", ".dicom", ".ima")
VOLUME_SOURCE_FILTERS = (
    "3D sources (*.tif *.tiff *.ome.tif *.ome.tiff "
    "*.nii *.nii.gz *.dcm *.dicom *.ima *.IMA *.zarr *.zarr.json *.zgroup);;All files (*.*)"
)
TIFF_SUFFIXES = (".tif", ".tiff")
OME_TIFF_SUFFIXES = (".ome.tif", ".ome.tiff")
_auto_ooc_raw = os.environ.get("ANNOLID_VTK_OUT_OF_CORE_THRESHOLD_MB", "2048") or ""
try:
    AUTO_OUT_OF_CORE_MB = float(_auto_ooc_raw.strip()) if _auto_ooc_raw else 0.0
except Exception:
    AUTO_OUT_OF_CORE_MB = 0.0
_max_voxels_raw = os.environ.get("ANNOLID_VTK_MAX_VOLUME_VOXELS", "134217728") or ""
try:
    MAX_VOLUME_VOXELS = int(float(_max_voxels_raw.strip()))
except Exception:
    MAX_VOLUME_VOXELS = 134217728
_slice_mode_bytes_raw = (
    os.environ.get("ANNOLID_VTK_SLICE_MODE_BYTES", "68719476736") or ""
)
try:
    SLICE_MODE_BYTES = float(_slice_mode_bytes_raw.strip())
except Exception:
    SLICE_MODE_BYTES = 68719476736.0

PLANE_DEFS = (
    (0, "Axial (Z)", "SetPlaneOrientationToZAxes"),
    (1, "Coronal (Y)", "SetPlaneOrientationToYAxes"),
    (2, "Sagittal (X)", "SetPlaneOrientationToXAxes"),
)
PLANE_COLORS: dict[int, tuple[float, float, float]] = {
    0: (0.9, 0.2, 0.2),
    1: (0.2, 0.9, 0.2),
    2: (0.2, 0.2, 0.9),
}

# Compact palette inspired by Allen Mouse Brain Atlas colors; repeats cyclically.
ALLEN_MOUSE_ATLAS_COLORS = (
    "#b22222",
    "#e25822",
    "#f3a530",
    "#ffd700",
    "#a8c81c",
    "#5cab1d",
    "#00a4a6",
    "#1e90ff",
    "#4169e1",
    "#6a5acd",
    "#8a2be2",
    "#ba55d3",
    "#c71585",
    "#ff69b4",
    "#cd5c5c",
    "#a0522d",
    "#d2691e",
    "#8b4513",
    "#2f4f4f",
    "#708090",
    "#556b2f",
    "#6b8e23",
    "#4682b4",
    "#5f9ea0",
    "#b0c4de",
    "#add8e6",
    "#ffb6c1",
    "#ffe4b5",
    "#98fb98",
    "#7fffd4",
    "#afeeee",
    "#dda0dd",
    "#f0e68c",
)


@dataclass
class _RegionSelectionEntry:
    item: QtWidgets.QListWidgetItem
    checkbox: QtWidgets.QCheckBox
    color_button: QtWidgets.QToolButton
    display_text: str


@dataclass
class _SlicePlaneControl:
    axis: int
    name: str
    slider: QtWidgets.QSlider
    checkbox: QtWidgets.QCheckBox
    label: QtWidgets.QLabel


@dataclass
class _VolumeData:
    array: Optional[np.ndarray]
    spacing: Optional[Tuple[float, float, float]]
    vmin: float
    vmax: float
    is_grayscale: bool = True
    is_out_of_core: bool = False
    backing_path: Optional[Path] = None
    vtk_image: Optional[vtkImageData] = None  # type: ignore[name-defined]
    slice_mode: bool = False
    slice_loader: Optional["_BaseSliceLoader"] = None
    slice_axis: int = 0
    volume_shape: Optional[tuple[int, int, int]] = None
    is_label_map: bool = False


@dataclass
class _OverlayVolumeEntry:
    path: Path
    actor: vtkVolume
    mapper: vtkSmartVolumeMapper
    property: vtkVolumeProperty
    label: str
    visible: bool = True


class _BaseSliceLoader:
    """Abstract loader that can retrieve 2D slices on demand."""

    def total_slices(self) -> int:
        raise NotImplementedError

    def shape(self) -> tuple[int, int, int]:
        raise NotImplementedError

    def dtype(self) -> np.dtype:
        raise NotImplementedError

    def read_slice(self, axis: int, index: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return


class _MemmapSliceLoader(_BaseSliceLoader):
    def __init__(self, array: np.ndarray):
        self._array = array
        if self._array.ndim == 2:
            self._array = self._array[np.newaxis, ...]

    def total_slices(self) -> int:
        return int(self._array.shape[0])

    def shape(self) -> tuple[int, int, int]:
        shp = self._array.shape
        if len(shp) == 2:
            return (1, int(shp[0]), int(shp[1]))
        return (int(shp[0]), int(shp[1]), int(shp[2]))

    def dtype(self) -> np.dtype:
        return self._array.dtype

    def read_slice(self, axis: int, index: int) -> np.ndarray:
        if axis != 0:
            raise NotImplementedError(
                "Memmap loader currently supports axial slices only."
            )
        index = max(0, min(int(index), self.total_slices() - 1))
        return np.array(self._array[index], copy=True)

    def close(self) -> None:
        self._array = None  # type: ignore[assignment]


class _TiffSliceLoader(_BaseSliceLoader):
    def __init__(self, path: Path, shape: tuple[int, int, int], dtype: np.dtype):
        import tifffile  # local import to avoid hard dependency at module load

        self._path = str(path)
        self._tif = tifffile.TiffFile(self._path)
        self._shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        self._dtype = np.dtype(dtype)

    def total_slices(self) -> int:
        return self._shape[0]

    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def dtype(self) -> np.dtype:
        return self._dtype

    def read_slice(self, axis: int, index: int) -> np.ndarray:
        if axis != 0:
            raise NotImplementedError(
                "TIFF slice loader currently supports axial slices only."
            )
        idx = max(0, min(int(index), self.total_slices() - 1))
        return self._tif.pages[idx].asarray()

    def close(self) -> None:
        try:
            self._tif.close()
        except Exception:
            pass


class _ZarrSliceLoader(_BaseSliceLoader):
    def __init__(
        self,
        zarr_array,
        zyx_axes: tuple[int, int, int],
        fixed_indices: Optional[dict[int, int]] = None,
    ):
        self._arr = zarr_array
        self._zyx_axes = zyx_axes
        self._fixed_indices = fixed_indices or {}

        # Cache shape and dtype to avoid repeated attribute access on slow backends
        full_shape = getattr(zarr_array, "shape", ())
        self._full_shape = full_shape

        # The visible 3D shape
        self._shape = (
            int(full_shape[zyx_axes[0]]),
            int(full_shape[zyx_axes[1]]),
            int(full_shape[zyx_axes[2]]),
        )
        self._dtype = np.dtype(getattr(zarr_array, "dtype", np.float32))

    def total_slices(self) -> int:
        return self._shape[0]

    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def dtype(self) -> np.dtype:
        return self._dtype

    def read_slice(self, axis: int, index: int) -> np.ndarray:
        # Clamp index
        idx = max(0, min(int(index), self.total_slices() - 1))

        # Build slicer for N-dimensions
        slicer: list[object] = [0] * len(self._full_shape)

        # 1. Apply fixed indices (e.g. Time=0, Channel=0)
        for dim, val in self._fixed_indices.items():
            slicer[dim] = val

        # 2. Apply the active ZYX mapping
        # For the slice mode, we are usually iterating over the Z axis (zyx_axes[0])
        # and fetching the whole YX plane.
        z_dim, y_dim, x_dim = self._zyx_axes

        slicer[z_dim] = idx  # The slice index
        slicer[y_dim] = slice(None)  # Keep full Y
        slicer[x_dim] = slice(None)  # Keep full X

        # 3. Retrieve data (this triggers the specific chunk read)
        try:
            arr = np.array(self._arr[tuple(slicer)])
        except Exception as e:
            logger.error(f"Zarr read failed at slice {idx}: {e}")
            return np.zeros((self._shape[1], self._shape[2]), dtype=self._dtype)

        # 4. Ensure 2D output
        # If the slicer leaves extra dimensions of size 1 (e.g. from slice(None)), squeeze them.
        # But we must preserve exactly 2 dimensions.
        if arr.ndim > 2:
            arr = arr.squeeze()

        # If squeezing reduced it too much (e.g. 1x1 pixel), reshape back
        if arr.ndim < 2:
            arr = arr.reshape(self._shape[1], self._shape[2])

        return arr

    def close(self) -> None:
        self._arr = None


class _ZarrV3Array:
    """Lightweight Zarr v3 reader for directory stores with default chunk keys."""

    def __init__(self, array_path: Path, metadata: Mapping[str, object]):
        self._path = Path(array_path)
        self.attrs = dict(metadata.get("attributes", {}) or {})
        shape_raw = metadata.get("shape", [])
        if not shape_raw:
            raise ValueError("Zarr metadata is missing shape.")
        self.shape = tuple(int(x) for x in shape_raw)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(metadata.get("data_type", "float32"))
        chunk_conf = (
            metadata.get("chunk_grid", {})
            .get("configuration", {})
            .get("chunk_shape", [])
        )
        if not chunk_conf:
            self._chunk_shape = tuple(1 for _ in self.shape)
        else:
            self._chunk_shape = tuple(int(x) for x in chunk_conf)
        self._fill_value = metadata.get("fill_value", 0)
        encoding_conf = metadata.get("chunk_key_encoding", {}).get("configuration", {})
        self._chunk_separator = encoding_conf.get("separator", "/") or "/"
        self._codecs = list(metadata.get("codecs", []) or [])
        self._chunk_cache: OrderedDict[tuple[int, ...], np.ndarray] = OrderedDict()
        # Cache a modest number of chunks to speed repeated slice access.
        self._cache_limit = 32
        self._chunk_root = self._path / "c"

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))
        if len(key) != self.ndim:
            raise IndexError(
                f"Expected {self.ndim} indices for Zarr array, got {len(key)}"
            )

        norm_slices: list[slice] = []
        squeeze_axes: list[int] = []
        for axis, k in enumerate(key):
            if isinstance(k, slice):
                start, stop, step = k.indices(self.shape[axis])
            elif isinstance(k, (int, np.integer)):
                idx = int(k)
                if idx < 0:
                    idx += self.shape[axis]
                start, stop, step = idx, idx + 1, 1
                squeeze_axes.append(axis)
            else:
                raise TypeError(f"Unsupported index type for Zarr data: {type(k)}")
            if step not in (None, 1):
                raise ValueError("Zarr reader currently supports step=1 slices only.")
            norm_slices.append(slice(start, stop, 1))

        out_shape = [sl.stop - sl.start for sl in norm_slices]
        if any(dim <= 0 for dim in out_shape):
            return np.empty(tuple(max(dim, 0) for dim in out_shape), dtype=self.dtype)
        result = np.full(out_shape, self._fill_value, dtype=self.dtype)

        chunk_ranges = []
        for axis, sl in enumerate(norm_slices):
            start_chunk = sl.start // self._chunk_shape[axis]
            end_chunk = (sl.stop - 1) // self._chunk_shape[axis]
            chunk_ranges.append(range(start_chunk, end_chunk + 1))

        for chunk_idx in itertools.product(*chunk_ranges):
            chunk_start = [
                int(chunk_idx[axis]) * self._chunk_shape[axis]
                for axis in range(self.ndim)
            ]
            chunk_shape = [
                min(self._chunk_shape[axis], self.shape[axis] - chunk_start[axis])
                for axis in range(self.ndim)
            ]
            chunk_arr = self._read_chunk(tuple(int(c) for c in chunk_idx), chunk_shape)
            if chunk_arr is None:
                continue

            chunk_slices: list[slice] = []
            out_slices: list[slice] = []
            skip_chunk = False
            for axis, sl in enumerate(norm_slices):
                local_start = max(sl.start - chunk_start[axis], 0)
                local_end = min(sl.stop - chunk_start[axis], chunk_shape[axis])
                if local_end <= local_start:
                    skip_chunk = True
                    break
                chunk_slices.append(slice(local_start, local_end))
                out_start = max(chunk_start[axis], sl.start) - sl.start
                out_end = out_start + (local_end - local_start)
                out_slices.append(slice(out_start, out_end))
            if skip_chunk:
                continue
            try:
                result[tuple(out_slices)] = chunk_arr[tuple(chunk_slices)]
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to map Zarr chunk %s: %s", chunk_idx, exc)

        if squeeze_axes:
            return np.squeeze(result, axis=tuple(squeeze_axes))
        return result

    def _chunk_path(self, chunk_idx: tuple[int, ...]) -> Path:
        parts = ["c"]
        parts.extend(str(int(c)) for c in chunk_idx)
        # Separator is kept for compatibility, but Path join covers typical layouts.
        return self._path.joinpath(*parts)

    def _read_chunk(
        self, chunk_idx: tuple[int, ...], expected_shape: Sequence[int]
    ) -> Optional[np.ndarray]:
        cached = self._chunk_cache.get(chunk_idx)
        if cached is not None:
            self._chunk_cache.move_to_end(chunk_idx)
            return cached

        chunk_path = self._chunk_path(chunk_idx)
        if not chunk_path.exists():
            arr = np.full(expected_shape, self._fill_value, dtype=self.dtype)
            self._store_chunk(chunk_idx, arr)
            return arr

        try:
            with open(chunk_path, "rb") as f:
                raw = f.read()
            arr = self._decode_chunk_bytes(raw, tuple(int(x) for x in expected_shape))
        except Exception as exc:
            logger.error("Failed to read Zarr chunk %s: %s", chunk_path, exc)
            arr = np.full(expected_shape, self._fill_value, dtype=self.dtype)
        self._store_chunk(chunk_idx, arr)
        return arr

    def _decode_chunk_bytes(
        self, raw: bytes, expected_shape: tuple[int, ...]
    ) -> np.ndarray:
        expected_bytes = int(np.prod(expected_shape)) * max(1, self.dtype.itemsize)
        data: object = raw
        last_error: Optional[Exception] = None
        for codec in reversed(self._codecs):
            name = codec.get("name") if isinstance(codec, Mapping) else None
            conf = codec.get("configuration", {}) if isinstance(codec, Mapping) else {}
            try:
                if name == "zstd":
                    data = self._decode_zstd(data, expected_bytes)
                elif name in ("gzip", "zlib"):
                    data = zlib.decompress(data)  # type: ignore[arg-type]
                elif name == "bytes":
                    endian = conf.get("endian", "<")
                    dt = self.dtype.newbyteorder(
                        "<" if endian in ("little", "<") else ">"
                    )
                    data = np.frombuffer(data, dtype=dt)
                else:
                    raise RuntimeError(f"Unsupported Zarr codec: {name}")
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                break

        if last_error is not None:
            raise RuntimeError(
                f"Zarr codec pipeline failed: {last_error}"
            ) from last_error

        if isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, dtype=self.dtype)
        else:
            arr = np.asarray(data, dtype=self.dtype)

        expected_elems = int(np.prod(expected_shape))
        full_chunk_elems = int(np.prod(self._chunk_shape))
        try:
            if arr.size == expected_elems:
                arr = arr.reshape(expected_shape)
            elif arr.size == full_chunk_elems:
                arr = arr.reshape(self._chunk_shape)
                crop = tuple(
                    slice(0, min(expected_shape[i], arr.shape[i]))
                    for i in range(len(expected_shape))
                )
                arr = arr[crop]
            else:
                arr = np.resize(arr, expected_shape)
        except Exception:  # pragma: no cover - defensive reshape
            arr = np.resize(arr, expected_shape)
        return arr

    def _decode_zstd(self, data: object, expected_bytes: int) -> bytes:
        """Decode Zstd with fallbacks; never throws unless all attempts fail."""
        last_error: Optional[Exception] = None
        max_out = max(expected_bytes * 2, expected_bytes + 1)
        try:
            import zstandard as zstd  # type: ignore

            dctx = zstd.ZstdDecompressor()
            # type: ignore[arg-type]
            return dctx.decompress(data, max_output_size=max_out)
        except Exception as exc:  # pragma: no cover - optional dependency
            last_error = exc

        try:
            import numcodecs  # type: ignore

            decoder = None
            try:
                from numcodecs.zstd import Zstd  # type: ignore

                decoder = Zstd()
            except Exception:
                try:
                    # type: ignore[attr-defined]
                    decoder = numcodecs.get_codec({"id": "zstd"})
                except Exception:
                    decoder = None
            if decoder is not None:
                return decoder.decode(data)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - optional dependency
            last_error = last_error or exc

        raise RuntimeError(f"zstd codec failed: {last_error}")

    def _store_chunk(self, key: tuple[int, ...], arr: np.ndarray) -> None:
        self._chunk_cache[key] = arr
        if len(self._chunk_cache) > self._cache_limit:
            self._chunk_cache.popitem(last=False)

    def first_nonempty_index(self, axis: int = 0) -> int:
        """Best-effort estimate of the first non-empty chunk along an axis."""
        chunk_size = self._chunk_shape[axis] if axis < len(self._chunk_shape) else 1
        if axis != 0:
            return 0
        try:
            for zdir in sorted(self._chunk_root.iterdir(), key=lambda p: int(p.name)):
                if not zdir.is_dir():
                    continue
                try:
                    for ydir in zdir.iterdir():
                        if not ydir.is_dir():
                            continue
                        for xfile in ydir.iterdir():
                            if xfile.is_file():
                                return max(0, int(zdir.name) * chunk_size)
                except Exception:
                    continue
        except Exception:
            return 0
        return 0


class VTKVolumeViewerDialog(QtWidgets.QMainWindow):
    _PLY_DTYPE_MAP: Mapping[str, str] = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "i2",
        "ushort": "u2",
        "int16": "i2",
        "uint16": "u2",
        "int": "i4",
        "uint": "u4",
        "int32": "i4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    """
    True 3D volume renderer using VTK's GPU volume mapper.

    - Loads a TIFF stack into a 3D volume
    - Interact with mouse: rotate, zoom, pan
    - Simple UI controls for opacity scaling and shading toggle
    """

    def __init__(
        self, src_path: Optional[str | Path], parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("3D Volume Renderer (VTK)")
        self.resize(1150, 820)
        self._source_path: Optional[Path] = None
        self._path = Path(".")
        if src_path:
            try:
                candidate = Path(src_path).expanduser()
            except Exception:
                candidate = Path(src_path)
            try:
                candidate = candidate.resolve()
            except Exception:
                pass
            self._source_path = candidate
            self._path = self._resolve_initial_source(candidate)

        # Volume state placeholders
        self._has_volume: bool = False
        self._volume_np: Optional[np.ndarray] = None
        self._vtk_img = None
        self._vmin = 0.0
        self._vmax = 1.0
        self._opacity_tf = None
        self._color_tf = None
        self._volume_shape: tuple[int, int, int] = (0, 0, 0)
        self._slice_plane_widgets: dict[int, vtkImagePlaneWidget] = {}
        self._slice_clipping_planes: dict[int, vtkPlane] = {}
        self._out_of_core_active = False
        self._out_of_core_backing_path: Optional[Path] = None
        self._out_of_core_array: Optional[np.memmap] = None
        self._volume_visible = True
        self._point_cloud_visible = True
        self._slice_mode = False
        self._slice_loader: Optional[_BaseSliceLoader] = None
        self._slice_actor: Optional[vtkImageActor] = None
        self._slice_current_index = 0
        self._slice_axis = 0
        self._slice_vmin = 0.0
        self._slice_vmax = 1.0
        self._slice_gamma = 1.0
        self._slice_window_override = False
        self._slice_last_data_min = 0.0
        self._slice_last_data_max = 1.0
        self._overlay_volumes: list[_OverlayVolumeEntry] = []
        self._overlay_list_updating = False
        self._slice_window_override = False
        self._slice_last_data_min = 0.0
        self._slice_last_data_max = 1.0
        self._label_volume_active = False
        self._slice_start_index_hint: Optional[int] = None
        self._ply_color_cache: dict[str, np.ndarray] = {}
        # Cache gaussian PLY vertex fields keyed by file path to avoid reparsing.
        self._gaussian_field_cache: dict[str, dict[str, object]] = {}
        self._gaussian_actor_data: dict[int, dict[str, object]] = {}
        self._gaussian_scale_mult: float = 1.0
        self._gaussian_opacity_mult: float = 1.0
        # Use a higher default glyph resolution for smoother gaussian ellipsoids.
        self._gaussian_glyph_res: int = 24
        self._scene_light: Optional[vtkLight] = None
        self._light_intensity: float = 1.0

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        central_layout = QtWidgets.QVBoxLayout(central_widget)
        central_layout.setContentsMargins(2, 2, 2, 2)
        central_layout.setSpacing(6)
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        central_layout.addWidget(self.vtk_widget, 1)

        # Controls panel (dockable)
        controls_panel = QtWidgets.QWidget()
        controls_panel.setMinimumWidth(180)
        controls_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setAlignment(QtCore.Qt.AlignTop)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        self._controls_dock = QtWidgets.QDockWidget("Controls", self)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(QtWidgets.QFrame.NoFrame)
        scroll_area.setWidget(controls_panel)
        self._controls_dock.setWidget(scroll_area)
        self._controls_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self._controls_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._controls_dock)
        self._apply_default_dock_width()

        # Quick access strip
        quick_group = QtWidgets.QGroupBox("Quick Actions")
        quick_layout = QtWidgets.QVBoxLayout()
        quick_layout.setContentsMargins(6, 6, 6, 6)
        quick_grid = QtWidgets.QGridLayout()
        quick_grid.setSpacing(6)
        buttons = [
            self._create_quick_button(
                "Load Volume",
                self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon),
                self._load_volume_dialog,
                "Open a 3D volume, DICOM folder, OME-TIFF, or Zarr store.",
            ),
            self._create_quick_button(
                "Reload Volume",
                self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload),
                self._reload_volume,
                "Re-read the active volume from disk (useful after toggling out-of-core).",
            ),
            self._create_quick_button(
                "Load Points",
                self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon),
                self._load_point_cloud_dialog,
                "Load CSV/PLY/XYZ point clouds.",
            ),
            self._create_quick_button(
                "Load Mesh",
                self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogNewFolder),
                self._load_mesh_dialog,
                "Load STL/OBJ/PLY meshes.",
            ),
            self._create_quick_button(
                "Snapshot",
                self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton),
                self._save_snapshot,
                "Save the current viewport as a PNG.",
            ),
            self._create_quick_button(
                "Reset View",
                self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload),
                self._reset_camera,
                "Reset camera to fit all visible data.",
            ),
            self._create_quick_button(
                "Help",
                self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxInformation),
                self._show_help_overlay,
                "Show navigation tips and shortcuts.",
            ),
        ]
        for idx, btn in enumerate(buttons):
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
            )
            btn.setMinimumHeight(28)
            r, c = divmod(idx, 3)
            quick_grid.addWidget(btn, r, c)
        for c in range(3):
            quick_grid.setColumnStretch(c, 1)
        quick_layout.addLayout(quick_grid)
        quick_hint = QtWidgets.QLabel(
            "Tip: press W to toggle window/level mode, R to reset camera, +/- to change opacity."
        )
        quick_hint.setWordWrap(True)
        quick_hint.setStyleSheet("color: #4a4a4a;")
        quick_layout.addWidget(quick_hint)
        quick_group.setLayout(quick_layout)
        controls_layout.addWidget(quick_group)

        self.volume_group = QtWidgets.QGroupBox("Volume Controls")
        volume_layout = QtWidgets.QGridLayout()
        self.volume_group.setLayout(volume_layout)

        # Blend mode
        volume_layout.addWidget(QtWidgets.QLabel("Blend:"), 0, 0)
        self.blend_combo = QtWidgets.QComboBox()
        self.blend_combo.addItems(["Composite", "MIP-Max", "MIP-Min", "Additive"])
        self.blend_combo.currentIndexChanged.connect(self._update_blend_mode)
        volume_layout.addWidget(self.blend_combo, 0, 1)

        # Colormap
        volume_layout.addWidget(QtWidgets.QLabel("Colormap:"), 0, 2)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["Grayscale", "Invert Gray", "Hot", "Allen Atlas"])
        self.cmap_combo.currentIndexChanged.connect(self._update_transfer_functions)
        volume_layout.addWidget(self.cmap_combo, 0, 3)

        # Intensity window (min/max)
        volume_layout.addWidget(QtWidgets.QLabel("Window:"), 1, 0)
        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.min_spin, self.max_spin):
            spin.setDecimals(3)
            spin.setKeyboardTracking(False)
        self.min_spin.valueChanged.connect(self._on_window_changed)
        self.max_spin.valueChanged.connect(self._on_window_changed)
        volume_layout.addWidget(self.min_spin, 1, 1)
        volume_layout.addWidget(self.max_spin, 1, 2)
        self.auto_window_btn = QtWidgets.QPushButton("Auto")
        self.auto_window_btn.clicked.connect(self._auto_window)
        volume_layout.addWidget(self.auto_window_btn, 1, 3)

        # Density (global opacity) and shading
        volume_layout.addWidget(QtWidgets.QLabel("Density:"), 2, 0)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(1, 100)
        self.opacity_slider.setValue(30)
        self.opacity_slider.valueChanged.connect(self._update_opacity)
        volume_layout.addWidget(self.opacity_slider, 2, 1, 1, 2)
        self.shade_checkbox = QtWidgets.QCheckBox("Shading")
        self.shade_checkbox.setChecked(True)
        self.shade_checkbox.stateChanged.connect(self._update_shading)
        volume_layout.addWidget(self.shade_checkbox, 2, 3)

        # Interpolation
        volume_layout.addWidget(QtWidgets.QLabel("Interpolation:"), 3, 0)
        self.interp_combo = QtWidgets.QComboBox()
        self.interp_combo.addItems(["Linear", "Nearest"])
        self.interp_combo.currentIndexChanged.connect(self._update_interpolation)
        volume_layout.addWidget(self.interp_combo, 3, 1)

        # Spacing (X, Y, Z)
        volume_layout.addWidget(QtWidgets.QLabel("Spacing X/Y/Z:"), 3, 2)
        self.spacing_x = QtWidgets.QDoubleSpinBox()
        self.spacing_y = QtWidgets.QDoubleSpinBox()
        self.spacing_z = QtWidgets.QDoubleSpinBox()
        for s in (self.spacing_x, self.spacing_y, self.spacing_z):
            s.setDecimals(3)
            s.setRange(0.001, 10000.0)
            s.setValue(1.0)
            s.valueChanged.connect(self._update_spacing)
        spacing_box = QtWidgets.QHBoxLayout()
        spacing_box.addWidget(self.spacing_x)
        spacing_box.addWidget(self.spacing_y)
        spacing_box.addWidget(self.spacing_z)
        spacing_widget = QtWidgets.QWidget()
        spacing_widget.setLayout(spacing_box)
        volume_layout.addWidget(spacing_widget, 3, 3)

        controls_layout.addWidget(self.volume_group)

        # Point cloud controls
        self.load_pc_btn = QtWidgets.QPushButton("Load Point Cloud…")
        self.load_pc_btn.clicked.connect(self._load_point_cloud_dialog)
        self.clear_pc_btn = QtWidgets.QPushButton("Clear Points")
        self.clear_pc_btn.clicked.connect(self._clear_point_clouds)
        self.point_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.point_size_slider.setRange(1, 12)
        self.point_size_slider.setValue(3)
        self.point_size_slider.setToolTip("Point size")
        self.point_size_slider.valueChanged.connect(self._update_point_sizes)

        self.point_group = QtWidgets.QGroupBox("Point Cloud")
        point_layout = QtWidgets.QVBoxLayout()
        point_layout.setContentsMargins(4, 4, 4, 4)
        load_row = QtWidgets.QHBoxLayout()
        load_row.addWidget(self.load_pc_btn)
        load_row.addStretch(1)
        point_layout.addLayout(load_row)

        self.point_detail_widget = QtWidgets.QWidget()
        point_detail_layout = QtWidgets.QVBoxLayout(self.point_detail_widget)
        point_detail_layout.setContentsMargins(0, 0, 0, 0)
        point_detail_layout.setSpacing(6)
        clear_row = QtWidgets.QHBoxLayout()
        clear_row.addWidget(self.clear_pc_btn)
        clear_row.addStretch(1)
        point_detail_layout.addLayout(clear_row)
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Point Size:"))
        size_row.addWidget(self.point_size_slider)
        size_row.addStretch(1)
        point_detail_layout.addLayout(size_row)

        brightness_row = QtWidgets.QHBoxLayout()
        brightness_row.addWidget(QtWidgets.QLabel("Scene Brightness:"))
        self.light_intensity_spin = QtWidgets.QDoubleSpinBox()
        self.light_intensity_spin.setRange(0.0, 3.0)
        self.light_intensity_spin.setSingleStep(0.05)
        self.light_intensity_spin.setValue(self._light_intensity)
        self.light_intensity_spin.valueChanged.connect(self._update_light_intensity)
        brightness_row.addWidget(self.light_intensity_spin)
        brightness_row.addStretch(1)
        point_detail_layout.addLayout(brightness_row)

        self.gaussian_group = QtWidgets.QGroupBox("Gaussian Splat")
        g_layout = QtWidgets.QGridLayout()
        g_layout.setContentsMargins(6, 6, 6, 6)
        g_layout.addWidget(QtWidgets.QLabel("Scale ×"), 0, 0)
        self.gaussian_scale_spin = QtWidgets.QDoubleSpinBox()
        self.gaussian_scale_spin.setRange(0.05, 5.0)
        self.gaussian_scale_spin.setSingleStep(0.05)
        self.gaussian_scale_spin.setValue(self._gaussian_scale_mult)
        self.gaussian_scale_spin.valueChanged.connect(self._update_gaussian_scale)
        g_layout.addWidget(self.gaussian_scale_spin, 0, 1)

        g_layout.addWidget(QtWidgets.QLabel("Opacity ×"), 1, 0)
        self.gaussian_opacity_spin = QtWidgets.QDoubleSpinBox()
        self.gaussian_opacity_spin.setRange(0.0, 3.0)
        self.gaussian_opacity_spin.setSingleStep(0.05)
        self.gaussian_opacity_spin.setValue(self._gaussian_opacity_mult)
        self.gaussian_opacity_spin.valueChanged.connect(self._update_gaussian_opacity)
        g_layout.addWidget(self.gaussian_opacity_spin, 1, 1)

        g_layout.addWidget(QtWidgets.QLabel("Glyph resolution"), 2, 0)
        self.gaussian_res_spin = QtWidgets.QSpinBox()
        self.gaussian_res_spin.setRange(6, 48)
        self.gaussian_res_spin.setValue(self._gaussian_glyph_res)
        self.gaussian_res_spin.valueChanged.connect(self._update_gaussian_resolution)
        g_layout.addWidget(self.gaussian_res_spin, 2, 1)
        self.gaussian_group.setLayout(g_layout)
        self.gaussian_group.setEnabled(False)
        point_detail_layout.addWidget(self.gaussian_group)

        self.region_group = QtWidgets.QGroupBox("Point Cloud Regions")
        region_layout = QtWidgets.QVBoxLayout()
        region_layout.setSpacing(4)
        region_btn_row = QtWidgets.QHBoxLayout()
        self.select_all_regions_btn = QtWidgets.QPushButton("Select All")
        self.deselect_all_regions_btn = QtWidgets.QPushButton("Deselect All")
        region_btn_row.addWidget(self.select_all_regions_btn)
        region_btn_row.addWidget(self.deselect_all_regions_btn)
        region_btn_row.addStretch(1)
        region_layout.addLayout(region_btn_row)
        self.select_all_regions_btn.clicked.connect(
            lambda: self._set_region_check_states(True)
        )
        self.deselect_all_regions_btn.clicked.connect(
            lambda: self._set_region_check_states(False)
        )
        self.region_search = QtWidgets.QLineEdit()
        self.region_search.setPlaceholderText("Filter regions…")
        self.region_search.textChanged.connect(self._filter_region_items)
        region_layout.addWidget(self.region_search)
        self.region_list_widget = QtWidgets.QListWidget()
        self.region_list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection
        )
        self.region_list_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        region_layout.addWidget(self.region_list_widget)
        self.region_group.setLayout(region_layout)
        self.region_group.setEnabled(False)
        point_detail_layout.addWidget(self.region_group)
        self.point_detail_widget.setVisible(False)
        point_layout.addWidget(self.point_detail_widget)
        self.point_group.setLayout(point_layout)
        controls_layout.addWidget(self.point_group)

        # Mesh controls
        self.load_mesh_btn = QtWidgets.QPushButton("Load Mesh…")
        self.load_mesh_btn.clicked.connect(self._load_mesh_dialog)
        self.clear_mesh_btn = QtWidgets.QPushButton("Clear Meshes")
        self.clear_mesh_btn.clicked.connect(self._clear_meshes)
        self.load_diffuse_tex_btn = QtWidgets.QPushButton("Load Diffuse Texture…")
        self.load_diffuse_tex_btn.clicked.connect(self._load_diffuse_texture)
        self.load_normal_tex_btn = QtWidgets.QPushButton("Load Normal Map…")
        self.load_normal_tex_btn.clicked.connect(self._load_normal_texture)

        self.mesh_group = QtWidgets.QGroupBox("Mesh")
        mesh_layout = QtWidgets.QVBoxLayout()
        mesh_layout.setContentsMargins(4, 4, 4, 4)
        mesh_load_row = QtWidgets.QHBoxLayout()
        mesh_load_row.addWidget(self.load_mesh_btn)
        mesh_load_row.addStretch(1)
        mesh_layout.addLayout(mesh_load_row)
        self.mesh_detail_widget = QtWidgets.QWidget()
        mesh_detail_layout = QtWidgets.QVBoxLayout(self.mesh_detail_widget)
        mesh_detail_layout.setContentsMargins(0, 0, 0, 0)
        mesh_detail_layout.setSpacing(6)
        mesh_clear_row = QtWidgets.QHBoxLayout()
        mesh_clear_row.addWidget(self.clear_mesh_btn)
        mesh_clear_row.addStretch(1)
        mesh_detail_layout.addLayout(mesh_clear_row)
        texture_group = QtWidgets.QGroupBox("Textures")
        texture_layout = QtWidgets.QHBoxLayout()
        texture_layout.setContentsMargins(4, 2, 4, 2)
        texture_layout.addWidget(self.load_diffuse_tex_btn)
        texture_layout.addWidget(self.load_normal_tex_btn)
        texture_group.setLayout(texture_layout)
        mesh_detail_layout.addWidget(texture_group)
        mesh_detail_layout.addStretch(1)
        self.mesh_detail_widget.setLayout(mesh_detail_layout)
        self.mesh_detail_widget.setVisible(False)
        mesh_layout.addWidget(self.mesh_detail_widget)
        self.mesh_group.setLayout(mesh_layout)
        controls_layout.addWidget(self.mesh_group)

        self.wl_mode_checkbox = QtWidgets.QCheckBox("Window/Level Mode")
        self.wl_mode_checkbox.setToolTip(
            "Enable to adjust intensity window by left-drag; camera interaction is paused"
        )
        self.wl_mode_checkbox.stateChanged.connect(self._toggle_wl_mode)

        self.general_group = QtWidgets.QGroupBox("General")
        general_layout = QtWidgets.QVBoxLayout()
        wl_layout = QtWidgets.QHBoxLayout()
        wl_layout.addWidget(self.wl_mode_checkbox)
        wl_layout.addStretch(1)
        general_layout.addLayout(wl_layout)
        hint_label = QtWidgets.QLabel(
            "Use Quick Actions above for load/reset/snapshot. Keep these toggles for visibility."
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #4a4a4a; font-size: 11px;")
        general_layout.addWidget(hint_label)
        visibility_layout = QtWidgets.QHBoxLayout()
        self.show_volume_checkbox = QtWidgets.QCheckBox("Show Volume")
        self.show_volume_checkbox.setChecked(True)
        self.show_volume_checkbox.stateChanged.connect(
            self._on_volume_visibility_changed
        )
        visibility_layout.addWidget(self.show_volume_checkbox)
        self.show_point_cloud_checkbox = QtWidgets.QCheckBox("Show Point Cloud")
        self.show_point_cloud_checkbox.setChecked(True)
        self.show_point_cloud_checkbox.stateChanged.connect(
            self._on_point_cloud_visibility_changed
        )
        visibility_layout.addWidget(self.show_point_cloud_checkbox)
        visibility_layout.addStretch(1)
        general_layout.addLayout(visibility_layout)
        self.general_group.setLayout(general_layout)
        controls_layout.addWidget(self.general_group)

        self.overlay_group = QtWidgets.QGroupBox("Overlay Volumes")
        overlay_layout = QtWidgets.QVBoxLayout()
        overlay_layout.setContentsMargins(4, 4, 4, 4)
        overlay_layout.setSpacing(4)
        self.overlay_list = QtWidgets.QListWidget()
        self.overlay_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.overlay_list.itemChanged.connect(self._on_overlay_item_changed)
        overlay_layout.addWidget(self.overlay_list)
        overlay_btn_row = QtWidgets.QHBoxLayout()
        self.overlay_remove_btn = QtWidgets.QPushButton("Remove Selected")
        self.overlay_remove_btn.clicked.connect(self._remove_selected_overlays)
        self.overlay_clear_btn = QtWidgets.QPushButton("Clear Overlays")
        self.overlay_clear_btn.clicked.connect(self._clear_overlay_volumes)
        overlay_btn_row.addWidget(self.overlay_remove_btn)
        overlay_btn_row.addWidget(self.overlay_clear_btn)
        overlay_btn_row.addStretch(1)
        overlay_layout.addLayout(overlay_btn_row)
        self.overlay_group.setLayout(overlay_layout)
        self.overlay_group.setVisible(False)
        controls_layout.addWidget(self.overlay_group)

        self.slice_view_group = QtWidgets.QGroupBox("Slice Viewer")
        slice_view_layout = QtWidgets.QVBoxLayout()
        slice_view_layout.setContentsMargins(4, 4, 4, 4)
        slice_view_layout.setSpacing(4)
        self.slice_hint_label = QtWidgets.QLabel(
            "Large TIFF detected. Showing on-demand axial slices."
        )
        self.slice_hint_label.setWordWrap(True)
        slice_view_layout.addWidget(self.slice_hint_label)
        self.slice_status_label = QtWidgets.QLabel("Slice: -/-")
        slice_view_layout.addWidget(self.slice_status_label)
        contrast_row = QtWidgets.QHBoxLayout()
        contrast_row.addWidget(QtWidgets.QLabel("Window min/max:"))
        self.slice_min_spin = QtWidgets.QDoubleSpinBox()
        self.slice_max_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.slice_min_spin, self.slice_max_spin):
            spin.setDecimals(3)
            spin.setRange(-1e9, 1e9)
            spin.setKeyboardTracking(False)
        self.slice_min_spin.valueChanged.connect(
            lambda _: self._on_slice_window_changed()
        )
        self.slice_max_spin.valueChanged.connect(
            lambda _: self._on_slice_window_changed()
        )
        contrast_row.addWidget(self.slice_min_spin)
        contrast_row.addWidget(self.slice_max_spin)
        self.slice_auto_btn = QtWidgets.QPushButton("Auto")
        self.slice_auto_btn.clicked.connect(self._slice_auto_window)
        contrast_row.addWidget(self.slice_auto_btn)
        slice_view_layout.addLayout(contrast_row)
        gamma_row = QtWidgets.QHBoxLayout()
        self.slice_gamma_label = QtWidgets.QLabel("Gamma: 1.00")
        self.slice_gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_gamma_slider.setRange(10, 400)  # 0.1 - 4.0
        self.slice_gamma_slider.setValue(100)
        self.slice_gamma_slider.valueChanged.connect(self._on_slice_gamma_changed)
        gamma_row.addWidget(self.slice_gamma_label)
        gamma_row.addWidget(self.slice_gamma_slider)
        slice_view_layout.addLayout(gamma_row)
        self.slice_index_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_index_slider.setRange(0, 0)
        self.slice_index_slider.setEnabled(False)
        self.slice_index_slider.valueChanged.connect(self._on_slice_index_changed)
        slice_view_layout.addWidget(self.slice_index_slider)
        self.slice_view_group.setLayout(slice_view_layout)
        self.slice_view_group.setVisible(False)
        controls_layout.addWidget(self.slice_view_group)

        self.slice_group = QtWidgets.QGroupBox("Slice Planes")
        slice_layout = QtWidgets.QVBoxLayout()
        slice_layout.setContentsMargins(4, 4, 4, 4)
        slice_layout.setSpacing(4)
        self._slice_plane_controls: dict[int, _SlicePlaneControl] = {}
        for axis, name, _ in PLANE_DEFS:
            row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(f"{name}: -/-")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 0)
            slider.setEnabled(False)
            slider.valueChanged.connect(partial(self._on_plane_slider_changed, axis))
            checkbox = QtWidgets.QCheckBox("Show")
            checkbox.setEnabled(False)
            checkbox.setToolTip(f"Show the {name} slice plane.")
            checkbox.stateChanged.connect(
                partial(self._on_plane_checkbox_changed, axis)
            )
            row.addWidget(label)
            row.addWidget(slider, 2)
            row.addWidget(checkbox)
            slice_layout.addLayout(row)
            self._slice_plane_controls[axis] = _SlicePlaneControl(
                axis=axis,
                name=name,
                slider=slider,
                checkbox=checkbox,
                label=label,
            )
        self.slice_group.setLayout(slice_layout)
        controls_layout.addWidget(self.slice_group)
        self._configure_plane_controls()

        self.status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QVBoxLayout()
        self.data_status_label = QtWidgets.QLabel("Data: none loaded")
        self.data_status_label.setWordWrap(True)
        self.mode_status_label = QtWidgets.QLabel("Mode: waiting for data")
        self.mode_status_label.setWordWrap(True)
        self.counts_status_label = QtWidgets.QLabel(
            "Overlays: 0 • Points: 0 • Meshes: 0"
        )
        self.counts_status_label.setWordWrap(True)
        self.volume_io_label = QtWidgets.QLabel("Volume I/O: idle")
        self.mesh_status_label = QtWidgets.QLabel("Mesh: none loaded")
        for lab in (
            self.data_status_label,
            self.mode_status_label,
            self.counts_status_label,
            self.volume_io_label,
            self.mesh_status_label,
        ):
            lab.setStyleSheet("color: #3a3a3a;")
        status_layout.addWidget(self.data_status_label)
        status_layout.addWidget(self.mode_status_label)
        status_layout.addWidget(self.counts_status_label)
        status_layout.addWidget(self.volume_io_label)
        status_layout.addWidget(self.mesh_status_label)
        self.status_group.setLayout(status_layout)
        controls_layout.addWidget(self.status_group)
        controls_layout.addStretch(1)

        # Build pipeline
        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self._configure_render_quality()

        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        # Camera interaction style (rotate/pan/zoom)
        self._style_trackball = vtkInteractorStyleTrackballCamera()
        self._style_inactive = vtkInteractorStyleUser()
        self.interactor.SetInteractorStyle(self._style_trackball)
        # Create a volume and add it
        self.volume = vtkVolume()
        self.mapper = vtkSmartVolumeMapper()
        self.property = vtkVolumeProperty()
        self.property.ShadeOn()
        self.property.SetInterpolationTypeToLinear()

        # Conditionally load volume if path looks like a volume source
        _loaded_volume = False
        if src_path and self._is_volume_candidate(self._path):
            try:
                _loaded_volume = self._load_volume()
            except Exception:
                _loaded_volume = False

        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.property)
        # Only add the volume actor if we actually loaded a volume
        if _loaded_volume:
            self.renderer.AddVolume(self.volume)
            self.renderer.ResetCamera()
        self.renderer.SetBackground(0.1, 0.1, 0.12)

        # Setup for interactive point picking
        self._picker = vtkPointPicker()
        self._picker.SetTolerance(0.005)  # Adjust sensitivity
        self._last_picked_id = -1
        self._last_picked_actor = None
        self._ensure_scene_light()

        # Setup interactive window/level mode and key/mouse bindings
        self._wl_mode = False
        self._wl_drag = False
        self._wl_last = (0, 0)
        self._install_interaction_bindings()

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self._point_actors: list[vtkActor] = []
        self._region_actors: dict[str, vtkActor] = {}
        self._region_entries: dict[str, _RegionSelectionEntry] = {}
        self._region_colors: dict[str, QtGui.QColor] = {}
        self._mesh_actors: list[vtkActor] = []
        self._mesh_textures: dict[int, dict[str, dict[str, object]]] = {}
        self._mesh_actor_names: dict[int, str] = {}
        self._active_mesh_actor: Optional[vtkActor] = None

        # If the provided path is a point cloud, add it now
        if (
            src_path
            and not _loaded_volume
            and self._is_point_cloud_candidate(self._path)
        ):
            try:
                ext = self._path.suffix.lower()
                if ext == ".ply":
                    self._add_point_cloud_ply(str(self._path))
                elif ext in (".csv", ".xyz"):
                    self._add_point_cloud_csv_or_xyz(str(self._path))
                self._update_point_sizes()
                self.renderer.ResetCamera()
                self.vtk_widget.GetRenderWindow().Render()
            except Exception:
                pass
        elif src_path and not _loaded_volume and self._is_mesh_candidate(self._path):
            try:
                self._load_mesh_file(str(self._path))
                self.renderer.ResetCamera()
                self.vtk_widget.GetRenderWindow().Render()
            except Exception:
                pass

        # Enable/disable volume-related controls based on whether a volume was loaded
        self._set_volume_controls_enabled(_loaded_volume)
        self._update_mesh_status_label()
        self._refresh_status_summary()

    def setModal(self, modal: bool):
        """QMainWindow cannot be modal; keep compatibility with QDialog API."""
        return

    def _apply_default_dock_width(self):
        if not hasattr(self, "_controls_dock") or not self._controls_dock:
            return
        try:
            total_width = max(1, self.width())
            preferred = max(220, min(360, int(total_width * 0.3)))
            self.resizeDocks([self._controls_dock], [preferred], QtCore.Qt.Horizontal)
        except Exception:
            pass

    def _create_quick_button(
        self,
        text: str,
        icon: QtGui.QIcon,
        callback: Callable[[], None],
        tooltip: str = "",
    ) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.setIcon(icon)
        btn.setIconSize(QtCore.QSize(16, 16))
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.setAutoRaise(True)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        if tooltip:
            btn.setToolTip(tooltip)
        try:
            btn.clicked.connect(callback)
        except Exception:
            pass
        return btn

    def _show_help_overlay(self):
        tips = [
            "Drag with the left mouse to rotate, middle to pan, and scroll to zoom.",
            "Press W to toggle window/level mode, then drag to adjust contrast.",
            "Press R to reset the camera, +/- to change opacity, and C to toggle shading.",
            "Use the slice index slider when large TIFF/Zarr volumes open in slice mode.",
            "Pick points to see region names; use the Quick Actions row to load common data.",
        ]
        QtWidgets.QMessageBox.information(
            self,
            "Viewer Tips",
            "\n\n".join(tips),
        )

    def _configure_render_quality(self) -> None:
        """Tune render window for better translucent gaussian splat compositing."""
        rw = self.vtk_widget.GetRenderWindow()
        try:
            rw.SetAlphaBitPlanes(1)
        except Exception:
            pass
        # Depth peeling improves alpha blending for dense, semi-transparent splats.
        try:
            rw.SetMultiSamples(0)
        except Exception:
            pass
        try:
            self.renderer.SetUseDepthPeeling(True)
            self.renderer.SetMaximumNumberOfPeels(50)
            self.renderer.SetOcclusionRatio(0.1)
        except Exception:
            pass
        # FXAA keeps edges smooth when MSAA is disabled for depth peeling.
        try:
            rw.SetUseFXAA(True)
        except Exception:
            pass

    def _install_interaction_bindings(self):
        # Mouse + key handlers
        self.interactor.AddObserver("LeftButtonPressEvent", self._vtk_on_left_press)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self._vtk_on_left_release)
        self.interactor.AddObserver("MouseMoveEvent", self._vtk_on_mouse_move)
        self.interactor.AddObserver("KeyPressEvent", self._vtk_on_key_press)

    def _set_volume_controls_enabled(self, enabled: bool):
        widgets = [
            self.min_spin,
            self.max_spin,
            self.auto_window_btn,
            self.shade_checkbox,
            self.opacity_slider,
            self.interp_combo,
            self.spacing_x,
            self.spacing_y,
            self.spacing_z,
            self.blend_combo,
            self.cmap_combo,
            self.wl_mode_checkbox,
        ]
        for w in widgets:
            try:
                w.setEnabled(enabled)
            except Exception:
                pass
        self.volume_group.setVisible(enabled)
        self.wl_mode_checkbox.setVisible(enabled)
        self.slice_group.setVisible(enabled)
        for axis, control in self._slice_plane_controls.items():
            available = self._plane_is_available(axis)
            control.slider.setEnabled(enabled and available)
            control.checkbox.setEnabled(enabled and available)
        if not enabled:
            self._clear_slice_clipping_planes()
        if hasattr(self, "slice_group"):
            self.slice_group.setVisible(enabled)
        if hasattr(self, "slice_view_group") and enabled and not self._slice_mode:
            self.slice_view_group.setVisible(False)

    def _reload_volume(self):
        if not self._is_volume_candidate(getattr(self, "_path", Path("."))):
            QtWidgets.QMessageBox.information(
                self,
                "Reload Volume",
                "No volume source is associated with this viewer.",
            )
            return
        try:
            self._teardown_slice_planes()
            self._teardown_slice_mode()
            self._clear_slice_clipping_planes()
            self._clear_overlay_volumes()
            self._volume_shape = (0, 0, 0)
            self._volume_np = None
            self._vtk_img = None
            self._has_volume = False
            loaded = self._load_volume()
            if loaded:
                self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Reload Volume",
                f"Failed to reload volume:\n{exc}",
            )

    def _load_volume_dialog(self):
        start_dir = (
            str(self._path.parent)
            if getattr(self, "_path", None) and self._path.exists()
            else "."
        )
        dialog = QtWidgets.QFileDialog(self, "Open 3D Volume")
        dialog.setDirectory(start_dir)
        dialog.setNameFilter(VOLUME_SOURCE_FILTERS)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QtWidgets.QFileDialog.ReadOnly, True)
        paths: list[str] = []
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            paths = dialog.selectedFiles()
        if not paths:
            fallback_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Open 3D Volume Folder",
                start_dir,
                QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.ReadOnly,
            )
            if fallback_dir:
                paths = [fallback_dir]
        candidates = []
        for raw in paths:
            if not raw:
                continue
            try:
                candidate = Path(raw).expanduser()
            except Exception:
                candidate = Path(raw)
            try:
                candidate = candidate.resolve()
            except Exception:
                pass
            candidates.append(self._normalize_volume_selection(candidate))
        if not candidates:
            return
        replace_primary = False
        if self._has_volume:
            resp = QtWidgets.QMessageBox.question(
                self,
                "Load Volume",
                "Replace the current primary volume with the first selection?\n"
                "Choose No to keep the current volume visible and add all selections as overlays.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            replace_primary = resp == QtWidgets.QMessageBox.Yes
        for idx, candidate in enumerate(candidates):
            if idx == 0 and (not self._has_volume or replace_primary):
                self._source_path = candidate
                self._path = self._resolve_initial_source(candidate)
                self._reload_volume()
            else:
                self._add_overlay_volume(candidate)

    def _add_overlay_volume(self, source_path: Path):
        if not self._has_volume or self._slice_mode:
            QtWidgets.QMessageBox.information(
                self,
                "Overlay Volume",
                "Load a primary volume first before adding overlays.",
            )
            return
        try:
            volume_data = self._read_volume_any(source_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Overlay Volume",
                f"Failed to read overlay volume:\n{exc}",
            )
            return
        if volume_data.slice_mode:
            QtWidgets.QMessageBox.warning(
                self,
                "Overlay Volume",
                "Overlay volumes must fit into memory; streaming-only volumes are not supported.",
            )
            return
        vtk_img, _ = self._prepare_vtk_image(volume_data)
        mapper = vtkSmartVolumeMapper()
        mapper.SetInputData(vtk_img)
        prop = vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        prop.SetScalarOpacityUnitDistance(self.property.GetScalarOpacityUnitDistance())
        prop.SetScalarOpacity(
            self._opacity_tf if self._opacity_tf else vtkPiecewiseFunction()
        )
        prop.SetColor(self._color_tf if self._color_tf else vtkColorTransferFunction())
        actor = vtkVolume()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)
        actor.SetVisibility(self._volume_visible)
        self.renderer.AddVolume(actor)
        entry = _OverlayVolumeEntry(
            path=source_path,
            actor=actor,
            mapper=mapper,
            property=prop,
            label=source_path.name,
            visible=True,
        )
        self._overlay_volumes.append(entry)
        self._refresh_overlay_list()
        self.vtk_widget.GetRenderWindow().Render()
        self._refresh_status_summary()

    def _load_volume(self) -> bool:
        old_array = self._out_of_core_array
        old_path = self._out_of_core_backing_path
        self._out_of_core_array = None
        self._out_of_core_backing_path = None
        self._label_volume_active = False
        volume_data = self._read_volume_any()
        self._label_volume_active = bool(getattr(volume_data, "is_label_map", False))
        if volume_data.slice_mode:
            self._init_slice_mode(volume_data)
            self._cleanup_out_of_core_backing(old_array, old_path)
            return False
        vtk_img, volume_array = self._prepare_vtk_image(volume_data)
        spacing = volume_data.spacing
        z, y, x = volume_array.shape
        self._volume_shape = (int(z), int(y), int(x))

        # Keep handles and stats
        self._vtk_img = vtk_img
        self._volume_np = volume_array
        self._out_of_core_active = volume_data.is_out_of_core
        if volume_data.is_out_of_core:
            self._out_of_core_array = volume_array  # keep memmap alive
            self._out_of_core_backing_path = volume_data.backing_path
        self._vmin = volume_data.vmin
        self._vmax = volume_data.vmax
        self._update_volume_io_label(self._out_of_core_active)

        # Initialize window controls
        for spin in (self.min_spin, self.max_spin):
            spin.blockSignals(True)
        self.min_spin.setRange(self._vmin, max(self._vmin + 1e-6, self._vmax))
        self.max_spin.setRange(self._vmin, self._vmax)
        self.min_spin.setValue(self._vmin)
        self.max_spin.setValue(self._vmax)
        for spin in (self.min_spin, self.max_spin):
            spin.blockSignals(False)

        # Create initial transfer functions
        self._opacity_tf = vtkPiecewiseFunction()
        self._color_tf = vtkColorTransferFunction()
        self._update_transfer_functions()

        self.property.SetScalarOpacity(self._opacity_tf)
        self.property.SetColor(self._color_tf)
        self.property.SetAmbient(0.1)
        self.property.SetDiffuse(0.9)
        self.property.SetSpecular(0.2)

        self.mapper.SetInputData(vtk_img)
        self._clear_slice_clipping_planes()
        self._setup_slice_plane()
        self._configure_plane_controls()
        # If spacing provided, reflect it in UI
        if spacing is not None and len(spacing) == 3:
            try:
                self.spacing_x.blockSignals(True)
                self.spacing_y.blockSignals(True)
                self.spacing_z.blockSignals(True)
                self.spacing_x.setValue(float(spacing[0]))
                self.spacing_y.setValue(float(spacing[1]))
                self.spacing_z.setValue(float(spacing[2]))
            finally:
                self.spacing_x.blockSignals(False)
                self.spacing_y.blockSignals(False)
                self.spacing_z.blockSignals(False)

        self._update_opacity()
        self._update_shading()
        self._update_blend_mode()
        if self._label_volume_active and hasattr(self, "cmap_combo"):
            idx = self.cmap_combo.findText("Allen Atlas")
            if idx >= 0:
                try:
                    self.cmap_combo.blockSignals(True)
                    self.cmap_combo.setCurrentIndex(idx)
                finally:
                    self.cmap_combo.blockSignals(False)
        self._has_volume = True
        self._set_volume_controls_enabled(True)
        self._cleanup_out_of_core_backing(old_array, old_path)
        self._ensure_volume_actor_added()
        self._set_volume_actor_visibility(self._volume_visible, force=True)
        self._refresh_status_summary()
        return True

    def _prepare_vtk_image(
        self, volume_data: _VolumeData
    ) -> tuple[vtkImageData, np.ndarray]:
        volume = volume_data.array
        if volume is None:
            raise RuntimeError("Volume source returned no data.")
        if (
            not volume_data.is_grayscale
            and volume.ndim == 4
            and volume.shape[-1] in (3, 4)
        ):
            volume = np.dot(
                volume[..., :3],
                [0.299, 0.587, 0.114],
            ).astype(volume.dtype)
        if not volume.flags.c_contiguous:
            volume = np.ascontiguousarray(volume)
        z, y, x = volume.shape
        vtk_img = vtkImageData()
        vtk_img.SetDimensions(int(x), int(y), int(z))
        spacing = volume_data.spacing
        if spacing is not None and len(spacing) == 3:
            vtk_img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
        else:
            vtk_img.SetSpacing(1.0, 1.0, 1.0)
        vtk_img.SetOrigin(0.0, 0.0, 0.0)
        vtk_array = numpy_to_vtk(
            num_array=volume.reshape(-1),
            deep=not volume_data.is_out_of_core,
        )
        vtk_img.GetPointData().SetScalars(vtk_array)
        return vtk_img, volume

    def _init_slice_mode(self, volume_data: _VolumeData):
        self._teardown_slice_mode()
        if volume_data.slice_loader is None:
            raise RuntimeError("Slice loader is not available for this volume.")
        self._slice_mode = True
        self._slice_loader = volume_data.slice_loader
        self._slice_axis = volume_data.slice_axis or 0
        self._slice_vmin = volume_data.vmin
        self._slice_vmax = volume_data.vmax
        self._slice_start_index_hint = self._initial_slice_index_for_loader(
            self._slice_loader
        )
        self._slice_window_override = False
        if self._overlay_volumes:
            self._clear_overlay_volumes()
        if hasattr(self, "volume"):
            try:
                self.renderer.RemoveVolume(self.volume)
            except Exception:
                pass
        self._set_volume_controls_enabled(False)
        if hasattr(self, "slice_group"):
            self.slice_group.setVisible(False)
        self.slice_view_group.setVisible(True)
        blank = vtkImageData()
        blank.SetDimensions(1, 1, 1)
        blank.SetSpacing(1.0, 1.0, 1.0)
        blank.SetOrigin(0.0, 0.0, 0.0)
        vtk_blank = numpy_to_vtk(np.zeros(1, dtype=np.float32), deep=True)
        blank.GetPointData().SetScalars(vtk_blank)
        try:
            self.mapper.SetInputData(blank)
        except Exception:
            pass
        try:
            self.renderer.RemoveVolume(self.volume)
        except Exception:
            pass
        try:
            self.volume.VisibilityOff()
        except Exception:
            pass
        total = self._slice_loader.total_slices() if self._slice_loader else 0
        self.slice_index_slider.blockSignals(True)
        self.slice_index_slider.setRange(0, max(0, total - 1))
        start_idx = (
            int(self._slice_start_index_hint)
            if self._slice_start_index_hint is not None
            else 0
        )
        start_idx = max(0, min(start_idx, max(0, total - 1)))
        self.slice_index_slider.setValue(start_idx)
        self.slice_index_slider.setEnabled(total > 1)
        self.slice_index_slider.blockSignals(False)
        self._update_slice_status_label(start_idx, total)
        self._slice_gamma = 1.0
        self.slice_gamma_slider.blockSignals(True)
        self.slice_gamma_slider.setValue(100)
        self.slice_gamma_slider.blockSignals(False)
        self._update_slice_gamma_label()
        self._configure_slice_window_controls()
        if self._slice_actor is None:
            self._slice_actor = vtkImageActor()
        try:
            self.renderer.RemoveActor(self._slice_actor)
        except Exception:
            pass
        self.renderer.AddActor(self._slice_actor)
        self._volume_shape = volume_data.volume_shape or (total, 0, 0)
        self._load_slice_image(start_idx)
        self._update_volume_io_label(True)

    def _load_slice_image(self, index: int):
        if not self._slice_loader or self._slice_actor is None:
            return
        total = max(1, self._slice_loader.total_slices())
        sanitized = max(0, min(int(index), total - 1))
        slice_array = self._slice_loader.read_slice(self._slice_axis, sanitized)
        arr = np.asarray(slice_array)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = self._convert_frame_to_plane(arr, arr.dtype)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                arr = arr[..., 0]

        if getattr(self, "_label_volume_active", False):
            raw_min = float(np.min(arr))
            raw_max = float(np.max(arr))
            self._slice_last_data_min = raw_min
            self._slice_last_data_max = raw_max
            rgb = self._labels_to_rgb(arr)
            h, w = rgb.shape[:2]
            vtk_img = vtkImageData()
            vtk_img.SetDimensions(int(w), int(h), 1)
            vtk_img.SetSpacing(1.0, 1.0, 1.0)
            vtk_img.SetOrigin(0.0, 0.0, 0.0)
            flat = rgb.reshape(-1, 3)
            try:
                array_type = get_vtk_array_type(np.dtype(np.uint8))
            except Exception:
                array_type = None
            vtk_array = numpy_to_vtk(flat, deep=True, array_type=array_type)  # type: ignore[arg-type]
            try:
                vtk_array.SetNumberOfComponents(3)
            except Exception:
                pass
            vtk_img.GetPointData().SetScalars(vtk_array)
            self._slice_actor.SetInputData(vtk_img)
            img_prop = self._slice_actor.GetProperty()
            img_prop.SetColorWindow(255.0)
            img_prop.SetColorLevel(127.5)
            self._slice_current_index = sanitized
            self._update_slice_status_label(sanitized, total)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            return

        raw_min = float(np.min(arr))
        raw_max = float(np.max(arr))
        self._slice_last_data_min = raw_min
        self._slice_last_data_max = raw_max
        need_window_update = False
        if not self._slice_window_override:
            self._slice_vmin = raw_min
            self._slice_vmax = raw_max
            need_window_update = True
        display_min = self._slice_vmin
        display_max = self._slice_vmax
        if display_max <= display_min:
            display_max = display_min + 1e-3
            need_window_update = True
        if raw_max <= display_min:
            display_min = raw_min
            display_max = max(raw_min + 1e-3, raw_max)
            need_window_update = True
        elif raw_min >= display_max:
            display_min = raw_min
            display_max = max(raw_max, raw_min + 1e-3)
            need_window_update = True
        if need_window_update:
            self._slice_vmin = display_min
            self._slice_vmax = display_max
            self._configure_slice_window_controls()
        norm = arr.astype(np.float32)
        span = float(display_max - display_min)
        span = max(span, 1e-6)
        norm = (norm - display_min) / span
        norm = np.clip(norm, 0.0, 1.0)
        gamma = max(0.1, float(self._slice_gamma))
        if abs(gamma - 1.0) > 1e-6:
            norm = np.power(norm, gamma)
        h, w = norm.shape
        vtk_img = vtkImageData()
        vtk_img.SetDimensions(int(w), int(h), 1)
        vtk_img.SetSpacing(1.0, 1.0, 1.0)
        vtk_img.SetOrigin(0.0, 0.0, 0.0)
        vtk_array = numpy_to_vtk(norm.reshape(-1), deep=True)
        vtk_img.GetPointData().SetScalars(vtk_array)
        self._slice_actor.SetInputData(vtk_img)

        # FIX: tell VTK our intensities are in [0, 1]
        img_prop = self._slice_actor.GetProperty()
        img_prop.SetColorWindow(1.0)  # max - min of the displayed range
        img_prop.SetColorLevel(0.5)  # center of the displayed range

        self._slice_current_index = sanitized
        self._update_slice_status_label(sanitized, total)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _configure_slice_window_controls(self):
        if not hasattr(self, "slice_min_spin"):
            return
        vmin = float(self._slice_vmin)
        vmax = float(self._slice_vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-3
        for spin in (self.slice_min_spin, self.slice_max_spin):
            spin.blockSignals(True)
            spin.setRange(vmin - 1e6, vmax + 1e6)
        self.slice_min_spin.setValue(vmin)
        self.slice_max_spin.setValue(vmax)
        for spin in (self.slice_min_spin, self.slice_max_spin):
            spin.blockSignals(False)
        self._slice_vmin = vmin
        self._slice_vmax = vmax

    def _on_slice_window_changed(self):
        if not self._slice_mode:
            return
        vmin = float(min(self.slice_min_spin.value(), self.slice_max_spin.value()))
        vmax = float(max(self.slice_min_spin.value(), self.slice_max_spin.value()))
        if vmax <= vmin:
            vmax = vmin + 1e-3
        self.slice_min_spin.blockSignals(True)
        self.slice_max_spin.blockSignals(True)
        self.slice_min_spin.setValue(vmin)
        self.slice_max_spin.setValue(vmax)
        self.slice_min_spin.blockSignals(False)
        self.slice_max_spin.blockSignals(False)
        self._slice_vmin = vmin
        self._slice_vmax = vmax
        self._slice_window_override = True
        self._load_slice_image(self._slice_current_index)

    def _slice_auto_window(self):
        if not self._slice_mode or not self._slice_loader:
            return
        idx = self._slice_current_index
        sample = self._slice_loader.read_slice(self._slice_axis, idx)
        arr = np.asarray(sample).astype(np.float32)
        if arr.size == 0:
            return
        p2 = float(np.percentile(arr, 2))
        p98 = float(np.percentile(arr, 98))
        if p98 <= p2:
            p2 = float(arr.min())
            p98 = float(arr.max())
        self._slice_vmin = p2
        self._slice_vmax = p98
        self._slice_window_override = True
        self._configure_slice_window_controls()
        self._load_slice_image(idx)

    def _on_slice_gamma_changed(self, value: int):
        self._slice_gamma = max(0.1, float(value) / 100.0)
        self._update_slice_gamma_label()
        if self._slice_mode:
            self._slice_window_override = True
            self._load_slice_image(self._slice_current_index)

    def _update_slice_gamma_label(self):
        if hasattr(self, "slice_gamma_label"):
            self.slice_gamma_label.setText(f"Gamma: {self._slice_gamma:.2f}")

    def _on_slice_index_changed(self, value: int):
        if not self._slice_mode:
            return
        self._load_slice_image(int(value))

    def _update_slice_status_label(self, index: int, total: int):
        if not hasattr(self, "slice_status_label"):
            return
        if total <= 0:
            self.slice_status_label.setText("Slice: -/-")
        else:
            self.slice_status_label.setText(f"Slice: {index + 1}/{total}")
        if total > 0:
            self.slice_hint_label.setText(
                "Large TIFF in slice mode. Adjust contrast or switch slices as needed."
            )

    def _teardown_slice_mode(self):
        self._slice_mode = False
        if self._slice_actor is not None:
            try:
                self.renderer.RemoveActor(self._slice_actor)
            except Exception:
                pass
            self._slice_actor = None
        self.slice_view_group.setVisible(False)
        self.slice_index_slider.blockSignals(True)
        self.slice_index_slider.setRange(0, 0)
        self.slice_index_slider.setValue(0)
        self.slice_index_slider.setEnabled(False)
        self.slice_index_slider.blockSignals(False)
        self._update_slice_status_label(0, 0)
        self._close_slice_loader()
        self._slice_vmin = 0.0
        self._slice_vmax = 1.0
        self._slice_gamma = 1.0
        self._update_slice_gamma_label()
        self._slice_window_override = False
        self._slice_start_index_hint = None
        self._refresh_status_summary()

    def _close_slice_loader(self):
        if self._slice_loader is None:
            return
        try:
            self._slice_loader.close()
        except Exception:
            pass
        self._slice_loader = None

    def _cleanup_out_of_core_backing(
        self,
        array: Optional[np.memmap] = None,
        path: Optional[Path] = None,
    ):
        if array is None and path is None:
            array = self._out_of_core_array
            path = self._out_of_core_backing_path
            self._out_of_core_array = None
            self._out_of_core_backing_path = None
        if array is not None:
            try:
                mmap_obj = getattr(array, "_mmap", None)
                if mmap_obj is not None:
                    mmap_obj.close()
            except Exception:
                pass
        if path:
            try:
                Path(path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except TypeError:
                try:
                    Path(path).unlink()
                except Exception:
                    pass
            except Exception:
                pass

    def _read_volume_any(self, source: Optional[Path] = None) -> _VolumeData:
        """Read a 3D volume from TIFF/NIfTI/DICOM or directory (DICOM series)."""
        path = source or self._path
        try:
            if path.is_dir() and self._is_zarr_candidate(path):
                return self._read_zarr(path)
            if path.is_dir():
                volume, spacing = self._read_dicom_series(path)
                return _VolumeData(
                    array=volume,
                    spacing=spacing,
                    vmin=float(volume.min()),
                    vmax=float(volume.max()),
                    is_grayscale=volume.ndim == 3,
                    is_out_of_core=False,
                    volume_shape=tuple(int(x) for x in volume.shape[:3]),
                )

            suffix = path.suffix.lower()
            name_lower = path.name.lower()
            if name_lower.endswith(".nii") or name_lower.endswith(".nii.gz"):
                # NIfTI via VTK reader
                try:
                    from vtkmodules.vtkIOImage import vtkNIFTIImageReader
                except Exception as exc:
                    raise RuntimeError(
                        "VTK NIFTI reader is not available in this build."
                    ) from exc
                reader = vtkNIFTIImageReader()
                reader.SetFileName(str(path))
                reader.Update()
                vtk_img = reader.GetOutput()
                vol = self._vtk_image_to_numpy(vtk_img)
                s = vtk_img.GetSpacing()
                spacing = (s[0], s[1], s[2])
                vol = self._normalize_to_float01(vol)
                return _VolumeData(
                    array=vol,
                    spacing=spacing,
                    vmin=float(vol.min()),
                    vmax=float(vol.max()),
                    is_grayscale=vol.ndim == 3,
                    is_out_of_core=False,
                    volume_shape=tuple(int(x) for x in vol.shape[:3]),
                )
            if suffix in (".dcm", ".ima", ".dicom"):
                # Treat as a DICOM series from the containing folder
                volume, spacing = self._read_dicom_series(path.parent)
                return _VolumeData(
                    array=volume,
                    spacing=spacing,
                    vmin=float(volume.min()),
                    vmax=float(volume.max()),
                    is_grayscale=volume.ndim == 3,
                    is_out_of_core=False,
                    volume_shape=tuple(int(x) for x in volume.shape[:3]),
                )
            if self._is_zarr_candidate(path):
                return self._read_zarr(path)

            if self._is_tiff_candidate(path):
                preferred_out_of_core = self._should_use_out_of_core_tiff(path)
                if preferred_out_of_core:
                    return self._read_tiff_out_of_core(path)
                try:
                    return self._read_tiff_eager(path)
                except MemoryError as exc:
                    logger.warning(
                        "Standard TIFF loading failed (%s). Retrying with out-of-core caching.",
                        exc,
                    )
                    return self._read_tiff_out_of_core(path)

            raise RuntimeError(f"Unsupported volume format: {path}")
        except Exception as e:
            # Re-raise with context so upper layer shows a concise message
            raise RuntimeError(f"Failed to read volume from '{path}': {e}")

    def _is_tiff_candidate(self, path: Path) -> bool:
        name_lower = path.name.lower()
        suffix = path.suffix.lower()
        if suffix in TIFF_SUFFIXES:
            return True
        return any(name_lower.endswith(ext) for ext in OME_TIFF_SUFFIXES)

    def _is_zarr_candidate(self, path: Path) -> bool:
        if not path:
            return False
        suffix = path.suffix.lower()
        if suffix == ".zarr":
            return True
        try:
            if path.is_file():
                name = path.name.lower()
                if name.endswith("zarr.json") or name.endswith(".zgroup"):
                    return True
                parent = path.parent
                if (
                    (parent / ".zarray").exists()
                    or (parent / "zarr.json").exists()
                    or (parent / ".zgroup").exists()
                ):
                    return True
                if (parent / "data" / ".zarray").exists():
                    return True
            if path.is_dir():
                if (
                    (path / ".zarray").exists()
                    or (path / "zarr.json").exists()
                    or (path / ".zgroup").exists()
                ):
                    return True
                if (path / "data" / ".zarray").exists() or (
                    path / "data" / "zarr.json"
                ).exists():
                    return True
        except Exception:
            return False
        return False

    def _should_use_out_of_core_tiff(self, path: Path) -> bool:
        size_bytes = self._safe_file_size(path)
        avail_bytes = self._available_memory_bytes()
        logger.debug(
            "VTK viewer source size=%s bytes, available RAM=%s bytes",
            size_bytes,
            avail_bytes,
        )
        if avail_bytes > 0 and size_bytes > 0 and size_bytes >= avail_bytes:
            logger.info(
                "TIFF stack (%s) is larger than available RAM; enabling out-of-core caching.",
                path,
            )
            return True
        if AUTO_OUT_OF_CORE_MB > 0 and size_bytes > 0:
            if size_bytes >= AUTO_OUT_OF_CORE_MB * 1024 * 1024:
                logger.info(
                    "TIFF stack (%s) exceeds configured threshold (%.0f MB); enabling out-of-core caching.",
                    path,
                    AUTO_OUT_OF_CORE_MB,
                )
                return True
        return False

    def _safe_file_size(self, path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except Exception:
            return 0

    def _available_memory_bytes(self) -> int:
        try:
            import psutil  # type: ignore
        except Exception:
            return 0
        try:
            mem = psutil.virtual_memory()
        except Exception:
            return 0
        return int(getattr(mem, "available", 0) or 0)

    def _read_tiff_eager(self, path: Path) -> _VolumeData:
        frames: list[np.ndarray] = []
        with Image.open(str(path)) as img:
            n = max(1, int(getattr(img, "n_frames", 1) or 1))
            for i in range(n):
                img.seek(i)
                frames.append(np.array(img))
        if not frames:
            raise RuntimeError("No frames found in TIFF stack.")
        vol = np.stack(frames, axis=0)
        vol = self._normalize_to_float01(vol)
        return _VolumeData(
            array=vol,
            spacing=None,
            vmin=float(vol.min()),
            vmax=float(vol.max()),
            is_grayscale=vol.ndim == 3,
            is_out_of_core=False,
            volume_shape=tuple(int(x) for x in vol.shape[:3]),
        )

    def _read_tiff_out_of_core(self, path: Path) -> _VolumeData:
        meta = self._probe_tiff_metadata(path)
        shape = meta[0] if meta else None
        dtype = meta[1] if meta else None
        memmap_arr = self._open_tiff_memmap(path)
        if memmap_arr is not None:
            mem_shape = tuple(int(x) for x in memmap_arr.shape)
            shape = shape or mem_shape
            dtype = dtype or memmap_arr.dtype
            if shape and dtype and self._should_use_slice_mode(shape, dtype):
                loader = _MemmapSliceLoader(memmap_arr)
                logger.info(
                    "Slice mode (memmap) for TIFF stack '%s' (shape=%s, dtype=%s)",
                    path,
                    loader.shape(),
                    loader.dtype(),
                )
                return self._make_slice_volume_data(loader)
            vmin, vmax = self._dtype_value_range(memmap_arr.dtype)
            logger.info(
                "Using tifffile.memmap for TIFF stack '%s' (shape=%s, dtype=%s)",
                path,
                mem_shape,
                memmap_arr.dtype,
            )
            return _VolumeData(
                array=memmap_arr,
                spacing=None,
                vmin=vmin,
                vmax=vmax,
                is_grayscale=True,
                is_out_of_core=True,
                backing_path=None,
                volume_shape=mem_shape,
            )
        if shape and dtype and self._should_use_slice_mode(shape, dtype):
            loader = _TiffSliceLoader(path, shape, dtype)
            logger.info(
                "Slice mode (paged) for TIFF stack '%s' (shape=%s, dtype=%s)",
                path,
                shape,
                dtype,
            )
            return self._make_slice_volume_data(loader, spacing=None)
        with Image.open(str(path)) as img:
            n_frames = max(1, int(getattr(img, "n_frames", 1) or 1))
            img.seek(0)
            first = np.array(img)
            convert_color = first.ndim == 3 and first.shape[-1] in (3, 4)
            if convert_color:
                target_dtype = first.dtype
                first_plane = self._convert_frame_to_plane(first, target_dtype)
            else:
                target_dtype = first.dtype
                first_plane = np.asarray(first, dtype=target_dtype)
                if first_plane.ndim == 3 and first_plane.shape[-1] == 1:
                    first_plane = first_plane[..., 0]
            if first_plane.ndim != 2:
                raise RuntimeError("Only grayscale TIFF stacks are supported.")
            plane_shape = first_plane.shape
            fd, tmp_path = tempfile.mkstemp(prefix="annolid_vtk_", suffix=".mmap")
            os.close(fd)
            backing_path = Path(tmp_path)
            try:
                writer = np.memmap(
                    backing_path,
                    mode="w+",
                    dtype=target_dtype,
                    shape=(n_frames, plane_shape[0], plane_shape[1]),
                )
                min_val = float(np.min(first_plane))
                max_val = float(np.max(first_plane))
                writer[0] = first_plane
                for idx in range(1, n_frames):
                    img.seek(idx)
                    arr = np.array(img)
                    if convert_color:
                        arr = self._convert_frame_to_plane(arr, target_dtype)
                    else:
                        arr = np.asarray(arr, dtype=target_dtype)
                        if arr.ndim == 3 and arr.shape[-1] == 1:
                            arr = arr[..., 0]
                    writer[idx] = arr
                    min_val = min(min_val, float(np.min(arr)))
                    max_val = max(max_val, float(np.max(arr)))
                writer.flush()
            except Exception:
                try:
                    backing_path.unlink()
                except Exception:
                    pass
                raise
            finally:
                try:
                    del writer
                except UnboundLocalError:
                    pass
        reader = np.memmap(
            backing_path,
            mode="r+",
            dtype=target_dtype,
            shape=(n_frames, plane_shape[0], plane_shape[1]),
        )
        return _VolumeData(
            reader,
            None,
            float(min_val),
            float(max_val),
            is_grayscale=True,
            is_out_of_core=True,
            backing_path=backing_path,
            volume_shape=(n_frames, plane_shape[0], plane_shape[1]),
        )

    def _first_3d_array_from_group(self, grp):
        """Breadth-first search for the first array with >=3 dimensions."""
        try:
            from zarr.hierarchy import Group  # type: ignore
        except Exception:
            Group = ()  # type: ignore[assignment]

        queue = [grp]
        while queue:
            node = queue.pop(0)
            try:
                # Arrays directly under this node
                for key in getattr(node, "array_keys", lambda: [])():
                    try:
                        arr = node[key]
                        shp = getattr(arr, "shape", ())
                        if shp and len(shp) >= 3:
                            try:
                                logger.info(
                                    "VTK viewer: BFS picked array '%s' (shape=%s)",
                                    getattr(arr, "path", None)
                                    or getattr(arr, "name", None),
                                    shp,
                                )
                            except Exception:
                                pass
                            return arr
                    except Exception:
                        continue
                # Child groups
                for key in getattr(node, "group_keys", lambda: [])():
                    try:
                        child = node[key]
                        if isinstance(child, Group):
                            queue.append(child)
                    except Exception:
                        continue
            except Exception:
                continue
        return None

    def _sample_zarr_minmax(
        self,
        arr_obj,
        zyx_axes: tuple[int, int, int],
        fixed_idx: dict[int, int],
    ) -> Optional[tuple[float, float]]:
        """Read a tiny block to estimate value range without loading everything."""
        shape = getattr(arr_obj, "shape", ())
        if not shape:
            return None
        slicer: list[object] = []
        for dim, size in enumerate(shape):
            if dim == zyx_axes[0]:
                slicer.append(0)
            elif dim in zyx_axes:
                slicer.append(slice(0, min(64, int(size))))
            else:
                slicer.append(fixed_idx.get(dim, 0))
        try:
            sample = np.asarray(arr_obj[tuple(slicer)])
            if sample.size == 0:
                return None
            sample = np.squeeze(sample)
            return float(np.min(sample)), float(np.max(sample))
        except Exception:
            return None

    def _is_label_volume(self, dtype: np.dtype, arr_obj, source_path: Path) -> bool:
        """Heuristic: keep integer masks (annotation/label/seg) un-normalized."""
        if not np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
            return False
        tokens = ("label", "mask", "annot", "seg")
        name = str(getattr(arr_obj, "path", "") or getattr(arr_obj, "name", "")).lower()
        if any(tok in name for tok in tokens):
            return True
        if any(tok in source_path.name.lower() for tok in tokens):
            return True
        attrs = getattr(arr_obj, "attrs", None)
        if attrs:
            try:
                keys = [str(k).lower() for k in getattr(attrs, "keys", lambda: [])()]
                if any(tok in k for k in keys for tok in tokens):
                    return True
                for key in (
                    "labels",
                    "label",
                    "annotation",
                    "annotations",
                    "image-label",
                    "image_label",
                    "segmentation",
                ):
                    if key in attrs:
                        return True
            except Exception:
                pass
        return False

    def _allen_mouse_color(self, label: int) -> tuple[float, float, float]:
        """Return an atlas-inspired RGB triple in [0,1] for a label id."""
        if label <= 0:
            return (0.0, 0.0, 0.0)
        try:
            idx = label % len(ALLEN_MOUSE_ATLAS_COLORS)
            hex_color = ALLEN_MOUSE_ATLAS_COLORS[idx].lstrip("#")
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return (r, g, b)
        except Exception:
            return (0.8, 0.8, 0.8)

    def _collect_label_values(self, limit: int = 256) -> list[int]:
        """Collect a limited set of label ids for transfer functions."""
        if self._volume_np is None:
            return []
        try:
            arr = np.asarray(self._volume_np)
            flat = arr.ravel()
            if flat.size > limit * 8000:
                step = max(1, flat.size // (limit * 8000))
                flat = flat[::step]
            vals = np.unique(flat)
            if vals.size > limit:
                vals = vals[:limit]
            return [int(v) for v in vals]
        except Exception:
            return []

    def _apply_allen_color_tf(
        self, label_values: Sequence[int], vmin: float, vmax: float
    ) -> None:
        """Populate the color transfer function with atlas-inspired colors."""
        self._color_tf.RemoveAllPoints()
        if not label_values:
            self._color_tf.AddRGBPoint(vmin, 0.0, 0.0, 0.0)
            self._color_tf.AddRGBPoint(vmax, 1.0, 1.0, 1.0)
            return
        added: set[int] = set()
        for val in label_values:
            if val in added:
                continue
            r, g, b = self._allen_mouse_color(val)
            self._color_tf.AddRGBPoint(float(val), r, g, b)
            added.add(val)
        if 0 not in added:
            self._color_tf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)

    def _labels_to_rgb(self, arr: np.ndarray) -> np.ndarray:
        """Convert a 2D integer label plane to RGB using the atlas palette."""
        flattened = np.asarray(arr).astype(np.int64, copy=False)
        vals = np.unique(flattened)
        palette = {
            int(v): (np.array(self._allen_mouse_color(int(v))) * 255).astype(np.uint8)
            for v in vals
        }
        rgb = np.zeros(flattened.shape + (3,), dtype=np.uint8)
        for v, color in palette.items():
            mask = flattened == v
            if np.any(mask):
                rgb[mask] = color
        return rgb

    def _load_zarr_json(self, meta_path: Path) -> Optional[dict]:
        """Safely load a zarr.json file or return None."""
        try:
            path = meta_path
            if path.is_dir():
                path = path / "zarr.json"
            if not path.exists():
                return None
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _find_zarr_array_metadata(
        self, base: Path
    ) -> tuple[Optional[Path], Optional[dict]]:
        """
        Locate the nearest Zarr array metadata starting from `base`.
        Returns the directory containing the metadata and the parsed JSON.
        """
        try:
            base_path = base
            if base_path.is_file():
                base_path = base_path.parent
        except Exception:
            base_path = base

        queue: list[tuple[Path, int]] = [(base_path, 0)]
        visited: set[Path] = set()
        max_depth = 3

        while queue:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            zarr_json = current / "zarr.json"
            dot_zarray = current / ".zarray"
            if zarr_json.exists():
                meta = self._load_zarr_json(zarr_json)
                if meta and (meta.get("node_type") == "array" or "shape" in meta):
                    return current, meta
            if dot_zarray.exists():
                try:
                    with open(dot_zarray, "r") as f:
                        meta = json.load(f)
                    meta.setdefault("zarr_format", 2)
                    meta.setdefault("node_type", "array")
                    return current, meta
                except Exception:
                    pass

            if depth >= max_depth:
                continue

            try:
                for child in current.iterdir():
                    if not child.is_dir():
                        continue
                    if child.name in {"c", ".zattrs", ".zgroup"}:
                        continue
                    if current.name == "c":
                        continue
                    if child.name.startswith("."):
                        continue
                    queue.append((child, depth + 1))
            except Exception:
                continue

        return None, None

    def _open_zarr_array(self, path: Path):
        """Open a Zarr array, with fallback support for v3 directory stores."""
        meta_path, meta = self._find_zarr_array_metadata(path)
        errors: list[str] = []

        zarr_mod = None
        try:
            import zarr as zarr_mod  # type: ignore
        except Exception as exc:
            errors.append(f"zarr not available: {exc}")

        if zarr_mod is not None:
            candidate = meta_path or path
            try:
                root = self._open_zarr_store(candidate, zarr_mod)
                if root is not None:
                    arr_obj = self._select_zarr_array(root)
                    try:
                        from zarr.core import Array  # type: ignore
                        from zarr.hierarchy import Group  # type: ignore
                    except Exception:
                        Array = ()  # type: ignore[assignment]
                        Group = ()  # type: ignore[assignment]
                    if isinstance(arr_obj, Group):
                        arr_obj = self._first_3d_array_from_group(arr_obj)
                    if isinstance(arr_obj, (Array, _ZarrV3Array)) or hasattr(
                        arr_obj, "shape"
                    ):
                        return arr_obj, root
            except Exception as exc:
                errors.append(f"zarr reader failed: {exc}")

        if meta and int(meta.get("zarr_format", 0) or 0) == 3:
            arr_dir = meta_path or path
            try:
                arr_obj = _ZarrV3Array(arr_dir, meta)
                root_meta = self._load_zarr_json(path) or {}
                root_proxy = types.SimpleNamespace(
                    attrs=root_meta.get("attributes", {}) or {}
                )
                return arr_obj, root_proxy
            except Exception as exc:
                errors.append(f"v3 fallback failed: {exc}")

        msg = "Could not open Zarr store."
        if errors:
            msg += " " + "; ".join(errors)
        raise RuntimeError(msg)

    def _read_zarr(self, path: Path) -> _VolumeData:
        """Robust Zarr reader supporting v2/v3 stores and slice-mode fallback."""
        arr_obj, root_obj = self._open_zarr_array(path)

        shape = tuple(int(x) for x in getattr(arr_obj, "shape", ()))
        if not shape or len(shape) < 3:
            raise RuntimeError(f"Zarr array must be at least 3D. Got: {shape}")
        dtype = np.dtype(getattr(arr_obj, "dtype", np.float32))
        is_label_volume = self._is_label_volume(dtype, arr_obj, path)

        zyx_axes, fixed_idx = self._zarr_axis_info(arr_obj)
        shape_zyx = (
            int(shape[zyx_axes[0]]),
            int(shape[zyx_axes[1]]),
            int(shape[zyx_axes[2]]),
        )
        spacing = self._extract_zarr_spacing(root_obj or arr_obj, arr_obj)
        sample_minmax = self._sample_zarr_minmax(arr_obj, zyx_axes, fixed_idx)

        total_voxels = int(np.prod(shape_zyx))
        itemsize = max(1, dtype.itemsize)
        bytes_needed = total_voxels * itemsize
        avail_ram = self._available_memory_bytes()

        # Heuristic: prefer full in-memory load when it will comfortably fit
        # (under configured threshold and <50% of available RAM).
        use_slice_mode = False
        if bytes_needed <= 0:
            use_slice_mode = False
        else:
            over_threshold = (
                AUTO_OUT_OF_CORE_MB > 0
                and (bytes_needed / 1024**2) > AUTO_OUT_OF_CORE_MB
            )
            over_memory = avail_ram > 0 and bytes_needed >= (avail_ram * 0.5)
            use_slice_mode = over_threshold or over_memory

        logger.info(
            "Zarr Load Strategy: %s (Size: %.2f MB, Shape: %s)",
            "Slice Mode" if use_slice_mode else "Full Load",
            bytes_needed / 1024**2,
            shape_zyx,
        )

        value_range = sample_minmax
        if value_range is None and is_label_volume:
            try:
                value_range = self._dtype_value_range(dtype)
            except Exception:
                value_range = None

        if use_slice_mode:
            loader = _ZarrSliceLoader(arr_obj, zyx_axes, fixed_idx)
            return self._make_slice_volume_data(
                loader,
                spacing=spacing,
                value_range=value_range,
                is_grayscale=True,
                is_label_map=is_label_volume,
            )

        try:
            vol = self._zarr_to_numpy_zyx(arr_obj, zyx_axes, fixed_idx)
            if not is_label_volume:
                vol = self._normalize_to_float01(vol)
            return _VolumeData(
                array=vol,
                spacing=spacing,
                vmin=float(vol.min()),
                vmax=float(vol.max()),
                is_grayscale=vol.ndim == 3,
                is_out_of_core=False,
                volume_shape=shape_zyx,
                is_label_map=is_label_volume,
            )
        except (MemoryError, RuntimeError) as exc:
            logger.warning(
                "In-memory Zarr load failed (%s), falling back to slice mode.", exc
            )
            loader = _ZarrSliceLoader(arr_obj, zyx_axes, fixed_idx)
            return self._make_slice_volume_data(
                loader,
                spacing=spacing,
                value_range=value_range,
                is_grayscale=True,
                is_label_map=is_label_volume,
            )

    def _open_tiff_memmap(self, path: Path) -> Optional[np.ndarray]:
        try:
            import tifffile  # type: ignore
        except Exception:
            return None
        try:
            try:
                reader = tifffile.memmap(str(path), mode="r+")
            except PermissionError:
                reader = tifffile.memmap(str(path), mode="r")
        except Exception:
            return None
        if reader.ndim not in (3, 4):
            return None
        if reader.ndim == 4 and reader.shape[-1] == 1:
            reader = reader[..., 0]
        if reader.ndim != 3:
            return None
        if not reader.flags.c_contiguous:
            reader = np.ascontiguousarray(reader)
        return reader

    def _make_slice_volume_data(
        self,
        loader: _BaseSliceLoader,
        spacing: Optional[tuple[float, float, float]] = None,
        value_range: Optional[tuple[float, float]] = None,
        is_grayscale: bool = True,
        is_label_map: bool = False,
    ) -> _VolumeData:
        shape = loader.shape()
        dtype = loader.dtype()
        if value_range is not None:
            vmin, vmax = value_range
        else:
            vmin, vmax = self._dtype_value_range(dtype)
        return _VolumeData(
            array=None,
            spacing=spacing,
            vmin=vmin,
            vmax=vmax,
            is_grayscale=is_grayscale,
            is_out_of_core=True,
            slice_mode=True,
            slice_loader=loader,
            slice_axis=0,
            volume_shape=shape,
            is_label_map=is_label_map,
        )

    def _initial_slice_index_for_loader(
        self, loader: Optional[_BaseSliceLoader]
    ) -> int:
        """Estimate a non-empty starting slice for sparse label volumes."""
        if loader is None:
            return 0
        try:
            arr = getattr(loader, "_arr", None)
            z_axis = getattr(loader, "_zyx_axes", (0, 1, 2))[0]
            if isinstance(arr, _ZarrV3Array) and z_axis == 0:
                return arr.first_nonempty_index(axis=0)
        except Exception:
            return 0
        return 0

    def _probe_tiff_metadata(
        self, path: Path
    ) -> Optional[tuple[tuple[int, int, int], np.dtype]]:
        try:
            import tifffile  # type: ignore
        except Exception:
            return None
        try:
            with tifffile.TiffFile(str(path)) as tif:
                series = tif.series[0]
                shape = tuple(int(x) for x in series.shape)
                dtype = np.dtype(series.dtype)
        except Exception:
            return None
        if len(shape) == 2:
            shape = (1, shape[0], shape[1])
        elif len(shape) > 3:
            shape = shape[:3]
        return shape, dtype

    def _should_use_slice_mode(
        self, shape: tuple[int, int, int], dtype: np.dtype
    ) -> bool:
        if not shape or len(shape) < 3:
            return False
        total_voxels = int(shape[0]) * int(shape[1]) * int(shape[2])
        itemsize = max(1, np.dtype(dtype).itemsize)
        size_bytes = total_voxels * itemsize
        if MAX_VOLUME_VOXELS > 0 and total_voxels > MAX_VOLUME_VOXELS:
            return True
        if SLICE_MODE_BYTES > 0 and size_bytes >= SLICE_MODE_BYTES:
            return True
        available = self._available_memory_bytes()
        if available > 0 and size_bytes >= available * 0.8:
            return True
        return False

    def _dtype_value_range(self, dtype: np.dtype) -> tuple[float, float]:
        if np.issubdtype(dtype, np.bool_):
            return 0.0, 1.0
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return float(info.min), float(info.max)
        if np.issubdtype(dtype, np.floating):
            return 0.0, 1.0
        return 0.0, 1.0

    def _convert_frame_to_plane(self, frame: np.ndarray, dtype: np.dtype) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            gray = np.dot(rgb, [0.299, 0.587, 0.114])
            if np.issubdtype(dtype, np.bool_):
                gray = (gray > 0.5).astype(dtype)
            elif np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                gray = np.clip(np.round(gray), info.min, info.max).astype(dtype)
            else:
                gray = gray.astype(dtype)
            return gray
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        return arr

    def _select_zarr_array(self, obj):
        """Smart selection of the image array from a Zarr group."""
        try:
            from zarr.hierarchy import Group
            from zarr.core import Array
        except ImportError:
            return obj

        if isinstance(obj, Array):
            return obj

        if isinstance(obj, Group):
            # Priority 1: OME-NGFF (multiscales) - Pick level 0 (highest res)
            if "multiscales" in obj.attrs:
                try:
                    datasets = obj.attrs["multiscales"][0]["datasets"]
                    path = datasets[0]["path"]
                    return obj[path]
                except (IndexError, KeyError):
                    pass

            # Priority 2: Common convention keys
            for key in ["0", "data", "image", "volume"]:
                if key in obj:
                    item = obj[key]
                    if isinstance(item, Array) and item.ndim >= 3:
                        return item
                    if isinstance(item, Group):
                        # Recurse once if we found a 'data' group
                        return self._select_zarr_array(item)

        return obj

    def _extract_zarr_spacing(self, obj, arr) -> Optional[tuple[float, float, float]]:
        """Attempt to read spacing from zarr attrs or OME-Zarr multiscales."""
        # Direct attrs on array
        for source in (getattr(arr, "attrs", None), getattr(obj, "attrs", None)):
            if not source:
                continue
            for key in ("spacing", "voxel_size", "voxel_spacing"):
                try:
                    candidate = source.get(key)
                    if candidate is not None and len(candidate) >= 3:
                        return (
                            float(candidate[0]),
                            float(candidate[1]),
                            float(candidate[2]),
                        )
                except Exception:
                    continue

        # OME-Zarr multiscales
        try:
            multiscales = getattr(obj, "attrs", {}).get("multiscales")
            if multiscales and isinstance(multiscales, (list, tuple)):
                meta = multiscales[0]
                axes = [
                    a.get("name", "").lower() if isinstance(a, dict) else str(a).lower()
                    for a in meta.get("axes", [])
                ]
                datasets = meta.get("datasets", [])
                if datasets:
                    transforms = datasets[0].get("coordinateTransformations", [])
                    for tf in transforms:
                        if tf.get("type") == "scale":
                            scale = tf.get("scale", [])
                            if axes and scale and len(scale) == len(axes):
                                axis_map = {ax: i for i, ax in enumerate(axes)}
                                try:
                                    return (
                                        float(scale[axis_map.get("z", -1)]),
                                        float(scale[axis_map.get("y", -1)]),
                                        float(scale[axis_map.get("x", -1)]),
                                    )
                                except Exception:
                                    pass
                            if scale and len(scale) >= 3:
                                return (
                                    float(scale[-3]),
                                    float(scale[-2]),
                                    float(scale[-1]),
                                )
        except Exception:
            pass
        return None

    def _normalize_volume_selection(self, path: Path) -> Path:
        """If user picks a file inside a Zarr store, return the store root."""
        try:
            p = path
            if p.is_file():
                if p.name.lower() in ("zarr.json", ".zgroup"):
                    return p.parent
                if (p.parent / ".zarray").exists():
                    return p.parent
                if (p.parent / "data" / ".zarray").exists() or (
                    p.parent / "data" / "zarr.json"
                ).exists():
                    return p.parent / "data"
            # Walk up a couple of levels to find a .zarr root
            cur = p
            for _ in range(3):
                if (
                    cur.name.lower().endswith(".zarr")
                    or (cur / ".zarray").exists()
                    or (cur / "zarr.json").exists()
                    or (cur / ".zgroup").exists()
                ):
                    return cur
                if (cur / "data" / ".zarray").exists() or (
                    cur / "data" / "zarr.json"
                ).exists():
                    return cur / "data"
                cur = cur.parent
        except Exception:
            pass
        return path

    def _zarr_axis_info(self, arr) -> tuple[tuple[int, int, int], dict[int, int]]:
        """
        Infer (Z, Y, X) axis indices.
        Returns:
            zyx_axes: tuple of (z_index, y_index, x_index)
            fixed_indices: dict {axis_index: default_value} for non-spatial axes (Time/Channel)
        """
        ndim = arr.ndim

        # Default to last 3 dimensions as Z, Y, X
        z_ix, y_ix, x_ix = ndim - 3, ndim - 2, ndim - 1

        # Try to find OME-Zarr axis names
        axes_meta = None
        if hasattr(arr, "attrs") and "multiscales" in arr.attrs:
            try:
                axes_meta = arr.attrs["multiscales"][0].get("axes")
            except Exception:
                pass

        # If we found metadata names, map them
        if axes_meta and len(axes_meta) == ndim:
            names = [
                x["name"].lower() if isinstance(x, dict) else x.lower()
                for x in axes_meta
            ]
            if "z" in names and "y" in names and "x" in names:
                z_ix = names.index("z")
                y_ix = names.index("y")
                x_ix = names.index("x")

        # Handle edge case: 2D array (treat as 1 slice Z)
        if ndim == 2:
            # We can't really handle 2D in a 3D viewer easily without faking Z
            # This assumes the data will be reshaped upstream or handled by loader
            return (0, 0, 1), {}

        # Identify "Extra" axes (Time, Channel)
        # We usually fix them to index 0 (first timepoint, first channel)
        fixed_indices = {}
        for i in range(ndim):
            if i not in (z_ix, y_ix, x_ix):
                # Default to the middle of the range? No, usually index 0 is safer.
                fixed_indices[i] = 0

        return (z_ix, y_ix, x_ix), fixed_indices

    def _zarr_to_numpy_zyx(
        self,
        arr,
        zyx_axes: tuple[int, int, int],
        fixed_indices: dict[int, int],
    ) -> np.ndarray:
        """Convert a zarr array to numpy in (Z, Y, X) order, selecting fixed indices for other axes."""
        slicer: list[object] = []
        keep_axes: list[int] = []
        for dim in range(len(getattr(arr, "shape", ()))):
            if dim in zyx_axes:
                slicer.append(slice(None))
                keep_axes.append(dim)
            else:
                slicer.append(fixed_indices.get(dim, 0))
        arr_sel = np.asarray(arr[tuple(slicer)])
        axis_map = {orig: idx for idx, orig in enumerate(keep_axes)}
        order = [
            axis_map.get(zyx_axes[0], 0),
            axis_map.get(zyx_axes[1], 1),
            axis_map.get(zyx_axes[2], 2),
        ]
        if order != [0, 1, 2]:
            arr_sel = np.moveaxis(arr_sel, order, [0, 1, 2])
        while arr_sel.ndim > 3:
            arr_sel = np.take(arr_sel, indices=0, axis=-1)
        return arr_sel

    def _open_zarr_store(self, path: Path, zarr_mod):
        """Robustly open Zarr path, handling consolidated metadata automatically."""
        path_str = str(path)

        # Attempt 1: Consolidated (Optimized for OME-Zarr)
        try:
            return zarr_mod.open_consolidated(path_str, mode="r")
        except Exception:
            pass

        # Attempt 2: Standard Group/Array open
        try:
            return zarr_mod.open(path_str, mode="r")
        except Exception:
            pass

        # Attempt 3: Check for nested 'data' folder (common in some exports)
        if (path / "data").exists():
            try:
                return zarr_mod.open(str(path / "data"), mode="r")
            except Exception:
                pass

        return None

    def _vtk_image_to_numpy(self, vtk_img) -> np.ndarray:
        from vtkmodules.util.numpy_support import vtk_to_numpy

        dims = vtk_img.GetDimensions()  # (x, y, z)
        scalars = vtk_img.GetPointData().GetScalars()
        if scalars is None:
            raise RuntimeError("No scalar data in volume.")
        arr = vtk_to_numpy(scalars)
        # VTK stores as x-fastest; reshape and permute to (Z, Y, X)
        arr = arr.reshape(dims[2], dims[1], dims[0])
        return arr

    def _normalize_to_float01(self, vol: np.ndarray) -> np.ndarray:
        if np.issubdtype(vol.dtype, np.integer):
            vmin = float(vol.min())
            vmax = float(vol.max())
            if vmax > vmin:
                vol = (vol.astype(np.float32) - vmin) / (vmax - vmin)
            else:
                vol = np.zeros_like(vol, dtype=np.float32)
        else:
            vol = vol.astype(np.float32)
            vol = np.clip(vol, 0.0, 1.0)
        return vol

    def _read_dicom_series(
        self, directory: Path
    ) -> tuple[np.ndarray, Optional[tuple[float, float, float]]]:
        # First try GDCM (if present), then fallback to VTK's basic reader
        reader = None
        if _HAS_GDCM:
            try:
                reader = vtkGDCMImageReader()  # type: ignore[name-defined]
                reader.SetDirectoryName(str(directory))
                reader.Update()
            except Exception:
                reader = None
        if reader is None:
            try:
                from vtkmodules.vtkIOImage import vtkDICOMImageReader
            except Exception as exc:
                raise RuntimeError(
                    "VTK DICOM reader is not available in this build."
                ) from exc
            reader = vtkDICOMImageReader()
            reader.SetDirectoryName(str(directory))
            reader.Update()

        vtk_img = reader.GetOutput()
        vol = self._vtk_image_to_numpy(vtk_img)
        s = vtk_img.GetSpacing()
        spacing = (s[0], s[1], s[2])
        vol = self._normalize_to_float01(vol)
        return vol, spacing

    def _update_opacity(self):
        if not getattr(self, "_has_volume", False):
            return
        # Adjust overall opacity scaling via unit distance
        val = self.opacity_slider.value() / 100.0
        # Smaller unit distance -> denser appearance
        unit = max(0.001, 2.0 * (1.0 - val) + 0.05)
        self.property.SetScalarOpacityUnitDistance(unit)
        try:
            self._ensure_volume_actor_added()
        except Exception:
            pass
        self.vtk_widget.GetRenderWindow().Render()

    def _update_shading(self):
        if not getattr(self, "_has_volume", False):
            return
        if self.shade_checkbox.isChecked():
            self.property.ShadeOn()
        else:
            self.property.ShadeOff()
        self.vtk_widget.GetRenderWindow().Render()

    def _configure_plane_controls(self):
        for axis, control in self._slice_plane_controls.items():
            total = self._plane_total(axis)
            available = total > 0 and axis in self._slice_plane_widgets
            control.slider.blockSignals(True)
            if available:
                control.slider.setRange(0, max(0, total - 1))
                control.slider.setValue(total // 2)
            else:
                control.slider.setRange(0, 0)
                control.slider.setValue(0)
            control.slider.blockSignals(False)
            control.slider.setEnabled(available)
            control.checkbox.blockSignals(True)
            control.checkbox.setEnabled(available)
            if not available:
                control.checkbox.setChecked(False)
                if axis in self._slice_plane_widgets:
                    self._slice_plane_widgets[axis].SetEnabled(False)
            control.checkbox.blockSignals(False)
            self._update_plane_label(axis, control.slider.value(), total)
            if available:
                self._apply_plane_slice(
                    axis, control.slider.value(), render=False, update_slider=False
                )

    def _plane_property(self, axis: int) -> vtkProperty:
        prop = vtkProperty()
        color = PLANE_COLORS.get(axis, (0.8, 0.8, 0.2))
        prop.SetColor(*color)
        prop.SetOpacity(0.65)
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)
        prop.SetInterpolationToFlat()
        return prop

    def _teardown_slice_planes(self):
        for widget in self._slice_plane_widgets.values():
            try:
                widget.SetEnabled(False)
            except Exception:
                pass
            try:
                widget.SetInteractor(None)
            except Exception:
                pass
        self._slice_plane_widgets.clear()

    def _setup_slice_plane(self):
        self._teardown_slice_planes()
        if not _HAS_PLANE_WIDGET or self._vtk_img is None:
            return
        for axis, _, orientation_name in PLANE_DEFS:
            try:
                widget = vtkImagePlaneWidget()
            except Exception:
                continue
            widget.SetInteractor(self.interactor)
            widget.SetInputData(self._vtk_img)
            orient_fn = getattr(widget, orientation_name, None)
            if callable(orient_fn):
                orient_fn()
            widget.TextureInterpolateOn()
            widget.DisplayTextOn()
            widget.SetEnabled(False)
            if getattr(self, "renderer", None) is not None:
                try:
                    widget.SetDefaultRenderer(self.renderer)
                except Exception:
                    pass
            try:
                widget.AlwaysOnTopOn()
            except Exception:
                pass
            widget.SetPlaneProperty(self._plane_property(axis))
            try:
                widget.SetResliceInterpolateToLinear()
            except Exception:
                pass
            self._slice_plane_widgets[axis] = widget

    def _plane_total(self, axis: int) -> int:
        if axis < len(self._volume_shape):
            return self._volume_shape[axis]
        return 0

    def _plane_is_available(self, axis: int) -> bool:
        return axis in self._slice_plane_widgets and self._plane_total(axis) > 0

    def _clear_slice_clipping_planes(self):
        if not getattr(self, "mapper", None):
            self._slice_clipping_planes.clear()
            return
        for plane in self._slice_clipping_planes.values():
            try:
                self.mapper.RemoveClippingPlane(plane)
            except Exception:
                pass
        self._slice_clipping_planes.clear()

    def _update_plane_label(self, axis: int, index: int, total: int):
        control = self._slice_plane_controls.get(axis)
        if control is None:
            return
        if total <= 0:
            control.label.setText(f"{control.name}: -/-")
        else:
            control.label.setText(f"{control.name}: {index + 1}/{total}")

    def _apply_plane_slice(
        self, axis: int, index: int, render: bool = True, update_slider: bool = True
    ):
        control = self._slice_plane_controls.get(axis)
        if control is None:
            return
        total = self._plane_total(axis)
        if total <= 0:
            return
        sanitized = max(0, min(index, total - 1))
        if update_slider:
            control.slider.blockSignals(True)
            control.slider.setValue(sanitized)
            control.slider.blockSignals(False)
        self._update_plane_label(axis, sanitized, total)
        widget = self._slice_plane_widgets.get(axis)
        if widget:
            widget.SetSliceIndex(sanitized)
            widget.SetEnabled(control.checkbox.isChecked())
        self._update_clipping_for_plane(axis, control.checkbox.isChecked())
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    def _on_plane_slider_changed(self, axis: int, value: int):
        control = self._slice_plane_controls.get(axis)
        if control and not control.checkbox.isChecked():
            control.checkbox.blockSignals(True)
            control.checkbox.setChecked(True)
            control.checkbox.blockSignals(False)
            self._on_plane_checkbox_changed(axis)
            return
        self._apply_plane_slice(axis, value)

    def _on_plane_checkbox_changed(self, axis: int, *_):
        widget = self._slice_plane_widgets.get(axis)
        control = self._slice_plane_controls.get(axis)
        if widget is None or control is None:
            return
        enabled = control.checkbox.isChecked()
        widget.SetEnabled(enabled)
        if enabled:
            self._apply_plane_slice(axis, control.slider.value())
        else:
            self._update_clipping_for_plane(axis, False)
            self.vtk_widget.GetRenderWindow().Render()

    def _update_clipping_for_plane(self, axis: int, enabled: bool):
        plane = self._slice_clipping_planes.get(axis)
        if not enabled:
            if plane is not None:
                try:
                    self.mapper.RemoveClippingPlane(plane)
                except Exception:
                    pass
                self._slice_clipping_planes.pop(axis, None)
            return
        widget = self._slice_plane_widgets.get(axis)
        control = self._slice_plane_controls.get(axis)
        if widget is None or control is None:
            return
        if plane is None:
            plane = vtkPlane()
            self._slice_clipping_planes[axis] = plane
            try:
                self.mapper.AddClippingPlane(plane)
            except Exception:
                pass
        origin = widget.GetCenter()
        normal = widget.GetNormal()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

    # -------------------- Interaction helpers --------------------
    def _toggle_wl_mode(self):
        if not getattr(self, "_has_volume", False):
            try:
                self.wl_mode_checkbox.setChecked(False)
            except Exception:
                pass
            self._wl_mode = False
            return
        self._wl_mode = self.wl_mode_checkbox.isChecked()
        # Disable camera interaction while in WL mode to avoid conflicts
        if self._wl_mode:
            self.interactor.SetInteractorStyle(self._style_inactive)
        else:
            self.interactor.SetInteractorStyle(self._style_trackball)

    def _vtk_on_left_press(self, obj, evt):
        if not getattr(self, "_has_volume", False) or not self._wl_mode:
            return
        self._wl_drag = True
        self._wl_last = self.interactor.GetEventPosition()

    def _vtk_on_left_release(self, obj, evt):
        if not getattr(self, "_has_volume", False) or not self._wl_mode:
            return
        self._wl_drag = False

    def _vtk_on_mouse_move(self, obj, evt):
        """Handles mouse movement for both window/level adjustment and point picking."""
        # --- BRANCH 1: Window/Level Drag Interaction ---
        # This logic takes precedence if the user is in W/L mode and dragging.
        if self._wl_mode and self._wl_drag:
            x, y = self.interactor.GetEventPosition()
            last_x, last_y = self._wl_last
            dx = x - last_x
            dy = y - last_y
            self._wl_last = (x, y)

            # Adjust window/level based on mouse delta
            wmin = float(self.min_spin.value())
            wmax = float(self.max_spin.value())
            window = max(1e-6, wmax - wmin)
            level = (wmax + wmin) * 0.5

            # Horizontal movement adjusts window width, vertical adjusts level/center
            new_window = max(1e-6, window * (1.0 + dx * 0.01))
            # Invert dy for natural feel
            new_level = level + (-dy) * (window * 0.005)

            vmin = new_level - new_window * 0.5
            vmax = new_level + new_window * 0.5

            # Clamp the new values to the full data range
            vmin = max(self._vmin, min(vmin, self._vmax - 1e-6))
            vmax = max(vmin + 1e-6, min(vmax, self._vmax))

            # Update the UI widgets without emitting signals to avoid loops
            self.min_spin.blockSignals(True)
            self.max_spin.blockSignals(True)
            self.min_spin.setValue(vmin)
            self.max_spin.setValue(vmax)
            self.min_spin.blockSignals(False)
            self.max_spin.blockSignals(False)

            self._update_transfer_functions()
            self.vtk_widget.GetRenderWindow().Render()

            # End processing for this event; do not proceed to picking logic
            return

        # --- BRANCH 2: Point Picking and Tooltip Interaction ---
        # This runs during normal mouse movement when not dragging in W/L mode.
        # We also prevent picking if the W/L mode checkbox is checked, even if not dragging.
        if self._wl_mode:
            return

        x, y = self.interactor.GetEventPosition()
        self._picker.Pick(x, y, 0, self.renderer)

        actor = self._picker.GetActor()
        point_id = self._picker.GetPointId()

        # Check if we successfully picked a point on a valid point cloud actor
        if (
            point_id > -1
            and hasattr(self, "_point_actors")
            and actor in self._point_actors
        ):
            # To prevent flickering, only update the tooltip if the picked point is new
            if point_id != self._last_picked_id or actor != self._last_picked_actor:
                self._last_picked_id = point_id
                self._last_picked_actor = actor

                polydata = actor.GetMapper().GetInput()

                # Attempt to retrieve the region label data array
                label_array_abstract = polydata.GetPointData().GetAbstractArray(
                    "RegionLabel"
                )

                tooltip_parts = []
                # If the array exists, get the string value for the picked point
                if label_array_abstract:
                    # The array must be safely cast to a vtkStringArray to get its value
                    label_array = vtkStringArray.SafeDownCast(label_array_abstract)
                    if label_array:
                        region_name = label_array.GetValue(point_id)
                        tooltip_parts.append(f"<b>Region:</b> {region_name}")

                # Always add point ID and coordinates to the tooltip
                coords = self._picker.GetPickPosition()
                tooltip_parts.append(f"<b>Point ID:</b> {point_id}")
                tooltip_parts.append(
                    f"<b>Coords:</b> ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
                )

                tooltip_text = "<br>".join(tooltip_parts)

                # Show the rich-text tooltip at the current cursor position
                QtWidgets.QToolTip.showText(
                    QtGui.QCursor.pos(), tooltip_text, self.vtk_widget
                )

        else:
            # If the cursor is not over a point (or moved off a point), hide the tooltip
            if self._last_picked_id != -1:
                QtWidgets.QToolTip.hideText()
                self._last_picked_id = -1
                self._last_picked_actor = None

    def _vtk_on_key_press(self, obj, evt):
        key = self.interactor.GetKeySym().lower()
        if key == "r":
            self._reset_camera()
        elif key == "w":
            self.wl_mode_checkbox.setChecked(not self.wl_mode_checkbox.isChecked())
        elif key == "c":
            self.shade_checkbox.setChecked(not self.shade_checkbox.isChecked())
        elif key == "+":
            self.opacity_slider.setValue(min(100, self.opacity_slider.value() + 5))
        elif key == "-":
            self.opacity_slider.setValue(max(1, self.opacity_slider.value() - 5))

    def _update_interpolation(self):
        if not getattr(self, "_has_volume", False):
            return
        if self.interp_combo.currentText() == "Nearest":
            self.property.SetInterpolationTypeToNearest()
        else:
            self.property.SetInterpolationTypeToLinear()
        self.vtk_widget.GetRenderWindow().Render()

    def _update_spacing(self):
        if not getattr(self, "_has_volume", False):
            return
        try:
            sx = float(self.spacing_x.value())
            sy = float(self.spacing_y.value())
            sz = float(self.spacing_z.value())
            self._vtk_img.SetSpacing(sx, sy, sz)
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass

    def _reset_camera(self):
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _save_snapshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Snapshot",
            str(self._path.with_suffix(".png")),
            "PNG Files (*.png)",
        )
        if not path:
            return
        w2i = vtkWindowToImageFilter()
        w2i.SetInput(self.vtk_widget.GetRenderWindow())
        w2i.Update()
        writer = vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Saved: {Path(path).name}")

    def _on_window_changed(self):
        # Ensure min <= max
        vmin = min(self.min_spin.value(), self.max_spin.value())
        vmax = max(self.min_spin.value(), self.max_spin.value())
        if vmin != self.min_spin.value():
            self.min_spin.blockSignals(True)
            self.min_spin.setValue(vmin)
            self.min_spin.blockSignals(False)
        if vmax != self.max_spin.value():
            self.max_spin.blockSignals(True)
            self.max_spin.setValue(vmax)
            self.max_spin.blockSignals(False)
        self._update_transfer_functions()
        self.vtk_widget.GetRenderWindow().Render()

    def _auto_window(self):
        if self._slice_mode:
            self._slice_auto_window()
            return
        if not getattr(self, "_has_volume", False) or self._volume_np is None:
            return
        # 2-98 percentile auto window on current volume
        vol = self._volume_np
        p2, p98 = float(np.percentile(vol, 2)), float(np.percentile(vol, 98))
        if p98 <= p2:
            p2, p98 = float(vol.min()), float(vol.max())
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(p2)
        self.max_spin.setValue(p98)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        self._update_transfer_functions()
        self.vtk_widget.GetRenderWindow().Render()

    def _update_transfer_functions(self):
        if self._slice_mode:
            return
        if (
            not getattr(self, "_has_volume", False)
            or self._opacity_tf is None
            or self._color_tf is None
        ):
            return
        is_label = getattr(self, "_label_volume_active", False)
        # Build opacity and color TF based on window and colormap
        vmin = float(self.min_spin.value()) if hasattr(self, "min_spin") else self._vmin
        vmax = float(self.max_spin.value()) if hasattr(self, "max_spin") else self._vmax
        if vmax <= vmin:
            vmax = vmin + 1e-3

        self._opacity_tf.RemoveAllPoints()
        label_values: list[int] = []
        if is_label:
            label_values = self._collect_label_values()
            added_any = False
            for val in label_values:
                opacity = 0.0 if val == 0 else 0.9
                self._opacity_tf.AddPoint(float(val), opacity)
                added_any = True
            if not added_any:
                self._opacity_tf.AddPoint(vmin, 0.0)
                self._opacity_tf.AddPoint(vmax, 0.9)
        else:
            # Opacity: ramp from vmin to vmax
            self._opacity_tf.AddPoint(vmin, 0.0)
            self._opacity_tf.AddPoint((vmin + vmax) * 0.5, 0.1)
            self._opacity_tf.AddPoint(vmax, 0.9)

        # Color map
        cmap = (
            self.cmap_combo.currentText()
            if hasattr(self, "cmap_combo")
            else "Grayscale"
        )
        self._color_tf.RemoveAllPoints()
        if cmap == "Allen Atlas" and is_label:
            self._apply_allen_color_tf(label_values, vmin, vmax)
        elif cmap == "Grayscale":
            self._color_tf.AddRGBPoint(vmin, 0.0, 0.0, 0.0)
            self._color_tf.AddRGBPoint(vmax, 1.0, 1.0, 1.0)
        elif cmap == "Invert Gray":
            self._color_tf.AddRGBPoint(vmin, 1.0, 1.0, 1.0)
            self._color_tf.AddRGBPoint(vmax, 0.0, 0.0, 0.0)
        elif cmap == "Hot":
            # Rough "hot" ramp: black -> red -> yellow -> white
            self._color_tf.AddRGBPoint(vmin, 0.0, 0.0, 0.0)
            self._color_tf.AddRGBPoint(vmin + (vmax - vmin) * 0.33, 1.0, 0.0, 0.0)
            self._color_tf.AddRGBPoint(vmin + (vmax - vmin) * 0.66, 1.0, 1.0, 0.0)
            self._color_tf.AddRGBPoint(vmax, 1.0, 1.0, 1.0)

    def _update_blend_mode(self):
        mode = (
            self.blend_combo.currentText()
            if hasattr(self, "blend_combo")
            else "Composite"
        )
        try:
            if mode == "MIP-Max":
                self.mapper.SetBlendModeToMaximumIntensity()
            elif mode == "MIP-Min":
                self.mapper.SetBlendModeToMinimumIntensity()
            elif mode == "Additive":
                self.mapper.SetBlendModeToAdditive()
            else:
                self.mapper.SetBlendModeToComposite()
        except Exception:
            # Some mappers may not support all modes; ignore
            pass
        self.vtk_widget.GetRenderWindow().Render()

    def _maybe_align_point_cloud_to_volume(self, pts: np.ndarray) -> np.ndarray:
        """Reorder axes/offset grid-derived points to match the active volume."""
        if pts is None or len(pts) == 0:
            return pts
        if not getattr(self, "_has_volume", False):
            return pts
        if not getattr(self, "_volume_shape", None) or len(self._volume_shape) < 3:
            return pts
        if any(ax <= 0 for ax in self._volume_shape[:3]):
            return pts

        spacing = (
            np.array(self._vtk_img.GetSpacing(), dtype=float)
            if getattr(self, "_vtk_img", None)
            else np.array([1.0, 1.0, 1.0])
        )
        if spacing.shape != (3,):
            spacing = np.array([1.0, 1.0, 1.0])
        expected_counts = np.array(
            [self._volume_shape[2], self._volume_shape[1], self._volume_shape[0]],
            dtype=float,
        )
        expected_world = expected_counts * spacing

        best_perm: tuple[int, int, int] | None = None
        best_err: float | None = None
        for perm in itertools.permutations((0, 1, 2)):
            perm_pts = pts[:, perm]
            span = perm_pts.max(axis=0) - perm_pts.min(axis=0)
            if not np.all(np.isfinite(span)):
                continue
            err_pix = np.mean(np.abs(span - expected_counts) / (expected_counts + 1e-6))
            err_world = np.mean(np.abs(span - expected_world) / (expected_world + 1e-6))
            err = min(err_pix, err_world)
            if best_err is None or err < best_err:
                best_err = err
                best_perm = perm

        if best_perm is None or best_err is None or best_err > 0.25:
            return pts

        aligned = np.array(pts[:, best_perm], copy=True)
        origin = (
            np.array(self._vtk_img.GetOrigin(), dtype=float)
            if getattr(self, "_vtk_img", None)
            else np.zeros(3, dtype=float)
        )
        mins = aligned.min(axis=0)
        offset = None
        if np.all(np.abs(mins - origin) <= spacing * 1.5):
            offset = origin - mins
        elif np.all(np.abs(mins - (origin + spacing * 0.5)) <= spacing * 1.5):
            offset = origin + spacing * 0.5 - mins
        if offset is not None:
            aligned += offset

        if best_perm != (0, 1, 2) or offset is not None:
            logger.info(
                "Auto-aligned point cloud to volume (perm=%s, err=%.3f, offset=%s)",
                best_perm,
                best_err,
                None if offset is None else tuple(float(x) for x in offset),
            )
        return aligned

    # -------------------- Point cloud support --------------------
    def _load_point_cloud_folder(self):
        start_dir = (
            str(self._path.parent)
            if getattr(self, "_path", None) and self._path.exists()
            else "."
        )
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Open Point Cloud Folder",
            start_dir,
        )
        if not folder:
            return
        directory = Path(folder)
        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            QtWidgets.QMessageBox.warning(
                self,
                "Point Cloud",
                "No CSV point clouds found in that directory.",
            )
            return
        self._load_point_cloud_files([str(path) for path in csv_files])

    def _load_point_cloud_dialog(self):
        start_dir = (
            str(self._path.parent)
            if getattr(self, "_path", None) and self._path.exists()
            else "."
        )
        files = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open Point Cloud Files",
            start_dir,
            "Point Clouds (*.csv *.CSV *.xyz *.XYZ *.ply *.PLY);;All files (*.*)",
        )
        if isinstance(files, tuple):
            selected = files[0]
        else:
            selected = files
        if not selected:
            return
        paths = [str(p) for p in selected if p]
        if not paths:
            return
        self._load_point_cloud_files(paths)

    def _load_point_cloud_files(self, paths: list[str]):
        if not paths:
            return
        self._clear_point_clouds()
        if not getattr(self, "_has_volume", False):
            try:
                self._source_path = Path(paths[0]).expanduser()
            except Exception:
                self._source_path = Path(paths[0])
        combined_bounds = None
        for path in paths:
            lowered = path.lower()
            try:
                if lowered.endswith(".ply"):
                    self._add_point_cloud_ply(path)
                    bounds = (
                        self._point_actors[-1].GetBounds()
                        if self._point_actors
                        else None
                    )
                elif lowered.endswith(".csv") or lowered.endswith(".xyz"):
                    bounds = self._add_point_cloud_csv_or_xyz(path, focus=False)
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Point Cloud",
                        f"Unsupported point cloud format: {path}",
                    )
                    continue
                combined_bounds = self._union_bounds(combined_bounds, bounds)
            except Exception as exc:
                logger.warning("Failed to load point cloud '%s': %s", path, exc)
        if combined_bounds:
            self._focus_on_bounds(combined_bounds)
        self._update_point_sizes()
        self.vtk_widget.GetRenderWindow().Render()
        self._refresh_status_summary()

    def _add_point_cloud_ply(self, path: str):
        try:
            from vtkmodules.vtkIOPLY import vtkPLYReader  # lazy import
        except Exception as exc:  # pragma: no cover - optional module
            raise RuntimeError(
                "VTK PLY reader module is not available. Install a VTK build with IOPLY support."
            ) from exc
        reader = vtkPLYReader()
        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("PLY contains no points")
        # If this is a gaussian-splat PLY, render with oriented glyphs first.
        if self._is_gaussian_splat_ply(path):
            try:
                actor = self._create_gaussian_splat_actor(path, poly)
                if actor is not None:
                    self.renderer.AddActor(actor)
                    self._point_actors.append(actor)
                    if hasattr(self, "show_point_cloud_checkbox"):
                        self.show_point_cloud_checkbox.blockSignals(True)
                        self.show_point_cloud_checkbox.setChecked(True)
                        self.show_point_cloud_checkbox.blockSignals(False)
                        self._point_cloud_visible = True
                    self._focus_on_bounds(actor.GetBounds())
                    self._update_point_controls_visibility()
                    return
            except Exception as exc:
                logger.warning(
                    "Falling back to point renderer for gaussian PLY '%s': %s",
                    path,
                    exc,
                )
        # Ask user for scale factors (default to 1, or volume spacing if available)
        scale = self._prompt_point_scale()
        if scale is not None:
            poly = self._scale_poly_points(poly, scale)
        try:
            pts_np = np.array(
                [poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())],
                dtype=np.float32,
            )
            aligned_pts = self._maybe_align_point_cloud_to_volume(pts_np)
            if aligned_pts.shape == pts_np.shape and not np.allclose(
                aligned_pts, pts_np
            ):
                vpts = vtkPoints()
                vpts.SetNumberOfPoints(len(aligned_pts))
                for i, (x, y, z) in enumerate(aligned_pts):
                    vpts.SetPoint(i, float(x), float(y), float(z))
                poly.SetPoints(vpts)
        except Exception:
            pass
        # Ensure vertices exist without vtkVertexGlyphFilter
        poly2 = self._ensure_vertices(poly)
        self._ensure_ply_point_colors(poly2, path)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly2)
        # Use embedded PLY colors if present
        scalars = None
        try:
            scalars = poly2.GetPointData().GetScalars()
        except Exception:
            scalars = None
        if scalars is not None and scalars.GetNumberOfComponents() >= 3:
            mapper.SetColorModeToDirectScalars()
            mapper.SetScalarModeToUsePointData()
            try:
                mapper.SelectColorArray(scalars.GetName() or "Colors")
            except Exception:
                pass
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(self.point_size_slider.value())
        actor.SetVisibility(self._point_cloud_visible)
        self.renderer.AddActor(actor)
        self._point_actors.append(actor)
        if hasattr(self, "show_point_cloud_checkbox"):
            self.show_point_cloud_checkbox.blockSignals(True)
            self.show_point_cloud_checkbox.setChecked(True)
            self.show_point_cloud_checkbox.blockSignals(False)
            self._point_cloud_visible = True
        self._focus_on_bounds(poly2.GetBounds())
        self._update_point_controls_visibility()

    def _ensure_ply_point_colors(self, poly: vtkPolyData, path: str) -> None:
        """Ensure the provided polydata has RGB(A) scalars, synthesizing them if needed."""
        if poly is None:
            return
        point_data = poly.GetPointData()
        if point_data is None:
            return
        try:
            scalars = point_data.GetScalars()
        except Exception:
            scalars = None
        if scalars is not None and scalars.GetNumberOfComponents() >= 3:
            if scalars.GetNumberOfTuples() == poly.GetNumberOfPoints():
                return
        colors = self._decode_gaussian_ply_colors(path, poly.GetNumberOfPoints())
        if colors is None:
            return
        vtk_colors = numpy_to_vtk(colors, deep=True)
        try:
            vtk_colors.SetNumberOfComponents(colors.shape[1])
        except Exception:
            pass
        vtk_colors.SetName("Colors")
        point_data.SetScalars(vtk_colors)
        try:
            point_data.SetActiveScalars(vtk_colors.GetName())
        except Exception:
            pass

    def _is_gaussian_splat_ply(self, path: str) -> bool:
        required = (
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        )
        data = self._read_gaussian_ply_fields(path, required)
        return data is not None and all(k in data for k in required)

    def _create_gaussian_splat_actor(
        self, path: str, poly: vtkPolyData
    ) -> Optional[vtkActor]:
        """Render gaussian-splat PLYs with oriented, scaled glyphs instead of flat points."""
        required = (
            "x",
            "y",
            "z",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
        )
        field_data = self._read_gaussian_ply_fields(path, required)
        if not field_data:
            return None
        try:
            pts = np.stack(
                [
                    np.asarray(field_data["x"], dtype=np.float32),
                    np.asarray(field_data["y"], dtype=np.float32),
                    np.asarray(field_data["z"], dtype=np.float32),
                ],
                axis=1,
            )
        except Exception:
            return None
        if pts.size == 0:
            return None
        # Apply optional auto-alignment to an active volume
        try:
            aligned = self._maybe_align_point_cloud_to_volume(np.array(pts, copy=True))
            if aligned.shape == pts.shape:
                pts = aligned
        except Exception:
            pass

        try:
            scales = np.stack(
                [
                    np.asarray(field_data["scale_0"], dtype=np.float32),
                    np.asarray(field_data["scale_1"], dtype=np.float32),
                    np.asarray(field_data["scale_2"], dtype=np.float32),
                ],
                axis=1,
            )
            # Gaussian splat scales are stored in log space; exponentiate to get radii.
            scales = np.exp(scales)
            # Guard against degenerate or runaway radii that create rendering artifacts.
            scales = np.clip(scales, 1e-4, 1e4)
        except Exception:
            scales = np.ones((len(pts), 3), dtype=np.float32)

        quaternion_array: Optional[np.ndarray] = None
        orientation_deg: Optional[np.ndarray]
        try:
            quats = np.stack(
                [
                    np.asarray(field_data["rot_0"], dtype=np.float32),
                    np.asarray(field_data["rot_1"], dtype=np.float32),
                    np.asarray(field_data["rot_2"], dtype=np.float32),
                    np.asarray(field_data["rot_3"], dtype=np.float32),
                ],
                axis=1,
            )
            quats = self._normalize_quaternions(quats)
            quaternion_array = np.ascontiguousarray(quats, dtype=np.float32)
            orientation_deg = self._quaternion_to_euler_deg(quaternion_array)
        except Exception:
            quaternion_array = None
            orientation_deg = np.zeros((len(pts), 3), dtype=np.float32)

        colors = self._colors_from_gaussian_field_data(field_data, len(pts))
        if colors is None:
            colors = self._decode_gaussian_ply_colors(path, len(pts))

        vpoints = vtkPoints()
        vpoints.SetData(
            numpy_to_vtk(np.ascontiguousarray(pts, dtype=np.float32), deep=True)
        )

        out = vtkPolyData()
        out.SetPoints(vpoints)
        # Add dummy verts so bounds work even without glyphs
        out = self._ensure_vertices(out)

        scale_arr = numpy_to_vtk(
            np.ascontiguousarray(scales, dtype=np.float32), deep=True
        )
        scale_arr.SetNumberOfComponents(3)
        scale_arr.SetName("Scale")
        out.GetPointData().AddArray(scale_arr)

        orient_arr = numpy_to_vtk(
            np.ascontiguousarray(
                orientation_deg
                if orientation_deg is not None
                else np.zeros((len(pts), 3), dtype=np.float32),
                dtype=np.float32,
            ),
            deep=True,
        )
        orient_arr.SetNumberOfComponents(3)
        orient_arr.SetName("Orientation")
        out.GetPointData().AddArray(orient_arr)
        if quaternion_array is not None and quaternion_array.size:
            quat_arr = numpy_to_vtk(quaternion_array, deep=True)
            quat_arr.SetNumberOfComponents(4)
            quat_arr.SetName("Quaternion")
            out.GetPointData().AddArray(quat_arr)

        if colors is not None:
            vtk_colors = numpy_to_vtk(colors, deep=True)
            vtk_colors.SetName("Colors")
            try:
                vtk_colors.SetNumberOfComponents(
                    colors.shape[1] if colors.ndim > 1 else 3
                )
            except Exception:
                pass
            out.GetPointData().SetScalars(vtk_colors)

        sphere = vtkSphereSource()
        sphere.SetRadius(1.0)
        sphere.SetThetaResolution(self._gaussian_glyph_res)
        sphere.SetPhiResolution(self._gaussian_glyph_res)

        mapper = vtkGlyph3DMapper()
        mapper.SetInputData(out)
        mapper.SetSourceConnection(sphere.GetOutputPort())
        mapper.SetScaling(True)
        try:
            mapper.SetScaleModeToScaleByComponents()
        except Exception:
            try:
                mapper.SetScaleMode(2)  # 2 == SCALE_BY_COMPONENTS
            except Exception:
                pass
        mapper.SetScaleArray("Scale")
        mapper.SetOrientationArray("Orientation")
        try:
            mapper.SetOrientationModeToRotation()
        except Exception:
            pass
        if quaternion_array is not None and quaternion_array.size:
            applied_quaternion = False
            if hasattr(mapper, "SetOrientationModeToQuaternion"):
                try:
                    mapper.SetOrientationArray("Quaternion")
                    mapper.SetOrientationModeToQuaternion()
                    applied_quaternion = True
                except Exception:
                    applied_quaternion = False
            if not applied_quaternion:
                mapper.SetOrientationArray("Orientation")
                try:
                    mapper.SetOrientationModeToRotation()
                except Exception:
                    pass
        if colors is not None:
            mapper.ScalarVisibilityOn()
            mapper.SetScalarModeToUsePointFieldData()
            try:
                mapper.SetColorModeToDirectScalars()
            except Exception:
                pass
            mapper.SelectColorArray("Colors")
        else:
            mapper.ScalarVisibilityOff()

        actor = vtkActor()
        actor.SetMapper(mapper)
        try:
            prop = actor.GetProperty()
            # Gaussian splats should be emissive; avoid harsh specular highlights.
            prop.ShadingOff()
            prop.SetInterpolationToFlat()
            prop.SetAmbient(1.0)
            prop.SetDiffuse(0.0)
            prop.SetSpecular(0.0)
        except Exception:
            pass
        actor.SetVisibility(self._point_cloud_visible)
        self._register_gaussian_actor(actor, scales, colors, sphere, out, mapper)
        return actor

    def _register_gaussian_actor(
        self,
        actor: vtkActor,
        base_scales: np.ndarray,
        base_colors: Optional[np.ndarray],
        sphere: object,
        poly: vtkPolyData,
        mapper: vtkGlyph3DMapper,
    ) -> None:
        if not hasattr(self, "_gaussian_actor_data"):
            self._gaussian_actor_data = {}
        self._gaussian_actor_data[id(actor)] = {
            "actor": actor,
            "base_scales": np.array(base_scales, copy=True),
            "base_colors": None
            if base_colors is None
            else np.array(base_colors, copy=True),
            "sphere": sphere,
            "poly": poly,
            "mapper": mapper,
        }
        self._apply_gaussian_scale_to_actor(actor)
        self._apply_gaussian_opacity_to_actor(actor)
        self._apply_gaussian_resolution_to_actor(actor)
        self._update_gaussian_controls_visibility()

    def _update_gaussian_controls_visibility(self) -> None:
        if hasattr(self, "gaussian_group"):
            has_gaussian = bool(getattr(self, "_gaussian_actor_data", {}))
            self.gaussian_group.setEnabled(has_gaussian)

    def _apply_gaussian_scale_to_actor(
        self, actor: vtkActor, multiplier: Optional[float] = None
    ) -> None:
        if not getattr(self, "_gaussian_actor_data", None):
            return
        data = self._gaussian_actor_data.get(id(actor))
        if not data:
            return
        m = float(self._gaussian_scale_mult if multiplier is None else multiplier)
        base = np.asarray(data.get("base_scales"))
        poly: vtkPolyData = data.get("poly")  # type: ignore[assignment]
        if base.size == 0 or poly is None:
            return
        scaled = np.ascontiguousarray(base * m, dtype=np.float32)
        v = numpy_to_vtk(scaled, deep=True)
        v.SetName("Scale")
        v.SetNumberOfComponents(3)
        pd = poly.GetPointData()
        try:
            pd.RemoveArray("Scale")
        except Exception:
            pass
        pd.AddArray(v)
        try:
            pd.SetActiveVectors("Scale")
        except Exception:
            pass
        poly.Modified()
        try:
            mapper = data.get("mapper")
            if mapper is not None:
                mapper.Modified()
        except Exception:
            pass

    def _apply_gaussian_opacity_to_actor(
        self, actor: vtkActor, multiplier: Optional[float] = None
    ) -> None:
        if not getattr(self, "_gaussian_actor_data", None):
            return
        data = self._gaussian_actor_data.get(id(actor))
        if not data:
            return
        base_colors = data.get("base_colors")
        if base_colors is None:
            return
        m = float(self._gaussian_opacity_mult if multiplier is None else multiplier)
        colors = np.array(base_colors, copy=True)
        if colors.ndim == 1:
            colors = colors.reshape(-1, 1)
        if colors.shape[1] >= 4:
            alpha = np.clip(colors[:, 3].astype(np.float32) * m, 0.0, 255.0)
            colors[:, 3] = alpha.astype(np.uint8)
        else:
            colors = np.clip(colors.astype(np.float32) * m, 0.0, 255.0).astype(np.uint8)
        poly: vtkPolyData = data.get("poly")  # type: ignore[assignment]
        vtk_colors = numpy_to_vtk(np.ascontiguousarray(colors), deep=True)
        vtk_colors.SetName("Colors")
        try:
            vtk_colors.SetNumberOfComponents(colors.shape[1] if colors.ndim > 1 else 3)
        except Exception:
            pass
        pd = poly.GetPointData()
        pd.SetScalars(vtk_colors)
        try:
            pd.SetActiveScalars("Colors")
        except Exception:
            pass
        poly.Modified()
        try:
            mapper = data.get("mapper")
            if mapper is not None:
                mapper.Modified()
        except Exception:
            pass

    def _apply_gaussian_resolution_to_actor(
        self, actor: vtkActor, res: Optional[int] = None
    ) -> None:
        if not getattr(self, "_gaussian_actor_data", None):
            return
        data = self._gaussian_actor_data.get(id(actor))
        if not data:
            return
        sphere = data.get("sphere")
        if sphere is None:
            return
        r = int(self._gaussian_glyph_res if res is None else res)
        try:
            sphere.SetThetaResolution(r)
            sphere.SetPhiResolution(r)
        except Exception:
            pass
        try:
            mapper = data.get("mapper")
            if mapper is not None:
                mapper.Modified()
        except Exception:
            pass

    def _apply_gaussian_scale_to_all(self) -> None:
        for data_id, data in list(getattr(self, "_gaussian_actor_data", {}).items()):
            actor = data.get("actor")
            if actor is None:
                continue
            self._apply_gaussian_scale_to_actor(actor, self._gaussian_scale_mult)
        self.vtk_widget.GetRenderWindow().Render()

    def _apply_gaussian_opacity_to_all(self) -> None:
        for data_id, data in list(getattr(self, "_gaussian_actor_data", {}).items()):
            actor = data.get("actor")
            if actor is None:
                continue
            self._apply_gaussian_opacity_to_actor(actor, self._gaussian_opacity_mult)
        self.vtk_widget.GetRenderWindow().Render()

    def _apply_gaussian_resolution_to_all(self) -> None:
        for data_id, data in list(getattr(self, "_gaussian_actor_data", {}).items()):
            actor = data.get("actor")
            if actor is None:
                continue
            self._apply_gaussian_resolution_to_actor(actor, self._gaussian_glyph_res)
        self.vtk_widget.GetRenderWindow().Render()

    def _decode_gaussian_ply_colors(
        self, path: str, expected_points: int
    ) -> Optional[np.ndarray]:
        """Decode gaussian-splat style color fields from a PLY file if RGB is missing."""
        cache = getattr(self, "_ply_color_cache", None)
        if cache:
            cached = cache.get(path)
            if cached is not None and len(cached) == expected_points:
                return cached
        required_fields = ("f_dc_0", "f_dc_1", "f_dc_2", "opacity")
        field_data = self._read_gaussian_ply_fields(path, required_fields)
        if not field_data or not all(
            name in field_data for name in ("f_dc_0", "f_dc_1", "f_dc_2")
        ):
            return None
        try:
            coeffs = np.stack(
                [
                    np.asarray(field_data["f_dc_0"], dtype=np.float32),
                    np.asarray(field_data["f_dc_1"], dtype=np.float32),
                    np.asarray(field_data["f_dc_2"], dtype=np.float32),
                ],
                axis=1,
            )
        except Exception:
            return None
        rgb = self._gaussian_dc_to_rgb(coeffs)
        if rgb is None or rgb.shape[0] == 0:
            return None
        if rgb.shape[0] != expected_points:
            logger.warning(
                "PLY color count (%d) does not match point count (%d); skipping color attachment.",
                rgb.shape[0],
                expected_points,
            )
            return None
        rgba = rgb
        if "opacity" in field_data:
            alpha = self._gaussian_opacity_to_alpha(field_data["opacity"])
            if alpha is not None and alpha.shape[0] == rgb.shape[0]:
                alpha_bytes = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
                rgba = np.concatenate([rgb, alpha_bytes[:, None]], axis=1)
        colors = np.ascontiguousarray(rgba.astype(np.uint8))
        if cache is not None and len(colors) == expected_points:
            cache[path] = colors
        return colors

    def _colors_from_gaussian_field_data(
        self, field_data: Mapping[str, np.ndarray], expected_points: int
    ) -> Optional[np.ndarray]:
        if not field_data:
            return None
        try:
            coeffs = np.stack(
                [
                    np.asarray(field_data["f_dc_0"], dtype=np.float32),
                    np.asarray(field_data["f_dc_1"], dtype=np.float32),
                    np.asarray(field_data["f_dc_2"], dtype=np.float32),
                ],
                axis=1,
            )
        except Exception:
            return None
        rgb = self._gaussian_dc_to_rgb(coeffs)
        if rgb is None or rgb.shape[0] != expected_points:
            return None
        rgba = rgb
        if "opacity" in field_data:
            alpha = self._gaussian_opacity_to_alpha(field_data["opacity"])
            if alpha is not None and alpha.shape[0] == rgb.shape[0]:
                alpha_bytes = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
                rgba = np.concatenate([rgb, alpha_bytes[:, None]], axis=1)
        return np.ascontiguousarray(rgba.astype(np.uint8))

    def _linear_to_srgb(self, rgb: np.ndarray) -> np.ndarray:
        """Convert linear RGB in [0,1] to sRGB for more faithful color output."""
        clipped = np.clip(rgb, 0.0, 1.0)
        return np.where(
            clipped <= 0.0031308,
            clipped * 12.92,
            1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
        )

    def _gaussian_dc_to_rgb(self, coeffs: np.ndarray) -> Optional[np.ndarray]:
        if coeffs is None or coeffs.ndim != 2 or coeffs.shape[1] < 3:
            return None
        sh_c0 = 0.28209479177387814  # Spherical harmonics constant for l=0,m=0
        rgb = 0.5 + sh_c0 * coeffs[:, :3]
        srgb = self._linear_to_srgb(rgb)
        return np.clip(srgb, 0.0, 1.0) * 255.0

    def _gaussian_opacity_to_alpha(self, opacity: np.ndarray) -> Optional[np.ndarray]:
        if opacity is None:
            return None
        raw = np.asarray(opacity, dtype=np.float32)
        if raw.size == 0:
            return None
        clipped = np.clip(raw, -50.0, 50.0)
        # Stored opacity is logit-space; convert back to [0, 1]
        alpha = 1.0 / (1.0 + np.exp(-clipped))
        return np.clip(alpha, 0.0, 1.0)

    def _read_gaussian_ply_fields(
        self, path: str, required: Sequence[str]
    ) -> Optional[dict[str, np.ndarray]]:
        """Return requested vertex fields with a simple on-disk cache to avoid reparsing."""
        if not required:
            return {}
        cache = getattr(self, "_gaussian_field_cache", None)
        if cache is None:
            self._gaussian_field_cache = {}
            cache = self._gaussian_field_cache
        signature = self._gaussian_ply_signature(path)
        entry = cache.get(path)
        if entry:
            cached_sig = entry.get("sig")
            if (
                cached_sig is not None
                and signature is not None
                and cached_sig != signature
            ):
                cache.pop(path, None)
                entry = None
        if entry is None:
            entry = {"sig": signature, "data": {}}
            cache[path] = entry
        fields: dict[str, np.ndarray] = entry["data"]  # type: ignore[index]
        missing = [name for name in required if name not in fields]
        if missing:
            data = self._read_gaussian_ply_fields_with_library(path, missing)
            if not data:
                data = self._read_gaussian_ply_fields_manual(path, missing)
            if not data:
                return None
            fields.update(data)
            entry["sig"] = signature
        if all(name in fields for name in required):
            return {name: fields[name] for name in required}
        return None

    def _gaussian_ply_signature(self, path: str) -> Optional[tuple[float, int]]:
        try:
            stat_res = os.stat(path)
        except OSError:
            return None
        return (stat_res.st_mtime, stat_res.st_size)

    def _read_gaussian_ply_fields_with_library(
        self, path: str, required: Sequence[str]
    ) -> Optional[dict[str, np.ndarray]]:
        try:
            from plyfile import PlyData  # type: ignore
        except Exception:
            return None
        try:
            ply = PlyData.read(path)
        except Exception as exc:
            logger.debug("plyfile failed to parse '%s': %s", path, exc)
            return None
        if "vertex" not in ply.elements:
            return None
        vertex = ply["vertex"].data
        if vertex.dtype.names is None:
            return None
        out: dict[str, np.ndarray] = {}
        for name in required:
            if name in vertex.dtype.names:
                try:
                    out[name] = np.asarray(vertex[name])
                except Exception:
                    return None
        return out if out else None

    def _read_gaussian_ply_fields_manual(
        self, path: str, required: Sequence[str]
    ) -> Optional[dict[str, np.ndarray]]:
        """Manual PLY vertex reader for gaussian splats when plyfile is unavailable."""
        try:
            with open(path, "rb") as fh:
                format_spec = None
                vertex_count = 0
                vertex_props: list[tuple[str, str]] = []
                reading_vertex = False
                while True:
                    line = fh.readline()
                    if not line:
                        return None
                    decoded = line.decode("ascii", errors="ignore").strip()
                    if not decoded:
                        continue
                    if decoded.startswith("format"):
                        parts = decoded.split()
                        if len(parts) >= 2:
                            format_spec = parts[1]
                    elif decoded.startswith("element"):
                        parts = decoded.split()
                        if len(parts) >= 3:
                            reading_vertex = parts[1] == "vertex"
                            if reading_vertex:
                                try:
                                    vertex_count = int(float(parts[2]))
                                except Exception:
                                    vertex_count = 0
                        else:
                            reading_vertex = False
                    elif reading_vertex and decoded.startswith("property"):
                        tokens = decoded.split()
                        if len(tokens) == 3:
                            _, ply_type, name = tokens
                            vertex_props.append((name, ply_type))
                        else:
                            return None
                    elif decoded == "end_header":
                        break
                if format_spec not in (
                    "binary_little_endian",
                    "binary_little_endian1.0",
                    "binary_little_endian1",
                ):
                    logger.warning(
                        "PLY '%s' uses unsupported format '%s' for gaussian colors.",
                        path,
                        format_spec or "unknown",
                    )
                    return None
                if vertex_count <= 0 or not vertex_props:
                    return None
                dtype_fields = []
                for name, ply_type in vertex_props:
                    code = self._PLY_DTYPE_MAP.get(ply_type)
                    if code is None:
                        return None
                    dtype_fields.append((name, "<" + code))
                dtype = np.dtype(dtype_fields)
                data = np.fromfile(fh, dtype=dtype, count=vertex_count)
        except Exception as exc:
            logger.warning("Manual PLY parsing failed for '%s': %s", path, exc)
            return None
        result: dict[str, np.ndarray] = {}
        for name in required:
            if name in data.dtype.names:
                result[name] = np.asarray(data[name])
        return result if result else None

    def _normalize_quaternions(self, quats: np.ndarray) -> np.ndarray:
        """Return normalized quaternions to avoid drift and improve glyph orientation."""
        if quats.ndim != 2 or quats.shape[1] < 4:
            return quats
        norms = np.linalg.norm(quats[:, :4], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = quats / norms
        return normalized.astype(np.float32, copy=False)

    def _quaternion_to_euler_deg(self, quats: np.ndarray) -> np.ndarray:
        """Convert quaternions [w, x, y, z] to Euler angles (deg) for VTK rotation arrays."""
        if quats.ndim != 2 or quats.shape[1] < 4:
            return np.zeros((len(quats), 3), dtype=np.float32)
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.degrees(np.arctan2(t0, t1))

        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.degrees(np.arcsin(t2))

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.degrees(np.arctan2(t3, t4))
        angles = np.stack([roll, pitch, yaw], axis=1).astype(np.float32)
        if not np.all(np.isfinite(angles)):
            angles = np.nan_to_num(angles, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )
        return angles

    def _add_point_cloud_csv_or_xyz(self, path: str, focus: bool = True):
        import numpy as np

        pts = None
        colors = None
        intensity = None
        region_labels = None
        if path.lower().endswith(".xyz"):
            data = np.loadtxt(path)
            if data.ndim != 2 or data.shape[1] < 3:
                raise RuntimeError("XYZ must have at least 3 columns: x y z")
            pts = data[:, :3].astype(np.float32)
            if data.shape[1] >= 6:
                col = data[:, 3:6]
                if col.max() <= 1.0:
                    col = col * 255.0
                colors = np.clip(col, 0, 255).astype(np.uint8)
            # Ask for scale factors
            scale = self._prompt_point_scale()
            if scale is not None:
                pts *= np.array(scale, dtype=np.float32)
        else:  # CSV with mapping dialog
            from .csv_mapping_dialog import CSVPointCloudMappingDialog

            df = pd.read_csv(path)
            parsed = self._auto_parse_point_cloud_csv(df)
            if parsed is not None:
                pts, colors, intensity, region_labels = parsed
            else:
                dlg = CSVPointCloudMappingDialog(df.columns, parent=self)
                if dlg.exec_() != QtWidgets.QDialog.Accepted:
                    return
                m = dlg.mapping()

                def get(name):
                    col = m.get(name)
                    return None if col is None else df[col]

                try:
                    pts = np.stack(
                        [get("x").to_numpy(), get("y").to_numpy(), get("z").to_numpy()],
                        axis=1,
                    ).astype(np.float32)
                except Exception:
                    raise RuntimeError("Invalid X/Y/Z column mapping")
                # Apply scale from dialog
                try:
                    sx, sy, sz = (
                        float(m.get("sx", 1.0)),
                        float(m.get("sy", 1.0)),
                        float(m.get("sz", 1.0)),
                    )
                    pts *= np.array([sx, sy, sz], dtype=np.float32)
                except Exception:
                    pass

                # Build colors
                color_by = m.get("color_by")
                mode = m.get("color_mode")
                if color_by:
                    series = get("color_by")
                    if mode == "categorical" or (series.dtype == object):
                        vals = series.astype(str).to_numpy()
                        colors = self._colors_for_categories(vals)
                    else:
                        raw = series.to_numpy(dtype=float)
                        v = raw[np.isfinite(raw)] if raw.size else raw
                        vmin = float(np.nanmin(v)) if v.size else 0.0
                        vmax = float(np.nanmax(v)) if v.size else 1.0
                        if vmax <= vmin:
                            vmax = vmin + 1e-6
                        norm = (raw - vmin) / (vmax - vmin)
                        norm = np.clip(norm, 0.0, 1.0)
                        colors = self._gradient_blue_red(norm)
                else:
                    inten_series = get("intensity")
                    if inten_series is not None:
                        inten = inten_series.to_numpy(dtype=float)
                        imax = np.nanmax(inten) if inten.size else 1.0
                        imin = np.nanmin(inten) if inten.size else 0.0
                        if imax <= imin:
                            imax = imin + 1e-6
                        g = np.clip((inten - imin) / (imax - imin), 0.0, 1.0)
                        g = (g * 255.0).astype(np.uint8)
                        colors = np.stack([g, g, g], axis=1)

                # Parse region labels if specified
                region_labels = None
                label_col = m.get("label_by")
                if label_col:
                    try:
                        # Ensure labels are strings for the tooltip
                        region_labels = get("label_by").astype(str).to_numpy()
                    except Exception as e:
                        print(
                            f"Warning: Could not parse region labels from column '{label_col}': {e}"
                        )

        if pts is None or len(pts) == 0:
            raise RuntimeError("No points parsed")

        if intensity is not None:
            intensity_values = intensity.astype(float)
        else:
            intensity_values = pts[:, 2].astype(float)

        if colors is None:
            vals = intensity_values
            finite_vals = vals[np.isfinite(vals)] if vals.size else vals
            vmin = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
            vmax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-6
            norm = (vals - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0.0, 1.0)
            colors = self._gradient_blue_red(norm)

        safe_pts = np.asarray(pts, dtype=np.float32)
        safe_pts = self._maybe_align_point_cloud_to_volume(safe_pts)
        bounds = (
            float(np.nanmin(safe_pts[:, 0])),
            float(np.nanmax(safe_pts[:, 0])),
            float(np.nanmin(safe_pts[:, 1])),
            float(np.nanmax(safe_pts[:, 1])),
            float(np.nanmin(safe_pts[:, 2])),
            float(np.nanmax(safe_pts[:, 2])),
        )

        self._region_actors = {}
        region_default_colors: dict[str, QtGui.QColor] = {}
        region_label_list: list[str] | None = None
        if region_labels is not None and len(region_labels) == len(pts):
            region_label_list = [str(label) for label in region_labels]

        if region_label_list:
            groups: dict[str, list[int]] = OrderedDict()
            for idx, label in enumerate(region_label_list):
                groups.setdefault(label, []).append(idx)

            for label, idxs in groups.items():
                indices = np.asarray(idxs, dtype=np.intp)
                subset_pts = safe_pts[indices]
                subset_colors = colors[indices] if colors is not None else None
                actor = self._create_point_actor(
                    subset_pts, subset_colors, region_label=label
                )
                if actor is None:
                    continue
                self.renderer.AddActor(actor)
                self._point_actors.append(actor)
                self._region_actors[label] = actor
                region_default_colors[label] = self._infer_region_color(subset_colors)
        else:
            actor = self._create_point_actor(safe_pts, colors)
            if actor is not None:
                self.renderer.AddActor(actor)
                self._point_actors.append(actor)
                if hasattr(self, "show_point_cloud_checkbox"):
                    self.show_point_cloud_checkbox.blockSignals(True)
                    self.show_point_cloud_checkbox.setChecked(True)
                    self.show_point_cloud_checkbox.blockSignals(False)
                    self._point_cloud_visible = True

        self._populate_region_selection(
            list(self._region_actors.keys()), region_default_colors
        )
        self._update_point_controls_visibility()
        if focus:
            self._focus_on_bounds(bounds)
        return bounds

    def _ensure_vertices(self, poly: vtkPolyData) -> vtkPolyData:
        """Create a polydata with Verts so that points render as glyphs.

        Avoids requiring vtkVertexGlyphFilter which may be absent in some VTK builds.
        """
        npts = poly.GetNumberOfPoints()
        verts = vtkCellArray()
        for i in range(npts):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        out = vtkPolyData()
        out.SetPoints(poly.GetPoints())
        out.SetVerts(verts)
        try:
            out.GetPointData().ShallowCopy(poly.GetPointData())
        except Exception:
            pass
        return out

    def _create_point_actor(
        self,
        pts: np.ndarray,
        colors: Optional[np.ndarray],
        region_label: Optional[str] = None,
    ) -> Optional[vtkActor]:
        if pts is None or len(pts) == 0:
            return None
        vpoints = vtkPoints()
        vpoints.SetNumberOfPoints(len(pts))
        for i, (x, y, z) in enumerate(pts):
            vpoints.SetPoint(i, float(x), float(y), float(z))
        poly = vtkPolyData()
        poly.SetPoints(vpoints)
        poly2 = self._ensure_vertices(poly)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly2)
        if colors is not None and len(colors):
            safe_colors = np.clip(colors, 0.0, 255.0).astype(np.uint8)
            c_arr = numpy_to_vtk(safe_colors, deep=True)
            try:
                c_arr.SetNumberOfComponents(3)
            except Exception:
                pass
            c_arr.SetName("Colors")
            poly2.GetPointData().SetScalars(c_arr)
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray("Colors")

        if region_label is not None:
            label_array = vtkStringArray()
            label_array.SetName("RegionLabel")
            label_array.SetNumberOfValues(len(pts))
            for i in range(len(pts)):
                label_array.SetValue(i, region_label)
            poly2.GetPointData().AddArray(label_array)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(self.point_size_slider.value())
        actor.SetVisibility(self._point_cloud_visible)
        return actor

    def _gradient_blue_red(self, norm: np.ndarray) -> np.ndarray:
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
        norm = np.clip(norm, 0.0, 1.0)
        c0 = np.array([30, 70, 200], dtype=np.float32)
        c1 = np.array([200, 50, 50], dtype=np.float32)
        rgb = c0[None, :] * (1.0 - norm[:, None]) + c1[None, :] * norm[:, None]
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _colors_for_categories(self, vals: np.ndarray) -> np.ndarray:
        palette = np.array(
            [
                [230, 25, 75],
                [60, 180, 75],
                [255, 225, 25],
                [0, 130, 200],
                [245, 130, 48],
                [145, 30, 180],
                [70, 240, 240],
                [240, 50, 230],
                [210, 245, 60],
                [250, 190, 190],
                [0, 128, 128],
                [230, 190, 255],
            ],
            dtype=np.uint8,
        )
        uniq = {}
        out = np.zeros((len(vals), 3), dtype=np.uint8)
        next_idx = 0
        for i, v in enumerate(vals):
            if v not in uniq:
                uniq[v] = next_idx
                next_idx = (next_idx + 1) % len(palette)
            out[i] = palette[uniq[v]]
        return out

    def _prompt_point_scale(self) -> Optional[Tuple[float, float, float]]:
        """Prompt user for point scale factors (spacing) before loading.

        Defaults to (1,1,1). If a volume exists, prefill with current volume spacing.
        Returns None if user cancels.
        """
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Point Spacing / Scale")
        form = QtWidgets.QFormLayout(dlg)
        sx = QtWidgets.QDoubleSpinBox(dlg)
        sy = QtWidgets.QDoubleSpinBox(dlg)
        sz = QtWidgets.QDoubleSpinBox(dlg)
        for sb in (sx, sy, sz):
            sb.setDecimals(6)
            sb.setRange(0.000001, 1e9)
            sb.setValue(1.0)
        # Prefill from volume spacing if available
        try:
            if getattr(self, "_vtk_img", None) is not None:
                sp = self._vtk_img.GetSpacing()
                sx.setValue(float(sp[0]))
                sy.setValue(float(sp[1]))
                sz.setValue(float(sp[2]))
        except Exception:
            pass
        row = QtWidgets.QHBoxLayout()
        row.addWidget(sx)
        row.addWidget(sy)
        row.addWidget(sz)
        w = QtWidgets.QWidget(dlg)
        w.setLayout(row)
        form.addRow("Scale X/Y/Z", w)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addWidget(buttons)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return float(sx.value()), float(sy.value()), float(sz.value())

    def _auto_parse_point_cloud_csv(
        self, df: "pd.DataFrame"
    ) -> Optional[tuple[np.ndarray, Optional[np.ndarray], np.ndarray, None]]:
        lookup = {col.lower(): col for col in df.columns}
        lower_cols = set(lookup)
        allowed = {"x", "y", "z", "intensity", "red", "green", "blue"}
        if not lower_cols.issubset(allowed):
            return None
        if not all(name in lower_cols for name in ("x", "y", "z")):
            return None

        try:
            coords = np.stack(
                [
                    df[lookup["x"]].to_numpy(dtype=float),
                    df[lookup["y"]].to_numpy(dtype=float),
                    df[lookup["z"]].to_numpy(dtype=float),
                ],
                axis=1,
            ).astype(np.float32)
        except Exception:
            return None

        intensity = None
        if "intensity" in lookup:
            intensity = df[lookup["intensity"]].to_numpy(dtype=float)
        else:
            intensity = coords[:, 2]

        colors = None
        color_keys = ("red", "green", "blue")
        if all(key in lookup for key in color_keys):
            colors = np.stack(
                [df[lookup[key]].to_numpy(dtype=float) for key in color_keys], axis=1
            )
        return coords, colors, intensity.astype(np.float32), None

    def _scale_poly_points(
        self, poly: vtkPolyData, scale: Tuple[float, float, float]
    ) -> vtkPolyData:
        """Return a copy of poly with points multiplied by scale per axis."""
        pts = poly.GetPoints()
        if pts is None:
            return poly
        n = pts.GetNumberOfPoints()
        new_pts = vtkPoints()
        new_pts.SetNumberOfPoints(n)
        sx, sy, sz = scale
        for i in range(n):
            x, y, z = pts.GetPoint(i)
            new_pts.SetPoint(i, float(x) * sx, float(y) * sy, float(z) * sz)
        out = vtkPolyData()
        out.SetPoints(new_pts)
        try:
            out.GetPointData().ShallowCopy(poly.GetPointData())
        except Exception:
            pass
        return out

    def _focus_on_bounds(
        self, bounds: Optional[Tuple[float, float, float, float, float, float]]
    ) -> None:
        """Adjust the camera to frame the provided bounds."""
        if not bounds:
            return
        values = np.array(bounds, dtype=float)
        if not np.all(np.isfinite(values)):
            return
        xmin, xmax, ymin, ymax, zmin, zmax = values
        center = (
            (xmin + xmax) * 0.5,
            (ymin + ymax) * 0.5,
            (zmin + zmax) * 0.5,
        )
        radius = max(xmax - xmin, ymax - ymin, zmax - zmin)
        if not np.isfinite(radius) or radius <= 0:
            radius = 1.0
        distance = radius * 2.5
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(center[0], center[1] - distance, center[2] + distance)
        camera.SetViewUp(0.0, 0.0, 1.0)
        self.renderer.ResetCameraClippingRange()

    def _union_bounds(
        self,
        first: Optional[Tuple[float, float, float, float, float, float]],
        second: Optional[Tuple[float, float, float, float, float, float]],
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        if first is None:
            return second
        if second is None:
            return first
        return (
            min(first[0], second[0]),
            max(first[1], second[1]),
            min(first[2], second[2]),
            max(first[3], second[3]),
            min(first[4], second[4]),
            max(first[5], second[5]),
        )

    def _update_point_sizes(self):
        size = int(self.point_size_slider.value())
        for actor in getattr(self, "_point_actors", []):
            actor.GetProperty().SetPointSize(size)
        self.vtk_widget.GetRenderWindow().Render()

    def _ensure_scene_light(self) -> None:
        if getattr(self, "_scene_light", None) is None:
            light = vtkLight()
            try:
                light.SetLightTypeToHeadlight()
            except Exception:
                pass
            light.SetIntensity(self._light_intensity)
            self.renderer.AddLight(light)
            self._scene_light = light

    def _update_light_intensity(self, value: float) -> None:
        self._light_intensity = float(value)
        self._ensure_scene_light()
        try:
            self._scene_light.SetIntensity(self._light_intensity)  # type: ignore[union-attr]
        except Exception:
            pass
        self.vtk_widget.GetRenderWindow().Render()

    def _update_gaussian_scale(self, value: float):
        self._gaussian_scale_mult = float(value)
        self._apply_gaussian_scale_to_all()

    def _update_gaussian_opacity(self, value: float):
        self._gaussian_opacity_mult = float(value)
        self._apply_gaussian_opacity_to_all()

    def _update_gaussian_resolution(self, value: int):
        self._gaussian_glyph_res = int(value)
        self._apply_gaussian_resolution_to_all()

    def _update_point_controls_visibility(self) -> None:
        has_points = bool(getattr(self, "_point_actors", []))
        self.point_detail_widget.setVisible(has_points)
        has_gaussian = bool(getattr(self, "_gaussian_actor_data", {}))
        if hasattr(self, "gaussian_group"):
            self.gaussian_group.setEnabled(has_gaussian)
        if not has_points:
            self._clear_region_selection()
        # Release cached gaussian data when points are cleared to free memory.
        self._ply_color_cache.clear()
        self._gaussian_field_cache.clear()

    def _update_mesh_controls_visibility(self) -> None:
        has_mesh = bool(getattr(self, "_mesh_actors", []))
        self.mesh_detail_widget.setVisible(has_mesh)

    def _clear_point_clouds(self):
        self._set_point_cloud_visibility(False)
        for actor in getattr(self, "_point_actors", []):
            try:
                self.renderer.RemoveActor(actor)
            except Exception:
                pass
        self._point_actors = []
        self._gaussian_actor_data = {}
        self._region_actors = {}
        self._clear_region_selection()
        self._update_point_controls_visibility()
        self.vtk_widget.GetRenderWindow().Render()
        self._point_cloud_visible = False
        if hasattr(self, "show_point_cloud_checkbox"):
            self.show_point_cloud_checkbox.blockSignals(True)
            self.show_point_cloud_checkbox.setChecked(False)
            self.show_point_cloud_checkbox.blockSignals(False)
        if not getattr(self, "_has_volume", False) and not getattr(
            self, "_mesh_actors", []
        ):
            self._source_path = None
        self._refresh_status_summary()

    def _clear_region_selection(self) -> None:
        if not hasattr(self, "region_list_widget"):
            return
        self.region_list_widget.blockSignals(True)
        self.region_list_widget.clear()
        self.region_list_widget.blockSignals(False)
        self.region_list_widget.setEnabled(False)
        self.region_group.setEnabled(False)
        self._region_entries.clear()
        self._region_colors.clear()
        if hasattr(self, "region_search"):
            self.region_search.blockSignals(True)
            self.region_search.clear()
            self.region_search.blockSignals(False)

    def _format_region_display(self, label: str) -> str:
        acronym, name = self._split_region_label(label)
        if acronym and name and acronym != name:
            return f"{acronym} - {name}"
        if acronym:
            return acronym
        if name:
            return name
        return str(label or "")

    def _split_region_label(self, label: str) -> tuple[str, str]:
        text = (label or "").strip()
        if not text:
            return "", ""
        paren_match = re.search(r"\(([^)]+)\)\s*$", text)
        if paren_match:
            acronym = paren_match.group(1).strip()
            name = text[: paren_match.start()].strip()
            if not name:
                name = text
            return acronym, name
        for delim in (":", "—", "-", "–"):
            if delim in text:
                left, right = [part.strip() for part in text.split(delim, 1)]
                if left and right:
                    if self._looks_like_region_acronym(left):
                        return left, right
                    if self._looks_like_region_acronym(right):
                        return right, left
        return "", text

    def _looks_like_region_acronym(self, segment: str) -> bool:
        seg = (segment or "").strip()
        if not seg:
            return False
        if seg.isupper() and len(seg) <= 8:
            return True
        upper_digit_count = sum(1 for c in seg if c.isupper() or c.isdigit())
        if upper_digit_count >= 2 and len(seg) <= 6:
            return True
        return False

    def _populate_region_selection(
        self,
        labels: Sequence[str],
        default_colors: Mapping[str, QtGui.QColor] | None = None,
    ) -> None:
        if not labels:
            self._clear_region_selection()
            return
        seen = set()
        ordered = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                ordered.append(label)

        self.region_list_widget.blockSignals(True)
        self.region_list_widget.clear()
        self._region_entries.clear()
        self._region_colors.clear()
        default_colors = default_colors or {}
        for label in ordered:
            display_text = self._format_region_display(label)
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.UserRole, label)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.region_list_widget.addItem(item)

            widget = QtWidgets.QWidget()
            widget_layout = QtWidgets.QHBoxLayout(widget)
            widget_layout.setContentsMargins(2, 1, 2, 1)
            widget_layout.setSpacing(4)
            color_button = QtWidgets.QToolButton()
            color_button.setAutoRaise(True)
            color_button.setFixedSize(20, 20)
            color_button.clicked.connect(
                lambda _, lbl=label: self._pick_region_color(lbl)
            )
            checkbox = QtWidgets.QCheckBox(display_text)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(
                lambda state, lbl=label: self._set_region_visibility(
                    lbl, state == QtCore.Qt.Checked
                )
            )
            widget_layout.addWidget(color_button)
            widget_layout.addWidget(checkbox)
            widget_layout.addStretch(1)
            self.region_list_widget.setItemWidget(item, widget)
            item.setSizeHint(widget.sizeHint())

            entry = _RegionSelectionEntry(
                item=item,
                checkbox=checkbox,
                color_button=color_button,
                display_text=display_text,
            )
            self._region_entries[label] = entry

            color = default_colors.get(label)
            if color is None:
                color = QtGui.QColor(255, 255, 255)
            self._region_colors[label] = color
            self._update_region_color_button(label, color)
        self.region_list_widget.blockSignals(False)
        self.region_group.setEnabled(True)
        self.region_list_widget.setEnabled(True)
        self._filter_region_items()

    def _filter_region_items(self, query: str | None = None) -> None:
        if query is None and hasattr(self, "region_search"):
            query = self.region_search.text()
        query = (query or "").strip().lower()
        should_filter = bool(query)
        for entry in self._region_entries.values():
            display = entry.display_text.lower()
            entry.item.setHidden(should_filter and query not in display)

    def _set_region_check_states(self, checked: bool) -> None:
        if not self._region_entries:
            return
        self.region_list_widget.blockSignals(True)
        for entry in self._region_entries.values():
            entry.checkbox.blockSignals(True)
            entry.checkbox.setChecked(checked)
            entry.checkbox.blockSignals(False)
        self.region_list_widget.blockSignals(False)
        self._refresh_region_visibilities()

    def _refresh_region_visibilities(self) -> None:
        for label, entry in self._region_entries.items():
            actor = self._region_actors.get(label)
            if actor:
                actor.SetVisibility(entry.checkbox.isChecked())
        self.vtk_widget.GetRenderWindow().Render()

    def _set_region_visibility(self, label: str, visible: bool) -> None:
        actor = self._region_actors.get(label)
        if actor:
            actor.SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()

    def _pick_region_color(self, label: str) -> None:
        current = self._region_colors.get(label, QtGui.QColor(255, 255, 255))
        color = QtWidgets.QColorDialog.getColor(current, self, "Pick region color")
        if not color.isValid():
            return
        self._region_colors[label] = color
        self._update_region_color_button(label, color)
        self._apply_region_color(label, color)

    def _update_region_color_button(self, label: str, color: QtGui.QColor) -> None:
        entry = self._region_entries.get(label)
        if entry is None:
            return
        size = QtCore.QSize(16, 16)
        pixmap = QtGui.QPixmap(size)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 1))
        painter.drawRect(1, 1, size.width() - 3, size.height() - 3)
        painter.end()
        entry.color_button.setIcon(QtGui.QIcon(pixmap))
        entry.color_button.setIconSize(size)
        entry.color_button.setToolTip(color.name())

    def _apply_region_color(self, label: str, color: QtGui.QColor) -> None:
        actor = self._region_actors.get(label)
        if actor is None:
            return
        prop = actor.GetProperty()
        prop.SetColor(color.redF(), color.greenF(), color.blueF())
        mapper = actor.GetMapper()
        if hasattr(mapper, "ScalarVisibilityOff"):
            mapper.ScalarVisibilityOff()
        self.vtk_widget.GetRenderWindow().Render()

    def _infer_region_color(self, colors: Optional[np.ndarray]) -> QtGui.QColor:
        fallback = QtGui.QColor(255, 255, 255)
        if colors is None or len(colors) == 0:
            return fallback
        arr = np.asarray(colors, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return fallback
        if arr.shape[1] < 3:
            arr = np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])), constant_values=0.0)
        arr = arr[:, :3]
        if arr.size == 0:
            return fallback
        max_val = np.nanmax(arr)
        if max_val <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        mean = np.nanmean(arr, axis=0)
        rgb = np.clip(mean, 0.0, 255.0).astype(int)
        return QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def _load_mesh_dialog(self):
        start_dir = (
            str(self._path.parent)
            if getattr(self, "_path", None) and self._path.exists()
            else "."
        )
        filters = "Meshes (*.stl *.STL *.obj *.OBJ *.ply *.PLY);;All files (*.*)"
        res = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Mesh",
            start_dir,
            filters,
        )
        path = res[0] if isinstance(res, tuple) else res
        if not path:
            return
        path = str(path)
        try:
            self._load_mesh_file(path)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Mesh Viewer", f"Failed to load: {e}")

    def _load_diffuse_texture(self):
        self._load_texture_dialog("diffuse")

    def _load_normal_texture(self):
        self._load_texture_dialog("normal")

    def _load_mesh_file(self, path: str):
        path_obj = Path(path)
        if not getattr(self, "_has_volume", False):
            self._source_path = path_obj
        ext = path_obj.suffix.lower()
        textures: dict[str, str] = {}
        if ext == ".stl":
            poly = self._read_stl_mesh(path)
        elif ext == ".obj":
            poly = self._read_obj_mesh(path)
            textures = self._parse_obj_mtl_textures(path_obj)
        elif ext == ".ply":
            poly = self._read_ply_mesh(path)
        else:
            raise RuntimeError(f"Unsupported mesh format: {ext}")
        actor = self._add_mesh_actor(poly, path_obj.name)
        if textures.get("diffuse"):
            self._apply_texture_from_path(actor, "diffuse", textures["diffuse"])
        if textures.get("normal"):
            self._apply_texture_from_path(actor, "normal", textures["normal"])

    def _parse_obj_mtl_textures(self, obj_path: Path) -> dict[str, str]:
        textures: dict[str, str] = {}
        mtl_files: list[Path] = []
        try:
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as obj_file:
                for line in obj_file:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    parts = stripped.split(maxsplit=1)
                    if parts[0].lower() != "mtllib" or len(parts) < 2:
                        continue
                    for name in parts[1].split():
                        candidate = obj_path.parent / name
                        mtl_files.append(candidate)
        except Exception:
            return textures
        for mtl_path in mtl_files:
            if not mtl_path.exists():
                continue
            try:
                with open(mtl_path, "r", encoding="utf-8", errors="ignore") as mtl_file:
                    for line in mtl_file:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        tokens = line.split(maxsplit=1)
                        if len(tokens) < 2:
                            continue
                        key = tokens[0].lower()
                        raw = tokens[1].split("#", 1)[0].strip()
                        if not raw:
                            continue
                        file_name = raw.split()[-1]
                        if not file_name:
                            continue
                        file_path = (mtl_path.parent / file_name).resolve()
                        if not file_path.exists():
                            continue
                        if key == "map_kd" and "diffuse" not in textures:
                            textures["diffuse"] = str(file_path)
                        elif (
                            key in {"map_bump", "bump", "norm"}
                            and "normal" not in textures
                        ):
                            textures["normal"] = str(file_path)
                        if "diffuse" in textures and "normal" in textures:
                            return textures
            except Exception:
                continue
        return textures

    def _add_mesh_actor(self, poly: vtkPolyData, source_name: str) -> vtkActor:
        mapper = self._create_mesh_mapper()
        mapper.SetInputData(poly)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToSurface()
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.renderer.AddActor(actor)
        self._mesh_actors.append(actor)
        self._mesh_textures[id(actor)] = {}
        self._mesh_actor_names[id(actor)] = source_name
        self._active_mesh_actor = actor
        self._update_mesh_status_label()
        self._update_mesh_controls_visibility()
        return actor

    def _clear_meshes(self):
        for actor in getattr(self, "_mesh_actors", []):
            try:
                self.renderer.RemoveActor(actor)
            except Exception:
                pass
        self._mesh_actors = []
        self._mesh_textures.clear()
        self._mesh_actor_names.clear()
        self._active_mesh_actor = None
        self._update_mesh_controls_visibility()
        self.vtk_widget.GetRenderWindow().Render()
        if not getattr(self, "_has_volume", False) and not getattr(
            self, "_point_actors", []
        ):
            self._source_path = None
        self._update_mesh_status_label()

    def _load_texture_dialog(self, kind: str):
        actor = self._current_mesh_actor()
        if actor is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Mesh Texture",
                "Load a mesh before applying textures.",
            )
            return
        filters = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)"
        res = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"Open {kind.capitalize()} Texture",
            str(self._path.parent) if getattr(self, "_path", None) else ".",
            filters,
        )
        path = res[0] if isinstance(res, tuple) else res
        if not path:
            return
        try:
            self._apply_texture_from_path(actor, kind, path)
            self.vtk_widget.GetRenderWindow().Render()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Mesh Texture",
                f"Failed to load {kind} texture: {exc}",
            )

    def _current_mesh_actor(self) -> Optional[vtkActor]:
        actor = self._active_mesh_actor
        if actor not in self._mesh_actors:
            actor = self._mesh_actors[-1] if self._mesh_actors else None
        if actor is not None:
            self._active_mesh_actor = actor
        return actor

    def _apply_texture_from_path(self, actor: vtkActor, kind: str, path: str):
        texture = self._create_texture_from_file(path)
        self._apply_texture_to_actor(actor, texture, kind, path)

    def _create_texture_from_file(self, path: str) -> vtkTexture:
        factory = vtkImageReader2Factory()
        reader = factory.CreateImageReader2(path)
        if reader is None:
            raise RuntimeError("Unsupported texture format")
        reader.SetFileName(path)
        reader.Update()
        image = reader.GetOutput()
        if image is None:
            raise RuntimeError("Texture image could not be read")
        texture = vtkTexture()
        texture.SetInputData(image)
        texture.InterpolateOn()
        return texture

    def closeEvent(self, event):  # type: ignore[override]
        try:
            self._teardown_slice_planes()
        except Exception:
            pass
        try:
            self._teardown_slice_mode()
        except Exception:
            pass
        try:
            self._clear_overlay_volumes()
        except Exception:
            pass
        self._cleanup_out_of_core_backing()
        super().closeEvent(event)

    def _apply_texture_to_actor(
        self,
        actor: vtkActor,
        texture: vtkTexture,
        kind: str,
        path: Optional[str] = None,
    ):
        actor.SetTexture(texture)
        self._mesh_textures.setdefault(id(actor), {})[kind] = {
            "texture": texture,
            "path": path,
        }
        if kind == "normal":
            mapper = actor.GetMapper()
            if hasattr(mapper, "SetNormalTexture"):
                mapper.SetNormalTexture(texture)
            if hasattr(mapper, "SetUseNormalTexture"):
                mapper.SetUseNormalTexture(True)
        self._update_mesh_status_label()

    def _create_mesh_mapper(self) -> vtkPolyDataMapper:
        if vtkOpenGLPolyDataMapper is not None:
            return vtkOpenGLPolyDataMapper()
        return vtkPolyDataMapper()

    def _update_mesh_status_label(self):
        if not hasattr(self, "mesh_status_label"):
            return
        actor = self._current_mesh_actor()
        if actor is None:
            self.mesh_status_label.setText("Mesh: none loaded")
            self._refresh_status_summary()
            return
        name = self._mesh_actor_names.get(id(actor), "mesh")
        textures = self._mesh_textures.get(id(actor), {})
        parts = [f"Mesh: {name}"]
        for kind_label, key in (("Diffuse", "diffuse"), ("Normal", "normal")):
            entry = textures.get(key)
            path = entry.get("path") if isinstance(entry, dict) else None
            if path:
                parts.append(f"{kind_label}: {Path(path).name}")
        self.mesh_status_label.setText(" | ".join(parts))
        self._refresh_status_summary()

    def _on_volume_visibility_changed(self, state: int):
        visible = state == QtCore.Qt.Checked
        self._volume_visible = visible
        self._set_volume_actor_visibility(visible, force=True)

    def _on_point_cloud_visibility_changed(self, state: int):
        visible = state == QtCore.Qt.Checked
        self._point_cloud_visible = visible
        self._set_point_cloud_visibility(visible)

    def _ensure_volume_actor_added(self) -> None:
        """Make sure the primary volume actor is present in the renderer."""
        base_actor = getattr(self, "volume", None)
        renderer = getattr(self, "renderer", None)
        if base_actor is None or renderer is None:
            return
        try:
            if not renderer.HasViewProp(base_actor):
                renderer.AddVolume(base_actor)
        except Exception:
            # Fallback: try to re-attach mapper/property if missing
            try:
                mapper = getattr(self, "mapper", None)
                prop = getattr(self, "property", None)
                if mapper is not None:
                    base_actor.SetMapper(mapper)
                if prop is not None:
                    base_actor.SetProperty(prop)
                renderer.AddVolume(base_actor)
            except Exception:
                pass

    def _set_volume_actor_visibility(self, visible: bool, force: bool = False):
        if self._slice_mode and not force:
            visible = False
        if not hasattr(self, "volume") or self.volume is None:
            base_actor = None
        else:
            base_actor = self.volume
        if base_actor is not None:
            try:
                base_actor.SetVisibility(
                    visible
                    and getattr(self, "_has_volume", False)
                    and not self._slice_mode
                )
            except Exception:
                pass
        for entry in getattr(self, "_overlay_volumes", []):
            try:
                entry.actor.SetVisibility(
                    visible and entry.visible and not self._slice_mode
                )
            except Exception:
                pass
        try:
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass

    def _set_point_cloud_visibility(self, visible: bool):
        for actor in getattr(self, "_point_actors", []):
            try:
                actor.SetVisibility(visible)
            except Exception:
                pass
        try:
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass

    def _refresh_overlay_list(self):
        if not hasattr(self, "overlay_list"):
            return
        self._overlay_list_updating = True
        try:
            self.overlay_list.clear()
        except Exception:
            self._overlay_list_updating = False
            return
        for idx, entry in enumerate(self._overlay_volumes):
            item = QtWidgets.QListWidgetItem(entry.label)
            item.setFlags(
                item.flags()
                | QtCore.Qt.ItemIsUserCheckable
                | QtCore.Qt.ItemIsSelectable
                | QtCore.Qt.ItemIsEnabled
            )
            item.setCheckState(
                QtCore.Qt.Checked if entry.visible else QtCore.Qt.Unchecked
            )
            item.setData(QtCore.Qt.UserRole, idx)
            self.overlay_list.addItem(item)
        if hasattr(self, "overlay_group"):
            self.overlay_group.setVisible(bool(self._overlay_volumes))
        self._overlay_list_updating = False

    def _on_overlay_item_changed(self, item: QtWidgets.QListWidgetItem):
        if self._overlay_list_updating:
            return
        idx = item.data(QtCore.Qt.UserRole)
        if idx is None:
            return
        try:
            entry = self._overlay_volumes[int(idx)]
        except (IndexError, ValueError):
            return
        entry.visible = item.checkState() == QtCore.Qt.Checked
        self._set_volume_actor_visibility(self._volume_visible, force=True)

    def _remove_selected_overlays(self):
        if not getattr(self, "_overlay_volumes", None):
            return
        selected = self.overlay_list.selectedIndexes()
        if not selected:
            return
        indexes = sorted({idx.row() for idx in selected}, reverse=True)
        for idx in indexes:
            self._remove_overlay_at_index(idx)
        self._refresh_overlay_list()
        self._refresh_status_summary()

    def _remove_overlay_at_index(self, idx: int):
        if idx < 0 or idx >= len(self._overlay_volumes):
            return
        entry = self._overlay_volumes.pop(idx)
        try:
            self.renderer.RemoveVolume(entry.actor)
        except Exception:
            pass
        try:
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass
        self._refresh_status_summary()

    def _clear_overlay_volumes(self):
        if not getattr(self, "_overlay_volumes", None):
            return
        for entry in self._overlay_volumes:
            try:
                self.renderer.RemoveVolume(entry.actor)
            except Exception:
                pass
        self._overlay_volumes = []
        self._refresh_overlay_list()
        try:
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass
        self._refresh_status_summary()

    def _refresh_status_summary(self) -> None:
        """Update the friendly status readout for the dock."""
        if not hasattr(self, "data_status_label"):
            return
        path = getattr(self, "_source_path", None)
        if path:
            try:
                name = Path(path).name
            except Exception:
                name = str(path)
            data_txt = f"Data: {name}"
        else:
            data_txt = "Data: none loaded"
        self.data_status_label.setText(data_txt)

        dims_txt = "-"
        if getattr(self, "_volume_shape", None):
            z, y, x = self._volume_shape
            if all(val > 0 for val in (z, y, x)):
                dims_txt = f"{int(x)}×{int(y)}×{int(z)}"

        if getattr(self, "_slice_mode", False):
            mode_txt = f"Mode: slice viewer ({dims_txt})"
        elif getattr(self, "_has_volume", False):
            mode_txt = f"Mode: volume ({dims_txt})"
        else:
            mode_txt = "Mode: scene (no volume loaded)"
        self.mode_status_label.setText(mode_txt)

        overlays = len(getattr(self, "_overlay_volumes", []))
        points = len(getattr(self, "_point_actors", []))
        meshes = len(getattr(self, "_mesh_actors", []))
        self.counts_status_label.setText(
            f"Overlays: {overlays} • Points: {points} • Meshes: {meshes}"
        )

    def _update_volume_io_label(self, out_of_core: bool):
        if not hasattr(self, "volume_io_label"):
            return
        if self._slice_mode:
            text = "Volume I/O: slice viewer (on-demand)"
        elif out_of_core:
            text = "Volume I/O: out-of-core (auto)"
        else:
            text = "Volume I/O: in-memory"
        self.volume_io_label.setText(text)
        self._refresh_status_summary()

    def _read_stl_mesh(self, path: str) -> vtkPolyData:
        try:
            from vtkmodules.vtkIOGeometry import vtkSTLReader
        except Exception as exc:
            raise RuntimeError(
                "VTK STL reader module is not available. Install VTK with IOGeometry support."
            ) from exc
        return self._read_polydata_from_reader(vtkSTLReader, path)

    def _read_obj_mesh(self, path: str) -> vtkPolyData:
        try:
            from vtkmodules.vtkIOGeometry import vtkOBJReader
        except Exception as exc:
            raise RuntimeError(
                "VTK OBJ reader module is not available. Install VTK with IOGeometry support."
            ) from exc
        return self._read_polydata_from_reader(vtkOBJReader, path)

    def _read_ply_mesh(self, path: str) -> vtkPolyData:
        try:
            from vtkmodules.vtkIOPLY import vtkPLYReader
        except Exception as exc:
            raise RuntimeError(
                "VTK PLY reader module is not available. Install VTK with IOPLY support."
            ) from exc
        return self._read_polydata_from_reader(vtkPLYReader, path)

    def _read_polydata_from_reader(
        self, reader_cls: Callable[[], object], path: str
    ) -> vtkPolyData:
        reader = reader_cls()
        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Mesh contains no points")
        return poly

    # -------------------- Source type helpers --------------------
    def _resolve_initial_source(self, path: Path) -> Path:
        """If a directory was provided, auto-pick the first supported file."""
        try:
            if not path.exists():
                return path
        except Exception:
            return path
        if not path.is_dir():
            return path
        if self._is_zarr_candidate(path):
            return path

        try:
            entries = sorted(path.iterdir())
        except Exception:
            return path

        # DICOM or other volume directories should remain directories
        for entry in entries:
            if entry.is_file() and entry.suffix.lower() in DICOM_EXTS:
                return path

        def _find(exts: Tuple[str, ...]) -> Optional[Path]:
            for entry in entries:
                if entry.is_file() and entry.suffix.lower() in exts:
                    return entry
                if entry.is_dir() and self._is_zarr_candidate(entry):
                    return entry
            return None

        for ext_group in (POINT_CLOUD_EXTS, MESH_EXTS, VOLUME_FILE_EXTS):
            candidate = _find(ext_group)
            if candidate is not None:
                logger.info(
                    "VTK viewer: auto-selecting '%s' inside '%s'.",
                    candidate.name,
                    path,
                )
                return candidate
        return path

    def _is_volume_candidate(self, path: Path) -> bool:
        if path.is_dir():
            try:
                if self._is_zarr_candidate(path):
                    return True
                return any(
                    entry.is_file() and entry.suffix.lower() in DICOM_EXTS
                    for entry in path.iterdir()
                )
            except Exception:
                return False
        ext = path.suffix.lower()
        name = path.name.lower()
        return (
            ext in VOLUME_FILE_EXTS
            or ext in DICOM_EXTS
            or name.endswith(".ome.tif")
            or name.endswith(".nii")
            or name.endswith(".nii.gz")
            or name.endswith(".zarr")
            or name.endswith("zarr.json")
            or name.endswith(".zgroup")
        )

    def _is_point_cloud_candidate(self, path: Path) -> bool:
        if path.is_dir():
            try:
                return any(
                    entry.is_file() and entry.suffix.lower() in POINT_CLOUD_EXTS
                    for entry in path.iterdir()
                )
            except Exception:
                return False
        return path.suffix.lower() in POINT_CLOUD_EXTS

    def _is_mesh_candidate(self, path: Path) -> bool:
        if path.is_dir():
            try:
                return any(
                    entry.is_file() and entry.suffix.lower() in MESH_EXTS
                    for entry in path.iterdir()
                )
            except Exception:
                return False
        return path.suffix.lower() in MESH_EXTS
