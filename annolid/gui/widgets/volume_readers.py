from __future__ import annotations
import struct
import types
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from annolid.gui.widgets import pyvista_runtime as pv_rt
from annolid.gui.widgets.volume_types import VolumeData
from annolid.gui.widgets.volume_slice_loaders import (
    _BaseSliceLoader,
    _TiffSliceLoader,
    _ZarrSliceLoader,
)
from annolid.utils.logger import logger


@dataclass(frozen=True)
class VolumeReaderConfig:
    dicom_exts: tuple[str, ...]
    tiff_suffixes: tuple[str, ...]
    ome_tiff_suffixes: tuple[str, ...]
    auto_out_of_core_mb: float
    max_volume_voxels: int
    slice_mode_bytes: float
    hdr_exts: tuple[str, ...] = (".hdr",)
    img_exts: tuple[str, ...] = (".img",)


class VolumeReaders:
    """Owns format-specific volume readers and file-type heuristics."""

    def __init__(
        self,
        *,
        config: VolumeReaderConfig,
        use_gdcm: bool = False,
    ) -> None:
        self._config = config
        self._use_gdcm = use_gdcm
        self._gdcm_image_reader_cls = pv_rt.vtkGDCMImageReader

    def _available_memory_bytes(self) -> int:
        try:
            import psutil  # type: ignore

            mem = psutil.virtual_memory()
            return int(getattr(mem, "available", 0) or 0)
        except Exception:
            return 0

    def normalize_volume_selection(self, path: Path) -> Path:
        try:
            p = path
            if p.is_file() and p.suffix.lower() in (".img", ".hdr"):
                hdr_candidate = self._find_companion_file(p, ".hdr")
                if hdr_candidate is not None:
                    return hdr_candidate
            if p.is_file():
                if p.name.lower() in ("zarr.json", ".zgroup"):
                    return p.parent
                if (p.parent / ".zarray").exists():
                    return p.parent
                if (p.parent / "data" / ".zarray").exists() or (
                    p.parent / "data" / "zarr.json"
                ).exists():
                    return p.parent / "data"
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

    def is_volume_candidate(self, path: Path) -> bool:
        if path.is_dir():
            try:
                if self.is_zarr_candidate(path):
                    return True
                return any(
                    entry.is_file() and entry.suffix.lower() in self._config.dicom_exts
                    for entry in path.iterdir()
                )
            except Exception:
                return False
        ext = path.suffix.lower()
        name = path.name.lower()
        return (
            ext in self._config.tiff_suffixes
            or any(name.endswith(extn) for extn in self._config.ome_tiff_suffixes)
            or ext in self._config.dicom_exts
            or name.endswith(".zarr")
            or name.endswith("zarr.json")
            or name.endswith(".zgroup")
            or name.endswith(".nii")
            or name.endswith(".nii.gz")
            or ext in (".hdr", ".img")
        )

    def is_point_cloud_candidate(self, path: Path) -> bool:
        if path.is_dir():
            try:
                return any(
                    entry.is_file() and entry.suffix.lower() in {".ply", ".csv", ".xyz"}
                    for entry in path.iterdir()
                )
            except Exception:
                return False
        return path.suffix.lower() in {".ply", ".csv", ".xyz"}

    def is_mesh_candidate(self, path: Path) -> bool:
        if path.is_dir():
            try:
                return any(
                    entry.is_file() and entry.suffix.lower() in {".stl", ".obj"}
                    for entry in path.iterdir()
                )
            except Exception:
                return False
        return path.suffix.lower() in {".stl", ".obj"}

    def resolve_initial_source(self, path: Path) -> Path:
        """If a directory was provided, auto-pick the first supported file."""
        try:
            if not path.exists():
                return path
        except Exception:
            return path
        if not path.is_dir():
            return path
        if self.is_zarr_candidate(path):
            return path

        try:
            entries = sorted(path.iterdir())
        except Exception:
            return path

        for entry in entries:
            if entry.is_file() and entry.suffix.lower() in self._config.dicom_exts:
                return path

        def _find(exts: tuple[str, ...]) -> Optional[Path]:
            for entry in entries:
                if entry.is_file() and (
                    entry.suffix.lower() in exts
                    or any(entry.name.lower().endswith(ext) for ext in exts)
                ):
                    return entry
                if entry.is_dir() and self.is_zarr_candidate(entry):
                    return entry
            return None

        for ext_group in (
            {
                ".tif",
                ".tiff",
                ".nii",
                ".nii.gz",
                ".hdr",
                ".img",
                ".zarr",
                ".zarr.json",
                ".zgroup",
            },
            {".stl", ".obj"},
            {".ply", ".csv", ".xyz"},
        ):
            candidate = _find(tuple(ext_group))
            if candidate is not None:
                logger.info(
                    "3D viewer: auto-selecting '%s' inside '%s'.",
                    candidate.name,
                    path,
                )
                return candidate
        return path

    def make_simple_volume_data(
        self, volume: np.ndarray, spacing: Optional[tuple[float, float, float]]
    ) -> VolumeData:
        vmin, vmax = self._finite_minmax(volume)
        shape = tuple(int(x) for x in volume.shape[:3])
        return VolumeData(
            array=volume,
            spacing=spacing,
            vmin=vmin,
            vmax=vmax,
            is_grayscale=volume.ndim == 3,
            is_out_of_core=False,
            volume_shape=shape,
            shape=shape,
        )

    def _image_has_scalars(self, vtk_img: Any) -> bool:
        try:
            dims = vtk_img.GetDimensions()
            if len(dims) < 3 or any(int(d) <= 0 for d in dims[:3]):
                return False
            point_data = vtk_img.GetPointData()
            if point_data is None:
                return False
            scalars = point_data.GetScalars()
            if scalars is None:
                return False
            return int(scalars.GetNumberOfTuples()) > 0
        except Exception:
            return False

    def vtk_image_to_numpy(self, vtk_img: Any) -> np.ndarray:
        vtk_to_numpy_func = pv_rt.vtk_to_numpy
        if vtk_to_numpy_func is None:
            raise RuntimeError("Runtime numpy bridge is unavailable.")
        dims = vtk_img.GetDimensions()
        pd = vtk_img.GetPointData()
        if pd is None:
            raise RuntimeError("No point data in volume.")
        scalars = pd.GetScalars()
        if scalars is None:
            raise RuntimeError("No scalar data in volume.")
        arr = vtk_to_numpy_func(scalars)
        return arr.reshape(dims[2], dims[1], dims[0])

    def _finite_minmax(
        self,
        arr: Any,
        *,
        fallback: tuple[float, float] = (0.0, 1.0),
    ) -> tuple[float, float]:
        try:
            data = np.asarray(arr)
            if data.size == 0:
                return fallback
            finite = np.isfinite(data)
            if np.any(finite):
                lo = float(np.min(data[finite]))
                hi = float(np.max(data[finite]))
                if hi <= lo:
                    hi = lo + 1e-6
                return lo, hi
        except Exception:
            pass
        return fallback

    def dtype_value_range(self, dtype: np.dtype) -> tuple[float, float]:
        if np.issubdtype(dtype, np.bool_):
            return 0.0, 1.0
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return float(info.min), float(info.max)
        return 0.0, 1.0

    def _safe_file_size(self, path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except Exception:
            return 0

    def _probe_tiff_metadata(
        self, path: Path
    ) -> Optional[tuple[tuple[int, int, int], np.dtype]]:
        try:
            import tifffile  # type: ignore

            with tifffile.TiffFile(str(path)) as tif:
                series = tif.series[0]
                shape = tuple(int(x) for x in series.shape)
                dtype = np.dtype(series.dtype)
            if len(shape) == 2:
                final_shape = (1, int(shape[0]), int(shape[1]))
            elif len(shape) >= 3:
                final_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
            else:
                return None
            return final_shape, dtype
        except Exception:
            return None

    def _open_tiff_memmap(self, path: Path) -> Optional[np.ndarray]:
        try:
            import tifffile  # type: ignore

            try:
                reader = tifffile.memmap(str(path), mode="r+")
            except PermissionError:
                reader = tifffile.memmap(str(path), mode="r")
            if reader.ndim not in (3, 4):
                return None
            if reader.ndim == 4 and reader.shape[-1] == 1:
                reader = reader[..., 0]
            if reader.ndim != 3:
                return None
            if not reader.flags.c_contiguous:
                reader = np.ascontiguousarray(reader)
            return reader
        except Exception:
            return None

    def should_use_slice_mode(self, shape: tuple[int, ...], dtype: np.dtype) -> bool:
        if not shape or len(shape) < 3:
            return False
        total_voxels = int(np.prod(shape[:3]))
        itemsize = max(1, np.dtype(dtype).itemsize)
        size_bytes = total_voxels * itemsize
        if (
            self._config.max_volume_voxels > 0
            and total_voxels > self._config.max_volume_voxels
        ):
            return True
        if (
            self._config.slice_mode_bytes > 0
            and size_bytes >= self._config.slice_mode_bytes
        ):
            return True
        available = self._available_memory_bytes()
        if available > 0 and size_bytes >= available * 0.8:
            return True
        return False

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

    def _is_label_volume(
        self, dtype: np.dtype, arr_obj: Any, source_path: Path
    ) -> bool:
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

    def is_tiff_candidate(self, path: Path) -> bool:
        name_lower = path.name.lower()
        suffix = path.suffix.lower()
        return suffix in self._config.tiff_suffixes or any(
            name_lower.endswith(ext) for ext in self._config.ome_tiff_suffixes
        )

    def is_zarr_candidate(self, path: Path) -> bool:
        if not path:
            return False
        if path.suffix.lower() == ".zarr":
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
            if path.is_dir():
                return (
                    (path / ".zarray").exists()
                    or (path / "zarr.json").exists()
                    or (path / ".zgroup").exists()
                    or (path / "data" / ".zarray").exists()
                )
        except Exception:
            pass
        return False

    def should_use_out_of_core_tiff(self, path: Path) -> bool:
        size_bytes = self._safe_file_size(path)
        avail_bytes = self._available_memory_bytes()
        if avail_bytes > 0 and size_bytes > 0 and size_bytes >= avail_bytes:
            return True
        if self._config.auto_out_of_core_mb > 0 and size_bytes > 0:
            if size_bytes >= self._config.auto_out_of_core_mb * 1024 * 1024:
                return True
        return False

    def _analyze_dtype(self, datatype_code: int, endian: str) -> np.dtype:
        mapping = {
            2: np.uint8,
            4: np.int16,
            8: np.int32,
            16: np.float32,
            64: np.float64,
            256: np.int8,
            512: np.uint16,
            768: np.uint32,
            1024: np.int64,
            1280: np.uint64,
        }
        base = mapping.get(int(datatype_code))
        if base is None:
            raise RuntimeError(f"Unsupported Analyze datatype code: {datatype_code}")
        dtype = np.dtype(base)
        if dtype.itemsize > 1:
            dtype = dtype.newbyteorder(endian)
        return dtype

    def _read_analyze_header(
        self, header_path: Path
    ) -> tuple[
        np.dtype, tuple[int, int, int, int], tuple[float, float, float], int, float
    ]:
        header = header_path.read_bytes()
        if len(header) < 348:
            raise RuntimeError("Analyze header is too small (expected >= 348 bytes).")

        sizeof_hdr_le = struct.unpack_from("<i", header, 0)[0]
        sizeof_hdr_be = struct.unpack_from(">i", header, 0)[0]
        if sizeof_hdr_le == 348:
            endian = "<"
        elif sizeof_hdr_be == 348:
            endian = ">"
        else:
            raise RuntimeError("Invalid Analyze header (sizeof_hdr != 348).")

        dims_raw = struct.unpack_from(f"{endian}8h", header, 40)
        ndim = max(0, int(dims_raw[0]))
        dims = [int(v) for v in dims_raw[1 : 1 + max(4, ndim)]]
        while len(dims) < 4:
            dims.append(1)
        dims = [d if d > 0 else 1 for d in dims]
        x, y, z, t = dims[0], dims[1], dims[2], dims[3]

        datatype = int(struct.unpack_from(f"{endian}h", header, 70)[0])
        pixdim = struct.unpack_from(f"{endian}8f", header, 76)
        spacing = (
            float(pixdim[1]) if float(pixdim[1]) > 0 else 1.0,
            float(pixdim[2]) if float(pixdim[2]) > 0 else 1.0,
            float(pixdim[3]) if float(pixdim[3]) > 0 else 1.0,
        )
        vox_offset = float(struct.unpack_from(f"{endian}f", header, 108)[0])
        offset = max(0, int(round(vox_offset)))
        scale = float(struct.unpack_from(f"{endian}f", header, 112)[0])
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        dtype = self._analyze_dtype(datatype, endian)
        return dtype, (x, y, z, t), spacing, offset, scale

    def _read_analyze_numpy(
        self, header_path: Path, image_path: Path
    ) -> tuple[np.ndarray, tuple[float, float, float], np.dtype]:
        dtype, (x, y, z, t), spacing, offset, scale = self._read_analyze_header(
            header_path
        )
        voxel_count = int(x * y * z * t)
        arr = np.fromfile(
            str(image_path), dtype=dtype, count=voxel_count, offset=offset
        )
        if arr.size != voxel_count and offset > 0:
            arr = np.fromfile(str(image_path), dtype=dtype, count=voxel_count, offset=0)
        if arr.size != voxel_count:
            raise RuntimeError(
                f"Analyze data size mismatch: expected {voxel_count} voxels, got {arr.size}."
            )
        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
        if t > 1:
            arr = arr.reshape((t, z, y, x))[0]
        else:
            arr = arr.reshape((z, y, x))
        if scale != 1.0:
            arr = arr.astype(np.float32, copy=False) * float(scale)
        return arr, spacing, np.dtype(dtype)

    def read_analyze_volume(self, path: Path) -> VolumeData:
        header = (
            path
            if path.suffix.lower() == ".hdr"
            else self._find_companion_file(path, ".hdr")
        )
        image = (
            path
            if path.suffix.lower() == ".img"
            else self._find_companion_file(path, ".img")
        )
        if not header or not image:
            raise RuntimeError("Analyze header/image missing.")

        header_path: Path = header
        image_path: Path = image

        try:
            vol, spacing, source_dtype = self._read_analyze_numpy(
                header_path, image_path
            )
        except Exception as exc:
            vtk_img = None
            if pv_rt.vtkAnalyzeReader:
                try:
                    reader = pv_rt.vtkAnalyzeReader()
                    reader.SetFileName(str(header_path))
                    reader.Update()
                    vtk_img = reader.GetOutput()
                except Exception:
                    pass
            if not vtk_img or not self._image_has_scalars(vtk_img):
                raise RuntimeError(f"Unable to decode Analyze volume: {exc}") from exc
            vol = self.vtk_image_to_numpy(vtk_img)
            s = vtk_img.GetSpacing()
            spacing = (s[0], s[1], s[2])
            source_dtype = np.dtype(vol.dtype)

        is_label = self._is_label_volume(
            source_dtype,
            types.SimpleNamespace(path=header_path.stem, name=header_path.name),
            path,
        )
        return dataclasses.replace(
            self.make_simple_volume_data(vol, spacing), is_label_map=is_label
        )

    def _select_zarr_array(self, obj: Any) -> Any:
        try:
            from zarr.core import Array
            from zarr.hierarchy import Group
        except ImportError:
            return obj
        if isinstance(obj, Array):
            return obj
        if isinstance(obj, Group):
            if "multiscales" in obj.attrs:
                try:
                    path = obj.attrs["multiscales"][0]["datasets"][0]["path"]
                    return obj[path]
                except (IndexError, KeyError):
                    pass
            for key in ["0", "data", "image", "volume"]:
                if key in obj:
                    item = obj[key]
                    if isinstance(item, Array) and item.ndim >= 3:
                        return item
                    if isinstance(item, Group):
                        return self._select_zarr_array(item)
        return obj

    def read_zarr(self, path: Path) -> VolumeData:
        import zarr  # type: ignore

        try:
            root = zarr.open_consolidated(str(path), mode="r")
        except Exception:
            try:
                root = zarr.open(str(path), mode="r")
            except Exception:
                raise RuntimeError(f"Failed to open Zarr store at {path}")

        arr = self._select_zarr_array(root)
        if hasattr(arr, "group_keys"):
            # If it's still a group, try to find the first 3D array
            queue = [arr]
            while queue:
                node = queue.pop(0)
                try:
                    for key in node.array_keys():
                        child = node[key]
                        if len(child.shape) >= 3:
                            arr = child
                            queue = []
                            break
                    if not queue:
                        for key in node.group_keys():
                            queue.append(node[key])
                except Exception:
                    continue

        if not hasattr(arr, "shape") or len(arr.shape) < 3:
            raise RuntimeError("Zarr array must be at least 3D.")

        dtype = np.dtype(arr.dtype)
        shape = tuple(int(x) for x in arr.shape)
        if self.should_use_slice_mode(shape, dtype):
            loader = _ZarrSliceLoader(
                arr, (shape[0] - 3, shape[0] - 2, shape[0] - 1)
            )  # Dummy axis mapping
            return self._build_slice_volume_data(loader)

        vol = np.asarray(arr)
        while vol.ndim > 3:
            vol = vol[0]
        return self.make_simple_volume_data(vol, None)

    def _build_slice_volume_data(
        self,
        loader: _BaseSliceLoader,
        spacing: Optional[tuple[float, float, float]] = None,
        is_label_map: bool = False,
    ) -> VolumeData:
        dtype = loader.dtype()
        vmin, vmax = self.dtype_value_range(dtype)
        shape = loader.shape()
        return VolumeData(
            array=None,
            spacing=spacing,
            vmin=0.0,
            vmax=1.0,
            is_grayscale=True,
            is_out_of_core=True,
            slice_mode=True,
            slice_loader=loader,
            volume_shape=shape,
            is_label_map=is_label_map,
            shape=shape,
        )

    def read_tiff_eager(self, path: Path) -> VolumeData:
        frames: list[np.ndarray] = []
        with Image.open(str(path)) as img:
            n = max(1, int(getattr(img, "n_frames", 1) or 1))
            for i in range(n):
                img.seek(i)
                frames.append(np.array(img))
        vol = np.stack(frames, axis=0)
        return self.make_simple_volume_data(vol, None)

    def read_tiff_out_of_core(self, path: Path) -> VolumeData:
        meta = self._probe_tiff_metadata(path)
        if meta and self.should_use_slice_mode(meta[0], meta[1]):
            loader = _TiffSliceLoader(path, meta[0], meta[1])
            return self._build_slice_volume_data(loader)

        memmap_arr = self._open_tiff_memmap(path)
        if memmap_arr is not None:
            return dataclasses.replace(
                self.make_simple_volume_data(memmap_arr, None), is_out_of_core=True
            )

        return self.read_tiff_eager(path)

    def read_dicom_series(self, directory: Path) -> VolumeData:
        vtk_dicom_reader_cls = pv_rt.vtkDICOMImageReader or pv_rt.vtkGDCMImageReader
        if not vtk_dicom_reader_cls:
            raise RuntimeError("DICOM reader unavailable.")
        reader = vtk_dicom_reader_cls()
        reader.SetDirectoryName(str(directory))
        reader.Update()
        vtk_img = reader.GetOutput()
        vol = self.vtk_image_to_numpy(vtk_img)
        s = vtk_img.GetSpacing()
        return self.make_simple_volume_data(vol, (s[0], s[1], s[2]))

    def read_volume_any(self, path: Path) -> VolumeData:
        if self.is_zarr_candidate(path):
            return self.read_zarr(path)
        if self.is_tiff_candidate(path):
            if self.should_use_out_of_core_tiff(path):
                return self.read_tiff_out_of_core(path)
            return self.read_tiff_eager(path)
        ext = path.suffix.lower()
        if ext in (".hdr", ".img"):
            return self.read_analyze_volume(path)
        if ext in self._config.dicom_exts or path.is_dir():
            # Assume DICOM series if it's a dir and not Zarr
            return self.read_dicom_series(path)

        raise RuntimeError(f"Unsupported volume format for path: {path}")

    def _find_companion_file(self, path: Path, suffix: str) -> Optional[Path]:
        cand = path.with_suffix(suffix)
        return cand if cand.exists() else None
