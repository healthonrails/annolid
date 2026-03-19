from __future__ import annotations

import json
import os
import struct
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from annolid.utils.logger import logger


@dataclass(frozen=True)
class VolumeReaderConfig:
    dicom_exts: tuple[str, ...]
    tiff_suffixes: tuple[str, ...]
    ome_tiff_suffixes: tuple[str, ...]
    auto_out_of_core_mb: float
    max_volume_voxels: int
    slice_mode_bytes: float


class VolumeReaders:
    """Owns format-specific volume readers and file-type heuristics."""

    def __init__(
        self,
        *,
        config: VolumeReaderConfig,
        make_volume_data: Callable[..., Any],
        make_slice_volume_data: Callable[..., Any],
        find_companion_file: Callable[[Path, str], Optional[Path]],
        memmap_slice_loader_cls: type,
        tiff_slice_loader_cls: type,
        zarr_slice_loader_cls: type,
        zarr_v3_array_cls: type,
        use_gdcm: bool = False,
        vtk_gdcm_image_reader_cls: Optional[type] = None,
    ) -> None:
        self._config = config
        self._make_volume_data_fn = make_volume_data
        self._make_slice_volume_data_fn = make_slice_volume_data
        self._find_companion_file = find_companion_file
        self._memmap_slice_loader_cls = memmap_slice_loader_cls
        self._tiff_slice_loader_cls = tiff_slice_loader_cls
        self._zarr_slice_loader_cls = zarr_slice_loader_cls
        self._zarr_v3_array_cls = zarr_v3_array_cls
        self._use_gdcm = use_gdcm
        self._vtk_gdcm_image_reader_cls = vtk_gdcm_image_reader_cls

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
            {".ply", ".csv", ".xyz"},
            {".stl", ".obj"},
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
        ):
            candidate = _find(tuple(ext_group))
            if candidate is not None:
                logger.info(
                    "VTK viewer: auto-selecting '%s' inside '%s'.",
                    candidate.name,
                    path,
                )
                return candidate
        return path

    def _make_simple_volume_data(
        self, volume: Any, spacing: Optional[tuple[float, float, float]]
    ) -> Any:
        vmin, vmax = self._finite_minmax(volume)
        return self._make_volume_data_fn(
            array=volume,
            spacing=spacing,
            vmin=vmin,
            vmax=vmax,
            is_grayscale=volume.ndim == 3,
            is_out_of_core=False,
            volume_shape=tuple(int(x) for x in volume.shape[:3]),
        )

    def make_simple_volume_data(
        self, volume: Any, spacing: Optional[tuple[float, float, float]]
    ) -> Any:
        return self._make_simple_volume_data(volume, spacing)

    def _vtk_image_has_scalars(self, vtk_img) -> bool:
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

    def _vtk_image_to_numpy(self, vtk_img) -> np.ndarray:
        from vtkmodules.util.numpy_support import vtk_to_numpy

        dims = vtk_img.GetDimensions()
        scalars = vtk_img.GetPointData().GetScalars()
        if scalars is None:
            raise RuntimeError("No scalar data in volume.")
        arr = vtk_to_numpy(scalars)
        return arr.reshape(dims[2], dims[1], dims[0])

    def vtk_image_to_numpy(self, vtk_img) -> np.ndarray:
        return self._vtk_image_to_numpy(vtk_img)

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
            else:
                lo, hi = fallback
        except Exception:
            return fallback
        if not np.isfinite(lo) or not np.isfinite(hi):
            return fallback
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    def _dtype_value_range(self, dtype: np.dtype) -> tuple[float, float]:
        if np.issubdtype(dtype, np.bool_):
            return 0.0, 1.0
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return float(info.min), float(info.max)
        if np.issubdtype(dtype, np.floating):
            return 0.0, 1.0
        return 0.0, 1.0

    def dtype_value_range(self, dtype: np.dtype) -> tuple[float, float]:
        return self._dtype_value_range(dtype)

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

    def _should_use_slice_mode(
        self, shape: tuple[int, int, int], dtype: np.dtype
    ) -> bool:
        if not shape or len(shape) < 3:
            return False
        total_voxels = int(shape[0]) * int(shape[1]) * int(shape[2])
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

    def should_use_slice_mode(
        self, shape: tuple[int, int, int], dtype: np.dtype
    ) -> bool:
        return self._should_use_slice_mode(shape, dtype)

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

    def _is_label_volume(self, dtype: np.dtype, arr_obj, source_path: Path) -> bool:
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
        if suffix in self._config.tiff_suffixes:
            return True
        return any(name_lower.endswith(ext) for ext in self._config.ome_tiff_suffixes)

    def is_zarr_candidate(self, path: Path) -> bool:
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

    def should_use_out_of_core_tiff(self, path: Path) -> bool:
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
        if self._config.auto_out_of_core_mb > 0 and size_bytes > 0:
            if size_bytes >= self._config.auto_out_of_core_mb * 1024 * 1024:
                logger.info(
                    "TIFF stack (%s) exceeds configured threshold (%.0f MB); enabling out-of-core caching.",
                    path,
                    self._config.auto_out_of_core_mb,
                )
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

    def _read_analyze_header(self, header_path: Path):
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
        bitpix = int(struct.unpack_from(f"{endian}h", header, 72)[0])
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
        expected_bits = int(dtype.itemsize * 8)
        if bitpix > 0 and bitpix != expected_bits:
            logger.warning(
                "Analyze header bitpix (%s) does not match dtype size (%s bits).",
                bitpix,
                expected_bits,
            )
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
            logger.info(
                "Analyze volume has %d timepoints; using the first volume for 3D view.",
                t,
            )
            arr = arr.reshape((t, z, y, x))[0]
        else:
            arr = arr.reshape((z, y, x))
        if scale != 1.0:
            arr = arr.astype(np.float32, copy=False) * float(scale)
        return arr, spacing, np.dtype(dtype)

    def _read_analyze_via_vtk_reader(self, header_path: Path):
        try:
            from vtkmodules.vtkIOImage import vtkAnalyzeReader
        except Exception:
            return None
        try:
            reader = vtkAnalyzeReader()
            reader.SetFileName(str(header_path))
            reader.Update()
            vtk_img = reader.GetOutput()
            if self._vtk_image_has_scalars(vtk_img):
                return vtk_img
        except Exception:
            return None
        return None

    def read_analyze_volume(self, path: Path) -> Any:
        header = path if path.suffix.lower() == ".hdr" else None
        image = path if path.suffix.lower() == ".img" else None
        if header is None:
            header = self._find_companion_file(path, ".hdr")
        if image is None:
            image = self._find_companion_file(path, ".img")
        if header is None:
            raise RuntimeError("Analyze header (.hdr) file was not found.")
        if image is None:
            raise RuntimeError("Analyze image (.img) file was not found.")

        analyze_error = None
        source_dtype: Optional[np.dtype] = None
        try:
            vol, spacing, source_dtype = self._read_analyze_numpy(header, image)
        except Exception as exc:
            analyze_error = exc
            vtk_img = self._read_analyze_via_vtk_reader(header)
            if vtk_img is None:
                raise RuntimeError(
                    f"Unable to decode Analyze volume. {analyze_error}"
                ) from exc
            vol = self._vtk_image_to_numpy(vtk_img)
            s = vtk_img.GetSpacing()
            spacing = (s[0], s[1], s[2])
            source_dtype = np.dtype(vol.dtype)

        if vol.size == 0:
            raise RuntimeError("Analyze volume contains no voxels.")
        is_label_volume = self._is_label_volume(
            source_dtype or np.dtype(vol.dtype),
            types.SimpleNamespace(path=header.stem, name=header.name),
            path,
        )
        vmin, vmax = self._finite_minmax(vol)
        return self._make_volume_data_fn(
            array=vol,
            spacing=spacing,
            vmin=vmin,
            vmax=vmax,
            is_grayscale=vol.ndim == 3,
            is_out_of_core=False,
            volume_shape=tuple(int(x) for x in vol.shape[:3]),
            is_label_map=is_label_volume,
        )

    def _first_3d_array_from_group(self, grp):
        try:
            from zarr.hierarchy import Group  # type: ignore
        except Exception:
            Group = ()  # type: ignore[assignment]

        queue = [grp]
        while queue:
            node = queue.pop(0)
            try:
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
            return self._finite_minmax(sample)
        except Exception:
            return None

    def _load_zarr_json(self, meta_path: Path) -> Optional[dict]:
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

    def load_zarr_json(self, meta_path: Path) -> Optional[dict]:
        return self._load_zarr_json(meta_path)

    def _find_zarr_array_metadata(
        self, base: Path
    ) -> tuple[Optional[Path], Optional[dict]]:
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

    def _select_zarr_array(self, obj):
        try:
            from zarr.hierarchy import Group
            from zarr.core import Array
        except ImportError:
            return obj

        if isinstance(obj, Array):
            return obj

        if isinstance(obj, Group):
            if "multiscales" in obj.attrs:
                try:
                    datasets = obj.attrs["multiscales"][0]["datasets"]
                    path = datasets[0]["path"]
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

    def _extract_zarr_spacing(self, obj, arr) -> Optional[tuple[float, float, float]]:
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

    def _zarr_axis_info(self, arr) -> tuple[tuple[int, int, int], dict[int, int]]:
        ndim = arr.ndim
        z_ix, y_ix, x_ix = ndim - 3, ndim - 2, ndim - 1
        axes_meta = None
        if hasattr(arr, "attrs") and "multiscales" in arr.attrs:
            try:
                axes_meta = arr.attrs["multiscales"][0].get("axes")
            except Exception:
                pass
        if axes_meta and len(axes_meta) == ndim:
            names = [
                x["name"].lower() if isinstance(x, dict) else x.lower()
                for x in axes_meta
            ]
            if "z" in names and "y" in names and "x" in names:
                z_ix = names.index("z")
                y_ix = names.index("y")
                x_ix = names.index("x")
        if ndim == 2:
            return (0, 0, 1), {}
        fixed_indices = {}
        for i in range(ndim):
            if i not in (z_ix, y_ix, x_ix):
                fixed_indices[i] = 0
        return (z_ix, y_ix, x_ix), fixed_indices

    def _zarr_to_numpy_zyx(
        self,
        arr,
        zyx_axes: tuple[int, int, int],
        fixed_indices: dict[int, int],
    ) -> np.ndarray:
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
        path_str = str(path)
        try:
            return zarr_mod.open_consolidated(path_str, mode="r")
        except Exception:
            pass
        try:
            return zarr_mod.open(path_str, mode="r")
        except Exception:
            pass
        if (path / "data").exists():
            try:
                return zarr_mod.open(str(path / "data"), mode="r")
            except Exception:
                pass
        return None

    def _normalize_volume_selection(self, path: Path) -> Path:
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

    def _first_3d_array_from_group(self, grp):
        try:
            from zarr.hierarchy import Group  # type: ignore
        except Exception:
            Group = ()  # type: ignore[assignment]
        queue = [grp]
        while queue:
            node = queue.pop(0)
            try:
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

    def _initial_slice_index_for_loader(self, loader) -> int:
        if loader is None:
            return 0
        try:
            arr = getattr(loader, "_arr", None)
            z_axis = getattr(loader, "_zyx_axes", (0, 1, 2))[0]
            if isinstance(arr, self._zarr_v3_array_cls) and z_axis == 0:
                return arr.first_nonempty_index(axis=0)
        except Exception:
            return 0
        return 0

    def initial_slice_index_for_loader(self, loader) -> int:
        return self._initial_slice_index_for_loader(loader)

    def convert_frame_to_plane(self, frame: np.ndarray, dtype: np.dtype) -> np.ndarray:
        return self._convert_frame_to_plane(frame, dtype)

    def _build_slice_volume_data(
        self,
        loader,
        spacing: Optional[tuple[float, float, float]] = None,
        value_range: Optional[tuple[float, float]] = None,
        is_grayscale: bool = True,
        is_label_map: bool = False,
    ) -> Any:
        dtype = loader.dtype()
        if value_range is not None:
            vmin, vmax = value_range
        else:
            try:
                vmin, vmax = self._finite_minmax(loader.read_slice(0, 0))
            except Exception:
                vmin, vmax = self._dtype_value_range(dtype)
        return self._make_slice_volume_data_fn(
            loader,
            spacing=spacing,
            value_range=(vmin, vmax),
            is_grayscale=is_grayscale,
            is_label_map=is_label_map,
        )

    def _open_zarr_array(self, path: Path):
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
                    if isinstance(arr_obj, (Array, self._zarr_v3_array_cls)) or hasattr(
                        arr_obj, "shape"
                    ):
                        return arr_obj, root
            except Exception as exc:
                errors.append(f"zarr reader failed: {exc}")

        if meta and int(meta.get("zarr_format", 0) or 0) == 3:
            arr_dir = meta_path or path
            try:
                arr_obj = self._zarr_v3_array_cls(arr_dir, meta)
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

    def read_zarr(self, path: Path) -> Any:
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
        use_slice_mode = False
        if bytes_needed > 0:
            over_threshold = (
                self._config.auto_out_of_core_mb > 0
                and (bytes_needed / 1024**2) > self._config.auto_out_of_core_mb
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
            loader = self._zarr_slice_loader_cls(arr_obj, zyx_axes, fixed_idx)
            return self._build_slice_volume_data(
                loader,
                spacing=spacing,
                value_range=value_range,
                is_grayscale=True,
                is_label_map=is_label_volume,
            )

        try:
            vol = self._zarr_to_numpy_zyx(arr_obj, zyx_axes, fixed_idx)
            vmin, vmax = self._finite_minmax(vol)
            return self._make_volume_data_fn(
                array=vol,
                spacing=spacing,
                vmin=vmin,
                vmax=vmax,
                is_grayscale=vol.ndim == 3,
                is_out_of_core=False,
                volume_shape=shape_zyx,
                is_label_map=is_label_volume,
            )
        except (MemoryError, RuntimeError) as exc:
            logger.warning(
                "In-memory Zarr load failed (%s), falling back to slice mode.", exc
            )
            loader = self._zarr_slice_loader_cls(arr_obj, zyx_axes, fixed_idx)
            return self._build_slice_volume_data(
                loader,
                spacing=spacing,
                value_range=value_range,
                is_grayscale=True,
                is_label_map=is_label_volume,
            )

    def read_tiff_eager(self, path: Path) -> Any:
        frames: list[np.ndarray] = []
        with Image.open(str(path)) as img:
            n = max(1, int(getattr(img, "n_frames", 1) or 1))
            for i in range(n):
                img.seek(i)
                frames.append(np.array(img))
        if not frames:
            raise RuntimeError("No frames found in TIFF stack.")
        vol = np.stack(frames, axis=0)
        vmin, vmax = self._finite_minmax(vol)
        return self._make_volume_data_fn(
            array=vol,
            spacing=None,
            vmin=vmin,
            vmax=vmax,
            is_grayscale=vol.ndim == 3,
            is_out_of_core=False,
            volume_shape=tuple(int(x) for x in vol.shape[:3]),
        )

    def read_tiff_out_of_core(self, path: Path) -> Any:
        meta = self._probe_tiff_metadata(path)
        shape = meta[0] if meta else None
        dtype = meta[1] if meta else None
        memmap_arr = self._open_tiff_memmap(path)
        if memmap_arr is not None:
            mem_shape = tuple(int(x) for x in memmap_arr.shape)
            shape = shape or mem_shape
            dtype = dtype or memmap_arr.dtype
            if shape and dtype and self._should_use_slice_mode(shape, dtype):
                loader = self._memmap_slice_loader_cls(memmap_arr)
                logger.info(
                    "Slice mode (memmap) for TIFF stack '%s' (shape=%s, dtype=%s)",
                    path,
                    loader.shape(),
                    loader.dtype(),
                )
                return self._build_slice_volume_data(loader)
            vmin, vmax = self._dtype_value_range(memmap_arr.dtype)
            logger.info(
                "Using tifffile.memmap for TIFF stack '%s' (shape=%s, dtype=%s)",
                path,
                mem_shape,
                memmap_arr.dtype,
            )
            return self._make_volume_data_fn(
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
            loader = self._tiff_slice_loader_cls(path, shape, dtype)
            logger.info(
                "Slice mode (paged) for TIFF stack '%s' (shape=%s, dtype=%s)",
                path,
                shape,
                dtype,
            )
            return self._build_slice_volume_data(loader, spacing=None)
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
        return self._make_volume_data_fn(
            reader,
            None,
            float(min_val),
            float(max_val),
            is_grayscale=True,
            is_out_of_core=True,
            backing_path=backing_path,
            volume_shape=(n_frames, plane_shape[0], plane_shape[1]),
        )

    def read_dicom_series(
        self, directory: Path
    ) -> tuple[np.ndarray, Optional[tuple[float, float, float]]]:
        reader = None
        if self._use_gdcm and self._vtk_gdcm_image_reader_cls is not None:
            try:
                reader = self._vtk_gdcm_image_reader_cls()  # type: ignore[misc]
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
        return vol, spacing
