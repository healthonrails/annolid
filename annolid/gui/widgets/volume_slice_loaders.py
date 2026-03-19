from __future__ import annotations
import zlib
import itertools
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional, Sequence, Any
import numpy as np
from annolid.utils.logger import logger


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

    def __init__(self, array_path: Path, metadata: Mapping[str, Any]):
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
        data: Any = raw
        last_error: Optional[Exception] = None
        for codec in reversed(self._codecs):
            name = codec.get("name") if isinstance(codec, Mapping) else None
            conf = codec.get("configuration", {}) if isinstance(codec, Mapping) else {}
            try:
                if name == "zstd":
                    data = self._decode_zstd(data, expected_bytes)
                elif name in ("gzip", "zlib"):
                    data = zlib.decompress(data)
                elif name == "bytes":
                    endian = conf.get("endian", "<")
                    dt = self.dtype.newbyteorder(
                        "<" if endian in ("little", "<") else ">"
                    )
                    data = np.frombuffer(data, dtype=dt)
                else:
                    raise RuntimeError(f"Unsupported Zarr codec: {name}")
            except Exception as exc:
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
        except Exception:
            arr = np.resize(arr, expected_shape)
        return arr

    def _decode_zstd(self, data: Any, expected_bytes: int) -> bytes:
        """Decode Zstd with fallbacks."""
        last_error: Optional[Exception] = None
        max_out = max(expected_bytes * 2, expected_bytes + 1)
        try:
            import zstandard as zstd

            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data, max_output_size=max_out)
        except Exception as exc:
            last_error = exc

        try:
            import numcodecs

            decoder = None
            try:
                from numcodecs.zstd import Zstd

                decoder = Zstd()
            except Exception:
                try:
                    decoder = numcodecs.get_codec({"id": "zstd"})
                except Exception:
                    decoder = None
            if decoder is not None:
                return decoder.decode(data)
        except Exception as exc:
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
