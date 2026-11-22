"""DeepLabCut & Lightning Pose interoperability (single/multi-animal CSV, H5, config.yaml)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import pickletools
import shutil
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

from annolid.annotation.keypoint_visibility import KeypointVisibility
from annolid.annotation.pose_interop.model import (
    PoseDataset,
    PoseFrame,
    PoseInstance,
    PosePoint,
    dedup_pairs,
    dedup_preserve_order,
    is_finite_number,
)
from annolid.utils.atomic_io import atomic_write_text

_COORD_NAMES = frozenset({"x", "y", "likelihood"})
_INDIVIDUAL_LEVEL_NAMES = frozenset({"individuals", "individual", "animals", "animal"})


@dataclass(frozen=True)
class DLCProjectConfig:
    """Subset of DeepLabCut ``config.yaml`` metadata relevant for pose interop."""

    scorer: str = "annolid"
    multianimal: bool = False
    individuals: Tuple[str, ...] = ()
    bodyparts: Tuple[str, ...] = ()
    skeleton: Tuple[Tuple[str, str], ...] = ()
    config_path: Optional[Path] = None


@dataclass(frozen=True)
class DLCTableInfo:
    """Normalized DeepLabCut / Lightning Pose DataFrame plus parsed schema."""

    df: pd.DataFrame
    multi_animal: bool
    scorer: str
    individuals: Tuple[str, ...]
    bodyparts: Tuple[str, ...]
    has_likelihood: bool
    path_columns: Tuple[Tuple[Any, ...], ...] = ()
    keypoint_columns: Tuple[Tuple[Any, ...], ...] = ()


def detect_dlc_csv_header_levels(csv_path: Path) -> int:
    """Detect whether a DeepLabCut / Lightning Pose CSV has 3 or 4 header rows."""
    try:
        with Path(csv_path).open("r", encoding="utf-8-sig", errors="replace") as fh:
            lines = [fh.readline() for _ in range(5)]
    except OSError:
        return 3

    first_cells = [
        (next(csv.reader([line]), [""])[0] if line else "").strip().lower()
        for line in lines
    ]
    if (
        len(first_cells) >= 4
        and first_cells[0] == "scorer"
        and first_cells[1] in _INDIVIDUAL_LEVEL_NAMES
        and first_cells[2] == "bodyparts"
        and first_cells[3] == "coords"
    ):
        return 4
    if (
        len(first_cells) >= 3
        and first_cells[0] == "scorer"
        and first_cells[1] == "bodyparts"
        and first_cells[2] == "coords"
    ):
        return 3
    return 3


def is_dlc_csv_file(csv_path: Path) -> bool:
    """Return True if ``csv_path`` starts with a DeepLabCut 3- or 4-row multi-index header."""
    try:
        with Path(csv_path).open("r", encoding="utf-8-sig", errors="replace") as fh:
            lines = [fh.readline() for _ in range(4)]
    except OSError:
        return False
    first_cells = [
        (next(csv.reader([line]), [""])[0] if line else "").strip().lower()
        for line in lines
    ]
    if (
        len(first_cells) >= 4
        and first_cells[0] == "scorer"
        and first_cells[1] in _INDIVIDUAL_LEVEL_NAMES
        and first_cells[2] == "bodyparts"
        and first_cells[3] == "coords"
    ):
        return True
    if (
        len(first_cells) >= 3
        and first_cells[0] == "scorer"
        and first_cells[1] == "bodyparts"
        and first_cells[2] == "coords"
    ):
        return True
    return False


def _decode_h5_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_h5_axis_index(group: Any, axis_prefix: str) -> pd.Index:
    """Decode a Pandas HDFStore fixed-format axis (Index or MultiIndex) via h5py."""
    nlevels = group.attrs.get(f"{axis_prefix}_nlevels")
    if nlevels is not None and int(nlevels) > 1:
        levels: List[List[Any]] = []
        codes: List[List[int]] = []
        names: List[Optional[str]] = []
        for lvl in range(int(nlevels)):
            lvl_ds = group[f"{axis_prefix}_level{lvl}"]
            lbl_ds = group[f"{axis_prefix}_label{lvl}"]
            lvl_vals = [_decode_h5_value(v) for v in lvl_ds[()]]
            lbl_vals = [int(c) for c in lbl_ds[()]]
            levels.append(lvl_vals)
            codes.append(lbl_vals)
            raw_name = lvl_ds.attrs.get("name")
            if raw_name is not None:
                decoded = str(_decode_h5_value(raw_name)).strip()
                names.append(None if decoded in ("", "N.") else decoded)
            else:
                names.append(None)
        return pd.MultiIndex(levels=levels, codes=codes, names=names)

    raw_vals = [_decode_h5_value(v) for v in group[axis_prefix][()]]
    return pd.Index(raw_vals)


_MAX_PICKLE_OPS = 200_000
_PICKLE_MARK = object()


def _safe_unpickle_literals(raw: bytes) -> Any:
    """Safely decode a literal-only pickle payload (lists/tuples/dicts/primitives).

    Strictly forbids all object-instantiation and callable opcodes (``GLOBAL``,
    ``STACK_GLOBAL``, ``REDUCE``, ``NEWOBJ``, ``BUILD``, ``INST``, ``OBJ``, etc.)
    so untrusted HDF5 attributes cannot execute arbitrary code.
    """
    stack: List[Any] = []
    memo: Dict[int, Any] = {}
    op_count = 0

    def _pop_mark() -> List[Any]:
        items: List[Any] = []
        while stack:
            top = stack.pop()
            if top is _PICKLE_MARK:
                items.reverse()
                return items
            items.append(top)
        raise ValueError("Malformed pickle attribute: missing MARK sentinel.")

    for opcode, arg, _pos in pickletools.genops(raw):
        op_count += 1
        if op_count > _MAX_PICKLE_OPS:
            raise ValueError(
                f"HDF5 attribute exceeds safe pickle opcode limit ({_MAX_PICKLE_OPS})."
            )
        name = opcode.name
        if name in ("PROTO", "FRAME"):
            continue
        if name == "STOP":
            return stack[-1] if stack else None
        if name == "MARK":
            stack.append(_PICKLE_MARK)
        elif name == "NONE":
            stack.append(None)
        elif name == "NEWTRUE":
            stack.append(True)
        elif name == "NEWFALSE":
            stack.append(False)
        elif name in (
            "INT",
            "BININT",
            "BININT1",
            "BININT2",
            "LONG",
            "LONG1",
            "LONG4",
            "FLOAT",
            "BINFLOAT",
            "STRING",
            "BINSTRING",
            "SHORT_BINSTRING",
            "UNICODE",
            "BINUNICODE",
            "SHORT_BINUNICODE",
            "BINUNICODE8",
            "BINBYTES",
            "SHORT_BINBYTES",
            "BINBYTES8",
            "BYTEARRAY8",
        ):
            stack.append(arg)
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "EMPTY_SET":
            stack.append(set())
        elif name == "LIST":
            stack.append(_pop_mark())
        elif name == "TUPLE":
            stack.append(tuple(_pop_mark()))
        elif name == "TUPLE1":
            a = stack.pop()
            stack.append((a,))
        elif name == "TUPLE2":
            b = stack.pop()
            a = stack.pop()
            stack.append((a, b))
        elif name == "TUPLE3":
            c = stack.pop()
            b = stack.pop()
            a = stack.pop()
            stack.append((a, b, c))
        elif name == "DICT":
            items = _pop_mark()
            d: Dict[Any, Any] = {}
            for i in range(0, len(items) - 1, 2):
                d[items[i]] = items[i + 1]
            stack.append(d)
        elif name == "APPEND":
            val = stack.pop()
            stack[-1].append(val)
        elif name == "APPENDS":
            items = _pop_mark()
            stack[-1].extend(items)
        elif name == "SETITEM":
            val = stack.pop()
            key = stack.pop()
            stack[-1][key] = val
        elif name == "SETITEMS":
            items = _pop_mark()
            target = stack[-1]
            for i in range(0, len(items) - 1, 2):
                target[items[i]] = items[i + 1]
        elif name == "ADDITEMS":
            items = _pop_mark()
            stack[-1].update(items)
        elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
            memo[int(arg)] = stack[-1]
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in ("GET", "BINGET", "LONG_BINGET"):
            stack.append(memo[int(arg)])
        elif name == "POP":
            stack.pop()
        elif name == "POP_MARK":
            _pop_mark()
        elif name == "DUP":
            stack.append(stack[-1])
        else:
            raise ValueError(
                f"Disallowed pickle opcode '{name}' in HDF5 attribute (only primitive literals allowed)."
            )
    return stack[-1] if stack else None


def _decode_pytables_attr(raw: Any) -> Any:
    """Decode a PyTables HDF5 attribute (native value, JSON, or safe literal pickle)."""
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            return _decode_pytables_attr(raw.item())
        return [_decode_pytables_attr(v) for v in raw.tolist()]
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        data = bytes(raw)
        if not data:
            return ""
        # Check for pickle protocol 0-5 markers before falling back to JSON/UTF-8.
        if data[:1] in (b"\x80", b"(", b"]", b"}", b"I", b"S", b"V", b"N", b"c"):
            try:
                return _safe_unpickle_literals(data)
            except ValueError as exc:
                if "Disallowed pickle opcode" in str(exc):
                    raise
            except Exception:
                pass
        text = data.decode("utf-8", errors="replace").strip()
        if text.startswith(("[", "{")):
            try:
                return json.loads(text)
            except Exception:
                pass
        return text
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except Exception:
                pass
        return stripped
    if isinstance(raw, np.generic):
        return raw.item()
    return raw


def _get_group_or_table_attr(group: Any, table_ds: Any, key: str) -> Any:
    if key in group.attrs:
        return _decode_pytables_attr(group.attrs[key])
    if key in table_ds.attrs:
        return _decode_pytables_attr(table_ds.attrs[key])
    return None


def _coerce_column_tuple(col: Any) -> Any:
    if isinstance(col, (list, tuple)):
        return tuple(_decode_h5_value(part) for part in col)
    return _decode_h5_value(col)


def _read_pandas_table_hdf5_group(group: Any, h5_path: Path) -> pd.DataFrame:
    """Read a Pandas PyTables ``format='table'`` HDF5 group using only ``h5py``."""
    table_ds = group["table"]
    try:
        rows = table_ds[()]
    except OSError as exc:
        msg = str(exc).lower()
        if "filter" in msg or "blosc" in msg or "compression" in msg:
            raise ImportError(
                f"{h5_path.name} uses an HDF5 compression filter not installed in h5py ({exc}). "
                "Install 'hdf5plugin' or 'tables', or use the sibling DeepLabCut .csv file."
            ) from exc
        raise

    field_names = list(rows.dtype.names or ())
    if not field_names:
        raise ValueError(
            f"PyTables 'table' dataset in {h5_path.name} has no compound fields."
        )

    levels_attr = _get_group_or_table_attr(group, table_ds, "levels")
    level_fields = [str(lvl) for lvl in (levels_attr or []) if str(lvl) in field_names]

    if level_fields:
        idx_tuples = [
            tuple(_decode_h5_value(rows[lf][i]) for lf in level_fields)
            for i in range(len(rows))
        ]
        index: pd.Index = pd.MultiIndex.from_tuples(idx_tuples)
    elif "index" in field_names:
        index = pd.Index([_decode_h5_value(v) for v in rows["index"]])
    else:
        index = pd.RangeIndex(len(rows))

    # Resolve declared non-index column labels from PyTables metadata.
    non_index_axes = _get_group_or_table_attr(group, table_ds, "non_index_axes") or []
    declared_cols: List[Any] = []
    if isinstance(non_index_axes, list):
        for entry in non_index_axes:
            if (
                isinstance(entry, (list, tuple))
                and len(entry) >= 2
                and isinstance(entry[1], (list, tuple))
            ):
                declared_cols = [_coerce_column_tuple(c) for c in entry[1]]
                break

    values_cols_attr = _get_group_or_table_attr(group, table_ds, "values_cols")
    if isinstance(values_cols_attr, list) and values_cols_attr:
        block_fields = [str(c) for c in values_cols_attr if str(c) in field_names]
    else:
        block_fields = [
            fn for fn in field_names if fn != "index" and fn not in level_fields
        ]

    col_labels: List[Any] = []
    col_arrays: List[np.ndarray] = []
    n_rows = len(rows)

    for bfield in block_fields:
        if bfield in level_fields:
            continue
        raw_arr = np.asarray(rows[bfield])
        if raw_arr.ndim == 1:
            raw_arr = raw_arr.reshape(n_rows, 1)
        kind_cols = _get_group_or_table_attr(group, table_ds, f"{bfield}_kind")
        if isinstance(kind_cols, (list, tuple)):
            block_col_labels = [_coerce_column_tuple(c) for c in kind_cols]
        elif (
            declared_cols
            and len(block_fields) == 1
            and len(declared_cols) == raw_arr.shape[1]
        ):
            block_col_labels = list(declared_cols)
        else:
            block_col_labels = (
                [bfield]
                if raw_arr.shape[1] == 1
                else [f"{bfield}_{j}" for j in range(raw_arr.shape[1])]
            )

        for j in range(min(raw_arr.shape[1], len(block_col_labels))):
            col_labels.append(block_col_labels[j])
            col_arrays.append(raw_arr[:, j])

    if not col_labels:
        raise ValueError(
            f"No value columns could be decoded from PyTables table in {h5_path.name}."
        )

    # Reorder columns to match non_index_axes declaration order when available.
    by_col = {lbl: arr for lbl, arr in zip(col_labels, col_arrays)}
    if declared_cols and all(c in by_col for c in declared_cols):
        ordered_labels = list(declared_cols)
    else:
        ordered_labels = list(col_labels)

    if all(isinstance(c, tuple) for c in ordered_labels):
        columns: pd.Index = pd.MultiIndex.from_tuples(ordered_labels)
    else:
        columns = pd.Index(ordered_labels)

    matrix = np.column_stack(
        [
            pd.to_numeric(pd.Series(by_col[c]), errors="coerce").to_numpy(
                dtype=np.float64
            )
            for c in ordered_labels
        ]
    )
    return pd.DataFrame(matrix, index=index, columns=columns)


def _read_pandas_hdf5_via_h5py(h5_path: Path) -> pd.DataFrame:
    """Read a DeepLabCut Pandas HDF5 file (``fixed`` or ``table`` format) using only ``h5py``."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading DeepLabCut .h5 files requires 'h5py' (or 'tables'). "
            "Install with: pip install 'annolid[sleap]' or export a .csv from DeepLabCut."
        ) from exc

    with h5py.File(h5_path, "r") as h5:
        target_group = None
        for preferred in ("df_with_missing", "tracks", "keypoints", "df"):
            if preferred in h5 and isinstance(h5[preferred], h5py.Group):
                target_group = h5[preferred]
                break
        if target_group is None:
            for key in h5.keys():
                obj = h5[key]
                if isinstance(obj, h5py.Group) and (
                    "axis0" in obj or "axis0_level0" in obj or "table" in obj
                ):
                    target_group = obj
                    break
        if target_group is None:
            raise ValueError(
                f"Could not find a DeepLabCut DataFrame group in {h5_path}. "
                f"Root keys: {list(h5.keys())}"
            )

        if "table" in target_group and "block0_values" not in target_group:
            return _read_pandas_table_hdf5_group(target_group, h5_path)

        columns = _read_h5_axis_index(target_group, "axis0")
        index = _read_h5_axis_index(target_group, "axis1")
        n_rows = len(index)
        n_cols = len(columns)

        data = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        n_blocks = int(target_group.attrs.get("nblocks", 0) or 0)
        if n_blocks <= 0:
            block_ids = sorted(
                int(k[5 : k.index("_values")])
                for k in target_group.keys()
                if k.startswith("block") and k.endswith("_values")
            )
        else:
            block_ids = list(range(n_blocks))

        if not block_ids:
            raise ValueError(f"No data blocks found in DeepLabCut HDF5 file: {h5_path}")

        for b_idx in block_ids:
            vals_key = f"block{b_idx}_values"
            items_key = f"block{b_idx}_items"
            if vals_key not in target_group:
                continue
            try:
                vals = np.asarray(target_group[vals_key][()])
            except OSError as exc:
                msg = str(exc).lower()
                if "filter" in msg or "blosc" in msg or "compression" in msg:
                    raise ImportError(
                        f"{h5_path.name} uses an HDF5 compression filter not installed in h5py ({exc}). "
                        "Install 'hdf5plugin' or 'tables', or use the sibling DeepLabCut .csv file."
                    ) from exc
                raise
            if vals.ndim == 1:
                vals = vals.reshape(n_rows, -1)
            # Pandas fixed HDF5 stores values as (n_rows, n_block_cols) or transposed.
            if vals.shape[0] != n_rows and vals.shape[1] == n_rows:
                vals = vals.T

            if items_key in target_group:
                raw_items = target_group[items_key][()]
                if np.issubdtype(np.asarray(raw_items).dtype, np.integer):
                    col_positions = [int(i) for i in raw_items]
                else:
                    # Match by column position sequentially when labels are stored directly.
                    col_positions = list(range(vals.shape[1]))
            else:
                col_positions = list(range(vals.shape[1]))

            for src_j, dst_col in enumerate(col_positions):
                if 0 <= dst_col < n_cols and src_j < vals.shape[1]:
                    try:
                        data[:, dst_col] = vals[:, src_j].astype(np.float64)
                    except (TypeError, ValueError):
                        pass

        return pd.DataFrame(data, index=index, columns=columns)


# Backward-compatible internal alias
_read_pandas_fixed_hdf5_via_h5py = _read_pandas_hdf5_via_h5py


def read_dlc_raw_dataframe(
    path: str | Path,
    *,
    for_training: bool = False,
) -> pd.DataFrame:
    """Read a DeepLabCut / Lightning Pose CSV or H5 file into a MultiIndex DataFrame."""
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"DeepLabCut file not found: {target}")

    suffix = target.suffix.lower()
    if suffix in (".h5", ".hdf", ".hdf5"):
        try:
            df = pd.read_hdf(target)
            if isinstance(df, pd.DataFrame):
                return df
        except ImportError:
            pass
        except Exception:
            # Fall back to pure-h5py fixed-format reader before propagating error.
            pass
        return _read_pandas_fixed_hdf5_via_h5py(target)

    header_levels = detect_dlc_csv_header_levels(target)
    if for_training:
        df = pd.read_csv(target, header=list(range(header_levels)))
    else:
        df = pd.read_csv(target, header=list(range(header_levels)), index_col=0)
    return df


def _clean_multiindex_columns(columns: pd.Index) -> pd.MultiIndex:
    if not isinstance(columns, pd.MultiIndex):
        raise ValueError(
            "Expected a 3- or 4-level MultiIndex header for DeepLabCut data."
        )
    cleaned_tuples = [
        tuple(str(part).strip() for part in col) for col in columns.tolist()
    ]
    nlevels = columns.nlevels
    if nlevels == 4:
        names = ["scorer", "individuals", "bodyparts", "coords"]
    elif nlevels == 3:
        names = ["scorer", "bodyparts", "coords"]
    else:
        raise ValueError(
            f"Unsupported DeepLabCut column header depth: expected 3 or 4 levels, got {nlevels}."
        )
    return pd.MultiIndex.from_tuples(cleaned_tuples, names=names)


def parse_dlc_table(
    path: str | Path,
    *,
    is_multi_animal: Optional[bool] = None,
    for_training: bool = False,
) -> DLCTableInfo:
    """Load and normalize a DeepLabCut / Lightning Pose CSV or H5 table."""
    df = read_dlc_raw_dataframe(path, for_training=for_training)
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels not in (3, 4):
        raise ValueError(
            f"Not a valid DeepLabCut multi-index table ({Path(path).name}): "
            f"expected 3 or 4 header levels."
        )

    df = df.copy()
    df.columns = _clean_multiindex_columns(df.columns)
    nlevels = df.columns.nlevels
    detected_multi = nlevels == 4
    if is_multi_animal is not None and bool(is_multi_animal) != detected_multi:
        # Honor the actual file structure rather than crashing with a KeyError,
        # while allowing explicit single-animal hints only when compatible.
        multi_animal = detected_multi
    else:
        multi_animal = detected_multi

    kp_cols: List[Tuple[Any, ...]] = []
    path_cols: List[Tuple[Any, ...]] = []
    scorers: List[str] = []
    individuals: List[str] = []
    bodyparts: List[str] = []
    has_likelihood = False

    for col in df.columns:
        coord = str(col[-1]).strip().lower()
        scorer = str(col[0]).strip()
        if coord not in _COORD_NAMES:
            path_cols.append(tuple(col))
            continue
        if (
            not scorer
            or scorer.lower().startswith("unnamed:")
            or scorer.lower() == "scorer"
        ):
            path_cols.append(tuple(col))
            continue
        bp = str(col[-2]).strip()
        if not bp or bp.lower().startswith("unnamed:"):
            continue
        if scorer not in scorers:
            scorers.append(scorer)
        if multi_animal:
            ind = str(col[1]).strip()
            if (
                ind
                and not ind.lower().startswith("unnamed:")
                and ind not in individuals
            ):
                individuals.append(ind)
        if bp not in bodyparts:
            bodyparts.append(bp)
        if coord == "likelihood":
            has_likelihood = True
        kp_cols.append(tuple(col))

    if not kp_cols:
        raise ValueError(
            f"No DeepLabCut coordinate columns (x/y/likelihood) found in {Path(path).name}."
        )

    return DLCTableInfo(
        df=df,
        multi_animal=multi_animal,
        scorer=scorers[0] if scorers else "annolid",
        individuals=tuple(individuals),
        bodyparts=tuple(bodyparts),
        has_likelihood=has_likelihood,
        path_columns=tuple(path_cols),
        keypoint_columns=tuple(kp_cols),
    )


def read_dlc_project_config(config_path: str | Path) -> Optional[DLCProjectConfig]:
    """Parse a DeepLabCut ``config.yaml`` if it exists."""
    p = Path(config_path).expanduser()
    if not p.is_file():
        return None
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    scorer = str(raw.get("scorer") or "annolid").strip() or "annolid"
    multianimal = bool(raw.get("multianimalproject", False))

    raw_individuals = raw.get("individuals") or []
    individuals = dedup_preserve_order(
        raw_individuals if isinstance(raw_individuals, list) else []
    )
    if len(individuals) > 1:
        multianimal = True

    bp_list: List[str] = []
    for key in ("multianimalbodyparts", "bodyparts", "uniquebodyparts"):
        val = raw.get(key)
        if isinstance(val, list):
            bp_list.extend(str(item).strip() for item in val if str(item).strip())
    bodyparts = dedup_preserve_order(bp_list)

    skeleton_edges: List[Tuple[str, str]] = []
    raw_skel = raw.get("skeleton")
    if isinstance(raw_skel, list):
        for pair in raw_skel:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                a = str(pair[0]).strip()
                b = str(pair[1]).strip()
                if a and b and a != b:
                    skeleton_edges.append((a, b))

    return DLCProjectConfig(
        scorer=scorer,
        multianimal=multianimal,
        individuals=tuple(individuals),
        bodyparts=tuple(bodyparts),
        skeleton=tuple(dedup_pairs(skeleton_edges)),
        config_path=p,
    )


def discover_dlc_config_near(path: str | Path) -> Optional[DLCProjectConfig]:
    """Search ``path`` and up to 2 ancestor directories for a DLC ``config.yaml``."""
    start = Path(path).expanduser().resolve()
    search_dirs = [start if start.is_dir() else start.parent]
    for parent in list(search_dirs[0].parents)[:2]:
        search_dirs.append(parent)
    for folder in search_dirs:
        candidate = folder / "config.yaml"
        cfg = read_dlc_project_config(candidate)
        if cfg is not None and (cfg.bodyparts or cfg.skeleton or cfg.individuals):
            return cfg
    return None


def find_dlc_analysis_file_for_video(video_path: str | Path) -> Optional[Path]:
    """Locate a matching DeepLabCut / Lightning Pose analysis CSV or H5 beside ``video_path``.

    Search priority:
    1. Exact sibling ``<video_stem>.csv`` (preserves legacy Annolid contract)
    2. Filtered DLC CSV ``<video_stem>DLC_*filtered*.csv``
    3. Standard DLC CSV ``<video_stem>DLC_*.csv``
    4. Exact sibling ``<video_stem>.h5``
    5. Filtered DLC H5 ``<video_stem>DLC_*filtered*.h5``
    6. Standard DLC H5 ``<video_stem>DLC_*.h5``
    """
    vp = Path(video_path).expanduser()
    parent = vp.parent
    stem = vp.stem
    if not parent.is_dir():
        return None

    exact_csv = vp.with_suffix(".csv")
    if exact_csv.is_file() and is_dlc_csv_file(exact_csv):
        return exact_csv

    dlc_csvs = sorted(
        p for p in parent.glob(f"{stem}DLC_*.csv") if p.is_file() and is_dlc_csv_file(p)
    )
    filtered_csvs = [p for p in dlc_csvs if "filtered" in p.stem.lower()]
    if filtered_csvs:
        return filtered_csvs[-1]
    if dlc_csvs:
        return dlc_csvs[-1]

    # Lightning Pose sometimes names predictions `<video_stem>_predictions.csv`
    lp_csvs = sorted(
        p for p in parent.glob(f"{stem}*.csv") if p.is_file() and is_dlc_csv_file(p)
    )
    if lp_csvs:
        return lp_csvs[0]

    exact_h5 = vp.with_suffix(".h5")
    if exact_h5.is_file():
        return exact_h5

    dlc_h5s = sorted(p for p in parent.glob(f"{stem}DLC_*.h5") if p.is_file())
    filtered_h5s = [p for p in dlc_h5s if "filtered" in p.stem.lower()]
    if filtered_h5s:
        return filtered_h5s[-1]
    if dlc_h5s:
        return dlc_h5s[-1]

    return None


def _probe_video_dimensions(
    video_path: Optional[str | Path],
) -> Tuple[Optional[int], Optional[int]]:
    if not video_path:
        return None, None
    vp = Path(video_path).expanduser()
    if not vp.is_file():
        return None, None
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise OSError(f"Error opening video file: {vp}")
    try:
        ret, frame = cap.read()
        if ret and frame is not None and hasattr(frame, "shape"):
            h, w = int(frame.shape[0]), int(frame.shape[1])
            if w > 0 and h > 0:
                return w, h
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w > 0 and h > 0:
            return w, h
        raise OSError(f"Could not read the first frame from video: {vp}")
    finally:
        cap.release()


def read_dlc_analysis(
    source_path: str | Path,
    *,
    video_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    min_likelihood: Optional[float] = None,
    is_multi_animal: Optional[bool] = None,
    default_instance: str = "mouse",
) -> PoseDataset:
    """Read a DeepLabCut / Lightning Pose analysis CSV or H5 into a ``PoseDataset``."""
    table_path = Path(source_path).expanduser().resolve()
    table = parse_dlc_table(
        table_path, is_multi_animal=is_multi_animal, for_training=False
    )

    cfg = (
        read_dlc_project_config(config_path)
        if config_path
        else discover_dlc_config_near(table_path)
    )
    img_w, img_h = _probe_video_dimensions(video_path)

    df = table.df
    index_numeric = pd.to_numeric(df.index, errors="coerce")
    use_positional = bool(index_numeric.isna().any())
    frame_indices = (
        list(range(len(df)))
        if use_positional
        else [int(v) for v in index_numeric.tolist()]
    )

    # Pre-group column tuples per (individual, bodypart) for fast O(N) extraction.
    if table.multi_animal:
        instance_order = list(table.individuals)
    else:
        inst_default = str(default_instance or "mouse").strip() or "mouse"
        instance_order = [inst_default]

    col_lookup: Dict[Tuple[str, str], Dict[str, Tuple[Any, ...]]] = {}
    for col in table.keypoint_columns:
        coord = str(col[-1]).strip().lower()
        if table.multi_animal:
            ind = str(col[1]).strip()
            bp = str(col[2]).strip()
        else:
            ind = instance_order[0]
            bp = str(col[1]).strip()
        col_lookup.setdefault((ind, bp), {})[coord] = col

    # Extract numpy arrays per column once to avoid slow per-cell pandas lookups.
    col_arrays: Dict[Tuple[Any, ...], np.ndarray] = {}
    for col in table.keypoint_columns:
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        col_arrays[col] = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )

    min_like_val = (
        float(min_likelihood)
        if min_likelihood is not None and is_finite_number(min_likelihood)
        else None
    )

    video_name = Path(video_path).stem if video_path else table_path.stem
    frames: List[PoseFrame] = []
    warnings: List[str] = []
    if use_positional:
        warnings.append(
            f"{table_path.name} has a non-numeric index; mapped rows to sequential frame indices."
        )

    for row_idx, frame_idx in enumerate(frame_indices):
        frame_instances: List[PoseInstance] = []
        for track_idx, ind in enumerate(instance_order):
            kps: Dict[str, PosePoint] = {}
            scores: List[float] = []
            for bp in table.bodyparts:
                coord_cols = col_lookup.get((ind, bp))
                if not coord_cols or "x" not in coord_cols or "y" not in coord_cols:
                    continue
                x_val = float(col_arrays[coord_cols["x"]][row_idx])
                y_val = float(col_arrays[coord_cols["y"]][row_idx])
                if not (math.isfinite(x_val) and math.isfinite(y_val)):
                    continue
                like_val: Optional[float] = None
                if "likelihood" in coord_cols:
                    raw_like = float(col_arrays[coord_cols["likelihood"]][row_idx])
                    if math.isfinite(raw_like):
                        like_val = raw_like
                if (
                    min_like_val is not None
                    and like_val is not None
                    and like_val < min_like_val
                ):
                    continue
                if like_val is not None:
                    scores.append(like_val)
                kps[bp] = PosePoint(
                    x=x_val,
                    y=y_val,
                    score=like_val,
                    visibility=int(KeypointVisibility.VISIBLE),
                )
            if kps:
                inst_score = float(np.mean(scores)) if scores else None
                frame_instances.append(
                    PoseInstance(
                        instance_name=ind,
                        keypoints=kps,
                        track_id=track_idx if table.multi_animal else None,
                        score=inst_score,
                        is_predicted=table.has_likelihood,
                    )
                )
        frames.append(
            PoseFrame(
                frame_index=int(frame_idx),
                instances=frame_instances,
                image_width=img_w,
                image_height=img_h,
                video_index=0,
                video_name=video_name,
            )
        )

    edges = list(cfg.skeleton) if cfg and cfg.skeleton else []
    ordered_kps = list(table.bodyparts)
    if cfg and cfg.bodyparts:
        # Preserve config bodypart order first, then any additional columns.
        ordered_kps = dedup_preserve_order(list(cfg.bodyparts) + ordered_kps)

    return PoseDataset(
        frames=frames,
        keypoints=ordered_kps,
        edges=edges,
        instances=list(instance_order) if table.multi_animal else [instance_order[0]],
        scorer=table.scorer,
        multi_animal=table.multi_animal,
        source_format="deeplabcut_h5"
        if table_path.suffix.lower() in (".h5", ".hdf", ".hdf5")
        else "deeplabcut_csv",
        video_paths={0: str(Path(video_path).expanduser())} if video_path else {},
        metadata={
            "source_path": str(table_path),
            "has_likelihood": table.has_likelihood,
            "config_path": str(cfg.config_path) if cfg and cfg.config_path else None,
        },
        warnings=warnings,
    )


def write_dlc_analysis_csv(
    dataset: PoseDataset,
    output_csv: str | Path,
    *,
    scorer: Optional[str] = None,
    multi_animal: Optional[bool] = None,
    default_likelihood: float = 1.0,
) -> Path:
    """Export a ``PoseDataset`` to a DeepLabCut / Lightning Pose / SimBA-compatible CSV.

    Produces:
    - 3 header rows (``scorer``, ``bodyparts``, ``coords``) for single-animal data, or
    - 4 header rows (``scorer``, ``individuals``, ``bodyparts``, ``coords``) for multi-animal data.
    """
    out_path = Path(output_csv).expanduser().resolve()
    keypoints = dataset.resolved_keypoints()
    if not keypoints:
        raise ValueError("Cannot export DeepLabCut CSV: dataset contains no keypoints.")

    scorer_name = str(scorer or dataset.scorer or "annolid").strip() or "annolid"
    use_multi = (
        bool(multi_animal) if multi_animal is not None else dataset.is_multi_animal()
    )
    instances = dataset.resolved_instances()
    if not instances:
        instances = ["individual1"]

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    if use_multi:
        row_scorer = ["scorer"]
        row_ind = ["individuals"]
        row_bp = ["bodyparts"]
        row_coords = ["coords"]
        for ind in instances:
            for bp in keypoints:
                for coord in ("x", "y", "likelihood"):
                    row_scorer.append(scorer_name)
                    row_ind.append(ind)
                    row_bp.append(bp)
                    row_coords.append(coord)
        writer.writerow(row_scorer)
        writer.writerow(row_ind)
        writer.writerow(row_bp)
        writer.writerow(row_coords)
    else:
        row_scorer = ["scorer"]
        row_bp = ["bodyparts"]
        row_coords = ["coords"]
        for bp in keypoints:
            for coord in ("x", "y", "likelihood"):
                row_scorer.append(scorer_name)
                row_bp.append(bp)
                row_coords.append(coord)
        writer.writerow(row_scorer)
        writer.writerow(row_bp)
        writer.writerow(row_coords)

    sorted_frames = sorted(dataset.frames, key=lambda f: (f.video_index, f.frame_index))
    for frame in sorted_frames:
        row: List[Any] = [int(frame.frame_index)]
        by_inst: Dict[str, PoseInstance] = {}
        for inst in frame.instances:
            name = str(inst.instance_name or "").strip() or instances[0]
            if name not in by_inst:
                by_inst[name] = inst

        target_instances = instances if use_multi else [instances[0]]
        for ind in target_instances:
            inst_obj = by_inst.get(ind)
            if inst_obj is None and not use_multi and frame.instances:
                inst_obj = frame.instances[0]
            for bp in keypoints:
                pt = inst_obj.keypoints.get(bp) if inst_obj is not None else None
                if pt is not None and pt.is_valid:
                    like = (
                        float(pt.score)
                        if pt.score is not None and math.isfinite(float(pt.score))
                        else float(default_likelihood)
                    )
                    row.extend(
                        [f"{float(pt.x):.4f}", f"{float(pt.y):.4f}", f"{like:.6f}"]
                    )
                else:
                    row.extend(["", "", ""])
        writer.writerow(row)

    atomic_write_text(out_path, buf.getvalue(), encoding="utf-8")
    return out_path


def write_dlc_training_project(
    dataset: PoseDataset,
    output_dir: str | Path,
    *,
    scorer: Optional[str] = None,
    video_name: str = "video",
    copy_images: bool = True,
    write_config_yaml: bool = True,
    multi_animal: Optional[bool] = None,
) -> Dict[str, Any]:
    """Export a ``PoseDataset`` to a DeepLabCut / Lightning Pose training project.

    Writes:
    - ``<output_dir>/labeled-data/<video_name>/CollectedData_<scorer>.csv``
    - Sidecar images under ``<output_dir>/labeled-data/<video_name>/`` when source images exist
    - ``<output_dir>/config.yaml`` with bodyparts, individuals, and skeleton edges
    """
    out_root = Path(output_dir).expanduser().resolve()
    keypoints = dataset.resolved_keypoints()
    if not keypoints:
        raise ValueError(
            "Cannot export DeepLabCut project: dataset contains no keypoints."
        )

    scorer_name = str(scorer or dataset.scorer or "annolid").strip() or "annolid"
    use_multi = (
        bool(multi_animal) if multi_animal is not None else dataset.is_multi_animal()
    )
    instances = dataset.resolved_instances()
    if not instances:
        instances = ["individual1"]

    video_folder_name = str(video_name or "video").strip() or "video"
    labeled_dir = out_root / "labeled-data" / video_folder_name
    labeled_dir.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    if use_multi:
        row_scorer = ["scorer", "", ""]
        row_ind = ["individuals", "", ""]
        row_bp = ["bodyparts", "", ""]
        row_coords = ["coords", "", ""]
        for ind in instances:
            for bp in keypoints:
                for coord in ("x", "y"):
                    row_scorer.append(scorer_name)
                    row_ind.append(ind)
                    row_bp.append(bp)
                    row_coords.append(coord)
        writer.writerow(row_scorer)
        writer.writerow(row_ind)
        writer.writerow(row_bp)
        writer.writerow(row_coords)
    else:
        row_scorer = ["scorer", "", ""]
        row_bp = ["bodyparts", "", ""]
        row_coords = ["coords", "", ""]
        for bp in keypoints:
            for coord in ("x", "y"):
                row_scorer.append(scorer_name)
                row_bp.append(bp)
                row_coords.append(coord)
        writer.writerow(row_scorer)
        writer.writerow(row_bp)
        writer.writerow(row_coords)

    copied_images = 0
    rows_written = 0
    sorted_frames = sorted(dataset.frames, key=lambda f: (f.video_index, f.frame_index))
    for frame in sorted_frames:
        img_name = (
            Path(frame.image_path).name
            if frame.image_path
            else f"img{int(frame.frame_index):06d}.png"
        )
        if copy_images and frame.source_image_path:
            src_img = Path(frame.source_image_path)
            if src_img.is_file():
                dst_img = labeled_dir / img_name
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
                    copied_images += 1

        row: List[Any] = ["labeled-data", video_folder_name, img_name]
        by_inst: Dict[str, PoseInstance] = {}
        for inst in frame.instances:
            name = str(inst.instance_name or "").strip() or instances[0]
            if name not in by_inst:
                by_inst[name] = inst

        target_instances = instances if use_multi else [instances[0]]
        for ind in target_instances:
            inst_obj = by_inst.get(ind)
            if inst_obj is None and not use_multi and frame.instances:
                inst_obj = frame.instances[0]
            for bp in keypoints:
                pt = inst_obj.keypoints.get(bp) if inst_obj is not None else None
                if pt is not None and pt.is_valid:
                    row.extend([f"{float(pt.x):.4f}", f"{float(pt.y):.4f}"])
                else:
                    row.extend(["", ""])
        writer.writerow(row)
        rows_written += 1

    csv_out = labeled_dir / f"CollectedData_{scorer_name}.csv"
    atomic_write_text(csv_out, buf.getvalue(), encoding="utf-8")

    config_out: Optional[Path] = None
    if write_config_yaml:
        schema = dataset.to_pose_schema(preset="auto")
        config_payload: Dict[str, Any] = {
            "Task": out_root.name,
            "scorer": scorer_name,
            "multianimalproject": bool(use_multi),
            "project_path": str(out_root),
            "skeleton": [[a, b] for a, b in schema.edges],
            "skeleton_color": "black",
            "pcutoff": 0.6,
            "dotsize": 8,
            "alphavalue": 0.7,
            "colormap": "rainbow",
        }
        if use_multi:
            config_payload["individuals"] = list(instances)
            config_payload["multianimalbodyparts"] = list(keypoints)
            config_payload["uniquebodyparts"] = []
            config_payload["bodyparts"] = "MULTI!"
        else:
            config_payload["bodyparts"] = list(keypoints)
        config_out = out_root / "config.yaml"
        atomic_write_text(
            config_out,
            yaml.safe_dump(config_payload, sort_keys=False),
            encoding="utf-8",
        )

    return {
        "output_dir": str(out_root),
        "collected_data_csv": str(csv_out),
        "config_yaml": str(config_out) if config_out else None,
        "frames_exported": rows_written,
        "images_copied": copied_images,
        "multi_animal": bool(use_multi),
        "keypoints": list(keypoints),
        "instances": list(instances),
    }
