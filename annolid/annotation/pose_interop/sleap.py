"""Bidirectional SLEAP interoperability (.slp, .pkg.slp, and .analysis.h5)."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import zipfile

import cv2
import numpy as np

from annolid.annotation.keypoint_visibility import KeypointVisibility
from annolid.annotation.pose_interop.model import (
    HDF5_MAGIC,
    ZIP_MAGIC,
    PoseDataset,
    PoseFrame,
    PoseInstance,
    PosePoint,
    dedup_pairs,
    dedup_preserve_order,
    sanitize_filename_token,
)

# Matches sleap-io InstanceType enum
INSTANCE_TYPE_USER = 0
INSTANCE_TYPE_PREDICTED = 1


def require_h5py() -> Any:
    """Import h5py, invoking Annolid's optional capability helper if needed."""
    try:
        import h5py

        return h5py
    except ImportError:
        try:
            from annolid.infrastructure.capabilities import ensure_capability

            ensure_capability("sleap", reason="SLEAP HDF5 (.slp/.h5) import/export")
            import h5py

            return h5py
        except Exception as exc:
            raise ImportError(
                "Missing optional dependency: h5py. Install with: pip install 'annolid[sleap]'"
            ) from exc


def _require_pil_image() -> Any:
    try:
        from PIL import Image

        return Image
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependency: pillow. Install with: pip install pillow"
        ) from exc


def detect_sleap_container(path: str | Path) -> str:
    """Detect whether a SLEAP file is an HDF5 container (``'h5'``) or ZIP (``'zip'``)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("rb") as fh:
        magic = fh.read(8)
    if magic.startswith(HDF5_MAGIC):
        return "h5"
    if magic.startswith(ZIP_MAGIC):
        return "zip"
    raise ValueError(f"Unrecognized SLEAP file container for {p} (magic={magic!r}).")


def detect_sleap_h5_kind(path: str | Path) -> str:
    """Distinguish SLEAP ``.slp``/``.pkg.slp`` (``'slp'``), Analysis HDF5 (``'analysis_h5'``), and ZIP ``.slp`` (``'zip_slp'``)."""
    p = Path(path).expanduser().resolve()
    kind = detect_sleap_container(p)
    if kind == "zip":
        return "zip_slp"
    h5py = require_h5py()
    with h5py.File(p, "r") as h5:
        keys = set(h5.keys())
    if {"frames", "instances", "points"}.issubset(keys):
        return "slp"
    if "tracks" in keys and "node_names" in keys:
        return "analysis_h5"
    raise ValueError(
        f"Unrecognized SLEAP HDF5 structure in {p.name}. "
        f"Expected either /frames+/instances+/points (.slp) or /tracks+/node_names (analysis .h5). "
        f"Found root keys: {sorted(keys)}"
    )


def _decode_str(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (bytes, bytearray, np.bytes_, np.void)):
        return bytes(raw).decode("utf-8", errors="ignore").strip("\x00").strip()
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        return _decode_str(raw.item())
    return str(raw).strip()


def safe_json_loads(raw: object) -> Optional[object]:
    """Safely parse a JSON string or bytes payload from an HDF5 dataset or attr."""
    if raw is None:
        return None
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, (bytes, bytearray, np.bytes_, np.void)):
        try:
            raw = bytes(raw).decode("utf-8", errors="ignore")
        except Exception:
            return None
    if isinstance(raw, str):
        text = raw.strip("\x00").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None
    return None


@dataclass(frozen=True)
class SkeletonSpec:
    names: List[str]
    edges: List[Tuple[str, str]]
    symmetry_pairs: List[Tuple[str, str]] = None  # type: ignore[assignment]
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.symmetry_pairs is None:
            object.__setattr__(self, "symmetry_pairs", [])


@dataclass(frozen=True)
class InstanceRecord:
    video_index: int
    frame_idx: int
    track_id: int
    skeleton_id: int
    points_xy: np.ndarray
    visible: Optional[np.ndarray] = None
    is_predicted: bool = False
    point_scores: Optional[np.ndarray] = None
    instance_score: Optional[float] = None
    track_name: Optional[str] = None


def _collect_jsonpickle_objects(obj: Any) -> Dict[int, Any]:
    """Collect ``py/object`` dicts by 1-based traversal ID (matching jsonpickle)."""
    table: Dict[int, Any] = {}
    counter = 0

    def walk(node: Any) -> None:
        nonlocal counter
        if isinstance(node, dict):
            if "py/id" in node:
                return
            if "py/object" in node:
                counter += 1
                table[counter] = node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return table


def _parse_sleap_metadata_skeletons(meta: object) -> Optional[List[SkeletonSpec]]:
    """Parse skeletons from SLEAP ``/metadata`` JSON (modern sleap-io & legacy jsonpickle)."""
    if not isinstance(meta, dict):
        return None

    nodes = meta.get("nodes")
    skels = meta.get("skeletons")
    if not (isinstance(nodes, list) and isinstance(skels, list) and skels):
        return None

    py_id_table = _collect_jsonpickle_objects(meta)

    global_names: List[str] = []
    node_id_to_name: Dict[int, str] = {}
    for idx, node in enumerate(nodes):
        if isinstance(node, dict) and "name" in node:
            nm = str(node.get("name", "")).strip() or f"node_{idx}"
        else:
            nm = f"node_{idx}"
        global_names.append(nm)
        node_id_to_name[idx] = nm

    def resolve_node_ref(ref: Any) -> Tuple[Optional[Any], Optional[str]]:
        if ref is None:
            return None, None
        if isinstance(ref, (int, np.integer)):
            idx = int(ref)
            return ("id", idx), node_id_to_name.get(idx, f"node_{idx}")
        if isinstance(ref, dict):
            if "id" in ref and isinstance(ref["id"], (int, np.integer)):
                idx = int(ref["id"])
                return ("id", idx), node_id_to_name.get(idx, f"node_{idx}")
            if "py/id" in ref and isinstance(ref["py/id"], (int, np.integer)):
                pid = int(ref["py/id"])
                target = py_id_table.get(pid)
                if isinstance(target, dict) and "name" in target:
                    return ("py/id", pid), str(target["name"]).strip() or f"node_{pid}"
                # Fallback: older writers sometimes used py/id directly as 0-based index
                if 0 <= pid < len(global_names):
                    return ("py/id", pid), global_names[pid]
                if 1 <= pid <= len(global_names):
                    return ("py/id", pid), global_names[pid - 1]
                return ("py/id", pid), f"node_{pid}"
            if "name" in ref:
                nm = str(ref.get("name") or "").strip()
                if nm:
                    return ("name", nm), nm
        return None, None

    out: List[SkeletonSpec] = []
    for sidx, sk in enumerate(skels):
        if not isinstance(sk, dict):
            continue
        sk_name = None
        graph = sk.get("graph")
        if isinstance(graph, dict) and graph.get("name"):
            sk_name = str(graph.get("name")).strip() or None

        sk_nodes = sk.get("nodes", [])
        if not isinstance(sk_nodes, list) or not sk_nodes:
            continue

        local_names: List[str] = []
        ref_to_local_name: Dict[Any, str] = {}
        for node_entry in sk_nodes:
            key, name = resolve_node_ref(node_entry)
            if key is None or name is None:
                continue
            local_names.append(name)
            ref_to_local_name[key] = name
            # Also index by raw integer when available for cross-referencing links.
            if isinstance(key, tuple) and key[0] in ("id", "py/id"):
                ref_to_local_name[key[1]] = name

        if not local_names:
            continue

        edges: List[Tuple[str, str]] = []
        sym: List[Tuple[str, str]] = []
        links = sk.get("links", [])
        if isinstance(links, list):
            for lk in links:
                if not isinstance(lk, dict):
                    continue
                src_key, src_name = resolve_node_ref(lk.get("source"))
                dst_key, dst_name = resolve_node_ref(lk.get("target"))
                a = ref_to_local_name.get(src_key) or (
                    ref_to_local_name.get(src_key[1])
                    if isinstance(src_key, tuple)
                    else src_name
                )
                b = ref_to_local_name.get(dst_key) or (
                    ref_to_local_name.get(dst_key[1])
                    if isinstance(dst_key, tuple)
                    else dst_name
                )
                if not a or not b or a == b:
                    continue
                # Link type 2 = symmetry edge, type 1 = skeleton edge
                link_type = lk.get("type")
                if isinstance(link_type, dict):
                    link_type = link_type.get("py/tuple", [None])[0]
                if link_type == 2 or str(link_type).lower() == "symmetry":
                    sym.append((a, b))
                else:
                    edges.append((a, b))

        out.append(
            SkeletonSpec(
                names=local_names,
                edges=dedup_pairs(edges),
                symmetry_pairs=dedup_pairs(sym),
                name=sk_name or f"skeleton_{sidx}",
            )
        )

    return out or None


class SleapH5Reader:
    """Reader for SLEAP HDF5 ``.slp`` and ``.pkg.slp`` files."""

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path).expanduser().resolve()
        self._h5py = require_h5py()
        self._h5 = self._h5py.File(self.h5_path, "r")

    def __enter__(self) -> "SleapH5Reader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def root_keys(self) -> List[str]:
        return list(self._h5.keys())

    def _require_ds(self, key: str) -> Any:
        if key not in self._h5:
            raise KeyError(
                f"Missing dataset/group: {key}. Root keys: {self.root_keys()}"
            )
        obj = self._h5[key]
        if not isinstance(obj, self._h5py.Dataset):
            raise TypeError(f"{key} is not a dataset (found {type(obj)}).")
        return obj

    def _maybe_ds(self, key: str) -> Optional[Any]:
        obj = self._h5.get(key, None)
        return obj if isinstance(obj, self._h5py.Dataset) else None

    def _maybe_grp(self, key: str) -> Optional[Any]:
        obj = self._h5.get(key, None)
        return obj if isinstance(obj, self._h5py.Group) else None

    def _read_metadata_obj(self) -> Optional[object]:
        meta_ds = self._maybe_ds("metadata")
        if meta_ds is not None:
            try:
                obj = safe_json_loads(meta_ds[()])
                if obj is not None:
                    return obj
            except Exception:
                pass

        meta_grp = self._maybe_grp("metadata")
        if meta_grp is not None:
            try:
                for _, v in meta_grp.attrs.items():
                    obj = safe_json_loads(v)
                    if obj is not None:
                        return obj
            except Exception:
                pass

        for k in self.root_keys():
            if k.endswith("_json") and k not in ("videos_json", "tracks_json"):
                ds = self._maybe_ds(k)
                if ds is None:
                    continue
                try:
                    obj = safe_json_loads(ds[()])
                    if obj is not None:
                        return obj
                except Exception:
                    continue
        return None

    @staticmethod
    def _get_id(x: Any) -> Optional[int]:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, dict):
            for k in ("id", "py/id"):
                if k in x and isinstance(x[k], (int, np.integer)):
                    return int(x[k])
        return None

    @staticmethod
    def _parse_sleap_metadata_skeletons(meta: object) -> Optional[List[SkeletonSpec]]:
        return _parse_sleap_metadata_skeletons(meta)

    def read_skeletons(self) -> List[SkeletonSpec]:
        meta = self._read_metadata_obj()
        skels = _parse_sleap_metadata_skeletons(meta) if meta is not None else None
        if skels:
            return skels

        inst = self._require_ds("instances")
        if inst.shape[0] == 0:
            raise ValueError("No instances found in file; cannot infer skeleton.")
        row0 = inst[0]
        n_nodes = max(0, int(row0["point_id_end"]) - int(row0["point_id_start"]))
        if n_nodes <= 0:
            raise ValueError(
                "Could not infer node count from instances[0].point_id_start/end."
            )
        return [
            SkeletonSpec(
                names=[f"node_{i}" for i in range(n_nodes)],
                edges=[],
                symmetry_pairs=[],
                name="skeleton_0",
            )
        ]

    def read_track_names(self) -> List[str]:
        """Return track names indexed by track_id from ``/tracks_json`` when present."""
        ds = self._maybe_ds("tracks_json")
        if ds is None:
            return []
        names: List[str] = []
        try:
            count = int(ds.shape[0])
        except Exception:
            return []
        for i in range(count):
            raw = ds[i]
            parsed = safe_json_loads(raw)
            name = ""
            if isinstance(parsed, list) and len(parsed) >= 2:
                # Standard sleap-io tracks_json entry: [spawned_on, "track_name"]
                name = str(parsed[1] or "").strip()
            elif isinstance(parsed, dict):
                name = str(parsed.get("name") or "").strip()
            elif isinstance(parsed, str):
                name = parsed.strip()
            else:
                name = _decode_str(raw)
            names.append(name or f"track_{i}")
        return names

    def _read_video_metadata_entries(self) -> List[Dict[str, Any]]:
        ds = self._maybe_ds("videos_json")
        if ds is None:
            return []
        entries: List[Dict[str, Any]] = []
        try:
            count = int(ds.shape[0])
        except Exception:
            return []
        for i in range(count):
            obj = safe_json_loads(ds[i])
            entries.append(obj if isinstance(obj, dict) else {})
        return entries

    def video_backend_info(self, video_index: int) -> Dict[str, Any]:
        """Return the ``backend`` metadata dict for ``video_index`` from ``/videos_json``."""
        entries = self._read_video_metadata_entries()
        target_grp = f"video{int(video_index)}"
        # Prefer explicit dataset group match (important for .pkg.slp files).
        for entry in entries:
            backend = entry.get("backend")
            if not isinstance(backend, dict):
                continue
            dataset = backend.get("dataset")
            if isinstance(dataset, str) and dataset.strip():
                if dataset.strip().split("/", 1)[0] == target_grp:
                    return backend
        # Fall back to positional lookup (standard for non-packaged .slp files).
        if 0 <= int(video_index) < len(entries):
            backend = entries[int(video_index)].get("backend")
            if isinstance(backend, dict):
                return backend
        return {}

    def video_shape_wh(self, video_index: int) -> Optional[Tuple[int, int]]:
        """Return ``(width, height)`` from ``/videos_json`` backend shape metadata if present."""
        backend = self.video_backend_info(video_index)
        shape = backend.get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) >= 3:
            try:
                h = int(shape[1])
                w = int(shape[2])
                if w > 0 and h > 0:
                    return w, h
            except (TypeError, ValueError):
                return None
        return None

    def video_filename(self, video_index: int) -> Optional[str]:
        backend = self.video_backend_info(video_index)
        filename = backend.get("filename")
        if isinstance(filename, str):
            cleaned = filename.strip()
            if cleaned and cleaned != ".":
                return cleaned
        return None

    def has_embedded_frames(self, video_index: int) -> bool:
        return self._maybe_ds(f"video{int(video_index)}/video") is not None

    def used_video_indices(self) -> List[int]:
        """Return sorted unique video indices referenced by ``/frames``."""
        frames = self._require_ds("frames")
        frame_fields = set(frames.dtype.fields.keys()) if frames.dtype.fields else set()
        if "video" not in frame_fields or frames.shape[0] == 0:
            return [0]
        vids = np.array(frames["video"][:], dtype=np.int64)
        return sorted({int(v) for v in np.unique(vids)})

    def video_name(self, video_index: int) -> str:
        """Best-effort stable name for ``video_index``."""
        fallback = f"video{int(video_index)}"
        filename = self.video_filename(video_index)
        if filename:
            # Handle both POSIX and Windows separators in stored SLEAP video paths.
            stem = Path(filename.replace("\\", "/")).stem
            return sanitize_filename_token(stem, fallback=fallback)
        return fallback

    def iter_instances(self, skeletons: List[SkeletonSpec]) -> Iterable[InstanceRecord]:
        frames = self._require_ds("frames")
        inst = self._require_ds("instances")
        pts = self._require_ds("points")
        pred_pts = self._maybe_ds("pred_points")
        track_names = self.read_track_names()

        pt_fields = set(pts.dtype.fields.keys()) if pts.dtype.fields else set()
        if not {"x", "y"}.issubset(pt_fields):
            raise KeyError(
                f"/points missing x/y columns. Found fields: {sorted(pt_fields)}"
            )

        inst_fields = set(inst.dtype.fields.keys()) if inst.dtype.fields else set()
        has_instance_type = "instance_type" in inst_fields
        has_skeleton_id = "skeleton" in inst_fields
        has_inst_score = "score" in inst_fields

        frame_fields = set(frames.dtype.fields.keys()) if frames.dtype.fields else set()
        has_video_idx = "video" in frame_fields

        for fr in frames:
            video_index = int(fr["video"]) if has_video_idx else 0
            frame_idx = int(fr["frame_idx"])
            i0 = int(fr["instance_id_start"])
            i1 = int(fr["instance_id_end"])
            if i1 <= i0:
                continue

            for inst_row in inst[i0:i1]:
                track_id = int(inst_row["track"]) if "track" in inst_fields else -1
                track_name = (
                    track_names[track_id] if 0 <= track_id < len(track_names) else None
                )
                sk_id = int(inst_row["skeleton"]) if has_skeleton_id else 0
                sk_id = sk_id if 0 <= sk_id < len(skeletons) else 0
                sk = skeletons[sk_id]

                ps = int(inst_row["point_id_start"])
                pe = int(inst_row["point_id_end"])
                n = pe - ps
                if n <= 0:
                    continue

                is_pred = False
                point_ds = pts
                if has_instance_type:
                    if int(inst_row["instance_type"]) == INSTANCE_TYPE_PREDICTED:
                        is_pred = True
                        if pred_pts is None:
                            raise KeyError(
                                "Found predicted instances but /pred_points dataset is missing."
                            )
                        point_ds = pred_pts

                pfields = (
                    set(point_ds.dtype.fields.keys())
                    if point_ds.dtype.fields
                    else set()
                )
                if not {"x", "y"}.issubset(pfields):
                    raise KeyError(
                        f"Point dataset missing x/y columns. Found fields: {sorted(pfields)}"
                    )

                xs = point_ds["x"][ps:pe].astype(np.float64, copy=False)
                ys = point_ds["y"][ps:pe].astype(np.float64, copy=False)
                xy = np.stack([xs, ys], axis=1)

                vis = None
                if "visible" in pfields:
                    vis = point_ds["visible"][ps:pe].astype(bool, copy=False)

                pt_scores = None
                if "score" in pfields:
                    pt_scores = point_ds["score"][ps:pe].astype(np.float64, copy=False)

                inst_score = None
                if has_inst_score:
                    raw_s = float(inst_row["score"])
                    if math.isfinite(raw_s):
                        inst_score = raw_s

                if n != len(sk.names):
                    raise ValueError(
                        "Point slice length does not match skeleton node count. "
                        f"slice n={n}, skeleton[{sk_id}] nodes={len(sk.names)}. "
                        "This is unexpected for standard .slp written by SLEAP/sleap-io."
                    )

                yield InstanceRecord(
                    video_index=video_index,
                    frame_idx=frame_idx,
                    track_id=track_id,
                    skeleton_id=sk_id,
                    points_xy=xy,
                    visible=vis,
                    is_predicted=is_pred,
                    point_scores=pt_scores,
                    instance_score=inst_score,
                    track_name=track_name,
                )

    def save_embedded_frames(
        self,
        out_dir: Path,
        video_index: int = 0,
        image_ext: str = ".png",
        name_prefix: str = "",
    ) -> Dict[int, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        video_key = f"video{video_index}/video"
        frame_nums_key = f"video{video_index}/frame_numbers"

        video_ds = self._maybe_ds(video_key)
        frame_nums_ds = self._maybe_ds(frame_nums_key)
        if video_ds is None:
            raise KeyError(
                f"Embedded frames not found at /{video_key}. Available root keys: {self.root_keys()}"
            )

        if frame_nums_ds is None:
            frame_numbers = np.arange(int(video_ds.shape[0]), dtype=np.int64)
        else:
            frame_numbers = np.array(frame_nums_ds[:], dtype=np.int64)

        mapping: Dict[int, str] = {}
        n = int(video_ds.shape[0])
        prefix = f"{name_prefix}_" if name_prefix else ""

        for i in range(n):
            frame_idx = int(frame_numbers[i])
            obj = video_ds[i]
            try:
                img = self._decode_embedded_frame(obj)
            except Exception as e:
                print(
                    f"[sleap2labelme] WARN: could not decode frame {frame_idx} (i={i}): {e}"
                )
                continue

            fname = f"{prefix}{frame_idx:09d}{image_ext}"
            fpath = out_dir / fname
            img.save(fpath)
            mapping[frame_idx] = fname

        return mapping

    @staticmethod
    def _decode_embedded_frame(obj: object) -> Any:
        Image = _require_pil_image()

        def try_decode_bytes(b: bytes) -> Optional[Any]:
            if not b:
                return None
            b2 = b.strip(b"\x00")
            for payload in (b, b2) if b2 != b else (b,):
                try:
                    im = Image.open(io.BytesIO(payload))
                    return im.convert("RGB")
                except Exception:
                    pass
            return None

        if isinstance(obj, (bytes, bytearray, np.void)):
            im = try_decode_bytes(bytes(obj))
            if im is not None:
                return im

        if isinstance(obj, np.ndarray):
            arr = obj
            if arr.dtype == object and arr.ndim == 0:
                return SleapH5Reader._decode_embedded_frame(arr.item())

            if arr.ndim == 1 and arr.dtype in (np.int8, np.uint8):
                b = arr.astype(np.uint8, copy=False).tobytes()
                im = try_decode_bytes(b)
                if im is not None:
                    return im
                raise ValueError(
                    f"Embedded frame looks like a 1D byte buffer but could not decode (len={arr.shape[0]}, dtype={arr.dtype})."
                )

            if arr.ndim == 2:
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return Image.fromarray(arr, mode="L").convert("RGB")

            if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                if arr.shape[2] == 4:
                    return Image.fromarray(arr, mode="RGBA").convert("RGB")
                if arr.shape[2] == 1:
                    return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
                return Image.fromarray(arr, mode="RGB")

            raise ValueError(
                f"Unsupported embedded frame ndarray shape={arr.shape}, dtype={arr.dtype}"
            )

        if isinstance(obj, memoryview):
            im = try_decode_bytes(obj.tobytes())
            if im is not None:
                return im

        raise ValueError(f"Unsupported embedded frame type: {type(obj)}")


def _extract_video_frames(
    video_path: Path,
    frame_indices: Sequence[int],
    out_dir: Path,
    *,
    name_prefix: str = "",
    image_ext: str = ".png",
) -> Tuple[Dict[int, str], Dict[int, Tuple[int, int]]]:
    """Extract specific frame indices from an external video file."""
    mapping: Dict[int, str] = {}
    sizes: Dict[int, Tuple[int, int]] = {}
    if not video_path.is_file() or not frame_indices:
        return mapping, sizes

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return mapping, sizes

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{name_prefix}_" if name_prefix else ""
    try:
        for fidx in sorted(set(int(f) for f in frame_indices if int(f) >= 0)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fidx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            fname = f"{prefix}{fidx:09d}{image_ext}"
            fpath = out_dir / fname
            if cv2.imwrite(str(fpath), frame):
                h, w = int(frame.shape[0]), int(frame.shape[1])
                mapping[fidx] = fname
                sizes[fidx] = (w, h)
    finally:
        cap.release()
    return mapping, sizes


def _normalize_sleap_tracks_array(tracks: np.ndarray, n_nodes: int) -> np.ndarray:
    """Normalize SLEAP Analysis HDF5 ``/tracks`` to shape ``(n_frames, n_tracks, n_nodes, 2)``."""
    arr = np.asarray(tracks, dtype=np.float64)
    if arr.ndim == 3:
        # Single-track shortcut: (2, n_nodes, n_frames) or (n_frames, n_nodes, 2)
        if arr.shape[0] == 2 and arr.shape[1] == n_nodes:
            return np.transpose(arr, (2, 1, 0))[:, None, :, :]
        if arr.shape[2] == 2 and arr.shape[1] == n_nodes:
            return arr[:, None, :, :]
        raise ValueError(
            f"Unsupported 3D SLEAP /tracks shape {arr.shape} for {n_nodes} nodes."
        )
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D SLEAP /tracks array, got shape {arr.shape}.")
    # Standard SLEAP analysis HDF5 layout: (n_tracks, 2, n_nodes, n_frames)
    if arr.shape[1] == 2 and arr.shape[2] == n_nodes:
        return np.transpose(arr, (3, 0, 2, 1))
    # Alternative Python row-major layout: (n_frames, n_tracks, n_nodes, 2)
    if arr.shape[3] == 2 and arr.shape[2] == n_nodes:
        return arr
    # Transposed alternative: (n_frames, n_nodes, 2, n_tracks)
    if arr.shape[2] == 2 and arr.shape[1] == n_nodes:
        return np.transpose(arr, (0, 3, 1, 2))
    # Fallback when n_nodes == 2 makes axes ambiguous: prefer standard SLEAP layout
    if arr.shape[1] == 2:
        return np.transpose(arr, (3, 0, 2, 1))
    if arr.shape[3] == 2:
        return arr
    raise ValueError(
        f"Cannot infer axis order for SLEAP /tracks shape {arr.shape} with n_nodes={n_nodes}."
    )


def _normalize_sleap_point_scores(
    scores: Optional[np.ndarray], n_frames: int, n_tracks: int, n_nodes: int
) -> Optional[np.ndarray]:
    if scores is None:
        return None
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 3:
        return None
    # Standard SLEAP analysis HDF5: (n_tracks, n_nodes, n_frames)
    if arr.shape == (n_tracks, n_nodes, n_frames):
        return np.transpose(arr, (2, 0, 1))
    if arr.shape == (n_frames, n_tracks, n_nodes):
        return arr
    if arr.shape == (n_frames, n_nodes, n_tracks):
        return np.transpose(arr, (0, 2, 1))
    return None


def _normalize_sleap_instance_scores(
    scores: Optional[np.ndarray], n_frames: int, n_tracks: int
) -> Optional[np.ndarray]:
    if scores is None:
        return None
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2:
        return None
    # Standard SLEAP analysis HDF5: (n_tracks, n_frames)
    if arr.shape == (n_tracks, n_frames):
        return arr.T
    if arr.shape == (n_frames, n_tracks):
        return arr
    return None


def read_sleap_analysis_h5(
    h5_path: str | Path,
    *,
    video_path: Optional[str | Path] = None,
    min_score: Optional[float] = None,
) -> PoseDataset:
    """Read a SLEAP Analysis HDF5 file (``/tracks``, ``/node_names``) into a ``PoseDataset``."""
    h5py = require_h5py()
    path = Path(h5_path).expanduser().resolve()

    with h5py.File(path, "r") as h5:
        if "tracks" not in h5 or "node_names" not in h5:
            raise KeyError(
                f"{path.name} is not a SLEAP Analysis HDF5 file (requires /tracks and /node_names)."
            )
        node_names = [
            _decode_str(v) or f"node_{i}" for i, v in enumerate(h5["node_names"][:])
        ]
        n_nodes = len(node_names)
        tracks_ftn2 = _normalize_sleap_tracks_array(h5["tracks"][:], n_nodes)
        n_frames, n_tracks, _, _ = tracks_ftn2.shape

        if "track_names" in h5:
            track_names = [
                _decode_str(v) or f"track_{i}"
                for i, v in enumerate(h5["track_names"][:])
            ]
        else:
            track_names = [f"track_{i}" for i in range(n_tracks)]
        while len(track_names) < n_tracks:
            track_names.append(f"track_{len(track_names)}")

        edges: List[Tuple[str, str]] = []
        if "edge_names" in h5:
            for pair in h5["edge_names"][:]:
                if len(pair) == 2:
                    a = _decode_str(pair[0])
                    b = _decode_str(pair[1])
                    if a and b and a != b:
                        edges.append((a, b))
        elif "edge_inds" in h5:
            for pair in h5["edge_inds"][:]:
                if len(pair) == 2:
                    ia, ib = int(pair[0]), int(pair[1])
                    if 0 <= ia < n_nodes and 0 <= ib < n_nodes and ia != ib:
                        edges.append((node_names[ia], node_names[ib]))

        pt_scores = _normalize_sleap_point_scores(
            h5["point_scores"][:] if "point_scores" in h5 else None,
            n_frames,
            n_tracks,
            n_nodes,
        )
        inst_scores = _normalize_sleap_instance_scores(
            h5["instance_scores"][:] if "instance_scores" in h5 else None,
            n_frames,
            n_tracks,
        )
        embedded_video_path = (
            _decode_str(h5["video_path"][()]) if "video_path" in h5 else ""
        )

    resolved_video = str(video_path) if video_path else embedded_video_path
    img_w: Optional[int] = None
    img_h: Optional[int] = None
    if resolved_video and Path(resolved_video).expanduser().is_file():
        cap = cv2.VideoCapture(str(Path(resolved_video).expanduser()))
        if cap.isOpened():
            try:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if w > 0 and h > 0:
                    img_w, img_h = w, h
            finally:
                cap.release()

    min_score_f = float(min_score) if min_score is not None else None
    video_stem = (
        sanitize_filename_token(
            Path(resolved_video.replace("\\", "/")).stem, fallback=path.stem
        )
        if resolved_video
        else sanitize_filename_token(path.stem, fallback="video0")
    )

    frames: List[PoseFrame] = []
    for fidx in range(n_frames):
        frame_instances: List[PoseInstance] = []
        for tidx in range(n_tracks):
            kps: Dict[str, PosePoint] = {}
            for nidx, kp_name in enumerate(node_names):
                x = float(tracks_ftn2[fidx, tidx, nidx, 0])
                y = float(tracks_ftn2[fidx, tidx, nidx, 1])
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                sc: Optional[float] = None
                if pt_scores is not None:
                    raw_sc = float(pt_scores[fidx, tidx, nidx])
                    if math.isfinite(raw_sc):
                        sc = raw_sc
                if min_score_f is not None and sc is not None and sc < min_score_f:
                    continue
                kps[kp_name] = PosePoint(
                    x=x,
                    y=y,
                    score=sc,
                    visibility=int(KeypointVisibility.VISIBLE),
                )
            if not kps:
                continue
            iscore: Optional[float] = None
            if inst_scores is not None:
                raw_is = float(inst_scores[fidx, tidx])
                if math.isfinite(raw_is):
                    iscore = raw_is
            frame_instances.append(
                PoseInstance(
                    instance_name=track_names[tidx],
                    keypoints=kps,
                    track_id=tidx,
                    score=iscore,
                    is_predicted=True,
                )
            )
        if frame_instances:
            frames.append(
                PoseFrame(
                    frame_index=int(fidx),
                    instances=frame_instances,
                    image_width=img_w,
                    image_height=img_h,
                    video_index=0,
                    video_name=video_stem,
                )
            )

    return PoseDataset(
        frames=frames,
        keypoints=list(node_names),
        edges=dedup_pairs(edges),
        instances=list(track_names[:n_tracks]),
        scorer="sleap",
        multi_animal=n_tracks > 1,
        source_format="sleap_analysis_h5",
        video_paths={0: resolved_video} if resolved_video else {},
        metadata={"source_path": str(path)},
    )


def read_sleap_slp(
    slp_path: str | Path,
    *,
    video_index: Optional[int] = None,
    prefer_user_instances: bool = True,
    include_predicted: bool = True,
    include_occluded: bool = True,
    min_score: Optional[float] = None,
) -> PoseDataset:
    """Read a SLEAP ``.slp`` or ``.pkg.slp`` HDF5 file into a ``PoseDataset``."""
    path = Path(slp_path).expanduser().resolve()
    with SleapH5Reader(path) as reader:
        skeletons = reader.read_skeletons()
        if not skeletons:
            raise ValueError(f"No skeletons found in {path.name}.")

        used_videos = reader.used_video_indices()
        if video_index is not None:
            if int(video_index) not in set(used_videos):
                raise ValueError(
                    f"video_index={video_index} is not present in /frames. "
                    f"Available video indices: {used_videos}"
                )
            selected_videos = [int(video_index)]
        else:
            selected_videos = used_videos
        selected_set = set(selected_videos)

        video_names = {vid: reader.video_name(vid) for vid in selected_videos}
        if len(selected_videos) > 1:
            counts: Dict[str, int] = {}
            for vid in selected_videos:
                counts[video_names[vid]] = counts.get(video_names[vid], 0) + 1
            for vid in selected_videos:
                if counts[video_names[vid]] > 1:
                    video_names[vid] = f"{video_names[vid]}_{vid}"

        video_paths = {
            vid: fname
            for vid in selected_videos
            if (fname := reader.video_filename(vid)) is not None
        }
        video_shapes = {vid: reader.video_shape_wh(vid) for vid in selected_videos}

        grouped: Dict[Tuple[int, int], List[InstanceRecord]] = {}
        for rec in reader.iter_instances(skeletons):
            if rec.video_index not in selected_set:
                continue
            if rec.is_predicted and not include_predicted:
                continue
            grouped.setdefault((rec.video_index, rec.frame_idx), []).append(rec)

        min_score_f = float(min_score) if min_score is not None else None
        frames: List[PoseFrame] = []
        observed_instances: List[str] = []

        for vid, fidx in sorted(grouped.keys()):
            recs = grouped[(vid, fidx)]
            if prefer_user_instances and any(not r.is_predicted for r in recs):
                recs = [r for r in recs if not r.is_predicted]

            frame_instances: List[PoseInstance] = []
            for local_idx, rec in enumerate(recs):
                sk = skeletons[rec.skeleton_id]
                kps: Dict[str, PosePoint] = {}
                for j, (x, y) in enumerate(rec.points_xy):
                    if j >= len(sk.names):
                        continue
                    xf, yf = float(x), float(y)
                    if not (math.isfinite(xf) and math.isfinite(yf)):
                        continue
                    is_vis = True if rec.visible is None else bool(rec.visible[j])
                    if not is_vis and not include_occluded:
                        continue
                    sc: Optional[float] = None
                    if rec.point_scores is not None and j < len(rec.point_scores):
                        raw_sc = float(rec.point_scores[j])
                        if math.isfinite(raw_sc):
                            sc = raw_sc
                    if (
                        rec.is_predicted
                        and min_score_f is not None
                        and sc is not None
                        and sc < min_score_f
                    ):
                        continue
                    vis_code = (
                        int(KeypointVisibility.VISIBLE)
                        if is_vis
                        else int(KeypointVisibility.OCCLUDED)
                    )
                    kps[sk.names[j]] = PosePoint(
                        x=xf,
                        y=yf,
                        score=sc,
                        visibility=vis_code,
                    )
                if not kps:
                    continue

                if rec.track_name:
                    inst_name = rec.track_name
                elif rec.track_id >= 0:
                    inst_name = f"track_{rec.track_id}"
                else:
                    inst_name = f"instance_{local_idx}"
                if inst_name not in observed_instances:
                    observed_instances.append(inst_name)

                frame_instances.append(
                    PoseInstance(
                        instance_name=inst_name,
                        keypoints=kps,
                        track_id=int(rec.track_id) if rec.track_id >= 0 else None,
                        skeleton_name=sk.name,
                        score=rec.instance_score,
                        is_predicted=rec.is_predicted,
                    )
                )

            wh = video_shapes.get(vid)
            frames.append(
                PoseFrame(
                    frame_index=int(fidx),
                    instances=frame_instances,
                    image_width=wh[0] if wh else None,
                    image_height=wh[1] if wh else None,
                    video_index=int(vid),
                    video_name=video_names[vid],
                )
            )

        primary_sk = skeletons[0]
        all_kps = dedup_preserve_order(kp for sk in skeletons for kp in sk.names)
        all_edges = dedup_pairs(edge for sk in skeletons for edge in sk.edges)
        all_sym = dedup_pairs(
            pair for sk in skeletons for pair in (sk.symmetry_pairs or [])
        )
        track_names = reader.read_track_names()

        return PoseDataset(
            frames=frames,
            keypoints=all_kps or list(primary_sk.names),
            edges=all_edges,
            symmetry_pairs=all_sym,
            instances=dedup_preserve_order(track_names + observed_instances),
            scorer="sleap",
            multi_animal=len(track_names) > 1
            or any(len(f.instances) > 1 for f in frames),
            source_format="sleap_slp",
            video_paths=video_paths,
            metadata={
                "source_path": str(path),
                "skeletons": [
                    {
                        "name": sk.name,
                        "names": list(sk.names),
                        "edges": list(sk.edges),
                        "symmetry_pairs": list(sk.symmetry_pairs or []),
                    }
                    for sk in skeletons
                ],
                "used_videos": used_videos,
                "video_names": video_names,
            },
        )


_MAX_ZIP_JSON_BYTES = 256 * 1024 * 1024
_MAX_ZIP_HDF5_BYTES = 2 * 1024 * 1024 * 1024
_MAX_ZIP_IMAGE_BYTES = 64 * 1024 * 1024
_ZIP_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
_ZIP_FRAME_NUM_RE = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")


def _is_safe_zip_member_name(name: str) -> bool:
    normalized = str(name or "").replace("\\", "/")
    if not normalized or normalized.startswith("/"):
        return False
    parts = [p for p in normalized.split("/") if p]
    if not parts or any(part == ".." for part in parts):
        return False
    if ":" in parts[0]:
        return False
    return True


def _read_zip_member_bounded(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, *, max_bytes: int
) -> bytes:
    if not _is_safe_zip_member_name(info.filename):
        raise ValueError(f"Unsafe path in ZIP archive: {info.filename!r}")
    if info.file_size > max_bytes:
        raise ValueError(
            f"ZIP member '{info.filename}' exceeds safety size limit ({max_bytes} bytes)."
        )
    chunks: List[bytes] = []
    total = 0
    with zf.open(info, "r") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"Decompressed ZIP member '{info.filename}' exceeds safety limit ({max_bytes} bytes)."
                )
            chunks.append(chunk)
    return b"".join(chunks)


def _find_zip_hdf5_member(zf: zipfile.ZipFile) -> Optional[zipfile.ZipInfo]:
    preferred: List[zipfile.ZipInfo] = []
    other: List[zipfile.ZipInfo] = []
    for info in zf.infolist():
        if info.is_dir() or not _is_safe_zip_member_name(info.filename):
            continue
        try:
            with zf.open(info, "r") as fh:
                magic = fh.read(8)
        except Exception:
            continue
        if magic.startswith(HDF5_MAGIC):
            if info.filename.lower().endswith((".slp", ".h5", ".hdf5")):
                preferred.append(info)
            else:
                other.append(info)
    if preferred:
        return sorted(preferred, key=lambda x: x.filename)[0]
    if other:
        return sorted(other, key=lambda x: x.filename)[0]
    return None


def _parse_sleap_json_labels(
    payload: Dict[str, Any],
    *,
    source_path: Path,
    video_index: Optional[int] = None,
    prefer_user_instances: bool = True,
    include_predicted: bool = True,
    include_occluded: bool = True,
    min_score: Optional[float] = None,
) -> PoseDataset:
    """Parse SLEAP JSON labels payload (from legacy JSON-in-ZIP ``.slp`` archives)."""
    py_id_table = _collect_jsonpickle_objects(payload)

    def resolve_ref(obj: Any) -> Any:
        if isinstance(obj, dict) and "py/id" in obj:
            pid = obj["py/id"]
            if isinstance(pid, int) and pid in py_id_table:
                return py_id_table[pid]
        return obj

    skeletons = _parse_sleap_metadata_skeletons(payload)
    if not skeletons:
        # Handle inline skeletons where nodes are embedded directly inside each skeleton.
        raw_skels = payload.get("skeletons") or []
        inferred_skels: List[SkeletonSpec] = []
        if isinstance(raw_skels, list):
            for sidx, sk_raw in enumerate(raw_skels):
                sk = resolve_ref(sk_raw)
                if not isinstance(sk, dict):
                    continue
                raw_nodes = sk.get("nodes") or []
                names: List[str] = []
                for n_i, n_entry in enumerate(raw_nodes):
                    n_obj = resolve_ref(n_entry)
                    if isinstance(n_obj, dict) and n_obj.get("name"):
                        names.append(str(n_obj["name"]).strip())
                    elif isinstance(n_obj, str) and n_obj.strip():
                        names.append(n_obj.strip())
                    else:
                        names.append(f"node_{n_i}")
                if not names:
                    continue
                edges: List[Tuple[str, str]] = []
                for edge in sk.get("edges") or []:
                    e_obj = resolve_ref(edge)
                    if isinstance(e_obj, dict):
                        src = resolve_ref(e_obj.get("source"))
                        dst = resolve_ref(
                            e_obj.get("destination") or e_obj.get("target")
                        )
                        a = (
                            str(src.get("name", "")).strip()
                            if isinstance(src, dict)
                            else ""
                        )
                        b = (
                            str(dst.get("name", "")).strip()
                            if isinstance(dst, dict)
                            else ""
                        )
                        if a and b and a != b:
                            edges.append((a, b))
                sym_pairs: List[Tuple[str, str]] = []
                for sym in sk.get("symmetries") or []:
                    s_obj = resolve_ref(sym)
                    nodes_pair = (
                        s_obj.get("nodes") if isinstance(s_obj, dict) else s_obj
                    )
                    if isinstance(nodes_pair, (list, tuple)) and len(nodes_pair) == 2:
                        na = resolve_ref(nodes_pair[0])
                        nb = resolve_ref(nodes_pair[1])
                        a = (
                            str(na.get("name", "")).strip()
                            if isinstance(na, dict)
                            else str(na).strip()
                        )
                        b = (
                            str(nb.get("name", "")).strip()
                            if isinstance(nb, dict)
                            else str(nb).strip()
                        )
                        if a and b and a != b:
                            sym_pairs.append((a, b))
                sk_name = str(sk.get("name") or "").strip() or f"skeleton_{sidx}"
                inferred_skels.append(
                    SkeletonSpec(
                        names=names,
                        edges=dedup_pairs(edges),
                        symmetry_pairs=dedup_pairs(sym_pairs),
                        name=sk_name,
                    )
                )
        skeletons = inferred_skels
    if not skeletons:
        raise ValueError(
            f"No skeletons found in SLEAP JSON payload inside {source_path.name}."
        )

    # Parse videos
    raw_videos = payload.get("videos") or []
    video_entries: List[Dict[str, Any]] = []
    video_obj_to_idx: Dict[int, int] = {}
    if isinstance(raw_videos, list):
        for v_i, v_raw in enumerate(raw_videos):
            v_obj = resolve_ref(v_raw)
            video_entries.append(v_obj if isinstance(v_obj, dict) else {})
            if isinstance(v_obj, dict):
                video_obj_to_idx[id(v_obj)] = v_i

    def _video_filename_for(v_i: int) -> Optional[str]:
        if 0 <= v_i < len(video_entries):
            v_obj = video_entries[v_i]
            backend = (
                v_obj.get("backend") if isinstance(v_obj.get("backend"), dict) else {}
            )
            fname = backend.get("filename") or v_obj.get("filename")
            if isinstance(fname, str) and fname.strip() and fname.strip() != ".":
                return fname.strip()
        return None

    # Parse tracks
    raw_tracks = payload.get("tracks") or []
    track_names: List[str] = []
    track_obj_to_idx: Dict[int, int] = {}
    if isinstance(raw_tracks, list):
        for t_i, t_raw in enumerate(raw_tracks):
            t_obj = resolve_ref(t_raw)
            name = ""
            if isinstance(t_obj, dict):
                name = str(t_obj.get("name") or "").strip()
                track_obj_to_idx[id(t_obj)] = t_i
            elif isinstance(t_obj, list) and len(t_obj) >= 2:
                name = str(t_obj[1] or "").strip()
            elif isinstance(t_obj, str):
                name = t_obj.strip()
            track_names.append(name or f"track_{t_i}")

    raw_frames = payload.get("labeled_frames") or payload.get("frames") or []
    if not isinstance(raw_frames, list):
        raise ValueError(f"Expected list of labeled_frames in {source_path.name}.")

    min_score_f = float(min_score) if min_score is not None else None
    grouped_frames: Dict[Tuple[int, int], List[PoseInstance]] = {}
    used_vids_set: set[int] = set()
    observed_instances: List[str] = []

    for lf_raw in raw_frames:
        lf = resolve_ref(lf_raw)
        if not isinstance(lf, dict):
            continue
        fidx_raw = lf.get("frame_idx", lf.get("frame_index", 0))
        try:
            fidx = int(fidx_raw)
        except (TypeError, ValueError):
            continue

        vid = 0
        v_ref = lf.get("video")
        if isinstance(v_ref, (int, np.integer)):
            vid = int(v_ref)
        elif isinstance(v_ref, dict):
            v_resolved = resolve_ref(v_ref)
            if isinstance(v_resolved, dict) and id(v_resolved) in video_obj_to_idx:
                vid = video_obj_to_idx[id(v_resolved)]
            elif "id" in v_ref and isinstance(v_ref["id"], int):
                vid = int(v_ref["id"])
        used_vids_set.add(vid)
        if video_index is not None and vid != int(video_index):
            continue

        raw_insts = lf.get("instances") or []
        if not isinstance(raw_insts, list):
            continue

        parsed_insts: List[PoseInstance] = []
        for local_idx, inst_raw in enumerate(raw_insts):
            inst = resolve_ref(inst_raw)
            if not isinstance(inst, dict):
                continue

            py_obj_name = str(inst.get("py/object") or "")
            is_pred = (
                "predicted" in py_obj_name.lower()
                or int(inst.get("instance_type", 0) or 0) == INSTANCE_TYPE_PREDICTED
                or bool(inst.get("is_predicted", False))
            )
            if is_pred and not include_predicted:
                continue

            sk_id = 0
            sk_ref = inst.get("skeleton")
            if isinstance(sk_ref, (int, np.integer)):
                sk_id = int(sk_ref)
            elif isinstance(sk_ref, dict) and isinstance(sk_ref.get("id"), int):
                sk_id = int(sk_ref["id"])
            sk_id = sk_id if 0 <= sk_id < len(skeletons) else 0
            sk = skeletons[sk_id]

            track_id: Optional[int] = None
            track_name: Optional[str] = None
            t_ref = inst.get("track")
            if isinstance(t_ref, (int, np.integer)) and int(t_ref) >= 0:
                track_id = int(t_ref)
                if track_id < len(track_names):
                    track_name = track_names[track_id]
            elif isinstance(t_ref, dict):
                t_resolved = resolve_ref(t_ref)
                if isinstance(t_resolved, dict):
                    if id(t_resolved) in track_obj_to_idx:
                        track_id = track_obj_to_idx[id(t_resolved)]
                    if t_resolved.get("name"):
                        track_name = str(t_resolved["name"]).strip()
                    elif track_id is not None and track_id < len(track_names):
                        track_name = track_names[track_id]
            elif isinstance(t_ref, str) and t_ref.strip():
                track_name = t_ref.strip()

            raw_iscore = inst.get("score")
            inst_score: Optional[float] = None
            if raw_iscore is not None:
                try:
                    sf = float(raw_iscore)
                    if math.isfinite(sf):
                        inst_score = sf
                except (TypeError, ValueError):
                    pass

            kps: Dict[str, PosePoint] = {}
            raw_pts = inst.get("points")
            pt_pairs: List[Tuple[str, Any]] = []
            if isinstance(raw_pts, dict):
                for raw_k, raw_v in raw_pts.items():
                    kp_name = str(raw_k).strip()
                    if kp_name.isdigit():
                        nidx = int(kp_name)
                        kp_name = (
                            sk.names[nidx]
                            if 0 <= nidx < len(sk.names)
                            else f"node_{nidx}"
                        )
                    pt_pairs.append((kp_name, raw_v))
            elif isinstance(raw_pts, list):
                for j, raw_v in enumerate(raw_pts):
                    if j < len(sk.names):
                        pt_pairs.append((sk.names[j], raw_v))

            for kp_name, pt_raw in pt_pairs:
                pt_obj = resolve_ref(pt_raw)
                if not isinstance(pt_obj, dict):
                    continue
                rx, ry = pt_obj.get("x"), pt_obj.get("y")
                if rx is None or ry is None:
                    continue
                try:
                    xf, yf = float(rx), float(ry)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(xf) and math.isfinite(yf)):
                    continue
                is_vis = (
                    True
                    if pt_obj.get("visible") is None
                    else bool(pt_obj.get("visible"))
                )
                if not is_vis and not include_occluded:
                    continue
                sc: Optional[float] = None
                if pt_obj.get("score") is not None:
                    try:
                        scf = float(pt_obj["score"])
                        if math.isfinite(scf):
                            sc = scf
                    except (TypeError, ValueError):
                        pass
                if (
                    is_pred
                    and min_score_f is not None
                    and sc is not None
                    and sc < min_score_f
                ):
                    continue
                vis_code = (
                    int(KeypointVisibility.VISIBLE)
                    if is_vis
                    else int(KeypointVisibility.OCCLUDED)
                )
                kps[kp_name] = PosePoint(x=xf, y=yf, score=sc, visibility=vis_code)

            if not kps:
                continue

            if track_name:
                inst_name = track_name
            elif track_id is not None and track_id >= 0:
                inst_name = f"track_{track_id}"
            else:
                inst_name = f"instance_{local_idx}"
            if inst_name not in observed_instances:
                observed_instances.append(inst_name)

            parsed_insts.append(
                PoseInstance(
                    instance_name=inst_name,
                    keypoints=kps,
                    track_id=track_id,
                    skeleton_name=sk.name,
                    score=inst_score,
                    is_predicted=is_pred,
                )
            )

        if prefer_user_instances and any(
            not inst.is_predicted for inst in parsed_insts
        ):
            parsed_insts = [inst for inst in parsed_insts if not inst.is_predicted]
        if parsed_insts:
            grouped_frames.setdefault((vid, fidx), []).extend(parsed_insts)

    used_videos = sorted(used_vids_set or {0})
    if video_index is not None and int(video_index) not in used_vids_set:
        raise ValueError(
            f"video_index={video_index} is not present in {source_path.name}. "
            f"Available video indices: {used_videos}"
        )
    selected_videos = [int(video_index)] if video_index is not None else used_videos

    video_names: Dict[int, str] = {}
    video_paths: Dict[int, str] = {}
    for vid in selected_videos:
        fname = _video_filename_for(vid)
        if fname:
            video_paths[vid] = fname
            video_names[vid] = sanitize_filename_token(
                Path(fname.replace("\\", "/")).stem, fallback=f"video{vid}"
            )
        else:
            video_names[vid] = f"video{vid}"

    frames = [
        PoseFrame(
            frame_index=fidx,
            instances=grouped_frames[(vid, fidx)],
            video_index=vid,
            video_name=video_names.get(vid, f"video{vid}"),
        )
        for (vid, fidx) in sorted(grouped_frames.keys())
    ]

    all_kps = dedup_preserve_order(kp for sk in skeletons for kp in sk.names)
    all_edges = dedup_pairs(edge for sk in skeletons for edge in sk.edges)
    all_sym = dedup_pairs(
        pair for sk in skeletons for pair in (sk.symmetry_pairs or [])
    )

    return PoseDataset(
        frames=frames,
        keypoints=all_kps or list(skeletons[0].names),
        edges=all_edges,
        symmetry_pairs=all_sym,
        instances=dedup_preserve_order(track_names + observed_instances),
        scorer="sleap",
        multi_animal=len(track_names) > 1 or any(len(f.instances) > 1 for f in frames),
        source_format="sleap_zip",
        video_paths=video_paths,
        metadata={
            "source_path": str(source_path),
            "skeletons": [
                {
                    "name": sk.name,
                    "names": list(sk.names),
                    "edges": list(sk.edges),
                    "symmetry_pairs": list(sk.symmetry_pairs or []),
                }
                for sk in skeletons
            ],
            "used_videos": used_videos,
            "video_names": video_names,
        },
    )


def read_sleap_zip(
    zip_path: str | Path,
    *,
    video_path: Optional[str | Path] = None,
    video_index: Optional[int] = None,
    prefer_user_instances: bool = True,
    include_predicted: bool = True,
    include_occluded: bool = True,
    min_score: Optional[float] = None,
) -> PoseDataset:
    """Read a ZIP-based SLEAP ``.slp`` archive (wrapping either HDF5 or SLEAP JSON)."""
    path = Path(zip_path).expanduser().resolve()
    with zipfile.ZipFile(path, "r") as zf:
        h5_member = _find_zip_hdf5_member(zf)
        if h5_member is not None:
            raw_h5 = _read_zip_member_bounded(
                zf, h5_member, max_bytes=_MAX_ZIP_HDF5_BYTES
            )
            with tempfile.TemporaryDirectory(prefix="annolid_sleap_zip_") as tmpdir:
                tmp_h5 = Path(tmpdir) / Path(h5_member.filename).name
                tmp_h5.write_bytes(raw_h5)
                dataset = read_sleap(
                    tmp_h5,
                    video_path=video_path,
                    video_index=video_index,
                    prefer_user_instances=prefer_user_instances,
                    include_predicted=include_predicted,
                    include_occluded=include_occluded,
                    min_score=min_score,
                )
            dataset.metadata["source_path"] = str(path)
            dataset.metadata["zip_embedded_h5_member"] = h5_member.filename
            return dataset

        json_members = sorted(
            (
                info
                for info in zf.infolist()
                if not info.is_dir()
                and _is_safe_zip_member_name(info.filename)
                and info.filename.lower().endswith(".json")
            ),
            key=lambda info: (
                0 if Path(info.filename).name.lower() == "labels.json" else 1,
                info.filename,
            ),
        )
        for info in json_members:
            raw_json = _read_zip_member_bounded(zf, info, max_bytes=_MAX_ZIP_JSON_BYTES)
            parsed = safe_json_loads(raw_json)
            if isinstance(parsed, dict) and (
                "labeled_frames" in parsed
                or "frames" in parsed
                or "skeletons" in parsed
            ):
                dataset = _parse_sleap_json_labels(
                    parsed,
                    source_path=path,
                    video_index=video_index,
                    prefer_user_instances=prefer_user_instances,
                    include_predicted=include_predicted,
                    include_occluded=include_occluded,
                    min_score=min_score,
                )
                dataset.metadata["zip_json_member"] = info.filename
                return dataset

    raise ValueError(
        f"ZIP file '{path.name}' does not contain a recognized SLEAP HDF5 or JSON labels payload."
    )


def extract_sleap_zip_frames(
    zip_path: str | Path,
    dataset: PoseDataset,
    out_dir: Path,
    *,
    filename_prefix: str = "",
) -> Tuple[Dict[Tuple[int, int], str], Dict[Tuple[int, int], Tuple[int, int]]]:
    """Extract embedded frames from a ZIP-based SLEAP ``.slp`` archive."""
    Image = _require_pil_image()
    path = Path(zip_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_to_image: Dict[Tuple[int, int], str] = {}
    frame_to_wh: Dict[Tuple[int, int], Tuple[int, int]] = {}

    used_vids = sorted({int(f.video_index) for f in dataset.frames})
    multi_video = len(used_vids) > 1
    vid_names = dataset.metadata.get("video_names") or {}

    with zipfile.ZipFile(path, "r") as zf:
        h5_member_name = dataset.metadata.get("zip_embedded_h5_member")
        if isinstance(h5_member_name, str) and h5_member_name:
            info = zf.getinfo(h5_member_name)
            raw_h5 = _read_zip_member_bounded(zf, info, max_bytes=_MAX_ZIP_HDF5_BYTES)
            with tempfile.TemporaryDirectory(prefix="annolid_sleap_zip_") as tmpdir:
                tmp_h5 = Path(tmpdir) / Path(h5_member_name).name
                tmp_h5.write_bytes(raw_h5)
                with SleapH5Reader(tmp_h5) as reader:
                    for vid in used_vids:
                        if not reader.has_embedded_frames(vid):
                            continue
                        prefix = (
                            str(vid_names.get(vid, f"video{vid}"))
                            if multi_video
                            else filename_prefix
                        )
                        mapping = reader.save_embedded_frames(
                            out_dir, video_index=vid, name_prefix=prefix
                        )
                        for fidx, rel in mapping.items():
                            frame_to_image[(vid, fidx)] = rel
                            with Image.open(out_dir / rel) as im:
                                frame_to_wh[(vid, fidx)] = (im.width, im.height)
            if frame_to_image:
                return frame_to_image, frame_to_wh

        # Extract image members stored directly inside the ZIP archive.
        for info in sorted(zf.infolist(), key=lambda x: x.filename):
            if info.is_dir() or not _is_safe_zip_member_name(info.filename):
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in _ZIP_IMAGE_EXTS:
                continue
            m = _ZIP_FRAME_NUM_RE.search(Path(info.filename).name)
            if not m:
                continue
            fidx = int(m.group(1))
            vid = 0
            parts = info.filename.replace("\\", "/").split("/")
            for seg in parts[:-1]:
                if seg.lower().startswith("video") and seg[5:].isdigit():
                    vid = int(seg[5:])
                    break
            raw_img = _read_zip_member_bounded(zf, info, max_bytes=_MAX_ZIP_IMAGE_BYTES)
            with Image.open(io.BytesIO(raw_img)) as im_raw:
                im = im_raw.convert("RGB")
                prefix = (
                    f"{vid_names.get(vid, f'video{vid}')}_"
                    if multi_video
                    else (f"{filename_prefix}_" if filename_prefix else "")
                )
                fname = f"{prefix}{fidx:09d}.png"
                im.save(out_dir / fname)
                frame_to_image[(vid, fidx)] = fname
                frame_to_wh[(vid, fidx)] = (im.width, im.height)

    return frame_to_image, frame_to_wh


def read_sleap(
    sleap_path: str | Path,
    *,
    video_path: Optional[str | Path] = None,
    video_index: Optional[int] = None,
    prefer_user_instances: bool = True,
    include_predicted: bool = True,
    include_occluded: bool = True,
    min_score: Optional[float] = None,
) -> PoseDataset:
    """Auto-detect and read SLEAP ``.slp``/``.pkg.slp`` (HDF5 or ZIP) or ``.analysis.h5``."""
    kind = detect_sleap_h5_kind(sleap_path)
    if kind == "analysis_h5":
        return read_sleap_analysis_h5(
            sleap_path, video_path=video_path, min_score=min_score
        )
    if kind == "zip_slp":
        return read_sleap_zip(
            sleap_path,
            video_path=video_path,
            video_index=video_index,
            prefer_user_instances=prefer_user_instances,
            include_predicted=include_predicted,
            include_occluded=include_occluded,
            min_score=min_score,
        )
    return read_sleap_slp(
        sleap_path,
        video_index=video_index,
        prefer_user_instances=prefer_user_instances,
        include_predicted=include_predicted,
        include_occluded=include_occluded,
        min_score=min_score,
    )


def _atomic_h5_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    return Path(raw)


def write_sleap_analysis_h5(
    dataset: PoseDataset,
    output_h5: str | Path,
    *,
    video_path: Optional[str | Path] = None,
    default_score: float = 1.0,
) -> Path:
    """Export a ``PoseDataset`` to SLEAP Analysis HDF5 format (``.analysis.h5``).

    Writes standard SLEAP analysis datasets:
    - ``/tracks``: ``(n_tracks, 2, n_nodes, n_frames)`` float64 with NaN for missing points
    - ``/point_scores``: ``(n_tracks, n_nodes, n_frames)`` float64
    - ``/instance_scores``: ``(n_tracks, n_frames)`` float64
    - ``/track_occupancy``: ``(n_frames, n_tracks)`` int8
    - ``/node_names``, ``/track_names``, ``/edge_names``, ``/edge_inds``
    - ``/video_path``, ``/provenance``
    """
    h5py = require_h5py()
    out_path = Path(output_h5).expanduser().resolve()
    keypoints = dataset.resolved_keypoints()
    if not keypoints:
        raise ValueError(
            "Cannot export SLEAP analysis H5: dataset contains no keypoints."
        )

    instances = dataset.resolved_instances() or ["track_0"]
    schema = dataset.to_pose_schema(preset="auto")

    frame_indices = [int(f.frame_index) for f in dataset.frames]
    n_frames = (max(frame_indices) + 1) if frame_indices else 0
    n_tracks = len(instances)
    n_nodes = len(keypoints)

    kp_idx = {name: i for i, name in enumerate(keypoints)}
    inst_idx = {name: i for i, name in enumerate(instances)}

    tracks = np.full((n_tracks, 2, n_nodes, n_frames), np.nan, dtype=np.float64)
    point_scores = np.full((n_tracks, n_nodes, n_frames), np.nan, dtype=np.float64)
    instance_scores = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
    track_occupancy = np.zeros((n_frames, n_tracks), dtype=np.int8)

    for frame in dataset.frames:
        fidx = int(frame.frame_index)
        if fidx < 0 or fidx >= n_frames:
            continue
        for inst in frame.instances:
            tname = str(inst.instance_name or "").strip() or instances[0]
            tidx = inst_idx.get(tname)
            if tidx is None:
                continue
            any_valid = False
            inst_pt_scores: List[float] = []
            for kp_name, pt in inst.keypoints.items():
                nidx = kp_idx.get(kp_name)
                if nidx is None or not pt.is_valid:
                    continue
                tracks[tidx, 0, nidx, fidx] = float(pt.x)
                tracks[tidx, 1, nidx, fidx] = float(pt.y)
                sc = (
                    float(pt.score)
                    if pt.score is not None and math.isfinite(float(pt.score))
                    else float(default_score)
                )
                point_scores[tidx, nidx, fidx] = sc
                inst_pt_scores.append(sc)
                any_valid = True
            if any_valid:
                track_occupancy[fidx, tidx] = 1
                if inst.score is not None and math.isfinite(float(inst.score)):
                    instance_scores[tidx, fidx] = float(inst.score)
                elif inst_pt_scores:
                    instance_scores[tidx, fidx] = float(np.mean(inst_pt_scores))

    edge_pairs: List[Tuple[str, str]] = [
        (a, b) for a, b in schema.edges if a in kp_idx and b in kp_idx
    ]
    edge_inds = (
        np.array([[kp_idx[a], kp_idx[b]] for a, b in edge_pairs], dtype=np.int64)
        if edge_pairs
        else np.zeros((0, 2), dtype=np.int64)
    )

    resolved_video = str(
        video_path
        or dataset.video_paths.get(0)
        or (dataset.frames[0].video_name if dataset.frames else "")
    )
    provenance = json.dumps(
        {
            "exporter": "annolid",
            "format": "sleap_analysis_h5",
            "scorer": dataset.scorer,
            "n_frames": n_frames,
            "n_tracks": n_tracks,
            "n_nodes": n_nodes,
        }
    )

    str_dtype = h5py.string_dtype(encoding="utf-8")
    tmp_path = _atomic_h5_path(out_path)
    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset("tracks", data=tracks, compression="gzip")
            h5.create_dataset("point_scores", data=point_scores, compression="gzip")
            h5.create_dataset(
                "instance_scores", data=instance_scores, compression="gzip"
            )
            h5.create_dataset("track_occupancy", data=track_occupancy)
            h5.create_dataset(
                "node_names", data=np.array(keypoints, dtype=object), dtype=str_dtype
            )
            h5.create_dataset(
                "track_names", data=np.array(instances, dtype=object), dtype=str_dtype
            )
            if edge_pairs:
                h5.create_dataset(
                    "edge_names",
                    data=np.array([[a, b] for a, b in edge_pairs], dtype=object),
                    dtype=str_dtype,
                )
            else:
                h5.create_dataset("edge_names", shape=(0, 2), dtype=str_dtype)
            h5.create_dataset("edge_inds", data=edge_inds)
            h5.create_dataset("video_path", data=resolved_video, dtype=str_dtype)
            h5.create_dataset("provenance", data=provenance, dtype=str_dtype)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return out_path


def write_sleap_slp(
    dataset: PoseDataset,
    output_slp: str | Path,
    *,
    video_path: Optional[str | Path] = None,
    embed_images: bool = False,
    skeleton_name: str = "annolid_skeleton",
) -> Path:
    """Export a ``PoseDataset`` to a SLEAP HDF5 ``.slp`` (or ``.pkg.slp`` when ``embed_images=True``)."""
    h5py = require_h5py()
    out_path = Path(output_slp).expanduser().resolve()
    keypoints = dataset.resolved_keypoints()
    if not keypoints:
        raise ValueError("Cannot export SLEAP .slp: dataset contains no keypoints.")

    schema = dataset.to_pose_schema(preset="auto")
    kp_idx = {name: i for i, name in enumerate(keypoints)}
    instances = dataset.resolved_instances()
    track_to_id = {name: i for i, name in enumerate(instances)}

    links: List[Dict[str, Any]] = []
    for edge_i, (a, b) in enumerate(schema.edges):
        if a in kp_idx and b in kp_idx:
            links.append(
                {
                    "edge_insert_idx": edge_i,
                    "key": 0,
                    "source": {"id": kp_idx[a]},
                    "target": {"id": kp_idx[b]},
                    "type": 1,
                }
            )
    for sym_i, (a, b) in enumerate(schema.symmetry_pairs):
        if a in kp_idx and b in kp_idx:
            links.append(
                {
                    "edge_insert_idx": sym_i,
                    "key": 0,
                    "source": {"id": kp_idx[a]},
                    "target": {"id": kp_idx[b]},
                    "type": 2,
                }
            )

    metadata = {
        "version": "2.0.0",
        "nodes": [{"name": name, "weight": 1.0} for name in keypoints],
        "skeletons": [
            {
                "directed": True,
                "multigraph": True,
                "graph": {"name": str(skeleton_name or "annolid_skeleton")},
                "nodes": [{"id": i} for i in range(len(keypoints))],
                "links": links,
            }
        ],
        "provenance": {"exporter": "annolid"},
    }

    sorted_frames = sorted(dataset.frames, key=lambda f: (f.video_index, f.frame_index))
    frames_dt = np.dtype(
        [
            ("frame_id", "<u8"),
            ("video", "<u4"),
            ("frame_idx", "<u8"),
            ("instance_id_start", "<u8"),
            ("instance_id_end", "<u8"),
        ]
    )
    inst_dt = np.dtype(
        [
            ("instance_id", "<i8"),
            ("instance_type", "u1"),
            ("frame_id", "<u8"),
            ("skeleton", "<u4"),
            ("track", "<i4"),
            ("from_predicted", "<i8"),
            ("score", "<f8"),
            ("point_id_start", "<u8"),
            ("point_id_end", "<u8"),
        ]
    )
    pts_dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("visible", "?"), ("complete", "?")])
    pred_pts_dt = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("visible", "?"),
            ("complete", "?"),
            ("score", "<f8"),
        ]
    )

    frame_rows: List[Tuple[int, int, int, int, int]] = []
    inst_rows: List[Tuple[int, int, int, int, int, int, float, int, int]] = []
    user_point_rows: List[Tuple[float, float, bool, bool]] = []
    pred_point_rows: List[Tuple[float, float, bool, bool, float]] = []

    embedded_by_video: Dict[int, List[Tuple[int, bytes]]] = {}
    video_dims: Dict[int, Tuple[int, int]] = {}

    for frame_id, frame in enumerate(sorted_frames):
        vid = int(frame.video_index)
        if frame.image_width and frame.image_height:
            video_dims.setdefault(
                vid, (int(frame.image_width), int(frame.image_height))
            )

        if embed_images and frame.source_image_path:
            img_p = Path(frame.source_image_path)
            if img_p.is_file():
                embedded_by_video.setdefault(vid, []).append(
                    (int(frame.frame_index), img_p.read_bytes())
                )

        inst_start = len(inst_rows)
        for inst in frame.instances:
            inst_id = len(inst_rows)
            tid = (
                track_to_id.get(str(inst.instance_name).strip(), -1)
                if inst.instance_name
                else (int(inst.track_id) if inst.track_id is not None else -1)
            )
            is_pred = bool(inst.is_predicted)
            inst_type = INSTANCE_TYPE_PREDICTED if is_pred else INSTANCE_TYPE_USER
            inst_score = (
                float(inst.score)
                if inst.score is not None and math.isfinite(float(inst.score))
                else (1.0 if is_pred else np.nan)
            )

            if is_pred:
                pt_start = len(pred_point_rows)
                for kp_name in keypoints:
                    pt = inst.keypoints.get(kp_name)
                    if pt is not None and pt.is_valid:
                        vis = int(pt.visibility) != int(KeypointVisibility.OCCLUDED)
                        sc = (
                            float(pt.score)
                            if pt.score is not None and math.isfinite(float(pt.score))
                            else 1.0
                        )
                        pred_point_rows.append(
                            (float(pt.x), float(pt.y), vis, False, sc)
                        )
                    else:
                        pred_point_rows.append((np.nan, np.nan, False, False, np.nan))
                pt_end = len(pred_point_rows)
            else:
                pt_start = len(user_point_rows)
                for kp_name in keypoints:
                    pt = inst.keypoints.get(kp_name)
                    if pt is not None and pt.is_valid:
                        vis = int(pt.visibility) != int(KeypointVisibility.OCCLUDED)
                        user_point_rows.append((float(pt.x), float(pt.y), vis, True))
                    else:
                        user_point_rows.append((np.nan, np.nan, False, False))
                pt_end = len(user_point_rows)

            inst_rows.append(
                (inst_id, inst_type, frame_id, 0, tid, -1, inst_score, pt_start, pt_end)
            )
        inst_end = len(inst_rows)
        frame_rows.append((frame_id, vid, int(frame.frame_index), inst_start, inst_end))

    used_vids = sorted({r[1] for r in frame_rows} or {0})
    videos_json_list: List[str] = []
    for vid in used_vids:
        vpath = str(
            video_path
            if (video_path and vid == 0)
            else (dataset.video_paths.get(vid) or ".")
        )
        dims = video_dims.get(vid)
        max_fidx = max((r[2] for r in frame_rows if r[1] == vid), default=0)
        backend: Dict[str, Any] = {"filename": vpath}
        if embed_images and vid in embedded_by_video:
            backend["dataset"] = f"video{vid}/video"
            backend["filename"] = "." if vpath == "." else Path(vpath).name
        if dims:
            backend["shape"] = [max_fidx + 1, int(dims[1]), int(dims[0]), 3]
        videos_json_list.append(json.dumps({"backend": backend}))

    tracks_json_list = [json.dumps([0, name]) for name in instances]
    str_dtype = h5py.string_dtype(encoding="utf-8")

    tmp_path = _atomic_h5_path(out_path)
    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset("metadata", data=json.dumps(metadata).encode("utf-8"))
            h5.create_dataset("frames", data=np.array(frame_rows, dtype=frames_dt))
            h5.create_dataset("instances", data=np.array(inst_rows, dtype=inst_dt))
            h5.create_dataset("points", data=np.array(user_point_rows, dtype=pts_dt))
            if pred_point_rows:
                h5.create_dataset(
                    "pred_points", data=np.array(pred_point_rows, dtype=pred_pts_dt)
                )
            h5.create_dataset(
                "videos_json",
                data=np.array(videos_json_list, dtype=object),
                dtype=str_dtype,
            )
            h5.create_dataset(
                "tracks_json",
                data=np.array(tracks_json_list, dtype=object),
                dtype=str_dtype,
            )

            if embed_images:
                vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))
                for vid, img_items in embedded_by_video.items():
                    grp = h5.create_group(f"video{vid}")
                    grp.create_dataset(
                        "frame_numbers",
                        data=np.array([f for f, _ in img_items], dtype=np.int64),
                    )
                    vds = grp.create_dataset("video", (len(img_items),), dtype=vlen_u8)
                    for idx, (_fidx, raw_bytes) in enumerate(img_items):
                        vds[idx] = np.frombuffer(raw_bytes, dtype=np.uint8)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return out_path
