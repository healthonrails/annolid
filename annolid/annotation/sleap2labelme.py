#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import h5py
except ImportError as e:
    raise SystemExit("Missing dependency: h5py. Install with: pip install h5py") from e

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit(
        "Missing dependency: pillow. Install with: pip install pillow"
    ) from e

try:
    # ✅ Use Annolid's PoseSchema implementation
    from annolid.annotation.pose_schema import PoseSchema  # type: ignore
except Exception as e:
    raise SystemExit(
        "Missing dependency: annolid (for PoseSchema). "
        "Install with: pip install annolid"
    ) from e


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ZIP_MAGIC = b"PK\x03\x04"


def _read_magic(p: Path, n: int = 8) -> bytes:
    with p.open("rb") as f:
        return f.read(n)


def detect_sleap_container(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    magic = _read_magic(p, 8)
    if magic.startswith(HDF5_MAGIC):
        return "h5"
    if magic.startswith(ZIP_MAGIC):
        return "zip"
    raise ValueError(f"Unrecognized SLEAP file container for {p} (magic={magic!r}).")


def safe_json_loads(raw: object) -> Optional[object]:
    """Try to parse a JSON string/bytes stored in HDF5 datasets or attrs."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray, np.void)):
        try:
            raw = bytes(raw).decode("utf-8", errors="ignore")
        except Exception:
            return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


@dataclass(frozen=True)
class SkeletonSpec:
    names: List[str]
    edges: List[Tuple[str, str]]


@dataclass(frozen=True)
class InstanceRecord:
    frame_idx: int
    track_id: int
    points_xy: np.ndarray
    visible: Optional[np.ndarray] = None
    node_ids: Optional[np.ndarray] = None


class SleapH5Reader:
    """
    Reader for common SLEAP HDF5 .slp/.pkg.slp structure.
    """

    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        self._h5 = h5py.File(h5_path, "r")

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def root_keys(self) -> List[str]:
        return list(self._h5.keys())

    def _require_ds(self, key: str) -> h5py.Dataset:
        if key not in self._h5:
            raise KeyError(
                f"Missing dataset/group: {key}. Root keys: {self.root_keys()}"
            )
        obj = self._h5[key]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"{key} is not a dataset (found {type(obj)}).")
        return obj

    def _maybe_ds(self, key: str) -> Optional[h5py.Dataset]:
        obj = self._h5.get(key, None)
        return obj if isinstance(obj, h5py.Dataset) else None

    def _maybe_grp(self, key: str) -> Optional[h5py.Group]:
        obj = self._h5.get(key, None)
        return obj if isinstance(obj, h5py.Group) else None

    # --------------------------
    # Skeleton extraction helpers
    # --------------------------
    @staticmethod
    def _extract_skeleton_from_obj(obj: object) -> Optional[SkeletonSpec]:
        """
        Try to find skeleton nodes+edges from any decoded JSON-like object.
        Handles patterns such as:
          {"skeletons":[{"nodes":[{"name":"head"}, ...], "edges":[[0,1], ...]}]}
          {"skeletons":[{"node_names":[...], "edges":[["head","thorax"], ...]}]}
          {"skeleton":{"nodes":[...], "edges":[...]}}, etc.
        """
        if not isinstance(obj, (dict, list)):
            return None

        def walk(x: object) -> Iterable[dict]:
            if isinstance(x, dict):
                yield x
                for v in x.values():
                    yield from walk(v)
            elif isinstance(x, list):
                for v in x:
                    yield from walk(v)

        def normalize_names(d: dict) -> Optional[List[str]]:
            # node_names: [str, ...]
            if (
                "node_names" in d
                and isinstance(d["node_names"], list)
                and d["node_names"]
            ):
                if all(isinstance(n, str) for n in d["node_names"]):
                    return [str(n).strip() for n in d["node_names"] if str(n).strip()]
            # nodes: [{name:...}]
            if "nodes" in d and isinstance(d["nodes"], list) and d["nodes"]:
                if all(isinstance(n, dict) and "name" in n for n in d["nodes"]):
                    names = [str(n.get("name", "")).strip() for n in d["nodes"]]
                    names = [n for n in names if n]
                    return names or None
            return None

        def normalize_edges(d: dict, names: List[str]) -> List[Tuple[str, str]]:
            out: List[Tuple[str, str]] = []
            edges = d.get("edges")
            if not edges or not isinstance(edges, list):
                return out

            def add_pair(a: Any, b: Any) -> None:
                a_s = str(a).strip() if a is not None else ""
                b_s = str(b).strip() if b is not None else ""
                if not a_s or not b_s or a_s == b_s:
                    return
                out.append((a_s, b_s))

            for e in edges:
                # edges could be [[0,1], ...] or [["head","thorax"], ...]
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    a, b = e
                    # indices
                    if isinstance(a, (int, np.integer)) and isinstance(
                        b, (int, np.integer)
                    ):
                        ai = int(a)
                        bi = int(b)
                        if 0 <= ai < len(names) and 0 <= bi < len(names):
                            add_pair(names[ai], names[bi])
                    else:
                        add_pair(a, b)
                # sometimes edges as dicts: {"source":0,"target":1} or {"src":..., "dst":...}
                elif isinstance(e, dict):
                    a = e.get("source", e.get("src", e.get("a")))
                    b = e.get("target", e.get("dst", e.get("b")))
                    if isinstance(a, (int, np.integer)) and isinstance(
                        b, (int, np.integer)
                    ):
                        ai = int(a)
                        bi = int(b)
                        if 0 <= ai < len(names) and 0 <= bi < len(names):
                            add_pair(names[ai], names[bi])
                    else:
                        add_pair(a, b)

            # de-dup while preserving order
            seen = set()
            dedup: List[Tuple[str, str]] = []
            for a, b in out:
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                dedup.append((a, b))
            return dedup

        for d in walk(obj):
            names = normalize_names(d)
            if not names:
                continue
            edges = normalize_edges(d, names)
            return SkeletonSpec(names=names, edges=edges)

        return None

    def _scan_attrs_for_skeleton(
        self, h5obj: Union[h5py.File, h5py.Group, h5py.Dataset]
    ) -> Optional[SkeletonSpec]:
        try:
            for _, v in h5obj.attrs.items():
                obj = safe_json_loads(v)
                if obj is None:
                    continue
                sk = self._extract_skeleton_from_obj(obj)
                if sk:
                    return sk
        except Exception:
            return None
        return None

    def infer_skeleton(self) -> SkeletonSpec:
        # 1) /metadata dataset
        meta_ds = self._maybe_ds("metadata")
        if meta_ds is not None:
            try:
                obj = safe_json_loads(meta_ds[()])
                if obj is not None:
                    sk = self._extract_skeleton_from_obj(obj)
                    if sk:
                        return sk
            except Exception:
                pass

        # 1b) /metadata group attrs
        meta_grp = self._maybe_grp("metadata")
        if meta_grp is not None:
            sk = self._scan_attrs_for_skeleton(meta_grp)
            if sk:
                return sk

        # 2) any *_json dataset
        for k in self.root_keys():
            if not k.endswith("_json"):
                continue
            ds = self._maybe_ds(k)
            if ds is None:
                continue
            try:
                obj = safe_json_loads(ds[()])
            except Exception:
                continue
            if obj is None:
                continue
            sk = self._extract_skeleton_from_obj(obj)
            if sk:
                return sk

        # 3) scan attrs broadly
        sk = self._scan_attrs_for_skeleton(self._h5)
        if sk:
            return sk

        found: Optional[SkeletonSpec] = None

        def visitor(_: str, obj: object) -> None:
            nonlocal found
            if found:
                return
            if isinstance(obj, (h5py.Group, h5py.Dataset)):
                got = self._scan_attrs_for_skeleton(obj)
                if got:
                    found = got

        try:
            self._h5.visititems(visitor)
        except Exception:
            pass
        if found:
            return found

        # 4) infer node count from /points node_id
        pts = self._maybe_ds("points")
        if pts is not None and pts.dtype.fields and "node_id" in pts.dtype.fields:
            try:
                node_ids = np.asarray(pts["node_id"][:], dtype=np.int64)
                node_ids = node_ids[np.isfinite(node_ids)]
                if node_ids.size:
                    n_nodes = int(np.max(node_ids)) + 1
                    if n_nodes > 0:
                        return SkeletonSpec(
                            names=[f"node_{i}" for i in range(n_nodes)],
                            edges=[],
                        )
            except Exception:
                pass

        # 5) fallback: first instance slice length
        inst = self._require_ds("instances")
        if inst.shape[0] == 0:
            raise ValueError("No instances found in file; cannot infer node count.")
        row0 = inst[0]
        ps = int(row0["point_id_start"])
        pe = int(row0["point_id_end"])
        n_nodes = max(0, pe - ps)
        if n_nodes <= 0:
            raise ValueError(
                "Could not infer node count from instances[0].point_id_start/end."
            )
        return SkeletonSpec(
            names=[f"node_{i}" for i in range(n_nodes)],
            edges=[],
        )

    # --------------------------
    # Instances / points reading
    # --------------------------
    def iter_instances(self, skeleton: SkeletonSpec) -> Iterable[InstanceRecord]:
        frames = self._require_ds("frames")
        inst = self._require_ds("instances")
        pts = self._require_ds("points")

        pt_fields = set(pts.dtype.fields.keys()) if pts.dtype.fields else set()
        if not {"x", "y"}.issubset(pt_fields):
            raise KeyError(
                f"/points missing x/y columns. Found fields: {sorted(pt_fields)}"
            )

        has_visible = "visible" in pt_fields
        has_node_id = "node_id" in pt_fields

        for fr in frames:
            frame_idx = int(fr["frame_idx"])
            i0 = int(fr["instance_id_start"])
            i1 = int(fr["instance_id_end"])
            if i1 <= i0:
                continue

            for inst_row in inst[i0:i1]:
                track_id = int(inst_row["track"])
                ps = int(inst_row["point_id_start"])
                pe = int(inst_row["point_id_end"])
                n = pe - ps
                if n <= 0:
                    continue

                xs = pts["x"][ps:pe].astype(np.float32, copy=False)
                ys = pts["y"][ps:pe].astype(np.float32, copy=False)
                xy = np.stack([xs, ys], axis=1)

                vis = None
                if has_visible:
                    vis = pts["visible"][ps:pe].astype(bool, copy=False)

                node_ids = None
                if has_node_id:
                    node_ids = pts["node_id"][ps:pe].astype(np.int64, copy=False)

                # ✅ Prevent silent mislabeling:
                # Without node_id, we can only safely map by index if the slice is dense.
                if node_ids is None and n != len(skeleton.names):
                    raise ValueError(
                        "Ambiguous point-to-node mapping: this SLEAP file stores a compact/variable "
                        f"point slice (n={n}) but provides no /points['node_id'] to identify nodes. "
                        "Re-export labels with node ids, or ensure each instance stores a full dense "
                        f"node vector of length {len(skeleton.names)}."
                    )

                yield InstanceRecord(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    points_xy=xy,
                    visible=vis,
                    node_ids=node_ids,
                )

    # --------------------------
    # Embedded frames extraction
    # --------------------------
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
    def _decode_embedded_frame(obj: object) -> Image.Image:
        def try_decode_bytes(b: bytes) -> Optional[Image.Image]:
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

            if (
                arr.ndim == 1
                and arr.dtype.kind in ("i", "u")
                and arr.itemsize in (2, 4, 8)
            ):
                b = (
                    (arr.astype(np.uint64, copy=False) & 0xFF)
                    .astype(np.uint8)
                    .tobytes()
                )
                im = try_decode_bytes(b)
                if im is not None:
                    return im

            raise ValueError(
                f"Unsupported embedded frame ndarray shape={arr.shape}, dtype={arr.dtype}"
            )

        if isinstance(obj, memoryview):
            im = try_decode_bytes(obj.tobytes())
            if im is not None:
                return im

        raise ValueError(f"Unsupported embedded frame type: {type(obj)}")


def make_labelme_json(
    image_relpath: str, image_size: Tuple[int, int], shapes: List[dict]
) -> dict:
    w, h = image_size
    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_relpath,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }


def _infer_symmetry_pairs_extended(keypoints: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Extended L/R inference:
    - uses Annolid PoseSchema.infer_symmetry_pairs when available
    - plus suffix patterns like wingL/wingR, eyeL/eyeR, forelegL4/forelegR4, etc.
    """
    pairs: List[Tuple[str, str]] = []
    try:
        pairs.extend(PoseSchema.infer_symmetry_pairs(keypoints))  # type: ignore[attr-defined]
    except Exception:
        pass

    names = [str(k).strip() for k in keypoints if str(k).strip()]
    lower_to_original = {n.lower(): n for n in names}
    used = {tuple(sorted((a.lower(), b.lower()))) for a, b in pairs}

    def add(a: str, b: str) -> None:
        key = tuple(sorted((a.lower(), b.lower())))
        if key in used:
            return
        used.add(key)
        pairs.append((a, b))

    for name in names:
        low = name.lower()

        # Suffix L/R: "...l" <-> "...r" (but avoid matching words like "all")
        if len(low) >= 2 and low[-1] in ("l", "r"):
            other = low[:-1] + ("r" if low[-1] == "l" else "l")
            if other in lower_to_original:
                add(name, lower_to_original[other])

        # Embedded token L/R: e.g. forelegL4 <-> forelegR4, eyeL <-> eyeR
        # Heuristic: replace 'l' with 'r' when adjacent to digit or end.
        for repl in (
            (("l",), ("r",)),
            (("left",), ("right",)),
            (("right",), ("left",)),
        ):
            # keep for future; suffix rule above covers the common SLEAP fly schema well.
            pass

        # Mid-string 'L'/'R' right before digits or end (case-insensitive)
        # Example: "forelegl4" -> "forelegr4"
        for idx in range(1, len(low) - 1):
            if low[idx] not in ("l", "r"):
                continue
            if not (low[idx + 1].isdigit() or idx == len(low) - 1):
                continue
            other = low[:idx] + ("r" if low[idx] == "l" else "l") + low[idx + 1 :]
            if other in lower_to_original:
                add(name, lower_to_original[other])

    return pairs


def build_label_mapper(
    skeleton_names: List[str], schema: Optional[PoseSchema]
) -> Dict[str, str]:
    """Map SLEAP skeleton node names -> canonical pose schema keypoint names (case-insensitive)."""
    mapper: Dict[str, str] = {}
    if schema is None or not getattr(schema, "keypoints", None):
        for n in skeleton_names:
            mapper[n] = n
        return mapper

    # Normalize schema if it has the helper
    try:
        schema.normalize_prefixed_keypoints()  # type: ignore[attr-defined]
    except Exception:
        pass

    canonical = {str(kp).strip().lower(): str(kp).strip() for kp in schema.keypoints}
    for n in skeleton_names:
        key = str(n).strip().lower()
        mapper[n] = canonical.get(key, n)
    return mapper


def write_default_pose_schema(out_dir: Path, skeleton: SkeletonSpec) -> Path:
    """
    ✅ By default, write a pose schema JSON to output folder.
    - keypoints: skeleton.names
    - edges: skeleton.edges (if available)
    - symmetry_pairs: inferred (extended)
    """
    schema = PoseSchema(
        version="1.0",
        keypoints=list(skeleton.names),
        edges=list(skeleton.edges) if skeleton.edges else [],
        symmetry_pairs=_infer_symmetry_pairs_extended(skeleton.names),
        flip_idx=None,
        instances=[],
        instance_separator="_",
    )
    out_path = out_dir / "pose_schema.json"
    schema.save(out_path)
    return out_path


def convert_sleap_h5_to_labelme(
    sleap_path: str | Path,
    out_dir: str | Path,
    *,
    save_frames: bool = True,
    video_index: int = 0,
    print_every: int = 200,
    pose_schema_path: Optional[str | Path] = None,
) -> Path:
    sleap_path = str(Path(sleap_path).expanduser())
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kind = detect_sleap_container(sleap_path)
    if kind != "h5":
        raise SystemExit(
            f"Currently this converter expects HDF5 .slp/.pkg.slp. "
            f"Your file looks like: {kind}. If you truly have a zip-based pkg, "
            f"export an HDF5 .slp or share the archive structure."
        )

    reader = SleapH5Reader(Path(sleap_path))
    try:
        skeleton = reader.infer_skeleton()

        # Load user-provided schema if available; otherwise create default
        pose_schema: Optional[PoseSchema]
        if pose_schema_path:
            pose_schema = PoseSchema.load(pose_schema_path)
        else:
            pose_schema = None

        # ✅ Always write a pose schema JSON to out_dir (default or normalized copy)
        if pose_schema is None:
            schema_path = write_default_pose_schema(out_dir, skeleton)
        else:
            # Save a normalized copy into out_dir for convenience
            try:
                # type: ignore[attr-defined]
                pose_schema.normalize_prefixed_keypoints()
            except Exception:
                pass
            schema_path = out_dir / "pose_schema.json"
            pose_schema.save(schema_path)

        label_map = build_label_mapper(skeleton.names, pose_schema)

        frame_to_image: Dict[int, str] = {}
        frame_to_wh: Dict[int, Tuple[int, int]] = {}

        if save_frames:
            frame_to_image = reader.save_embedded_frames(
                out_dir, video_index=video_index
            )
            for fidx, rel in frame_to_image.items():
                p = out_dir / rel
                with Image.open(p) as im:
                    frame_to_wh[fidx] = (im.width, im.height)

        per_frame_shapes: Dict[int, List[dict]] = {}

        for k, rec in enumerate(reader.iter_instances(skeleton)):
            per_frame_shapes.setdefault(rec.frame_idx, [])

            for j, (x, y) in enumerate(rec.points_xy):
                # Skip non-visible points if visibility exists
                if rec.visible is not None and not bool(rec.visible[j]):
                    continue
                # Skip NaNs
                if np.isnan(x) or np.isnan(y):
                    continue

                # ✅ Correct node mapping:
                if rec.node_ids is not None:
                    nid = int(rec.node_ids[j])
                else:
                    # Safe only because iter_instances ensures dense slice (n == n_nodes).
                    nid = j

                if nid < 0 or nid >= len(skeleton.names):
                    continue

                skel_name = skeleton.names[nid]
                label = label_map.get(skel_name, skel_name)

                gid = int(rec.track_id)
                group_id = None if gid < 0 else gid

                shape = {
                    "label": label,
                    "points": [[float(x), float(y)]],
                    "group_id": group_id,
                    "shape_type": "point",
                    "flags": {},
                }
                per_frame_shapes[rec.frame_idx].append(shape)

            if print_every and (k + 1) % print_every == 0:
                print(f"[sleap2labelme] processed {k + 1} instances...")

        for frame_idx, shapes in per_frame_shapes.items():
            image_rel = frame_to_image.get(frame_idx, "")
            wh = frame_to_wh.get(frame_idx, (0, 0))
            data = make_labelme_json(image_rel, wh, shapes)
            out_json = out_dir / f"{frame_idx:09d}.json"
            out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

        print(
            f"[sleap2labelme] wrote {len(per_frame_shapes)} LabelMe JSONs to: {out_dir}"
        )
        if save_frames:
            print(
                f"[sleap2labelme] saved {len(frame_to_image)} embedded frames to: {out_dir}"
            )
        print(f"[sleap2labelme] wrote pose schema to: {schema_path}")
        return out_dir

    finally:
        reader.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleap", required=True, help="Path to .slp or .pkg.slp (HDF5)")
    ap.add_argument(
        "--out", required=True, help="Output directory for LabelMe JSON + images/"
    )
    ap.add_argument(
        "--no-save-frames", action="store_true", help="Do not extract embedded frames."
    )
    ap.add_argument(
        "--video-index",
        type=int,
        default=0,
        help="video index to read (video0, video1, ...)",
    )
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument(
        "--pose-schema",
        default=None,
        help="Optional: pose schema JSON/YAML to map skeleton names. "
        "Regardless, pose_schema.json will be written to the output folder.",
    )
    args = ap.parse_args()

    convert_sleap_h5_to_labelme(
        args.sleap,
        args.out,
        save_frames=not args.no_save_frames,
        video_index=args.video_index,
        print_every=args.print_every,
        pose_schema_path=args.pose_schema,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
