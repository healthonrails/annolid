#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
        "Missing dependency: annolid (for PoseSchema). Install with: pip install annolid"
    ) from e


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ZIP_MAGIC = b"PK\x03\x04"

# Matches sleap-io InstanceType
INSTANCE_TYPE_USER = 0
INSTANCE_TYPE_PREDICTED = 1


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
    symmetry_pairs: List[Tuple[str, str]] = None  # type: ignore[assignment]
    name: Optional[str] = None


@dataclass(frozen=True)
class InstanceRecord:
    frame_idx: int
    track_id: int
    skeleton_id: int
    points_xy: np.ndarray
    visible: Optional[np.ndarray] = None
    is_predicted: bool = False


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
    # Metadata / skeleton parsing
    # --------------------------
    def _read_metadata_obj(self) -> Optional[object]:
        # /metadata dataset often stores JSON
        meta_ds = self._maybe_ds("metadata")
        if meta_ds is not None:
            try:
                obj = safe_json_loads(meta_ds[()])
                if obj is not None:
                    return obj
            except Exception:
                pass

        # /metadata group may store attrs
        meta_grp = self._maybe_grp("metadata")
        if meta_grp is not None:
            try:
                for _, v in meta_grp.attrs.items():
                    obj = safe_json_loads(v)
                    if obj is not None:
                        return obj
            except Exception:
                pass

        # Sometimes other *_json datasets exist
        for k in self.root_keys():
            if k.endswith("_json"):
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
        # Handles {"py/id": 3}, {"id": 3}, or direct ints
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
        """
        Parse skeletons as stored by SLEAP in /metadata:
          meta["nodes"] = [{"name": ...}, ...]
          meta["skeletons"] = [
             {
               "nodes": [{"id": <global_node_idx>}, ...],
               "links": [{"source": {"id":...}, "target": {"id":...}, "type": 1|2}, ...],
               "graph": {"name": "..."} (optional)
             },
             ...
          ]
        link.type==1 => edges
        link.type==2 => symmetry_pairs
        """
        if not isinstance(meta, dict):
            return None

        nodes = meta.get("nodes")
        skels = meta.get("skeletons")
        if not (isinstance(nodes, list) and isinstance(skels, list) and skels):
            return None

        global_names: List[str] = []
        for n in nodes:
            if isinstance(n, dict) and "name" in n:
                nm = str(n.get("name", "")).strip()
                global_names.append(nm if nm else f"node_{len(global_names)}")
            else:
                global_names.append(f"node_{len(global_names)}")

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

            node_ids: List[int] = []
            for node in sk_nodes:
                nid = SleapH5Reader._get_id(node)
                if nid is None:
                    continue
                node_ids.append(nid)

            if not node_ids:
                continue

            local_names: List[str] = []
            for gid in node_ids:
                if 0 <= gid < len(global_names):
                    local_names.append(global_names[gid])
                else:
                    local_names.append(f"node_{gid}")

            idx_map = {gid: li for li, gid in enumerate(node_ids)}

            edges: List[Tuple[str, str]] = []
            sym: List[Tuple[str, str]] = []

            links = sk.get("links", [])
            if isinstance(links, list):
                for lk in links:
                    if not isinstance(lk, dict):
                        continue
                    src = SleapH5Reader._get_id(lk.get("source"))
                    dst = SleapH5Reader._get_id(lk.get("target"))
                    if src is None or dst is None:
                        continue
                    if src not in idx_map or dst not in idx_map:
                        continue
                    a = local_names[idx_map[src]]
                    b = local_names[idx_map[dst]]
                    if not a or not b or a == b:
                        continue
                    t = lk.get("type")
                    if t == 2:
                        sym.append((a, b))
                    else:
                        # default to edges for type==1 (and any unknowns)
                        edges.append((a, b))

            # de-dup preserve order
            def dedup(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
                seen = set()
                out2: List[Tuple[str, str]] = []
                for a, b in pairs:
                    key = (a, b)
                    if key in seen:
                        continue
                    seen.add(key)
                    out2.append((a, b))
                return out2

            out.append(
                SkeletonSpec(
                    names=local_names,
                    edges=dedup(edges),
                    symmetry_pairs=dedup(sym),
                    name=sk_name or f"skeleton_{sidx}",
                )
            )

        return out or None

    def read_skeletons(self) -> List[SkeletonSpec]:
        meta = self._read_metadata_obj()
        skels = self._parse_sleap_metadata_skeletons(meta) if meta is not None else None
        if skels:
            return skels

        # Fallback: try to infer a single skeleton by points-per-instance
        inst = self._require_ds("instances")
        if inst.shape[0] == 0:
            raise ValueError("No instances found in file; cannot infer skeleton.")
        row0 = inst[0]
        ps = int(row0["point_id_start"])
        pe = int(row0["point_id_end"])
        n_nodes = max(0, pe - ps)
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

    # --------------------------
    # Instances / points reading
    # --------------------------
    def iter_instances(self, skeletons: List[SkeletonSpec]) -> Iterable[InstanceRecord]:
        frames = self._require_ds("frames")
        inst = self._require_ds("instances")
        pts = self._require_ds("points")
        pred_pts = self._maybe_ds("pred_points")

        pt_fields = set(pts.dtype.fields.keys()) if pts.dtype.fields else set()
        if not {"x", "y"}.issubset(pt_fields):
            raise KeyError(
                f"/points missing x/y columns. Found fields: {sorted(pt_fields)}"
            )

        inst_fields = set(inst.dtype.fields.keys()) if inst.dtype.fields else set()
        has_instance_type = "instance_type" in inst_fields
        has_skeleton_id = "skeleton" in inst_fields

        for fr in frames:
            frame_idx = int(fr["frame_idx"])
            i0 = int(fr["instance_id_start"])
            i1 = int(fr["instance_id_end"])
            if i1 <= i0:
                continue

            for inst_row in inst[i0:i1]:
                track_id = int(inst_row["track"]) if "track" in inst_fields else -1
                sk_id = int(inst_row["skeleton"]) if has_skeleton_id else 0
                sk_id = sk_id if 0 <= sk_id < len(skeletons) else 0
                sk = skeletons[sk_id]

                ps = int(inst_row["point_id_start"])
                pe = int(inst_row["point_id_end"])
                n = pe - ps
                if n <= 0:
                    continue

                is_pred = False
                point_ds: h5py.Dataset = pts
                if has_instance_type:
                    it = int(inst_row["instance_type"])
                    if it == INSTANCE_TYPE_PREDICTED:
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

                xs = point_ds["x"][ps:pe].astype(np.float32, copy=False)
                ys = point_ds["y"][ps:pe].astype(np.float32, copy=False)
                xy = np.stack([xs, ys], axis=1)

                vis = None
                if "visible" in pfields:
                    vis = point_ds["visible"][ps:pe].astype(bool, copy=False)

                # ✅ This is the key sleap-io invariant:
                # points slice is ordered to match skeleton.node_names.
                if n != len(sk.names):
                    raise ValueError(
                        "Point slice length does not match skeleton node count. "
                        f"slice n={n}, skeleton[{sk_id}] nodes={len(sk.names)}. "
                        "This is unexpected for standard .slp written by SLEAP/sleap-io."
                    )

                yield InstanceRecord(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    skeleton_id=sk_id,
                    points_xy=xy,
                    visible=vis,
                    is_predicted=is_pred,
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
    pairs: List[Tuple[str, str]] = []
    try:
        pairs.extend(PoseSchema.infer_symmetry_pairs(keypoints))  # type: ignore[attr-defined]
    except Exception:
        pass
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

    try:
        schema.normalize_prefixed_keypoints()  # type: ignore[attr-defined]
    except Exception:
        pass

    canonical = {str(kp).strip().lower(): str(kp).strip() for kp in schema.keypoints}
    for n in skeleton_names:
        key = str(n).strip().lower()
        mapper[n] = canonical.get(key, n)
    return mapper


def write_pose_schema(out_dir: Path, skeleton: SkeletonSpec, *, filename: str) -> Path:
    """
    Write pose schema JSON to output folder.
    - keypoints: skeleton.names
    - edges: skeleton.edges
    - symmetry_pairs: skeleton.symmetry_pairs if present else inferred
    """
    sym = skeleton.symmetry_pairs or _infer_symmetry_pairs_extended(skeleton.names)
    schema = PoseSchema(
        version="1.0",
        keypoints=list(skeleton.names),
        edges=list(skeleton.edges) if skeleton.edges else [],
        symmetry_pairs=sym,
        flip_idx=None,
        instances=[],
        instance_separator="_",
    )
    out_path = out_dir / filename
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
            "Currently this converter expects HDF5 .slp/.pkg.slp. "
            f"Your file looks like: {kind}."
        )

    reader = SleapH5Reader(Path(sleap_path))
    try:
        skeletons = reader.read_skeletons()
        if not skeletons:
            raise ValueError("No skeletons found.")

        # Load user-provided schema if available
        pose_schema: Optional[PoseSchema]
        if pose_schema_path:
            pose_schema = PoseSchema.load(pose_schema_path)
        else:
            pose_schema = None

        # ✅ Always write pose schema(s) with real edges from SLEAP skeletons
        schema_paths: List[Path] = []
        if len(skeletons) == 1:
            # single skeleton => write pose_schema.json
            schema_paths.append(
                write_pose_schema(out_dir, skeletons[0], filename="pose_schema.json")
            )
        else:
            # multiple skeletons => write per-skeleton + a default pose_schema.json for skeleton_0
            schema_paths.append(
                write_pose_schema(out_dir, skeletons[0], filename="pose_schema.json")
            )
            for i, sk in enumerate(skeletons):
                schema_paths.append(
                    write_pose_schema(out_dir, sk, filename=f"pose_schema_{i}.json")
                )

        # If user provided pose schema, also save normalized copy as pose_schema_user.json
        if pose_schema is not None:
            try:
                # type: ignore[attr-defined]
                pose_schema.normalize_prefixed_keypoints()
            except Exception:
                pass
            user_copy = out_dir / "pose_schema_user.json"
            pose_schema.save(user_copy)

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

        for k, rec in enumerate(reader.iter_instances(skeletons)):
            per_frame_shapes.setdefault(rec.frame_idx, [])
            sk = skeletons[rec.skeleton_id]

            # Label mapping (per skeleton)
            label_map = build_label_mapper(sk.names, pose_schema)

            for j, (x, y) in enumerate(rec.points_xy):
                # Skip non-visible points if visibility exists
                if rec.visible is not None and not bool(rec.visible[j]):
                    continue
                # Skip NaNs
                if np.isnan(x) or np.isnan(y):
                    continue

                # ✅ Correct node mapping: index j corresponds to sk.names[j]
                if j >= len(sk.names):
                    continue

                skel_name = sk.names[j]
                label = label_map.get(skel_name, skel_name)

                gid = int(rec.track_id)
                group_id = None if gid < 0 else gid

                shape = {
                    "label": label,
                    "points": [[float(x), float(y)]],
                    "group_id": group_id,
                    "shape_type": "point",
                    "flags": {"predicted": bool(rec.is_predicted)},
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
        print("[sleap2labelme] wrote pose schema(s):")
        for p in schema_paths:
            print(f"  - {p.name}")
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
        "A normalized copy is written as pose_schema_user.json.",
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
