#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    # Fallback: many .slp are HDF5 even without a perfect magic read (rare),
    # but don't guess—raise a helpful error.
    raise ValueError(f"Unrecognized SLEAP file container for {p} (magic={magic!r}).")


def safe_json_loads(raw: object) -> Optional[object]:
    """Try to parse a JSON string/bytes stored in HDF5 datasets."""
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
class NodeSpec:
    names: List[str]


@dataclass(frozen=True)
class InstanceRecord:
    frame_idx: int
    track_id: int
    points_xy: np.ndarray  # (n_nodes, 2) float
    visible: Optional[np.ndarray] = None  # (n_nodes,) bool


class SleapH5Reader:
    """
    Reader for the schema you printed:
      /frames (frame_idx, instance_id_start/end, ...)
      /instances (track, point_id_start/end, frame_id, ...)
      /points (x, y, visible, complete)
      /video0/video (object array: embedded frames) + /video0/frame_numbers
      /videos_json, /tracks_json (may be empty)
      /metadata (often exists; may contain skeleton info, depending on exporter)
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

    def _require(self, key: str) -> h5py.Dataset:
        if key not in self._h5:
            raise KeyError(
                f"Missing dataset/group: {key}. Root keys: {self.root_keys()}"
            )
        obj = self._h5[key]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"{key} is not a dataset (found {type(obj)}).")
        return obj

    def _maybe(self, key: str) -> Optional[h5py.Dataset]:
        obj = self._h5.get(key, None)
        if isinstance(obj, h5py.Dataset):
            return obj
        return None

    def infer_nodes(self) -> NodeSpec:
        """
        Best-effort:
        1) Try to parse skeleton node names from /metadata or *_json datasets.
        2) Otherwise infer node count from first instance's point slice length and name nodes node_0..node_{n-1}.
        """
        # (1) metadata dataset sometimes contains JSON with skeleton definitions
        meta_ds = self._maybe("metadata")
        if meta_ds is not None and len(meta_ds.shape) == 0:
            meta_obj = safe_json_loads(meta_ds[()])
            names = self._extract_node_names_from_metadata(meta_obj)
            if names:
                return NodeSpec(names=names)

        # (1b) sometimes videos_json or other json datasets carry skeletons
        for k in self.root_keys():
            if k.endswith("_json"):
                ds = self._maybe(k)
                if ds is None:
                    continue
                try:
                    raw = ds[()]
                except Exception:
                    continue
                obj = safe_json_loads(raw)
                names = self._extract_node_names_from_metadata(obj)
                if names:
                    return NodeSpec(names=names)

        # (2) infer from first instance
        inst = self._require("instances")
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
        return NodeSpec(names=[f"node_{i}" for i in range(n_nodes)])

    @staticmethod
    def _extract_node_names_from_metadata(obj: object) -> Optional[List[str]]:
        """
        Very defensive JSON probing. Different SLEAP writers differ.
        We'll search for structures like:
          {"skeletons":[{"nodes":[{"name":"head"}, ...]}]}
          {"skeletons":[{"node_names":[...]}]}
          {"skeleton":{"nodes":[...]}}, etc.
        """
        if not isinstance(obj, dict):
            return None

        candidates: List[object] = []
        for key in ("skeletons", "skeleton", "project", "labels"):
            if key in obj:
                candidates.append(obj[key])

        # Flatten some common patterns
        def walk(x: object) -> Iterable[dict]:
            if isinstance(x, dict):
                yield x
                for v in x.values():
                    yield from walk(v)
            elif isinstance(x, list):
                for v in x:
                    yield from walk(v)

        for d in walk(candidates):
            # node_names
            if (
                "node_names" in d
                and isinstance(d["node_names"], list)
                and d["node_names"]
            ):
                if all(isinstance(n, str) for n in d["node_names"]):
                    return list(d["node_names"])
            # nodes: [{name:...}]
            if "nodes" in d and isinstance(d["nodes"], list) and d["nodes"]:
                if all(isinstance(n, dict) and "name" in n for n in d["nodes"]):
                    names = [str(n["name"]) for n in d["nodes"]]
                    if names:
                        return names
        return None

    def iter_instances(self, nodes: NodeSpec) -> Iterable[InstanceRecord]:
        frames = self._require("frames")
        inst = self._require("instances")
        pts = self._require("points")

        # Validate /points has x,y at minimum (your print shows it does)
        pt_fields = set(pts.dtype.fields.keys()) if pts.dtype.fields else set()
        if not {"x", "y"}.issubset(pt_fields):
            raise KeyError(
                f"/points missing x/y columns. Found fields: {sorted(pt_fields)}"
            )

        has_visible = "visible" in pt_fields

        # frames rows give a contiguous range of instances
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

                # Some files can have varying node counts; clamp to known node list
                n_use = min(n, len(nodes.names))

                xs = pts["x"][ps : ps + n_use].astype(np.float32, copy=False)
                ys = pts["y"][ps : ps + n_use].astype(np.float32, copy=False)
                xy = np.stack([xs, ys], axis=1)

                vis = None
                if has_visible:
                    vis = pts["visible"][ps : ps + n_use].astype(bool, copy=False)

                yield InstanceRecord(
                    frame_idx=frame_idx, track_id=track_id, points_xy=xy, visible=vis
                )

    def save_embedded_frames(
        self,
        out_dir: Path,
        video_index: int = 0,
        image_ext: str = ".png",
        name_prefix: str = "",
    ) -> Dict[int, str]:
        """
        Save embedded frames from /video{idx}/video directly into out_dir/ (same folder as JSONs).
        Returns mapping: frame_idx -> relative image filename (e.g., "frame_000123.png").
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        video_key = f"video{video_index}/video"
        frame_nums_key = f"video{video_index}/frame_numbers"

        video_ds = self._maybe(video_key)
        frame_nums_ds = self._maybe(frame_nums_key)

        if video_ds is None:
            raise KeyError(
                f"Embedded frames not found at /{video_key}. "
                f"Available root keys: {self.root_keys()}"
            )

        if frame_nums_ds is None:
            frame_numbers = np.arange(int(video_ds.shape[0]), dtype=np.int64)
        else:
            frame_numbers = np.array(frame_nums_ds[:], dtype=np.int64)

        mapping: Dict[int, str] = {}
        n = int(video_ds.shape[0])

        # normalize prefix
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

            # matches your JSON naming style
            fname = f"{prefix}{frame_idx:09d}{image_ext}"
            fpath = out_dir / fname
            img.save(fpath)

            # IMPORTANT: return relative filename (same folder as JSON)
            mapping[frame_idx] = fname

        return mapping

    @staticmethod
    def _decode_embedded_frame(obj: object) -> Image.Image:
        """
        Decode an embedded frame that may be stored as:
          - bytes/bytearray/np.void (encoded PNG/JPEG bytes)
          - 1D numpy array of int8/uint8 (encoded bytes buffer)
          - 2D/3D numpy array (raw pixel array)
        """

        def try_decode_bytes(b: bytes) -> Optional[Image.Image]:
            if not b:
                return None
            # Heuristic: trim leading/trailing zeros sometimes present
            # (safe: only removes zeros at ends)
            b2 = b.strip(b"\x00")
            for payload in (b, b2) if b2 != b else (b,):
                try:
                    im = Image.open(io.BytesIO(payload))
                    return im.convert("RGB")
                except Exception:
                    pass
            return None

        # Case 1: bytes-like already
        if isinstance(obj, (bytes, bytearray, np.void)):
            im = try_decode_bytes(bytes(obj))
            if im is not None:
                return im

        # Case 2: numpy array
        if isinstance(obj, np.ndarray):
            arr = obj

            # 2a: object array element might itself be bytes
            if arr.dtype == object and arr.ndim == 0:
                inner = arr.item()
                return SleapH5Reader._decode_embedded_frame(inner)

            # 2b: 1D buffer of int8/uint8 -> encoded image bytes
            if arr.ndim == 1 and arr.dtype in (np.int8, np.uint8):
                # int8 can be negative; reinterpret as unsigned bytes
                b = arr.astype(np.uint8, copy=False).tobytes()
                im = try_decode_bytes(b)
                if im is not None:
                    return im
                # Fall through: maybe it's raw grayscale samples (rare)
                # If so, we can't infer width/height reliably.
                raise ValueError(
                    f"Embedded frame looks like a 1D byte buffer but could not decode as an image "
                    f"(len={arr.shape[0]}, dtype={arr.dtype})."
                )

            # 2c: raw pixel arrays
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

            # Sometimes HDF5 stores encoded bytes as (N,) but dtype=int16/int32
            if (
                arr.ndim == 1
                and arr.dtype.kind in ("i", "u")
                and arr.itemsize in (2, 4, 8)
            ):
                # Try interpret low 8 bits as bytes
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

        # Case 3: other python objects (e.g., memoryview)
        if isinstance(obj, memoryview):
            im = try_decode_bytes(obj.tobytes())
            if im is not None:
                return im

        raise ValueError(f"Unsupported embedded frame type: {type(obj)}")


def make_labelme_json(
    image_relpath: str,
    image_size: Tuple[int, int],
    shapes: List[dict],
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


def convert_sleap_h5_to_labelme(
    sleap_path: str | Path,
    out_dir: str | Path,
    *,
    save_frames: bool = True,
    video_index: int = 0,
    print_every: int = 200,
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
        nodes = reader.infer_nodes()

        frame_to_image: Dict[int, str] = {}
        frame_to_wh: Dict[int, Tuple[int, int]] = {}

        if save_frames:
            frame_to_image = reader.save_embedded_frames(
                out_dir, video_index=video_index
            )
            # Read one saved image per frame to get W,H for LabelMe JSON
            for fidx, rel in frame_to_image.items():
                p = out_dir / rel
                with Image.open(p) as im:
                    frame_to_wh[fidx] = (im.width, im.height)

        # Accumulate shapes per frame
        per_frame_shapes: Dict[int, List[dict]] = {}

        for k, rec in enumerate(reader.iter_instances(nodes)):
            # If we didn't save frames, still write jsons, but imagePath will be blank.
            if rec.frame_idx not in per_frame_shapes:
                per_frame_shapes[rec.frame_idx] = []

            # Create one point shape per node
            for node_i, (x, y) in enumerate(rec.points_xy):
                label = (
                    nodes.names[node_i]
                    if node_i < len(nodes.names)
                    else f"node_{node_i}"
                )
                if np.isnan(x) or np.isnan(y):
                    continue
                shape = {
                    "label": label,
                    "points": [[float(x), float(y)]],
                    "group_id": int(rec.track_id),
                    "shape_type": "point",
                    "flags": {},
                }
                per_frame_shapes[rec.frame_idx].append(shape)

            if print_every and (k + 1) % print_every == 0:
                print(f"[sleap2labelme] processed {k + 1} instances...")

        # Write LabelMe json per frame
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
    args = ap.parse_args()

    convert_sleap_h5_to_labelme(
        args.sleap,
        args.out,
        save_frames=not args.no_save_frames,
        video_index=args.video_index,
        print_every=args.print_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
