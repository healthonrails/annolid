from __future__ import annotations

from dataclasses import dataclass
import json
import math
import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    visibility_from_labelme_shape,
)
from annolid.features import Dinov3Config, Dinov3FeatureExtractor
from annolid.segmentation.dino_kpseg.keypoints import infer_flip_idx_from_names


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_LABELME_SUFFIX = ".json"


def _resolve_yaml_path(value: str, *, yaml_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (yaml_path.parent / path).resolve()


def _resolve_dataset_path(
    value: str,
    *,
    yaml_path: Path,
    root_path: Optional[Path],
) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if root_path is not None:
        return (root_path / path).resolve()
    return (yaml_path.parent / path).resolve()


def _yolo_list_images(images_root: Path) -> List[Path]:
    if not images_root.exists():
        return []
    if images_root.is_file():
        if images_root.suffix.lower() in {".txt", ".list"}:
            try:
                lines = images_root.read_text(encoding="utf-8").splitlines()
            except Exception:
                return []
            base = images_root.parent
            out: List[Path] = []
            for line in lines:
                raw = str(line).strip()
                if not raw:
                    continue
                p = Path(raw).expanduser()
                if not p.is_absolute():
                    p = (base / p).resolve()
                if p.exists():
                    out.append(p)
            return out
    paths: List[Path] = []
    for suffix in _IMAGE_SUFFIXES:
        paths.extend(sorted(images_root.rglob(f"*{suffix}")))
    return paths


def _image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
    except ValueError:
        pass
    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def _image_to_labelme_path(image_path: Path) -> Path:
    """Best-effort mapping from image path -> LabelMe JSON path.

    When `images/` appears in the path, it is swapped for `annotations/`.
    Otherwise, the JSON is assumed to be a sidecar next to the image.
    """
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "annotations"
    except ValueError:
        pass
    return Path(*parts).with_suffix(_LABELME_SUFFIX)


@dataclass(frozen=True)
class YoloPoseLabelsSummary:
    images_total: int
    label_files_found: int
    images_with_pose_instances: int
    pose_instances_total: int
    invalid_lines_total: int
    example_issues: List[str]


def summarize_yolo_pose_labels(
    image_paths: Iterable[Path],
    *,
    kpt_count: int,
    kpt_dims: int,
    max_issues: int = 10,
) -> YoloPoseLabelsSummary:
    """Lightweight integrity check for a YOLO pose dataset."""
    total = 0
    label_found = 0
    images_with_instances = 0
    instances_total = 0
    invalid_lines = 0
    issues: List[str] = []

    for image_path in image_paths:
        total += 1
        label_path = _image_to_label_path(Path(image_path))
        if not label_path.exists():
            if len(issues) < int(max_issues):
                issues.append(f"Missing label file: {label_path}")
            continue
        label_found += 1
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            if len(issues) < int(max_issues):
                issues.append(f"Unreadable label file: {label_path}")
            continue

        valid_instances = 0
        for line in lines:
            tokens = str(line).strip().split()
            if not tokens:
                continue
            kpts = _parse_yolo_pose_line(
                tokens, kpt_count=int(kpt_count), dims=int(kpt_dims)
            )
            if kpts is None:
                invalid_lines += 1
                continue
            valid_instances += 1
        if valid_instances <= 0:
            if len(issues) < int(max_issues):
                issues.append(f"No pose instances in: {label_path}")
            continue
        images_with_instances += 1
        instances_total += valid_instances

    return YoloPoseLabelsSummary(
        images_total=int(total),
        label_files_found=int(label_found),
        images_with_pose_instances=int(images_with_instances),
        pose_instances_total=int(instances_total),
        invalid_lines_total=int(invalid_lines),
        example_issues=issues,
    )


@dataclass(frozen=True)
class LabelMePoseLabelsSummary:
    images_total: int
    label_files_found: int
    images_with_pose_instances: int
    pose_instances_total: int
    invalid_shapes_total: int
    example_issues: List[str]


def _safe_float_pair(points: object) -> Optional[Tuple[float, float]]:
    if not isinstance(points, (list, tuple)) or not points:
        return None
    pt = points[0]
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None
    try:
        return float(pt[0]), float(pt[1])
    except Exception:
        return None


def _labelme_keypoint_instances_dict_from_payload(
    payload: Dict[str, object],
    *,
    keypoint_names: List[str],
    kpt_dims: int,
    image_hw: Tuple[int, int],
) -> Dict[int, np.ndarray]:
    """Extract per-instance keypoint arrays from a LabelMe payload.

    - Keypoints are `shape_type == "point"`.
    - Instance association uses `group_id` (defaulting to 0 when missing).
    - Output is always `[K,3]` with YOLO/COCO visibility in the last column.
    """
    height, width = int(image_hw[0]), int(image_hw[1])
    if width <= 0 or height <= 0:
        return {}

    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        return {}

    name_to_idx = {str(n): int(i) for i, n in enumerate(keypoint_names)}
    instances: Dict[int, np.ndarray] = {}

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if shape.get("shape_type") != "point":
            continue
        label = str(shape.get("label") or "").strip()
        if not label or label not in name_to_idx:
            continue
        xy = _safe_float_pair(shape.get("points"))
        if xy is None:
            continue
        x_px, y_px = xy
        x_norm = x_px / max(1.0, float(width))
        y_norm = y_px / max(1.0, float(height))
        x_norm = float(np.clip(x_norm, 0.0, 1.0))
        y_norm = float(np.clip(y_norm, 0.0, 1.0))

        try:
            gid_raw = shape.get("group_id")
            gid = int(gid_raw) if gid_raw is not None else 0
        except Exception:
            gid = 0

        inst = instances.get(gid)
        if inst is None:
            inst = np.zeros((len(keypoint_names), 3), dtype=np.float32)
            instances[gid] = inst

        idx = int(name_to_idx[label])
        v = visibility_from_labelme_shape(shape)
        if kpt_dims >= 3:
            vis = int(v) if v is not None else int(KeypointVisibility.VISIBLE)
        else:
            vis = int(KeypointVisibility.VISIBLE)

        # Prefer the most-visible annotation when duplicates exist.
        if inst[idx, 2] <= 0 or float(vis) > float(inst[idx, 2]):
            inst[idx, 0] = float(x_norm)
            inst[idx, 1] = float(y_norm)
            inst[idx, 2] = float(vis)

    return instances


def _labelme_keypoint_instances_from_payload(
    payload: Dict[str, object],
    *,
    keypoint_names: List[str],
    kpt_dims: int,
    image_hw: Tuple[int, int],
) -> List[np.ndarray]:
    instances = _labelme_keypoint_instances_dict_from_payload(
        payload,
        keypoint_names=keypoint_names,
        kpt_dims=kpt_dims,
        image_hw=image_hw,
    )
    return [arr for _, arr in sorted(instances.items(), key=lambda kv: kv[0])]


def _labelme_keypoint_instances_with_group_ids_from_payload(
    payload: Dict[str, object],
    *,
    keypoint_names: List[str],
    kpt_dims: int,
    image_hw: Tuple[int, int],
) -> List[Tuple[int, np.ndarray]]:
    instances = _labelme_keypoint_instances_dict_from_payload(
        payload,
        keypoint_names=keypoint_names,
        kpt_dims=kpt_dims,
        image_hw=image_hw,
    )
    return [(gid, arr) for gid, arr in sorted(instances.items(), key=lambda kv: kv[0])]


def summarize_labelme_pose_labels(
    image_paths: Iterable[Path],
    *,
    label_paths: Optional[Iterable[Optional[Path]]] = None,
    keypoint_names: List[str],
    kpt_dims: int,
    max_issues: int = 10,
) -> LabelMePoseLabelsSummary:
    """Lightweight integrity check for a LabelMe pose dataset."""
    total = 0
    label_found = 0
    images_with_instances = 0
    instances_total = 0
    invalid_shapes = 0
    issues: List[str] = []

    label_paths_list = list(label_paths) if label_paths is not None else None

    from annolid.datasets.labelme_collection import resolve_image_path
    from annolid.utils.annotation_store import load_labelme_json

    for idx, image_path in enumerate(image_paths):
        total += 1
        image_path = Path(image_path)
        json_path = None
        if label_paths_list is not None and idx < len(label_paths_list):
            json_path = label_paths_list[idx]
        if json_path is None:
            candidate = _image_to_labelme_path(image_path)
            json_path = candidate if candidate.exists() else None

        if json_path is None or not Path(json_path).exists():
            if len(issues) < int(max_issues):
                issues.append(f"Missing LabelMe JSON for image: {image_path}")
            continue
        label_found += 1

        try:
            payload = load_labelme_json(json_path)
        except Exception:
            if len(issues) < int(max_issues):
                issues.append(f"Unreadable LabelMe JSON: {json_path}")
            continue
        if not isinstance(payload, dict):
            if len(issues) < int(max_issues):
                issues.append(f"Invalid LabelMe JSON payload: {json_path}")
            continue

        h = payload.get("imageHeight")
        w = payload.get("imageWidth")
        try:
            image_hw = (int(h), int(w))
        except Exception:
            image_hw = (0, 0)

        if image_hw[0] <= 0 or image_hw[1] <= 0:
            resolved = resolve_image_path(Path(json_path))
            if resolved and resolved.exists():
                try:
                    with Image.open(resolved) as pil:
                        pil = ImageOps.exif_transpose(pil)
                        w2, h2 = pil.size
                    image_hw = (int(h2), int(w2))
                except Exception:
                    image_hw = (0, 0)

        inst = _labelme_keypoint_instances_from_payload(
            payload,
            keypoint_names=list(keypoint_names),
            kpt_dims=int(kpt_dims),
            image_hw=image_hw,
        )
        if not inst:
            shapes = payload.get("shapes")
            if isinstance(shapes, list):
                invalid_shapes += len(shapes)
            if len(issues) < int(max_issues):
                issues.append(f"No keypoint instances in: {json_path}")
            continue

        valid_instances = 0
        for arr in inst:
            if arr.ndim != 2 or arr.shape[1] < 3:
                invalid_shapes += 1
                continue
            if not bool((arr[:, 2] > 0).any()):
                continue
            valid_instances += 1
        if valid_instances <= 0:
            if len(issues) < int(max_issues):
                issues.append(f"No visible keypoints in: {json_path}")
            continue

        images_with_instances += 1
        instances_total += valid_instances

    return LabelMePoseLabelsSummary(
        images_total=int(total),
        label_files_found=int(label_found),
        images_with_pose_instances=int(images_with_instances),
        pose_instances_total=int(instances_total),
        invalid_shapes_total=int(invalid_shapes),
        example_issues=issues,
    )


def _parse_yolo_pose_line(
    tokens: List[str], *, kpt_count: int, dims: int
) -> Optional[np.ndarray]:
    # YOLO pose format: cls x y w h (kpt_count * dims)
    start = 5
    required = start + kpt_count * dims
    if len(tokens) < required:
        return None
    raw = tokens[start:required]
    try:
        values = [float(v) for v in raw]
    except ValueError:
        return None

    values = np.asarray(values, dtype=np.float32).reshape(kpt_count, dims)
    if dims == 2:
        xy = values
        vis = np.ones((kpt_count, 1), dtype=np.float32) * 2.0
        return np.concatenate([xy, vis], axis=1)
    if dims >= 3:
        return values[:, :3].astype(np.float32, copy=False)
    return None


def _parse_yolo_pose_instance(
    tokens: List[str],
    *,
    kpt_count: int,
    dims: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # YOLO pose format: cls x y w h (kpt_count * dims)
    start = 5
    required = start + kpt_count * dims
    if len(tokens) < required:
        return None
    try:
        bbox = np.asarray(
            [float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])],
            dtype=np.float32,
        )
    except ValueError:
        return None
    kpts = _parse_yolo_pose_line(tokens, kpt_count=kpt_count, dims=dims)
    if kpts is None:
        return None
    return bbox, kpts


def _crop_instance(
    pil: Image.Image,
    *,
    bbox: np.ndarray,
    keypoints: np.ndarray,
    bbox_scale: float,
) -> Tuple[Image.Image, np.ndarray]:
    width, height = pil.size
    if width <= 1 or height <= 1:
        return pil, keypoints
    if bbox.shape[0] < 4:
        return pil, keypoints

    cx, cy, bw, bh = [float(x) for x in bbox[:4]]
    if bw <= 0 or bh <= 0:
        return pil, keypoints

    bw_px = float(bw) * float(width) * float(bbox_scale)
    bh_px = float(bh) * float(height) * float(bbox_scale)
    if bw_px <= 1.0 or bh_px <= 1.0:
        return pil, keypoints

    x1 = cx * float(width) - bw_px / 2.0
    y1 = cy * float(height) - bh_px / 2.0
    x2 = cx * float(width) + bw_px / 2.0
    y2 = cy * float(height) + bh_px / 2.0

    x1 = max(0, int(math.floor(x1)))
    y1 = max(0, int(math.floor(y1)))
    x2 = min(int(width), int(math.ceil(x2)))
    y2 = min(int(height), int(math.ceil(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return pil, keypoints

    crop = pil.crop((x1, y1, x2, y2))
    crop_w = float(max(1, x2 - x1))
    crop_h = float(max(1, y2 - y1))

    kpts = keypoints.astype(np.float32, copy=True)
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return crop, kpts

    x_px = kpts[:, 0] * float(width)
    y_px = kpts[:, 1] * float(height)
    x_norm = (x_px - float(x1)) / crop_w
    y_norm = (y_px - float(y1)) / crop_h
    inside = (x_norm >= 0.0) & (x_norm <= 1.0) & (y_norm >= 0.0) & (y_norm <= 1.0)

    kpts[:, 0] = np.clip(x_norm, 0.0, 1.0)
    kpts[:, 1] = np.clip(y_norm, 0.0, 1.0)
    if kpts.shape[1] >= 3:
        vis = kpts[:, 2]
        vis = np.where(inside, vis, np.zeros_like(vis))
        kpts[:, 2] = vis
    return crop, kpts


def _build_keypoint_targets(
    keypoints_instances: Iterable[np.ndarray],
    *,
    grid_hw: Tuple[int, int],
    patch_size: int,
    resized_hw_px: Tuple[int, int],
    radius_px: float,
    original_hw_px: Tuple[int, int],
    mask_type: str = "gaussian",
    heatmap_sigma_px: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kpt_instances = list(keypoints_instances)
    if not kpt_instances:
        raise ValueError("Expected at least one keypoint instance array")

    kpt_count = int(kpt_instances[0].shape[0])
    h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
    resized_h, resized_w = int(resized_hw_px[0]), int(resized_hw_px[1])
    orig_h, orig_w = int(original_hw_px[0]), int(original_hw_px[1])

    scale = 0.5 * ((resized_w / max(1, orig_w)) + (resized_h / max(1, orig_h)))
    radius_res = max(1.0, float(radius_px) * float(scale))
    radius2 = radius_res * radius_res

    x_centers = (np.arange(w_p, dtype=np.float32) + 0.5) * float(patch_size)
    y_centers = (np.arange(h_p, dtype=np.float32) + 0.5) * float(patch_size)
    yy = y_centers[:, None]
    xx = x_centers[None, :]

    mask_mode = str(mask_type or "gaussian").strip().lower()
    if mask_mode not in {"disk", "gaussian"}:
        raise ValueError(f"Unsupported mask_type: {mask_type!r}")

    sigma_res = None
    denom = None
    if mask_mode == "gaussian":
        if heatmap_sigma_px is not None and float(heatmap_sigma_px) > 0:
            sigma_res = float(heatmap_sigma_px) * float(scale)
        else:
            sigma_res = radius_res / 2.0
        sigma_res = max(1.0, float(sigma_res))
        denom = 2.0 * sigma_res * sigma_res

    masks = np.zeros((kpt_count, h_p, w_p), dtype=np.float32)
    coords = np.zeros((kpt_count, 2), dtype=np.float32)
    coord_counts = np.zeros((kpt_count,), dtype=np.int32)

    for keypoints in kpt_instances:
        if keypoints.shape[0] != kpt_count:
            continue
        for k in range(kpt_count):
            x_norm, y_norm, v = keypoints[k].tolist()
            if v <= 0:
                continue
            x_res = float(x_norm) * float(resized_w)
            y_res = float(y_norm) * float(resized_h)
            dist2 = (xx - x_res) ** 2 + (yy - y_res) ** 2
            if mask_mode == "gaussian" and denom is not None:
                blob = np.exp(-dist2 / float(denom)).astype(np.float32, copy=False)
            else:
                blob = (dist2 <= radius2).astype(np.float32)
            masks[k] = np.maximum(masks[k], blob)

            if coord_counts[k] == 0:
                coords[k, 0] = x_res
                coords[k, 1] = y_res
                coord_counts[k] = 1
            else:
                coord_counts[k] += 1

    coord_mask = (coord_counts == 1).astype(np.float32)
    return (
        torch.from_numpy(masks),
        torch.from_numpy(coords),
        torch.from_numpy(coord_mask),
    )


@dataclass(frozen=True)
class YoloPoseDatasetSpec:
    root: Path
    train_images: List[Path]
    val_images: List[Path]
    kpt_count: int
    kpt_dims: int
    keypoint_names: Optional[List[str]]
    flip_idx: Optional[List[int]]


def load_yolo_pose_spec(data_yaml: Path) -> YoloPoseDatasetSpec:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML: {data_yaml}")

    root = payload.get("path")
    if root:
        root_path = _resolve_yaml_path(str(root), yaml_path=data_yaml)
    else:
        root_path = data_yaml.parent

    train_val: Dict[str, List[Path]] = {}
    for split in ("train", "val"):
        split_value = payload.get(split)
        if not split_value:
            train_val[split] = []
            continue
        split_path = _resolve_dataset_path(
            str(split_value),
            yaml_path=data_yaml,
            root_path=root_path,
        )
        train_val[split] = _yolo_list_images(split_path)

    kpt_shape = payload.get("kpt_shape") or []
    if not isinstance(kpt_shape, (list, tuple)) or len(kpt_shape) < 2:
        raise ValueError("data.yaml missing kpt_shape for pose dataset")
    kpt_count = int(kpt_shape[0])
    kpt_dims = int(kpt_shape[1])

    keypoint_names = None
    kpt_names = payload.get("kpt_names")
    if isinstance(kpt_names, dict) and kpt_names:
        if 0 in kpt_names:
            names = kpt_names.get(0)
        elif "0" in kpt_names:
            names = kpt_names.get("0")
        else:
            names = next(iter(kpt_names.values()))
        if isinstance(names, list):
            cleaned = [str(k).strip() for k in names if str(k).strip()]
            if cleaned:
                keypoint_names = cleaned

    if keypoint_names is None:
        kpt_labels = payload.get("kpt_labels")
        if isinstance(kpt_labels, dict) and kpt_labels:
            names_by_idx: Dict[int, str] = {}
            for key, value in kpt_labels.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                label = str(value or "").strip()
                if label:
                    names_by_idx[idx] = label
            if names_by_idx:
                ordered = [names_by_idx.get(i, "") for i in range(kpt_count)]
                if all(ordered):
                    keypoint_names = ordered

    flip_idx = None
    raw_flip = payload.get("flip_idx")
    if isinstance(raw_flip, (list, tuple)) and raw_flip:
        try:
            candidate = [int(v) for v in raw_flip]
        except Exception:
            candidate = None
        if candidate and len(candidate) == kpt_count:
            flip_idx = candidate
    if flip_idx is None and keypoint_names:
        inferred = _infer_flip_idx_from_names(keypoint_names, kpt_count=kpt_count)
        if inferred is not None:
            flip_idx = inferred

    return YoloPoseDatasetSpec(
        root=root_path,
        train_images=train_val["train"],
        val_images=train_val["val"],
        kpt_count=kpt_count,
        kpt_dims=kpt_dims,
        keypoint_names=keypoint_names,
        flip_idx=flip_idx,
    )


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


@dataclass(frozen=True)
class CocoPoseDatasetSpec:
    root: Path
    image_root: Path
    train_ann: Optional[Path]
    val_ann: Optional[Path]
    kpt_count: int
    kpt_dims: int
    keypoint_names: List[str]
    flip_idx: Optional[List[int]]
    category_names: List[str]
    category_id_to_index: Dict[int, int]
    val_split: float
    val_seed: int
    auto_val_split: bool


def _load_coco_payload(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read COCO annotation JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid COCO annotation JSON: {path}")
    return payload


def _resolve_coco_image_path(
    file_name: str,
    *,
    root: Path,
    image_root: Path,
) -> Optional[Path]:
    p = Path(str(file_name)).expanduser()
    if p.is_absolute():
        return p if p.exists() else None

    candidates = [
        (image_root / p).resolve(),
        (root / p).resolve(),
        (root / "images" / p).resolve(),
        (image_root / p.name).resolve(),
        (root / "images" / p.name).resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _install_file(src: Path, dst: Path, *, mode: str = "hardlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src)
        return
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported link mode: {mode!r}")


def load_coco_pose_spec(data_yaml: Path) -> CocoPoseDatasetSpec:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML: {data_yaml}")

    fmt = str(payload.get("format") or payload.get("type") or "coco").strip().lower()
    if fmt not in {"coco", "coco_pose", "coco_keypoints"}:
        raise ValueError(f"Unsupported COCO spec type/format: {fmt!r}")

    root = payload.get("path")
    root_path = (
        _resolve_yaml_path(str(root), yaml_path=data_yaml) if root else data_yaml.parent
    )

    image_root_val = payload.get("image_root") or payload.get("images") or "."
    image_root = _resolve_dataset_path(
        str(image_root_val),
        yaml_path=data_yaml,
        root_path=root_path,
    )

    train_ann = None
    val_ann = None
    if payload.get("train"):
        train_ann = _resolve_dataset_path(
            str(payload.get("train")),
            yaml_path=data_yaml,
            root_path=root_path,
        )
    if payload.get("val"):
        val_ann = _resolve_dataset_path(
            str(payload.get("val")),
            yaml_path=data_yaml,
            root_path=root_path,
        )
    if train_ann is None and val_ann is None:
        raise ValueError("COCO spec requires at least one of 'train' or 'val'")

    ann_payload = None
    for candidate in (train_ann, val_ann):
        if candidate is not None and candidate.exists():
            ann_payload = _load_coco_payload(candidate)
            break
    if ann_payload is None:
        missing = [str(p) for p in (train_ann, val_ann) if p is not None]
        raise ValueError(f"No readable COCO annotation JSON found: {missing}")

    categories = ann_payload.get("categories")
    cat_list = categories if isinstance(categories, list) else []
    category_names: List[str] = []
    category_id_to_index: Dict[int, int] = {}
    for cat in cat_list:
        if not isinstance(cat, dict):
            continue
        cid = _safe_int(cat.get("id"))
        if cid is None or cid in category_id_to_index:
            continue
        category_id_to_index[cid] = len(category_id_to_index)
        category_names.append(str(cat.get("name") or f"class_{cid}"))
    if not category_names:
        category_names = ["animal"]
        category_id_to_index = {1: 0}

    kpt_shape = payload.get("kpt_shape") or []
    kpt_count = 0
    kpt_dims = 3
    if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 1:
        kpt_count = int(kpt_shape[0])
        if len(kpt_shape) >= 2:
            kpt_dims = int(kpt_shape[1])

    category_keypoints: List[str] = []
    for cat in cat_list:
        if not isinstance(cat, dict):
            continue
        raw = cat.get("keypoints")
        if isinstance(raw, list):
            category_keypoints = [str(x).strip() for x in raw if str(x).strip()]
            if category_keypoints:
                break

    if kpt_count <= 0 and category_keypoints:
        kpt_count = len(category_keypoints)

    if kpt_count <= 0:
        annotations = ann_payload.get("annotations")
        ann_list = annotations if isinstance(annotations, list) else []
        for ann in ann_list:
            if not isinstance(ann, dict):
                continue
            kpts = ann.get("keypoints")
            if isinstance(kpts, list) and len(kpts) >= 3:
                kpt_count = len(kpts) // 3
                break
    if kpt_count <= 0:
        raise ValueError("Could not infer keypoint count from COCO spec/annotations")

    if kpt_dims not in (2, 3):
        raise ValueError(f"Unsupported kpt_dims: {kpt_dims} (expected 2 or 3)")

    keypoint_names_raw = payload.get("keypoint_names")
    if isinstance(keypoint_names_raw, list):
        keypoint_names = [str(x).strip() for x in keypoint_names_raw if str(x).strip()]
    elif category_keypoints:
        keypoint_names = list(category_keypoints)
    else:
        keypoint_names = [f"kp_{i}" for i in range(int(kpt_count))]

    if len(keypoint_names) != int(kpt_count):
        raise ValueError(
            f"Expected {int(kpt_count)} keypoint names, got {len(keypoint_names)}"
        )

    flip_idx = None
    raw_flip = payload.get("flip_idx")
    if isinstance(raw_flip, (list, tuple)) and raw_flip:
        try:
            candidate = [int(v) for v in raw_flip]
        except Exception:
            candidate = None
        if candidate and len(candidate) == int(kpt_count):
            flip_idx = candidate
    if flip_idx is None and keypoint_names:
        inferred = _infer_flip_idx_from_names(keypoint_names, kpt_count=int(kpt_count))
        if inferred is not None:
            flip_idx = inferred

    raw_val_split = payload.get("val_split")
    val_split = _safe_float(raw_val_split)
    if val_split is None:
        val_split = 0.1 if val_ann is None else 0.0
    val_split = float(max(0.0, min(0.9, val_split)))

    raw_seed = payload.get("val_seed")
    val_seed = _safe_int(raw_seed)
    if val_seed is None:
        val_seed = 0
    auto_val_split = _safe_bool(payload.get("auto_val_split"), default=True)

    return CocoPoseDatasetSpec(
        root=root_path,
        image_root=image_root,
        train_ann=train_ann,
        val_ann=val_ann,
        kpt_count=int(kpt_count),
        kpt_dims=int(kpt_dims),
        keypoint_names=list(keypoint_names),
        flip_idx=flip_idx,
        category_names=list(category_names),
        category_id_to_index=dict(category_id_to_index),
        val_split=val_split,
        val_seed=int(val_seed),
        auto_val_split=bool(auto_val_split),
    )


def materialize_coco_pose_as_yolo(
    *,
    spec: CocoPoseDatasetSpec,
    output_dir: Path,
    link_mode: str = "hardlink",
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    def _convert_split(
        split_name: str,
        ann_path: Optional[Path],
        *,
        include_image_ids: Optional[set[int]] = None,
    ) -> int:
        if ann_path is None:
            return 0
        payload = _load_coco_payload(ann_path)
        images_raw = payload.get("images")
        ann_raw = payload.get("annotations")
        images_list = images_raw if isinstance(images_raw, list) else []
        ann_list = ann_raw if isinstance(ann_raw, list) else []

        images_by_id: Dict[int, Dict[str, object]] = {}
        for rec in images_list:
            if not isinstance(rec, dict):
                continue
            img_id = _safe_int(rec.get("id"))
            if img_id is None:
                continue
            images_by_id[img_id] = rec

        ann_by_image: Dict[int, List[Dict[str, object]]] = {}
        for rec in ann_list:
            if not isinstance(rec, dict):
                continue
            img_id = _safe_int(rec.get("image_id"))
            if img_id is None:
                continue
            ann_by_image.setdefault(img_id, []).append(rec)

        written = 0
        for image_id in sorted(images_by_id.keys()):
            if include_image_ids is not None and int(image_id) not in include_image_ids:
                continue
            image_rec = images_by_id[image_id]
            file_name = str(image_rec.get("file_name") or "").strip()
            if not file_name:
                continue
            src_image = _resolve_coco_image_path(
                file_name,
                root=spec.root,
                image_root=spec.image_root,
            )
            if src_image is None:
                continue

            width = _safe_int(image_rec.get("width")) or 0
            height = _safe_int(image_rec.get("height")) or 0
            if width <= 0 or height <= 0:
                try:
                    with Image.open(src_image) as pil:
                        width, height = pil.size
                except Exception:
                    continue
            width_f = float(width)
            height_f = float(height)
            if width_f <= 0.0 or height_f <= 0.0:
                continue

            stem = f"{Path(file_name).stem}__{int(image_id)}"
            suffix = src_image.suffix or ".jpg"
            dst_image = output_dir / "images" / split_name / f"{stem}{suffix}"
            _install_file(src_image, dst_image, mode=link_mode)
            dst_label = output_dir / "labels" / split_name / f"{stem}.txt"

            lines: List[str] = []
            for ann in ann_by_image.get(image_id, []):
                bbox = ann.get("bbox")
                if not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                try:
                    x, y, bw, bh = [float(v) for v in bbox[:4]]
                except Exception:
                    continue
                if bw <= 0.0 or bh <= 0.0:
                    continue

                cx = (x + bw / 2.0) / width_f
                cy = (y + bh / 2.0) / height_f
                nw = bw / width_f
                nh = bh / height_f
                cx = float(np.clip(cx, 0.0, 1.0))
                cy = float(np.clip(cy, 0.0, 1.0))
                nw = float(np.clip(nw, 1e-8, 1.0))
                nh = float(np.clip(nh, 1e-8, 1.0))

                category_id = _safe_int(ann.get("category_id"))
                cls_idx = 0
                if category_id is not None:
                    cls_idx = int(spec.category_id_to_index.get(category_id, 0))

                raw_kpts = ann.get("keypoints")
                if not isinstance(raw_kpts, list) or len(raw_kpts) < (
                    int(spec.kpt_count) * 3
                ):
                    continue

                kp_tokens: List[str] = []
                for i in range(int(spec.kpt_count)):
                    try:
                        x_px = float(raw_kpts[3 * i + 0])
                        y_px = float(raw_kpts[3 * i + 1])
                        v = float(raw_kpts[3 * i + 2])
                    except Exception:
                        x_px, y_px, v = 0.0, 0.0, 0.0

                    x_n = float(np.clip(x_px / width_f, 0.0, 1.0))
                    y_n = float(np.clip(y_px / height_f, 0.0, 1.0))
                    if int(spec.kpt_dims) == 2:
                        kp_tokens.extend([f"{x_n:.6f}", f"{y_n:.6f}"])
                    else:
                        if v < 0.0:
                            vis = 0
                        elif v > 2.0:
                            vis = 2
                        else:
                            vis = int(round(v))
                        kp_tokens.extend([f"{x_n:.6f}", f"{y_n:.6f}", f"{vis:d}"])

                line = " ".join(
                    [
                        str(int(cls_idx)),
                        f"{cx:.6f}",
                        f"{cy:.6f}",
                        f"{nw:.6f}",
                        f"{nh:.6f}",
                        *kp_tokens,
                    ]
                )
                lines.append(line)

            dst_label.write_text(
                ("\n".join(lines) + ("\n" if lines else "")),
                encoding="utf-8",
            )
            written += 1
        return int(written)

    def _split_train_ids_for_val_from_train(
        ann_path: Path,
        *,
        val_split: float,
        seed: int,
    ) -> Tuple[set[int], set[int]]:
        payload = _load_coco_payload(ann_path)
        images_raw = payload.get("images")
        images_list = images_raw if isinstance(images_raw, list) else []
        image_ids = [
            int(img_id)
            for img_id in (
                _safe_int(rec.get("id")) if isinstance(rec, dict) else None
                for rec in images_list
            )
            if img_id is not None
        ]
        image_ids = sorted(set(image_ids))
        if len(image_ids) <= 1 or float(val_split) <= 0.0:
            return set(image_ids), set()

        val_count = int(round(float(len(image_ids)) * float(val_split)))
        val_count = max(1, min(len(image_ids) - 1, val_count))
        rng = random.Random(int(seed))
        val_ids = set(rng.sample(image_ids, val_count))
        train_ids = {int(i) for i in image_ids if int(i) not in val_ids}
        return train_ids, val_ids

    auto_val_split_used = False
    written_train = 0
    written_val = 0
    if (
        spec.train_ann is not None
        and spec.val_ann is None
        and spec.auto_val_split
        and float(spec.val_split) > 0.0
    ):
        train_ids, val_ids = _split_train_ids_for_val_from_train(
            spec.train_ann,
            val_split=float(spec.val_split),
            seed=int(spec.val_seed),
        )
        written_train = _convert_split(
            "train",
            spec.train_ann,
            include_image_ids=train_ids,
        )
        written_val = _convert_split(
            "val",
            spec.train_ann,
            include_image_ids=val_ids,
        )
        auto_val_split_used = bool(written_val > 0)
    else:
        written_train = _convert_split("train", spec.train_ann)
        written_val = _convert_split("val", spec.val_ann)

    names = list(spec.category_names or ["animal"])
    yolo_payload: Dict[str, object] = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": int(max(1, len(names))),
        "names": names,
        "kpt_shape": [int(spec.kpt_count), int(spec.kpt_dims)],
        "kpt_names": {
            int(i): list(spec.keypoint_names) for i in range(int(max(1, len(names))))
        },
    }
    if spec.flip_idx:
        yolo_payload["flip_idx"] = [int(x) for x in spec.flip_idx]
    if auto_val_split_used:
        yolo_payload["auto_val_split"] = True
        yolo_payload["val_split"] = float(spec.val_split)
        yolo_payload["val_seed"] = int(spec.val_seed)
        yolo_payload["source_val_missing"] = True
    yolo_payload["images_train_count"] = int(written_train)
    yolo_payload["images_val_count"] = int(written_val)

    yolo_yaml = output_dir / "data.yaml"
    yolo_yaml.write_text(
        yaml.safe_dump(yolo_payload, sort_keys=False),
        encoding="utf-8",
    )
    return yolo_yaml


@dataclass(frozen=True)
class LabelMePoseDatasetSpec:
    root: Path
    train_images: List[Path]
    val_images: List[Path]
    train_json: List[Optional[Path]]
    val_json: List[Optional[Path]]
    kpt_count: int
    kpt_dims: int
    keypoint_names: List[str]
    flip_idx: Optional[List[int]]


def _load_kpt_names(payload: Dict[str, object], *, kpt_count: int) -> List[str]:
    names: List[str] = []

    raw = payload.get("keypoint_names")
    if isinstance(raw, list):
        names = [str(x).strip() for x in raw if str(x).strip()]

    if not names:
        raw = payload.get("kpt_names")
        if isinstance(raw, list):
            names = [str(x).strip() for x in raw if str(x).strip()]

    if not names:
        raw = payload.get("kpt_names")
        if isinstance(raw, dict) and raw:
            if 0 in raw:
                candidate = raw.get(0)
            elif "0" in raw:
                candidate = raw.get("0")
            else:
                candidate = next(iter(raw.values()))
            if isinstance(candidate, list):
                names = [str(x).strip() for x in candidate if str(x).strip()]

    if not names:
        kpt_labels = payload.get("kpt_labels")
        if isinstance(kpt_labels, dict) and kpt_labels:
            by_idx: Dict[int, str] = {}
            for key, value in kpt_labels.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                label = str(value or "").strip()
                if label:
                    by_idx[idx] = label
            if by_idx:
                ordered = [by_idx.get(i, "") for i in range(int(kpt_count))]
                if all(ordered):
                    names = ordered

    if len(names) != int(kpt_count):
        raise ValueError(f"Expected {int(kpt_count)} keypoint names, got {len(names)}")
    return names


def _labelme_iter_pairs_from_dir(root: Path) -> Tuple[List[Path], List[Optional[Path]]]:
    from annolid.datasets.labelme_collection import resolve_image_path

    images: List[Path] = []
    jsons: List[Optional[Path]] = []
    for json_path in sorted(Path(root).rglob(f"*{_LABELME_SUFFIX}")):
        image_path = resolve_image_path(Path(json_path))
        if image_path is None:
            continue
        images.append(Path(image_path).expanduser().resolve())
        jsons.append(Path(json_path).expanduser().resolve())
    return images, jsons


def _labelme_iter_pairs_from_jsonl(
    index_path: Path,
) -> Tuple[List[Path], List[Optional[Path]]]:
    from annolid.datasets.labelme_collection import iter_label_index_records

    images: List[Path] = []
    jsons: List[Optional[Path]] = []
    for rec in iter_label_index_records(Path(index_path)):
        img = rec.get("image_path")
        js = rec.get("json_path")
        if not isinstance(img, str) or not img:
            continue
        img_path = Path(img).expanduser().resolve()
        js_path = (
            Path(js).expanduser().resolve() if isinstance(js, str) and js else None
        )
        if not img_path.exists():
            continue
        images.append(img_path)
        jsons.append(js_path if js_path and js_path.exists() else None)
    return images, jsons


def _labelme_iter_pairs_from_list(
    list_path: Path,
) -> Tuple[List[Path], List[Optional[Path]]]:
    from annolid.datasets.labelme_collection import resolve_image_path

    try:
        lines = Path(list_path).read_text(encoding="utf-8").splitlines()
    except Exception:
        return [], []

    images: List[Path] = []
    jsons: List[Optional[Path]] = []
    base = Path(list_path).parent
    for line in lines:
        raw = str(line).strip()
        if not raw or raw.startswith("#"):
            continue
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        if not p.exists():
            continue
        if p.suffix.lower() == _LABELME_SUFFIX:
            image_path = resolve_image_path(p)
            if image_path is None or not Path(image_path).exists():
                continue
            images.append(Path(image_path).expanduser().resolve())
            jsons.append(Path(p).expanduser().resolve())
            continue
        if p.suffix.lower() in _IMAGE_SUFFIXES:
            images.append(Path(p).expanduser().resolve())
            candidate = _image_to_labelme_path(Path(p))
            jsons.append(candidate if candidate.exists() else None)
            continue
    return images, jsons


def load_labelme_pose_spec(data_yaml: Path) -> LabelMePoseDatasetSpec:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML: {data_yaml}")

    fmt = str(payload.get("format") or payload.get("type") or "labelme").strip().lower()
    if fmt not in {"labelme", "labelme_pose", "labelme_dinokpseg"}:
        raise ValueError(f"Unsupported LabelMe spec type/format: {fmt!r}")

    root = payload.get("path")
    if root:
        root_path = _resolve_yaml_path(str(root), yaml_path=data_yaml)
    else:
        root_path = data_yaml.parent

    kpt_shape = payload.get("kpt_shape") or []
    if not isinstance(kpt_shape, (list, tuple)) or len(kpt_shape) < 2:
        raise ValueError("LabelMe spec missing kpt_shape")
    kpt_count = int(kpt_shape[0])
    kpt_dims = int(kpt_shape[1])
    if kpt_dims not in (2, 3):
        raise ValueError(f"Unsupported kpt_dims: {kpt_dims} (expected 2 or 3)")

    keypoint_names = _load_kpt_names(payload, kpt_count=kpt_count)

    flip_idx = None
    raw_flip = payload.get("flip_idx")
    if isinstance(raw_flip, (list, tuple)) and raw_flip:
        try:
            candidate = [int(v) for v in raw_flip]
        except Exception:
            candidate = None
        if candidate and len(candidate) == kpt_count:
            flip_idx = candidate
    if flip_idx is None and keypoint_names:
        inferred = _infer_flip_idx_from_names(keypoint_names, kpt_count=kpt_count)
        if inferred is not None:
            flip_idx = inferred

    def _load_split(value: object) -> Tuple[List[Path], List[Optional[Path]]]:
        if not value:
            return [], []
        split_path = _resolve_dataset_path(
            str(value),
            yaml_path=data_yaml,
            root_path=root_path,
        )
        if not split_path.exists():
            from annolid.datasets.labelme_collection import DEFAULT_LABEL_INDEX_DIRNAME

            fallback = root_path / DEFAULT_LABEL_INDEX_DIRNAME / Path(value).name
            if fallback.exists():
                split_path = fallback
        if split_path.is_dir():
            return _labelme_iter_pairs_from_dir(split_path)
        if split_path.is_file():
            if split_path.suffix.lower() == ".jsonl":
                return _labelme_iter_pairs_from_jsonl(split_path)
            if split_path.suffix.lower() in {".txt", ".list"}:
                return _labelme_iter_pairs_from_list(split_path)
            if split_path.suffix.lower() == _LABELME_SUFFIX:
                # Single JSON: treat like a 1-line list.
                return _labelme_iter_pairs_from_list(split_path)
            return _labelme_iter_pairs_from_list(split_path)
        return [], []

    train_images, train_json = _load_split(payload.get("train"))
    val_images, val_json = _load_split(payload.get("val"))

    return LabelMePoseDatasetSpec(
        root=root_path,
        train_images=train_images,
        val_images=val_images,
        train_json=train_json,
        val_json=val_json,
        kpt_count=kpt_count,
        kpt_dims=kpt_dims,
        keypoint_names=keypoint_names,
        flip_idx=flip_idx,
    )


def _infer_flip_idx_from_names(
    names: List[str], *, kpt_count: int
) -> Optional[List[int]]:
    # Backwards-compatible wrapper for older imports/tests.
    return infer_flip_idx_from_names(names, kpt_count=int(kpt_count))


def merge_feature_layers(feats: torch.Tensor, *, mode: str = "concat") -> torch.Tensor:
    """Merge multi-layer DINO features into a single CHW tensor.

    Supported modes:
      - concat: [L,D,H,W] -> [L*D,H,W]
      - mean:   [L,D,H,W] -> [D,H,W]
      - max:    [L,D,H,W] -> [D,H,W]
    """
    if feats.ndim == 3:
        return feats
    if feats.ndim != 4:
        raise ValueError("Expected DINO features as CHW or LDHW")
    mode_norm = str(mode or "concat").strip().lower()
    layer_count, dim, h, w = feats.shape
    if mode_norm == "concat":
        return feats.contiguous().view(int(layer_count * dim), int(h), int(w))
    if mode_norm == "mean":
        return feats.mean(dim=0)
    if mode_norm == "max":
        return feats.max(dim=0).values
    raise ValueError(
        f"Unsupported feature merge mode: {mode!r} (expected concat/mean/max)"
    )


@dataclass(frozen=True)
class DinoKPSEGAugmentConfig:
    """YOLO-style, lightweight pose augmentations.

    Note: This intentionally avoids mosaic/mixup to keep geometry and feature extraction simple.
    """

    enabled: bool = False
    hflip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.0
    scale: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    seed: Optional[int] = None


def _pil_color_jitter(
    pil: Image.Image,
    *,
    rng: np.random.Generator,
    brightness: float,
    contrast: float,
    saturation: float,
) -> Image.Image:
    if brightness <= 0 and contrast <= 0 and saturation <= 0:
        return pil

    from PIL import ImageEnhance

    out = pil
    if brightness > 0:
        out = ImageEnhance.Brightness(out).enhance(
            float(rng.uniform(1.0 - brightness, 1.0 + brightness))
        )
    if contrast > 0:
        out = ImageEnhance.Contrast(out).enhance(
            float(rng.uniform(1.0 - contrast, 1.0 + contrast))
        )
    if saturation > 0:
        out = ImageEnhance.Color(out).enhance(
            float(rng.uniform(1.0 - saturation, 1.0 + saturation))
        )
    return out


def _affine_matrix(
    *,
    width: int,
    height: int,
    rng: np.random.Generator,
    degrees: float,
    translate: float,
    scale: float,
) -> Optional[np.ndarray]:
    if degrees <= 0 and translate <= 0 and scale <= 0:
        return None

    angle = float(rng.uniform(-degrees, degrees)) if degrees > 0 else 0.0
    sc = float(rng.uniform(1.0 - scale, 1.0 + scale)) if scale > 0 else 1.0
    tx = float(rng.uniform(-translate, translate) * width) if translate > 0 else 0.0
    ty = float(rng.uniform(-translate, translate) * height) if translate > 0 else 0.0

    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    rad = np.deg2rad(angle)
    c = float(np.cos(rad))
    s = float(np.sin(rad))

    m = np.array(
        [
            [c * sc, -s * sc, 0.0],
            [s * sc, c * sc, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    t1 = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    t2 = np.array(
        [[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return (t2 @ m @ t1).astype(np.float32, copy=False)


def _apply_affine_to_keypoints(
    keypoints_instances: List[np.ndarray],
    *,
    width: int,
    height: int,
    m: np.ndarray,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for kpts in keypoints_instances:
        if kpts.ndim != 2 or kpts.shape[1] < 3:
            out.append(kpts)
            continue

        xy = kpts[:, :2].astype(np.float32, copy=True)
        v = kpts[:, 2:3].astype(np.float32, copy=True)

        xy[:, 0] *= float(width)
        xy[:, 1] *= float(height)

        ones = np.ones((xy.shape[0], 1), dtype=np.float32)
        pts = np.concatenate([xy, ones], axis=1)  # [K,3]
        pts_t = (pts @ m.T)[:, :2]

        x_norm = pts_t[:, 0] / max(1.0, float(width))
        y_norm = pts_t[:, 1] / max(1.0, float(height))

        inside = (x_norm >= 0.0) & (x_norm <= 1.0) & (y_norm >= 0.0) & (y_norm <= 1.0)
        v = np.where(inside[:, None], v, np.zeros_like(v))
        updated = np.concatenate([x_norm[:, None], y_norm[:, None], v], axis=1).astype(
            np.float32, copy=False
        )
        out.append(updated)
    return out


def _apply_hflip_to_keypoints(
    keypoints_instances: List[np.ndarray],
    *,
    flip_idx: Optional[List[int]],
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for kpts in keypoints_instances:
        if kpts.ndim != 2 or kpts.shape[1] < 2:
            out.append(kpts)
            continue
        updated = kpts.astype(np.float32, copy=True)
        updated[:, 0] = 1.0 - updated[:, 0]
        if flip_idx and len(flip_idx) == int(updated.shape[0]):
            updated = updated[np.asarray(flip_idx, dtype=np.int64)]
        out.append(updated)
    return out


def _apply_pose_augmentations(
    pil: Image.Image,
    keypoints_instances: List[np.ndarray],
    *,
    cfg: DinoKPSEGAugmentConfig,
    flip_idx: Optional[List[int]],
    rng: np.random.Generator,
) -> tuple[Image.Image, List[np.ndarray]]:
    if not cfg.enabled:
        return pil, keypoints_instances

    width, height = pil.size
    if width <= 1 or height <= 1:
        return pil, keypoints_instances

    pil = _pil_color_jitter(
        pil,
        rng=rng,
        brightness=float(cfg.brightness),
        contrast=float(cfg.contrast),
        saturation=float(cfg.saturation),
    )

    m = _affine_matrix(
        width=width,
        height=height,
        rng=rng,
        degrees=float(cfg.degrees),
        translate=float(cfg.translate),
        scale=float(cfg.scale),
    )
    if m is not None:
        try:
            coeffs = _invert_affine_3x3_to_pil_coeffs(m)
            pil = pil.transform(
                (width, height),
                Image.Transform.AFFINE,
                coeffs,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(114, 114, 114),
            )
        except Exception:
            m = None
        if m is not None:
            keypoints_instances = _apply_affine_to_keypoints(
                keypoints_instances,
                width=width,
                height=height,
                m=m,
            )

    if float(cfg.hflip_prob) > 0 and float(rng.random()) < float(cfg.hflip_prob):
        pil = ImageOps.mirror(pil)
        keypoints_instances = _apply_hflip_to_keypoints(
            keypoints_instances, flip_idx=flip_idx
        )

    return pil, keypoints_instances


def _invert_affine_3x3_to_pil_coeffs(
    m: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Invert a 3x3 affine matrix (with last row [0,0,1]) and return PIL coeffs.

    PIL expects (a, b, c, d, e, f) mapping output->input:
      x_in = a*x_out + b*y_out + c
      y_in = d*x_out + e*y_out + f
    """
    mat = np.asarray(m, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError("Expected 3x3 matrix")
    if not np.allclose(mat[2, :], [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("Expected affine matrix with last row [0,0,1]")

    a, b, c = float(mat[0, 0]), float(mat[0, 1]), float(mat[0, 2])
    d, e, f = float(mat[1, 0]), float(mat[1, 1]), float(mat[1, 2])
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError("Singular affine matrix")

    inv_a = e / det
    inv_b = -b / det
    inv_d = -d / det
    inv_e = a / det
    inv_c = (b * f - c * e) / det
    inv_f = (c * d - a * f) / det
    return (inv_a, inv_b, inv_c, inv_d, inv_e, inv_f)


class DinoKPSEGPoseDataset(torch.utils.data.Dataset):
    """Dataset yielding frozen DINO features + keypoint masks at patch resolution."""

    @dataclass(frozen=True)
    class _InstanceRecord:
        image_path: Path
        bbox: np.ndarray
        kpts: np.ndarray
        cache_salt: bytes

    def __init__(
        self,
        image_paths: List[Path],
        *,
        kpt_count: int,
        kpt_dims: int,
        radius_px: float,
        extractor: Dinov3FeatureExtractor,
        label_format: str = "yolo",
        label_paths: Optional[List[Optional[Path]]] = None,
        keypoint_names: Optional[List[str]] = None,
        flip_idx: Optional[List[int]] = None,
        augment: Optional[DinoKPSEGAugmentConfig] = None,
        cache_dir: Optional[Path] = None,
        mask_type: str = "gaussian",
        heatmap_sigma_px: Optional[float] = None,
        instance_mode: str = "union",
        bbox_scale: float = 1.25,
        cache_dtype: torch.dtype = torch.float16,
        return_images: bool = False,
        return_keypoints: bool = False,
        feature_merge: str = "concat",
    ) -> None:
        self.image_paths = list(image_paths)
        self.kpt_count = int(kpt_count)
        self.kpt_dims = int(kpt_dims)
        self.radius_px = float(radius_px)
        self.extractor = extractor
        self.label_format = str(label_format or "yolo").strip().lower()
        if self.label_format not in {"yolo", "labelme"}:
            raise ValueError(f"Unsupported label_format: {self.label_format!r}")
        self.label_paths = list(label_paths) if label_paths is not None else None
        self.keypoint_names = (
            list(keypoint_names) if keypoint_names is not None else None
        )
        if self.label_format == "labelme" and not self.keypoint_names:
            raise ValueError("LabelMe datasets require `keypoint_names`.")
        self.flip_idx = list(flip_idx) if flip_idx else None
        self.augment = augment or DinoKPSEGAugmentConfig(enabled=False)
        self.rng = np.random.default_rng(self.augment.seed)
        self.return_images = bool(return_images)
        self.return_keypoints = bool(return_keypoints)
        self.mask_type = str(mask_type or "gaussian").strip().lower()
        self.heatmap_sigma_px = (
            float(heatmap_sigma_px) if heatmap_sigma_px is not None else None
        )
        self.instance_mode = str(instance_mode or "auto").strip().lower()
        if self.instance_mode not in {"auto", "union", "per_instance"}:
            raise ValueError(f"Unsupported instance_mode: {self.instance_mode!r}")
        if self.instance_mode == "auto":
            self.instance_mode = self._infer_instance_mode()
        self.bbox_scale = float(bbox_scale) if bbox_scale is not None else 1.25
        self._instance_records: List[DinoKPSEGPoseDataset._InstanceRecord] = []

        # Feature caching is incompatible with random image augmentations.
        self.cache_dir = None if self.augment.enabled else cache_dir
        self.cache_dtype = cache_dtype
        self.feature_merge = str(feature_merge or "concat").strip().lower()
        if self.feature_merge not in {"concat", "mean", "max"}:
            raise ValueError(
                f"Unsupported feature_merge: {self.feature_merge!r} "
                "(expected concat/mean/max)"
            )

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.instance_mode == "per_instance":
            self._build_instance_records()

    def __len__(self) -> int:  # pragma: no cover - trivial
        if self.instance_mode == "per_instance":
            return len(self._instance_records)
        return len(self.image_paths)

    def _label_path_for(self, idx: int, image_path: Path) -> Optional[Path]:
        if self.label_format == "yolo":
            return _image_to_label_path(Path(image_path))
        if self.label_paths is not None and idx < len(self.label_paths):
            p = self.label_paths[idx]
            return Path(p) if p is not None else None
        candidate = _image_to_labelme_path(Path(image_path))
        return candidate if candidate.exists() else None

    def _load_pose_instances_for_mode_infer(self, label_path: Path) -> List[np.ndarray]:
        if self.label_format == "yolo":
            try:
                lines = Path(label_path).read_text(encoding="utf-8").splitlines()
            except Exception:
                return []
            out: List[np.ndarray] = []
            for line in lines:
                tokens = str(line).strip().split()
                if not tokens:
                    continue
                parsed = _parse_yolo_pose_instance(
                    tokens, kpt_count=self.kpt_count, dims=self.kpt_dims
                )
                if parsed is None:
                    continue
                _, kpts = parsed
                if kpts.ndim != 2 or kpts.shape[1] < 2:
                    continue
                if kpts.shape[1] >= 3 and not bool((kpts[:, 2] > 0).any()):
                    continue
                out.append(kpts.astype(np.float32, copy=False))
            return out

        from annolid.utils.annotation_store import load_labelme_json

        try:
            payload = load_labelme_json(label_path)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []

        try:
            h = int(payload.get("imageHeight") or 0)
            w = int(payload.get("imageWidth") or 0)
        except Exception:
            h, w = 0, 0
        image_hw = (h, w)
        if image_hw[0] <= 0 or image_hw[1] <= 0:
            from annolid.datasets.labelme_collection import resolve_image_path

            img = resolve_image_path(Path(label_path))
            if img and Path(img).exists():
                try:
                    with Image.open(img) as pil:
                        pil = ImageOps.exif_transpose(pil)
                        w2, h2 = pil.size
                    image_hw = (int(h2), int(w2))
                except Exception:
                    image_hw = (0, 0)

        return _labelme_keypoint_instances_from_payload(
            payload,
            keypoint_names=list(self.keypoint_names or []),
            kpt_dims=int(self.kpt_dims),
            image_hw=image_hw,
        )

    def _load_pose_instances_with_bbox(
        self, label_path: Path
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.label_format == "yolo":
            try:
                lines = Path(label_path).read_text(encoding="utf-8").splitlines()
            except Exception:
                return []
            out: List[Tuple[np.ndarray, np.ndarray]] = []
            for line in lines:
                tokens = str(line).strip().split()
                if not tokens:
                    continue
                parsed = _parse_yolo_pose_instance(
                    tokens, kpt_count=self.kpt_count, dims=self.kpt_dims
                )
                if parsed is None:
                    continue
                out.append(parsed)
            return out

        from annolid.utils.annotation_store import load_labelme_json

        try:
            payload = load_labelme_json(label_path)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []

        try:
            h = int(payload.get("imageHeight") or 0)
            w = int(payload.get("imageWidth") or 0)
        except Exception:
            h, w = 0, 0
        image_hw = (h, w)
        if image_hw[0] <= 0 or image_hw[1] <= 0:
            from annolid.datasets.labelme_collection import resolve_image_path

            img = resolve_image_path(Path(label_path))
            if img and Path(img).exists():
                try:
                    with Image.open(img) as pil:
                        pil = ImageOps.exif_transpose(pil)
                        w2, h2 = pil.size
                    image_hw = (int(h2), int(w2))
                except Exception:
                    image_hw = (0, 0)

        kpt_instances = _labelme_keypoint_instances_with_group_ids_from_payload(
            payload,
            keypoint_names=list(self.keypoint_names or []),
            kpt_dims=int(self.kpt_dims),
            image_hw=image_hw,
        )
        if not kpt_instances:
            return []

        shapes = payload.get("shapes")
        poly_by_gid: Dict[int, List[Tuple[float, float]]] = {}
        if isinstance(shapes, list):
            for shape in shapes:
                if not isinstance(shape, dict):
                    continue
                if shape.get("shape_type") != "polygon":
                    continue
                pts = shape.get("points")
                if not isinstance(pts, list) or len(pts) < 3:
                    continue
                flat: List[Tuple[float, float]] = []
                for pt in pts:
                    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                        continue
                    try:
                        flat.append((float(pt[0]), float(pt[1])))
                    except Exception:
                        continue
                if len(flat) < 3:
                    continue
                try:
                    gid_raw = shape.get("group_id")
                    gid = int(gid_raw) if gid_raw is not None else 0
                except Exception:
                    gid = 0
                poly_by_gid.setdefault(gid, []).extend(flat)

        height, width = int(image_hw[0]), int(image_hw[1])
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for gid, kpts in kpt_instances:
            pts = poly_by_gid.get(gid)
            xs: List[float] = []
            ys: List[float] = []
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
            else:
                visible = kpts[:, 2] > 0
                if bool(visible.any()):
                    xs = (kpts[visible, 0] * float(width)).tolist()
                    ys = (kpts[visible, 1] * float(height)).tolist()
            if not xs or not ys:
                continue
            x1, x2 = float(min(xs)), float(max(xs))
            y1, y2 = float(min(ys)), float(max(ys))
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0
            bbox = np.asarray(
                [
                    cx / max(1.0, float(width)),
                    cy / max(1.0, float(height)),
                    bw / max(1.0, float(width)),
                    bh / max(1.0, float(height)),
                ],
                dtype=np.float32,
            )
            out.append((bbox, kpts.astype(np.float32, copy=True)))
        return out

    def _infer_instance_mode(self) -> str:
        """Pick a default instance mode based on the label files.

        - Use `per_instance` when any label file contains >1 valid pose instances.
        - Otherwise default to `union`.
        """
        for idx, image_path in enumerate(self.image_paths):
            label_path = self._label_path_for(idx, Path(image_path))
            if label_path is None or not label_path.exists():
                continue
            if len(self._load_pose_instances_for_mode_infer(label_path)) > 1:
                return "per_instance"
        return "union"

    def _build_instance_records(self) -> None:
        records: List[DinoKPSEGPoseDataset._InstanceRecord] = []
        for idx, image_path in enumerate(self.image_paths):
            image_path = Path(image_path)
            label_path = self._label_path_for(idx, image_path)
            if label_path is None or not Path(label_path).exists():
                continue

            for inst_idx, (bbox, kpts) in enumerate(
                self._load_pose_instances_with_bbox(Path(label_path))
            ):
                if kpts.ndim != 2 or kpts.shape[1] < 2:
                    continue
                if kpts.shape[1] >= 3 and not bool((kpts[:, 2] > 0).any()):
                    continue
                bbox_f32 = bbox.astype(np.float32, copy=True)
                kpts_f32 = kpts.astype(np.float32, copy=True)
                cache_salt = b"|".join(
                    [
                        b"per_instance",
                        str(label_path).encode("utf-8", errors="ignore"),
                        str(int(inst_idx)).encode("utf-8"),
                        str(float(self.bbox_scale)).encode("utf-8"),
                        bbox_f32.tobytes(),
                    ]
                )
                records.append(
                    DinoKPSEGPoseDataset._InstanceRecord(
                        image_path=image_path,
                        bbox=bbox_f32,
                        kpts=kpts_f32,
                        cache_salt=cache_salt,
                    )
                )
        self._instance_records = records

    def _cache_path(
        self, image_path: Path, *, cache_salt: Optional[bytes] = None
    ) -> Path:
        assert self.cache_dir is not None
        # Cache key must include extractor config; otherwise switching DINO backbones
        # will reuse incompatible cached features (e.g., 384 vs 1024 channels).
        cfg = getattr(self.extractor, "cfg", None)
        layers = getattr(cfg, "layers", None)
        if layers is None:
            layers_token = ""
        else:
            try:
                layers_token = ",".join(
                    str(int(x))
                    # type: ignore[arg-type]
                    for x in layers
                )
            except Exception:
                layers_token = str(layers)

        model_id = str(getattr(self.extractor, "model_id", "") or "")
        short_side = str(getattr(cfg, "short_side", "") or "")
        patch_size = str(getattr(self.extractor, "patch_size", "") or "")
        payload = f"{model_id}|{short_side}|{patch_size}|{layers_token}|{self.feature_merge}|{image_path}".encode(
            "utf-8", errors="ignore"
        )
        if cache_salt:
            payload = payload + b"|" + bytes(cache_salt)
        digest = hashlib.sha1(payload).hexdigest()
        return self.cache_dir / f"{digest}.pt"

    def _load_or_compute_features(
        self,
        image_path: Path,
        pil: Image.Image,
        *,
        cache_salt: Optional[bytes] = None,
    ) -> torch.Tensor:
        if self.cache_dir is None:
            feats = self.extractor.extract(pil, return_type="torch")
            feats = merge_feature_layers(feats, mode=self.feature_merge)
            return feats.to(dtype=self.cache_dtype)

        cache_path = self._cache_path(image_path, cache_salt=cache_salt)
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu")
                if isinstance(payload, torch.Tensor):
                    cached = merge_feature_layers(payload, mode=self.feature_merge)
                    if self._is_compatible_cached_features(cached):
                        return cached
                if isinstance(payload, dict) and isinstance(
                    payload.get("feats"), torch.Tensor
                ):
                    cached = merge_feature_layers(
                        payload["feats"], mode=self.feature_merge
                    )
                    if self._is_compatible_cached_features(cached):
                        return cached
            except Exception:
                pass

        feats = self.extractor.extract(pil, return_type="torch")
        feats = merge_feature_layers(feats, mode=self.feature_merge).to(
            dtype=self.cache_dtype
        )
        try:
            torch.save({"feats": feats}, cache_path)
        except Exception:
            pass
        return feats

    def _is_compatible_cached_features(self, feats: torch.Tensor) -> bool:
        try:
            if feats.ndim == 4:
                feats = merge_feature_layers(feats, mode=self.feature_merge)
            if feats.ndim != 3:
                return False
            expected = getattr(getattr(self.extractor, "model", None), "config", None)
            expected_dim = int(getattr(expected, "hidden_size", 0) or 0)
            if expected_dim <= 0:
                return True
            if self.feature_merge == "concat":
                layers = getattr(getattr(self.extractor, "cfg", None), "layers", None)
                if layers is not None:
                    try:
                        layer_count = len(list(layers))
                    except Exception:
                        layer_count = 1
                else:
                    layer_count = 1
                return int(feats.shape[0]) == int(expected_dim) * int(layer_count)
            return int(feats.shape[0]) == int(expected_dim)
        except Exception:
            return False

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.instance_mode == "per_instance":
            record = self._instance_records[idx]
            image_path = record.image_path
            instance_bbox = record.bbox
            instance_kpts = record.kpts
            cache_salt = record.cache_salt
        else:
            image_path = self.image_paths[idx]
            instance_kpts = None
            cache_salt = None

        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil.convert("RGB"))
        width, height = pil.size

        keypoint_instances: List[np.ndarray] = []
        if self.instance_mode == "per_instance":
            keypoint_instances = [instance_kpts] if instance_kpts is not None else []
            if instance_bbox is not None and keypoint_instances:
                pil, instance_kpts = _crop_instance(
                    pil,
                    bbox=instance_bbox,
                    keypoints=keypoint_instances[0],
                    bbox_scale=self.bbox_scale,
                )
                keypoint_instances = [instance_kpts]
        else:
            label_path = self._label_path_for(idx, Path(image_path))
            if label_path is not None and Path(label_path).exists():
                if self.label_format == "yolo":
                    try:
                        lines = (
                            Path(label_path).read_text(encoding="utf-8").splitlines()
                        )
                    except Exception:
                        lines = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        tokens = line.split()
                        kpts = _parse_yolo_pose_line(
                            tokens, kpt_count=self.kpt_count, dims=self.kpt_dims
                        )
                        if kpts is not None:
                            keypoint_instances.append(kpts)
                else:
                    from annolid.utils.annotation_store import load_labelme_json

                    try:
                        payload = load_labelme_json(Path(label_path))
                    except Exception:
                        payload = None
                    if isinstance(payload, dict):
                        keypoint_instances.extend(
                            _labelme_keypoint_instances_from_payload(
                                payload,
                                keypoint_names=list(self.keypoint_names or []),
                                kpt_dims=int(self.kpt_dims),
                                image_hw=(int(height), int(width)),
                            )
                        )

        pil, keypoint_instances = _apply_pose_augmentations(
            pil,
            keypoint_instances,
            cfg=self.augment,
            flip_idx=self.flip_idx,
            rng=self.rng,
        )
        width, height = pil.size

        feats = self._load_or_compute_features(image_path, pil, cache_salt=cache_salt)
        _, h_p, w_p = feats.shape
        resized_h = int(h_p) * int(self.extractor.patch_size)
        resized_w = int(w_p) * int(self.extractor.patch_size)

        if not keypoint_instances:
            masks = torch.zeros((self.kpt_count, h_p, w_p), dtype=torch.float32)
            coords = torch.zeros((self.kpt_count, 2), dtype=torch.float32)
            coord_mask = torch.zeros((self.kpt_count,), dtype=torch.float32)
        else:
            masks, coords, coord_mask = _build_keypoint_targets(
                keypoint_instances,
                grid_hw=(h_p, w_p),
                patch_size=int(self.extractor.patch_size),
                resized_hw_px=(resized_h, resized_w),
                radius_px=self.radius_px,
                original_hw_px=(height, width),
                mask_type=self.mask_type,
                heatmap_sigma_px=self.heatmap_sigma_px,
            )

        sample = {
            "feats": feats.to(dtype=torch.float32),
            "masks": masks.to(dtype=torch.float32),
            "coords": coords.to(dtype=torch.float32),
            "coord_mask": coord_mask.to(dtype=torch.float32),
        }
        if self.return_keypoints:
            sample["gt_instances"] = [
                np.asarray(kpt, dtype=np.float32) for kpt in keypoint_instances
            ]
            sample["image_hw"] = (int(height), int(width))
        if self.return_images:
            try:
                pil_resized = pil.resize(
                    (int(resized_w), int(resized_h)), resample=Image.BILINEAR
                )
                # Use a writable array for safe torch conversion on all platforms.
                arr = np.array(pil_resized, dtype=np.uint8, copy=True)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    arr = arr[..., :3]
                img = (
                    torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32)
                    / 255.0
                )
                sample["image"] = img
            except Exception:
                pass
        return sample


def build_extractor(
    *,
    model_name: str,
    short_side: int,
    layers: Tuple[int, ...] = (-1,),
    device: Optional[str] = None,
) -> Dinov3FeatureExtractor:
    return_layer = "all" if len(layers) > 1 else "last"
    cfg = Dinov3Config(
        model_name=model_name,
        short_side=int(short_side),
        device=device,
        layers=layers,
        return_layer=return_layer,
    )
    return Dinov3FeatureExtractor(cfg)
