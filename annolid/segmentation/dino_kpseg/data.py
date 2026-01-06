from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

from annolid.features import Dinov3Config, Dinov3FeatureExtractor
from annolid.segmentation.dino_kpseg.keypoints import infer_flip_idx_from_names


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _resolve_yaml_path(value: str, *, yaml_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (yaml_path.parent / path).resolve()


def _yolo_list_images(images_root: Path) -> List[Path]:
    if not images_root.exists():
        return []
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
                tokens, kpt_count=int(kpt_count), dims=int(kpt_dims))
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


def _parse_yolo_pose_line(tokens: List[str], *, kpt_count: int, dims: int) -> Optional[np.ndarray]:
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
            [float(tokens[1]), float(tokens[2]),
             float(tokens[3]), float(tokens[4])],
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
    inside = (x_norm >= 0.0) & (x_norm <= 1.0) & (
        y_norm >= 0.0) & (y_norm <= 1.0)

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
                blob = np.exp(-dist2 / float(denom)
                              ).astype(np.float32, copy=False)
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
        split_path = _resolve_yaml_path(str(split_value), yaml_path=data_yaml)
        if not split_path.is_absolute():
            split_path = (root_path / split_path).resolve()
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
        inferred = _infer_flip_idx_from_names(
            keypoint_names, kpt_count=kpt_count)
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


def _infer_flip_idx_from_names(names: List[str], *, kpt_count: int) -> Optional[List[int]]:
    # Backwards-compatible wrapper for older imports/tests.
    return infer_flip_idx_from_names(names, kpt_count=int(kpt_count))


def merge_feature_layers(feats: torch.Tensor) -> torch.Tensor:
    """Merge multi-layer DINO features into a single CHW tensor.

    When multiple transformer layers are requested, the extractor returns
    [L, D, H, W]. We flatten layers into the channel dimension -> [L*D, H, W].
    """
    if feats.ndim == 3:
        return feats
    if feats.ndim != 4:
        raise ValueError("Expected DINO features as CHW or LDHW")
    l, d, h, w = feats.shape
    return feats.contiguous().view(int(l * d), int(h), int(w))


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
            float(rng.uniform(1.0 - brightness, 1.0 + brightness)))
    if contrast > 0:
        out = ImageEnhance.Contrast(out).enhance(
            float(rng.uniform(1.0 - contrast, 1.0 + contrast)))
    if saturation > 0:
        out = ImageEnhance.Color(out).enhance(
            float(rng.uniform(1.0 - saturation, 1.0 + saturation)))
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
    tx = float(rng.uniform(-translate, translate)
               * width) if translate > 0 else 0.0
    ty = float(rng.uniform(-translate, translate)
               * height) if translate > 0 else 0.0

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
    t1 = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    t2 = np.array([[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
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

        inside = (x_norm >= 0.0) & (x_norm <= 1.0) & (
            y_norm >= 0.0) & (y_norm <= 1.0)
        v = np.where(inside[:, None], v, np.zeros_like(v))
        updated = np.concatenate([x_norm[:, None], y_norm[:, None], v], axis=1).astype(
            np.float32, copy=False)
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
            keypoints_instances, flip_idx=flip_idx)

    return pil, keypoints_instances


def _invert_affine_3x3_to_pil_coeffs(m: np.ndarray) -> tuple[float, float, float, float, float, float]:
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

    def __init__(
        self,
        image_paths: List[Path],
        *,
        kpt_count: int,
        kpt_dims: int,
        radius_px: float,
        extractor: Dinov3FeatureExtractor,
        flip_idx: Optional[List[int]] = None,
        augment: Optional[DinoKPSEGAugmentConfig] = None,
        cache_dir: Optional[Path] = None,
        mask_type: str = "gaussian",
        heatmap_sigma_px: Optional[float] = None,
        instance_mode: str = "union",
        bbox_scale: float = 1.25,
        cache_dtype: torch.dtype = torch.float16,
        return_images: bool = False,
    ) -> None:
        self.image_paths = list(image_paths)
        self.kpt_count = int(kpt_count)
        self.kpt_dims = int(kpt_dims)
        self.radius_px = float(radius_px)
        self.extractor = extractor
        self.flip_idx = list(flip_idx) if flip_idx else None
        self.augment = augment or DinoKPSEGAugmentConfig(enabled=False)
        self.rng = np.random.default_rng(self.augment.seed)
        self.return_images = bool(return_images)
        self.mask_type = str(mask_type or "gaussian").strip().lower()
        self.heatmap_sigma_px = (
            float(heatmap_sigma_px) if heatmap_sigma_px is not None else None
        )
        self.instance_mode = str(instance_mode or "union").strip().lower()
        if self.instance_mode not in {"union", "per_instance"}:
            raise ValueError(
                f"Unsupported instance_mode: {self.instance_mode!r}"
            )
        self.bbox_scale = float(bbox_scale) if bbox_scale is not None else 1.25
        self._instance_records: List[Tuple[Path, np.ndarray, np.ndarray]] = []

        # Feature caching is incompatible with random image augmentations.
        self.cache_dir = None if self.augment.enabled else cache_dir
        self.cache_dtype = cache_dtype

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.instance_mode == "per_instance":
            self._build_instance_records()
            # Avoid cache collisions when using per-instance crops.
            self.cache_dir = None

    def __len__(self) -> int:  # pragma: no cover - trivial
        if self.instance_mode == "per_instance":
            return len(self._instance_records)
        return len(self.image_paths)

    def _build_instance_records(self) -> None:
        records: List[Tuple[Path, np.ndarray, np.ndarray]] = []
        for image_path in self.image_paths:
            label_path = _image_to_label_path(Path(image_path))
            if not label_path.exists():
                continue
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line in lines:
                tokens = line.strip().split()
                if not tokens:
                    continue
                parsed = _parse_yolo_pose_instance(
                    tokens, kpt_count=self.kpt_count, dims=self.kpt_dims
                )
                if parsed is None:
                    continue
                bbox, kpts = parsed
                if kpts.ndim != 2 or kpts.shape[1] < 3:
                    continue
                if not bool((kpts[:, 2] > 0).any()):
                    continue
                records.append(
                    (
                        Path(image_path),
                        bbox.astype(np.float32, copy=True),
                        kpts.astype(np.float32, copy=True),
                    )
                )
        self._instance_records = records

    def _cache_path(self, image_path: Path) -> Path:
        assert self.cache_dir is not None
        # Cache key must include extractor config; otherwise switching DINO backbones
        # will reuse incompatible cached features (e.g., 384 vs 1024 channels).
        cfg = getattr(self.extractor, "cfg", None)
        layers = getattr(cfg, "layers", None)
        if layers is None:
            layers_token = ""
        else:
            try:
                layers_token = ",".join(str(int(x))
                                        for x in layers)  # type: ignore[arg-type]
            except Exception:
                layers_token = str(layers)

        model_id = str(getattr(self.extractor, "model_id", "") or "")
        short_side = str(getattr(cfg, "short_side", "") or "")
        patch_size = str(getattr(self.extractor, "patch_size", "") or "")
        payload = f"{model_id}|{short_side}|{patch_size}|{layers_token}|{image_path}".encode(
            "utf-8", errors="ignore"
        )
        digest = hashlib.sha1(payload).hexdigest()
        return self.cache_dir / f"{digest}.pt"

    def _load_or_compute_features(self, image_path: Path, pil: Image.Image) -> torch.Tensor:
        if self.cache_dir is None:
            feats = self.extractor.extract(pil, return_type="torch")
            feats = merge_feature_layers(feats)
            return feats.to(dtype=self.cache_dtype)

        cache_path = self._cache_path(image_path)
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu")
                if isinstance(payload, torch.Tensor):
                    cached = merge_feature_layers(payload)
                    if self._is_compatible_cached_features(cached):
                        return cached
                if isinstance(payload, dict) and isinstance(payload.get("feats"), torch.Tensor):
                    cached = merge_feature_layers(payload["feats"])
                    if self._is_compatible_cached_features(cached):
                        return cached
            except Exception:
                pass

        feats = self.extractor.extract(
            pil, return_type="torch")
        feats = merge_feature_layers(feats).to(dtype=self.cache_dtype)
        try:
            torch.save({"feats": feats}, cache_path)
        except Exception:
            pass
        return feats

    def _is_compatible_cached_features(self, feats: torch.Tensor) -> bool:
        try:
            if feats.ndim == 4:
                feats = merge_feature_layers(feats)
            if feats.ndim != 3:
                return False
            expected = getattr(
                getattr(self.extractor, "model", None), "config", None)
            expected_dim = int(getattr(expected, "hidden_size", 0) or 0)
            if expected_dim <= 0:
                return True
            layers = getattr(
                getattr(self.extractor, "cfg", None), "layers", None)
            if layers is not None:
                try:
                    layer_count = len(list(layers))
                except Exception:
                    layer_count = 1
            else:
                layer_count = 1
            return int(feats.shape[0]) == int(expected_dim) * int(layer_count)
        except Exception:
            return False

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.instance_mode == "per_instance":
            image_path, instance_bbox, instance_kpts = self._instance_records[idx]
        else:
            image_path = self.image_paths[idx]
            instance_kpts = None

        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil.convert("RGB"))
        width, height = pil.size

        keypoint_instances: List[np.ndarray] = []
        if self.instance_mode == "per_instance":
            keypoint_instances = [
                instance_kpts] if instance_kpts is not None else []
            if instance_bbox is not None and keypoint_instances:
                pil, instance_kpts = _crop_instance(
                    pil,
                    bbox=instance_bbox,
                    keypoints=keypoint_instances[0],
                    bbox_scale=self.bbox_scale,
                )
                keypoint_instances = [instance_kpts]
        else:
            label_path = _image_to_label_path(image_path)
            if label_path.exists():
                try:
                    lines = label_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    kpts = _parse_yolo_pose_line(
                        tokens, kpt_count=self.kpt_count, dims=self.kpt_dims)
                    if kpts is not None:
                        keypoint_instances.append(kpts)

        pil, keypoint_instances = _apply_pose_augmentations(
            pil,
            keypoint_instances,
            cfg=self.augment,
            flip_idx=self.flip_idx,
            rng=self.rng,
        )
        width, height = pil.size

        feats = self._load_or_compute_features(image_path, pil)
        _, h_p, w_p = feats.shape
        resized_h = int(h_p) * int(self.extractor.patch_size)
        resized_w = int(w_p) * int(self.extractor.patch_size)

        if not keypoint_instances:
            masks = torch.zeros((self.kpt_count, h_p, w_p),
                                dtype=torch.float32)
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
        if self.return_images:
            try:
                pil_resized = pil.resize(
                    (int(resized_w), int(resized_h)), resample=Image.BILINEAR)
                # Use a writable array for safe torch conversion on all platforms.
                arr = np.array(pil_resized, dtype=np.uint8, copy=True)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    arr = arr[..., :3]
                img = torch.from_numpy(arr).permute(
                    2, 0, 1).to(dtype=torch.float32) / 255.0
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
