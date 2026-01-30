from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import yaml
import torch

from annolid.segmentation.dino_kpseg.cli_utils import parse_layers
from annolid.segmentation.dino_kpseg.data import (
    _image_to_label_path,
    _image_to_labelme_path,
    _labelme_keypoint_instances_dict_from_payload,
    _parse_yolo_pose_line,
    DinoKPSEGPoseDataset,
    build_extractor,
    load_labelme_pose_spec,
    load_yolo_pose_spec,
)
from annolid.segmentation.dino_kpseg.keypoints import infer_left_right_pairs
from annolid.utils.annotation_store import load_labelme_json


@dataclass
class _KeypointAuditStats:
    present: int = 0
    visible: int = 0
    occluded: int = 0
    missing: int = 0
    out_of_bounds: int = 0


@dataclass
class _SplitAudit:
    images_total: int = 0
    label_files_found: int = 0
    images_with_instances: int = 0
    instances_total: int = 0
    invalid_lines: int = 0
    bbox_out_of_bounds: int = 0
    bbox_invalid: int = 0
    kpt_out_of_bounds: int = 0
    class_out_of_range: int = 0
    keypoint_stats: Dict[int, _KeypointAuditStats] = field(default_factory=dict)
    examples: Dict[str, List[str]] = field(default_factory=dict)
    crop_stats: Dict[str, float] = field(default_factory=dict)
    swap_ambiguity: Dict[str, object] = field(default_factory=dict)


def _issue_add(
    examples: Dict[str, List[str]], key: str, msg: str, *, limit: int = 8
) -> None:
    bucket = examples.setdefault(key, [])
    if len(bucket) < int(limit):
        bucket.append(str(msg))


def _safe_float(token: str) -> Optional[float]:
    try:
        val = float(token)
    except Exception:
        return None
    if math.isfinite(val):
        return float(val)
    return None


def _parse_yolo_pose_tokens(
    tokens: Sequence[str],
    *,
    kpt_count: int,
    dims: int,
) -> Tuple[
    Optional[int],
    Optional[Tuple[float, float, float, float]],
    Optional[np.ndarray],
    Optional[str],
]:
    """Parse a YOLO pose line into class id, bbox, and keypoints.

    Returns (cls, bbox_xywh, kpts, error). On error, returns None and message.
    """
    if not tokens:
        return None, None, None, "empty"
    start = 5
    required = start + int(kpt_count) * int(dims)
    if len(tokens) < required:
        return None, None, None, "too_few_tokens"
    cls_raw = tokens[0]
    try:
        cls = int(float(cls_raw))
    except Exception:
        return None, None, None, "invalid_class_id"
    nums: List[float] = []
    for tok in tokens[1:required]:
        val = _safe_float(tok)
        if val is None:
            return None, None, None, "invalid_number"
        nums.append(float(val))
    x, y, w, h = nums[:4]
    kpt_flat = nums[4:]
    if len(kpt_flat) != int(kpt_count) * int(dims):
        return None, None, None, "invalid_kpt_count"
    kpts = np.array(kpt_flat, dtype=np.float32).reshape(int(kpt_count), int(dims))
    return cls, (float(x), float(y), float(w), float(h)), kpts, None


def _suggest_fix(
    cls: int,
    bbox: Tuple[float, float, float, float],
    kpts: np.ndarray,
    *,
    dims: int,
) -> Optional[List[str]]:
    """Create a suggested YOLO pose line with clamped values."""
    if cls < 0:
        return None
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    w = min(1.0, max(1e-6, w))
    h = min(1.0, max(1e-6, h))
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return None
    fixed = kpts.copy()
    fixed[:, 0] = np.clip(fixed[:, 0], 0.0, 1.0)
    fixed[:, 1] = np.clip(fixed[:, 1], 0.0, 1.0)
    if dims >= 3:
        fixed[:, 2] = np.clip(fixed[:, 2], 0.0, 2.0)
    tokens = [str(int(cls)), f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for row in fixed:
        tokens.append(f"{row[0]:.6f}")
        tokens.append(f"{row[1]:.6f}")
        if dims >= 3:
            tokens.append(f"{row[2]:.0f}")
    return tokens


def _iter_splits(
    spec,
    *,
    split: str,
) -> List[Tuple[str, List[Path]]]:
    split_norm = str(split or "both").strip().lower()
    if split_norm == "train":
        return [("train", list(spec.train_images))]
    if split_norm == "val":
        return [("val", list(spec.val_images))]
    return [("train", list(spec.train_images)), ("val", list(spec.val_images))]


def audit_yolo_pose_dataset(
    data_yaml: Path,
    *,
    split: str = "both",
    max_images: Optional[int] = None,
    seed: int = 0,
    instance_mode: str = "auto",
    bbox_scale: float = 1.25,
    max_issue_examples: int = 8,
    max_fix_suggestions: int = 200,
) -> Dict[str, object]:
    """Audit a YOLO pose dataset and return a structured report."""
    data_yaml = Path(data_yaml)
    spec = load_yolo_pose_spec(Path(data_yaml))
    rng = random.Random(int(seed))
    try:
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        payload = {}
    nc = payload.get("nc")
    if nc is None and isinstance(payload.get("names"), (list, tuple)):
        nc = len(payload.get("names"))
    report: Dict[str, object] = {
        "data_yaml": str(Path(data_yaml).expanduser().resolve()),
        "kpt_count": int(spec.kpt_count),
        "kpt_dims": int(spec.kpt_dims),
        "keypoint_names": list(spec.keypoint_names or []),
        "flip_idx": list(spec.flip_idx or []),
        "splits": {},
        "fix_suggestions": [],
    }

    keypoint_names = list(spec.keypoint_names or [])
    lr_pairs = infer_left_right_pairs(keypoint_names, flip_idx=spec.flip_idx)

    for split_name, images in _iter_splits(spec, split=split):
        audit = _SplitAudit(
            keypoint_stats={
                idx: _KeypointAuditStats() for idx in range(int(spec.kpt_count))
            },
        )
        if not images:
            report["splits"][split_name] = audit.__dict__
            continue

        if max_images is not None:
            rng.shuffle(images)
            images = images[: max(0, int(max_images))]

        swap_examples: List[Dict[str, object]] = []
        swap_ambiguous = 0
        swap_total = 0

        outside_crop = 0
        visible_in_crop = 0

        for image_path in images:
            audit.images_total += 1
            label_path = _image_to_label_path(Path(image_path))
            if not label_path.exists():
                _issue_add(
                    audit.examples,
                    "missing_label",
                    f"{label_path}",
                    limit=max_issue_examples,
                )
                continue

            audit.label_files_found += 1
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                _issue_add(
                    audit.examples,
                    "unreadable_label",
                    f"{label_path}",
                    limit=max_issue_examples,
                )
                continue

            try:
                pil = Image.open(image_path)
                pil = ImageOps.exif_transpose(pil.convert("RGB"))
                width, height = pil.size
            except Exception:
                width, height = 0, 0

            valid_instances = 0
            for line_idx, line in enumerate(lines):
                tokens = str(line).strip().split()
                if not tokens:
                    continue
                cls, bbox, kpts, err = _parse_yolo_pose_tokens(
                    tokens, kpt_count=spec.kpt_count, dims=spec.kpt_dims
                )
                if err is not None or kpts is None or bbox is None or cls is None:
                    audit.invalid_lines += 1
                    _issue_add(
                        audit.examples,
                        "invalid_line",
                        f"{label_path}:{line_idx + 1} ({err})",
                        limit=max_issue_examples,
                    )
                    continue

                valid_instances += 1
                x, y, w, h = bbox
                line_bbox_oob = False
                line_kpt_oob = False
                if w <= 0 or h <= 0:
                    audit.bbox_invalid += 1
                    _issue_add(
                        audit.examples,
                        "invalid_bbox",
                        f"{label_path}:{line_idx + 1} (w={w:.3f}, h={h:.3f})",
                        limit=max_issue_examples,
                    )
                if not (
                    0.0 <= x <= 1.0
                    and 0.0 <= y <= 1.0
                    and 0.0 < w <= 1.0
                    and 0.0 < h <= 1.0
                ):
                    audit.bbox_out_of_bounds += 1
                    line_bbox_oob = True
                    _issue_add(
                        audit.examples,
                        "bbox_out_of_bounds",
                        f"{label_path}:{line_idx + 1}",
                        limit=max_issue_examples,
                    )

                if cls < 0:
                    audit.class_out_of_range += 1
                    _issue_add(
                        audit.examples,
                        "class_out_of_range",
                        f"{label_path}:{line_idx + 1} (cls={cls})",
                        limit=max_issue_examples,
                    )
                if nc is not None and cls >= int(nc):
                    audit.class_out_of_range += 1
                    _issue_add(
                        audit.examples,
                        "class_out_of_range",
                        f"{label_path}:{line_idx + 1} (cls={cls})",
                        limit=max_issue_examples,
                    )

                for k in range(int(spec.kpt_count)):
                    stats = audit.keypoint_stats[k]
                    if k >= int(kpts.shape[0]):
                        stats.missing += 1
                        continue
                    xk = float(kpts[k, 0])
                    yk = float(kpts[k, 1])
                    v = float(kpts[k, 2]) if int(spec.kpt_dims) >= 3 else 2.0
                    if v <= 0:
                        stats.missing += 1
                        continue
                    stats.present += 1
                    if v <= 1:
                        stats.occluded += 1
                    else:
                        stats.visible += 1
                    if not (0.0 <= xk <= 1.0 and 0.0 <= yk <= 1.0):
                        stats.out_of_bounds += 1
                        audit.kpt_out_of_bounds += 1
                        line_kpt_oob = True
                        _issue_add(
                            audit.examples,
                            "kpt_out_of_bounds",
                            f"{label_path}:{line_idx + 1} (kpt={k})",
                            limit=max_issue_examples,
                        )

                if lr_pairs and width > 0 and height > 0:
                    for li, ri in lr_pairs:
                        if li >= kpts.shape[0] or ri >= kpts.shape[0]:
                            continue
                        lv = float(kpts[li, 2]) if int(spec.kpt_dims) >= 3 else 2.0
                        rv = float(kpts[ri, 2]) if int(spec.kpt_dims) >= 3 else 2.0
                        if lv <= 0 or rv <= 0:
                            continue
                        lx = float(kpts[li, 0]) * float(width)
                        rx = float(kpts[ri, 0]) * float(width)
                        swap_total += 1
                        if abs(lx - rx) <= max(4.0, 0.05 * float(width)):
                            swap_ambiguous += 1
                            if len(swap_examples) < int(max_issue_examples):
                                swap_examples.append(
                                    {
                                        "image": str(image_path),
                                        "pair": [int(li), int(ri)],
                                        "left": [
                                            float(lx),
                                            float(kpts[li, 1]) * float(height),
                                        ],
                                        "right": [
                                            float(rx),
                                            float(kpts[ri, 1]) * float(height),
                                        ],
                                    }
                                )

                mode_norm = str(instance_mode or "auto").strip().lower()
                if mode_norm == "auto":
                    mode_norm = "per_instance" if valid_instances > 1 else "union"
                if mode_norm == "per_instance" and width > 0 and height > 0:
                    cx = float(x)
                    cy = float(y)
                    half_w = float(w) * float(bbox_scale) / 2.0
                    half_h = float(h) * float(bbox_scale) / 2.0
                    x1 = cx - half_w
                    x2 = cx + half_w
                    y1 = cy - half_h
                    y2 = cy + half_h
                    for k in range(int(spec.kpt_count)):
                        if k >= kpts.shape[0]:
                            continue
                        v = float(kpts[k, 2]) if int(spec.kpt_dims) >= 3 else 2.0
                        if v <= 0:
                            continue
                        visible_in_crop += 1
                        xk = float(kpts[k, 0])
                        yk = float(kpts[k, 1])
                        if xk < x1 or xk > x2 or yk < y1 or yk > y2:
                            outside_crop += 1

                # Fix suggestion for out-of-range values.
                if line_bbox_oob or line_kpt_oob:
                    suggestion = _suggest_fix(cls, bbox, kpts, dims=int(spec.kpt_dims))
                    if suggestion and len(report["fix_suggestions"]) < int(
                        max_fix_suggestions
                    ):
                        report["fix_suggestions"].append(
                            {
                                "label_path": str(label_path),
                                "line_index": int(line_idx + 1),
                                "issue": "out_of_bounds",
                                "suggested_line": " ".join(suggestion),
                            }
                        )

            if valid_instances > 0:
                audit.images_with_instances += 1
                audit.instances_total += int(valid_instances)
            else:
                _issue_add(
                    audit.examples,
                    "no_instances",
                    f"{label_path}",
                    limit=max_issue_examples,
                )

        if visible_in_crop > 0:
            audit.crop_stats = {
                "keypoints_visible": int(visible_in_crop),
                "keypoints_outside_crop": int(outside_crop),
                "outside_crop_frac": float(outside_crop) / float(visible_in_crop),
            }
        if swap_total > 0:
            audit.swap_ambiguity = {
                "pairs_total": int(swap_total),
                "ambiguous_pairs": int(swap_ambiguous),
                "ambiguous_frac": float(swap_ambiguous) / float(swap_total),
                "examples": swap_examples,
            }

        report["splits"][split_name] = {
            **audit.__dict__,
            "keypoint_stats": {
                str(k): stats.__dict__ for k, stats in audit.keypoint_stats.items()
            },
        }

    return report


def audit_labelme_pose_dataset(
    data_yaml: Path,
    *,
    split: str = "both",
    max_images: Optional[int] = None,
    seed: int = 0,
    max_issue_examples: int = 8,
) -> Dict[str, object]:
    """Audit a LabelMe pose dataset and return a structured report."""
    data_yaml = Path(data_yaml)
    spec = load_labelme_pose_spec(Path(data_yaml))
    rng = random.Random(int(seed))

    report: Dict[str, object] = {
        "data_yaml": str(Path(data_yaml).expanduser().resolve()),
        "format": "labelme",
        "kpt_count": int(spec.kpt_count),
        "kpt_dims": int(spec.kpt_dims),
        "keypoint_names": list(spec.keypoint_names or []),
        "flip_idx": list(spec.flip_idx or []),
        "splits": {},
        "fix_suggestions": [],
    }

    def _iter_labelme_splits() -> List[Tuple[str, List[Path], List[Optional[Path]]]]:
        split_norm = str(split or "both").strip().lower()
        if split_norm == "train":
            return [("train", list(spec.train_images), list(spec.train_json))]
        if split_norm == "val":
            return [("val", list(spec.val_images), list(spec.val_json))]
        return [
            ("train", list(spec.train_images), list(spec.train_json)),
            ("val", list(spec.val_images), list(spec.val_json)),
        ]

    for split_name, images, jsons in _iter_labelme_splits():
        audit = _SplitAudit(
            keypoint_stats={
                idx: _KeypointAuditStats() for idx in range(int(spec.kpt_count))
            },
        )
        if not images:
            report["splits"][split_name] = audit.__dict__
            continue

        if max_images is not None:
            combined = list(zip(images, jsons))
            rng.shuffle(combined)
            combined = combined[: max(0, int(max_images))]
            images = [x for x, _ in combined]
            jsons = [y for _, y in combined]

        for idx, image_path in enumerate(images):
            audit.images_total += 1
            json_path = None
            if idx < len(jsons):
                json_path = jsons[idx]
            if json_path is None:
                candidate = _image_to_labelme_path(Path(image_path))
                json_path = candidate if candidate.exists() else None
            if json_path is None or not Path(json_path).exists():
                _issue_add(
                    audit.examples,
                    "missing_label",
                    f"{image_path}",
                    limit=max_issue_examples,
                )
                continue
            audit.label_files_found += 1

            try:
                payload = load_labelme_json(Path(json_path))
            except Exception:
                _issue_add(
                    audit.examples,
                    "unreadable_label",
                    f"{json_path}",
                    limit=max_issue_examples,
                )
                continue
            if not isinstance(payload, dict):
                _issue_add(
                    audit.examples,
                    "invalid_label",
                    f"{json_path}",
                    limit=max_issue_examples,
                )
                continue

            try:
                h = int(payload.get("imageHeight") or 0)
                w = int(payload.get("imageWidth") or 0)
            except Exception:
                h, w = 0, 0
            if h <= 0 or w <= 0:
                try:
                    pil = Image.open(image_path)
                    pil = ImageOps.exif_transpose(pil.convert("RGB"))
                    w, h = pil.size
                except Exception:
                    h, w = 0, 0

            instances = _labelme_keypoint_instances_dict_from_payload(
                payload,
                keypoint_names=list(spec.keypoint_names),
                kpt_dims=int(spec.kpt_dims),
                image_hw=(int(h), int(w)),
            )
            if not instances:
                _issue_add(
                    audit.examples,
                    "no_instances",
                    f"{json_path}",
                    limit=max_issue_examples,
                )
                continue

            valid_instances = 0
            for _, kpts in sorted(instances.items(), key=lambda kv: kv[0]):
                if (
                    kpts.ndim != 2
                    or int(kpts.shape[0]) != int(spec.kpt_count)
                    or int(kpts.shape[1]) < 3
                ):
                    audit.invalid_lines += 1
                    continue
                if not bool((kpts[:, 2] > 0).any()):
                    continue
                valid_instances += 1
                for k in range(int(spec.kpt_count)):
                    stats = audit.keypoint_stats[int(k)]
                    v = float(kpts[k, 2])
                    stats.present += 1
                    if v <= 0:
                        stats.missing += 1
                    elif v >= 2:
                        stats.visible += 1
                    else:
                        stats.occluded += 1
            if valid_instances > 0:
                audit.images_with_instances += 1
                audit.instances_total += int(valid_instances)

        report["splits"][split_name] = {
            **audit.__dict__,
            "keypoint_stats": {
                str(k): stats.__dict__ for k, stats in audit.keypoint_stats.items()
            },
        }

    return report


def _draw_keypoint_histogram(
    *,
    keypoint_names: Sequence[str],
    stats: Dict[str, _KeypointAuditStats],
    width: int = 900,
    height: int = 300,
) -> Image.Image:
    width = max(320, int(width))
    height = max(200, int(height))
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n = max(1, len(stats))
    margin = 24
    bar_w = max(6, (width - 2 * margin) // n)
    max_h = height - 2 * margin
    for idx, (k, st) in enumerate(stats.items()):
        total = max(1, st.present + st.missing)
        visible_frac = st.visible / float(total)
        occluded_frac = st.occluded / float(total)
        missing_frac = st.missing / float(total)
        x0 = margin + idx * bar_w
        x1 = x0 + bar_w - 2
        h_vis = int(max_h * visible_frac)
        h_occ = int(max_h * occluded_frac)
        h_mis = int(max_h * missing_frac)
        y0 = height - margin
        if h_vis > 0:
            draw.rectangle([x0, y0 - h_vis, x1, y0], fill=(64, 170, 92))
        if h_occ > 0:
            draw.rectangle(
                [x0, y0 - h_vis - h_occ, x1, y0 - h_vis], fill=(230, 179, 62)
            )
        if h_mis > 0:
            draw.rectangle(
                [x0, y0 - h_vis - h_occ - h_mis, x1, y0 - h_vis - h_occ],
                fill=(210, 77, 87),
            )
        if idx < len(keypoint_names):
            label = str(keypoint_names[idx])
        else:
            label = str(k)
        draw.text((x0, height - margin + 2), label[:6], fill=(50, 50, 50))
    return img


def log_dataset_health(
    *,
    tb_writer,
    report: Dict[str, object],
    split_name: str,
    image_paths: Iterable[Path],
    kpt_count: int,
    kpt_dims: int,
    keypoint_names: Sequence[str],
    max_images: int = 12,
    seed: int = 0,
    tag_prefix: str = "dataset",
) -> None:
    rng = random.Random(int(seed))
    images = list(image_paths)
    if not images:
        return
    rng.shuffle(images)
    images = images[: max(1, int(max_images))]

    rendered: List[np.ndarray] = []
    for image_path in images:
        label_path = _image_to_label_path(Path(image_path))
        if not label_path.exists():
            continue
        try:
            pil = Image.open(image_path)
            pil = ImageOps.exif_transpose(pil.convert("RGB"))
        except Exception:
            continue
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        draw = ImageDraw.Draw(pil)
        for line in lines[:4]:
            tokens = str(line).strip().split()
            if not tokens:
                continue
            kpts = _parse_yolo_pose_line(
                tokens, kpt_count=int(kpt_count), dims=int(kpt_dims)
            )
            if kpts is None:
                continue
            for idx in range(min(int(kpt_count), int(kpts.shape[0]))):
                x = float(kpts[idx, 0]) * float(pil.size[0])
                y = float(kpts[idx, 1]) * float(pil.size[1])
                v = float(kpts[idx, 2]) if int(kpt_dims) >= 3 else 2.0
                if v <= 0:
                    continue
                radius = 3
                color = (64, 170, 92) if v > 1 else (230, 179, 62)
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    outline=color,
                    width=2,
                )
        rendered.append(np.array(pil, dtype=np.uint8, copy=True))

    if rendered:
        arr = np.stack(rendered, axis=0)  # NHWC
        arr = arr.astype(np.float32) / 255.0
        arr = np.transpose(arr, (0, 3, 1, 2))
        tb_writer.add_images(f"{tag_prefix}/{split_name}/samples", arr, global_step=0)

    split = report.get("splits", {}).get(split_name, {})
    stats_dict = split.get("keypoint_stats", {})
    stats: Dict[str, _KeypointAuditStats] = {}
    for k, v in stats_dict.items():
        stats[str(k)] = _KeypointAuditStats(**v)
    if stats:
        hist = _draw_keypoint_histogram(keypoint_names=keypoint_names, stats=stats)
        hist_arr = np.array(hist, dtype=np.uint8, copy=True)
        hist_arr = hist_arr.astype(np.float32) / 255.0
        hist_arr = np.transpose(hist_arr, (2, 0, 1))
        tb_writer.add_image(
            f"{tag_prefix}/{split_name}/keypoint_presence", hist_arr, global_step=0
        )

    tb_writer.add_text(
        f"{tag_prefix}/{split_name}/audit_summary",
        json.dumps(split, indent=2),
        0,
    )


def _group_key_from_path(
    path: Path,
    *,
    group_by: str,
    regex: Optional[str],
) -> str:
    group_by = str(group_by or "parent").strip().lower()
    if group_by == "parent":
        return path.parent.name or path.name
    if group_by == "grandparent":
        return path.parent.parent.name or path.parent.name or path.name
    if group_by == "stem_prefix":
        stem = path.stem
        for sep in ("_", "-", "."):
            if sep in stem:
                return stem.split(sep)[0]
        return stem
    if group_by == "regex" and regex:
        match = re.search(regex, path.as_posix())
        if match:
            if match.groups():
                return match.group(1)
            return match.group(0)
    return path.parent.name or path.name


def stratified_split(
    data_yaml: Path,
    *,
    output_dir: Path,
    val_size: float = 0.1,
    seed: int = 0,
    group_by: str = "parent",
    group_regex: Optional[str] = None,
    include_val: bool = True,
) -> Dict[str, object]:
    """Create a stratified split and write train/val list files."""
    spec = load_yolo_pose_spec(Path(data_yaml))
    all_images = list(spec.train_images)
    if include_val:
        all_images.extend(spec.val_images)
    if not all_images:
        raise ValueError("No images found to split.")

    groups: Dict[str, List[Path]] = {}
    for path in all_images:
        key = _group_key_from_path(Path(path), group_by=group_by, regex=group_regex)
        groups.setdefault(key, []).append(Path(path))

    rng = random.Random(int(seed))
    keys = list(groups.keys())
    rng.shuffle(keys)

    target_val = max(1, int(round(float(val_size) * len(all_images))))
    val: List[Path] = []
    train: List[Path] = []
    for key in keys:
        bucket = groups[key]
        if len(val) < target_val:
            val.extend(bucket)
        else:
            train.extend(bucket)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_list = output_dir / "train.txt"
    val_list = output_dir / "val.txt"
    train_list.write_text(
        "\n".join(str(p.resolve()) for p in sorted(set(train))),
        encoding="utf-8",
    )
    val_list.write_text(
        "\n".join(str(p.resolve()) for p in sorted(set(val))),
        encoding="utf-8",
    )

    payload = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    payload["train"] = str(train_list)
    payload["val"] = str(val_list)
    split_yaml = output_dir / "data_split.yaml"
    split_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    return {
        "train_list": str(train_list),
        "val_list": str(val_list),
        "split_yaml": str(split_yaml),
        "groups": len(groups),
        "train_images": len(train),
        "val_images": len(val),
        "val_size": float(val_size),
    }


def precompute_features(
    *,
    data_yaml: Path,
    data_format: str = "auto",
    model_name: str,
    short_side: int,
    layers: Sequence[int],
    device: Optional[str] = None,
    split: str = "both",
    instance_mode: str = "auto",
    bbox_scale: float = 1.25,
    cache_dir: Optional[Path] = None,
    cache_dtype: str = "float16",
) -> Dict[str, object]:
    """Precompute and cache DINOv3 features for a pose dataset (YOLO or LabelMe)."""
    data_yaml = Path(data_yaml)
    data_format_norm = str(data_format or "auto").strip().lower()
    if data_format_norm not in {"auto", "yolo", "labelme"}:
        raise ValueError(f"Unsupported data_format: {data_format_norm!r}")
    if data_format_norm == "auto":
        try:
            payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        except Exception:
            payload = {}
        fmt_token = ""
        if isinstance(payload, dict):
            fmt_token = (
                str(payload.get("format") or payload.get("type") or "").strip().lower()
            )
        data_format_norm = "labelme" if "labelme" in fmt_token else "yolo"

    if data_format_norm == "labelme":
        spec_lm = load_labelme_pose_spec(Path(data_yaml))
        train_images = list(spec_lm.train_images)
        val_images = list(spec_lm.val_images)
        train_labels = list(spec_lm.train_json)
        val_labels = list(spec_lm.val_json)
        kpt_count = int(spec_lm.kpt_count)
        kpt_dims = int(spec_lm.kpt_dims)
        keypoint_names = list(spec_lm.keypoint_names)
        flip_idx = spec_lm.flip_idx
    else:
        spec = load_yolo_pose_spec(Path(data_yaml))
        train_images = list(spec.train_images)
        val_images = list(spec.val_images)
        train_labels = None
        val_labels = None
        kpt_count = int(spec.kpt_count)
        kpt_dims = int(spec.kpt_dims)
        keypoint_names = list(spec.keypoint_names or [])
        flip_idx = spec.flip_idx

    extractor = build_extractor(
        model_name=str(model_name),
        short_side=int(short_side),
        layers=tuple(int(x) for x in layers),
        device=(str(device).strip() if device else None),
    )
    if cache_dir is None:
        cache_root = Path.home() / ".cache" / "annolid" / "dinokpseg" / "features"
        fingerprint = f"{model_name}|{short_side}|{tuple(int(x) for x in layers)}"
        digest = (
            __import__("hashlib").sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
        )
        cache_dir = cache_root / digest
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_dtype_norm = str(cache_dtype).strip().lower()
    torch_dtype = None
    if cache_dtype_norm == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    summary: Dict[str, object] = {
        "cache_dir": str(cache_dir),
        "model_name": str(model_name),
        "short_side": int(short_side),
        "layers": [int(x) for x in layers],
        "split": str(split),
        "data_format": str(data_format_norm),
        "counts": {},
    }

    split_norm = str(split or "both").strip().lower()
    split_items: List[Tuple[str, List[Path], Optional[List[Optional[Path]]]]] = []
    if split_norm == "train":
        split_items = [("train", train_images, train_labels)]
    elif split_norm == "val":
        split_items = [("val", val_images, val_labels)]
    else:
        split_items = [
            ("train", train_images, train_labels),
            ("val", val_images, val_labels),
        ]

    for split_name, images, label_paths in split_items:
        if not images:
            summary["counts"][split_name] = 0
            continue
        ds = DinoKPSEGPoseDataset(
            list(images),
            kpt_count=kpt_count,
            kpt_dims=kpt_dims,
            radius_px=6.0,
            extractor=extractor,
            label_format=str(data_format_norm),
            label_paths=label_paths,
            keypoint_names=keypoint_names,
            flip_idx=flip_idx,
            augment=None,
            cache_dir=cache_dir,
            mask_type="gaussian",
            heatmap_sigma_px=None,
            instance_mode=str(instance_mode),
            bbox_scale=float(bbox_scale),
            cache_dtype=torch_dtype,
            return_images=False,
        )
        for idx in range(len(ds)):
            _ = ds[int(idx)]
        summary["counts"][split_name] = int(len(ds))

    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DinoKPSEG dataset tools (audit + stratified split)."
    )
    sub = p.add_subparsers(dest="command", required=True)

    audit_p = sub.add_parser("audit", help="Audit a pose dataset (YOLO or LabelMe).")
    audit_p.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML (YOLO pose data.yaml or LabelMe spec.yaml)",
    )
    audit_p.add_argument(
        "--data-format",
        choices=("auto", "yolo", "labelme"),
        default="auto",
        help="Dataset annotation format (default: auto-detect from YAML).",
    )
    audit_p.add_argument("--split", choices=("train", "val", "both"), default="both")
    audit_p.add_argument("--max-images", type=int, default=None)
    audit_p.add_argument("--seed", type=int, default=0)
    audit_p.add_argument(
        "--instance-mode", choices=("auto", "union", "per_instance"), default="auto"
    )
    audit_p.add_argument("--bbox-scale", type=float, default=1.25)
    audit_p.add_argument("--out", default=None, help="Optional output JSON path")

    split_p = sub.add_parser("split", help="Create a stratified train/val split.")
    split_p.add_argument("--data", required=True, help="Path to YOLO pose data.yaml")
    split_p.add_argument(
        "--output", required=True, help="Output directory for split lists"
    )
    split_p.add_argument("--val-size", type=float, default=0.1)
    split_p.add_argument("--seed", type=int, default=0)
    split_p.add_argument(
        "--group-by",
        choices=("parent", "grandparent", "stem_prefix", "regex"),
        default="parent",
    )
    split_p.add_argument("--group-regex", default=None)
    split_p.add_argument(
        "--include-val",
        action="store_true",
        help="Include existing val images when splitting",
    )

    pre_p = sub.add_parser(
        "precompute",
        help="Precompute and cache DINOv3 features for a dataset.",
    )
    pre_p.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML (YOLO pose data.yaml or LabelMe spec.yaml)",
    )
    pre_p.add_argument(
        "--data-format",
        choices=("auto", "yolo", "labelme"),
        default="auto",
        help="Dataset annotation format (default: auto-detect from YAML).",
    )
    pre_p.add_argument(
        "--model-name", required=True, help="Hugging Face model id or dinov3 alias"
    )
    pre_p.add_argument("--short-side", type=int, default=768)
    pre_p.add_argument("--layers", type=str, default="-1")
    pre_p.add_argument("--device", default=None)
    pre_p.add_argument("--split", choices=("train", "val", "both"), default="both")
    pre_p.add_argument(
        "--instance-mode", choices=("auto", "union", "per_instance"), default="auto"
    )
    pre_p.add_argument("--bbox-scale", type=float, default=1.25)
    pre_p.add_argument("--cache-dir", default=None)
    pre_p.add_argument(
        "--cache-dtype", choices=("float16", "float32"), default="float16"
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.command == "audit":
        data_yaml = Path(args.data).expanduser().resolve()
        data_format_norm = (
            str(getattr(args, "data_format", "auto") or "auto").strip().lower()
        )
        if data_format_norm not in {"auto", "yolo", "labelme"}:
            raise ValueError(f"Unsupported data_format: {data_format_norm!r}")
        if data_format_norm == "auto":
            try:
                payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
            except Exception:
                payload = {}
            fmt_token = ""
            if isinstance(payload, dict):
                fmt_token = (
                    str(payload.get("format") or payload.get("type") or "")
                    .strip()
                    .lower()
                )
            data_format_norm = "labelme" if "labelme" in fmt_token else "yolo"

        if data_format_norm == "labelme":
            report = audit_labelme_pose_dataset(
                data_yaml,
                split=str(args.split),
                max_images=args.max_images,
                seed=int(args.seed),
            )
        else:
            report = audit_yolo_pose_dataset(
                data_yaml,
                split=str(args.split),
                max_images=args.max_images,
                seed=int(args.seed),
                instance_mode=str(args.instance_mode),
                bbox_scale=float(args.bbox_scale),
            )
        out = args.out
        if out:
            out_path = Path(out).expanduser().resolve()
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            print(json.dumps(report, indent=2))
        return 0
    if args.command == "split":
        summary = stratified_split(
            Path(args.data).expanduser().resolve(),
            output_dir=Path(args.output),
            val_size=float(args.val_size),
            seed=int(args.seed),
            group_by=str(args.group_by),
            group_regex=str(args.group_regex) if args.group_regex else None,
            include_val=bool(args.include_val),
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "precompute":
        layers = parse_layers(str(args.layers))
        summary = precompute_features(
            data_yaml=Path(args.data).expanduser().resolve(),
            data_format=str(args.data_format),
            model_name=str(args.model_name),
            short_side=int(args.short_side),
            layers=layers,
            device=(str(args.device).strip() if args.device else None),
            split=str(args.split),
            instance_mode=str(args.instance_mode),
            bbox_scale=float(args.bbox_scale),
            cache_dir=(
                Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
            ),
            cache_dtype=str(args.cache_dtype),
        )
        print(json.dumps(summary, indent=2))
        return 0
    raise SystemExit("Unknown command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
