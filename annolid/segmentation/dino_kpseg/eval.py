from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from annolid.segmentation.dino_kpseg.cli_utils import normalize_device
from annolid.segmentation.dino_kpseg.data import (
    _image_to_label_path,
    _parse_yolo_pose_line,
    load_yolo_pose_spec,
)
from annolid.segmentation.dino_kpseg.keypoints import (
    infer_left_right_pairs,
    symmetric_pairs_from_flip_idx,
)
from annolid.segmentation.dino_kpseg.model import checkpoint_unpack
from annolid.segmentation.dino_kpseg.data import (
    build_extractor,
    merge_feature_layers,
)

def _average_precision(
    scores: Sequence[float],
    is_true_positive: Sequence[bool],
    *,
    num_gt: int,
) -> float:
    if num_gt <= 0:
        return 0.0
    n = len(scores)
    if n == 0:
        return 0.0

    order = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)
    tp = [1.0 if bool(is_true_positive[i]) else 0.0 for i in order]
    fp = [0.0 if bool(is_true_positive[i]) else 1.0 for i in order]

    tp_cum: List[float] = []
    fp_cum: List[float] = []
    t, f = 0.0, 0.0
    for i in range(n):
        t += tp[i]
        f += fp[i]
        tp_cum.append(t)
        fp_cum.append(f)

    recalls = [tpi / float(num_gt) for tpi in tp_cum]
    precisions = [tpi / max(1e-12, (tpi + fpi))
                  for tpi, fpi in zip(tp_cum, fp_cum)]

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return float(ap)


@dataclass
class _KeypointStats:
    count: int = 0
    error_sum: float = 0.0
    pck_counts: Dict[float, int] = field(default_factory=dict)


class DinoKPSEGEvalAccumulator:
    def __init__(
        self,
        *,
        kpt_count: int,
        thresholds_px: Sequence[float],
        keypoint_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.kpt_count = int(kpt_count)
        thresholds = [float(t) for t in thresholds_px]
        self.thresholds_px = sorted(set(thresholds))
        if not self.thresholds_px:
            raise ValueError("thresholds_px must be non-empty")
        self.keypoint_names = list(keypoint_names) if keypoint_names else None

        self.images_total = 0
        self.images_used = 0
        self.instances_total = 0
        self.keypoints_visible_total = 0
        self.error_sum = 0.0
        self.pck_counts = {thr: 0 for thr in self.thresholds_px}

        self.swap_pairs_total = 0
        self.swap_pairs_swapped = 0

        self.keypoint_stats: Dict[int, _KeypointStats] = {
            idx: _KeypointStats(
                pck_counts={thr: 0 for thr in self.thresholds_px})
            for idx in range(self.kpt_count)
        }

    def update(
        self,
        *,
        pred_xy: Sequence[Tuple[float, float]],
        gt_instances: Sequence[np.ndarray],
        image_hw: Tuple[int, int],
        lr_pairs: Sequence[Tuple[int, int]],
    ) -> None:
        self.images_total += 1
        if not gt_instances:
            return

        self.images_used += 1
        self.instances_total += int(len(gt_instances))

        height, width = int(image_hw[0]), int(image_hw[1])
        if width <= 0 or height <= 0:
            return

        # Optional left/right swap rate: only meaningful for single-instance images.
        if len(gt_instances) == 1 and lr_pairs:
            gt = gt_instances[0]
            swaps, total = _count_swaps(
                pred_xy=pred_xy,
                gt_xy=gt,
                image_hw=(height, width),
                lr_pairs=lr_pairs,
            )
            self.swap_pairs_total += total
            self.swap_pairs_swapped += swaps

        for kpt_idx in range(self.kpt_count):
            if kpt_idx >= len(pred_xy):
                continue
            candidates = _collect_gt_candidates(
                gt_instances,
                kpt_idx=kpt_idx,
                image_hw=(height, width),
            )
            if not candidates:
                continue

            pred = pred_xy[kpt_idx]
            err = _min_error_px(pred, candidates)
            if err is None:
                continue

            self.keypoints_visible_total += 1
            self.error_sum += float(err)
            for thr in self.thresholds_px:
                if err <= thr:
                    self.pck_counts[thr] += 1

            kp_stats = self.keypoint_stats.get(kpt_idx)
            if kp_stats is None:
                continue
            kp_stats.count += 1
            kp_stats.error_sum += float(err)
            for thr in self.thresholds_px:
                if err <= thr:
                    kp_stats.pck_counts[thr] += 1

    def summary(self, *, include_per_keypoint: bool = False) -> Dict[str, object]:
        mean_error = None
        if self.keypoints_visible_total > 0:
            mean_error = self.error_sum / float(self.keypoints_visible_total)

        pck = {
            str(thr): (
                self.pck_counts[thr] / float(self.keypoints_visible_total)
                if self.keypoints_visible_total > 0
                else None
            )
            for thr in self.thresholds_px
        }

        swap_rate = None
        if self.swap_pairs_total > 0:
            swap_rate = self.swap_pairs_swapped / float(self.swap_pairs_total)

        payload: Dict[str, object] = {
            "images_total": self.images_total,
            "images_used": self.images_used,
            "instances_total": self.instances_total,
            "keypoints_visible_total": self.keypoints_visible_total,
            "mean_error_px": mean_error,
            "pck": pck,
            "swap_pairs_total": self.swap_pairs_total,
            "swap_pairs_swapped": self.swap_pairs_swapped,
            "swap_rate": swap_rate,
        }

        if include_per_keypoint:
            kp_payload: Dict[str, object] = {}
            for idx in range(self.kpt_count):
                stats = self.keypoint_stats.get(idx)
                if stats is None or stats.count <= 0:
                    continue
                name = (
                    self.keypoint_names[idx]
                    if self.keypoint_names and idx < len(self.keypoint_names)
                    else f"kp_{idx}"
                )
                kp_payload[name] = {
                    "count": stats.count,
                    "mean_error_px": stats.error_sum / float(stats.count),
                    "pck": {
                        str(thr): stats.pck_counts[thr] / float(stats.count)
                        for thr in self.thresholds_px
                    },
                }
            payload["per_keypoint"] = kp_payload
        return payload


def _collect_gt_candidates(
    gt_instances: Sequence[np.ndarray],
    *,
    kpt_idx: int,
    image_hw: Tuple[int, int],
) -> List[Tuple[float, float]]:
    height, width = int(image_hw[0]), int(image_hw[1])
    candidates: List[Tuple[float, float]] = []
    for kpts in gt_instances:
        if kpts.ndim != 2 or kpts.shape[1] < 2:
            continue
        if kpt_idx >= int(kpts.shape[0]):
            continue
        x_norm = float(kpts[kpt_idx, 0])
        y_norm = float(kpts[kpt_idx, 1])
        v = float(kpts[kpt_idx, 2]) if kpts.shape[1] >= 3 else 2.0
        if v <= 0:
            continue
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            continue
        candidates.append((x_norm * float(width), y_norm * float(height)))
    return candidates


def _min_error_px(
    pred_xy: Tuple[float, float],
    gt_candidates: Iterable[Tuple[float, float]],
) -> Optional[float]:
    px, py = float(pred_xy[0]), float(pred_xy[1])
    best = None
    for gx, gy in gt_candidates:
        dx = float(px) - float(gx)
        dy = float(py) - float(gy)
        dist = float((dx * dx + dy * dy) ** 0.5)
        if best is None or dist < best:
            best = dist
    return best


def _count_swaps(
    *,
    pred_xy: Sequence[Tuple[float, float]],
    gt_xy: np.ndarray,
    image_hw: Tuple[int, int],
    lr_pairs: Sequence[Tuple[int, int]],
) -> Tuple[int, int]:
    if gt_xy.ndim != 2 or gt_xy.shape[1] < 2:
        return (0, 0)
    height, width = int(image_hw[0]), int(image_hw[1])
    swaps = 0
    total = 0
    for li, ri in lr_pairs:
        li = int(li)
        ri = int(ri)
        if li >= len(pred_xy) or ri >= len(pred_xy):
            continue
        if li >= int(gt_xy.shape[0]) or ri >= int(gt_xy.shape[0]):
            continue

        lvis = float(gt_xy[li, 2]) if gt_xy.shape[1] >= 3 else 2.0
        rvis = float(gt_xy[ri, 2]) if gt_xy.shape[1] >= 3 else 2.0
        if lvis <= 0 or rvis <= 0:
            continue

        lx = float(gt_xy[li, 0]) * float(width)
        ly = float(gt_xy[li, 1]) * float(height)
        rx = float(gt_xy[ri, 0]) * float(width)
        ry = float(gt_xy[ri, 1]) * float(height)

        pred_l = pred_xy[li]
        pred_r = pred_xy[ri]

        d_same = _min_error_px(
            pred_l, [(lx, ly)]) + _min_error_px(pred_r, [(rx, ry)])
        d_swap = _min_error_px(
            pred_l, [(rx, ry)]) + _min_error_px(pred_r, [(lx, ly)])
        total += 1
        if d_swap + 1e-6 < d_same:
            swaps += 1
    return swaps, total


def _load_pose_instances(
    label_path: Path,
    *,
    kpt_count: int,
    kpt_dims: int,
) -> List[np.ndarray]:
    if not label_path.exists():
        return []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    instances: List[np.ndarray] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        kpts = _parse_yolo_pose_line(
            tokens, kpt_count=int(kpt_count), dims=int(kpt_dims))
        if kpts is not None:
            instances.append(kpts)
    return instances


def _predict_keypoints(
    probs: torch.Tensor,
    *,
    patch_size: int,
) -> List[Tuple[float, float]]:
    if probs.ndim != 3:
        raise ValueError("Expected probs in KHW format")
    k, h_p, w_p = int(probs.shape[0]), int(probs.shape[1]), int(probs.shape[2])
    norm = probs.sum(dim=(1, 2), keepdim=False).clamp(min=1e-6)
    xs = (torch.arange(w_p, device=probs.device,
          dtype=probs.dtype) + 0.5) * float(patch_size)
    ys = (torch.arange(h_p, device=probs.device,
          dtype=probs.dtype) + 0.5) * float(patch_size)
    x_exp = (probs.sum(dim=1) * xs[None, :]).sum(dim=1) / norm
    y_exp = (probs.sum(dim=2) * ys[None, :]).sum(dim=1) / norm
    return [(float(x), float(y)) for x, y in zip(x_exp.tolist(), y_exp.tolist())]


def _predict_keypoints_and_scores(
    probs: torch.Tensor,
    *,
    patch_size: int,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    coords = _predict_keypoints(probs, patch_size=patch_size)
    flat = probs.view(int(probs.shape[0]), -1)
    best_idx = torch.argmax(flat, dim=1)
    scores = torch.gather(flat, 1, best_idx[:, None]).squeeze(1).tolist()
    return coords, [float(s) for s in scores]


def _parse_thresholds(value: str) -> List[float]:
    raw = str(value or "").strip()
    if not raw:
        return [4.0, 8.0, 16.0]
    out: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out if out else [4.0, 8.0, 16.0]


def _parse_threshold_grid(value: str) -> List[float]:
    raw = str(value or "").strip()
    if not raw:
        return [round(x, 2) for x in np.linspace(0.05, 0.95, 19).tolist()]
    if ":" in raw:
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) >= 2:
            start = float(parts[0])
            stop = float(parts[1])
            step = float(parts[2]) if len(parts) >= 3 else 0.05
            if step <= 0:
                step = 0.05
            values: List[float] = []
            v = start
            while v <= stop + 1e-9:
                values.append(float(v))
                v += step
            return values
    out: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out if out else [round(x, 2) for x in np.linspace(0.05, 0.95, 19).tolist()]


def evaluate(
    *,
    data_yaml: Path,
    weights: Path,
    split: str = "val",
    thresholds_px: Sequence[float] = (4.0, 8.0, 16.0),
    device: Optional[str] = None,
    max_images: Optional[int] = None,
    include_per_keypoint: bool = False,
) -> Dict[str, object]:
    spec = load_yolo_pose_spec(data_yaml)
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    image_paths = spec.train_images if split == "train" else spec.val_images
    if not image_paths:
        raise ValueError(f"No images found for split '{split}'")

    payload = torch.load(weights, map_location="cpu")
    head, meta = checkpoint_unpack(payload)
    if int(spec.kpt_count) != int(meta.num_parts):
        raise ValueError(
            "Keypoint count mismatch between dataset and checkpoint: "
            f"dataset kpt_count={spec.kpt_count} vs checkpoint num_parts={meta.num_parts}."
        )
    device_str = normalize_device(device)
    device_t = torch.device(device_str)

    extractor = build_extractor(
        model_name=meta.model_name,
        short_side=meta.short_side,
        layers=meta.layers,
        device=device_str,
    )

    head = head.to(device_t).eval()

    flip_idx = meta.flip_idx or spec.flip_idx
    keypoint_names = meta.keypoint_names or spec.keypoint_names or []
    lr_pairs = (
        infer_left_right_pairs(keypoint_names, flip_idx=flip_idx)
        if (keypoint_names and flip_idx)
        else []
    )
    if not lr_pairs and flip_idx:
        lr_pairs = symmetric_pairs_from_flip_idx(flip_idx)

    acc = DinoKPSEGEvalAccumulator(
        kpt_count=int(meta.num_parts),
        thresholds_px=thresholds_px,
        keypoint_names=meta.keypoint_names or spec.keypoint_names,
    )

    limit = int(max_images) if max_images is not None else None
    for idx, image_path in enumerate(image_paths):
        if limit is not None and idx >= limit:
            break
        try:
            pil = Image.open(image_path)
        except Exception:
            continue
        pil = ImageOps.exif_transpose(pil.convert("RGB"))
        width, height = pil.size

        label_path = _image_to_label_path(Path(image_path))
        gt_instances = _load_pose_instances(
            label_path, kpt_count=spec.kpt_count, kpt_dims=spec.kpt_dims)
        if not gt_instances:
            acc.update(
                pred_xy=[],
                gt_instances=[],
                image_hw=(height, width),
                lr_pairs=lr_pairs,
            )
            continue

        feats = extractor.extract(pil, return_type="torch")
        feats = merge_feature_layers(feats)
        if feats.ndim != 3:
            continue
        if int(feats.shape[0]) != int(head.in_dim):
            raise RuntimeError(
                "DinoKPSEG checkpoint/backbone mismatch in eval: "
                f"checkpoint expects {head.in_dim} channels but extractor produced {int(feats.shape[0])}. "
                "Retrain with matching backbone/layers or clear the DinoKPSEG cache."
            )
        patch_size = int(extractor.patch_size)
        h_p, w_p = int(feats.shape[1]), int(feats.shape[2])
        resized_h, resized_w = h_p * patch_size, w_p * patch_size

        with torch.no_grad():
            logits = head(feats.unsqueeze(0).to(
                device_t, dtype=torch.float32))[0]
        probs = torch.sigmoid(logits).to("cpu")
        pred_resized = _predict_keypoints(probs, patch_size=patch_size)

        pred_xy = []
        for x_res, y_res in pred_resized:
            x_orig = x_res * (float(width) / float(resized_w))
            y_orig = y_res * (float(height) / float(resized_h))
            pred_xy.append((float(x_orig), float(y_orig)))

        acc.update(
            pred_xy=pred_xy,
            gt_instances=gt_instances,
            image_hw=(height, width),
            lr_pairs=lr_pairs,
        )

    return acc.summary(include_per_keypoint=include_per_keypoint)


def calibrate_thresholds(
    *,
    data_yaml: Path,
    weights: Path,
    split: str = "val",
    metric: str = "pck",
    pck_px: float = 8.0,
    threshold_grid: Sequence[float] = (),
    device: Optional[str] = None,
    max_images: Optional[int] = None,
    per_keypoint: bool = False,
) -> Dict[str, object]:
    spec = load_yolo_pose_spec(data_yaml)
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    image_paths = spec.train_images if split == "train" else spec.val_images
    if not image_paths:
        raise ValueError(f"No images found for split '{split}'")

    payload = torch.load(weights, map_location="cpu")
    head, meta = checkpoint_unpack(payload)
    if int(spec.kpt_count) != int(meta.num_parts):
        raise ValueError(
            "Keypoint count mismatch between dataset and checkpoint: "
            f"dataset kpt_count={spec.kpt_count} vs checkpoint num_parts={meta.num_parts}."
        )
    device_str = normalize_device(device)
    device_t = torch.device(device_str)

    extractor = build_extractor(
        model_name=meta.model_name,
        short_side=meta.short_side,
        layers=meta.layers,
        device=device_str,
    )
    head = head.to(device_t).eval()

    keypoint_names = meta.keypoint_names or spec.keypoint_names or []
    records: Dict[int, List[Tuple[float, float]]] = {
        idx: [] for idx in range(int(meta.num_parts))
    }

    limit = int(max_images) if max_images is not None else None
    for idx, image_path in enumerate(image_paths):
        if limit is not None and idx >= limit:
            break
        try:
            pil = Image.open(image_path)
        except Exception:
            continue
        pil = ImageOps.exif_transpose(pil.convert("RGB"))
        width, height = pil.size

        label_path = _image_to_label_path(Path(image_path))
        gt_instances = _load_pose_instances(
            label_path, kpt_count=spec.kpt_count, kpt_dims=spec.kpt_dims)
        if not gt_instances:
            continue

        feats = extractor.extract(pil, return_type="torch")
        feats = merge_feature_layers(feats)
        if feats.ndim != 3:
            continue
        if int(feats.shape[0]) != int(head.in_dim):
            raise RuntimeError(
                "DinoKPSEG checkpoint/backbone mismatch in calibration: "
                f"checkpoint expects {head.in_dim} channels but extractor produced {int(feats.shape[0])}. "
                "Retrain with matching backbone/layers or clear the DinoKPSEG cache."
            )
        patch_size = int(extractor.patch_size)
        h_p, w_p = int(feats.shape[1]), int(feats.shape[2])
        resized_h, resized_w = h_p * patch_size, w_p * patch_size

        with torch.no_grad():
            logits = head(feats.unsqueeze(0).to(
                device_t, dtype=torch.float32))[0]
        probs = torch.sigmoid(logits).to("cpu")
        pred_resized, scores = _predict_keypoints_and_scores(
            probs, patch_size=patch_size)

        pred_xy = []
        for x_res, y_res in pred_resized:
            x_orig = x_res * (float(width) / float(resized_w))
            y_orig = y_res * (float(height) / float(resized_h))
            pred_xy.append((float(x_orig), float(y_orig)))

        for kpt_idx in range(int(meta.num_parts)):
            if kpt_idx >= len(pred_xy):
                continue
            candidates = _collect_gt_candidates(
                gt_instances, kpt_idx=kpt_idx, image_hw=(height, width)
            )
            if not candidates:
                continue
            err = _min_error_px(pred_xy[kpt_idx], candidates)
            if err is None:
                continue
            score = float(scores[kpt_idx]) if kpt_idx < len(scores) else 0.0
            records[kpt_idx].append((score, float(err)))

    grid = [float(v) for v in threshold_grid] if threshold_grid else [
        round(x, 2) for x in np.linspace(0.05, 0.95, 19).tolist()
    ]
    metric_name = str(metric or "pck").strip().lower()
    if metric_name not in ("pck", "ap"):
        metric_name = "pck"

    def _score_threshold(pairs: Sequence[Tuple[float, float]], thr: float) -> float:
        if not pairs:
            return 0.0
        if metric_name == "pck":
            hits = sum(
                1 for score, err in pairs if score >= thr and err <= pck_px
            )
            return float(hits) / float(len(pairs))
        scores = [float(score) for score, _ in pairs if score >= thr]
        oks = [float(err) <= pck_px for score, err in pairs if score >= thr]
        return _average_precision(scores, oks, num_gt=len(pairs))

    all_pairs: List[Tuple[float, float]] = []
    for pairs in records.values():
        all_pairs.extend(pairs)

    best_global = None
    best_global_metric = -1.0
    for thr in grid:
        value = _score_threshold(all_pairs, float(thr))
        if value > best_global_metric:
            best_global_metric = value
            best_global = float(thr)

    payload: Dict[str, object] = {
        "metric": metric_name,
        "pck_px": float(pck_px),
        "threshold_grid": [float(v) for v in grid],
        "best_threshold": best_global,
        "best_metric": float(best_global_metric),
    }

    if per_keypoint:
        per_payload: Dict[str, object] = {}
        for idx, pairs in records.items():
            best_thr = None
            best_val = -1.0
            for thr in grid:
                value = _score_threshold(pairs, float(thr))
                if value > best_val:
                    best_val = value
                    best_thr = float(thr)
            name = (
                keypoint_names[idx]
                if keypoint_names and idx < len(keypoint_names)
                else f"kp_{idx}"
            )
            per_payload[name] = {
                "best_threshold": best_thr,
                "best_metric": float(best_val),
                "count": len(pairs),
            }
        payload["per_keypoint"] = per_payload

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate DinoKPSEG checkpoints on a YOLO pose dataset.")
    parser.add_argument("--data", required=True,
                        help="Path to YOLO pose data.yaml")
    parser.add_argument("--weights", required=True,
                        help="Path to DinoKPSEG checkpoint (.pt)")
    parser.add_argument("--split", default="val",
                        choices=("train", "val"))
    parser.add_argument("--thresholds", default="4,8,16",
                        help="Comma-separated PCK thresholds in pixels")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional cap on number of images to evaluate")
    parser.add_argument("--per-keypoint", action="store_true",
                        help="Include per-keypoint metrics in the output")
    parser.add_argument("--auto-threshold", action="store_true",
                        help="Calibrate confidence thresholds on the split")
    parser.add_argument(
        "--auto-threshold-metric",
        default="pck",
        choices=("pck", "ap"),
        help="Metric to optimize when calibrating thresholds",
    )
    parser.add_argument(
        "--auto-threshold-pck-px",
        type=float,
        default=8.0,
        help="Pixel radius for PCK/OKS positives during calibration",
    )
    parser.add_argument(
        "--auto-threshold-grid",
        default="0.05:0.95:0.05",
        help="Threshold grid (comma list or start:stop:step)",
    )
    parser.add_argument(
        "--auto-threshold-per-keypoint",
        action="store_true",
        help="Include per-keypoint threshold calibration",
    )
    parser.add_argument("--out", default=None,
                        help="Optional JSON output path (default: stdout)")
    args = parser.parse_args(argv)

    thresholds = _parse_thresholds(args.thresholds)
    data_yaml = Path(args.data).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()

    if bool(args.auto_threshold):
        grid = _parse_threshold_grid(args.auto_threshold_grid)
        output = calibrate_thresholds(
            data_yaml=data_yaml,
            weights=weights,
            split=str(args.split),
            metric=str(args.auto_threshold_metric),
            pck_px=float(args.auto_threshold_pck_px),
            threshold_grid=grid,
            device=(str(args.device).strip() if args.device else None),
            max_images=(int(args.max_images) if args.max_images else None),
            per_keypoint=bool(args.auto_threshold_per_keypoint),
        )
    else:
        summary = evaluate(
            data_yaml=data_yaml,
            weights=weights,
            split=str(args.split),
            thresholds_px=thresholds,
            device=(str(args.device).strip() if args.device else None),
            max_images=(int(args.max_images) if args.max_images else None),
            include_per_keypoint=bool(args.per_keypoint),
        )
        output = summary

    text = json.dumps(output, indent=2)
    out_path = args.out
    if out_path:
        Path(out_path).expanduser().resolve(
        ).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
