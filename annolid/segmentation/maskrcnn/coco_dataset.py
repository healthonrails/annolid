"""
COCO dataset utilities for torchvision-based Mask R-CNN and Keypoint R-CNN training.

Replaces detectron2's ``register_coco_instances`` / ``MetadataCatalog``
with pure torchvision + pycocotools equivalents.

Keypoint support (``AnnolidCocoKeypointDataset``) reads the ``keypoints``
field from COCO annotations and provides ``num_keypoints``, ``keypoint_names``
and ``skeleton`` metadata consumed by :mod:`annolid.segmentation.maskrcnn.keypoint_rcnn_train`.
"""

from __future__ import annotations

from pathlib import Path
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

from annolid.annotation.pose_schema import PoseSchema
from annolid.datasets.coco import (
    load_coco_category_id_map,
    load_coco_class_names,
    load_coco_keypoint_meta,
)


# ------------------------------------------------------------------
# Metadata helpers  (replace D2 MetadataCatalog / register_coco_instances)
# ------------------------------------------------------------------


def load_class_names(annotations_json: str | Path) -> List[str]:
    """Read category names from a COCO ``annotations.json`` file.

    Returns:
        Sorted list of category names (order matches category id).
    """
    return load_coco_class_names(annotations_json)


def load_keypoint_meta(
    annotations_json: str | Path,
) -> Dict[str, Any]:
    """Read keypoint metadata from a COCO annotations file.

    Returns a dict with keys:

    * ``num_keypoints`` (int): number of keypoints per instance.
    * ``keypoint_names`` (list[str]): ordered keypoint labels.
    * ``skeleton`` (list[list[int]]): 1-indexed bone pairs from COCO spec.

    Falls back to empty/zero values when no keypoint data is present.
    """
    return load_coco_keypoint_meta(annotations_json)


def load_category_id_map(annotations_json: str | Path) -> Dict[int, int]:
    """Build a mapping from COCO category_id → contiguous class index (0-based).

    Returns:
        Dict mapping original COCO category IDs to 0-based indices.
    """
    return load_coco_category_id_map(annotations_json)


# ------------------------------------------------------------------
# Dataset wrappers
# ------------------------------------------------------------------


def _get_transforms(train: bool = True) -> T.Compose:
    """Build torchvision v2 transforms for Mask R-CNN training/eval."""
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    return T.Compose(transforms)


class KeypointAugmentTransform:
    """Pose-aware data augmentation for ``[N, K, 3]`` Keypoint R-CNN targets.

    Applies:
    - Random horizontal flip with ``flip_idx`` left/right keypoint channel swapping,
    - Random affine rotation & scale with exact keypoint + bounding-box transformation,
    - Mild photometric brightness/contrast/saturation jitter.

    Preserves torchvision's ``[N, K, 3]`` float32 ``(x, y, visibility)`` keypoint
    tensor and ``XYXY`` ``tv_tensors.BoundingBoxes`` contract.
    """

    def __init__(
        self,
        *,
        flip_idx: Optional[Sequence[int]] = None,
        keypoint_names: Optional[Sequence[str]] = None,
        hflip_prob: float = 0.5,
        affine_prob: float = 0.5,
        rotation_deg: float = 15.0,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        color_jitter: bool = True,
    ) -> None:
        self.hflip_prob = max(0.0, min(1.0, float(hflip_prob)))
        self.affine_prob = max(0.0, min(1.0, float(affine_prob)))
        self.rotation_deg = max(0.0, float(rotation_deg))
        self.scale_range = (
            max(0.2, float(scale_range[0])),
            max(float(scale_range[0]), float(scale_range[1])),
        )
        self.color_jitter = bool(color_jitter)
        self.flip_idx: Optional[List[int]] = None

        if flip_idx is not None:
            self.flip_idx = [int(i) for i in flip_idx]
        elif keypoint_names:
            names = [str(n) for n in keypoint_names]
            schema = PoseSchema(
                keypoints=names,
                symmetry_pairs=PoseSchema.infer_symmetry_pairs(names),
            )
            computed = schema.compute_flip_idx(names)
            if computed is not None:
                self.flip_idx = list(computed)

        self._to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        self._jitter = (
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)
            if self.color_jitter
            else None
        )

    def _apply_hflip(
        self, img: torch.Tensor, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, h, w = img.shape
        img = F.horizontal_flip(img)
        target = dict(target)

        boxes = target.get("boxes")
        if boxes is not None and boxes.numel() > 0:
            raw = boxes.as_subclass(torch.Tensor).clone()
            x1 = raw[:, 0].clone()
            x2 = raw[:, 2].clone()
            raw[:, 0] = (float(w) - x2).clamp(0.0, float(w))
            raw[:, 2] = (float(w) - x1).clamp(0.0, float(w))
            target["boxes"] = tv_tensors.BoundingBoxes(
                raw,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(int(h), int(w)),
            )

        kps = target.get("keypoints")
        if kps is not None and kps.numel() > 0:
            kps = kps.clone()
            vis_mask = kps[..., 2] > 0
            kps[..., 0] = torch.where(vis_mask, float(w) - kps[..., 0], kps[..., 0])
            if self.flip_idx is not None and len(self.flip_idx) == int(kps.shape[1]):
                perm = torch.tensor(self.flip_idx, dtype=torch.long, device=kps.device)
                kps = kps[:, perm, :]
            target["keypoints"] = kps

        return img, target

    def _apply_affine(
        self,
        img: torch.Tensor,
        target: Dict[str, Any],
        *,
        angle_deg: float,
        scale: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, h, w = img.shape
        cx, cy = float(w) * 0.5, float(h) * 0.5
        img_out = F.affine(
            img,
            angle=float(angle_deg),
            translate=[0, 0],
            scale=float(scale),
            shear=[0.0, 0.0],
        )

        # Torchvision affine rotates clockwise for positive angle in image coords (y-down).
        rad = math.radians(float(angle_deg))
        cos_a = math.cos(rad) * float(scale)
        sin_a = math.sin(rad) * float(scale)

        def _transform_xy(
            x: torch.Tensor, y: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            dx = x - cx
            dy = y - cy
            nx = cx + cos_a * dx - sin_a * dy
            ny = cy + sin_a * dx + cos_a * dy
            return nx, ny

        target_out = dict(target)
        boxes = target.get("boxes")
        kps = target.get("keypoints")

        if boxes is not None and boxes.numel() > 0:
            raw = boxes.as_subclass(torch.Tensor)
            x1, y1, x2, y2 = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
            corners_x = torch.stack([x1, x2, x2, x1], dim=1)
            corners_y = torch.stack([y1, y1, y2, y2], dim=1)
            tx, ty = _transform_xy(corners_x, corners_y)
            nx1 = tx.min(dim=1).values.clamp(0.0, float(w))
            ny1 = ty.min(dim=1).values.clamp(0.0, float(h))
            nx2 = tx.max(dim=1).values.clamp(0.0, float(w))
            ny2 = ty.max(dim=1).values.clamp(0.0, float(h))
            valid_box = (nx2 - nx1 > 1.0) & (ny2 - ny1 > 1.0)
            if not bool(valid_box.all()):
                # Avoid dropping all boxes on extreme edge crops; fall back to untransformed.
                return img, target
            new_boxes = torch.stack([nx1, ny1, nx2, ny2], dim=1)
            target_out["boxes"] = tv_tensors.BoundingBoxes(
                new_boxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(int(h), int(w)),
            )
            target_out["area"] = (nx2 - nx1) * (ny2 - ny1)

        if kps is not None and kps.numel() > 0:
            kps_out = kps.clone()
            vis = kps_out[..., 2] > 0
            kx, ky = _transform_xy(kps_out[..., 0], kps_out[..., 1])
            in_bounds = (
                vis & (kx >= 0.0) & (kx <= float(w)) & (ky >= 0.0) & (ky <= float(h))
            )
            kps_out[..., 0] = torch.where(in_bounds, kx, torch.zeros_like(kx))
            kps_out[..., 1] = torch.where(in_bounds, ky, torch.zeros_like(ky))
            kps_out[..., 2] = torch.where(
                in_bounds, kps_out[..., 2], torch.zeros_like(kps_out[..., 2])
            )
            target_out["keypoints"] = kps_out

        return img_out, target_out

    def __call__(
        self, img: Any, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_t, target = self._to_tensor(img, target)
        if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
            img_t, target = self._apply_hflip(img_t, target)
        if (
            self.affine_prob > 0.0
            and (self.rotation_deg > 0.0 or self.scale_range != (1.0, 1.0))
            and random.random() < self.affine_prob
        ):
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            img_t, target = self._apply_affine(
                img_t, target, angle_deg=angle, scale=scale
            )
        if self._jitter is not None:
            img_t = self._jitter(img_t)
        return img_t, target


def _get_keypoint_transforms(
    train: bool = True,
    *,
    keypoint_names: Optional[Sequence[str]] = None,
    flip_idx: Optional[Sequence[int]] = None,
    augment: bool = True,
    hflip_prob: float = 0.5,
    rotation_deg: float = 15.0,
    scale_range: Tuple[float, float] = (0.85, 1.15),
    color_jitter: bool = True,
) -> Any:
    """Build transforms for Keypoint R-CNN datasets.

    When ``train=True`` and ``augment=True``, applies pose-aware horizontal flip
    (with ``flip_idx`` L/R keypoint channel swapping), mild affine rotation/scale,
    and color jitter while keeping keypoints as ``[N, K, 3]`` float32 tensors.
    """
    if train and augment:
        return KeypointAugmentTransform(
            flip_idx=flip_idx,
            keypoint_names=keypoint_names,
            hflip_prob=hflip_prob,
            rotation_deg=rotation_deg,
            scale_range=scale_range,
            color_jitter=color_jitter,
        )
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


class AnnolidCocoDataset(torch.utils.data.Dataset):
    """COCO instance segmentation dataset for torchvision Mask R-CNN.

    Each ``__getitem__`` returns ``(image_tensor, target_dict)`` in the
    format expected by ``torchvision.models.detection`` models.
    """

    def __init__(
        self,
        root: str | Path,
        annotations_json: str | Path,
        transforms: Optional[Any] = None,
    ) -> None:
        from pycocotools.coco import COCO

        self.root = Path(root)
        self.coco = COCO(str(annotations_json))
        self.ids = sorted(self.coco.getImgIds())
        self.cat_id_map = load_category_id_map(annotations_json)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        coco = self.coco
        img_id = self.ids[idx]
        img_info = coco.loadImgs(img_id)[0]
        img_path = self.root / img_info["file_name"]

        from PIL import Image

        img = Image.open(img_path).convert("RGB")

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_map[ann["category_id"]] + 1)  # +1: bg=0
            mask = coco.annToMask(ann)
            masks.append(mask)
            areas.append(ann.get("area", w * h))

        if len(boxes) == 0:
            # Empty image — still need valid tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img.height, img.width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(img.height, img.width),
            ),
            "labels": labels,
            "masks": tv_tensors.Mask(masks),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": areas,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# ------------------------------------------------------------------
# DataLoader builders
# ------------------------------------------------------------------


def _collate_fn(batch):
    """Custom collate that keeps each (image, target) separate."""
    return tuple(zip(*batch))


def build_train_loader(
    dataset_dir: str | Path,
    batch_size: int = 2,
    num_workers: int = 2,
    device: str = "cpu",
) -> DataLoader:
    """Build a training DataLoader for a COCO-format dataset.

    Expects ``{dataset_dir}/train/annotations.json`` and images in
    ``{dataset_dir}/train/``.
    """
    dataset_dir = Path(dataset_dir)
    ds = AnnolidCocoDataset(
        root=dataset_dir / "train",
        annotations_json=dataset_dir / "train" / "annotations.json",
        transforms=_get_transforms(train=True),
    )
    # pin_memory only helps for CUDA; on MPS and CPU it causes a warning and
    # provides no benefit.
    pin_memory = str(device).startswith("cuda")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin_memory,
    )


def build_val_loader(
    dataset_dir: str | Path,
    batch_size: int = 1,
    num_workers: int = 2,
) -> DataLoader:
    """Build a validation DataLoader for a COCO-format dataset."""
    dataset_dir = Path(dataset_dir)
    ds = AnnolidCocoDataset(
        root=dataset_dir / "valid",
        annotations_json=dataset_dir / "valid" / "annotations.json",
        transforms=_get_transforms(train=False),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


# ------------------------------------------------------------------
# Keypoint dataset
# ------------------------------------------------------------------


class AnnolidCocoKeypointDataset(torch.utils.data.Dataset):
    """COCO keypoint dataset for torchvision Keypoint R-CNN.

    Each ``__getitem__`` returns ``(image_tensor, target_dict)`` where
    ``target_dict`` contains:

    * ``boxes`` – BoundingBoxes tensor ``[N, 4]`` (XYXY).
    * ``labels`` – Int64 tensor ``[N]`` (1-indexed foreground class).
    * ``keypoints`` – Float32 tensor ``[N, K, 3]`` columns ``(x, y, visibility)``.
    * ``image_id``, ``area``, ``iscrowd`` – standard COCO fields.

    Torchvision Keypoint R-CNN expects *exactly* this format.
    """

    def __init__(
        self,
        root,
        annotations_json,
        transforms=None,
    ):
        from pycocotools.coco import COCO

        self.root = Path(root)
        self.coco = COCO(str(annotations_json))
        self.ids = sorted(self.coco.getImgIds())
        self.cat_id_map = load_category_id_map(annotations_json)
        self.transforms = transforms

        # Keypoint metadata
        meta = load_keypoint_meta(annotations_json)
        self.num_keypoints = meta["num_keypoints"]
        self.keypoint_names = meta["keypoint_names"]
        self.skeleton = meta["skeleton"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        img_info = coco.loadImgs(img_id)[0]
        img_path = self.root / img_info["file_name"]

        from PIL import Image

        img = Image.open(img_path).convert("RGB")

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels_list = []
        keypoints_list = []
        areas = []
        K = self.num_keypoints

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            raw_kps = ann.get("keypoints", [])
            if len(raw_kps) != K * 3:
                continue
            kps = [
                [
                    float(raw_kps[i * 3]),
                    float(raw_kps[i * 3 + 1]),
                    float(raw_kps[i * 3 + 2]),
                ]
                for i in range(K)
            ]
            boxes.append([x, y, x + w, y + h])
            labels_list.append(self.cat_id_map.get(ann["category_id"], 0) + 1)
            keypoints_list.append(kps)
            areas.append(ann.get("area", w * h))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            keypoints_t = torch.zeros((0, K, 3), dtype=torch.float32)
            areas_t = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels_list, dtype=torch.int64)
            keypoints_t = torch.as_tensor(keypoints_list, dtype=torch.float32)
            areas_t = torch.as_tensor(areas, dtype=torch.float32)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes_t,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(img.height, img.width),
            ),
            "labels": labels_t,
            # Keep [N, K, 3] tensor for Keypoint R-CNN training targets.
            # torchvision.tv_tensors.KeyPoints currently expects (..., 2) xy.
            "keypoints": keypoints_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": torch.zeros((len(labels_t),), dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def build_keypoint_train_loader(
    dataset_dir,
    batch_size=2,
    num_workers=0,
    device="cpu",
    augment=True,
    flip_idx=None,
):
    """Build a training DataLoader for a COCO keypoints dataset.

    Expects ``{dataset_dir}/train/annotations.json`` with
    ``categories[*].keypoints`` populated.
    """
    dataset_dir = Path(dataset_dir)
    ann_json = dataset_dir / "train" / "annotations.json"
    kp_meta = load_keypoint_meta(ann_json)
    ds = AnnolidCocoKeypointDataset(
        root=dataset_dir / "train",
        annotations_json=ann_json,
        transforms=_get_keypoint_transforms(
            train=True,
            keypoint_names=kp_meta.get("keypoint_names"),
            flip_idx=flip_idx,
            augment=bool(augment),
        ),
    )
    pin_memory = str(device).startswith("cuda")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin_memory,
    )


def build_keypoint_val_loader(
    dataset_dir,
    batch_size=1,
    num_workers=0,
):
    """Build a validation DataLoader for a COCO keypoints dataset."""
    dataset_dir = Path(dataset_dir)
    ds = AnnolidCocoKeypointDataset(
        root=dataset_dir / "valid",
        annotations_json=dataset_dir / "valid" / "annotations.json",
        transforms=_get_keypoint_transforms(train=False),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )
