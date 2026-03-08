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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

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


def _get_keypoint_transforms(train: bool = True) -> T.Compose:
    """Build transforms for Keypoint R-CNN datasets.

    Keep keypoints as a plain ``[N, K, 3]`` tensor (x, y, visibility) for model
    compatibility across torchvision versions. We intentionally avoid geometric
    augmentations here, because v2 keypoint tv_tensors currently model xy only.
    """
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    return T.Compose(transforms)


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
):
    """Build a training DataLoader for a COCO keypoints dataset.

    Expects ``{dataset_dir}/train/annotations.json`` with
    ``categories[*].keypoints`` populated.
    """
    dataset_dir = Path(dataset_dir)
    ds = AnnolidCocoKeypointDataset(
        root=dataset_dir / "train",
        annotations_json=dataset_dir / "train" / "annotations.json",
        transforms=_get_keypoint_transforms(train=True),
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
