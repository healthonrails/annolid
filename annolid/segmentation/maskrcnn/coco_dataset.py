"""
COCO dataset utilities for torchvision-based Mask R-CNN training.

Replaces detectron2's ``register_coco_instances`` / ``MetadataCatalog``
with pure torchvision + pycocotools equivalents.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metadata helpers  (replace D2 MetadataCatalog / register_coco_instances)
# ------------------------------------------------------------------


def load_class_names(annotations_json: str | Path) -> List[str]:
    """Read category names from a COCO ``annotations.json`` file.

    Returns:
        Sorted list of category names (order matches category id).
    """
    with open(annotations_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    return [c["name"] for c in cats]


def load_category_id_map(annotations_json: str | Path) -> Dict[int, int]:
    """Build a mapping from COCO category_id → contiguous class index (0-based).

    Returns:
        Dict mapping original COCO category IDs to 0-based indices.
    """
    with open(annotations_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    return {c["id"]: idx for idx, c in enumerate(cats)}


# ------------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------------


def _get_transforms(train: bool = True) -> T.Compose:
    """Build torchvision v2 transforms for Mask R-CNN training/eval."""
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
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
