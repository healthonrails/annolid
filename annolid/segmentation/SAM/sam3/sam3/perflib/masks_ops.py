# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import numpy as np
import torch
from pycocotools import mask as mask_util


def masks_to_boxes(masks: torch.Tensor, obj_ids: list[int]):
    with torch.autograd.profiler.record_function("perflib: masks_to_boxes"):
        # Sanity check based on callsite for replacement
        assert masks.shape[0] == len(obj_ids)
        assert masks.dim() == 3

        # Based on torchvision masks_to_boxes
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        N, H, W = masks.shape
        device = masks.device
        y = torch.arange(H, device=device).view(1, H)
        x = torch.arange(W, device=device).view(1, W)

        masks_with_obj = masks != 0  # N, H, W
        masks_with_obj_x = masks_with_obj.amax(
            dim=1
        )  # N, H (which columns have objects)
        masks_with_obj_y = masks_with_obj.amax(dim=2)  # N, W (which rows have objects)
        masks_without_obj_x = ~masks_with_obj_x
        masks_without_obj_y = ~masks_with_obj_y

        bounding_boxes_0 = torch.amin(
            (masks_without_obj_x * W) + (masks_with_obj_x * x), dim=1
        )
        bounding_boxes_1 = torch.amin(
            (masks_without_obj_y * H) + (masks_with_obj_y * y), dim=1
        )
        bounding_boxes_2 = torch.amax(masks_with_obj_x * x, dim=1)
        bounding_boxes_3 = torch.amax(masks_with_obj_y * y, dim=1)

        bounding_boxes = torch.stack(
            [bounding_boxes_0, bounding_boxes_1, bounding_boxes_2, bounding_boxes_3],
            dim=1,
        ).to(dtype=torch.float)
        assert bounding_boxes.shape == (N, 4)
        assert bounding_boxes.device == masks.device
        assert bounding_boxes.dtype == torch.float
        return bounding_boxes


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the IoU (Intersection over Union) between predicted masks and ground truth masks.
    Uses matmul-based vectorized intersection for Tensor Core acceleration.

    Args:
      - pred_masks: (N, H, W) bool Tensor, containing binary predicted segmentation masks
      - gt_masks: (M, H, W) bool Tensor, containing binary ground truth segmentation masks
    Returns:
      - ious: (N, M) float Tensor, containing IoUs for each pair of predicted and ground truth masks
    """
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    assert pred_masks.shape[1:] == gt_masks.shape[1:]

    # Matmul-based intersection (uses Tensor Cores via float mm)
    m1_flat = pred_masks.flatten(1).float()
    m2_flat = gt_masks.flatten(1).float()
    intersection = torch.mm(m1_flat, m2_flat.t())

    area1 = m1_flat.sum(dim=1)
    area2 = m2_flat.sum(dim=1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1)


def mask_iom(masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
    """
    Intersection-over-min-area for two mask sets.
    """
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool

    m1_flat = masks1.flatten(1).float()
    m2_flat = masks2.flatten(1).float()
    intersection = torch.mm(m1_flat, m2_flat.t())
    area1 = m1_flat.sum(dim=1)
    area2 = m2_flat.sum(dim=1)
    min_area = torch.min(area1[:, None], area2[None, :]).clamp(min=1e-8)
    return intersection / min_area


@torch.no_grad()
def rle_encode(orig_mask: torch.Tensor, return_areas: bool = False):
    """
    Encode a batch of boolean masks to COCO RLE dictionaries.
    """
    assert orig_mask.ndim == 3, "Mask must be of shape (N, H, W)"
    assert orig_mask.dtype == torch.bool, "Mask must have dtype=torch.bool"

    if orig_mask.numel() == 0:
        return []

    mask = orig_mask.transpose(1, 2)
    flat_mask = mask.reshape(mask.shape[0], -1)
    if return_areas:
        mask_areas = flat_mask.sum(-1).tolist()

    differences = torch.ones(
        mask.shape[0], flat_mask.shape[1] + 1, device=mask.device, dtype=torch.bool
    )
    differences[:, 1:-1] = flat_mask[:, :-1] != flat_mask[:, 1:]
    differences[:, 0] = flat_mask[:, 0]
    _, change_indices = torch.where(differences)

    try:
        boundaries = torch.cumsum(differences.sum(-1), 0).cpu()
    except RuntimeError:
        boundaries = torch.cumsum(differences.cpu().sum(-1), 0)

    change_indices_clone = change_indices.clone()
    for i in range(mask.shape[0]):
        beg = 0 if i == 0 else boundaries[i - 1].item()
        end = boundaries[i].item()
        change_indices[beg + 1 : end] -= change_indices_clone[beg : end - 1]

    change_indices = change_indices.tolist()
    batch_rles = []
    for i in range(mask.shape[0]):
        beg = 0 if i == 0 else boundaries[i - 1].item()
        end = boundaries[i].item()
        run_lengths = change_indices[beg:end]

        uncompressed_rle = {"counts": run_lengths, "size": list(orig_mask.shape[1:])}
        h, w = uncompressed_rle["size"]
        rle = mask_util.frPyObjects(uncompressed_rle, h, w)
        rle["counts"] = rle["counts"].decode("utf-8")
        if return_areas:
            rle["area"] = mask_areas[i]
        batch_rles.append(rle)

    return batch_rles


def robust_rle_encode(masks: torch.Tensor):
    """
    Encode masks to COCO RLE, falling back to CPU conversion on runtime errors.
    """
    assert masks.ndim == 3, "Mask must be of shape (N, H, W)"
    assert masks.dtype == torch.bool, "Mask must have dtype=torch.bool"

    try:
        return rle_encode(masks)
    except RuntimeError:
        masks_cpu = masks.cpu().numpy()
        rles = [
            mask_util.encode(
                np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F")
            )[0]
            for mask in masks_cpu
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")
        return rles
