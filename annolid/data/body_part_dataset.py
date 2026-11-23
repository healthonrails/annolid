"""Body-part segmentation dataset schema, builders, and format exporters.

Supports converting pixel-exact per-instance body-part ID maps (such as those
produced by Annolid's 3D two-mouse simulation or tracked body-part pipelines)
into three synchronized training formats:

1. **YOLO-Seg hierarchical polygons** (``images/``, ``labels/``, ``data.yaml``):
   parent animal body polygons (``resident``, ``intruder``) plus anatomical
   body-part polygons (``head``, ``snout``, ``ear``, ``torso``, ``tail``) so a
   single-stage instance segmentation model (e.g. ``YOLO11n-seg``) can localize
   both animal identities and body parts.
2. **COCO Instance Segmentation JSON** (``annotations/{train,val}.json``) with
   polygon segmentations, bounding boxes, and areas.
3. **Dense per-instance semantic part maps** (``semantic_part_masks/*.png``)
   encoding canonical anatomical part IDs (``0=bg, 1=head, 2=snout, 3=ear,
   4=torso, 5=tail``) alongside parent ``track_id`` masks for temporal VOS
   body-part segmentation models (e.g. Compressed Cutie-BodyPartSeg).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

# Raw 3D fine mesh IDs emitted by Annolid's two_mice.html segmentation pass.
SIM_FINE_ID_TO_NAME: Dict[int, str] = {
    1: "head",
    2: "snout",
    3: "left_ear",
    4: "right_ear",
    5: "neck",
    6: "torso",
    7: "left_forelimb",
    8: "right_forelimb",
    9: "left_hindlimb",
    10: "right_hindlimb",
    11: "tail",
}

# Canonical semantic body-part IDs for pose-oriented mouse body-part segmentation.
# Keeping ears chirality-agnostic (`ear`) avoids mirror confusion during pixel
# segmentation; left/right identity is resolved deterministically from spine heading.
SEMANTIC_PART_NAMES: Tuple[str, ...] = (
    "background",
    "head",
    "snout",
    "ear",
    "torso",
    "tail",
)
SEMANTIC_PART_NAME_TO_ID: Dict[str, int] = {
    name: idx for idx, name in enumerate(SEMANTIC_PART_NAMES)
}

# Hierarchical YOLO-Seg / COCO-Seg class order: whole-animal classes first,
# followed by anatomical body-part classes.
HIERARCHICAL_SEG_CLASSES: Tuple[str, ...] = (
    "resident",
    "intruder",
    "head",
    "snout",
    "ear",
    "torso",
    "tail",
)
HIERARCHICAL_CLASS_TO_ID: Dict[str, int] = {
    name: idx for idx, name in enumerate(HIERARCHICAL_SEG_CLASSES)
}


@dataclass(frozen=True)
class BodyPartPolygonRecord:
    """Single polygon instance in a body-part segmentation dataset."""

    class_id: int
    class_name: str
    points_xy: List[Tuple[float, float]]
    bbox_xywh: Tuple[float, float, float, float]
    area: float
    track_id: int
    is_parent_instance: bool = False


@dataclass
class BodyPartDatasetStats:
    """Summary statistics validating a built body-part segmentation dataset."""

    images_by_split: Dict[str, int] = field(default_factory=dict)
    polygons_by_class: Dict[str, int] = field(default_factory=dict)
    semantic_masks_by_split: Dict[str, int] = field(default_factory=dict)
    empty_label_files: int = 0


def expand_snout_muzzle_mask(
    snout_tip_mask: np.ndarray,
    head_mask: np.ndarray,
    torso_mask: Optional[np.ndarray] = None,
    *,
    radius_ratio: float = 0.42,
    min_radius_px: float = 6.0,
    max_radius_px: float = 14.0,
) -> np.ndarray:
    """Expand a tiny rhinarium tip mask into a segmentable anterior muzzle/snout cap.

    In top-down rodent imagery the anatomical rhinarium mesh can be only 2–4 px
    wide, too small for 512 px instance segmentation or 64x64 ROI semantic
    decoders. This helper defines the snout as the anterior muzzle cap of
    ``head_mask`` centered on the rhinarium tip (or, if the rhinarium is
    occluded, the anterior 22% of ``head_mask`` farthest from ``torso_mask``).
    """
    snout_b = np.asarray(snout_tip_mask, dtype=bool)
    head_b = np.asarray(head_mask, dtype=bool)
    head_or_snout = head_b | snout_b
    h_area = int(np.count_nonzero(head_or_snout))
    if h_area < 6:
        return snout_b

    # Compute effective head diameter to scale muzzle radius cleanly across resolutions
    head_diam = float(np.sqrt(4.0 * float(h_area) / np.pi))
    radius = float(np.clip(radius_ratio * head_diam, min_radius_px, max_radius_px))

    if np.any(snout_b):
        sy, sx = np.nonzero(snout_b)
        tip_cx, tip_cy = float(sx.mean()), float(sy.mean())
        hy, hx = np.nonzero(head_or_snout)
        d2 = (hx.astype(np.float32) - tip_cx) ** 2 + (
            hy.astype(np.float32) - tip_cy
        ) ** 2
        keep = d2 <= (radius * radius)
        out = np.zeros_like(snout_b, dtype=bool)
        out[hy[keep], hx[keep]] = True
        out |= snout_b
        return out

    # Fallback when rhinarium tip is occluded (e.g., head tilted down):
    # anterior 22% of head pixels along torso -> head direction
    if torso_mask is not None and int(np.count_nonzero(torso_mask)) >= 12:
        ty, tx = np.nonzero(torso_mask)
        hy, hx = np.nonzero(head_b)
        fwd = np.array(
            [float(hx.mean() - tx.mean()), float(hy.mean() - ty.mean())],
            dtype=np.float64,
        )
        nrm = float(np.linalg.norm(fwd))
        if nrm > 2.0:
            fwd /= nrm
            proj = hx.astype(np.float64) * fwd[0] + hy.astype(np.float64) * fwd[1]
            thresh = float(np.percentile(proj, 78.0))
            keep = proj >= thresh
            out = np.zeros_like(snout_b, dtype=bool)
            out[hy[keep], hx[keep]] = True
            return out

    return snout_b


def fine_id_map_to_semantic_parts(
    track_map: np.ndarray,
    fine_map: np.ndarray,
    *,
    target_track_id: Optional[int] = None,
    include_limbs_in_torso: bool = True,
    expand_snout_muzzle: bool = True,
) -> np.ndarray:
    """Map raw 3D fine mesh IDs into canonical 6-class semantic part IDs.

    Mapping:
    - ``fine_id == 1`` (head) -> ``1`` (head)
    - ``fine_id == 2`` (snout, optionally expanded to anterior muzzle cap) -> ``2`` (snout)
    - ``fine_id in {3, 4}`` (left_ear, right_ear) -> ``3`` (ear)
    - ``fine_id in {5, 6}`` (neck, torso) + optional limbs ``{7..10}`` -> ``4`` (torso)
    - ``fine_id == 11`` (tail) -> ``5`` (tail)
    """
    sem = np.zeros(fine_map.shape[:2], dtype=np.uint8)
    track_ids = (
        [int(target_track_id)]
        if target_track_id is not None
        else [int(t) for t in np.unique(track_map) if int(t) > 0]
    )
    for tid in track_ids:
        fg = track_map == tid
        if not np.any(fg):
            continue
        head_m = fg & (fine_map == 1)
        snout_m = fg & (fine_map == 2)
        ear_m = fg & ((fine_map == 3) | (fine_map == 4))
        torso_m = fg & ((fine_map == 5) | (fine_map == 6))
        if include_limbs_in_torso:
            torso_m |= fg & (fine_map >= 7) & (fine_map <= 10)
        tail_m = fg & (fine_map == 11)

        if expand_snout_muzzle:
            snout_m = expand_snout_muzzle_mask(snout_m, head_m, torso_m)

        sem[torso_m] = SEMANTIC_PART_NAME_TO_ID["torso"]
        sem[tail_m] = SEMANTIC_PART_NAME_TO_ID["tail"]
        sem[head_m] = SEMANTIC_PART_NAME_TO_ID["head"]
        sem[snout_m & (~ear_m)] = SEMANTIC_PART_NAME_TO_ID["snout"]
        sem[ear_m] = SEMANTIC_PART_NAME_TO_ID["ear"]
    return sem


def mask_to_clean_polygons(
    binary_mask: np.ndarray,
    *,
    min_area_px: float = 12.0,
    approx_epsilon: float = 1.0,
    max_contours: int = 2,
    dilate_px: int = 0,
) -> List[Tuple[List[Tuple[float, float]], Tuple[float, float, float, float], float]]:
    """Extract clean exterior polygons ``[(points_xy, bbox_xywh, area), ...]`` from a binary mask."""
    mask_u8 = (np.asarray(binary_mask, dtype=bool)).astype(np.uint8) * 255
    if dilate_px > 0 and np.any(mask_u8):
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        mask_u8 = cv2.dilate(mask_u8, k)
    if int(np.count_nonzero(mask_u8)) < max(4, int(min_area_px)):
        return []

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    out: List[
        Tuple[List[Tuple[float, float]], Tuple[float, float, float, float], float]
    ] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area_px:
            continue
        approx = cv2.approxPolyDP(cnt, float(approx_epsilon), True)
        pts_arr = approx if len(approx) >= 3 else cnt
        if len(pts_arr) < 3:
            continue
        pts_xy = [(float(pt[0][0]), float(pt[0][1])) for pt in pts_arr]
        xs = [p[0] for p in pts_xy]
        ys = [p[1] for p in pts_xy]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if (x1 - x0) < 1.0 or (y1 - y0) < 1.0:
            continue
        out.append((pts_xy, (x0, y0, x1 - x0, y1 - y0), area))
        if len(out) >= max_contours:
            break
    return out


def extract_hierarchical_body_part_polygons(
    track_map: np.ndarray,
    fine_map: np.ndarray,
    coarse_map: np.ndarray,
    *,
    track_id_to_parent_class: Optional[Mapping[int, str]] = None,
    include_proximal_tail_only_px: Optional[float] = None,
) -> List[BodyPartPolygonRecord]:
    """Convert `(track_map, fine_map, coarse_map)` into hierarchical YOLO/COCO polygons.

    Emits:
    - Parent animal body mask (`resident`/`intruder`, excluding thin tail so parent boxes
      stay tight around the animal body and match pose detection scale).
    - Per-animal anatomical body parts:
      - `head` (union of skull + snout + ears, giving a robust large head mask even
        when a cranial headstage partially covers the skull!)
      - `snout` (anterior muzzle/nose region, slightly dilated when tiny so it stays
        segmentable at 512px)
      - `ear` (separate left & right pinna polygons, each slightly stabilized)
      - `torso` (neck + torso + limbs trunk)
      - `tail` (tail mask, optionally clipped to proximal segment if requested)
    """
    parent_map = dict(track_id_to_parent_class or {1: "resident", 2: "intruder"})
    records: List[BodyPartPolygonRecord] = []

    unique_tracks = [int(t) for t in np.unique(track_map) if int(t) > 0]
    for tid in unique_tracks:
        parent_name = parent_map.get(tid)
        if parent_name not in HIERARCHICAL_CLASS_TO_ID:
            continue

        # 1. Parent whole-body silhouette (exclude tail coarse_id == 4)
        body_mask = (track_map == tid) & (coarse_map != 4)
        for pts_xy, bbox, area in mask_to_clean_polygons(
            body_mask, min_area_px=40.0, approx_epsilon=1.2, max_contours=1
        ):
            records.append(
                BodyPartPolygonRecord(
                    class_id=HIERARCHICAL_CLASS_TO_ID[parent_name],
                    class_name=parent_name,
                    points_xy=pts_xy,
                    bbox_xywh=bbox,
                    area=area,
                    track_id=tid,
                    is_parent_instance=True,
                )
            )

        fg = track_map == tid

        # 2. Head union (skull + snout + ears) — large robust head target
        head_union = fg & (
            (fine_map == 1) | (fine_map == 2) | (fine_map == 3) | (fine_map == 4)
        )
        for pts_xy, bbox, area in mask_to_clean_polygons(
            head_union, min_area_px=20.0, approx_epsilon=1.0, max_contours=1
        ):
            records.append(
                BodyPartPolygonRecord(
                    class_id=HIERARCHICAL_CLASS_TO_ID["head"],
                    class_name="head",
                    points_xy=pts_xy,
                    bbox_xywh=bbox,
                    area=area,
                    track_id=tid,
                )
            )

        # 3. Snout (anterior muzzle cap anchored on rhinarium) — large enough to segment reliably
        torso_mask_tmp = fg & (fine_map >= 5) & (fine_map <= 10)
        snout_mask = expand_snout_muzzle_mask(
            fg & (fine_map == 2),
            fg & (fine_map == 1),
            torso_mask_tmp,
        )
        for pts_xy, bbox, area in mask_to_clean_polygons(
            snout_mask,
            min_area_px=12.0,
            approx_epsilon=0.8,
            max_contours=1,
            dilate_px=1,
        ):
            records.append(
                BodyPartPolygonRecord(
                    class_id=HIERARCHICAL_CLASS_TO_ID["snout"],
                    class_name="snout",
                    points_xy=pts_xy,
                    bbox_xywh=bbox,
                    area=area,
                    track_id=tid,
                )
            )

        # 4. Left & right ears individually as chirality-agnostic `ear` polygons
        for ear_fine_id in (3, 4):
            ear_mask = fg & (fine_map == ear_fine_id)
            for pts_xy, bbox, area in mask_to_clean_polygons(
                ear_mask,
                min_area_px=8.0,
                approx_epsilon=0.8,
                max_contours=1,
                dilate_px=1,
            ):
                records.append(
                    BodyPartPolygonRecord(
                        class_id=HIERARCHICAL_CLASS_TO_ID["ear"],
                        class_name="ear",
                        points_xy=pts_xy,
                        bbox_xywh=bbox,
                        area=area,
                        track_id=tid,
                    )
                )

        # 5. Torso (neck + torso + limbs)
        torso_mask = fg & (fine_map >= 5) & (fine_map <= 10)
        for pts_xy, bbox, area in mask_to_clean_polygons(
            torso_mask, min_area_px=30.0, approx_epsilon=1.1, max_contours=1
        ):
            records.append(
                BodyPartPolygonRecord(
                    class_id=HIERARCHICAL_CLASS_TO_ID["torso"],
                    class_name="torso",
                    points_xy=pts_xy,
                    bbox_xywh=bbox,
                    area=area,
                    track_id=tid,
                )
            )

        # 6. Tail (dilated 1px because mouse tails are thin 3-5px structures)
        tail_mask = fg & (fine_map == 11)
        if (
            include_proximal_tail_only_px is not None
            and np.any(tail_mask)
            and np.any(torso_mask)
        ):
            # Keep tail pixels within `include_proximal_tail_only_px` of torso so tail boxes stay compact
            inv_torso = (~torso_mask).astype(np.uint8)
            dist = cv2.distanceTransform(inv_torso, cv2.DIST_L2, 3)
            tail_mask = tail_mask & (dist <= float(include_proximal_tail_only_px))

        for pts_xy, bbox, area in mask_to_clean_polygons(
            tail_mask, min_area_px=12.0, approx_epsilon=1.0, max_contours=1, dilate_px=1
        ):
            records.append(
                BodyPartPolygonRecord(
                    class_id=HIERARCHICAL_CLASS_TO_ID["tail"],
                    class_name="tail",
                    points_xy=pts_xy,
                    bbox_xywh=bbox,
                    area=area,
                    track_id=tid,
                )
            )

    return records


def format_yolo_seg_lines(
    records: Sequence[BodyPartPolygonRecord],
    *,
    image_width: int,
    image_height: int,
) -> List[str]:
    """Serialize polygon records into normalized Ultralytics YOLO-Seg label lines."""
    w = float(max(1, image_width))
    h = float(max(1, image_height))
    lines: List[str] = []
    for rec in records:
        if len(rec.points_xy) < 3:
            continue
        coords: List[str] = [str(int(rec.class_id))]
        for x, y in rec.points_xy:
            nx = max(0.0, min(1.0, float(x) / w))
            ny = max(0.0, min(1.0, float(y) / h))
            coords.append(f"{nx:.6f}")
            coords.append(f"{ny:.6f}")
        lines.append(" ".join(coords))
    return lines


def write_yolo_seg_data_yaml(
    dataset_root: str | Path,
    *,
    class_names: Sequence[str] = HIERARCHICAL_SEG_CLASSES,
) -> Path:
    """Write a clean Ultralytics YOLO-Seg ``data.yaml`` file."""
    root = Path(dataset_root).resolve()
    lines = [
        f"path: {root}",
        "train: images/train",
        "val: images/val",
        "names:",
        *[f"  {idx}: {name}" for idx, name in enumerate(class_names)],
        "",
    ]
    out_path = root / "data.yaml"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_coco_seg_split(
    out_json_path: str | Path,
    *,
    images: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    class_names: Sequence[str] = HIERARCHICAL_SEG_CLASSES,
) -> Path:
    """Write a standard COCO instance segmentation annotation file."""
    out_path = Path(out_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "info": {
            "description": "Annolid hierarchical animal body-part segmentation dataset",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [
            {
                "id": idx + 1,
                "name": name,
                "supercategory": "animal" if idx < 2 else "body_part",
            }
            for idx, name in enumerate(class_names)
        ],
        "images": list(images),
        "annotations": list(annotations),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def summarize_body_part_dataset(dataset_root: str | Path) -> BodyPartDatasetStats:
    """Audit a YOLO-Seg + semantic-mask body-part dataset directory."""
    root = Path(dataset_root)
    stats = BodyPartDatasetStats()
    for split in ("train", "val"):
        imgs = sorted((root / "images" / split).glob("*.jpg")) + sorted(
            (root / "images" / split).glob("*.png")
        )
        stats.images_by_split[split] = len(imgs)
        sem_dir = root / "semantic_masks" / split
        if sem_dir.is_dir():
            stats.semantic_masks_by_split[split] = len(list(sem_dir.glob("*.png")))
        lbl_dir = root / "labels" / split
        if not lbl_dir.is_dir():
            continue
        for lp in sorted(lbl_dir.glob("*.txt")):
            raw = lp.read_text(encoding="utf-8").strip()
            if not raw:
                stats.empty_label_files += 1
                continue
            for line in raw.splitlines():
                toks = line.split()
                if len(toks) < 7:
                    continue
                cid = int(toks[0])
                cname = (
                    HIERARCHICAL_SEG_CLASSES[cid]
                    if 0 <= cid < len(HIERARCHICAL_SEG_CLASSES)
                    else f"class_{cid}"
                )
                stats.polygons_by_class[cname] = (
                    stats.polygons_by_class.get(cname, 0) + 1
                )
    return stats
