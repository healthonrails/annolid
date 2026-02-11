"""Convert LabelMe JSON annotations to COCO datasets."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import os
from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

from annolid.datasets.labelme_collection import resolve_image_path
from annolid.utils.annotation_store import load_labelme_json

logger = logging.getLogger(__name__)

try:
    import pycocotools.mask as coco_mask
except Exception:  # pragma: no cover - optional at import time
    coco_mask = None

try:
    from annolid.utils import annotation_compat as labelme
except Exception:  # pragma: no cover
    labelme = None


_IGNORE_LABELS = {"__ignore__", "_background_"}


def _require_runtime_deps() -> None:
    if coco_mask is None:
        raise RuntimeError(
            "labelme2coco requires pycocotools. Install with: pip install pycocotools"
        )
    if labelme is None:
        raise RuntimeError("labelme2coco requires annolid annotation_compat utilities.")


def _normalize_train_count(total: int, train_valid_split: float) -> int:
    if total <= 0:
        return 0
    value = float(train_valid_split)
    if value <= 0:
        return 0
    if value <= 1:
        return max(0, min(total, int(round(total * value))))
    return max(0, min(total, int(round(value))))


def _list_labelme_pairs(input_dir: Path) -> List[Tuple[Path, Path]]:
    out: List[Tuple[Path, Path]] = []
    for json_path in sorted(input_dir.glob("*.json")):
        image_path = resolve_image_path(json_path)
        if image_path is None or not image_path.exists():
            continue
        try:
            payload = load_labelme_json(json_path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        shapes = payload.get("shapes")
        if not isinstance(shapes, list) or not shapes:
            continue
        out.append((json_path, image_path))
    return out


def _read_label_names(
    labels_file: Optional[str], pairs: Sequence[Tuple[Path, Path]]
) -> List[str]:
    if labels_file:
        path = Path(labels_file)
        if not path.exists():
            raise FileNotFoundError(f"labels_file not found: {path}")
        labels: List[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            name = str(raw).strip()
            if not name or name in _IGNORE_LABELS:
                continue
            labels.append(name)
        return labels

    names: Dict[str, None] = {}
    for json_path, _ in pairs:
        try:
            payload = load_labelme_json(json_path)
        except Exception:
            continue
        for shape in payload.get("shapes", []) if isinstance(payload, dict) else []:
            if not isinstance(shape, dict):
                continue
            name = str(shape.get("label") or "").strip()
            if not name or name in _IGNORE_LABELS:
                continue
            names[name] = None
    return sorted(names.keys())


def _shape_to_polygon_points(
    shape: Dict[str, object],
    *,
    image_hw: Tuple[int, int],
    radius_ratio: float,
) -> Optional[List[List[float]]]:
    points = shape.get("points")
    shape_type = str(shape.get("shape_type") or "polygon").lower().strip()
    if not isinstance(points, list) or not points:
        return None

    def _pt(p: object) -> Optional[Tuple[float, float]]:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        try:
            return float(p[0]), float(p[1])
        except Exception:
            return None

    if shape_type == "polygon":
        poly = [_pt(p) for p in points]
        out = [[x, y] for x, y in poly if x is not None and y is not None]  # type: ignore[misc]
        return out if len(out) >= 3 else None

    if shape_type == "rectangle" and len(points) >= 2:
        p1, p2 = _pt(points[0]), _pt(points[1])
        if p1 is None or p2 is None:
            return None
        x1, y1 = p1
        x2, y2 = p2
        xa, xb = sorted([x1, x2])
        ya, yb = sorted([y1, y2])
        return [[xa, ya], [xb, ya], [xb, yb], [xa, yb]]

    if shape_type == "circle" and len(points) >= 2:
        c, e = _pt(points[0]), _pt(points[1])
        if c is None or e is None:
            return None
        cx, cy = c
        ex, ey = e
        r = math.hypot(ex - cx, ey - cy)
        if r <= 0:
            return None
        num = 32
        return [
            [
                cx + r * math.cos(2 * math.pi * i / num),
                cy + r * math.sin(2 * math.pi * i / num),
            ]
            for i in range(num)
        ]

    if shape_type == "point" and len(points) >= 1:
        p = _pt(points[0])
        if p is None:
            return None
        x, y = p
        h, w = image_hw
        r = max(1.0, float(min(h, w)) * float(radius_ratio))
        num = 20
        return [
            [
                x + r * math.cos(2 * math.pi * i / num),
                y + r * math.sin(2 * math.pi * i / num),
            ]
            for i in range(num)
        ]

    return None


def _shape_point(shape: Dict[str, object]) -> Optional[Tuple[float, float]]:
    points = shape.get("points")
    if not isinstance(points, list) or not points:
        return None
    p0 = points[0]
    if not isinstance(p0, (list, tuple)) or len(p0) < 2:
        return None
    try:
        return float(p0[0]), float(p0[1])
    except Exception:
        return None


def _poly_flat(poly: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if len(poly) < 3:
        return None
    flat: List[float] = []
    for p in poly:
        if len(p) < 2:
            return None
        flat.extend([float(p[0]), float(p[1])])
    return flat if len(flat) >= 6 else None


def _encode_mask(mask: np.ndarray) -> Dict[str, object]:
    if coco_mask is None:
        raise RuntimeError("pycocotools is required")
    enc = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(enc, list):  # pragma: no cover
        enc = enc[0]
    return enc


def _build_dataset_skeleton(
    categories: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    now = _dt.datetime.utcnow().isoformat() + "Z"
    return {
        "info": {
            "description": "Annolid LabelMe to COCO export",
            "version": "1.0",
            "year": _dt.datetime.utcnow().year,
            "date_created": now,
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": list(categories),
    }


def _build_segmentation_categories(
    category_names: Sequence[str],
) -> List[Dict[str, object]]:
    return [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(category_names)
    ]


def _build_keypoint_categories(
    category_names: Sequence[str], keypoint_names: Sequence[str]
) -> List[Dict[str, object]]:
    categories: List[Dict[str, object]] = []
    for idx, name in enumerate(category_names):
        categories.append(
            {
                "id": idx + 1,
                "name": name,
                "supercategory": "animal",
                "keypoints": list(keypoint_names),
                "skeleton": [],
            }
        )
    return categories


def convert(
    input_annotated_dir: str,
    output_annotated_dir: str,
    labels_file: Optional[str] = "labels.txt",
    vis: bool = False,  # kept for backward compatibility
    save_mask: bool = True,  # kept for backward compatibility
    train_valid_split: float = 0.7,
    radius_ratio: float = 0.2,
    output_mode: str = "segmentation",
) -> Iterator[Tuple[int, str]]:
    """Convert LabelMe directory to train/valid COCO JSON files.

    Args:
        output_mode: "segmentation" (default) or "keypoints".

    Yields:
        (progress_percent, json_path)
    """
    del vis, save_mask
    _require_runtime_deps()

    mode = str(output_mode or "segmentation").strip().lower()
    if mode in {"coco_keypoints", "keypoint", "pose"}:
        mode = "keypoints"
    if mode not in {"segmentation", "keypoints"}:
        raise ValueError(
            f"Unsupported output_mode={output_mode!r}. Use 'segmentation' or 'keypoints'."
        )

    input_dir = Path(input_annotated_dir).expanduser().resolve()
    out_dir = Path(output_annotated_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pairs = _list_labelme_pairs(input_dir)
    total = len(pairs)
    if total == 0:
        raise ValueError(f"No valid LabelMe json/image pairs found in {input_dir}")

    last_progress = -1
    progress_step = max(1, total // 20)

    if mode == "segmentation":
        category_names = _read_label_names(labels_file, pairs)
        if not category_names:
            raise ValueError(
                "No categories found. Provide labels_file or valid shape labels."
            )
        categories = _build_segmentation_categories(category_names)
    else:
        # Keypoints mode needs schema inference first. Emit staged progress so
        # users don't see a frozen progress window on large datasets.
        keypoint_names_map: Dict[str, None] = {}
        instance_names_map: Dict[str, None] = {}
        for pidx, (json_path, _image_path) in enumerate(pairs, start=1):
            payload = load_labelme_json(json_path)
            if isinstance(payload, dict):
                for shape in payload.get("shapes", []):
                    if not isinstance(shape, dict):
                        continue
                    label_name = str(shape.get("label") or "").strip()
                    if not label_name or label_name in _IGNORE_LABELS:
                        continue
                    shape_type = (
                        str(shape.get("shape_type") or "polygon").lower().strip()
                    )
                    if shape_type == "point":
                        keypoint_names_map[label_name] = None
                    else:
                        instance_names_map[label_name] = None

            if pidx == total or pidx % progress_step == 0:
                pre_progress = min(20, max(1, int(round((pidx / total) * 20))))
                if pre_progress != last_progress:
                    last_progress = pre_progress
                    yield pre_progress, str(json_path)

        keypoint_names = sorted(keypoint_names_map.keys())
        if not keypoint_names:
            raise ValueError(
                "No keypoint labels found. COCO keypoints mode requires point shapes."
            )

        if instance_names_map:
            category_names = sorted(instance_names_map.keys())
        else:
            category_names = ["animal"]
            if labels_file:
                inferred = _read_label_names(labels_file, pairs)
                if inferred:
                    category_names = [inferred[0]]

        categories = _build_keypoint_categories(category_names, keypoint_names)

    name_to_id = {str(cat["name"]): int(cat["id"]) for cat in categories}

    shuffled = list(pairs)
    random.Random(0).shuffle(shuffled)
    train_n = _normalize_train_count(total, train_valid_split)
    train_set = set(shuffled[:train_n])

    train_img_dir = out_dir / "train" / "JPEGImages"
    valid_img_dir = out_dir / "valid" / "JPEGImages"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    valid_img_dir.mkdir(parents=True, exist_ok=True)

    train_data = _build_dataset_skeleton(categories)
    valid_data = _build_dataset_skeleton(categories)

    train_img_id = 1
    valid_img_id = 1
    train_ann_id = 1
    valid_ann_id = 1

    keypoint_index: Dict[str, int] = {}
    default_class_name = category_names[0]
    if mode == "keypoints":
        keypoint_index = {name: idx for idx, name in enumerate(keypoint_names)}

    for idx, (json_path, image_path) in enumerate(shuffled, start=1):
        if mode == "keypoints":
            progress = min(100, int(round(20 + (idx / total) * 80)))
        else:
            progress = int(round((idx / total) * 100))
        if progress != last_progress:
            last_progress = progress
            yield progress, str(json_path)

        payload = load_labelme_json(json_path)
        if not isinstance(payload, dict):
            continue

        with Image.open(image_path) as pil:
            pil = ImageOps.exif_transpose(pil.convert("RGB"))
            w, h = pil.size

            is_train = (json_path, image_path) in train_set
            if is_train:
                dst_img = train_img_dir / f"{json_path.stem}.jpg"
                pil.save(dst_img, format="JPEG", quality=95)
                image_id = train_img_id
                train_img_id += 1
                target = train_data
                ann_id_ref = "train"
            else:
                dst_img = valid_img_dir / f"{json_path.stem}.jpg"
                pil.save(dst_img, format="JPEG", quality=95)
                image_id = valid_img_id
                valid_img_id += 1
                target = valid_data
                ann_id_ref = "valid"

        ann_rel = os.path.relpath(
            dst_img,
            (out_dir / ("train" if is_train else "valid")),
        ).replace("\\", "/")
        target["images"].append(
            {
                "id": int(image_id),
                "license": 1,
                "file_name": ann_rel,
                "height": int(h),
                "width": int(w),
                "date_captured": None,
            }
        )

        if mode == "segmentation":
            grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
            for sidx, shape in enumerate(payload.get("shapes", [])):
                if not isinstance(shape, dict):
                    continue
                label_name = str(shape.get("label") or "").strip()
                if not label_name or label_name not in name_to_id:
                    continue
                gid = shape.get("group_id")
                gid_key = str(gid) if gid is not None else f"shape_{sidx}"
                grouped.setdefault((label_name, gid_key), []).append(shape)

            for (label_name, _gid), shapes in grouped.items():
                mask_union = np.zeros((int(h), int(w)), dtype=np.uint8)
                segmentations: List[List[float]] = []
                for shape in shapes:
                    poly = _shape_to_polygon_points(
                        shape,
                        image_hw=(int(h), int(w)),
                        radius_ratio=float(radius_ratio),
                    )
                    if poly is None:
                        continue
                    flat = _poly_flat(poly)
                    if flat is not None:
                        segmentations.append(flat)

                    try:
                        shape_mask = labelme.utils.shape_to_mask(
                            (int(h), int(w)), poly, "polygon"
                        )
                        mask_union = np.maximum(mask_union, shape_mask.astype(np.uint8))
                    except Exception:
                        continue

                if int(mask_union.sum()) <= 0:
                    continue

                enc = _encode_mask(mask_union)
                area = float(coco_mask.area(enc))
                bbox = coco_mask.toBbox(enc).astype(float).tolist()
                if bbox and len(bbox) == 4:
                    bbox = [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ]

                if not segmentations:
                    x, y, bw, bh = bbox
                    segmentations = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]

                ann = {
                    "id": int(train_ann_id if ann_id_ref == "train" else valid_ann_id),
                    "image_id": int(image_id),
                    "category_id": int(name_to_id[label_name]),
                    "segmentation": segmentations,
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0,
                }
                target["annotations"].append(ann)
                if ann_id_ref == "train":
                    train_ann_id += 1
                else:
                    valid_ann_id += 1
        else:
            # Group by instance id only so one COCO annotation is emitted per instance.
            grouped: Dict[str, List[Dict[str, object]]] = {}
            for sidx, shape in enumerate(payload.get("shapes", [])):
                if not isinstance(shape, dict):
                    continue
                gid = shape.get("group_id")
                gid_key = str(gid) if gid is not None else f"shape_{sidx}"
                grouped.setdefault(gid_key, []).append(shape)

            for _gid, shapes in grouped.items():
                non_point_shapes: List[Dict[str, object]] = []
                point_shapes: List[Dict[str, object]] = []
                for shape in shapes:
                    st = str(shape.get("shape_type") or "polygon").lower().strip()
                    if st == "point":
                        point_shapes.append(shape)
                    else:
                        non_point_shapes.append(shape)

                if not non_point_shapes and not point_shapes:
                    continue

                class_name = default_class_name
                for shape in non_point_shapes:
                    label_name = str(shape.get("label") or "").strip()
                    if label_name and label_name in name_to_id:
                        class_name = label_name
                        break

                kpt_values: List[float] = [0.0] * (len(keypoint_index) * 3)
                num_keypoints = 0
                xy_visible: List[Tuple[float, float]] = []
                for shape in point_shapes:
                    label_name = str(shape.get("label") or "").strip()
                    kidx = keypoint_index.get(label_name)
                    if kidx is None:
                        continue
                    xy = _shape_point(shape)
                    if xy is None:
                        continue
                    x, y = xy
                    if x < 0 or y < 0 or x >= w or y >= h:
                        # Still keep keypoint visible in COCO, but clamp for safety.
                        x = min(max(float(x), 0.0), float(max(w - 1, 0)))
                        y = min(max(float(y), 0.0), float(max(h - 1, 0)))
                    off = kidx * 3
                    # Keep first valid keypoint instance for duplicate labels.
                    if kpt_values[off + 2] > 0:
                        continue
                    kpt_values[off] = float(x)
                    kpt_values[off + 1] = float(y)
                    kpt_values[off + 2] = 2.0
                    num_keypoints += 1
                    xy_visible.append((float(x), float(y)))

                segmentations: List[List[float]] = []
                mask_union = np.zeros((int(h), int(w)), dtype=np.uint8)
                for shape in non_point_shapes:
                    poly = _shape_to_polygon_points(
                        shape,
                        image_hw=(int(h), int(w)),
                        radius_ratio=float(radius_ratio),
                    )
                    if poly is None:
                        continue
                    flat = _poly_flat(poly)
                    if flat is not None:
                        segmentations.append(flat)
                    try:
                        shape_mask = labelme.utils.shape_to_mask(
                            (int(h), int(w)), poly, "polygon"
                        )
                        mask_union = np.maximum(mask_union, shape_mask.astype(np.uint8))
                    except Exception:
                        continue

                if int(mask_union.sum()) > 0:
                    enc = _encode_mask(mask_union)
                    area = float(coco_mask.area(enc))
                    bbox = coco_mask.toBbox(enc).astype(float).tolist()
                    if bbox and len(bbox) == 4:
                        bbox = [
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ]
                elif xy_visible:
                    xs = [xy[0] for xy in xy_visible]
                    ys = [xy[1] for xy in xy_visible]
                    pad = 2.0
                    x0 = max(0.0, min(xs) - pad)
                    y0 = max(0.0, min(ys) - pad)
                    x1 = min(float(w - 1), max(xs) + pad)
                    y1 = min(float(h - 1), max(ys) + pad)
                    bbox = [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]
                    area = float(bbox[2] * bbox[3])
                    segmentations = []
                else:
                    # No geometry and no keypoints => skip.
                    continue

                ann = {
                    "id": int(train_ann_id if ann_id_ref == "train" else valid_ann_id),
                    "image_id": int(image_id),
                    "category_id": int(
                        name_to_id.get(class_name, name_to_id[default_class_name])
                    ),
                    "segmentation": segmentations,
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "keypoints": [float(v) for v in kpt_values],
                    "num_keypoints": int(num_keypoints),
                }
                target["annotations"].append(ann)
                if ann_id_ref == "train":
                    train_ann_id += 1
                else:
                    valid_ann_id += 1

    train_ann = out_dir / "train" / "annotations.json"
    valid_ann = out_dir / "valid" / "annotations.json"
    train_ann.parent.mkdir(parents=True, exist_ok=True)
    valid_ann.parent.mkdir(parents=True, exist_ok=True)
    train_ann.write_text(json.dumps(train_data, indent=2), encoding="utf-8")
    valid_ann.write_text(json.dumps(valid_data, indent=2), encoding="utf-8")

    (out_dir / "annotations_train.json").write_text(
        json.dumps(train_data, indent=2), encoding="utf-8"
    )
    (out_dir / "annotations_valid.json").write_text(
        json.dumps(valid_data, indent=2), encoding="utf-8"
    )

    meta = {
        "DATASET": {
            "name": input_dir.name,
            "train_info": str(Path("train") / "annotations.json"),
            "train_images": str(Path("train") / "JPEGImages"),
            "valid_info": str(Path("valid") / "annotations.json"),
            "valid_images": str(Path("valid") / "JPEGImages"),
            "class_names": list(category_names),
            "output_mode": mode,
        }
    }
    (out_dir / "data.yaml").write_text(
        "\n".join(
            [
                "DATASET:",
                f"  name: '{meta['DATASET']['name']}'",
                f"  train_info: '{meta['DATASET']['train_info']}'",
                f"  train_images: '{meta['DATASET']['train_images']}'",
                f"  valid_info: '{meta['DATASET']['valid_info']}'",
                f"  valid_images: '{meta['DATASET']['valid_images']}'",
                f"  class_names: {meta['DATASET']['class_names']}",
                f"  output_mode: '{meta['DATASET']['output_mode']}'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info("COCO export complete: %s (mode=%s)", out_dir, mode)
