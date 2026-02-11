"""Convert COCO datasets to LabelMe JSON files."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence


logger = logging.getLogger(__name__)


def _ensure_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _bbox_to_rect_points(bbox_xywh: Sequence[object]) -> Optional[List[List[float]]]:
    if len(bbox_xywh) < 4:
        return None
    x = _ensure_float(bbox_xywh[0])
    y = _ensure_float(bbox_xywh[1])
    w = _ensure_float(bbox_xywh[2])
    h = _ensure_float(bbox_xywh[3])
    if w <= 0 or h <= 0:
        return None
    return [[x, y], [x + w, y + h]]


def _segmentation_polygons(segmentation: object) -> List[List[List[float]]]:
    out: List[List[List[float]]] = []
    if not isinstance(segmentation, list):
        return out
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        pts: List[List[float]] = []
        for i in range(0, len(poly) - 1, 2):
            pts.append([_ensure_float(poly[i]), _ensure_float(poly[i + 1])])
        if len(pts) >= 3:
            out.append(pts)
    return out


def _keypoint_shapes(
    keypoints: object,
    *,
    keypoint_names: Optional[List[str]],
    group_id: Optional[int],
) -> List[Dict[str, object]]:
    if not isinstance(keypoints, list) or len(keypoints) < 3:
        return []
    out: List[Dict[str, object]] = []
    count = len(keypoints) // 3
    for idx in range(count):
        x = _ensure_float(keypoints[idx * 3 + 0])
        y = _ensure_float(keypoints[idx * 3 + 1])
        v = _ensure_float(keypoints[idx * 3 + 2])
        if v <= 0:
            continue
        label = (
            keypoint_names[idx]
            if keypoint_names and idx < len(keypoint_names)
            else f"kp_{idx}"
        )
        out.append(
            {
                "label": str(label),
                "points": [[float(x), float(y)]],
                "group_id": group_id,
                "shape_type": "point",
                "flags": {},
            }
        )
    return out


def _resolve_image_path(
    image_file_name: str,
    *,
    coco_json_path: Path,
    images_dir: Optional[Path],
) -> Path:
    rel = Path(str(image_file_name))
    if rel.is_absolute():
        return rel
    candidates: List[Path] = []
    if images_dir is not None:
        candidates.append((images_dir / rel).resolve())
        candidates.append((images_dir / rel.name).resolve())

    # Common COCO layouts:
    # - <root>/annotations/train.json + file_name=images/xxx.png
    # - <root>/train/annotations.json + file_name=xxx.png
    ann_dir = coco_json_path.parent
    root_dir = ann_dir.parent
    candidates.extend(
        [
            (ann_dir / rel).resolve(),
            (root_dir / rel).resolve(),
            (root_dir / "images" / rel).resolve(),
            (root_dir / "images" / rel.name).resolve(),
            (ann_dir / "images" / rel).resolve(),
            (ann_dir / "images" / rel.name).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Preserve previous behavior for warning messages.
    return (ann_dir / rel).resolve()


def _image_target_relative(file_name: str, *, image_id: int) -> Path:
    rel = Path(str(file_name))
    # Keep relative structure when provided by COCO. For absolute paths,
    # place under a stable local images/ folder.
    if rel.is_absolute():
        rel = Path("images") / rel.name
    if not rel.suffix:
        rel = rel.with_suffix(".png")
    # Avoid collisions in flat or duplicated names.
    return rel.with_name(f"{rel.stem}__{int(image_id)}{rel.suffix}")


def _install_file(src: Path, dst: Path, *, mode: str = "hardlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
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


def convert_coco_json_to_labelme(
    coco_json_path: Path | str,
    *,
    output_dir: Path | str,
    images_dir: Optional[Path | str] = None,
    include_polygons: bool = True,
    include_keypoints: bool = True,
    include_bbox_when_missing: bool = True,
) -> Dict[str, int]:
    """Convert one COCO annotation JSON file into per-image LabelMe JSON files."""

    coco_json_path = Path(coco_json_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    images_dir_path = (
        Path(images_dir).expanduser().resolve() if images_dir is not None else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(coco_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid COCO payload: {coco_json_path}")

    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    image_list = images if isinstance(images, list) else []
    ann_list = annotations if isinstance(annotations, list) else []
    cat_list = categories if isinstance(categories, list) else []

    categories_by_id: Dict[int, Dict[str, object]] = {}
    for cat in cat_list:
        if not isinstance(cat, dict):
            continue
        try:
            cid = int(cat.get("id"))
        except Exception:
            continue
        categories_by_id[cid] = cat

    anns_by_image: Dict[int, List[Dict[str, object]]] = {}
    for ann in ann_list:
        if not isinstance(ann, dict):
            continue
        try:
            image_id = int(ann.get("image_id"))
        except Exception:
            continue
        anns_by_image.setdefault(image_id, []).append(ann)

    converted = 0
    missing_images = 0
    shapes_total = 0

    for image in image_list:
        if not isinstance(image, dict):
            continue
        try:
            image_id = int(image.get("id"))
        except Exception:
            continue

        file_name = str(image.get("file_name") or "").strip()
        if not file_name:
            continue
        resolved_image = _resolve_image_path(
            file_name, coco_json_path=coco_json_path, images_dir=images_dir_path
        )
        if not resolved_image.exists():
            missing_images += 1
            logger.warning("Image file missing for COCO record: %s", resolved_image)
            continue

        image_width = int(_ensure_float(image.get("width"), default=0))
        image_height = int(_ensure_float(image.get("height"), default=0))

        shapes: List[Dict[str, object]] = []
        for ann in anns_by_image.get(image_id, []):
            category_id = int(_ensure_float(ann.get("category_id"), default=0))
            category = categories_by_id.get(category_id) or {}
            category_name = str(category.get("name") or f"class_{category_id}")
            keypoint_names = (
                [str(x) for x in category.get("keypoints", [])]
                if isinstance(category.get("keypoints"), list)
                else None
            )
            group_id = (
                int(_ensure_float(ann.get("id"))) if ann.get("id") is not None else None
            )

            segmentation = ann.get("segmentation")
            polygons = _segmentation_polygons(segmentation)
            if include_polygons and polygons:
                for poly in polygons:
                    shapes.append(
                        {
                            "label": category_name,
                            "points": poly,
                            "group_id": group_id,
                            "shape_type": "polygon",
                            "flags": {},
                        }
                    )

            if include_bbox_when_missing and not polygons:
                bbox = ann.get("bbox")
                if isinstance(bbox, list):
                    rect = _bbox_to_rect_points(bbox)
                    if rect is not None:
                        shapes.append(
                            {
                                "label": category_name,
                                "points": rect,
                                "group_id": group_id,
                                "shape_type": "rectangle",
                                "flags": {},
                            }
                        )

            if include_keypoints:
                shapes.extend(
                    _keypoint_shapes(
                        ann.get("keypoints"),
                        keypoint_names=keypoint_names,
                        group_id=group_id,
                    )
                )

        out_json = output_dir / f"{Path(file_name).stem}.json"
        labelme_payload = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": str(resolved_image),
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }
        out_json.write_text(json.dumps(labelme_payload, indent=2), encoding="utf-8")
        converted += 1
        shapes_total += len(shapes)

    return {
        "images_total": len(image_list),
        "converted_images": int(converted),
        "missing_images": int(missing_images),
        "shapes_total": int(shapes_total),
    }


def convert_coco_json_to_labelme_dataset(
    coco_json_path: Path | str,
    *,
    output_dir: Path | str,
    images_dir: Optional[Path | str] = None,
    include_polygons: bool = True,
    include_keypoints: bool = True,
    include_bbox_when_missing: bool = True,
    link_mode: str = "hardlink",
) -> Dict[str, int]:
    """Convert one COCO JSON into a LabelMe dataset with sidecar JSON next to images."""

    coco_json_path = Path(coco_json_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    images_dir_path = (
        Path(images_dir).expanduser().resolve() if images_dir is not None else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(coco_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid COCO payload: {coco_json_path}")

    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    image_list = images if isinstance(images, list) else []
    ann_list = annotations if isinstance(annotations, list) else []
    cat_list = categories if isinstance(categories, list) else []

    categories_by_id: Dict[int, Dict[str, object]] = {}
    for cat in cat_list:
        if not isinstance(cat, dict):
            continue
        try:
            cid = int(cat.get("id"))
        except Exception:
            continue
        categories_by_id[cid] = cat

    anns_by_image: Dict[int, List[Dict[str, object]]] = {}
    for ann in ann_list:
        if not isinstance(ann, dict):
            continue
        try:
            image_id = int(ann.get("image_id"))
        except Exception:
            continue
        anns_by_image.setdefault(image_id, []).append(ann)

    converted = 0
    missing_images = 0
    shapes_total = 0
    copied_images = 0

    for image in image_list:
        if not isinstance(image, dict):
            continue
        try:
            image_id = int(image.get("id"))
        except Exception:
            continue
        file_name = str(image.get("file_name") or "").strip()
        if not file_name:
            continue
        src_image = _resolve_image_path(
            file_name, coco_json_path=coco_json_path, images_dir=images_dir_path
        )
        if not src_image.exists():
            missing_images += 1
            logger.warning("Image file missing for COCO record: %s", src_image)
            continue

        rel_target = _image_target_relative(file_name, image_id=image_id)
        dst_image = (output_dir / rel_target).resolve()
        _install_file(src_image, dst_image, mode=link_mode)
        copied_images += 1

        image_width = int(_ensure_float(image.get("width"), default=0))
        image_height = int(_ensure_float(image.get("height"), default=0))
        if image_width <= 0 or image_height <= 0:
            try:
                from PIL import Image as PILImage

                with PILImage.open(dst_image) as pil:
                    image_width, image_height = pil.size
            except Exception:
                pass

        shapes: List[Dict[str, object]] = []
        for ann in anns_by_image.get(image_id, []):
            category_id = int(_ensure_float(ann.get("category_id"), default=0))
            category = categories_by_id.get(category_id) or {}
            category_name = str(category.get("name") or f"class_{category_id}")
            keypoint_names = (
                [str(x) for x in category.get("keypoints", [])]
                if isinstance(category.get("keypoints"), list)
                else None
            )
            group_id = (
                int(_ensure_float(ann.get("id"))) if ann.get("id") is not None else None
            )

            segmentation = ann.get("segmentation")
            polygons = _segmentation_polygons(segmentation)
            if include_polygons and polygons:
                for poly in polygons:
                    shapes.append(
                        {
                            "label": category_name,
                            "points": poly,
                            "group_id": group_id,
                            "shape_type": "polygon",
                            "flags": {},
                        }
                    )

            if include_bbox_when_missing and not polygons:
                bbox = ann.get("bbox")
                if isinstance(bbox, list):
                    rect = _bbox_to_rect_points(bbox)
                    if rect is not None:
                        shapes.append(
                            {
                                "label": category_name,
                                "points": rect,
                                "group_id": group_id,
                                "shape_type": "rectangle",
                                "flags": {},
                            }
                        )

            if include_keypoints:
                shapes.extend(
                    _keypoint_shapes(
                        ann.get("keypoints"),
                        keypoint_names=keypoint_names,
                        group_id=group_id,
                    )
                )

        out_json = dst_image.with_suffix(".json")
        labelme_payload = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": dst_image.name,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }
        out_json.write_text(json.dumps(labelme_payload, indent=2), encoding="utf-8")
        converted += 1
        shapes_total += len(shapes)

    return {
        "images_total": len(image_list),
        "converted_images": int(converted),
        "missing_images": int(missing_images),
        "shapes_total": int(shapes_total),
        "copied_images": int(copied_images),
    }


class COCO2Labeme:
    """Backward-compatible wrapper for converting COCO JSON(s) to LabelMe."""

    def __init__(self, annotations: str, images_dir: str) -> None:
        self.annotations = annotations
        self.images_dir = images_dir

    def get_annos(self) -> List[str]:
        return sorted(str(p) for p in Path(self.annotations).glob("*.json"))

    def parse_coco_json(self, json_file: str) -> Dict[str, object]:
        return json.loads(Path(json_file).read_text(encoding="utf-8"))

    def to_labelme(self, json_data: Dict[str, object]) -> None:
        # Preserve legacy API: conversion requires file path context, so prefer convert().
        raise NotImplementedError("Use convert() with on-disk COCO JSON files.")

    def convert(self) -> None:
        for jf in self.get_annos():
            convert_coco_json_to_labelme(
                jf,
                output_dir=self.images_dir,
                images_dir=self.images_dir,
            )
            logger.info("Finished %s.", jf)


def convert_coco_dir_to_labelme(
    annotations_dir: Path | str,
    *,
    output_dir: Path | str,
    images_dir: Optional[Path | str] = None,
) -> List[Dict[str, int]]:
    """Convert all COCO JSON files in a directory to LabelMe."""
    annotations_dir = Path(annotations_dir).expanduser().resolve()
    results: List[Dict[str, int]] = []
    for coco_json in sorted(annotations_dir.glob("*.json")):
        results.append(
            convert_coco_json_to_labelme(
                coco_json,
                output_dir=output_dir,
                images_dir=images_dir,
            )
        )
    return results


def convert_coco_annotations_dir_to_labelme_dataset(
    annotations_dir: Path | str,
    *,
    output_dir: Path | str,
    images_dir: Optional[Path | str] = None,
    recursive: bool = True,
    link_mode: str = "hardlink",
) -> Dict[str, int]:
    """Convert all COCO JSON files under annotations_dir into one LabelMe dataset.

    Output contains images and sidecar LabelMe JSON files together.
    """
    annotations_dir = Path(annotations_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    globber = annotations_dir.rglob if recursive else annotations_dir.glob
    coco_jsons = sorted(p for p in globber("*.json") if p.is_file())

    totals = {
        "json_files_total": len(coco_jsons),
        "images_total": 0,
        "converted_images": 0,
        "missing_images": 0,
        "shapes_total": 0,
        "copied_images": 0,
    }
    for coco_json in coco_jsons:
        summary = convert_coco_json_to_labelme_dataset(
            coco_json,
            output_dir=output_dir,
            images_dir=images_dir,
            link_mode=link_mode,
        )
        totals["images_total"] += int(summary.get("images_total", 0))
        totals["converted_images"] += int(summary.get("converted_images", 0))
        totals["missing_images"] += int(summary.get("missing_images", 0))
        totals["shapes_total"] += int(summary.get("shapes_total", 0))
        totals["copied_images"] += int(summary.get("copied_images", 0))
    return totals
