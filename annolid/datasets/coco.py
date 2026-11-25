from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image


_COCO_SPLIT_FILENAMES = {
    # Prefer keypoint annotations when multiple COCO variants are present.
    "train": ("person_keypoints_train.json", "instances_train.json", "train.json"),
    "val": ("person_keypoints_val.json", "instances_val.json", "val.json"),
    "test": ("person_keypoints_test.json", "instances_test.json", "test.json"),
}

_COCO_SPLIT_DIRS = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}

_COCO_SPLIT_LOCAL_FILENAMES = (
    "annotations.json",
    "person_keypoints.json",
    "instances.json",
)


def looks_like_coco_spec(data_cfg: Dict[str, Any]) -> bool:
    fmt = str(data_cfg.get("format") or data_cfg.get("type") or "").strip().lower()
    if fmt in {"coco", "coco_pose", "coco_keypoints"}:
        return True

    for split in ("train", "val", "test"):
        value = data_cfg.get(split)
        entries = value if isinstance(value, (list, tuple)) else [value]
        for entry in entries:
            if isinstance(entry, str) and entry.strip().lower().endswith(".json"):
                return True
    return False


def build_coco_spec_from_annotations_dir(
    annotations_dir: Path,
) -> Optional[Dict[str, Any]]:
    if not annotations_dir.is_dir():
        return None

    split_files: Dict[str, Path] = {}
    for split, candidates in _COCO_SPLIT_FILENAMES.items():
        for name in candidates:
            candidate = annotations_dir / name
            if candidate.exists():
                split_files[split] = candidate
                break

    if "train" not in split_files and "val" not in split_files:
        return None

    root_path = annotations_dir
    image_root = "."
    if (annotations_dir / "images").exists():
        image_root = "images"
    elif (annotations_dir.parent / "images").exists():
        root_path = annotations_dir.parent
        image_root = "images"

    payload: Dict[str, Any] = {
        "format": "coco",
        "path": str(root_path.resolve()),
        "image_root": image_root,
    }
    resolved_root = root_path.resolve()
    for split, ann_path in split_files.items():
        payload[split] = str(ann_path.resolve().relative_to(resolved_root))
    return payload


def build_coco_spec_from_split_dirs(dataset_root: Path) -> Optional[Dict[str, Any]]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        return None

    split_files: Dict[str, Path] = {}
    for split, dir_names in _COCO_SPLIT_DIRS.items():
        for dir_name in dir_names:
            split_dir = dataset_root / dir_name
            if not split_dir.is_dir():
                continue
            for file_name in _COCO_SPLIT_LOCAL_FILENAMES:
                ann_path = split_dir / file_name
                if ann_path.exists():
                    split_files[split] = ann_path
                    break
            if split in split_files:
                break

    if "train" not in split_files and "val" not in split_files:
        return None

    payload: Dict[str, Any] = {
        "format": "coco",
        "path": str(dataset_root),
        "image_root": ".",
    }
    for split, ann_path in split_files.items():
        payload[split] = str(ann_path.relative_to(dataset_root))
    return payload


def build_coco_spec_from_dataset_path(path: Path) -> Optional[Dict[str, Any]]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        return None

    payload = build_coco_spec_from_annotations_dir(candidate)
    if payload is not None:
        return payload

    common = candidate / "annotations"
    if common.is_dir():
        payload = build_coco_spec_from_annotations_dir(common)
        if payload is not None:
            return payload

    payload = build_coco_spec_from_split_dirs(candidate)
    if payload is not None:
        return payload

    for depth in (1, 2):
        for pattern in (
            "/".join(["*"] * depth + [name])
            for name in _COCO_SPLIT_FILENAMES["train"] + _COCO_SPLIT_LOCAL_FILENAMES
        ):
            for json_path in candidate.glob(pattern):
                parent = json_path.parent
                payload = build_coco_spec_from_annotations_dir(parent)
                if payload is not None:
                    return payload
                payload = build_coco_spec_from_split_dirs(parent.parent)
                if payload is not None:
                    return payload
    return None


def discover_coco_annotations_dir(
    input_dir: Path,
    *,
    max_depth: int = 2,
) -> Optional[Path]:
    """Find a COCO annotations directory from a user-selected dataset path."""
    candidate = Path(input_dir).expanduser().resolve()
    if not candidate.is_dir():
        return None

    if build_coco_spec_from_annotations_dir(candidate) is not None:
        return candidate

    common = candidate / "annotations"
    if common.is_dir() and build_coco_spec_from_annotations_dir(common) is not None:
        return common

    for split_dir_names in _COCO_SPLIT_DIRS.values():
        for dir_name in split_dir_names:
            split_dir = candidate / dir_name
            if not split_dir.is_dir():
                continue
            for file_name in _COCO_SPLIT_LOCAL_FILENAMES:
                if (split_dir / file_name).exists():
                    return candidate

    depth = max(1, int(max_depth))
    nested_names = (
        tuple(name for names in _COCO_SPLIT_FILENAMES.values() for name in names)
        + _COCO_SPLIT_LOCAL_FILENAMES
    )
    for name in nested_names:
        for prefix in ("", "train/", "val/", "valid/", "validation/", "test/"):
            pattern = (
                "/".join(["*"] * depth + [prefix + name])
                if prefix
                else "/".join(["*"] * depth + [name])
            )
            for json_path in candidate.glob(pattern):
                parent = json_path.parent
                if (
                    parent.is_dir()
                    and build_coco_spec_from_annotations_dir(parent) is not None
                ):
                    return parent
                if build_coco_spec_from_split_dirs(parent.parent) is not None:
                    return parent.parent
    return None


def _annotation_has_segmentation(ann: Dict[str, Any]) -> bool:
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                return True
        return False
    if isinstance(seg, dict):
        counts = seg.get("counts")
        size = seg.get("size")
        return bool(
            counts is not None and isinstance(size, (list, tuple)) and len(size) == 2
        )
    return False


def _coco_has_segmentation(*, config_path: Path, payload: Dict[str, Any]) -> bool:
    ann_paths = resolve_coco_annotation_paths(config_path=config_path, payload=payload)
    for ann_path in ann_paths.values():
        if ann_path is None or not ann_path.exists():
            continue
        ann_payload = _load_json_dict(ann_path)
        annotations = ann_payload.get("annotations")
        for ann in annotations if isinstance(annotations, list) else []:
            if isinstance(ann, dict) and _annotation_has_segmentation(ann):
                return True
    return False


def infer_coco_task(*, config_path: Path, payload: Dict[str, Any]) -> str:
    if (
        payload.get("kpt_shape")
        or payload.get("keypoint_names")
        or payload.get("flip_idx")
    ):
        return "pose"

    has_seg = False
    ann_paths = resolve_coco_annotation_paths(config_path=config_path, payload=payload)
    for ann_path in ann_paths.values():
        if ann_path is None or not ann_path.exists():
            continue
        ann_payload = _load_json_dict(ann_path)
        categories = ann_payload.get("categories")
        for category in categories if isinstance(categories, list) else []:
            kpts = category.get("keypoints") if isinstance(category, dict) else None
            if isinstance(kpts, list) and len(kpts) > 0:
                return "pose"
        annotations = ann_payload.get("annotations")
        for ann in annotations if isinstance(annotations, list) else []:
            if not isinstance(ann, dict):
                continue
            keypoints = ann.get("keypoints")
            if isinstance(keypoints, list) and len(keypoints) >= 3:
                return "pose"
            if not has_seg and _annotation_has_segmentation(ann):
                has_seg = True
    return "seg" if has_seg else "detect"


def _normalize_coco_expected_task(expected_task: Optional[str]) -> Optional[str]:
    if expected_task is None:
        return None
    token = str(expected_task).strip().lower()
    if not token:
        return None
    if token in {"seg", "segment", "segmentation"}:
        return "seg"
    if token in {"detect", "detection", "bbox"}:
        return "detect"
    return token


def materialize_coco_spec_as_yolo(
    *,
    config_path: Path,
    output_dir: Optional[Path] = None,
    link_mode: str = "hardlink",
    expected_task: Optional[str] = None,
) -> Path:
    payload = read_yaml_dict(config_path)
    inferred_task = infer_coco_task(config_path=config_path, payload=payload)
    norm_expected = _normalize_coco_expected_task(expected_task)

    if norm_expected is None:
        target_task = inferred_task
    elif norm_expected == "pose":
        if inferred_task != "pose":
            raise ValueError(
                f"Expected a COCO pose dataset, but inferred {inferred_task} from {config_path}"
            )
        target_task = "pose"
    elif norm_expected == "seg":
        if inferred_task != "seg" and not _coco_has_segmentation(
            config_path=config_path, payload=payload
        ):
            raise ValueError(
                f"Expected a COCO seg dataset, but inferred {inferred_task} from {config_path}"
            )
        target_task = "seg"
    elif norm_expected == "detect":
        # COCO segmentation and pose datasets also carry bounding boxes and can train detectors.
        target_task = "detect"
    else:
        if inferred_task != norm_expected:
            raise ValueError(
                f"Expected a COCO {expected_task} dataset, but inferred {inferred_task} from {config_path}"
            )
        target_task = norm_expected

    target_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(tempfile.mkdtemp(prefix=f"{config_path.stem}_coco_yolo_")).resolve()
    )

    if target_task == "pose":
        from annolid.segmentation.dino_kpseg.data import (
            load_coco_pose_spec,
            materialize_coco_pose_as_yolo,
        )

        spec = load_coco_pose_spec(config_path)
        return materialize_coco_pose_as_yolo(
            spec=spec,
            output_dir=target_dir,
            link_mode=link_mode,
        )
    if target_task == "seg":
        return materialize_coco_segmentation_as_yolo(
            config_path=config_path,
            output_dir=target_dir,
            link_mode=link_mode,
        )
    return materialize_coco_detection_as_yolo(
        config_path=config_path,
        output_dir=target_dir,
        link_mode=link_mode,
    )


def materialize_coco_detection_as_yolo(
    *,
    config_path: Path,
    output_dir: Path,
    link_mode: str = "hardlink",
) -> Path:
    return _materialize_coco_boxes_or_segments_as_yolo(
        config_path=config_path,
        output_dir=output_dir,
        link_mode=link_mode,
        segmentation=False,
    )


def materialize_coco_segmentation_as_yolo(
    *,
    config_path: Path,
    output_dir: Path,
    link_mode: str = "hardlink",
) -> Path:
    return _materialize_coco_boxes_or_segments_as_yolo(
        config_path=config_path,
        output_dir=output_dir,
        link_mode=link_mode,
        segmentation=True,
    )


def _materialize_coco_boxes_or_segments_as_yolo(
    *,
    config_path: Path,
    output_dir: Path,
    link_mode: str = "hardlink",
    segmentation: bool = False,
) -> Path:
    payload = read_yaml_dict(config_path)
    root_path = resolve_coco_root_path(config_path=config_path, payload=payload)
    image_root = payload.get("image_root") or payload.get("images") or "."
    image_root_path = resolve_dataset_path(
        value=str(image_root),
        config_path=config_path,
        root_path=root_path,
    )
    ann_paths = resolve_coco_annotation_paths(config_path=config_path, payload=payload)

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)

    first_existing_ann = next(iter_existing_paths(ann_paths.values()), None)
    names, category_id_to_index = _load_categories(
        first_existing_ann, default_name="object"
    )

    raw_val_split = payload.get("val_split")
    val_split = _safe_float(raw_val_split)
    if val_split is None:
        val_split = 0.1 if ann_paths.get("val") is None else 0.0
    val_split = float(max(0.0, min(0.9, val_split)))
    val_seed = _safe_int(payload.get("val_seed"))
    if val_seed is None:
        val_seed = 0
    auto_val_split = _safe_bool(payload.get("auto_val_split"), default=True)

    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    auto_val_split_used = False

    if (
        ann_paths.get("train") is not None
        and ann_paths.get("val") is None
        and auto_val_split
        and val_split > 0.0
    ):
        train_ids, val_ids = _split_image_ids_for_val(
            ann_paths["train"],
            val_split=val_split,
            seed=int(val_seed),
        )
        counts["train"] = _convert_coco_split(
            split_name="train",
            ann_path=ann_paths["train"],
            image_root=image_root_path,
            root_path=root_path,
            output_dir=output_dir,
            category_id_to_index=category_id_to_index,
            include_image_ids=train_ids,
            link_mode=link_mode,
            segmentation=segmentation,
        )
        counts["val"] = _convert_coco_split(
            split_name="val",
            ann_path=ann_paths["train"],
            image_root=image_root_path,
            root_path=root_path,
            output_dir=output_dir,
            category_id_to_index=category_id_to_index,
            include_image_ids=val_ids,
            link_mode=link_mode,
            segmentation=segmentation,
        )
        auto_val_split_used = counts["val"] > 0
    else:
        for split in ("train", "val", "test"):
            ann_path = ann_paths.get(split)
            if ann_path is None:
                continue
            counts[split] = _convert_coco_split(
                split_name=split,
                ann_path=ann_path,
                image_root=image_root_path,
                root_path=root_path,
                output_dir=output_dir,
                category_id_to_index=category_id_to_index,
                include_image_ids=None,
                link_mode=link_mode,
                segmentation=segmentation,
            )

    yolo_payload: Dict[str, Any] = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": int(max(1, len(names))),
        "names": list(names),
        "images_train_count": int(counts["train"]),
        "images_val_count": int(counts["val"]),
    }
    if counts["test"] or ann_paths.get("test") is not None:
        yolo_payload["test"] = "images/test"
        yolo_payload["images_test_count"] = int(counts["test"])
    if auto_val_split_used:
        yolo_payload["auto_val_split"] = True
        yolo_payload["val_split"] = float(val_split)
        yolo_payload["val_seed"] = int(val_seed)
        yolo_payload["source_val_missing"] = True

    yolo_yaml = output_dir / "data.yaml"
    yolo_yaml.write_text(
        yaml.safe_dump(yolo_payload, sort_keys=False),
        encoding="utf-8",
    )
    return yolo_yaml


def resolve_coco_annotation_paths(
    *,
    config_path: Path,
    payload: Dict[str, Any],
) -> Dict[str, Optional[Path]]:
    root_path = resolve_coco_root_path(config_path=config_path, payload=payload)
    return {
        split: resolve_dataset_path(
            value=str(payload[split]),
            config_path=config_path,
            root_path=root_path,
        )
        if payload.get(split)
        else None
        for split in ("train", "val", "test")
    }


def resolve_coco_root_path(*, config_path: Path, payload: Dict[str, Any]) -> Path:
    raw_root = payload.get("path")
    if not raw_root:
        return config_path.parent.resolve()
    root = Path(str(raw_root)).expanduser()
    if root.is_absolute():
        return root.resolve()
    return (config_path.parent / root).resolve()


def resolve_dataset_path(*, config_path: Path, root_path: Path, value: str) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root_path / candidate).resolve()


def read_yaml_dict(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return dict(payload)


def load_coco_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid COCO JSON: {path}")
    return payload


def load_coco_categories(path: str | Path) -> List[Dict[str, Any]]:
    payload = load_coco_json(path)
    categories = payload.get("categories", [])
    categories = categories if isinstance(categories, list) else []
    parsed: List[Dict[str, Any]] = []
    for category in categories:
        if isinstance(category, dict):
            parsed.append(category)
    return sorted(parsed, key=lambda category: int(category.get("id", 0)))


def load_coco_class_names(path: str | Path) -> List[str]:
    return [str(category.get("name") or "") for category in load_coco_categories(path)]


def load_coco_keypoint_meta(path: str | Path) -> Dict[str, Any]:
    categories = load_coco_categories(path)
    if not categories:
        return {"num_keypoints": 0, "keypoint_names": [], "skeleton": []}
    category = categories[0]
    keypoint_names = category.get("keypoints", [])
    skeleton = category.get("skeleton", [])
    keypoint_names = keypoint_names if isinstance(keypoint_names, list) else []
    skeleton = skeleton if isinstance(skeleton, list) else []
    return {
        "num_keypoints": len(keypoint_names),
        "keypoint_names": [str(name) for name in keypoint_names],
        "skeleton": skeleton,
    }


def load_coco_category_id_map(path: str | Path) -> Dict[int, int]:
    return {
        int(category["id"]): idx
        for idx, category in enumerate(load_coco_categories(path))
        if "id" in category
    }


def iter_existing_paths(paths: Iterable[Optional[Path]]) -> Iterable[Path]:
    for path in paths:
        if path is not None and path.exists():
            yield path


def _merge_multi_segment(
    segments: List[List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """Merge multi-part polygon contours into a single closed YOLO-seg contour."""
    if not segments:
        return []
    merged = list(segments[0])
    for seg in segments[1:]:
        if len(seg) < 3:
            continue
        best_dist = float("inf")
        best_i = 0
        best_j = 0
        for i, (mx, my) in enumerate(merged):
            for j, (sx, sy) in enumerate(seg):
                dx = mx - sx
                dy = my - sy
                dist_sq = dx * dx + dy * dy
                if dist_sq < best_dist:
                    best_dist = dist_sq
                    best_i = i
                    best_j = j
        rotated = seg[best_j:] + seg[:best_j] + [seg[best_j]]
        merged = (
            merged[: best_i + 1] + rotated + [merged[best_i]] + merged[best_i + 1 :]
        )
    return merged


def _decode_coco_rle_to_polygons(
    rle: Dict[str, Any],
) -> List[List[Tuple[float, float]]]:
    """Decode uncompressed or compressed COCO RLE dict into polygon contours."""
    import cv2
    import numpy as np

    counts = rle.get("counts")
    size = rle.get("size")
    if counts is None or not isinstance(size, (list, tuple)) or len(size) != 2:
        return []
    try:
        height, width = int(size[0]), int(size[1])
    except Exception:
        return []
    if height <= 0 or width <= 0:
        return []

    mask: Optional[Any] = None
    if isinstance(counts, list):
        flat = np.zeros(height * width, dtype=np.uint8)
        pos = 0
        val = 0
        for run in counts:
            try:
                length = max(0, int(run))
            except Exception:
                return []
            end = min(flat.size, pos + length)
            if val == 1 and end > pos:
                flat[pos:end] = 1
            pos = end
            val = 1 - val
            if pos >= flat.size:
                break
        mask = flat.reshape((height, width), order="F")
    elif isinstance(counts, (str, bytes)):
        try:
            from pycocotools import mask as coco_mask_util  # type: ignore

            rle_payload = {
                "size": [height, width],
                "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
            }
            decoded = coco_mask_util.decode(rle_payload)
            mask = np.asarray(decoded, dtype=np.uint8)
        except Exception:
            return []
    else:
        return []

    if mask is None or mask.size == 0:
        return []
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[Tuple[float, float]]] = []
    for contour in contours:
        pts = contour.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        polygons.append([(float(x), float(y)) for x, y in pts])
    return polygons


def _extract_coco_segmentation_polygon(
    ann: Dict[str, Any],
) -> List[Tuple[float, float]]:
    """Extract a single merged polygon (pixel coordinates) from a COCO annotation."""
    seg = ann.get("segmentation")
    segments: List[List[Tuple[float, float]]] = []
    if isinstance(seg, list):
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts: List[Tuple[float, float]] = []
            try:
                for idx in range(0, len(poly) - 1, 2):
                    pts.append((float(poly[idx]), float(poly[idx + 1])))
            except Exception:
                continue
            if len(pts) >= 3:
                segments.append(pts)
    elif isinstance(seg, dict):
        segments = _decode_coco_rle_to_polygons(seg)

    if segments:
        return _merge_multi_segment(segments)

    # Fallback: convert bbox to a 4-corner polygon so mixed annotations stay valid YOLO-seg.
    bbox = ann.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            x, y, bw, bh = [float(v) for v in bbox[:4]]
        except Exception:
            return []
        if bw > 0.0 and bh > 0.0:
            return [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
    return []


def _convert_coco_split(
    *,
    split_name: str,
    ann_path: Path,
    image_root: Path,
    root_path: Path,
    output_dir: Path,
    category_id_to_index: Dict[int, int],
    include_image_ids: Optional[set[int]],
    link_mode: str,
    segmentation: bool = False,
) -> int:
    payload = _load_json_dict(ann_path)
    images = payload.get("images")
    annotations = payload.get("annotations")
    images_list = images if isinstance(images, list) else []
    ann_list = annotations if isinstance(annotations, list) else []

    images_by_id: Dict[int, Dict[str, Any]] = {}
    for image_rec in images_list:
        if not isinstance(image_rec, dict):
            continue
        image_id = _safe_int(image_rec.get("id"))
        if image_id is None:
            continue
        images_by_id[image_id] = image_rec

    ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in ann_list:
        if not isinstance(ann, dict):
            continue
        image_id = _safe_int(ann.get("image_id"))
        if image_id is None:
            continue
        ann_by_image.setdefault(image_id, []).append(ann)

    (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    written = 0
    for image_id in sorted(images_by_id.keys()):
        if include_image_ids is not None and int(image_id) not in include_image_ids:
            continue
        image_rec = images_by_id[image_id]
        file_name = str(image_rec.get("file_name") or "").strip()
        if not file_name:
            continue
        src_image = _resolve_coco_image_path(
            file_name=file_name,
            root_path=root_path,
            image_root=image_root,
            ann_path=ann_path,
        )
        if src_image is None:
            continue

        width = _safe_int(image_rec.get("width")) or 0
        height = _safe_int(image_rec.get("height")) or 0
        if width <= 0 or height <= 0:
            try:
                with Image.open(src_image) as pil:
                    width, height = pil.size
            except Exception:
                continue
        if width <= 0 or height <= 0:
            continue

        stem = f"{Path(file_name).stem}__{int(image_id)}"
        dst_image = (
            output_dir / "images" / split_name / f"{stem}{src_image.suffix or '.jpg'}"
        )
        _install_file(src_image, dst_image, mode=link_mode)
        dst_label = output_dir / "labels" / split_name / f"{stem}.txt"

        lines: List[str] = []
        for ann in ann_by_image.get(image_id, []):
            if ann.get("iscrowd") and not segmentation:
                continue
            category_id = _safe_int(ann.get("category_id"))
            if category_id is None:
                continue
            cls_idx = int(category_id_to_index.get(category_id, 0))

            if segmentation:
                poly_pts = _extract_coco_segmentation_polygon(ann)
                if len(poly_pts) < 3:
                    continue
                coords: List[str] = []
                for px, py in poly_pts:
                    nx = min(1.0, max(0.0, px / float(width)))
                    ny = min(1.0, max(0.0, py / float(height)))
                    coords.append(f"{nx:.6f} {ny:.6f}")
                lines.append(f"{cls_idx} " + " ".join(coords))
            else:
                bbox = ann.get("bbox")
                if not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                try:
                    x, y, bw, bh = [float(v) for v in bbox[:4]]
                except Exception:
                    continue
                if bw <= 0.0 or bh <= 0.0:
                    continue
                cx = min(1.0, max(0.0, (x + bw / 2.0) / float(width)))
                cy = min(1.0, max(0.0, (y + bh / 2.0) / float(height)))
                nw = min(1.0, max(0.0, bw / float(width)))
                nh = min(1.0, max(0.0, bh / float(height)))
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dst_label.write_text(
            ("\n".join(lines) + ("\n" if lines else "")),
            encoding="utf-8",
        )
        written += 1
    return int(written)


def _convert_coco_detection_split(
    *,
    split_name: str,
    ann_path: Path,
    image_root: Path,
    root_path: Path,
    output_dir: Path,
    category_id_to_index: Dict[int, int],
    include_image_ids: Optional[set[int]],
    link_mode: str,
) -> int:
    return _convert_coco_split(
        split_name=split_name,
        ann_path=ann_path,
        image_root=image_root,
        root_path=root_path,
        output_dir=output_dir,
        category_id_to_index=category_id_to_index,
        include_image_ids=include_image_ids,
        link_mode=link_mode,
        segmentation=False,
    )


def _split_image_ids_for_val(
    ann_path: Path,
    *,
    val_split: float,
    seed: int,
) -> Tuple[set[int], set[int]]:
    payload = _load_json_dict(ann_path)
    images = payload.get("images")
    image_ids = sorted(
        {
            int(image_id)
            for image_id in (
                _safe_int(rec.get("id")) if isinstance(rec, dict) else None
                for rec in (images if isinstance(images, list) else [])
            )
            if image_id is not None
        }
    )
    if len(image_ids) <= 1 or float(val_split) <= 0.0:
        return set(image_ids), set()

    val_count = int(round(float(len(image_ids)) * float(val_split)))
    val_count = max(1, min(len(image_ids) - 1, val_count))
    rng = random.Random(int(seed))
    val_ids = set(rng.sample(image_ids, val_count))
    train_ids = {
        int(image_id) for image_id in image_ids if int(image_id) not in val_ids
    }
    return train_ids, val_ids


def _resolve_coco_image_path(
    *,
    file_name: str,
    root_path: Path,
    image_root: Path,
    ann_path: Optional[Path] = None,
) -> Optional[Path]:
    raw = Path(file_name)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if ann_path is not None:
            candidates.append(ann_path.parent / raw)
            candidates.append(ann_path.parent / raw.name)
        candidates.append(image_root / raw)
        candidates.append(root_path / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_categories(
    ann_path: Optional[Path], *, default_name: str
) -> Tuple[List[str], Dict[int, int]]:
    if ann_path is None or not ann_path.exists():
        return [default_name], {1: 0}
    payload = _load_json_dict(ann_path)
    categories = payload.get("categories")
    names: List[str] = []
    category_id_to_index: Dict[int, int] = {}
    for category in categories if isinstance(categories, list) else []:
        if not isinstance(category, dict):
            continue
        category_id = _safe_int(category.get("id"))
        if category_id is None or category_id in category_id_to_index:
            continue
        category_id_to_index[category_id] = len(names)
        names.append(str(category.get("name") or f"class_{category_id}"))
    if not names:
        names = [default_name]
        category_id_to_index = {1: 0}
    return names, category_id_to_index


def _load_json_dict(path: Path) -> Dict[str, Any]:
    return load_coco_json(path)


def _install_file(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)
