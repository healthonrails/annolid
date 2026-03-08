from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image

from annolid.segmentation.dino_kpseg.data import (
    load_coco_pose_spec,
    materialize_coco_pose_as_yolo,
)


_COCO_SPLIT_FILENAMES = {
    "train": ("train.json", "instances_train.json", "person_keypoints_train.json"),
    "val": ("val.json", "instances_val.json", "person_keypoints_val.json"),
    "test": ("test.json", "instances_test.json", "person_keypoints_test.json"),
}


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

    depth = max(1, int(max_depth))
    for split_names in _COCO_SPLIT_FILENAMES.values():
        for name in split_names:
            pattern = "/".join(["*"] * depth + [name])
            for json_path in candidate.glob(pattern):
                parent = json_path.parent
                if (
                    parent.is_dir()
                    and build_coco_spec_from_annotations_dir(parent) is not None
                ):
                    return parent
    return None


def infer_coco_task(*, config_path: Path, payload: Dict[str, Any]) -> str:
    if (
        payload.get("kpt_shape")
        or payload.get("keypoint_names")
        or payload.get("flip_idx")
    ):
        return "pose"

    ann_paths = resolve_coco_annotation_paths(config_path=config_path, payload=payload)
    for ann_path in ann_paths.values():
        if ann_path is None or not ann_path.exists():
            continue
        ann_payload = _load_json_dict(ann_path)
        categories = ann_payload.get("categories")
        for category in categories if isinstance(categories, list) else []:
            if isinstance(category, dict) and isinstance(
                category.get("keypoints"), list
            ):
                return "pose"
        annotations = ann_payload.get("annotations")
        for ann in annotations if isinstance(annotations, list) else []:
            keypoints = ann.get("keypoints") if isinstance(ann, dict) else None
            if isinstance(keypoints, list) and len(keypoints) >= 3:
                return "pose"
    return "detect"


def materialize_coco_spec_as_yolo(
    *,
    config_path: Path,
    output_dir: Optional[Path] = None,
    link_mode: str = "hardlink",
    expected_task: Optional[str] = None,
) -> Path:
    payload = read_yaml_dict(config_path)
    task = infer_coco_task(config_path=config_path, payload=payload)
    if expected_task is not None and task != str(expected_task).strip().lower():
        raise ValueError(
            f"Expected a COCO {expected_task} dataset, but inferred {task} from {config_path}"
        )

    target_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(tempfile.mkdtemp(prefix=f"{config_path.stem}_coco_yolo_")).resolve()
    )

    if task == "pose":
        spec = load_coco_pose_spec(config_path)
        return materialize_coco_pose_as_yolo(
            spec=spec,
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
        counts["train"] = _convert_coco_detection_split(
            split_name="train",
            ann_path=ann_paths["train"],
            image_root=image_root_path,
            root_path=root_path,
            output_dir=output_dir,
            category_id_to_index=category_id_to_index,
            include_image_ids=train_ids,
            link_mode=link_mode,
        )
        counts["val"] = _convert_coco_detection_split(
            split_name="val",
            ann_path=ann_paths["train"],
            image_root=image_root_path,
            root_path=root_path,
            output_dir=output_dir,
            category_id_to_index=category_id_to_index,
            include_image_ids=val_ids,
            link_mode=link_mode,
        )
        auto_val_split_used = counts["val"] > 0
    else:
        for split in ("train", "val", "test"):
            ann_path = ann_paths.get(split)
            if ann_path is None:
                continue
            counts[split] = _convert_coco_detection_split(
                split_name=split,
                ann_path=ann_path,
                image_root=image_root_path,
                root_path=root_path,
                output_dir=output_dir,
                category_id_to_index=category_id_to_index,
                include_image_ids=None,
                link_mode=link_mode,
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
            if ann.get("iscrowd"):
                continue
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            try:
                x, y, bw, bh = [float(v) for v in bbox[:4]]
            except Exception:
                continue
            if bw <= 0.0 or bh <= 0.0:
                continue
            category_id = _safe_int(ann.get("category_id"))
            if category_id is None:
                continue
            cls_idx = int(category_id_to_index.get(category_id, 0))

            cx = (x + bw / 2.0) / float(width)
            cy = (y + bh / 2.0) / float(height)
            nw = bw / float(width)
            nh = bh / float(height)
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dst_label.write_text(
            ("\n".join(lines) + ("\n" if lines else "")),
            encoding="utf-8",
        )
        written += 1
    return int(written)


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
) -> Optional[Path]:
    raw = Path(file_name)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
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
