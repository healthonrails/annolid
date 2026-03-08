from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from annolid.datasets.coco import materialize_coco_spec_as_yolo
from annolid.segmentation.dino_kpseg.data import (
    _image_to_labelme_path,
    _resolve_split_images,
    load_labelme_pose_spec,
    load_yolo_pose_spec,
)
from annolid.segmentation.dino_kpseg.format_utils import (
    normalize_dino_kpseg_data_format,
)


@dataclass(frozen=True)
class ResolvedPoseDataset:
    data_yaml: Path
    source_yaml: Path
    staged_yolo_yaml: Optional[Path]
    data_format: str
    label_format: str
    train_images: List[Path]
    val_images: List[Path]
    train_label_paths: Optional[List[Path]]
    val_label_paths: Optional[List[Path]]
    keypoint_names: Optional[List[str]]
    flip_idx: Optional[List[int]]
    kpt_count: int
    kpt_dims: int
    raw_train_entry: object
    raw_val_entry: object
    test_images: List[Path] = field(default_factory=list)
    test_label_paths: Optional[List[Path]] = None
    raw_test_entry: object = None

    def split_images(self, split: str) -> List[Path]:
        if split == "train":
            return list(self.train_images)
        if split == "val":
            return list(self.val_images)
        if split == "test":
            return list(self.test_images)
        raise ValueError("split must be 'train', 'val', or 'test'")

    def split_labels(self, split: str) -> Optional[List[Path]]:
        if split == "train":
            if self.train_label_paths is None:
                return None
            return list(self.train_label_paths)
        if split == "val":
            if self.val_label_paths is None:
                return None
            return list(self.val_label_paths)
        if split == "test":
            if self.test_label_paths is None:
                return None
            return list(self.test_label_paths)
        raise ValueError("split must be 'train', 'val', or 'test'")


def _resolve_root_path(
    source_yaml: Path, source_payload: Dict[str, Any]
) -> Optional[Path]:
    root = source_payload.get("path")
    if not root:
        return source_yaml.parent
    root_path = Path(str(root)).expanduser()
    if root_path.is_absolute():
        return root_path
    return (source_yaml.parent / root_path).resolve()


def _resolve_test_images(
    *,
    source_yaml: Path,
    source_payload: Dict[str, Any],
    raw_test_entry: object,
) -> List[Path]:
    if not raw_test_entry:
        return []
    root_path = _resolve_root_path(source_yaml, source_payload)
    return list(
        _resolve_split_images(
            raw_test_entry,
            yaml_path=source_yaml,
            root_path=root_path,
        )
    )


def _guess_labelme_json_paths(images: List[Path]) -> List[Path]:
    paths: List[Path] = []
    for img_path in images:
        json_path = _image_to_labelme_path(Path(img_path))
        paths.append(json_path if json_path.exists() else json_path)
    return paths


def resolve_pose_dataset(
    *,
    data_yaml: Path,
    data_format: str = "auto",
    coco_staging_dir: Optional[Path] = None,
    coco_temp_prefix: str = "dino_kpseg_coco_",
) -> ResolvedPoseDataset:
    requested_data_format = str(data_format or "auto").strip().lower()
    if requested_data_format not in {"auto", "yolo", "labelme", "coco"}:
        raise ValueError(f"Unsupported data_format: {requested_data_format!r}")

    payload = _read_yaml_dict(Path(data_yaml))
    data_format_norm = normalize_dino_kpseg_data_format(
        payload,
        data_format=requested_data_format,
    )

    source_yaml = Path(data_yaml)
    staged_yolo_yaml: Optional[Path] = None
    label_format = "yolo"
    if data_format_norm == "coco":
        if coco_staging_dir is None:
            staging_root = Path(
                tempfile.mkdtemp(
                    prefix=str(coco_temp_prefix), dir=tempfile.gettempdir()
                )
            )
        else:
            staging_root = Path(coco_staging_dir).resolve()
            if staging_root.exists():
                shutil.rmtree(staging_root)
        staged_yolo_yaml = materialize_coco_spec_as_yolo(
            config_path=source_yaml,
            output_dir=staging_root,
            expected_task="pose",
        )
        source_yaml = staged_yolo_yaml
        label_format = "yolo"

    source_payload = _read_yaml_dict(source_yaml)
    raw_train_entry = source_payload.get("train")
    raw_val_entry = source_payload.get("val")
    raw_test_entry = source_payload.get("test")
    test_images = _resolve_test_images(
        source_yaml=source_yaml,
        source_payload=source_payload,
        raw_test_entry=raw_test_entry,
    )

    if data_format_norm == "labelme":
        spec_lm = load_labelme_pose_spec(source_yaml)
        return ResolvedPoseDataset(
            data_yaml=Path(data_yaml),
            source_yaml=source_yaml,
            staged_yolo_yaml=staged_yolo_yaml,
            data_format=data_format_norm,
            label_format="labelme",
            train_images=list(spec_lm.train_images),
            val_images=list(spec_lm.val_images),
            train_label_paths=list(spec_lm.train_json),
            val_label_paths=list(spec_lm.val_json),
            test_images=list(test_images),
            test_label_paths=_guess_labelme_json_paths(test_images)
            if test_images
            else [],
            keypoint_names=list(spec_lm.keypoint_names),
            flip_idx=spec_lm.flip_idx,
            kpt_count=int(spec_lm.kpt_count),
            kpt_dims=int(spec_lm.kpt_dims),
            raw_train_entry=raw_train_entry,
            raw_val_entry=raw_val_entry,
            raw_test_entry=raw_test_entry,
        )

    spec = load_yolo_pose_spec(source_yaml)
    return ResolvedPoseDataset(
        data_yaml=Path(data_yaml),
        source_yaml=source_yaml,
        staged_yolo_yaml=staged_yolo_yaml,
        data_format=data_format_norm,
        label_format=label_format,
        train_images=list(spec.train_images),
        val_images=list(spec.val_images),
        train_label_paths=None,
        val_label_paths=None,
        test_images=list(test_images),
        test_label_paths=None,
        keypoint_names=(list(spec.keypoint_names) if spec.keypoint_names else None),
        flip_idx=spec.flip_idx,
        kpt_count=int(spec.kpt_count),
        kpt_dims=int(spec.kpt_dims),
        raw_train_entry=raw_train_entry,
        raw_val_entry=raw_val_entry,
        raw_test_entry=raw_test_entry,
    )


def _read_yaml_dict(path: Path) -> Dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return {}
    return dict(payload)
