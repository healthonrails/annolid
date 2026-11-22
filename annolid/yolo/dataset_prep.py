from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from annolid.datasets.coco import (
    build_coco_spec_from_annotations_dir,
    build_coco_spec_from_dataset_path,
    infer_coco_task,
    iter_existing_paths,
    load_coco_class_names,
    load_coco_json,
    load_coco_keypoint_meta,
    looks_like_coco_spec,
    materialize_coco_detection_as_yolo,
    materialize_coco_segmentation_as_yolo,
    materialize_coco_spec_as_yolo,
    read_yaml_dict,
    resolve_coco_annotation_paths,
    resolve_coco_root_path,
    resolve_dataset_path,
)
from annolid.utils.logger import logger

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class YOLODatasetError(ValueError):
    """Structured dataset validation/preparation error with UI-friendly title & message."""

    def __init__(self, title: str, message: str, *, critical: bool = False) -> None:
        super().__init__(message)
        self.title = str(title)
        self.message = str(message)
        self.critical = bool(critical)


@dataclass(frozen=True)
class YOLODatasetSummary:
    """Inspection summary for a YOLO or COCO dataset path."""

    source_path: Path
    format_kind: str  # "yolo" | "coco" | "coco_dir" | "unknown"
    task: str  # "pose" | "seg" | "detect" | "unknown"
    class_names: Tuple[str, ...] = ()
    train_images: int = 0
    val_images: int = 0
    test_images: int = 0
    kpt_shape: Optional[Tuple[int, ...]] = None
    flip_idx: Optional[Tuple[int, ...]] = None
    issues: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0


@dataclass
class PreparedYOLODataset:
    """Resolved YOLO data.yaml plus any temporary staging files/dirs created for training."""

    data_yaml: Path
    data_cfg: Dict[str, Any]
    temp_configs: List[Path] = field(default_factory=list)
    temp_dataset_dirs: List[Path] = field(default_factory=list)

    def cleanup(self) -> None:
        while self.temp_configs:
            temp_path = self.temp_configs.pop()
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Could not remove temporary YOLO config: %s", temp_path)
        while self.temp_dataset_dirs:
            dataset_dir = self.temp_dataset_dirs.pop()
            try:
                for child in sorted(dataset_dir.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                dataset_dir.rmdir()
            except FileNotFoundError:
                continue
            except Exception:
                logger.debug(
                    "Could not remove temporary YOLO dataset dir: %s", dataset_dir
                )

    def __enter__(self) -> "PreparedYOLODataset":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()


def is_remote_dataset_entry(path_str: str) -> bool:
    return str(path_str or "").startswith(("http://", "https://", "rtsp://", "rtmp://"))


def normalize_class_names(value: Any) -> Optional[List[str]]:
    if isinstance(value, str):
        label = value.strip()
        return [label] if label else None
    if isinstance(value, (list, tuple)):
        names = [str(item).strip() for item in value if str(item).strip()]
        return names if names else None
    if isinstance(value, dict):
        entries: List[Tuple[int, str]] = []
        for key, name in value.items():
            label = str(name).strip()
            if not label:
                continue
            try:
                key_i = int(key)
            except Exception:
                continue
            entries.append((key_i, label))
        if not entries:
            return None
        entries.sort(key=lambda item: item[0])
        return [label for _, label in entries]
    return None


def infer_flip_idx(data_cfg: Dict[str, Any]) -> Optional[List[int]]:
    """Infer left/right symmetric keypoint swap indices from kpt_labels or kpt_names."""
    kpt_shape = data_cfg.get("kpt_shape")
    if not isinstance(kpt_shape, (list, tuple)) or len(kpt_shape) < 1:
        return None
    try:
        nkpt = int(kpt_shape[0])
    except Exception:
        return None
    if nkpt <= 0:
        return None

    labels_by_idx: Dict[int, str] = {}
    kpt_labels = data_cfg.get("kpt_labels")
    if isinstance(kpt_labels, dict) and kpt_labels:
        for key, value in kpt_labels.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if idx < 0 or idx >= nkpt:
                continue
            label = str(value or "").strip()
            if label:
                labels_by_idx[idx] = label

    if len(labels_by_idx) != nkpt:
        kpt_names = data_cfg.get("kpt_names")
        names_list: Optional[List[Any]] = None
        if isinstance(kpt_names, list):
            names_list = kpt_names
        elif isinstance(kpt_names, dict) and kpt_names:
            raw = (
                kpt_names.get(0)
                if 0 in kpt_names
                else kpt_names.get("0", next(iter(kpt_names.values())))
            )
            if isinstance(raw, list):
                names_list = raw
        if names_list and len(names_list) == nkpt:
            labels_by_idx = {
                idx: str(val or "").strip()
                for idx, val in enumerate(names_list)
                if str(val or "").strip()
            }

    if len(labels_by_idx) != nkpt:
        return None

    idx_by_label = {lbl.lower(): i for i, lbl in labels_by_idx.items()}

    def counterpart(label: str) -> Optional[str]:
        s = label.lower()
        swaps = [
            ("left", "right"),
            ("right", "left"),
            ("l_", "r_"),
            ("r_", "l_"),
            ("l-", "r-"),
            ("r-", "l-"),
        ]
        for a, b in swaps:
            if a in s:
                return s.replace(a, b)
        return None

    flip_idx: List[int] = []
    for i in range(nkpt):
        label = labels_by_idx[i]
        other = counterpart(label)
        if other and other in idx_by_label:
            flip_idx.append(idx_by_label[other])
        else:
            flip_idx.append(i)

    if len(flip_idx) != nkpt:
        return None
    return flip_idx


def count_dataset_images(
    entry: Union[str, List[str], Tuple[str, ...], Path, None],
    *,
    root_dir: Optional[Path] = None,
) -> int:
    """Count images referenced by a YOLO split entry (directory, .txt list, or list of entries)."""
    if not entry:
        return 0

    if isinstance(entry, (list, tuple)):
        return sum(count_dataset_images(item, root_dir=root_dir) for item in entry)

    path_str = str(entry).strip()
    if not path_str or is_remote_dataset_entry(path_str):
        return 0
    if any(ch in path_str for ch in ("*", "?", "[")):
        return 0

    p = Path(path_str).expanduser()
    if not p.is_absolute() and root_dir is not None:
        p = root_dir / p
    try:
        if p.is_dir():
            return sum(1 for f in p.rglob("*") if f.suffix.lower() in _IMAGE_EXTENSIONS)
        if p.is_file() and p.suffix.lower() == ".txt":
            count = 0
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                candidate = Path(line)
                if not candidate.is_absolute():
                    candidate = p.parent / candidate
                if candidate.exists():
                    count += 1
            return count
    except OSError:
        return 0
    return 0


def _resolve_root_dir(data_cfg: Dict[str, Any], base_dir: Path) -> Path:
    raw_root = data_cfg.get("path")
    if raw_root:
        root_candidate = Path(str(raw_root)).expanduser()
        if not root_candidate.is_absolute():
            root_candidate = (base_dir / root_candidate).resolve()
        return root_candidate
    return base_dir


def _resolve_split_entry(entry: Any, root_dir: Path) -> Any:
    if isinstance(entry, str):
        if not entry.strip() or is_remote_dataset_entry(entry):
            return entry
        entry_path = Path(entry).expanduser()
        resolved = entry_path if entry_path.is_absolute() else root_dir / entry_path
        return str(resolved.resolve())
    if isinstance(entry, (list, tuple)):
        return [_resolve_split_entry(item, root_dir) for item in entry]
    return entry


def _infer_yolo_task_from_labels(
    data_cfg: Dict[str, Any],
    *,
    root_dir: Path,
    sample_limit: int = 25,
) -> str:
    """Infer YOLO dataset task ('pose', 'seg', or 'detect') from YAML metadata + label files."""
    if data_cfg.get("kpt_shape"):
        return "pose"

    candidate_label_dirs: List[Path] = []
    for split in ("train", "val", "test"):
        val = data_cfg.get(split)
        entries = val if isinstance(val, (list, tuple)) else [val]
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                continue
            p = Path(entry).expanduser()
            if not p.is_absolute():
                p = root_dir / p
            parts = list(p.parts)
            if "images" in parts:
                idx = len(parts) - 1 - parts[::-1].index("images")
                parts[idx] = "labels"
                candidate_label_dirs.append(Path(*parts))
            candidate_label_dirs.append(p.parent / "labels")
    candidate_label_dirs.extend([root_dir / "labels" / "train", root_dir / "labels"])

    seen: set[Path] = set()
    checked_files = 0
    saw_polygon = False
    saw_box = False

    for label_dir in candidate_label_dirs:
        try:
            resolved = label_dir.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        try:
            txt_files = sorted(resolved.rglob("*.txt"))
        except OSError:
            continue
        for txt_path in txt_files:
            if checked_files >= sample_limit:
                break
            try:
                content = txt_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in content.splitlines():
                tokens = line.strip().split()
                if len(tokens) < 5:
                    continue
                checked_files += 1
                if len(tokens) == 5:
                    saw_box = True
                elif len(tokens) >= 7 and (len(tokens) - 1) % 2 == 0:
                    saw_polygon = True
                if checked_files >= sample_limit:
                    break
        if saw_polygon:
            return "seg"

    if saw_polygon:
        return "seg"
    if any(
        "segment" in str(v).lower() or "-seg" in str(v).lower()
        for v in data_cfg.values()
    ):
        return "seg"
    if saw_box:
        return "detect"
    return "detect"


def _infer_class_names_from_coco(
    *,
    root_dir: Path,
    config_path: Path,
) -> Optional[List[str]]:
    candidates: List[Path] = [root_dir, config_path.parent]
    seen: set[Path] = set()
    for candidate_root in candidates:
        if candidate_root in seen:
            continue
        seen.add(candidate_root)
        payload = build_coco_spec_from_dataset_path(candidate_root)
        if not payload:
            continue
        ann_paths = resolve_coco_annotation_paths(
            config_path=candidate_root / "auto_discovered_coco_spec.yaml",
            payload=payload,
        )
        for split in ("train", "val", "test"):
            ann_path = ann_paths.get(split)
            if ann_path is None or not ann_path.exists():
                continue
            names = [name for name in load_coco_class_names(ann_path) if name]
            if names:
                return names
    return None


def _ensure_required_class_metadata(
    data_cfg: Dict[str, Any],
    *,
    root_dir: Path,
    config_path: Path,
) -> None:
    names = normalize_class_names(data_cfg.get("names"))
    if names:
        data_cfg["names"] = names
        data_cfg["nc"] = len(names)
        return

    try:
        nc = int(data_cfg.get("nc"))
    except Exception:
        nc = 0
    if nc > 0:
        data_cfg["names"] = [f"class_{i}" for i in range(nc)]
        data_cfg["nc"] = nc
        return

    kpt_names = data_cfg.get("kpt_names")
    if isinstance(kpt_names, dict) and kpt_names:
        class_count = len(kpt_names)
        if class_count > 0:
            data_cfg["names"] = [f"class_{i}" for i in range(class_count)]
            data_cfg["nc"] = class_count
            return

    coco_names = _infer_class_names_from_coco(
        root_dir=root_dir,
        config_path=config_path,
    )
    if coco_names:
        data_cfg["names"] = coco_names
        data_cfg["nc"] = len(coco_names)
        return

    # Ultralytics requires at least one class definition for all training YAMLs.
    data_cfg["names"] = ["animal"]
    data_cfg["nc"] = 1


def _fill_missing_splits(
    data_cfg: Dict[str, Any],
    *,
    root_dir: Path,
) -> List[str]:
    missing = [split for split in ("train", "val") if not data_cfg.get(split)]
    if not missing:
        return []

    guessed = {
        "train": [
            root_dir / "images" / "train",
            root_dir / "train",
        ],
        "val": [
            root_dir / "images" / "val",
            root_dir / "val",
            root_dir / "images" / "valid",
            root_dir / "valid",
            root_dir / "images" / "validation",
            root_dir / "validation",
        ],
    }
    for split in list(missing):
        for candidate in guessed[split]:
            if candidate.exists():
                data_cfg[split] = str(candidate.resolve())
                missing.remove(split)
                break
    return missing


def write_temp_yolo_yaml(data_cfg: Dict[str, Any], anchor_path: Path) -> Path:
    """Persist a temporary resolved YOLO YAML beside anchor_path."""
    parent_dir = anchor_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f"{anchor_path.stem}_resolved_",
        suffix=".yaml",
        dir=str(parent_dir),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data_cfg, handle, sort_keys=False)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temp_path).unlink(missing_ok=True)
        raise
    return Path(temp_path)


def _resolve_directory_input(
    directory: Path,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Resolve a user-supplied folder into either a YOLO data.yaml or a COCO spec payload."""
    for candidate in (
        directory / "data.yaml",
        directory / "YOLO_dataset" / "data.yaml",
    ):
        if candidate.is_file():
            return candidate.resolve(), None
    coco_payload = build_coco_spec_from_dataset_path(directory)
    if coco_payload is not None:
        return None, coco_payload
    return None, None


def inspect_yolo_dataset(path: Union[str, Path]) -> YOLODatasetSummary:
    """Inspect a YOLO data.yaml, COCO spec YAML, or dataset directory without mutating disk."""
    raw = str(path or "").strip()
    if not raw:
        return YOLODatasetSummary(
            source_path=Path(),
            format_kind="unknown",
            task="unknown",
            issues=("Dataset path is empty.",),
        )

    target = Path(raw).expanduser()
    try:
        target = target.resolve()
    except OSError:
        pass

    if not target.exists():
        return YOLODatasetSummary(
            source_path=target,
            format_kind="unknown",
            task="unknown",
            issues=(f"Dataset path not found: {target}",),
        )

    if target.is_dir():
        yaml_candidate, coco_payload = _resolve_directory_input(target)
        if yaml_candidate is not None:
            target = yaml_candidate
        elif coco_payload is not None:
            return _inspect_coco_payload(
                source_path=target,
                config_path=target / "coco_spec.yaml",
                payload=coco_payload,
                format_kind="coco_dir",
            )
        else:
            return YOLODatasetSummary(
                source_path=target,
                format_kind="unknown",
                task="unknown",
                issues=(
                    "Folder does not contain data.yaml or recognizable COCO train/val annotations.",
                ),
            )

    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return YOLODatasetSummary(
            source_path=target,
            format_kind="unknown",
            task="unknown",
            issues=(f"Failed to read YAML: {exc}",),
        )

    if not isinstance(loaded, dict):
        return YOLODatasetSummary(
            source_path=target,
            format_kind="unknown",
            task="unknown",
            issues=("Dataset configuration must be a YAML mapping.",),
        )

    if looks_like_coco_spec(loaded):
        return _inspect_coco_payload(
            source_path=target,
            config_path=target,
            payload=loaded,
            format_kind="coco",
        )

    data_cfg = dict(loaded)
    root_dir = _resolve_root_dir(data_cfg, target.parent)
    for split in ("train", "val", "test"):
        if data_cfg.get(split):
            data_cfg[split] = _resolve_split_entry(data_cfg[split], root_dir)

    missing_splits = _fill_missing_splits(data_cfg, root_dir=root_dir)
    issues: List[str] = []
    warnings: List[str] = []
    if missing_splits:
        issues.append(f"Missing required splits: {', '.join(missing_splits)}")

    for split in ("train", "val", "test"):
        value = data_cfg.get(split)
        if not value:
            continue
        entries = value if isinstance(value, list) else [value]
        for entry in entries:
            if isinstance(entry, str) and not is_remote_dataset_entry(entry):
                if any(ch in entry for ch in ("*", "?", "[")):
                    continue
                if not Path(entry).exists():
                    issues.append(f"Missing {split} path: {entry}")

    names = normalize_class_names(data_cfg.get("names")) or []
    if not names:
        try:
            nc = int(data_cfg.get("nc") or 0)
        except Exception:
            nc = 0
        if nc > 0:
            names = [f"class_{i}" for i in range(nc)]

    train_count = count_dataset_images(data_cfg.get("train"), root_dir=root_dir)
    val_count = count_dataset_images(data_cfg.get("val"), root_dir=root_dir)
    test_count = count_dataset_images(data_cfg.get("test"), root_dir=root_dir)
    task = _infer_yolo_task_from_labels(data_cfg, root_dir=root_dir)

    kpt_shape_raw = data_cfg.get("kpt_shape")
    kpt_shape = (
        tuple(int(x) for x in kpt_shape_raw)
        if isinstance(kpt_shape_raw, (list, tuple)) and kpt_shape_raw
        else None
    )
    flip_raw = data_cfg.get("flip_idx") or (
        infer_flip_idx(data_cfg) if kpt_shape else None
    )
    flip_idx = (
        tuple(int(x) for x in flip_raw)
        if isinstance(flip_raw, (list, tuple)) and flip_raw
        else None
    )
    if train_count and train_count < 50:
        warnings.append(
            f"Small training set ({train_count} images); consider the 'Small Lab Dataset (Fine-tune)' preset."
        )
    if val_count and val_count < 10:
        warnings.append(
            f"Small validation set ({val_count} images); validation metrics may be noisy."
        )

    return YOLODatasetSummary(
        source_path=target,
        format_kind="yolo",
        task=task,
        class_names=tuple(names),
        train_images=train_count,
        val_images=val_count,
        test_images=test_count,
        kpt_shape=kpt_shape,
        flip_idx=flip_idx,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _inspect_coco_payload(
    *,
    source_path: Path,
    config_path: Path,
    payload: Dict[str, Any],
    format_kind: str,
) -> YOLODatasetSummary:
    ann_paths = resolve_coco_annotation_paths(config_path=config_path, payload=payload)
    issues: List[str] = []
    warnings: List[str] = []

    train_ann = ann_paths.get("train")
    val_ann = ann_paths.get("val")
    test_ann = ann_paths.get("test")
    if train_ann is None or not train_ann.exists():
        issues.append("COCO training annotations JSON not found.")

    names: List[str] = []
    train_images = 0
    val_images = 0
    test_images = 0
    has_seg = False
    kpt_shape: Optional[Tuple[int, ...]] = None

    for split_name, ann_path in (
        ("train", train_ann),
        ("val", val_ann),
        ("test", test_ann),
    ):
        if ann_path is None or not ann_path.exists():
            continue
        try:
            coco_data = load_coco_json(ann_path)
        except Exception as exc:
            issues.append(f"Invalid COCO {split_name} JSON: {exc}")
            continue
        img_count = len(coco_data.get("images") or [])
        if split_name == "train":
            train_images = img_count
        elif split_name == "val":
            val_images = img_count
        else:
            test_images = img_count
        if not names:
            names = [n for n in load_coco_class_names(ann_path) if n]
        if kpt_shape is None:
            meta = load_coco_keypoint_meta(ann_path)
            nkpt = int(meta.get("num_keypoints") or 0)
            if nkpt > 0:
                kpt_shape = (nkpt, 3)
        if not has_seg:
            for ann in (coco_data.get("annotations") or [])[:50]:
                if isinstance(ann, dict) and ann.get("segmentation"):
                    has_seg = True
                    break

    inferred = infer_coco_task(config_path=config_path, payload=payload)
    if inferred == "pose":
        task = "pose"
    elif has_seg:
        task = "seg"
    else:
        task = "detect"

    if train_images and val_images == 0:
        warnings.append(
            "No val.json found; Annolid will auto-split validation from train."
        )

    return YOLODatasetSummary(
        source_path=source_path,
        format_kind=format_kind,
        task=task,
        class_names=tuple(names),
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        kpt_shape=kpt_shape,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _upgrade_from_nearby_coco(
    *,
    config_path: Path,
    data_cfg: Dict[str, Any],
    expected_task: Optional[str],
    temp_configs: List[Path],
) -> Tuple[Path, Dict[str, Any]]:
    task = str(expected_task or "").strip().lower()
    if task != "pose":
        return config_path, data_cfg
    if data_cfg.get("kpt_shape") or looks_like_coco_spec(data_cfg):
        return config_path, data_cfg

    candidate_roots: List[Path] = [config_path.parent]
    raw_root = data_cfg.get("path")
    if raw_root:
        root_candidate = Path(str(raw_root)).expanduser()
        if not root_candidate.is_absolute():
            root_candidate = (config_path.parent / root_candidate).resolve()
        candidate_roots.append(root_candidate)

    seen: set[Path] = set()
    for candidate_root in candidate_roots:
        if candidate_root in seen or not candidate_root.exists():
            continue
        seen.add(candidate_root)
        payload = build_coco_spec_from_dataset_path(candidate_root)
        if not payload:
            continue
        ann_paths = resolve_coco_annotation_paths(
            config_path=candidate_root / "auto_pose_spec.yaml",
            payload=payload,
        )
        keypoint_ann = next(
            (
                ann_path
                for ann_path in ann_paths.values()
                if ann_path is not None and ann_path.exists()
            ),
            None,
        )
        if keypoint_ann is None:
            continue
        keypoint_meta = load_coco_keypoint_meta(keypoint_ann)
        if int(keypoint_meta.get("num_keypoints") or 0) <= 0:
            continue

        temp_spec_path = write_temp_yolo_yaml(
            payload,
            candidate_root / "auto_pose_spec.yaml",
        )
        temp_configs.append(temp_spec_path)
        try:
            upgraded_cfg = (
                yaml.safe_load(temp_spec_path.read_text(encoding="utf-8")) or {}
            )
        except Exception:
            upgraded_cfg = dict(payload)
        return temp_spec_path, upgraded_cfg

    return config_path, data_cfg


def prepare_yolo_dataset(
    config_file: Union[str, Path],
    *,
    expected_task: Optional[str] = None,
) -> PreparedYOLODataset:
    """Resolve and validate a YOLO/COCO dataset config into a ready-to-train temporary YAML.

    Accepts:
      - A YOLO ``data.yaml`` file
      - A COCO spec YAML file (automatically materialized into YOLO format via hardlinks)
      - A directory containing ``data.yaml``, ``YOLO_dataset/data.yaml``, or COCO annotations

    Raises:
        YOLODatasetError: When the dataset is missing, unreadable, or incompatible.
    """
    temp_configs: List[Path] = []
    temp_dirs: List[Path] = []

    def _cleanup_on_error() -> None:
        PreparedYOLODataset(
            data_yaml=Path(),
            data_cfg={},
            temp_configs=temp_configs,
            temp_dataset_dirs=temp_dirs,
        ).cleanup()

    try:
        config_path = Path(config_file).expanduser().resolve()
        if not config_path.exists():
            raise YOLODatasetError(
                "Invalid Dataset",
                f"Dataset configuration file not found:\n{config_path}",
            )

        if config_path.is_dir():
            yaml_candidate, coco_payload = _resolve_directory_input(config_path)
            if yaml_candidate is not None:
                config_path = yaml_candidate
            elif coco_payload is not None:
                cfg_stub = config_path / "coco_pose_spec.yaml"
                temp_spec = write_temp_yolo_yaml(coco_payload, cfg_stub)
                temp_configs.append(temp_spec)
                config_path = temp_spec
            else:
                raise YOLODatasetError(
                    "Invalid COCO Folder",
                    "Could not find YOLO data.yaml or COCO train/val annotations. "
                    "Select a YOLO data.yaml, a COCO annotations folder, a dataset root "
                    "containing an annotations subfolder, or a dataset root with split-local "
                    "annotations.json files.",
                )

        try:
            with config_path.open("r", encoding="utf-8") as stream:
                data_cfg: Any = yaml.safe_load(stream)
            if not isinstance(data_cfg, dict):
                raise ValueError("Dataset configuration must be a YAML mapping.")
        except Exception as exc:
            logger.exception("Failed to load YOLO dataset config: %s", config_path)
            raise YOLODatasetError(
                "Dataset Error",
                f"Could not read dataset configuration:\n{config_path}\n\n{exc}",
                critical=True,
            ) from exc

        config_path, data_cfg = _upgrade_from_nearby_coco(
            config_path=config_path,
            data_cfg=data_cfg,
            expected_task=expected_task,
            temp_configs=temp_configs,
        )

        if looks_like_coco_spec(data_cfg):
            try:
                stage_dir = Path(
                    tempfile.mkdtemp(prefix=f"{config_path.stem}_coco_yolo_")
                ).resolve()
                temp_dirs.append(stage_dir)
                staged_yaml = materialize_coco_spec_as_yolo(
                    config_path=config_path,
                    output_dir=stage_dir,
                    link_mode="hardlink",
                    expected_task=expected_task,
                )
            except Exception as exc:
                logger.exception("Failed to convert COCO spec to a YOLO dataset.")
                raise YOLODatasetError(
                    "Dataset Error",
                    f"Failed to convert COCO annotations into a YOLO dataset.\n\n{exc}",
                    critical=True,
                ) from exc

            config_path = staged_yaml
            try:
                with config_path.open("r", encoding="utf-8") as stream:
                    data_cfg = yaml.safe_load(stream)
                if not isinstance(data_cfg, dict):
                    raise ValueError("Dataset configuration must be a YAML mapping.")
            except Exception as exc:
                logger.exception(
                    "Failed to load staged YOLO dataset config: %s", config_path
                )
                raise YOLODatasetError(
                    "Dataset Error",
                    f"Could not read staged YOLO dataset config:\n{config_path}\n\n{exc}",
                    critical=True,
                ) from exc

        base_dir = config_path.parent
        root_dir = _resolve_root_dir(data_cfg, base_dir)
        data_cfg["path"] = str(root_dir)

        for split in ("train", "val", "test"):
            if split in data_cfg and data_cfg[split]:
                data_cfg[split] = _resolve_split_entry(data_cfg[split], root_dir)

        missing_splits = _fill_missing_splits(data_cfg, root_dir=root_dir)
        if missing_splits:
            raise YOLODatasetError(
                "Dataset Split Required",
                "YOLO training requires both 'train' and 'val' in the dataset YAML.\n\n"
                f"Missing: {', '.join(missing_splits)}",
            )

        missing_paths: List[str] = []
        for split in ("train", "val", "test"):
            value = data_cfg.get(split)
            if not value:
                continue
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                if isinstance(entry, str) and not is_remote_dataset_entry(entry):
                    if any(char in entry for char in ("*", "?", "[")):
                        continue
                    if not Path(entry).exists():
                        missing_paths.append(f"{split}: {entry}")

        if missing_paths:
            raise YOLODatasetError(
                "Dataset Missing Files",
                "The following dataset paths could not be found:\n"
                + "\n".join(missing_paths),
            )

        _ensure_required_class_metadata(
            data_cfg,
            root_dir=root_dir,
            config_path=config_path,
        )

        # Pose datasets: auto-generate flip_idx if missing so fliplr augmentations work.
        if data_cfg.get("kpt_shape") and not data_cfg.get("flip_idx"):
            flip_idx = infer_flip_idx(data_cfg)
            if flip_idx:
                data_cfg["flip_idx"] = flip_idx

        if str(expected_task or "").strip().lower() == "pose" and not data_cfg.get(
            "kpt_shape"
        ):
            raise YOLODatasetError(
                "Pose Dataset Required",
                "The selected YOLO pose model requires a dataset with 'kpt_shape'.\n\n"
                "Choose a YOLO pose dataset or a COCO keypoints dataset.",
            )

        resolved_yaml = write_temp_yolo_yaml(data_cfg, config_path)
        temp_configs.append(resolved_yaml)
        return PreparedYOLODataset(
            data_yaml=resolved_yaml,
            data_cfg=data_cfg,
            temp_configs=temp_configs,
            temp_dataset_dirs=temp_dirs,
        )
    except Exception:
        _cleanup_on_error()
        raise


def check_training_preflight(
    data_config_path: Union[str, Path],
    *,
    batch_size: int,
    device: Optional[str],
    plots: bool,
    expected_task: Optional[str] = None,
) -> List[str]:
    """Return human-readable preflight warnings for a YOLO training configuration."""
    warnings: List[str] = []
    data_cfg: Dict[str, Any] = {}
    resolved_yaml = Path(data_config_path).expanduser()
    try:
        loaded = yaml.safe_load(resolved_yaml.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data_cfg = loaded
    except Exception:
        data_cfg = {}

    root_dir = _resolve_root_dir(data_cfg, resolved_yaml.parent)
    train_count = count_dataset_images(data_cfg.get("train"), root_dir=root_dir)
    val_count = count_dataset_images(data_cfg.get("val"), root_dir=root_dir)

    if train_count and batch_size > train_count:
        warnings.append(
            f"Batch size ({batch_size}) is larger than the training set ({train_count} images). "
            "This can result in very few optimizer steps and poor convergence."
        )

    if train_count and train_count < 50:
        warnings.append(
            f"Training set is very small ({train_count} images). Consider labeling more frames "
            "or using the 'Small Lab Dataset (Fine-tune)' preset."
        )

    if val_count and val_count < 10:
        warnings.append(
            f"Validation set is very small ({val_count} images). Metrics may be unstable/noisy."
        )

    kpt_shape = data_cfg.get("kpt_shape")
    if kpt_shape and not data_cfg.get("flip_idx"):
        warnings.append(
            "Pose dataset has 'kpt_shape' but no 'flip_idx' in data.yaml; "
            "Ultralytics disables flip augmentations."
        )

    if expected_task:
        exp = str(expected_task).strip().lower()
        detected = _infer_yolo_task_from_labels(data_cfg, root_dir=root_dir)
        if exp == "seg" and detected == "detect":
            warnings.append(
                "Selected a segmentation model (-seg), but sampled labels look like detection boxes. "
                "Verify your dataset contains polygon masks."
            )
        elif exp == "detect" and detected == "pose":
            warnings.append(
                "Dataset looks like a pose dataset (kpt_shape present), but a detection model was selected."
            )

    from annolid.yolo.runtime import resolve_yolo_device

    resolved_device = resolve_yolo_device(device)
    if resolved_device == "cpu":
        warnings.append(
            "Training on CPU is usually much slower. If available, prefer 'mps' (Apple Silicon) or CUDA GPU."
        )

    if plots and platform.system().lower() == "darwin":
        warnings.append(
            "Ultralytics training plots may crash on macOS in GUI/threaded runs (Bus error). "
            "If this happens, disable plots."
        )

    return warnings


__all__ = [
    "PreparedYOLODataset",
    "YOLODatasetError",
    "YOLODatasetSummary",
    "build_coco_spec_from_annotations_dir",
    "build_coco_spec_from_dataset_path",
    "check_training_preflight",
    "count_dataset_images",
    "infer_coco_task",
    "infer_flip_idx",
    "inspect_yolo_dataset",
    "is_remote_dataset_entry",
    "iter_existing_paths",
    "looks_like_coco_spec",
    "materialize_coco_detection_as_yolo",
    "materialize_coco_segmentation_as_yolo",
    "materialize_coco_spec_as_yolo",
    "normalize_class_names",
    "prepare_yolo_dataset",
    "read_yaml_dict",
    "resolve_coco_annotation_paths",
    "resolve_coco_root_path",
    "resolve_dataset_path",
    "write_temp_yolo_yaml",
]
