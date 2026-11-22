from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from annolid.utils.runs import shared_runs_root


YOLO_FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("YOLO11", "yolo11"),
    ("YOLO26", "yolo26"),
)

YOLO_TASK_SUFFIXES: Dict[str, str] = {
    "Detection": "",
    "Instance Segmentation": "-seg",
    "Pose Estimation": "-pose",
    "Oriented Detection": "-obb",
    "Classification": "-cls",
}

YOLO_TASK_KEY_TO_LABEL: Dict[str, str] = {
    "detect": "Detection",
    "detection": "Detection",
    "seg": "Instance Segmentation",
    "segment": "Instance Segmentation",
    "segmentation": "Instance Segmentation",
    "pose": "Pose Estimation",
    "obb": "Oriented Detection",
    "cls": "Classification",
    "classify": "Classification",
    "classification": "Classification",
}

YOLO_TASK_LABEL_TO_KEY: Dict[str, str] = {
    "Detection": "detect",
    "Instance Segmentation": "seg",
    "Pose Estimation": "pose",
    "Oriented Detection": "obb",
    "Classification": "cls",
}

YOLO_MODEL_SIZES: Tuple[str, ...] = ("n", "s", "m", "l", "x")

_WEIGHT_NAME_RE = re.compile(
    r"^(yolo(?:11|26|v8|v9|v10|v12|e-11|e-26))([nslmx])(?:(-seg|-pose|-obb|-cls))?(?:-pf)?\.pt$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class YOLOTrainPreset:
    """Curated training / fine-tuning defaults for common research workflows."""

    key: str
    display_name: str
    description: str
    epochs: int
    imgsz: int
    batch: int
    lr0: float
    lrf: float
    weight_decay: float
    patience: int
    close_mosaic: int
    optimizer: str
    cos_lr: bool
    freeze: int
    cache: Any

    def to_overrides(self) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {
            "lr0": float(self.lr0),
            "lrf": float(self.lrf),
            "weight_decay": float(self.weight_decay),
            "patience": int(self.patience),
            "close_mosaic": int(self.close_mosaic),
            "optimizer": str(self.optimizer),
            "cos_lr": bool(self.cos_lr),
            "cache": self.cache,
        }
        if int(self.freeze) > 0:
            overrides["freeze"] = int(self.freeze)
        return overrides


YOLO_TRAIN_PRESETS: Dict[str, YOLOTrainPreset] = {
    "balanced": YOLOTrainPreset(
        key="balanced",
        display_name="Balanced (Recommended)",
        description="Good accuracy/speed tradeoff for most lab datasets (100+ frames).",
        epochs=100,
        imgsz=640,
        batch=8,
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        patience=50,
        close_mosaic=10,
        optimizer="auto",
        cos_lr=True,
        freeze=0,
        cache=False,
    ),
    "small_dataset": YOLOTrainPreset(
        key="small_dataset",
        display_name="Small Lab Dataset (Fine-tune)",
        description=(
            "Freezes early backbone layers with AdamW + cosine LR for stable "
            "fine-tuning on small datasets (~20-150 labeled frames)."
        ),
        epochs=120,
        imgsz=640,
        batch=8,
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        patience=40,
        close_mosaic=15,
        optimizer="AdamW",
        cos_lr=True,
        freeze=10,
        cache=True,
    ),
    "fast": YOLOTrainPreset(
        key="fast",
        display_name="Quick Smoke Test",
        description="Fast 30-epoch pilot run to verify labels, classes, and pipeline.",
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=0.01,
        lrf=0.05,
        weight_decay=0.0005,
        patience=15,
        close_mosaic=5,
        optimizer="auto",
        cos_lr=False,
        freeze=0,
        cache=False,
    ),
    "high_accuracy": YOLOTrainPreset(
        key="high_accuracy",
        display_name="High Accuracy (960px)",
        description="Higher input resolution and longer cosine schedule for small animals/parts.",
        epochs=200,
        imgsz=960,
        batch=8,
        lr0=0.005,
        lrf=0.005,
        weight_decay=0.0005,
        patience=60,
        close_mosaic=20,
        optimizer="AdamW",
        cos_lr=True,
        freeze=0,
        cache=False,
    ),
}


def iter_yolo_train_presets() -> Tuple[YOLOTrainPreset, ...]:
    return tuple(YOLO_TRAIN_PRESETS.values())


def get_yolo_train_preset(name: str) -> YOLOTrainPreset:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in YOLO_TRAIN_PRESETS:
        return YOLO_TRAIN_PRESETS[key]
    for preset in YOLO_TRAIN_PRESETS.values():
        if preset.display_name.lower() == str(name or "").strip().lower():
            return preset
    available = ", ".join(YOLO_TRAIN_PRESETS.keys())
    raise ValueError(f"Unknown YOLO preset {name!r}. Available: {available}")


def recommend_preset_for_dataset(train_images: Optional[int]) -> YOLOTrainPreset:
    """Suggest a sensible training preset based on training set size."""
    try:
        count = int(train_images or 0)
    except Exception:
        count = 0
    if 0 < count <= 150:
        return YOLO_TRAIN_PRESETS["small_dataset"]
    return YOLO_TRAIN_PRESETS["balanced"]


def task_key_to_label(task_key: Optional[str]) -> Optional[str]:
    key = str(task_key or "").strip().lower()
    if not key:
        return None
    if key in YOLO_TASK_KEY_TO_LABEL:
        return YOLO_TASK_KEY_TO_LABEL[key]
    for label in YOLO_TASK_SUFFIXES:
        if label.lower() == key:
            return label
    return None


def task_label_to_key(task_label: Optional[str]) -> Optional[str]:
    text = str(task_label or "").strip()
    if not text:
        return None
    if text in YOLO_TASK_LABEL_TO_KEY:
        return YOLO_TASK_LABEL_TO_KEY[text]
    return YOLO_TASK_LABEL_TO_KEY.get(task_key_to_label(text) or "")


def compute_yolo_weight_name(
    task_text: str,
    size_text: str,
    *,
    family: str = "yolo11",
) -> str:
    """Compute a standard Ultralytics pretrained weight filename (e.g. yolo11n-seg.pt)."""
    fam = str(family or "yolo11").strip().lower()
    if fam not in {"yolo11", "yolo26", "yolov8"}:
        fam = "yolo11"
    label = task_key_to_label(task_text) or task_text
    suffix = YOLO_TASK_SUFFIXES.get(label, "")
    size = str(size_text or "n").strip().lower()[:1] or "n"
    if size not in YOLO_MODEL_SIZES:
        size = "n"
    return f"{fam}{size}{suffix}.pt"


def parse_yolo_weight_spec(
    weight_file: str,
) -> Optional[Tuple[str, str, str]]:
    """Parse standard Ultralytics filenames into (family, task_label, size_upper)."""
    raw = str(weight_file or "").strip()
    if not raw:
        return None
    name = Path(raw).name.lower()
    match = _WEIGHT_NAME_RE.match(name)
    if not match:
        return None
    family = match.group(1).lower()
    size = match.group(2).upper()
    suffix = match.group(3) or ""
    suffix_to_task = {v: k for k, v in YOLO_TASK_SUFFIXES.items()}
    task_label = suffix_to_task.get(suffix, "Detection")
    return family, task_label, size


def infer_yolo_task(
    weight_or_path: Optional[str],
    *,
    fallback_weight: Optional[str] = None,
) -> Optional[str]:
    """Infer YOLO task key ('pose', 'seg', 'obb', 'cls', 'detect') from a weight name/path."""
    for candidate in (weight_or_path, fallback_weight):
        raw = str(candidate or "").strip()
        if not raw:
            continue
        parsed = parse_yolo_weight_spec(raw)
        if parsed is not None:
            return task_label_to_key(parsed[1])
        token = raw.lower()
        if "-pose" in token or token.endswith("pose.pt") or "/pose/" in token:
            return "pose"
        if (
            "-seg" in token
            or token.endswith("seg.pt")
            or "/segment/" in token
            or "/seg/" in token
        ):
            return "seg"
        if "-obb" in token or token.endswith("obb.pt") or "/obb/" in token:
            return "obb"
        if "-cls" in token or token.endswith("cls.pt") or "/classify/" in token:
            return "cls"
        # Check sibling args.yaml if a trained run checkpoint path was supplied.
        try:
            path = Path(raw).expanduser()
            for parent in (path.parent, path.parent.parent):
                args_yaml = parent / "args.yaml"
                if args_yaml.is_file():
                    import yaml

                    data = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
                    if isinstance(data, dict):
                        task_val = str(data.get("task") or "").strip().lower()
                        mapped = task_label_to_key(task_key_to_label(task_val))
                        if mapped:
                            return mapped
        except Exception:
            pass
    return None


def is_yoloe_model(weight: Optional[str]) -> bool:
    return "yoloe" in str(weight or "").strip().lower()


def is_prompt_free_yoloe(weight: Optional[str]) -> bool:
    w = str(weight or "").strip().lower()
    return "yoloe" in w and ("-pf." in w or w.endswith("-pf") or "-pf_" in w)


@dataclass(frozen=True)
class DiscoveredYOLOCheckpoint:
    """Metadata for a trained YOLO checkpoint discovered on disk."""

    path: Path
    run_dir: Path
    model_name: str
    run_name: str
    task: Optional[str]
    mtime: float

    @property
    def display_name(self) -> str:
        if self.model_name and self.run_name and self.model_name != self.run_name:
            return f"YOLO: {self.model_name} ({self.run_name})"
        label = self.run_name or self.model_name or self.path.stem
        return f"YOLO: {label}"


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_iterdir(path: Path) -> List[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def discover_yolo_checkpoints(
    *,
    runs_root: Optional[Path] = None,
    extra_roots: Optional[Iterable[Path]] = None,
    limit: int = 20,
    include_last: bool = False,
) -> List[DiscoveredYOLOCheckpoint]:
    """Discover recently trained YOLO checkpoints across standard Annolid and Ultralytics layouts.

    Supported layouts under each search root:
      - ``<root>/yolo/<model_stem>/<run_name>/weights/best.pt`` (Annolid shared runs)
      - ``<root>/train/<run_name>/weights/best.pt`` (legacy runs layout)
      - ``<root>/{detect,segment,pose,obb,classify}/<run_name>/weights/best.pt``
      - ``<root>/<run_name>/weights/best.pt`` (custom output_dir runs)
    """
    roots: List[Path] = []
    if runs_root is not None:
        roots.append(Path(runs_root).expanduser())
    else:
        try:
            roots.append(shared_runs_root())
        except Exception:
            pass
    if extra_roots:
        for r in extra_roots:
            roots.append(Path(r).expanduser())

    weight_names: Sequence[str] = (
        ("best.pt", "last.pt") if include_last else ("best.pt",)
    )
    seen_paths: set[str] = set()
    discovered: List[DiscoveredYOLOCheckpoint] = []

    def _record_run(run_dir: Path, model_hint: str) -> None:
        weights_dir = run_dir / "weights"
        if not _safe_is_dir(weights_dir):
            return
        for wname in weight_names:
            wpath = weights_dir / wname
            if not _safe_is_file(wpath):
                continue
            try:
                resolved = wpath.resolve()
                key = str(resolved)
            except OSError:
                resolved = wpath
                key = str(wpath)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            task = infer_yolo_task(str(resolved), fallback_weight=model_hint)
            discovered.append(
                DiscoveredYOLOCheckpoint(
                    path=resolved,
                    run_dir=run_dir,
                    model_name=model_hint,
                    run_name=run_dir.name,
                    task=task,
                    mtime=_safe_mtime(resolved),
                )
            )
            # Prefer best.pt per run_dir unless best.pt is absent.
            break

    for root in roots:
        if not _safe_is_dir(root):
            continue

        # 1. Annolid layout: <root>/yolo/<model>/<run>/weights/best.pt
        yolo_root = root / "yolo"
        if _safe_is_dir(yolo_root):
            for model_dir in _safe_iterdir(yolo_root):
                if not _safe_is_dir(model_dir):
                    continue
                if _safe_is_dir(model_dir / "weights"):
                    _record_run(model_dir, model_dir.name)
                for run_dir in _safe_iterdir(model_dir):
                    if _safe_is_dir(run_dir):
                        _record_run(run_dir, model_dir.name)

        # 2. Legacy & standard Ultralytics task dirs under root
        for task_subdir in ("train", "detect", "segment", "pose", "obb", "classify"):
            parent = root / task_subdir
            if not _safe_is_dir(parent):
                continue
            if _safe_is_dir(parent / "weights"):
                _record_run(parent, task_subdir)
            for run_dir in _safe_iterdir(parent):
                if _safe_is_dir(run_dir):
                    _record_run(run_dir, task_subdir if task_subdir != "train" else "")

        # 3. Direct children with weights/ (when user pointed at a custom project dir)
        for child in _safe_iterdir(root):
            if not _safe_is_dir(child):
                continue
            if child.name in {
                "yolo",
                "dino_kpseg",
                "maskrcnn",
                "keypoint_rcnn",
                "train",
                "detect",
                "segment",
                "pose",
                "obb",
                "classify",
            }:
                continue
            if _safe_is_dir(child / "weights"):
                _record_run(child, "")

    discovered.sort(key=lambda item: item.mtime, reverse=True)
    if limit > 0:
        return discovered[: int(limit)]
    return discovered
