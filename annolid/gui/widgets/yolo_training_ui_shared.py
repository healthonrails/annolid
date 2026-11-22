from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional

from qtpy import QtWidgets

from annolid.gui.models_registry import MODEL_REGISTRY, ModelConfig
from annolid.yolo import (
    YOLO_FAMILIES,
    YOLO_MODEL_SIZES,
    YOLO_TASK_LABEL_TO_KEY,
    compute_yolo_weight_name,
    discover_yolo_checkpoints,
    get_yolo_train_preset,
    infer_yolo_task,
    iter_yolo_train_presets,
    list_available_yolo_devices,
    parse_yolo_weight_spec,
    task_key_to_label,
    task_label_to_key,
)


def default_yolo_plots_enabled() -> bool:
    """Default plots flag for GUI dialogs (disabled on macOS by default for stability)."""
    return sys.platform != "darwin"


def yolo_registry_weight_configs(*, include_yoloe: bool = False) -> List[ModelConfig]:
    """Return Ultralytics YOLO model configs registered in MODEL_REGISTRY."""
    configs: List[ModelConfig] = []
    for cfg in MODEL_REGISTRY:
        weight = str(getattr(cfg, "weight_file", "") or "").strip()
        if not weight:
            continue
        weight_l = weight.lower()
        if not weight_l.endswith((".pt", ".pth")):
            continue
        if not weight_l.startswith("yolo"):
            continue
        if not include_yoloe and weight_l.startswith("yoloe"):
            continue
        configs.append(cfg)
    return configs


def format_yolo_weight_display(weight: str) -> str:
    """Return a concise human-readable label for a YOLO weight name or checkpoint path."""
    text = str(weight or "").strip()
    if not text:
        return ""
    for cfg in yolo_registry_weight_configs(include_yoloe=True):
        if str(cfg.weight_file).strip().lower() == text.lower():
            return f"{cfg.display_name} ({cfg.weight_file})"
    parsed = parse_yolo_weight_spec(text)
    if parsed is not None:
        family, task_label, size = parsed
        return f"{family.upper()} {size} ({task_label}) — {Path(text).name}"
    p = Path(text)
    if p.parent != Path("."):
        parent_hint = (
            p.parent.parent.name if p.parent.name == "weights" else p.parent.name
        )
        return f"{parent_hint}/{p.name}" if parent_hint else p.name
    return p.name


def populate_yolo_family_combo(
    combo: QtWidgets.QComboBox, *, default: str = "yolo11"
) -> None:
    combo.clear()
    for label, key in YOLO_FAMILIES:
        combo.addItem(label, key)
    norm = str(default or "yolo11").strip().lower()
    idx = combo.findData(norm)
    if idx < 0:
        idx = combo.findText(str(default or "YOLO11"))
    if idx >= 0:
        combo.setCurrentIndex(idx)


def populate_yolo_task_combo(
    combo: QtWidgets.QComboBox, *, default_task: str = "seg"
) -> None:
    combo.clear()
    for label, task_key in YOLO_TASK_LABEL_TO_KEY.items():
        combo.addItem(label, task_key)
    norm_key = task_label_to_key(default_task) or "seg"
    idx = combo.findData(norm_key)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def populate_yolo_size_combo(combo: QtWidgets.QComboBox, *, default: str = "n") -> None:
    combo.clear()
    for size in YOLO_MODEL_SIZES:
        combo.addItem(size.upper(), size.upper())
    idx = combo.findData(str(default or "N").strip().upper()[:1] or "N")
    if idx >= 0:
        combo.setCurrentIndex(idx)


def populate_yolo_weight_combo(
    combo: QtWidgets.QComboBox,
    *,
    default_weight: str = "yolo11n-seg.pt",
    include_recent_runs: bool = True,
    max_recent: int = 8,
) -> None:
    """Populate a combo with built-in YOLO presets and optionally recent trained runs."""
    combo.clear()
    seen_values: set[str] = set()
    for cfg in yolo_registry_weight_configs():
        weight = str(cfg.weight_file).strip()
        if weight.lower() in seen_values:
            continue
        seen_values.add(weight.lower())
        combo.addItem(f"{cfg.display_name} ({weight})", weight)

    if include_recent_runs:
        try:
            checkpoints = discover_yolo_checkpoints(limit=max_recent)
        except Exception:
            checkpoints = []
        if checkpoints:
            combo.insertSeparator(combo.count())
            for ckpt in checkpoints:
                path_str = str(ckpt.path)
                if path_str.lower() in seen_values:
                    continue
                seen_values.add(path_str.lower())
                combo.addItem(f"Trained: {ckpt.display_name}", path_str)

    if default_weight:
        select_or_insert_yolo_weight(combo, default_weight)


def select_or_insert_yolo_weight(
    combo: QtWidgets.QComboBox,
    weight: str,
    *,
    display_label: Optional[str] = None,
) -> str:
    """Select an existing weight item by data or insert a computed/custom entry."""
    target = str(weight or "").strip()
    if not target:
        return ""
    target_lower = target.lower()
    for idx in range(combo.count()):
        data = combo.itemData(idx)
        if data is None:
            continue
        data_str = str(data).strip()
        if data_str == target or data_str.lower() == target_lower:
            combo.setCurrentIndex(idx)
            return data_str

    label = display_label or format_yolo_weight_display(target)
    combo.addItem(label, target)
    combo.setCurrentIndex(combo.count() - 1)
    return target


def populate_yolo_device_combo(
    combo: QtWidgets.QComboBox,
    *,
    default_device: Optional[str] = None,
) -> None:
    """Populate a device combo using hardware detection (Auto, CUDA, MPS, CPU)."""
    combo.clear()
    for label, value in list_available_yolo_devices():
        combo.addItem(label, value)

    if default_device is not None:
        target = str(default_device).strip()
        idx = combo.findData(target)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif target:
            combo.addItem(f"Custom ({target})", target)
            combo.setCurrentIndex(combo.count() - 1)


def populate_yolo_preset_combo(
    combo: QtWidgets.QComboBox,
    *,
    default_key: str = "balanced",
    include_custom: bool = True,
) -> None:
    """Populate fine-tuning preset choices with descriptions as item tooltips."""
    combo.clear()
    for preset in iter_yolo_train_presets():
        combo.addItem(preset.display_name, preset.key)
        combo.setItemData(combo.count() - 1, preset.description, 3)  # Qt.ToolTipRole
    if include_custom:
        combo.addItem("Custom", "custom")
        combo.setItemData(
            combo.count() - 1,
            "Manually customized training hyperparameters.",
            3,
        )
    idx = combo.findData(str(default_key or "balanced").strip().lower())
    if idx >= 0:
        combo.setCurrentIndex(idx)


def compute_weight_for_selectors(
    *,
    family: str,
    task: str,
    size: str,
) -> str:
    """Compute canonical Ultralytics weight filename from selector values."""
    return compute_yolo_weight_name(task, size, family=family)


def format_yolo_dataset_summary(summary: Any) -> str:
    """Format a YOLODatasetSummary (duck-typed) for inline GUI feedback."""
    if summary is None:
        return ""
    task = str(getattr(summary, "task", "detect") or "detect")
    task_label = task_key_to_label(task) or task.title()
    train_n = int(getattr(summary, "train_images", 0) or 0)
    val_n = int(getattr(summary, "val_images", 0) or 0)
    names = list(getattr(summary, "class_names", ()) or ())
    num_classes = len(names)
    source_kind = str(
        getattr(summary, "format_kind", "") or getattr(summary, "source_kind", "") or ""
    )
    kpt_shape = getattr(summary, "kpt_shape", None)

    parts = [f"Task: {task_label}"]
    if source_kind and source_kind != "unknown":
        parts.append(f"Format: {source_kind.upper()}")
    parts.append(f"Train: {train_n} imgs")
    parts.append(f"Val: {val_n} imgs")
    if num_classes:
        preview = ", ".join(names[:5])
        if len(names) > 5:
            preview += f", +{len(names) - 5} more"
        parts.append(
            f"Classes ({num_classes}): {preview}"
            if preview
            else f"Classes: {num_classes}"
        )
    if kpt_shape:
        parts.append(f"Keypoints: {kpt_shape[0]}")
    return "  •  ".join(parts)


__all__ = [
    "compute_weight_for_selectors",
    "default_yolo_plots_enabled",
    "format_yolo_dataset_summary",
    "format_yolo_weight_display",
    "get_yolo_train_preset",
    "infer_yolo_task",
    "parse_yolo_weight_spec",
    "populate_yolo_device_combo",
    "populate_yolo_family_combo",
    "populate_yolo_preset_combo",
    "populate_yolo_size_combo",
    "populate_yolo_task_combo",
    "populate_yolo_weight_combo",
    "select_or_insert_yolo_weight",
    "task_key_to_label",
    "task_label_to_key",
    "yolo_registry_weight_configs",
]
