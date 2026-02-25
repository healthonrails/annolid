from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable


_KNOWN_DATA_FORMATS = {"auto", "yolo", "labelme", "coco"}


def normalize_dino_kpseg_data_format(
    data_cfg: Dict[str, Any],
    *,
    data_format: str = "auto",
) -> str:
    """Normalize dataset format for DinoKPSEG training.

    Resolution order:
    1) Explicit `data_format` when valid and non-auto.
    2) YAML `format`/`type` token.
    3) Heuristics from `train`/`val`/`test` entries.
    4) Fallback to `yolo`.
    """
    fmt = str(data_format or "auto").strip().lower()
    if fmt not in _KNOWN_DATA_FORMATS:
        fmt = "auto"
    if fmt != "auto":
        return fmt

    token = str(data_cfg.get("format") or data_cfg.get("type") or "").strip().lower()
    if "labelme" in token:
        return "labelme"
    if "coco" in token:
        return "coco"
    if "yolo" in token:
        return "yolo"

    split_entries = list(_iter_split_entries(data_cfg))
    if any(_looks_like_coco_annotation_path(v) for v in split_entries):
        return "coco"
    if any(_looks_like_labelme_index_path(v) for v in split_entries):
        return "labelme"

    if data_cfg.get("keypoint_names") or data_cfg.get("kpt_names"):
        return "labelme"

    return "yolo"


def _iter_split_entries(data_cfg: Dict[str, Any]) -> Iterable[str]:
    for key in ("train", "val", "test"):
        value = data_cfg.get(key)
        if isinstance(value, str):
            raw = value.strip()
            if raw:
                yield raw
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    raw = item.strip()
                    if raw:
                        yield raw


def _looks_like_coco_annotation_path(value: str) -> bool:
    p = Path(str(value).strip())
    suffix = p.suffix.lower()
    if suffix != ".json":
        return False
    name = p.name.lower()
    if "annotation" in name:
        return True
    parent = p.parent.name.lower()
    return parent in {"annotations", "ann", "anno"}


def _looks_like_labelme_index_path(value: str) -> bool:
    p = Path(str(value).strip())
    suffix = p.suffix.lower()
    return suffix in {".jsonl", ".list"}
