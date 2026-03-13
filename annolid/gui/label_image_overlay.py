from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


ID_COLUMN_CANDIDATES = (
    "id",
    "region_id",
    "label_id",
    "structure_id",
    "atlas_id",
    "value",
    "index",
)
ACRONYM_COLUMN_CANDIDATES = (
    "acronym",
    "abbreviation",
    "abbr",
    "short_name",
    "label",
)
NAME_COLUMN_CANDIDATES = (
    "name",
    "region_name",
    "full_name",
    "structure_name",
    "annotation",
)


def _normalize_header(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _select_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_header(name): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_label_mapping_csv(path: str | Path) -> dict[int, dict[str, Any]]:
    mapping_path = Path(path)
    raw = mapping_path.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",\t;")
    except Exception:
        dialect = csv.excel
    reader = csv.DictReader(raw.splitlines(), dialect=dialect)
    fieldnames = list(reader.fieldnames or [])
    id_col = _select_column(fieldnames, ID_COLUMN_CANDIDATES)
    acronym_col = _select_column(fieldnames, ACRONYM_COLUMN_CANDIDATES)
    name_col = _select_column(fieldnames, NAME_COLUMN_CANDIDATES)
    if id_col is None:
        raise ValueError("Could not find an id column in the label mapping file.")
    results: dict[int, dict[str, Any]] = {}
    for row in reader:
        try:
            label_id = int(float(str(row.get(id_col, "")).strip()))
        except Exception:
            continue
        results[label_id] = {
            "id": label_id,
            "acronym": str(row.get(acronym_col, "")).strip() if acronym_col else "",
            "name": str(row.get(name_col, "")).strip() if name_col else "",
            "raw": dict(row),
        }
    return results


def label_entry_text(
    label_value: int, mapping: dict[int, dict[str, Any]] | None
) -> str:
    if int(label_value) <= 0:
        return "background"
    record = dict((mapping or {}).get(int(label_value)) or {})
    acronym = str(record.get("acronym", "") or "").strip()
    name = str(record.get("name", "") or "").strip()
    if acronym and name and acronym.lower() != name.lower():
        return f"{label_value}: {acronym} ({name})"
    if name:
        return f"{label_value}: {name}"
    if acronym:
        return f"{label_value}: {acronym}"
    return str(int(label_value))


def label_color_table() -> np.ndarray:
    try:
        import imgviz

        cmap = np.asarray(imgviz.label_colormap(), dtype=np.uint8)
        if cmap.ndim == 2 and cmap.shape[1] >= 3:
            return cmap[:, :3]
    except Exception:
        pass
    indices = np.arange(256, dtype=np.uint32)
    r = ((indices * 37) % 255).astype(np.uint8)
    g = ((indices * 67) % 255).astype(np.uint8)
    b = ((indices * 97) % 255).astype(np.uint8)
    table = np.stack([r, g, b], axis=1)
    table[0] = np.array([0, 0, 0], dtype=np.uint8)
    return table


_LABEL_COLOR_TABLE = label_color_table()


def colorize_label_image(
    labels: np.ndarray,
    *,
    opacity: float = 0.45,
    selected_label: int | None = None,
) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D label image, got shape {arr.shape}")
    label_values = arr.astype(np.int64, copy=False)
    indices = np.mod(label_values, len(_LABEL_COLOR_TABLE)).astype(np.intp, copy=False)
    rgb = _LABEL_COLOR_TABLE[indices].copy()
    alpha = np.zeros(label_values.shape, dtype=np.uint8)
    foreground = label_values > 0
    base_alpha = int(max(0.0, min(1.0, float(opacity))) * 255.0)
    alpha[foreground] = base_alpha
    selected = int(selected_label) if selected_label is not None else None
    if selected is not None and selected > 0:
        selected_mask = label_values == selected
        dimmed = foreground & ~selected_mask
        alpha[dimmed] = np.minimum(alpha[dimmed], max(18, base_alpha // 4))
        alpha[selected_mask] = max(base_alpha, 210)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
    rgba[~foreground] = np.array([0, 0, 0, 0], dtype=np.uint8)
    return rgba
