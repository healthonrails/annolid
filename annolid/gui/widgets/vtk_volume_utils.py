from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def path_matches_ext(path: Path, exts: Tuple[str, ...]) -> bool:
    """Match both simple suffixes and compound extensions like '.nii.gz'."""
    name = path.name.lower()
    suffix = path.suffix.lower()
    for ext in exts:
        ext_l = str(ext or "").lower()
        if not ext_l:
            continue
        if suffix == ext_l or name.endswith(ext_l):
            return True
    return False


def first_file_with_suffix(path: Path, exts: Tuple[str, ...]) -> Optional[Path]:
    try:
        entries = sorted(path.iterdir())
    except Exception:
        return None
    for entry in entries:
        if entry.is_file() and path_matches_ext(entry, exts):
            return entry
    return None


def normalize_to_float01(
    vol: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    out = np.asarray(vol, dtype=np.float32)
    if out.size == 0:
        return out

    finite = np.isfinite(out)
    if not np.any(finite):
        return np.zeros_like(out, dtype=np.float32)

    # Keep percentile bounds valid and deterministic even for bad caller inputs.
    lower = float(np.clip(lower_percentile, 0.0, 100.0))
    upper = float(np.clip(upper_percentile, 0.0, 100.0))
    if lower > upper:
        lower, upper = upper, lower
    if lower == upper:
        lower, upper = 0.0, 100.0

    data = out[finite]
    # Robust scaling keeps contrast stable for medical volumes with heavy tails.
    vmin = float(np.percentile(data, lower))
    vmax = float(np.percentile(data, upper))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(data))
        vmax = float(np.max(data))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(out, dtype=np.float32)

    scale = vmax - vmin
    out_finite = out[finite]
    out[finite] = np.clip((out_finite - vmin) / scale, 0.0, 1.0)
    out[~finite] = 0.0
    return out
