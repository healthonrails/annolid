from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional


DEFAULT_RUNS_DIRNAME = "annolid_logs/runs"


def _slugify(value: str, *, max_len: int = 64) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value[:max_len] if value else "run"


def shared_runs_root(*, base_dir: Optional[Path] = None) -> Path:
    """Return the shared runs root for TensorBoard and training artifacts.

    Precedence:
      1) ANNOLID_RUNS_ROOT
      2) ANNOLID_LOG_ROOT / ANNOLID_LOG_DIR
      3) base_dir/annolid_logs/runs (when base_dir is provided)
      4) ~/annolid_logs/runs
    """
    for key in ("ANNOLID_RUNS_ROOT", "ANNOLID_LOG_ROOT", "ANNOLID_LOG_DIR"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser().resolve()

    if base_dir is not None:
        base_dir = Path(base_dir).expanduser().resolve()
        return (base_dir / "annolid_logs" / "runs").resolve()

    return (Path.home() / "annolid_logs" / "runs").resolve()


def new_run_dir(
    *,
    task: str,
    model: str,
    runs_root: Optional[Path] = None,
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Create a unique run directory under the shared runs root."""
    root = Path(runs_root) if runs_root is not None else shared_runs_root()
    root = root.expanduser().resolve()
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    task_slug = _slugify(task)
    model_slug = _slugify(model)
    name_slug = _slugify(run_name) if run_name else ts
    return (root / task_slug / model_slug / name_slug).resolve()


def find_latest_checkpoint(
    *,
    task: str,
    model: str,
    filename: str = "best.pt",
    runs_root: Optional[Path] = None,
) -> Optional[Path]:
    """Find the most recently modified checkpoint under the runs root.

    Expected layout:
      <runs_root>/<task>/<model>/<run_name>/weights/<filename>
    """
    root = Path(runs_root) if runs_root is not None else shared_runs_root()
    root = root.expanduser().resolve()
    if not root.exists():
        return None

    task_slug = _slugify(task)
    model_slug = _slugify(model)
    pattern = f"{task_slug}/{model_slug}/*/weights/{filename}"
    candidates = [p for p in root.glob(pattern) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
