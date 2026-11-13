"""Atomic file-writing utilities to protect user annotations and configs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np


def _json_scalar_default(value: Any) -> Any:
    """Convert NumPy scalar values to native Python primitives for JSON."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_text(
    path: Union[str, Path],
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Atomically write text to ``path`` via a synced sibling temporary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    existing_mode: Optional[int] = None
    try:
        if target.exists():
            existing_mode = target.stat().st_mode & 0o777
    except OSError:
        existing_mode = None

    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(text)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # Some virtual or network filesystems do not support fsync.
                pass

        if existing_mode is not None:
            try:
                os.chmod(tmp_path, existing_mode)
            except OSError:
                pass

        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def atomic_write_json(
    path: Union[str, Path],
    payload: Any,
    *,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
    separators: Optional[Tuple[str, str]] = None,
    default: Optional[Callable[[Any], Any]] = None,
    encoding: str = "utf-8",
) -> None:
    """Serialize ``payload`` to JSON first, then atomically replace ``path``."""
    serialized = json.dumps(
        payload,
        indent=indent,
        ensure_ascii=ensure_ascii,
        separators=separators,
        default=_json_scalar_default if default is None else default,
    )
    atomic_write_text(path, serialized, encoding=encoding)


__all__ = [
    "atomic_write_json",
    "atomic_write_text",
]
