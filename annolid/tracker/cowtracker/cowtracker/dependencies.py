"""Dependency gateway for CowTracker third-party integrations.

This module centralizes imports for:
- vendored VGGT runtime subset (``cowtracker/thirdparty/vggt``)
- Annolid's bundled Depth-Anything utility blocks

Keeping these imports in one place avoids scattered ``sys.path`` side effects and
makes dependency failures easier to diagnose.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

try:
    # Import path when loaded as top-level `cowtracker` package.
    from cowtracker.vendor.vggt_runtime import (
        IMPORT_HINT as _VGGT_IMPORT_HINT,
        INSTALL_HINT as _INSTALL_HINT,
        missing_runtime_files,
        vendored_runtime_is_complete,
        vendored_vggt_root,
    )
except ImportError:  # pragma: no cover - fallback for annolid package path
    from .vendor.vggt_runtime import (
        IMPORT_HINT as _VGGT_IMPORT_HINT,
        INSTALL_HINT as _INSTALL_HINT,
        missing_runtime_files,
        vendored_runtime_is_complete,
        vendored_vggt_root,
    )


def ensure_vggt_importable() -> Path:
    """Ensure VGGT is importable, preferring vendored runtime files when present.

    Returns:
        Path to the vendored VGGT root if it is present and complete.

    Raises:
        RuntimeError: If neither a complete vendored runtime tree nor an importable
            external `vggt` package is available.
    """
    root = vendored_vggt_root()
    if root.exists():
        if not vendored_runtime_is_complete(root):
            missing = ", ".join(missing_runtime_files(root))
            raise RuntimeError(
                f"Vendored VGGT directory is incomplete at '{root}'. "
                f"Missing: {missing}. "
                "CowTracker needs the VGGT runtime subset listed in "
                "`annolid/tracker/cowtracker/README.md`. "
                f"{_INSTALL_HINT}"
            )
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return root

    try:
        importlib.import_module("vggt")
    except Exception as exc:
        raise RuntimeError(
            f"VGGT runtime not found. {_VGGT_IMPORT_HINT} {_INSTALL_HINT}"
        ) from exc

    return root


def _import_module_or_raise(module_name: str, *, context: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import failure path
        raise RuntimeError(
            f"Failed to import {context} ({module_name}). "
            f"{_VGGT_IMPORT_HINT} {_INSTALL_HINT}"
        ) from exc


def get_vggt_aggregator_cls() -> type:
    ensure_vggt_importable()
    module = _import_module_or_raise(
        "vggt.models.aggregator", context="VGGT aggregator"
    )
    return getattr(module, "Aggregator")


def get_vggt_dpt_head_cls() -> type:
    ensure_vggt_importable()
    module = _import_module_or_raise("vggt.heads.dpt_head", context="VGGT DPTHead")
    return getattr(module, "DPTHead")


def get_depth_anything_blocks() -> tuple[Any, Any]:
    module = _import_module_or_raise(
        "annolid.depth.video_depth_anything.util.blocks",
        context="Depth-Anything utility blocks",
    )
    return getattr(module, "FeatureFusionBlock"), getattr(module, "_make_scratch")


def validate_cowtracker_runtime() -> None:
    """Preflight validation for CowTracker runtime dependencies."""
    ensure_vggt_importable()
    _ = get_vggt_aggregator_cls()
    _ = get_vggt_dpt_head_cls()
    _ = get_depth_anything_blocks()
