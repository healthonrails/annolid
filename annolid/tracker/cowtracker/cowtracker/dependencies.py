"""Dependency gateway for CowTracker third-party integrations.

This module centralizes imports for:
- bundled VGGT (under ``cowtracker/thirdparty/vggt``)
- Annolid's bundled Depth-Anything utility blocks

Keeping these imports in one place avoids scattered ``sys.path`` side effects and
makes dependency failures easier to diagnose.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

_INSTALL_HINT = (
    'Install CowTracker dependencies via `pip install "annolid[cowtracker]"` '
    'or `pip install "safetensors>=0.4.0"`.'
)


def _thirdparty_vggt_root() -> Path:
    return Path(__file__).resolve().parent / "thirdparty" / "vggt"


def ensure_vggt_importable() -> Path:
    """Ensure the vendored VGGT package root is importable."""
    root = _thirdparty_vggt_root()
    if not root.exists():
        raise RuntimeError(
            f"Bundled VGGT directory not found at '{root}'. "
            "CowTracker requires the vendored `thirdparty/vggt` tree. "
            f"{_INSTALL_HINT}"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _import_module_or_raise(module_name: str, *, context: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import failure path
        raise RuntimeError(
            f"Failed to import {context} ({module_name}). {_INSTALL_HINT}"
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
