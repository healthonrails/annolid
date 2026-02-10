"""Central spec for CowTracker's vendored VGGT runtime subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Iterable

INSTALL_HINT: Final[str] = (
    'Install CowTracker dependencies via `pip install "annolid[cowtracker]"` '
    'or `pip install "safetensors>=0.4.0"`.'
)
IMPORT_HINT: Final[str] = (
    "Expected either a vendored runtime tree at "
    "`annolid/tracker/cowtracker/cowtracker/thirdparty/vggt`, or an importable "
    "`vggt` Python package."
)

# Keep this in sync with thirdparty/vggt/VENDORED_MANIFEST.json.
REQUIRED_RUNTIME_FILES: Final[tuple[str, ...]] = (
    "vggt/__init__.py",
    "vggt/models/__init__.py",
    "vggt/models/aggregator.py",
    "vggt/heads/__init__.py",
    "vggt/heads/dpt_head.py",
    "vggt/heads/head_act.py",
    "vggt/heads/utils.py",
    "vggt/layers/__init__.py",
    "vggt/layers/attention.py",
    "vggt/layers/block.py",
    "vggt/layers/drop_path.py",
    "vggt/layers/layer_scale.py",
    "vggt/layers/mlp.py",
    "vggt/layers/patch_embed.py",
    "vggt/layers/rope.py",
    "vggt/layers/swiglu_ffn.py",
    "vggt/layers/vision_transformer.py",
    "LICENSE.txt",
)


def vendored_vggt_root() -> Path:
    return Path(__file__).resolve().parents[1] / "thirdparty" / "vggt"


def vendored_manifest_path() -> Path:
    return vendored_vggt_root() / "VENDORED_MANIFEST.json"


def missing_runtime_files(root: Path) -> list[str]:
    return [rel for rel in REQUIRED_RUNTIME_FILES if not (root / rel).exists()]


def vendored_runtime_is_complete(root: Path) -> bool:
    return not missing_runtime_files(root)


def load_manifest_required_files() -> tuple[str, ...]:
    manifest = vendored_manifest_path()
    if not manifest.exists():
        return ()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    files = data.get("required_runtime_files", [])
    return tuple(str(path) for path in files)


def iter_runtime_files(root: Path) -> Iterable[Path]:
    for rel_path in REQUIRED_RUNTIME_FILES:
        yield root / rel_path
