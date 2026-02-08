from __future__ import annotations

from pathlib import Path

THREEJS_MODEL_EXTENSIONS = frozenset({".stl", ".obj", ".ply", ".csv", ".xyz"})


def supports_threejs_canvas(path: str | Path) -> bool:
    try:
        suffix = Path(path).suffix.lower()
    except Exception:
        return False
    return suffix in THREEJS_MODEL_EXTENSIONS
