from __future__ import annotations

from pathlib import Path

THREEJS_MODEL_EXTENSIONS = frozenset(
    {".stl", ".obj", ".ply", ".csv", ".xyz", ".glb", ".gltf", ".zarr"}
)
THREEJS_PANORAMA_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
)
THREEJS_CANVAS_EXTENSIONS = frozenset(
    set(THREEJS_MODEL_EXTENSIONS) | set(THREEJS_PANORAMA_EXTENSIONS)
)


def supports_threejs_canvas(path: str | Path) -> bool:
    try:
        suffix = Path(path).suffix.lower()
    except Exception:
        return False
    return suffix in THREEJS_CANVAS_EXTENSIONS
