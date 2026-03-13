from __future__ import annotations

TILE_NATIVE_CREATE_MODES = (
    "point",
    "line",
    "linestrip",
    "polygon",
    "rectangle",
    "circle",
)

LARGE_IMAGE_DRAW_MODE_LABELS = {
    "polygon": "Polygon",
    "rectangle": "Rectangle",
    "circle": "Circle",
    "point": "Point",
    "line": "Line",
    "linestrip": "Line Strip",
    "ai_polygon": "AI Polygon",
    "ai_mask": "AI Mask",
    "grounding_sam": "Grounding SAM",
    "polygonSAM": "Polygon SAM",
}


def is_tile_native_create_mode(create_mode: str | None) -> bool:
    return str(create_mode or "").lower() in TILE_NATIVE_CREATE_MODES


def large_image_draw_mode_label(create_mode: str | None) -> str:
    mode = str(create_mode or "")
    return LARGE_IMAGE_DRAW_MODE_LABELS.get(mode, mode or "editing")
