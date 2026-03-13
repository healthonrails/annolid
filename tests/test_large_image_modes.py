from annolid.gui.large_image_modes import (
    is_tile_native_create_mode,
    large_image_draw_mode_label,
)


def test_tile_native_create_modes_cover_supported_manual_tools() -> None:
    for mode in ("point", "line", "linestrip", "polygon", "rectangle", "circle"):
        assert is_tile_native_create_mode(mode) is True

    for mode in ("ai_polygon", "ai_mask", "grounding_sam", "polygonSAM", ""):
        assert is_tile_native_create_mode(mode) is False


def test_large_image_draw_mode_label_provides_user_facing_names() -> None:
    assert large_image_draw_mode_label("polygon") == "Polygon"
    assert large_image_draw_mode_label("ai_polygon") == "AI Polygon"
    assert large_image_draw_mode_label("grounding_sam") == "Grounding SAM"
    assert large_image_draw_mode_label("unknown_mode") == "unknown_mode"
