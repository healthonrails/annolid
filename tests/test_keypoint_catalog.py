from __future__ import annotations

from annolid.gui.keypoint_catalog import merge_keypoint_lists, normalize_keypoint_names


def test_merge_keypoint_lists_preserves_order_and_deduplicates_casefold() -> None:
    merged = merge_keypoint_lists(
        ["nose", "left_ear"],
        ["Nose", "right_ear"],
        ["left_ear", "tail_base"],
    )
    assert merged == ["nose", "left_ear", "right_ear", "tail_base"]


def test_normalize_keypoint_names_ignores_blank_values() -> None:
    names = normalize_keypoint_names(["", " nose ", None, "tail"])
    assert names == ["nose", "tail"]
