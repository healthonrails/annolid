from __future__ import annotations

from annolid.gui.widgets.video_slider import VideoSlider


def test_parse_frame_input_accepts_integers_only() -> None:
    assert VideoSlider._parse_frame_input("29") == 29
    assert VideoSlider._parse_frame_input("  42  ") == 42
    assert VideoSlider._parse_frame_input("29.0") is None
    assert VideoSlider._parse_frame_input("") is None
    assert VideoSlider._parse_frame_input("frame-3") is None
