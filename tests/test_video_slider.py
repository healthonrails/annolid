from __future__ import annotations

from annolid.gui.widgets.video_slider import VideoSlider, VideoSliderMark


def test_parse_frame_input_accepts_integers_only() -> None:
    assert VideoSlider._parse_frame_input("29") == 29
    assert VideoSlider._parse_frame_input("  42  ") == 42
    assert VideoSlider._parse_frame_input("29.0") is None
    assert VideoSlider._parse_frame_input("") is None
    assert VideoSlider._parse_frame_input("frame-3") is None


def test_missing_instance_mark_has_stable_visuals() -> None:
    mark = VideoSliderMark(mark_type="missing_instance", val=12)
    assert mark.color == (220, 38, 38)
    assert mark.visual_width == 1
