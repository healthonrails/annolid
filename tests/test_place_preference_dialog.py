from __future__ import annotations

from annolid.gui.widgets.place_preference_dialog import TrackingAnalyzerDialog


def test_latency_reference_parser_rejects_float_strings() -> None:
    assert TrackingAnalyzerDialog._parse_integer_frame_text("29") == 29
    assert TrackingAnalyzerDialog._parse_integer_frame_text(" 42 ") == 42
    assert TrackingAnalyzerDialog._parse_integer_frame_text("29.0") is None
    assert TrackingAnalyzerDialog._parse_integer_frame_text("") is None
