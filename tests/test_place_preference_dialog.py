from __future__ import annotations

from qtpy import QtWidgets

from annolid.gui.widgets.place_preference_dialog import TrackingAnalyzerDialog


def _ensure_qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_latency_reference_parser_rejects_float_strings() -> None:
    assert TrackingAnalyzerDialog._parse_integer_frame_text("29") == 29
    assert TrackingAnalyzerDialog._parse_integer_frame_text(" 42 ") == 42
    assert TrackingAnalyzerDialog._parse_integer_frame_text("29.0") is None
    assert TrackingAnalyzerDialog._parse_integer_frame_text("") is None


def test_mode_specs_cover_all_exports() -> None:
    specs = TrackingAnalyzerDialog._build_mode_specs()
    assert len(specs) == 4

    by_key = {str(spec["key"]): spec for spec in specs}
    assert set(by_key.keys()) == {
        "legacy_csv",
        "zone_metrics",
        "assay_summary",
        "social_summary",
    }
    assert by_key["legacy_csv"]["method"] == "save_all_instances_zone_time_to_csv"
    assert by_key["legacy_csv"]["assay_profile"] == "generic"
    assert by_key["zone_metrics"]["method"] == "save_zone_metrics_to_csv"
    assert by_key["assay_summary"]["method"] == "save_assay_summary_report"
    assert by_key["social_summary"]["method"] == "save_social_summary_report"
    assert by_key["social_summary"]["requires_latency"] is True


def test_dialog_uses_scroll_area_for_compact_layout() -> None:
    _ensure_qapp()
    dialog = TrackingAnalyzerDialog()
    scroll_areas = dialog.findChildren(QtWidgets.QScrollArea)
    assert scroll_areas
    assert all(area.widgetResizable() for area in scroll_areas)
