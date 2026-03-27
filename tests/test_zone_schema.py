from __future__ import annotations

import json

import pandas as pd

from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_schema import build_zone_shape, load_zone_shapes


def test_build_zone_shape_adds_explicit_semantics():
    shape = build_zone_shape(
        "north_chamber",
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        zone_kind="chamber",
        phase="phase_1",
        occupant_role="stim",
        access_state="open",
    )

    assert shape["flags"]["semantic_type"] == "zone"
    assert shape["flags"]["shape_category"] == "zone"
    assert shape["flags"]["zone_kind"] == "chamber"
    assert shape["flags"]["phase"] == "phase_1"
    assert shape["flags"]["occupant_role"] == "stim"
    assert shape["flags"]["access_state"] == "open"


def test_load_zone_shapes_supports_explicit_and_legacy_zone_shapes():
    zone_data = {
        "shapes": [
            {
                "label": "north_chamber",
                "shape_type": "polygon",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "description": "",
                "flags": {
                    "semantic_type": "zone",
                    "zone_kind": "chamber",
                    "phase": "phase_2",
                    "occupant_role": "rover",
                },
            },
            {
                "label": "legacy_zone",
                "shape_type": "rectangle",
                "points": [[20, 20], [40, 40]],
                "description": "legacy zone marker",
                "flags": {},
            },
            {
                "label": "analysis_roi",
                "shape_type": "polygon",
                "points": [[50, 50], [60, 50], [60, 60]],
                "description": "",
                "flags": {},
            },
        ]
    }

    zone_specs = load_zone_shapes(zone_data)

    assert [spec.display_label for spec in zone_specs] == [
        "north_chamber",
        "legacy_zone",
    ]
    assert zone_specs[0].zone_kind == "chamber"
    assert zone_specs[0].phase == "phase_2"
    assert zone_specs[1].inferred_from_legacy is True
    assert zone_specs[1].compatibility_mode == "legacy_compat"
    assert zone_specs[1].source_shape_type == "rectangle"
    assert len(zone_specs[1].analysis_points) == 5


def test_tracking_results_analyzer_uses_normalized_zone_shapes_without_mutation(
    tmp_path,
):
    video_path = tmp_path / "session.mp4"
    zone_path = tmp_path / "session_zone.json"

    zone_data = {
        "shapes": [
            {
                "label": "zone_legacy",
                "shape_type": "rectangle",
                "points": [[10, 10], [30, 30]],
                "description": "legacy zone",
                "flags": {},
            }
        ]
    }
    zone_path.write_text(json.dumps(zone_data), encoding="utf-8")

    analyzer = TrackingResultsAnalyzer(
        str(video_path), zone_file=str(zone_path), fps=30
    )
    analyzer.merged_df = pd.DataFrame(
        {
            "frame_number": [0, 1, 2],
            "instance_name": ["mouse", "mouse", "mouse"],
            "cx_tracking": [15, 25, 50],
            "cy_tracking": [15, 25, 50],
        }
    )

    result = analyzer.determine_time_in_zone("mouse")

    assert result["zone_legacy"] == 2
    assert analyzer.zone_data["shapes"][0]["points"] == [[10, 10], [30, 30]]
