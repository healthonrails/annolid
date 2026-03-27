from __future__ import annotations


import pandas as pd

from annolid.gui.widgets.zone_manager_utils import (
    generate_arena_layout_preset,
    shape_to_zone_payload,
    write_zone_json,
)
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_analysis_engine import GenericZoneEngine
from annolid.postprocessing.zone_schema import build_zone_shape, load_zone_shapes


def test_legacy_zone_json_is_accepted():
    zone_data = {
        "shapes": [
            {
                "label": "legacy_zone",
                "shape_type": "rectangle",
                "points": [[10, 10], [30, 30]],
                "description": "legacy zone marker",
                "flags": {},
            }
        ]
    }

    zone_specs = load_zone_shapes(zone_data)

    assert len(zone_specs) == 1
    assert zone_specs[0].inferred_from_legacy is True
    assert zone_specs[0].compatibility_mode == "legacy_compat"
    assert zone_specs[0].display_label == "legacy_zone"


def test_explicit_zone_metadata_is_preserved():
    payload = shape_to_zone_payload(
        build_zone_shape(
            "north_chamber",
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            zone_kind="chamber",
            phase="phase_1",
            occupant_role="stim",
            access_state="open",
            description="north chamber",
        ),
        tags=["phase_1", "corner"],
    )

    assert payload["flags"]["semantic_type"] == "zone"
    assert payload["flags"]["zone_kind"] == "chamber"
    assert payload["flags"]["phase"] == "phase_1"
    assert payload["flags"]["occupant_role"] == "stim"
    assert payload["flags"]["access_state"] == "open"
    assert payload["flags"]["tags"] == ["phase_1", "corner"]


def test_chamber_preset_generates_nine_shapes():
    shapes = generate_arena_layout_preset("3x3_chamber", 900, 900)

    assert len(shapes) == 9
    assert shapes[0].label == "north_west_chamber"
    assert shapes[4].label == "center_chamber"
    assert shapes[8].label == "south_east_chamber"
    assert all(shape.flags["zone_kind"] == "chamber" for shape in shapes)


def test_per_zone_occupancy_counts_frames_inside_each_zone():
    zone_data = {
        "shapes": [
            build_zone_shape(
                "left_chamber",
                [[0, 0], [50, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                phase="phase_1",
                occupant_role="stim",
                access_state="open",
                description="left chamber",
            ),
            build_zone_shape(
                "right_chamber",
                [[50, 0], [100, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                phase="phase_1",
                occupant_role="rover",
                access_state="open",
                description="right chamber",
            ),
        ]
    }
    engine = GenericZoneEngine(load_zone_shapes(zone_data), fps=10)
    dataframe = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 10, "cy": 50},
            {"frame_number": 1, "instance_name": "mouse", "cx": 15, "cy": 50},
            {"frame_number": 2, "instance_name": "mouse", "cx": 75, "cy": 50},
        ]
    )

    result = engine.analyze_instance(dataframe, "mouse")

    assert result.occupancy_frames["left_chamber"] == 2
    assert result.occupancy_frames["right_chamber"] == 1
    assert result.entry_counts["right_chamber"] == 1


def test_phase_dependent_metrics_change_with_same_saved_zones(tmp_path):
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"fake video")
    tracked_csv = tmp_path / "session_tracked.csv"
    tracking_csv = tmp_path / "session_tracking.csv"

    pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 10, "cy": 50},
            {"frame_number": 1, "instance_name": "mouse", "cx": 15, "cy": 50},
            {"frame_number": 2, "instance_name": "mouse", "cx": 75, "cy": 50},
        ]
    ).to_csv(tracked_csv, index=False)
    pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 10, "cy": 50},
            {"frame_number": 1, "instance_name": "mouse", "cx": 15, "cy": 50},
            {"frame_number": 2, "instance_name": "mouse", "cx": 75, "cy": 50},
        ]
    ).to_csv(tracking_csv, index=False)

    zone_path = tmp_path / "session_zones.json"
    write_zone_json(
        zone_path,
        shapes=[
            build_zone_shape(
                "left_open",
                [[0, 0], [50, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                phase="custom",
                occupant_role="stim",
                access_state="open",
                description="left chamber open",
            ),
            build_zone_shape(
                "right_blocked",
                [[50, 0], [100, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                phase="custom",
                occupant_role="stim",
                access_state="blocked",
                description="right chamber blocked",
            ),
        ],
        image_path=str(video_path),
        image_width=100,
        image_height=100,
    )

    phase1 = TrackingResultsAnalyzer(
        video_path,
        zone_file=zone_path,
        fps=10,
        assay_profile="phase_1",
    )
    phase2 = TrackingResultsAnalyzer(
        video_path,
        zone_file=zone_path,
        fps=10,
        assay_profile="phase_2",
    )

    phase1_csv = phase1.save_zone_metrics_to_csv(tmp_path / "phase1_metrics.csv")
    phase2_csv = phase2.save_zone_metrics_to_csv(tmp_path / "phase2_metrics.csv")

    phase1_summary = pd.read_csv(phase1_csv, index_col=0)
    phase2_summary = pd.read_csv(phase2_csv, index_col=0)

    assert phase1_summary.loc["mouse", "assay_profile"] == "phase_1"
    assert phase2_summary.loc["mouse", "assay_profile"] == "phase_2"
    assert "occupancy_frames__right_blocked" not in phase1_summary.columns
    assert phase2_summary.loc["mouse", "occupancy_frames__right_blocked"] == 1
    assert phase1_summary.loc["mouse", "metrics_computed"].startswith(
        "occupancy_frames"
    )
