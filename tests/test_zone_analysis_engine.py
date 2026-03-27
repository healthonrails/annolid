from __future__ import annotations

from pathlib import Path

import pandas as pd

from annolid.gui.widgets.zone_manager_utils import write_zone_json
from annolid.postprocessing.zone_analysis_engine import GenericZoneEngine
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_schema import build_zone_shape, load_zone_shapes


def _build_test_engine() -> GenericZoneEngine:
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
                "center_chamber",
                [[50, 0], [100, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                phase="phase_1",
                occupant_role="neutral",
                access_state="open",
                description="center chamber",
            ),
            build_zone_shape(
                "mesh_edge",
                [[0, 0], [100, 20]],
                shape_type="rectangle",
                zone_kind="barrier_edge",
                phase="phase_1",
                occupant_role="unknown",
                access_state="blocked",
                description="mesh barrier edge",
            ),
        ]
    }
    return GenericZoneEngine(load_zone_shapes(zone_data), fps=10)


def test_generic_zone_engine_computes_occupancy_entries_transitions_and_dwell():
    engine = _build_test_engine()
    dataframe = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 10, "cy": 50},
            {"frame_number": 1, "instance_name": "mouse", "cx": 12, "cy": 55},
            {"frame_number": 2, "instance_name": "mouse", "cx": 70, "cy": 50},
            {"frame_number": 3, "instance_name": "mouse", "cx": 75, "cy": 55},
            {"frame_number": 4, "instance_name": "mouse", "cx": 80, "cy": 10},
            {"frame_number": 5, "instance_name": "mouse", "cx": 82, "cy": 15},
            {"frame_number": 6, "instance_name": "mouse", "cx": 72, "cy": 60},
            {"frame_number": 7, "instance_name": "mouse", "cx": 74, "cy": 62},
        ]
    )

    result = engine.analyze_instance(dataframe, "mouse")

    assert result.occupancy_frames["left_chamber"] == 2
    assert result.occupancy_frames["center_chamber"] == 4
    assert result.occupancy_frames["mesh_edge"] == 2
    assert result.entry_counts["left_chamber"] == 1
    assert result.entry_counts["center_chamber"] == 2
    assert result.entry_counts["mesh_edge"] == 1
    assert result.first_entry_frames["left_chamber"] == 0
    assert result.first_entry_frames["center_chamber"] == 2
    assert result.first_entry_frames["mesh_edge"] == 4
    assert result.dwell_frames["left_chamber"] == 2
    assert result.dwell_frames["center_chamber"] == 4
    assert result.dwell_frames["mesh_edge"] == 2
    assert result.barrier_adjacent_frames == 2
    assert result.transition_counts["left_chamber"]["center_chamber"] == 1
    assert result.transition_counts["center_chamber"]["mesh_edge"] == 1
    assert result.transition_counts["mesh_edge"]["center_chamber"] == 1
    assert result.outside_frames == 0
    assert len(result.segments) == 4

    summary = result.to_summary_row(fps=10, include_transition_columns=True)
    assert summary["occupancy_seconds__left_chamber"] == 0.2
    assert summary["entry_count__center_chamber"] == 2
    assert summary["first_entry_frame__center_chamber"] == 2
    assert summary["first_entry_seconds__mesh_edge"] == 0.4
    assert summary["barrier_adjacent_seconds"] == 0.2
    assert summary["transition_count__left_chamber__center_chamber"] == 1


def test_generic_zone_engine_ignores_frames_outside_any_zone():
    engine = _build_test_engine()
    dataframe = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 10, "cy": 50},
            {"frame_number": 1, "instance_name": "mouse", "cx": 500, "cy": 500},
            {"frame_number": 2, "instance_name": "mouse", "cx": 10, "cy": 50},
        ]
    )

    result = engine.analyze_instance(dataframe, "mouse")

    assert result.occupancy_frames["left_chamber"] == 2
    assert result.outside_frames == 1
    assert result.entry_counts["left_chamber"] == 2
    assert result.total_transitions() == 0


def test_tracking_results_analyzer_exports_generic_zone_metrics(tmp_path):
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"fake video container")
    tracked_csv = tmp_path / "session_tracked.csv"
    tracking_csv = tmp_path / "session_tracking.csv"

    dataframe = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 10, "cy": 50},
            {"frame_number": 1, "instance_name": "mouse", "cx": 15, "cy": 50},
            {"frame_number": 2, "instance_name": "mouse", "cx": 75, "cy": 50},
        ]
    )
    dataframe.to_csv(tracked_csv, index=False)
    dataframe.to_csv(tracking_csv, index=False)

    zone_path = tmp_path / "session_zones.json"
    write_zone_json(
        zone_path,
        shapes=[
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
        ],
        image_path=str(video_path),
        image_width=100,
        image_height=100,
    )

    analyzer = TrackingResultsAnalyzer(video_path, zone_file=zone_path, fps=10)
    output_csv = analyzer.save_zone_metrics_to_csv()
    summary = pd.read_csv(output_csv, index_col=0)

    assert "mouse" in summary.index
    assert summary.loc["mouse", "occupancy_frames__left_chamber"] == 2
    assert summary.loc["mouse", "occupancy_frames__right_chamber"] == 1
    assert summary.loc["mouse", "entry_count__right_chamber"] == 1
    assert summary.loc["mouse", "total_transitions"] == 1


def test_assay_profiles_reuse_same_zone_shapes_but_produce_different_summaries(
    tmp_path,
):
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"fake video container")
    tracked_csv = tmp_path / "session_tracked.csv"
    tracking_csv = tmp_path / "session_tracking.csv"

    dataframe = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 10, "cy": 50},
            {"frame_number": 1, "instance_name": "mouse", "cx": 15, "cy": 50},
            {"frame_number": 2, "instance_name": "mouse", "cx": 75, "cy": 50},
        ]
    )
    dataframe.to_csv(tracked_csv, index=False)
    dataframe.to_csv(tracking_csv, index=False)

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
            build_zone_shape(
                "mesh_edge",
                [[0, 0], [100, 10]],
                shape_type="rectangle",
                zone_kind="barrier_edge",
                phase="custom",
                occupant_role="unknown",
                access_state="blocked",
                description="mesh edge",
            ),
        ],
        image_path=str(video_path),
        image_width=100,
        image_height=100,
    )

    phase1_analyzer = TrackingResultsAnalyzer(
        video_path,
        zone_file=zone_path,
        fps=10,
        assay_profile="phase_1",
    )
    phase2_analyzer = TrackingResultsAnalyzer(
        video_path,
        zone_file=zone_path,
        fps=10,
        assay_profile="phase_2",
    )

    phase1_csv = phase1_analyzer.save_zone_metrics_to_csv(
        tmp_path / "phase1_zone_metrics.csv"
    )
    phase2_csv = phase2_analyzer.save_zone_metrics_to_csv(
        tmp_path / "phase2_zone_metrics.csv"
    )

    phase1_summary = pd.read_csv(phase1_csv, index_col=0)
    phase2_summary = pd.read_csv(phase2_csv, index_col=0)

    assert phase1_summary.loc["mouse", "assay_profile"] == "phase_1"
    assert phase2_summary.loc["mouse", "assay_profile"] == "phase_2"
    assert "profile_description" in phase1_summary.columns
    assert "metrics_computed" in phase1_summary.columns
    assert "included_zone_labels" in phase1_summary.columns
    assert "blocked_zone_labels" in phase1_summary.columns
    assert "occupancy_frames__right_blocked" not in phase1_summary.columns
    assert phase2_summary.loc["mouse", "occupancy_frames__right_blocked"] == 1
    assert phase1_summary.loc["mouse", "occupancy_frames__left_open"] == 2
    assert phase2_summary.loc["mouse", "occupancy_frames__left_open"] == 2
    assert phase1_summary.loc["mouse", "barrier_adjacent_frames"] == 0
    assert phase2_summary.loc["mouse", "barrier_adjacent_frames"] == 0


def test_assay_summary_report_labels_profile_and_zone_access(tmp_path):
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"fake video container")
    tracked_csv = tmp_path / "session_tracked.csv"
    tracking_csv = tmp_path / "session_tracking.csv"

    dataframe = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 10, "cy": 50},
            {"frame_number": 1, "instance_name": "mouse", "cx": 15, "cy": 50},
            {"frame_number": 2, "instance_name": "mouse", "cx": 75, "cy": 50},
        ]
    )
    dataframe.to_csv(tracked_csv, index=False)
    dataframe.to_csv(tracking_csv, index=False)

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
            build_zone_shape(
                "mesh_edge",
                [[0, 0], [100, 10]],
                shape_type="rectangle",
                zone_kind="barrier_edge",
                phase="custom",
                occupant_role="unknown",
                access_state="blocked",
                description="mesh edge",
            ),
        ],
        image_path=str(video_path),
        image_width=100,
        image_height=100,
    )

    analyzer = TrackingResultsAnalyzer(
        video_path,
        zone_file=zone_path,
        fps=10,
        assay_profile="phase_1",
    )
    md_path = analyzer.save_assay_summary_report(tmp_path / "reports")
    csv_path = tmp_path / "reports" / "session_phase_1_assay_summary.csv"

    assert md_path.endswith("session_phase_1_assay_summary.md")
    assert csv_path.is_file()

    markdown = Path(md_path).read_text()
    assert "# Phase 1 Assay Summary" in markdown
    assert "- **Profile:** `phase_1`" in markdown
    assert "## Phase Rules" in markdown
    assert "Profile-included zones" in markdown
    assert "`right_blocked`" in markdown
    assert "Accessible zones" in markdown
    assert "Blocked zones" in markdown
    assert "## Metrics Computed" in markdown
    assert "`occupancy_frames`" in markdown
