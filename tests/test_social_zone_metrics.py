from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from annolid.gui.widgets.zone_manager_utils import write_zone_json
from annolid.postprocessing.social_zone_metrics import (
    build_anchor_dataframe,
    compute_pairwise_centroid_summary,
    load_pose_keypoint_observations,
)
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_schema import build_zone_shape


def _write_keypoint_frame(
    folder: Path,
    frame_number: int,
    *,
    rover_nose: tuple[float, float] | None = None,
    stim_nose: tuple[float, float] | None = None,
) -> None:
    shapes = []
    if rover_nose is not None:
        shapes.append(
            {
                "label": "nose",
                "shape_type": "point",
                "points": [[rover_nose[0], rover_nose[1]]],
                "instance_label": "rover",
                "flags": {},
            }
        )
    if stim_nose is not None:
        shapes.append(
            {
                "label": "nose",
                "shape_type": "point",
                "points": [[stim_nose[0], stim_nose[1]]],
                "instance_label": "stim",
                "flags": {},
            }
        )
    payload = {
        "version": "Annolid",
        "frame_index": frame_number,
        "imageHeight": 100,
        "imageWidth": 100,
        "imagePath": f"session_{frame_number:09d}.png",
        "shapes": shapes,
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"session_{frame_number:09d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_anchor_dataframe_prefers_nose_when_available(tmp_path):
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"fake video")
    keypoint_folder = tmp_path / "session"
    _write_keypoint_frame(
        keypoint_folder,
        0,
        rover_nose=(12, 50),
        stim_nose=(78, 50),
    )

    tracking_df = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "rover", "cx": 30, "cy": 50},
            {"frame_number": 0, "instance_name": "stim", "cx": 70, "cy": 50},
        ]
    )

    keypoint_df = load_pose_keypoint_observations(video_path)
    anchor_df = build_anchor_dataframe(tracking_df, keypoint_df)

    rover_row = anchor_df[anchor_df["instance_name"] == "rover"].iloc[0]
    stim_row = anchor_df[anchor_df["instance_name"] == "stim"].iloc[0]

    assert rover_row["anchor_source"] == "nose"
    assert rover_row["anchor_x"] == 12
    assert rover_row["anchor_y"] == 50
    assert stim_row["anchor_source"] == "nose"
    assert stim_row["anchor_x"] == 78


def test_pairwise_centroid_summary_reports_neighbor_distance(tmp_path):
    dataframe = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "rover", "cx": 10, "cy": 10},
            {"frame_number": 0, "instance_name": "stim", "cx": 30, "cy": 10},
            {"frame_number": 1, "instance_name": "rover", "cx": 12, "cy": 10},
            {"frame_number": 1, "instance_name": "stim", "cx": 32, "cy": 10},
        ]
    )

    summary_df, detail_df = compute_pairwise_centroid_summary(dataframe)

    assert not summary_df.empty
    assert "rover" in summary_df.index
    assert summary_df.loc["rover", "mean_nearest_neighbor_distance_px"] == 20.0
    assert summary_df.loc["rover", "min_nearest_neighbor_distance_px"] == 20.0
    assert len(detail_df) == 2


def test_social_summary_export_uses_nose_anchor_and_latency_reference(tmp_path):
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"fake video")
    tracking_csv = tmp_path / "session_tracking.csv"
    tracked_csv = tmp_path / "session_tracked.csv"

    pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "rover", "cx": 30, "cy": 50},
            {"frame_number": 1, "instance_name": "rover", "cx": 30, "cy": 50},
            {"frame_number": 0, "instance_name": "stim", "cx": 70, "cy": 50},
            {"frame_number": 1, "instance_name": "stim", "cx": 70, "cy": 50},
        ]
    ).to_csv(tracking_csv, index=False)
    pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "rover", "cx": 30, "cy": 50},
            {"frame_number": 1, "instance_name": "rover", "cx": 30, "cy": 50},
            {"frame_number": 0, "instance_name": "stim", "cx": 70, "cy": 50},
            {"frame_number": 1, "instance_name": "stim", "cx": 70, "cy": 50},
        ]
    ).to_csv(tracked_csv, index=False)

    keypoint_folder = tmp_path / "session"
    _write_keypoint_frame(
        keypoint_folder,
        0,
        rover_nose=(10, 50),
        stim_nose=(78, 50),
    )
    _write_keypoint_frame(
        keypoint_folder,
        1,
        rover_nose=(10, 50),
        stim_nose=(78, 50),
    )

    zone_path = tmp_path / "session_zones.json"
    write_zone_json(
        zone_path,
        shapes=[
            build_zone_shape(
                "left_social_zone",
                [[0, 25], [20, 75]],
                shape_type="rectangle",
                zone_kind="interaction_zone",
                phase="social",
                occupant_role="rover",
                access_state="open",
                description="rover-side social zone",
            ),
            build_zone_shape(
                "right_social_zone",
                [[80, 25], [100, 75]],
                shape_type="rectangle",
                zone_kind="interaction_zone",
                phase="social",
                occupant_role="rover",
                access_state="open",
                description="rover-side social zone",
            ),
        ],
        image_path=str(video_path),
        image_width=100,
        image_height=100,
    )

    analyzer = TrackingResultsAnalyzer(video_path, zone_file=zone_path, fps=10)
    md_path = analyzer.save_social_summary_report(
        tmp_path / "reports",
        assay_profile="generic",
        latency_reference_frame=0,
    )

    csv_path = tmp_path / "reports" / "session_generic_social_summary.csv"
    summary_df = pd.read_csv(csv_path, index_col=0)

    assert md_path.endswith("session_generic_social_summary.md")
    assert summary_df.loc["rover", "anchor_source"] == "nose"
    assert summary_df.loc["rover", "latency_reference_text"] == "frame 0"
    assert summary_df.loc["rover", "first_entry_frame__left_social_zone"] == 0
    assert summary_df.loc["rover", "first_entry_seconds__left_social_zone"] == 0.0
    assert summary_df.loc["rover", "latency_frame__left_social_zone"] == 0
    assert "mean_nearest_neighbor_distance_px" in summary_df.columns
