from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from annolid.gui.widgets.zone_manager_utils import write_zone_json
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_analysis_engine import GenericZoneEngine
from annolid.postprocessing.zone_occupancy_policy import apply_zone_occupancy_policy
from annolid.postprocessing.zone_schema import build_zone_shape, load_zone_shapes


def _zone_specs():
    data = {
        "shapes": [
            build_zone_shape(
                "chamber_D",
                [[0, 0], [50, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                extra_flags={"zone_group": "chamber"},
            ),
            build_zone_shape(
                "chamber_south",
                [[50, 0], [100, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                extra_flags={"zone_group": "chamber"},
            ),
            build_zone_shape(
                "tether_D",
                [[0, 0], [30, 100]],
                shape_type="rectangle",
                zone_kind="tether",
                extra_flags={"zone_group": "tether"},
            ),
        ]
    }
    return load_zone_shapes(data)


def test_zone_occupancy_policy_forces_chamber_without_moving_centroid() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "frame_number": 1,
                "instance_name": "stim_D",
                "cx": 75.0,
                "cy": 50.0,
                "chamber_D": 0,
                "chamber_south": 1,
                "tether_D": 0,
            }
        ]
    )
    policy = {
        "instance_policies": [
            {
                "instance_name": "stim_D",
                "rules": [
                    {
                        "name": "stim_D_chamber_prior",
                        "zone_group": "chamber",
                        "mode": "force_one",
                        "zone": "chamber_D",
                    }
                ],
            }
        ]
    }

    result = apply_zone_occupancy_policy(dataframe, _zone_specs(), policy)

    row = result.dataframe.iloc[0]
    assert row["cx"] == 75.0
    assert row["cy"] == 50.0
    assert row["chamber_D"] == 1
    assert row["chamber_south"] == 0
    assert row["tether_D"] == 0
    assert result.audit.iloc[0]["rule_name"] == "stim_D_chamber_prior"


def test_zone_engine_counts_overlapping_zone_columns() -> None:
    engine = GenericZoneEngine(_zone_specs(), fps=10)
    dataframe = pd.DataFrame(
        [
            {
                "frame_number": 0,
                "instance_name": "stim_D",
                "cx": 75.0,
                "cy": 50.0,
                "chamber_D": 1,
                "chamber_south": 0,
                "tether_D": 1,
            }
        ]
    )

    result = engine.analyze_instance(dataframe, "stim_D")

    assert result.occupancy_frames["chamber_D"] == 1
    assert result.occupancy_frames["tether_D"] == 1


def test_tracking_analyzer_exports_zone_corrected_tracked_csv(tmp_path: Path) -> None:
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"fake")
    tracked_csv = tmp_path / "session_tracked.csv"
    tracking_csv = tmp_path / "session_tracking.csv"
    raw_df = pd.DataFrame(
        [
            {
                "frame_number": 1,
                "instance_name": "stim_D",
                "cx": 75.0,
                "cy": 50.0,
                "chamber_D": 0,
                "chamber_south": 1,
            }
        ]
    )
    raw_df.to_csv(tracked_csv, index=False)
    raw_df.to_csv(tracking_csv, index=False)
    zone_path = tmp_path / "session_zones.json"
    write_zone_json(
        zone_path,
        shapes=[
            build_zone_shape(
                "chamber_D",
                [[0, 0], [50, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                extra_flags={"zone_group": "chamber"},
            ),
            build_zone_shape(
                "chamber_south",
                [[50, 0], [100, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                extra_flags={"zone_group": "chamber"},
            ),
        ],
        image_path=str(video_path),
        image_width=100,
        image_height=100,
    )
    policy_path = tmp_path / "zone_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "instance_policies": [
                    {
                        "instance_name": "stim_D",
                        "rules": [
                            {
                                "name": "stim_D_legal_chamber",
                                "zone_group": "chamber",
                                "mode": "force_one",
                                "zone": "chamber_D",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    analyzer = TrackingResultsAnalyzer(
        video_path,
        zone_file=zone_path,
        fps=10,
        zone_policy_file=policy_path,
    )
    output_csv = analyzer.save_zone_corrected_tracked_csv()
    corrected = pd.read_csv(output_csv)
    audit = pd.read_csv(
        Path(output_csv).with_name("session_tracked_zone_corrected_audit.csv")
    )

    assert corrected.loc[0, "cx"] == 75.0
    assert corrected.loc[0, "chamber_D"] == 1
    assert corrected.loc[0, "chamber_south"] == 0
    assert audit.loc[0, "rule_name"] == "stim_D_legal_chamber"
