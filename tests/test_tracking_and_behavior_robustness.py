from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
import pickle
from pathlib import Path
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from annolid.behavior.irr import (
    PointBehaviorEvent,
    _rasterize_series,
    compute_binary_cohens_kappa,
    compute_fleiss_kappa,
)
from annolid.behavior.polygon_classifier_workflow import (
    predict_polygon_classifier_csv,
    train_polygon_classifier,
)
from annolid.behavior.tcn import load_tcn_checkpoint, safe_load_behavior_checkpoint
from annolid.utils.safe_torch_load import SafeCheckpointMetadata
from annolid.behavior.time_budget import (
    BehaviorInterval,
    compute_time_budget,
    extract_behavior_intervals,
)
from annolid.gui.widgets.zone_manager_utils import write_zone_json
from annolid.postprocessing.freezing_event_analyzer import FreezingEventAnalyzer
from annolid.postprocessing.motion_analyzer import analyze_motion_index
from annolid.postprocessing.quality_control import TracksResults
from annolid.postprocessing.social_zone_metrics import compute_pairwise_centroid_summary
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_analysis_engine import GenericZoneEngine
from annolid.postprocessing.zone_occupancy_policy import apply_zone_occupancy_policy
from annolid.postprocessing.zone_schema import (
    ZoneShapeSpec,
    build_zone_shape,
    load_zone_shapes,
    zone_shape_covers_point,
    zone_shape_covers_points_xy,
)
from annolid.tracker.bbox_iou_tracker import BBoxIOUTracker
from annolid.tracker.simple_instances import SimpleInstances
from annolid.tracking.identity_continuity import detect_centroid_assignment_ambiguity


@dataclass
class _CustomLabConfig:
    """Harmless custom research dataclass used to verify extra_globals / trust_checkpoint."""

    experiment_tag: str = "cohort_alpha"
    fps: float = 30.0


class _MaliciousPickleGadget:
    """Simulated untrusted pickle gadget that must never execute on checkpoint load."""

    def __reduce__(self):
        import os

        return (os.getpid, ())


class _EvalPickleGadget:
    """Simulated builtins.eval gadget that must be blocked even under trust_checkpoint=True."""

    def __reduce__(self):
        return (eval, ("1 + 1",))


class _ReflectionPickleGadget:
    """Simulated operator.attrgetter gadget that must be blocked even under trust_checkpoint=True."""

    def __reduce__(self):
        return (operator.attrgetter("name"), (Path("/tmp"),))


class _BenignModuleReduceGadget:
    """Simulated non-denylisted module REDUCE callable (json.loads) — must still be blocked."""

    def __reduce__(self):
        return (json.loads, ('{"ok": 1}',))


class _SetstateHookClass:
    """Local class with a custom __setstate__ hook — must never execute on load."""

    SIDE_EFFECT_TRIGGERED = False

    def __init__(self) -> None:
        self.value = 42

    def __setstate__(self, state: dict) -> None:
        type(self).SIDE_EFFECT_TRIGGERED = True
        self.__dict__.update(state)


class _CustomNewClass:
    """Local class with a custom __new__ constructor — must never execute on load."""

    SIDE_EFFECT_TRIGGERED = False

    def __new__(cls, *args, **kwargs):
        cls.SIDE_EFFECT_TRIGGERED = True
        return super().__new__(cls)


class _CustomSetattrClass:
    """Local class with a custom __setattr__ hook — must never execute on load."""

    SIDE_EFFECT_TRIGGERED = False

    def __setattr__(self, key: str, value: object) -> None:
        type(self).SIDE_EFFECT_TRIGGERED = True
        object.__setattr__(self, key, value)


def test_identity_ambiguity_ignores_intra_animal_body_part_crossover():
    # mouse_1 turns around so its head crosses near its torso; should NOT trigger
    # an identity swap warning because 'mouse_1:head' and 'mouse_1:torso' have
    # distinct semantic roles.
    prev_centroids = {
        "mouse_1": (50.0, 50.0),
        "mouse_1:head": (40.0, 50.0),
        "mouse_1:torso": (60.0, 50.0),
    }
    curr_centroids = {
        "mouse_1": (50.0, 50.0),
        "mouse_1:head": (60.0, 50.0),
        "mouse_1:torso": (40.0, 50.0),
    }
    assert (
        detect_centroid_assignment_ambiguity(
            prev_centroids,
            curr_centroids,
            frame_shape=(100, 100),
            min_relative_improvement=0.2,
            min_absolute_improvement_px=1.0,
        )
        is None
    )

    # Genuine cross-animal head swap between mouse_1:head and mouse_2:head MUST
    # still be detected.
    prev_two_animals = {
        "mouse_1:head": (10.0, 20.0),
        "mouse_2:head": (30.0, 20.0),
    }
    curr_swapped = {
        "mouse_1:head": (29.5, 20.0),
        "mouse_2:head": (10.5, 20.0),
    }
    warning = detect_centroid_assignment_ambiguity(
        prev_two_animals,
        curr_swapped,
        frame_shape=(100, 100),
        min_relative_improvement=0.2,
        min_absolute_improvement_px=1.0,
    )
    assert warning is not None
    changed_labels = {src for src, _ in warning.changed_assignments}
    assert "mouse_1:head" in changed_labels
    assert "mouse_2:head" in changed_labels


def test_bbox_iou_tracker_handles_empty_frames_zero_dims_and_mask_recovery():
    tracker = BBoxIOUTracker(
        video_height=0,
        video_width=0,
        max_lost_frame_count=3,
        track_iou_threshold=0.3,
        min_instance_period=0,
    )

    # Frame 0: 1 detection with a segmentation mask
    inst0 = SimpleInstances(image_size=(20, 20))
    inst0.pred_boxes = torch.tensor([[2.0, 2.0, 10.0, 10.0]], dtype=torch.float32)
    inst0.scores = torch.tensor([0.95], dtype=torch.float32)
    inst0.pred_classes = torch.tensor([0], dtype=torch.int64)
    mask0 = torch.zeros((1, 20, 20), dtype=torch.bool)
    mask0[0, 2:10, 2:10] = True
    inst0.pred_masks = mask0

    out0 = tracker.update(inst0)
    assert len(out0) == 1
    first_id = int(out0.ID[0])

    # Frame 1: 0 detections (1D empty pred_boxes) — must normalize to (0, 4) and
    # synthesize lost track with pred_masks without SimpleInstances.cat mismatch!
    empty_inst = SimpleInstances(image_size=(20, 20))
    empty_inst.pred_boxes = torch.FloatTensor([])
    empty_inst.scores = torch.FloatTensor([])
    empty_inst.pred_classes = torch.IntTensor([])
    empty_inst.pred_masks = torch.empty((0, 20, 20), dtype=torch.bool)

    out1 = tracker.update(empty_inst)
    assert len(out1) == 1
    assert int(out1.ID[0]) == first_id
    assert out1.pred_boxes.shape == (1, 4)
    assert out1.pred_masks.dtype == torch.bool


def test_bowtie_zone_and_nan_centroids_do_not_crash_or_teleport_to_origin():
    # Self-intersecting bow-tie polygon: (0,0)->(20,20)->(20,0)->(0,20)
    # Left triangle covers (5, 10); right triangle covers (15, 10).
    bowtie_spec = ZoneShapeSpec(
        label="origin_bowtie",
        points=[[0.0, 0.0], [20.0, 20.0], [20.0, 0.0], [0.0, 20.0]],
        shape_type="polygon",
        source_shape_type="polygon",
    )
    assert zone_shape_covers_point(bowtie_spec, (15.0, 10.0)) is True
    assert zone_shape_covers_point(bowtie_spec, (float("nan"), 10.0)) is False

    vec_mask = zone_shape_covers_points_xy(
        bowtie_spec,
        np.array([5.0, 15.0, float("nan"), 100.0]),
        np.array([10.0, 10.0, 10.0, 100.0]),
    )
    assert vec_mask.tolist() == [True, True, False, False]

    # Origin-covering rectangle [0, 0] -> [50, 50]: NaN/None centroids must NOT
    # be coerced to (0.0, 0.0) and falsely counted inside origin_zone!
    zone_data = {
        "shapes": [
            build_zone_shape(
                "origin_zone",
                [[0, 0], [50, 50]],
                shape_type="rectangle",
                zone_kind="chamber",
            )
        ]
    }
    engine = GenericZoneEngine(load_zone_shapes(zone_data), fps=10)
    df = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": None, "cy": None},
            {
                "frame_number": 1,
                "instance_name": "mouse",
                "cx": float("nan"),
                "cy": float("nan"),
            },
            {"frame_number": 2, "instance_name": "mouse", "cx": 10.0, "cy": 10.0},
        ]
    )
    res = engine.analyze_instance(df, "mouse")
    assert res.occupancy_frames.get("origin_zone", 0) == 1
    assert res.outside_frames == 2


def test_tracking_results_analyzer_preserves_anchor_part_across_profiles(
    tmp_path: Path,
):
    video_path = tmp_path / "assay.mp4"
    video_path.write_bytes(b"fake")
    # Whole body centroid is in right_chamber (x=75), while mouse:head is in left_chamber (x=15)
    df = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 75.0, "cy": 50.0},
            {"frame_number": 0, "instance_name": "mouse:head", "cx": 15.0, "cy": 50.0},
            {"frame_number": 1, "instance_name": "mouse", "cx": 75.0, "cy": 50.0},
            {"frame_number": 1, "instance_name": "mouse:head", "cx": 15.0, "cy": 50.0},
        ]
    )
    df.to_csv(tmp_path / "assay_tracking.csv", index=False)
    df.to_csv(tmp_path / "assay_tracked.csv", index=False)

    zone_path = tmp_path / "zones.json"
    write_zone_json(
        zone_path,
        shapes=[
            build_zone_shape(
                "left_chamber",
                [[0, 0], [50, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                access_state="open",
            ),
            build_zone_shape(
                "right_chamber",
                [[50, 0], [100, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                access_state="open",
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
        anchor_part="head",
    )
    occ = analyzer.determine_time_in_zone("mouse")
    assert occ.get("left_chamber", 0) == 2
    assert occ.get("right_chamber", 0) == 0

    metrics_csv = analyzer.save_zone_metrics_to_csv(tmp_path / "zone_metrics.csv")
    summary = pd.read_csv(metrics_csv, index_col=0)
    assert list(summary.index) == ["mouse"]
    assert summary.loc["mouse", "occupancy_frames__left_chamber"] == 2

    # Single whole-animal dataset: merge_and_calculate_distance preserves canonical columns
    analyzer.merge_and_calculate_distance()
    assert list(analyzer.distances_df.columns) == [
        "frame_number",
        "instance_name_1",
        "instance_name_2",
        "distance",
    ]
    assert analyzer.distances_df.empty


def test_pairwise_centroid_summary_three_animals_and_body_parts_exact_nearest_neighbor():
    # mouse_1 at (0, 0), mouse_2 at (5, 0), mouse_3 at (505, 0), plus mouse_1:torso at (0.5, 0)
    df = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse_1", "cx": 0.0, "cy": 0.0},
            {"frame_number": 0, "instance_name": "mouse_1:torso", "cx": 0.5, "cy": 0.0},
            {"frame_number": 0, "instance_name": "mouse_2", "cx": 5.0, "cy": 0.0},
            {"frame_number": 0, "instance_name": "mouse_3", "cx": 505.0, "cy": 0.0},
            {"frame_number": 1, "instance_name": "mouse_1", "cx": 0.0, "cy": 0.0},
            {"frame_number": 1, "instance_name": "mouse_1:torso", "cx": 0.5, "cy": 0.0},
            {"frame_number": 1, "instance_name": "mouse_2", "cx": 5.0, "cy": 0.0},
            {"frame_number": 1, "instance_name": "mouse_3", "cx": 505.0, "cy": 0.0},
        ]
    )
    summary_df, detail_df = compute_pairwise_centroid_summary(df)

    # Body-part row must be excluded from animal-to-animal neighbor summary
    assert "mouse_1:torso" not in summary_df.index
    assert set(summary_df.index) == {"mouse_1", "mouse_2", "mouse_3"}
    # 3 animals x 2 frames = 6 undirected pairs total
    assert len(detail_df) == 6

    # mouse_1's nearest neighbor on both frames is mouse_2 at 5.0 px, while
    # explicit all-conspecifics group dispersion columns preserve (5 + 505)/2 = 255.0 px!
    assert summary_df.loc[
        "mouse_1", "mean_nearest_neighbor_distance_px"
    ] == pytest.approx(5.0)
    assert summary_df.loc[
        "mouse_1", "min_nearest_neighbor_distance_px"
    ] == pytest.approx(5.0)
    assert summary_df.loc["mouse_1", "closest_neighbor_label"] == "mouse_2"
    assert summary_df.loc["mouse_1", "neighbor_observations"] == 2
    assert summary_df.loc["mouse_1", "mean_pairwise_distance_px"] == pytest.approx(
        255.0
    )
    assert summary_df.loc["mouse_1", "max_pairwise_distance_px"] == pytest.approx(505.0)
    assert summary_df.loc["mouse_1", "pairwise_observations"] == 4

    # mouse_3's nearest neighbor on both frames is mouse_2 at 500.0 px
    assert summary_df.loc[
        "mouse_3", "mean_nearest_neighbor_distance_px"
    ] == pytest.approx(500.0)
    assert summary_df.loc["mouse_3", "closest_neighbor_label"] == "mouse_2"


def test_zone_occupancy_policy_ignores_nan_zone_flags():
    specs = load_zone_shapes(
        {
            "shapes": [
                build_zone_shape(
                    "chamber_A",
                    [[0, 0], [50, 100]],
                    shape_type="rectangle",
                    zone_kind="chamber",
                    extra_flags={"zone_group": "chamber"},
                ),
                build_zone_shape(
                    "chamber_B",
                    [[50, 0], [100, 100]],
                    shape_type="rectangle",
                    zone_kind="chamber",
                    extra_flags={"zone_group": "chamber"},
                ),
            ]
        }
    )
    df = pd.DataFrame(
        [
            {
                "frame_number": 0,
                "instance_name": "mouse",
                "cx": 25.0,
                "cy": 50.0,
                "chamber_A": 1,
                "chamber_B": float("nan"),
            }
        ]
    )
    policy = {
        "instance_policies": [
            {
                "instance_name": "mouse",
                "rules": [
                    {
                        "name": "prefer_B_only_when_active",
                        "zone_group": "chamber",
                        "mode": "prefer",
                        "zones": ["chamber_B"],
                    }
                ],
            }
        ]
    }
    result = apply_zone_occupancy_policy(df, specs, policy)
    # chamber_B is NaN (not truthy), so chamber_A must remain 1!
    assert result.dataframe.loc[0, "chamber_A"] == 1


def test_freezing_analyzer_motion_analyzer_and_qc_edge_cases(tmp_path: Path):
    # 1. FreezingEventAnalyzer: sentinel -1.0 ignored, frame gaps split bouts,
    #    missing 'timestamps' column handled gracefully.
    rows = []
    # Frames 0..4: -1.0 optical-flow unavailable sentinel -> NOT freezing
    for f in range(0, 5):
        rows.append(
            {"frame_number": f, "instance_name": "mouse_1", "motion_index": -1.0}
        )
    # Frames 10..13 (4 frames, unsorted): valid freezing (< 0.5)
    for f in [12, 10, 13, 11]:
        rows.append(
            {"frame_number": f, "instance_name": "mouse_1", "motion_index": 0.05}
        )
    # Gap from frame 13 -> 50! Frames 50..52 (3 frames) must NOT merge with 10..13!
    for f in range(50, 53):
        rows.append(
            {"frame_number": f, "instance_name": "mouse_1", "motion_index": 0.05}
        )
    csv_path = tmp_path / "freezing.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary = FreezingEventAnalyzer(
        str(csv_path), "mouse_1", threshold=0.5, min_duration=4
    ).analyze()
    assert len(summary) == 1
    assert int(summary.iloc[0]["start_frame"]) == 10
    assert int(summary.iloc[0]["end_frame"]) == 13
    assert list(summary.columns) == [
        "freezing_group",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
    ]

    # 2. analyze_motion_index closes matplotlib figures and handles multi-animal CSVs
    motion_csv = tmp_path / "multi_motion.csv"
    pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "m1", "motion_index": 0.0},
            {"frame_number": 1, "instance_name": "m1", "motion_index": 10.0},
            {"frame_number": 2, "instance_name": "m2", "motion_index": 12.0},
            {"frame_number": 3, "instance_name": "m1", "motion_index": 0.2},
            {"frame_number": 4, "instance_name": "m2", "motion_index": 0.3},
        ]
        * 5
    ).to_csv(motion_csv, index=False)
    open_before = len(plt.get_fignums())
    out_dir = tmp_path / "dash_out"
    analyze_motion_index(str(motion_csv), str(out_dir))
    assert (out_dir / "multi_motion_dashboard.png").is_file()
    assert len(plt.get_fignums()) == open_before

    # 3. TracksResults: empty instance missing frames & None width switch_left_right
    qc = TracksResults(tracking_csv=str(motion_csv))
    assert qc.get_missing_instance_frames("nonexistent_animal") == set()
    assert (
        TracksResults.switch_left_right(
            {"instance_name": "Left_mouse", "cx": 900.0}, width=None
        )
        == "Left_mouse"
    )


def test_time_budget_hhmmss_trailing_comma_and_irr_vectorized_parity(tmp_path: Path):
    # Trailing comma creates key=None in csv.DictReader; HH:MM:SS + timestamp alias
    events_csv = tmp_path / "events.csv"
    events_csv.write_text(
        "Timestamp,Subject,Behavior,Event,\n"
        "00:01:00.0,MouseA,grooming,start,extra_cell\n"
        "00:01:05.5,MouseA,grooming,stop,extra_cell\n"
        "nan,MouseA,grooming,start,extra_cell\n",
        encoding="utf-8",
    )
    rows, warnings = compute_time_budget(events_csv)
    assert len(rows) == 1
    assert rows[0].subject == "MouseA"
    assert rows[0].behavior == "grooming"
    assert rows[0].total_duration == pytest.approx(5.5)
    assert any("missing fields" in w for w in warnings)

    # Same-timestamp abutting stop + start in user log order
    ivs, iv_warns = extract_behavior_intervals(
        [
            {"time": "1.0", "subject": "S1", "behavior": "sniff", "event": "start"},
            {"time": "3.0", "subject": "S1", "behavior": "sniff", "event": "stop"},
            {"time": "3.0", "subject": "S1", "behavior": "sniff", "event": "start"},
            {"time": "5.0", "subject": "S1", "behavior": "sniff", "event": "stop"},
        ]
    )
    assert not iv_warns
    assert [(iv.start, iv.end) for iv in ivs] == [(1.0, 3.0), (3.0, 5.0)]

    # Vectorized IRR parity against multi-bin, midpoint-fallback, zero-duration, and point events
    seq_a = [1, 1, 0, 1, 0, 0, 1, 0] * 50
    seq_b = [1, 0, 0, 1, 1, 0, 1, 0] * 50
    seq_c = [1, 1, 0, 0, 0, 0, 1, 1] * 50
    stats = compute_binary_cohens_kappa(seq_a, seq_b)
    assert -1.0 <= stats["cohens_kappa"] <= 1.0
    assert stats["observed_agreement"] == pytest.approx(6.0 / 8.0)
    fk = compute_fleiss_kappa([seq_a, seq_b, seq_c])
    assert -1.0 <= fk <= 1.0

    grid = _rasterize_series(
        intervals=[
            BehaviorInterval("S1", "b", 0.02, 0.48),  # spans bins 0..4 at bin=0.1
            BehaviorInterval(
                "S1", "b", 0.68, 0.72
            ),  # short interval straddling 0.7 -> midpoint fallback
            BehaviorInterval("S1", "b", 0.90, 0.90),  # zero-duration interval at bin 9
        ],
        points=[PointBehaviorEvent("S1", "b", 0.85, 1)],
        window_start=0.0,
        window_end=1.0,
        bin_seconds=0.1,
        n_bins=10,
    )
    assert grid[:5] == [1, 1, 1, 1, 1]
    assert grid[8] == 1
    assert grid[9] == 1


def test_polygon_and_tcn_checkpoints_preserve_frame_zero_and_block_malicious_pickles(
    tmp_path: Path,
):
    # 1. Malicious pickle gadget MUST be blocked by weights_only=True safe loader
    malicious_pt = tmp_path / "malicious.pt"
    torch.save({"model_state": {}, "evil": _MaliciousPickleGadget()}, malicious_pt)

    feature_csv = tmp_path / "features.csv"
    rows = [
        {
            "video": "vid0",
            "frame": f"vid0_{frame:09d}.json",
            "frame_number": frame,
            "label": "idle" if frame < 3 else "rearing",
            "mouse_features": [float(frame), 0.0, 2.0, 0.0, 2.0, 2.0],
            "mouse_area": 4.0,
            "mouse_centroid": [float(frame), 1.0],
            "mouse_perimeter": 8.0,
        }
        for frame in range(6)
    ]
    pd.DataFrame.from_records(rows).to_csv(feature_csv, index=False)

    with pytest.raises(pickle.UnpicklingError):
        predict_polygon_classifier_csv(
            feature_csv=feature_csv,
            checkpoint_path=malicious_pt,
            output_csv=tmp_path / "evil_out.csv",
            device="cpu",
        )

    with pytest.raises(pickle.UnpicklingError):
        load_tcn_checkpoint(malicious_pt, device="cpu")

    # 2. Valid ConvNet checkpoint loads cleanly under weights_only=True AND preserves frame_number=0!
    with torch.no_grad():
        outcome = train_polygon_classifier(
            train_csv=feature_csv,
            test_csv=feature_csv,
            output_dir=tmp_path / "runs",
            num_epochs=1,
            batch_size=2,
            hidden_dim=8,
            num_residual_blocks=1,
            window_size=3,
            device="cpu",
        )
    pred_out = predict_polygon_classifier_csv(
        feature_csv=feature_csv,
        checkpoint_path=outcome.checkpoint_path,
        output_csv=tmp_path / "preds.csv",
        device="cpu",
    )
    preds_df = pd.read_csv(pred_out.output_csv)
    assert preds_df["frame_number"].tolist() == [0, 1, 2, 3, 4, 5]


def test_body_part_only_multi_animal_and_custom_checkpoint_residual_risks(
    tmp_path: Path,
):
    # 1. Body-part-only multi-animal CSV (no whole-body mouse_1 / mouse_2 rows)
    bp_df = pd.DataFrame(
        [
            {
                "frame_number": 0,
                "instance_name": "mouse_1:head",
                "cx": 10.0,
                "cy": 50.0,
            },
            {
                "frame_number": 0,
                "instance_name": "mouse_1:torso",
                "cx": 20.0,
                "cy": 50.0,
            },
            {
                "frame_number": 0,
                "instance_name": "mouse_2:head",
                "cx": 80.0,
                "cy": 50.0,
            },
            {
                "frame_number": 0,
                "instance_name": "mouse_2:torso",
                "cx": 90.0,
                "cy": 50.0,
            },
            {
                "frame_number": 1,
                "instance_name": "mouse_1:head",
                "cx": 60.0,
                "cy": 50.0,
            },
            {
                "frame_number": 1,
                "instance_name": "mouse_1:torso",
                "cx": 30.0,
                "cy": 50.0,
            },
            {
                "frame_number": 1,
                "instance_name": "mouse_2:head",
                "cx": 80.0,
                "cy": 50.0,
            },
            {
                "frame_number": 1,
                "instance_name": "mouse_2:torso",
                "cx": 90.0,
                "cy": 50.0,
            },
        ]
    )
    # Social neighbor summary aggregates mouse_1=(15, 50)/(45, 50) vs mouse_2=(85, 50)
    # and never pairs mouse_1:head with mouse_1:torso!
    nn_summary, nn_detail = compute_pairwise_centroid_summary(bp_df)
    assert set(nn_summary.index) == {"mouse_1", "mouse_2"}
    assert len(nn_detail) == 2
    assert nn_summary.loc[
        "mouse_1", "mean_nearest_neighbor_distance_px"
    ] == pytest.approx(55.0)
    assert nn_summary.loc["mouse_1", "closest_neighbor_label"] == "mouse_2"

    # Single-animal bare body-part CSV ('head', 'tail_base') must NOT pair head with tail_base
    bare_df = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "head", "cx": 10.0, "cy": 10.0},
            {"frame_number": 0, "instance_name": "tail_base", "cx": 30.0, "cy": 10.0},
        ]
    )
    bare_summary, bare_detail = compute_pairwise_centroid_summary(bare_df)
    assert bare_summary.empty
    assert bare_detail.empty

    # GenericZoneEngine + TrackingResultsAnalyzer resolve mouse_1 and mouse_2 on body-part-only CSVs
    zone_specs = load_zone_shapes(
        {
            "shapes": [
                build_zone_shape(
                    "right_zone",
                    [[50, 0], [100, 100]],
                    shape_type="rectangle",
                    zone_kind="chamber",
                    access_state="open",
                )
            ]
        }
    )
    # Default mean-part centroid: mouse_1 is at x=15 (f0) and x=45 (f1) -> 0 frames in right_zone;
    # with anchor_part='head', mouse_1:head is at x=10 (f0) and x=60 (f1) -> 1 frame in right_zone!
    default_res = GenericZoneEngine(zone_specs, fps=10).analyze_dataframe(bp_df)
    head_res = GenericZoneEngine(
        zone_specs, fps=10, anchor_part="head"
    ).analyze_dataframe(bp_df)
    assert set(default_res.keys()) == {"mouse_1", "mouse_2"}
    assert default_res["mouse_1"].occupancy_frames.get("right_zone", 0) == 0
    assert default_res["mouse_2"].occupancy_frames.get("right_zone", 0) == 2
    assert head_res["mouse_1"].occupancy_frames.get("right_zone", 0) == 1

    vid_path = tmp_path / "bp_only.mp4"
    vid_path.write_bytes(b"fake")
    bp_df.to_csv(tmp_path / "bp_only_tracking.csv", index=False)
    bp_df.to_csv(tmp_path / "bp_only_tracked.csv", index=False)
    tra = TrackingResultsAnalyzer(vid_path, fps=10)
    tra.merge_and_calculate_distance()
    assert len(tra.distances_df) == 2
    assert set(tra.distances_df["instance_name_1"]).union(
        set(tra.distances_df["instance_name_2"])
    ) == {"mouse_1", "mouse_2"}

    # 2. Custom checkpoint classes: blocked by default with actionable diagnostics,
    #    allowed cleanly via extra_globals=(...) or explicit trust_checkpoint=True,
    #    and PosixPath metadata works out-of-the-box under weights_only=True!
    path_ckpt = tmp_path / "with_path.pt"
    torch.save(
        {"weights": torch.ones(2), "source_path": Path("/tmp/session_a")}, path_ckpt
    )
    loaded_path_ckpt = safe_load_behavior_checkpoint(path_ckpt)
    assert Path(loaded_path_ckpt["source_path"]).name == "session_a"

    custom_ckpt = tmp_path / "custom_lab.pt"
    torch.save(
        {"weights": torch.zeros(2), "lab_cfg": _CustomLabConfig(experiment_tag="exp7")},
        custom_ckpt,
    )
    with pytest.raises(pickle.UnpicklingError, match="extra_globals") as exc_info:
        safe_load_behavior_checkpoint(custom_ckpt)
    assert "_CustomLabConfig" in str(exc_info.value)

    loaded_via_allowlist = safe_load_behavior_checkpoint(
        custom_ckpt, extra_globals=(_CustomLabConfig,)
    )
    assert loaded_via_allowlist["lab_cfg"].experiment_tag == "exp7"

    loaded_via_trust = safe_load_behavior_checkpoint(custom_ckpt, trust_checkpoint=True)
    assert loaded_via_trust["lab_cfg"].experiment_tag == "exp7"

    # 3. Even when trust_checkpoint=True (and without expected_sha256), OS/eval/reflection,
    #    non-denylisted module REDUCE functions (json.loads), and local classes with custom
    #    __setstate__ / __new__ / __setattr__ hooks remain hard-blocked with zero execution!
    _SetstateHookClass.SIDE_EFFECT_TRIGGERED = False
    setstate_obj = _SetstateHookClass()
    new_obj = object.__new__(_CustomNewClass)
    _CustomNewClass.SIDE_EFFECT_TRIGGERED = False
    setattr_obj = object.__new__(_CustomSetattrClass)
    object.__setattr__(setattr_obj, "x", 7)
    _CustomSetattrClass.SIDE_EFFECT_TRIGGERED = False

    for idx, gadget in enumerate(
        (
            _MaliciousPickleGadget(),
            _EvalPickleGadget(),
            _ReflectionPickleGadget(),
            _BenignModuleReduceGadget(),
            setstate_obj,
            new_obj,
            setattr_obj,
        )
    ):
        exploit_pt = tmp_path / f"exploit_{idx}.pt"
        torch.save({"weights": torch.zeros(1), "payload": gadget}, exploit_pt)
        with pytest.raises(pickle.UnpicklingError):
            safe_load_behavior_checkpoint(exploit_pt, trust_checkpoint=True)

    assert _SetstateHookClass.SIDE_EFFECT_TRIGGERED is False
    assert _CustomNewClass.SIDE_EFFECT_TRIGGERED is False
    assert _CustomSetattrClass.SIDE_EFFECT_TRIGGERED is False

    # Passing a custom-hook class in extra_globals is also rejected before torch.load
    setstate_pt = tmp_path / "setstate_extra.pt"
    torch.save({"weights": torch.zeros(1), "payload": setstate_obj}, setstate_pt)
    with pytest.raises(pickle.UnpicklingError):
        safe_load_behavior_checkpoint(setstate_pt, extra_globals=(_SetstateHookClass,))
    assert _SetstateHookClass.SIDE_EFFECT_TRIGGERED is False

    # 4. Unimported external-lab config classes load under weights_only=True into inert
    #    SafeCheckpointMetadata containers without importing any external module!
    ext_mod_name = "_uninstalled_collab_lab_cfg_mod"
    ext_mod = types.ModuleType(ext_mod_name)
    ExtCfg = type("ExternalExperimentConfig", (), {"__module__": ext_mod_name})
    ext_mod.ExternalExperimentConfig = ExtCfg  # type: ignore[attr-defined]
    sys.modules[ext_mod_name] = ext_mod
    try:
        ext_inst = ExtCfg()
        ext_inst.cohort = "beta"
        ext_inst.window = 15
        ext_ckpt = tmp_path / "external_lab.pt"
        torch.save({"weights": torch.ones(3), "cfg": ext_inst}, ext_ckpt)
    finally:
        sys.modules.pop(ext_mod_name, None)

    loaded_ext = safe_load_behavior_checkpoint(ext_ckpt, trust_checkpoint=True)
    assert ext_mod_name not in sys.modules
    assert isinstance(loaded_ext["cfg"], SafeCheckpointMetadata)
    assert loaded_ext["cfg"].cohort == "beta"
    assert loaded_ext["cfg"].to_dict() == {"cohort": "beta", "window": 15}

    # 5. Optional expected_sha256 verifies cryptographic integrity prior to unpickling
    valid_digest = hashlib.sha256(custom_ckpt.read_bytes()).hexdigest()
    verified_ckpt = safe_load_behavior_checkpoint(
        custom_ckpt,
        extra_globals=(_CustomLabConfig,),
        expected_sha256=f"sha256:{valid_digest}",
    )
    assert verified_ckpt["lab_cfg"].experiment_tag == "exp7"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        safe_load_behavior_checkpoint(
            custom_ckpt,
            extra_globals=(_CustomLabConfig,),
            expected_sha256="0" * 64,
        )
