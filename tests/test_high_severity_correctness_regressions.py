"""Focused deterministic regression tests for High-severity correctness items C-3 through C-8."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from annolid.behavior.cohort import (
    CohortRecordingSpec,
    _logrank_test,
    analyze_behavior_cohort,
)
from annolid.behavior.irr import _bootstrap_mean_cohens_kappa_ci95
from annolid.postprocessing.kinematics_stats import (
    _polygon_centroid_and_area,
    _savgol_smooth_contiguous,
    analyze_kinematics,
)
from annolid.postprocessing.quality_control import TracksResults, pred_dict_to_labelme
from annolid.postprocessing.spatial_calibration import (
    CalibrationError,
    PlanarCalibration,
    resolve_planar_calibration,
)
from annolid.services.chat_tracking_stats import _infer_tracking_source
from annolid.services.unified_reporting import (
    _resolve_recording_bundle,
    build_unified_analysis_report,
)
from annolid.utils.annotation_store import AnnotationStore


def test_c3_pred_dict_to_labelme_nan_segmentation_and_closed_polygon_integrity(
    tmp_path: Path,
) -> None:
    import pycocotools.mask as mask_util

    # Blank CSV cells become float('nan') under pandas.read_csv — must fall back to rectangle
    for empty_seg in (float("nan"), "", "nan", "None", None):
        shapes = pred_dict_to_labelme(
            {
                "frame_number": 0,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 30.0,
                "y2": 40.0,
                "instance_name": "Left_ear",
                "class_score": 0.9,
                "segmentation": empty_seg,
                "tracking_id": float("nan"),
            }
        )
        assert len(shapes) == 1
        assert shapes[0].shape_type == "rectangle"
        assert shapes[0].label == "Left_ear"

    # Valid mask polygon must call shape.close() without desynchronizing parallel point lists
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:16, 4:16] = 1
    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    poly_shapes = pred_dict_to_labelme(
        {
            "frame_number": 0,
            "x1": 4.0,
            "y1": 4.0,
            "x2": 16.0,
            "y2": 16.0,
            "instance_name": "mouse",
            "class_score": 0.95,
            "segmentation": str(rle),
            "tracking_id": 2.0,
        },
        keypoint_area_threshold=10,
    )
    assert len(poly_shapes) == 1
    poly = poly_shapes[0]
    assert poly.label == "mouse_2"
    assert poly.isClosed() is True
    assert len(poly.points) == len(poly.point_labels) == len(poly.shared_vertex_ids)
    assert poly.points[0] != poly.points[-1]

    # TracksResults.to_labelme_json must preserve Left/Right labels by default across arena midline
    video_path = tmp_path / "arena.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (100, 80)
    )
    for _ in range(2):
        writer.write(np.zeros((80, 100, 3), dtype=np.uint8))
    writer.release()

    csv_path = tmp_path / "arena_tracking.csv"
    pd.DataFrame(
        [
            {
                "frame_number": 0,
                "x1": 70.0,
                "y1": 20.0,
                "x2": 80.0,
                "y2": 30.0,
                "cx": 75.0,
                "cy": 25.0,
                "instance_name": "Left_ear",
                "class_score": 0.95,
                "segmentation": np.nan,
                "tracking_id": -1,
            },
            {
                "frame_number": 0,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 20.0,
                "y2": 30.0,
                "cx": 15.0,
                "cy": 25.0,
                "instance_name": "Right_ear",
                "class_score": 0.95,
                "segmentation": np.nan,
                "tracking_id": -1,
            },
        ]
    ).to_csv(csv_path, index=False)

    out_dir = tmp_path / "exported_frames"
    tr = TracksResults(video_file=str(video_path), tracking_csv=str(csv_path))
    events = list(tr.to_labelme_json(str(out_dir), skip_frames=1))
    assert any(msg.endswith(".json") for _, msg in events)
    saved_json = json.loads(
        (out_dir / "arena_000000000.json").read_text(encoding="utf-8")
    )
    exported_labels = {s["label"] for s in saved_json["shapes"]}
    assert exported_labels == {"Left_ear", "Right_ear"}
    assert list(tr.df["instance_name"]) == ["Left_ear", "Right_ear"]


def test_c4_kinematics_gap_isolation_sparse_freezing_and_circle_geometry(
    tmp_path: Path,
) -> None:
    # 1. Circle geometry returns true center and pi * r^2 area
    geom = _polygon_centroid_and_area([[50.0, 60.0], [60.0, 60.0]], "circle")
    assert geom is not None
    cx, cy, area = geom
    assert (cx, cy) == pytest.approx((50.0, 60.0))
    assert area == pytest.approx(math.pi * 100.0)

    # 2. Contiguous Savitzky-Golay does not bleed across frame gaps
    frames = np.array([0, 1, 2, 3, 4, 100, 101, 102, 103, 104], dtype=np.int64)
    vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 500.0, 500.0, 500.0, 500.0])
    smoothed = _savgol_smooth_contiguous(frames, vals, 5, 2)
    assert np.allclose(smoothed[:5], 0.0)
    assert np.allclose(smoothed[5:], 500.0)

    # 3. Trajectory with a multi-second occlusion gap must not create false freezing or accel spikes
    gap_csv = tmp_path / "gap_tracking.csv"
    rows = ["frame_number,instance_name,cx,cy"]
    # Constant fast motion (10 px/frame at 10 fps = 100 px/s) before and after a 50-frame gap
    for f in range(10):
        rows.append(f"{f},mouse_1,{float(f * 10)},20.0")
    for idx, f in enumerate(range(60, 70)):
        rows.append(f"{f},mouse_1,{200.0 + float(idx * 10)},20.0")
    gap_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    rep_gap = analyze_kinematics(gap_csv, fps=10.0, smooth_window=5)
    inst = rep_gap.instances[0]
    assert inst.mean_speed_px_s == pytest.approx(100.0, rel=1e-3)
    assert inst.freezing_bout_count == 0
    assert inst.freezing_percent == 0.0
    assert inst.max_accel_px_s2 == pytest.approx(0.0, abs=1e-6)

    # 4. Every-2nd-frame stationary annotations must never report freezing_seconds > duration_s
    sparse_csv = tmp_path / "sparse_tracking.csv"
    s_rows = ["frame_number,instance_name,cx,cy"]
    for f in range(0, 22, 2):
        s_rows.append(f"{f},mouse_a,30.0,30.0")
        s_rows.append(f"{f},mouse_b,40.0,30.0")
    sparse_csv.write_text("\n".join(s_rows) + "\n", encoding="utf-8")

    rep_sparse = analyze_kinematics(
        sparse_csv, fps=10.0, smooth_window=1, min_freeze_duration_s=0.5
    )
    inst_s = rep_sparse.instances[0]
    assert inst_s.freezing_seconds <= inst_s.duration_s
    assert inst_s.freezing_percent == pytest.approx(100.0)
    assert rep_sparse.pairwise[0].close_contact_percent == pytest.approx(100.0)

    uni = build_unified_analysis_report(
        sparse_csv,
        fps=10.0,
        output_dir=tmp_path / "sparse_rep",
        include_plots=False,
    )
    assert "100.0%" in uni.markdown_path.read_text(encoding="utf-8")
    assert "100.0%" in uni.html_path.read_text(encoding="utf-8")


def test_c5_planar_calibration_reflected_homography_and_invalid_px_per_cm() -> None:
    # Flipped Y-axis homography diag(0.1, -0.1, 1.0) must preserve negative Y in project_points
    cal_flip = PlanarCalibration.from_homography(
        [[0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 1.0]]
    )
    assert cal_flip.mode == "homography"
    xw, yw = cal_flip.project_points([20.0], [30.0])
    assert float(xw[0]) == pytest.approx(2.0)
    assert float(yw[0]) == pytest.approx(-3.0)

    # Positive isotropic homography still uses fast uniform mode
    cal_pos = PlanarCalibration.from_homography(
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]]
    )
    assert cal_pos.mode == "uniform"

    # Non-positive or non-finite px_per_cm must raise CalibrationError instead of returning None
    assert resolve_planar_calibration(px_per_cm=None) is None
    for bad_val in (0.0, -12.5, float("nan"), float("inf")):
        with pytest.raises(CalibrationError):
            resolve_planar_calibration(px_per_cm=bad_val)


def test_c6_cohort_logrank_near_ties_zero_variance_dyad_rounding_and_irr_block_bootstrap(
    tmp_path: Path,
) -> None:
    # 1. Near-tied event timestamps within 1e-12 must not double-count Mantel-Cox chi2
    exact_ties = {
        "ctrl": [(10.0, True), (10.0, True), (30.0, False)],
        "drug": [(25.0, True), (30.0, False), (30.0, False)],
    }
    near_ties = {
        "ctrl": [(10.0, True), (10.0 + 1e-13, True), (30.0, False)],
        "drug": [(25.0, True), (30.0, False), (30.0, False)],
    }
    chi2_exact, p_exact = _logrank_test(exact_ties, ["ctrl", "drug"])
    chi2_near, p_near = _logrank_test(near_ties, ["ctrl", "drug"])
    assert chi2_near == pytest.approx(chi2_exact, rel=1e-12)
    assert p_near == pytest.approx(p_exact, rel=1e-12)

    # 2. Zero within-group variance + dyad [0, 1] occurrence rounding
    specs = []
    for i in range(3):
        p_ctrl = tmp_path / f"zv_ctrl_{i}.csv"
        p_ctrl.write_text(
            "Recording time,Subject,Behavior,Event\n"
            "1.0,resident,other,point\n"
            "1.0,intruder,other,point\n",
            encoding="utf-8",
        )
        specs.append(
            CohortRecordingSpec(
                recording_id=f"c{i}",
                events_csv=p_ctrl,
                group="ctrl",
                session_duration_s=20.0,
            )
        )
        p_drug = tmp_path / f"zv_drug_{i}.csv"
        # Resident attacks once for exactly 10.0s; intruder has 0 occurrences -> dyad mean 0.5 -> 1
        p_drug.write_text(
            "Recording time,Subject,Behavior,Event\n"
            "2.0,resident,attack,start\n"
            "12.0,resident,attack,stop\n"
            "1.0,intruder,other,point\n",
            encoding="utf-8",
        )
        specs.append(
            CohortRecordingSpec(
                recording_id=f"d{i}",
                events_csv=p_drug,
                group="drug",
                session_duration_s=20.0,
            )
        )

    rep = analyze_behavior_cohort(specs, analysis_unit="recording")
    attack_cmp = next(c for c in rep.comparisons if c.behavior == "attack")
    assert attack_cmp.p_parametric == 0.0
    assert attack_cmp.q_parametric_bh == 0.0
    assert (
        math.isfinite(attack_cmp.parametric_stat) and attack_cmp.parametric_stat != 0.0
    )
    serialized = json.dumps(attack_cmp.to_dict(), allow_nan=False)
    assert "attack" in serialized

    drug_attack_summary = next(
        s for s in rep.group_summaries if s.group == "drug" and s.behavior == "attack"
    )
    assert drug_attack_summary.n_responders == 3
    assert drug_attack_summary.mean_occurrences == pytest.approx(1.0)

    # 3. IRR circular block bootstrap respects bout autocorrelation (wider CI than i.i.d. bins)
    # Construct 600 bins (60s at 0.1s) with 5s bouts where raters agree on half the bouts
    r1_grid = ([1] * 50 + [0] * 50) * 6
    r2_grid = ([1] * 50 + [0] * 50) * 3 + ([0] * 50 + [0] * 50) * 3
    grids = {"r1": r1_grid, "r2": r2_grid}
    lo_iid, hi_iid = _bootstrap_mean_cohens_kappa_ci95(
        grids, [("r1", "r2")], bin_seconds=0.1, block_seconds=0.05, n_boot=300, seed=7
    )
    lo_blk, hi_blk = _bootstrap_mean_cohens_kappa_ci95(
        grids, [("r1", "r2")], bin_seconds=0.1, block_seconds=5.0, n_boot=300, seed=7
    )
    assert (hi_blk - lo_blk) > (hi_iid - lo_iid)


def test_c7_c8_features_lazy_exports_and_ndjson_store_discovery(
    tmp_path: Path,
) -> None:
    import annolid.features as features_pkg
    import annolid.postprocessing.visualizer as visualizer_mod

    assert "Embedding" in features_pkg.__all__
    assert "CLIPModel" in features_pkg.__all__
    emb = visualizer_mod.Embedding(pretrained=False)
    dummy_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    vec = emb([dummy_rgb])
    assert vec.shape == (1, 512)

    # NDJSON-only annotation folder (persist_json=False) must be recognized as annotation_dir
    ndjson_dir = tmp_path / "trial_01"
    ndjson_dir.mkdir()
    store = AnnotationStore(ndjson_dir / "trial_01_annotations.ndjson")
    store.append_frame(
        {
            "frame": 0,
            "version": "5.0.0",
            "flags": {},
            "shapes": [
                {
                    "label": "mouse_1",
                    "shape_type": "rectangle",
                    "points": [[10.0, 10.0], [30.0, 30.0]],
                    "flags": {},
                }
            ],
            "imagePath": "trial_01_000000000.png",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 100,
        }
    )
    assert not list(ndjson_dir.glob("*.json"))
    bundle = _resolve_recording_bundle(ndjson_dir)
    assert bundle["annotation_dir"] == ndjson_dir
    assert _infer_tracking_source(ndjson_dir) == ndjson_dir

    # Workspace root with project.annolid.json + single *_tracking.csv resolves to the CSV
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    (ws_dir / "project.annolid.json").write_text("{}", encoding="utf-8")
    ws_csv = ws_dir / "rec_tracking.csv"
    ws_csv.write_text(
        "frame_number,instance_name,cx,cy\n0,m1,10,10\n", encoding="utf-8"
    )
    ws_bundle = _resolve_recording_bundle(ws_dir)
    assert ws_bundle["tracking_csv"] == ws_csv
    assert _infer_tracking_source(ws_dir) == ws_csv
