"""Comprehensive deterministic tests for the SLEAP/DeepLabCut-parity pose estimation workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

from annolid.annotation.keypoint_visibility import KeypointVisibility
from annolid.annotation.pose_interop.converters import (
    export_pose_dataset,
    read_pose_dataset,
)
from annolid.annotation.pose_interop.labelme import write_pose_to_labelme_dir
from annolid.annotation.pose_interop.model import (
    PoseDataset,
    PoseFrame,
    PoseInstance,
    PosePoint,
)
from annolid.annotation.pose_schema import (
    PoseSchema,
    get_pose_schema_preset,
    list_pose_schema_presets,
)
from annolid.data.frame_sampling import (
    extract_frames_smart,
    extract_pose_outlier_frames,
    score_pose_outlier_frames,
    select_kmeans_frame_indices,
    select_uniform_frame_indices,
)
from annolid.data.videos import extract_frames as legacy_extract_frames
from annolid.engine import cli as engine_cli
from annolid.postprocessing.identity_governor import (
    EvidenceRule,
    GovernorPolicy,
    IdentityGovernor,
    MetricCondition,
)
from annolid.postprocessing.pose_evaluation import (
    evaluate_pose_predictions,
    format_pose_eval_table,
    write_pose_eval_report_bundle,
)
from annolid.postprocessing.pose_filtering import (
    PoseFilterConfig,
    filter_pose_dataset,
    filter_pose_source,
    swap_pose_instances_from_frame,
)
from annolid.postprocessing.pose_kinematics import (
    analyze_pose_kinematics,
    format_pose_kinematics_table,
    write_pose_kinematics_bundle,
)
from annolid.postprocessing.pose_video_renderer import (
    PoseVideoRenderConfig,
    render_pose_video,
)
from annolid.postprocessing.temporal_identity_repair import (
    run_temporal_identity_repair,
)
from annolid.postprocessing.track_qc import evaluate_track_qc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_synthetic_video(
    path: Path,
    *,
    num_frames: int = 30,
    width: int = 160,
    height: int = 120,
    fps: float = 30.0,
) -> Path:
    """Create a deterministic multi-cluster video with distinct colored visual states."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Failed to open synthetic video writer at {path}"
    try:
        for idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cluster = idx // max(1, num_frames // 3)
            if cluster == 0:
                frame[:, :] = (30, 30, 140)
                cv2.circle(frame, (35 + idx, 35), 14, (240, 240, 240), -1)
            elif cluster == 1:
                frame[:, :] = (30, 140, 30)
                cv2.rectangle(frame, (90, 20), (140, 70), (20, 240, 240), -1)
            else:
                frame[:, :] = (140, 40, 30)
                cv2.circle(frame, (80, 90), 18, (255, 180, 20), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _make_mouse_schema() -> PoseSchema:
    return get_pose_schema_preset("Mouse (5 keypoints)")


def _make_two_mouse_dataset(
    *,
    num_frames: int = 12,
    inject_noise: bool = False,
    swap_order_in_pred: bool = False,
) -> PoseDataset:
    schema = _make_mouse_schema()
    frames: list[PoseFrame] = []
    for fidx in range(num_frames):
        # Mouse A moves smoothly rightward along y=40
        ax = 40.0 + fidx * 4.0
        ay = 40.0
        # Mouse B moves smoothly rightward along y=90
        bx = 40.0 + fidx * 3.0
        by = 90.0

        def _pts(
            cx: float,
            cy: float,
            *,
            low_conf_nose: bool = False,
            spike_tail: bool = False,
            lr_swap: bool = False,
        ):
            nose_x = cx + 15.0
            nose_y = cy
            le_x, le_y = cx + 8.0, cy - 8.0
            re_x, re_y = cx + 8.0, cy + 8.0
            if lr_swap:
                le_x, le_y, re_x, re_y = re_x, re_y, le_x, le_y
            tb_x = cx - 20.0 + (80.0 if spike_tail else 0.0)
            tb_y = cy
            return {
                "nose": PosePoint(
                    nose_x + (12.0 if low_conf_nose else 0.0),
                    nose_y,
                    score=0.20 if low_conf_nose else 0.95,
                    visibility=int(KeypointVisibility.VISIBLE),
                ),
                "left_ear": PosePoint(
                    le_x, le_y, score=0.92, visibility=int(KeypointVisibility.VISIBLE)
                ),
                "right_ear": PosePoint(
                    re_x, re_y, score=0.92, visibility=int(KeypointVisibility.VISIBLE)
                ),
                "centroid": PosePoint(
                    cx, cy, score=0.96, visibility=int(KeypointVisibility.VISIBLE)
                ),
                "tail_base": PosePoint(
                    tb_x, tb_y, score=0.90, visibility=int(KeypointVisibility.VISIBLE)
                ),
            }

        inst_a = PoseInstance(
            instance_name="mouse_a",
            keypoints=_pts(
                ax,
                ay,
                low_conf_nose=inject_noise and fidx == 3,
                spike_tail=inject_noise and fidx == 6,
                lr_swap=inject_noise and fidx == 8,
            ),
            track_id=0,
            score=0.94,
            is_predicted=inject_noise,
        )
        inst_b = PoseInstance(
            instance_name="mouse_b",
            keypoints=_pts(bx, by),
            track_id=1,
            score=0.93,
            is_predicted=inject_noise,
        )
        inst_list = [inst_b, inst_a] if swap_order_in_pred else [inst_a, inst_b]
        frames.append(
            PoseFrame(
                frame_index=fidx,
                instances=inst_list,
                image_path=f"vid_{fidx:09d}.png",
                image_width=160,
                image_height=120,
            )
        )
    return PoseDataset(
        frames=frames,
        keypoints=list(schema.keypoints),
        edges=list(schema.edges),
        symmetry_pairs=list(schema.symmetry_pairs),
        instances=["mouse_a", "mouse_b"],
        multi_animal=True,
        source_format="labelme_dir",
    )


# ---------------------------------------------------------------------------
# 1. Schema & Interop Fixes
# ---------------------------------------------------------------------------
def test_pose_schema_new_presets_bilateral_guard_and_reordered_flip_idx(
    tmp_path: Path,
) -> None:
    presets = list_pose_schema_presets()
    for expected in (
        "Mouse CalMS21 / MARS (7 keypoints)",
        "Mouse MaBE2022 (12 keypoints)",
        "Rat (8 keypoints)",
        "Fly Full-Body (12 keypoints)",
        "Zebrafish Spine (5 keypoints)",
    ):
        assert expected in presets
        sc = get_pose_schema_preset(expected)
        assert sc.validate() == []
        assert len(sc.keypoints) >= 5
        assert len(sc.edges) >= 4

    # Bilateral-only schema must NOT split left/right into instances=["left", "right"]!
    bilateral = PoseSchema.from_dict(
        {
            "keypoints": ["left_ear", "right_ear", "left_hip", "right_hip"],
            "instances": [],
            "instance_separator": "_",
        }
    )
    bilateral.normalize_prefixed_keypoints()
    assert bilateral.instances == []
    assert bilateral.keypoints == ["left_ear", "right_ear", "left_hip", "right_hip"]

    # Explicit flip_idx must remap accurately when keypoint_order is permuted.
    sc_flip = PoseSchema.from_dict(
        {
            "keypoints": ["nose", "left_ear", "right_ear"],
            "flip_idx": [0, 2, 1],
        }
    )
    assert sc_flip.compute_flip_idx(["left_ear", "nose", "right_ear"]) == [2, 1, 0]

    # Interop: export DLC project -> read_pose_dataset(deeplabcut_training) -> export COCO JSON -> read back!
    ds = _make_two_mouse_dataset(num_frames=3)
    lm_dir = tmp_path / "lm"
    write_pose_to_labelme_dir(ds, lm_dir, write_pose_schema=True)

    dlc_proj = tmp_path / "dlc_proj"
    export_pose_dataset(
        lm_dir,
        dlc_proj,
        target_format="deeplabcut-project",
        scorer="lab",
        video_name="trial",
    )
    loaded_train = read_pose_dataset(dlc_proj)
    assert loaded_train.source_format == "deeplabcut_training"
    assert len(loaded_train.frames) == 3
    assert set(loaded_train.resolved_keypoints()) == set(ds.resolved_keypoints())

    coco_out = tmp_path / "exported_coco.json"
    coco_summary = export_pose_dataset(dlc_proj, coco_out, target_format="coco-json")
    assert coco_summary["target_format"] == "coco_json"
    assert coco_summary["annotations_exported"] == 6
    loaded_coco = read_pose_dataset(coco_out)
    assert len(loaded_coco.frames) == 3


# ---------------------------------------------------------------------------
# 2. Smart Frame Extraction (Uniform, K-Means) & Active-Learning Outliers
# ---------------------------------------------------------------------------
def test_smart_frame_extraction_kmeans_uniform_and_legacy_videos_bridge(
    tmp_path: Path,
) -> None:
    vid = _write_synthetic_video(tmp_path / "demo.mp4", num_frames=30)

    unif = select_uniform_frame_indices(30, 5)
    assert unif == [0, 7, 14, 22, 29]

    km_a = select_kmeans_frame_indices(vid, 3, seed=7)
    km_b = select_kmeans_frame_indices(vid, 3, seed=7)
    assert km_a == km_b
    assert len(km_a) == 3
    # Should pick one frame from each of the 3 distinct visual clusters [0..9], [10..19], [20..29].
    clusters_hit = {idx // 10 for idx in km_a}
    assert clusters_hit == {0, 1, 2}

    res_km = extract_frames_smart(
        vid, tmp_path / "out_km", algo="kmeans", num_frames=4, seed=42
    )
    assert len(res_km.saved_paths) == 4
    assert all(Path(p).is_file() for p in res_km.saved_paths)

    res_u = extract_frames_smart(
        vid, tmp_path / "out_unif", algo="uniform", num_frames=4
    )
    assert len(res_u.saved_paths) == 4

    # Legacy videos.extract_frames generator accepts algo="kmeans" & algo="uniform".
    events = list(
        legacy_extract_frames(
            video_file=str(vid),
            num_frames=3,
            out_dir=str(tmp_path / "legacy"),
            algo="kmeans",
        )
    )
    assert events[-1][0] == 100
    assert len(list((tmp_path / "legacy" / "demo_kmeans").glob("*.png"))) == 3


def test_active_learning_outlier_scoring_and_extraction(tmp_path: Path) -> None:
    vid = _write_synthetic_video(tmp_path / "trial.mp4", num_frames=12)
    noisy = _make_two_mouse_dataset(num_frames=12, inject_noise=True)
    schema = _make_mouse_schema()

    scored = score_pose_outlier_frames(noisy, pose_schema=schema, min_confidence=0.6)
    top_frames = [item.frame_index for item in scored[:3]]
    # Frame 6 has a huge tail_base jump + bone stretch; frame 3 has low confidence (0.20).
    assert 6 in top_frames
    assert 3 in top_frames

    out_dir = tmp_path / "outliers"
    res = extract_pose_outlier_frames(
        noisy,
        vid,
        out_dir,
        pose_schema=schema,
        num_frames=2,
        min_frame_gap=2,
        min_confidence=0.6,
    )
    assert res.frames_extracted == 2
    assert Path(res.manifest_path).is_file()
    assert res.pose_schema_path is not None and Path(res.pose_schema_path).is_file()
    extracted_indices = {s.frame_index for s in res.selected_frames}
    assert {3, 6} <= extracted_indices or 6 in extracted_indices
    # Editable LabelMe JSONs should exist alongside PNGs.
    json_files = [
        p
        for p in out_dir.glob("*.json")
        if p.name not in ("pose_schema.json", "outlier_frames_manifest.json")
    ]
    assert len(json_files) == 2


# ---------------------------------------------------------------------------
# 3. Unified Pose Evaluation Engine
# ---------------------------------------------------------------------------
def test_pose_evaluation_dlc_sleap_oks_swaps_and_report_bundle(tmp_path: Path) -> None:
    gt = _make_two_mouse_dataset(num_frames=10, inject_noise=False)
    # Predictions have swapped instance list order (tests Hungarian matching!),
    # plus low-conf nose error on frame 3, tail spike on frame 6, and L/R ear swap on frame 8.
    pred = _make_two_mouse_dataset(
        num_frames=10, inject_noise=True, swap_order_in_pred=True
    )
    schema = _make_mouse_schema()

    report = evaluate_pose_predictions(
        gt,
        pred,
        pose_schema=schema,
        pcutoff=0.6,
        pck_thresholds_px=(2.0, 8.0),
        pck_norm_thresholds=(0.1, 0.2),
    )
    assert report.frames_evaluated == 10
    assert report.matched_instances_total == 20
    assert report.rmse_px is not None and report.rmse_pcutoff_px is not None
    # Excluding low-confidence nose error (12px at score=0.20) must change pcutoff count.
    assert report.pcutoff_keypoints_total == report.matched_keypoints_total - 1
    assert report.p50_px is not None and report.p50_px == pytest.approx(0.0, abs=1e-6)
    assert report.p95_px is not None
    assert 0.0 < report.oks_map <= 1.0
    assert report.oks_ap50 > 0.8
    # Multi-animal L/R swap on mouse_a frame 8 must be detected!
    assert report.swap_pairs_swapped == 1
    assert report.swap_rate is not None and report.swap_rate > 0.0
    # Frame 6 (80px spike) and frame 8 (L/R swap) should rank among worst frames.
    worst_ids = [wf.frame_index for wf in report.worst_frames[:3]]
    assert 6 in worst_ids
    assert 8 in worst_ids

    text = format_pose_eval_table(report)
    assert "Pose Evaluation Summary" in text
    assert "L/R Swap Rate" in text

    bundle = write_pose_eval_report_bundle(report, tmp_path / "eval_bundle")
    assert Path(bundle["json"]).is_file()
    assert Path(bundle["per_keypoint_csv"]).is_file()
    assert Path(bundle["worst_frames_csv"]).is_file()
    html_body = Path(bundle["html"]).read_text(encoding="utf-8")
    assert "<svg" in html_body
    assert "Annolid Pose Evaluation Report" in html_body


# ---------------------------------------------------------------------------
# 4. Offline Pose Filtering, Interpolation, Forward Swap, & Identity Sync
# ---------------------------------------------------------------------------
def test_pose_filtering_masks_interpolates_smooths_and_preserves_polygons(
    tmp_path: Path,
) -> None:
    schema = _make_mouse_schema()
    lm_dir = tmp_path / "pose_with_polys"
    lm_dir.mkdir()
    schema.save(lm_dir / "pose_schema.json")

    # 11 frames: steady linear motion x = 20 + 5*t, with:
    # - Frame 3: low-confidence nose spike (score=0.1, x=250) -> masked & interpolated!
    # - Frame 7: high-conf single-frame jump spike on tail_base (x=300) -> masked & interpolated!
    # - Every frame also has a non-point polygon that MUST survive filtering untouched!
    for t in range(11):
        cx = 20.0 + 5.0 * t
        nose_x = 250.0 if t == 3 else (cx + 15.0)
        nose_score = 0.10 if t == 3 else 0.95
        tb_x = 300.0 if t == 7 else (cx - 15.0)
        payload = {
            "version": "5.1.1",
            "flags": {},
            "imagePath": f"frame_{t:09d}.png",
            "imageData": None,
            "imageHeight": 120,
            "imageWidth": 320,
            "shapes": [
                {
                    "label": "mouse",
                    "points": [
                        [cx - 20, 30],
                        [cx + 20, 30],
                        [cx + 20, 60],
                        [cx - 20, 60],
                    ],
                    "group_id": 0,
                    "shape_type": "polygon",
                    "flags": {},
                },
                {
                    "label": "nose",
                    "points": [[nose_x, 45.0]],
                    "group_id": 0,
                    "shape_type": "point",
                    "flags": {"kp_visible": True},
                    "instance_label": "mouse",
                    "display_label": "nose",
                    "score": nose_score,
                },
                {
                    "label": "tail_base",
                    "points": [[tb_x, 45.0]],
                    "group_id": 0,
                    "shape_type": "point",
                    "flags": {"kp_visible": True},
                    "instance_label": "mouse",
                    "display_label": "tail_base",
                    "score": 0.95,
                },
            ],
        }
        (lm_dir / f"frame_{t:09d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    out_dir = tmp_path / "pose_filtered"
    cfg = PoseFilterConfig(
        min_confidence=0.5,
        max_jump_px=40.0,
        interpolate="linear",
        max_gap_frames=3,
        method="median",
        window_length=3,
    )
    report = filter_pose_source(
        lm_dir, output_path=out_dir, config=cfg, pose_schema=schema
    )
    assert report.masked_low_confidence == 1
    assert report.masked_jump_outliers >= 1
    assert report.gaps_interpolated >= 2

    # Check frame 3: polygon preserved AND nose repaired back near 20 + 15 + 15 = 50!
    f3 = json.loads((out_dir / "frame_000000003.json").read_text(encoding="utf-8"))
    polys = [s for s in f3["shapes"] if s["shape_type"] == "polygon"]
    assert len(polys) == 1 and polys[0]["label"] == "mouse"
    nose_f3 = [
        s for s in f3["shapes"] if s["shape_type"] == "point" and s["label"] == "nose"
    ][0]
    assert abs(nose_f3["points"][0][0] - 50.0) < 3.0

    # Check frame 7: tail_base repaired back near 20 + 35 - 15 = 40!
    f7 = json.loads((out_dir / "frame_000000007.json").read_text(encoding="utf-8"))
    tb_f7 = [
        s
        for s in f7["shapes"]
        if s["shape_type"] == "point" and s["label"] == "tail_base"
    ][0]
    assert abs(tb_f7["points"][0][0] - 40.0) < 3.0

    # Also test cubic interpolation + savgol + one_euro on PoseDataset directly.
    ds = _make_two_mouse_dataset(num_frames=10)
    ds_filt, rep_sg = filter_pose_dataset(
        ds, PoseFilterConfig(interpolate="cubic", method="savgol", window_length=5)
    )
    assert rep_sg.frames_processed == 10
    assert len(ds_filt.frames) == 10
    _, rep_oe = filter_pose_dataset(ds, PoseFilterConfig(method="one_euro", fps=30.0))
    assert rep_oe.points_smoothed > 0


def test_swap_pose_instances_from_frame_and_child_keypoint_identity_sync(
    tmp_path: Path,
) -> None:
    # Build 6 frames with two animals (polygon + keypoints per animal).
    folder = tmp_path / "swap_seq"
    folder.mkdir()
    for fidx in range(6):
        swapped = fidx >= 3
        a_x = 120.0 if swapped else 30.0
        b_x = 30.0 if swapped else 120.0
        payload = {
            "version": "5.1.1",
            "flags": {},
            "imagePath": f"s_{fidx:09d}.png",
            "imageData": None,
            "imageHeight": 120,
            "imageWidth": 200,
            "shapes": [
                {
                    "label": "mouse_1",
                    "points": [
                        [a_x - 10, 30],
                        [a_x + 10, 30],
                        [a_x + 10, 50],
                        [a_x - 10, 50],
                    ],
                    "group_id": 1,
                    "track_id": 1,
                    "instance_label": "mouse_1",
                    "shape_type": "polygon",
                    "flags": {},
                },
                {
                    "label": "mouse_1_nose",
                    "points": [[a_x + 5, 40]],
                    "group_id": 1,
                    "track_id": 1,
                    "instance_label": "mouse_1",
                    "display_label": "nose",
                    "shape_type": "point",
                    "description": "dinokpseg",
                    "flags": {"kp_visible": True},
                },
                {
                    "label": "mouse_2",
                    "points": [
                        [b_x - 10, 30],
                        [b_x + 10, 30],
                        [b_x + 10, 50],
                        [b_x - 10, 50],
                    ],
                    "group_id": 2,
                    "track_id": 2,
                    "instance_label": "mouse_2",
                    "shape_type": "polygon",
                    "flags": {},
                },
                {
                    "label": "mouse_2_nose",
                    "points": [[b_x + 5, 40]],
                    "group_id": 2,
                    "track_id": 2,
                    "instance_label": "mouse_2",
                    "display_label": "nose",
                    "shape_type": "point",
                    "description": "dinokpseg",
                    "flags": {"kp_visible": True},
                },
            ],
        }
        (folder / f"s_{fidx:09d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    # 1. Dry-run forward swap does not mutate disk.
    dry = swap_pose_instances_from_frame(
        folder, start_frame=3, instance_a="mouse_1", instance_b="mouse_2", dry_run=True
    )
    assert dry.frames_modified == 3
    assert dry.points_swapped == 6
    assert dry.polygons_swapped == 6

    # 2. Test temporal_identity_repair synchronizes child pose keypoints when repairing parent polygons!
    rep = run_temporal_identity_repair(folder, apply_changes=True)
    assert len(rep.proposed_corrections) >= 2
    f4_after = json.loads((folder / "s_000000004.json").read_text(encoding="utf-8"))
    pts_by_lbl = {
        s["label"]: s["points"][0][0]
        for s in f4_after["shapes"]
        if s["shape_type"] == "point"
    }
    # After repair, mouse_1_nose should be back on the left (x=35) and mouse_2_nose on the right (x=125)!
    assert pts_by_lbl["mouse_1_nose"] == pytest.approx(35.0)
    assert pts_by_lbl["mouse_2_nose"] == pytest.approx(125.0)

    # 3. Apply explicit swap_pose_instances_from_frame from frame 4 onward.
    applied = swap_pose_instances_from_frame(
        folder, start_frame=4, instance_a="mouse_1", instance_b="mouse_2", dry_run=False
    )
    assert applied.frames_modified == 2
    f3_untouched = json.loads((folder / "s_000000003.json").read_text(encoding="utf-8"))
    f4_swapped = json.loads((folder / "s_000000004.json").read_text(encoding="utf-8"))
    f3_nose = {
        s["label"]: s["points"][0][0]
        for s in f3_untouched["shapes"]
        if s["shape_type"] == "point"
    }
    f4_nose = {
        s["label"]: s["points"][0][0]
        for s in f4_swapped["shapes"]
        if s["shape_type"] == "point"
    }
    assert f3_nose["mouse_1_nose"] == pytest.approx(35.0)
    assert f4_nose["mouse_1_nose"] == pytest.approx(125.0)
    assert f4_nose["mouse_2_nose"] == pytest.approx(35.0)

    # 4. Verify IdentityGovernor._apply_child_body_part_remaps also remaps point keypoints.
    dummy_rule = EvidenceRule(
        name="left_mouse",
        assign_label="mouse_1",
        conditions=(MetricCondition(metric="centroid_x", op="lt", value=100.0),),
    )
    gov = IdentityGovernor(folder, GovernorPolicy(rules=(dummy_rule,)))
    shapes_test = [
        {
            "label": "mouse_1_nose",
            "shape_type": "point",
            "instance_label": "mouse_1",
            "flags": {},
        },
        {"label": "mouse_2:tail_base", "shape_type": "point", "flags": {}},
    ]
    n_remapped = gov._apply_child_body_part_remaps(
        shapes_test, {"mouse_1": "mouse_2", "mouse_2": "mouse_1"}
    )
    assert n_remapped == 2
    assert shapes_test[0]["label"] == "mouse_2_nose"
    assert shapes_test[0]["instance_label"] == "mouse_2"
    assert shapes_test[1]["label"] == "mouse_1:tail_base"


def test_track_qc_detects_collapsed_bones_and_keypoint_velocity_spikes(
    tmp_path: Path,
) -> None:
    schema = PoseSchema.from_dict(
        {
            "keypoints": ["nose", "tail_base"],
            "edges": [["nose", "tail_base"]],
        }
    )
    lm_dir = tmp_path / "qc_pose"
    lm_dir.mkdir()
    schema.save(lm_dir / "pose_schema.json")

    for t in range(8):
        # Normal bone length is 40px. On frame 4, nose collapses onto tail_base (length=2px) AND jumps 38px!
        nose_x = 42.0 if t == 4 else 80.0
        payload = {
            "version": "5.1.1",
            "flags": {},
            "imagePath": f"q_{t:09d}.png",
            "imageData": None,
            "imageHeight": 120,
            "imageWidth": 160,
            "shapes": [
                {
                    "label": "nose",
                    "points": [[nose_x, 50.0]],
                    "shape_type": "point",
                    "flags": {"kp_visible": True},
                },
                {
                    "label": "tail_base",
                    "points": [[40.0, 50.0]],
                    "shape_type": "point",
                    "flags": {"kp_visible": True},
                },
            ],
        }
        (lm_dir / f"q_{t:09d}.json").write_text(json.dumps(payload), encoding="utf-8")

    qc = evaluate_track_qc(lm_dir, pose_schema=schema, min_jump_px=20.0)
    codes = {issue.code for issue in qc.issues}
    assert "anatomical_edge_outlier" in codes
    assert any("collapsed" in issue.message for issue in qc.issues)
    assert "keypoint_velocity_spike" in codes


# ---------------------------------------------------------------------------
# 5. Pose Kinematics (Speed, Bone Lengths, Joint Angles, Egocentric, Dyadic)
# ---------------------------------------------------------------------------
def test_pose_kinematics_calibrated_angles_egocentric_and_dyadic(
    tmp_path: Path,
) -> None:
    # 2 animals over 5 frames:
    # - animal A faces +y (rotated 90 deg from +x): tail_base=(50, 50+10t), neck=(50, 70+10t), nose=(50, 90+10t),
    #   plus right_ear=(30, 70+10t) forming an exact 90-degree angle nose-neck-right_ear!
    # - animal B stands nearby so dyadic nose-to-nose distance is exact.
    schema = PoseSchema.from_dict(
        {
            "keypoints": ["nose", "right_ear", "neck", "tail_base"],
            "edges": [
                ["nose", "neck"],
                ["right_ear", "neck"],
                ["neck", "tail_base"],
            ],
        }
    )
    frames: list[PoseFrame] = []
    for t in range(5):
        y0 = 20.0 + 10.0 * t  # 10 px/frame at 10 fps = 100 px/s = 10 cm/s (at 10 px/cm)
        inst_a = PoseInstance(
            instance_name="resident",
            keypoints={
                "tail_base": PosePoint(50.0, y0, score=0.95),
                "neck": PosePoint(50.0, y0 + 20.0, score=0.95),
                "nose": PosePoint(50.0, y0 + 40.0, score=0.95),
                "right_ear": PosePoint(30.0, y0 + 20.0, score=0.95),
            },
        )
        inst_b = PoseInstance(
            instance_name="intruder",
            keypoints={
                "tail_base": PosePoint(70.0, y0, score=0.95),
                "neck": PosePoint(70.0, y0 + 20.0, score=0.95),
                "nose": PosePoint(70.0, y0 + 40.0, score=0.95),
                "right_ear": PosePoint(90.0, y0 + 20.0, score=0.95),
            },
        )
        frames.append(PoseFrame(frame_index=t, instances=[inst_a, inst_b]))

    ds = PoseDataset(
        frames=frames,
        keypoints=list(schema.keypoints),
        edges=list(schema.edges),
        instances=["resident", "intruder"],
        multi_animal=True,
    )
    report = analyze_pose_kinematics(ds, pose_schema=schema, fps=10.0, px_per_cm=10.0)
    assert report.frames_analyzed == 5
    assert report.has_calibration is True

    # Speed check: 100 px/s = 10 cm/s for resident nose.
    res_nose = [
        r
        for r in report.keypoint_rows
        if r.instance == "resident" and r.keypoint == "nose"
    ][0]
    assert res_nose.mean_speed_px_s == pytest.approx(100.0, abs=1e-3)
    assert res_nose.mean_speed_cm_s == pytest.approx(10.0, abs=1e-3)

    # Bone length check: nose-neck == 20px == 2.0cm.
    nose_neck = [
        b
        for b in report.bone_rows
        if b.instance == "resident" and {b.kp_a, b.kp_b} == {"nose", "neck"}
    ][0]
    assert nose_neck.median_px == pytest.approx(20.0, abs=1e-3)
    assert nose_neck.median_cm == pytest.approx(2.0, abs=1e-3)

    # Joint angle checks: nose-neck-right_ear == 90 deg; nose-neck-tail_base == 180 deg.
    angles = {
        (j.instance, j.vertex, frozenset((j.kp_a, j.kp_b))): j.mean_deg
        for j in report.joint_angle_rows
    }
    assert angles[
        ("resident", "neck", frozenset(("nose", "right_ear")))
    ] == pytest.approx(90.0, abs=1e-3)
    assert angles[
        ("resident", "neck", frozenset(("nose", "tail_base")))
    ] == pytest.approx(180.0, abs=1e-3)

    # Egocentric check: anchor is neck, heading is tail_base -> nose (+y = 90 deg),
    # so in egocentric coordinates nose must lie at (+20, 0) and tail_base at (-20, 0)!
    res_ego = [row for row in report.egocentric_rows if row["instance"] == "resident"][
        0
    ]
    assert res_ego["nose_ego_x"] == pytest.approx(20.0, abs=1e-3)
    assert res_ego["nose_ego_y"] == pytest.approx(0.0, abs=1e-3)
    assert res_ego["tail_base_ego_x"] == pytest.approx(-20.0, abs=1e-3)

    # Dyadic nose-to-nose distance is exactly 20px (2cm) across all 5 frames!
    nn = [d for d in report.dyadic_rows if d.kp_a == "nose" and d.kp_b == "nose"][0]
    assert nn.mean_distance_px == pytest.approx(20.0, abs=1e-3)
    assert nn.mean_distance_cm == pytest.approx(2.0, abs=1e-3)
    assert nn.close_contact_frames == 5

    table = format_pose_kinematics_table(report)
    assert "Pose Kinematics Summary" in table

    bundle = write_pose_kinematics_bundle(report, tmp_path / "kin_out")
    for key in (
        "summary_json",
        "keypoint_kinematics_csv",
        "bone_lengths_csv",
        "joint_angles_csv",
        "dyadic_keypoints_csv",
        "egocentric_pose_csv",
    ):
        assert Path(bundle[key]).is_file()


# ---------------------------------------------------------------------------
# 6. Offline Labeled Skeleton Video Renderer
# ---------------------------------------------------------------------------
def test_render_pose_video_color_modes_trails_and_comparison(tmp_path: Path) -> None:
    vid = _write_synthetic_video(tmp_path / "source.mp4", num_frames=8)
    ds = _make_two_mouse_dataset(num_frames=8, inject_noise=True)
    schema = _make_mouse_schema()

    for mode in ("instance", "bodypart", "confidence"):
        out_vid = tmp_path / f"labeled_{mode}.mp4"
        res = render_pose_video(
            vid,
            ds,
            out_vid,
            pose_schema=schema,
            comparison_source=ds if mode == "confidence" else None,
            config=PoseVideoRenderConfig(
                color_mode=mode,
                min_confidence=0.5 if mode == "confidence" else None,
                trail_length=4,
                draw_labels=True,
                draw_instance_names=True,
                draw_frame_index=True,
            ),
        )
        assert out_vid.is_file() and out_vid.stat().st_size > 0
        assert res.frames_rendered == 8
        assert res.frames_with_pose == 8
        cap = cv2.VideoCapture(str(out_vid))
        assert cap.isOpened()
        ret, frame = cap.read()
        cap.release()
        assert ret and frame is not None and frame.shape[:2] == (120, 160)


# ---------------------------------------------------------------------------
# 7. CLI End-to-End Subcommands
# ---------------------------------------------------------------------------
def test_cli_pose_workflow_subcommands_end_to_end(tmp_path: Path, capsys) -> None:
    vid = _write_synthetic_video(tmp_path / "cli_vid.mp4", num_frames=10)
    gt_dir = tmp_path / "gt_lm"
    pred_dir = tmp_path / "pred_lm"
    write_pose_to_labelme_dir(
        _make_two_mouse_dataset(num_frames=10, inject_noise=False), gt_dir
    )
    write_pose_to_labelme_dir(
        _make_two_mouse_dataset(num_frames=10, inject_noise=True), pred_dir
    )

    # extract-frames
    rc = engine_cli.main(
        [
            "extract-frames",
            str(vid),
            "--algo",
            "kmeans",
            "--num-frames",
            "3",
            "-o",
            str(tmp_path / "kf"),
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["num_extracted"] == 3

    # extract-outliers
    rc = engine_cli.main(
        [
            "extract-outliers",
            str(pred_dir),
            "--video",
            str(vid),
            "--num-frames",
            "2",
            "--min-frame-gap",
            "1",
            "-o",
            str(tmp_path / "cli_outliers"),
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["frames_extracted"] == 2

    # eval-pose
    rc = engine_cli.main(
        [
            "eval-pose",
            str(gt_dir),
            str(pred_dir),
            "--output-dir",
            str(tmp_path / "cli_eval"),
            "--json",
        ]
    )
    assert rc == 0
    eval_payload = json.loads(capsys.readouterr().out)
    assert eval_payload["frames_evaluated"] == 10
    assert Path(eval_payload["artifacts"]["html"]).is_file()

    # filter-pose
    rc = engine_cli.main(
        [
            "filter-pose",
            str(pred_dir),
            "-o",
            str(tmp_path / "cli_filt"),
            "--method",
            "median",
            "--min-confidence",
            "0.5",
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["masked_low_confidence"] >= 1

    # swap-pose (dry-run)
    rc = engine_cli.main(
        [
            "swap-pose",
            str(pred_dir),
            "--start-frame",
            "5",
            "--instance-a",
            "mouse_a",
            "--instance-b",
            "mouse_b",
            "--dry-run",
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["frames_modified"] == 5

    # pose-kinematics
    rc = engine_cli.main(
        [
            "pose-kinematics",
            str(gt_dir),
            "--fps",
            "30",
            "--output-dir",
            str(tmp_path / "cli_kin"),
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["frames_analyzed"] == 10

    # render-pose-video
    rc = engine_cli.main(
        [
            "render-pose-video",
            str(vid),
            str(pred_dir),
            "-o",
            str(tmp_path / "cli_labeled.mp4"),
            "--color-mode",
            "confidence",
            "--trail-length",
            "3",
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["frames_rendered"] == 10


# ---------------------------------------------------------------------------
# 8. GUI Ergonomics & Pose Workbench Smoke Tests
# ---------------------------------------------------------------------------
def _ensure_qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_gui_sequencer_skip_prev_auto_advance_and_extract_dialog() -> None:
    _ensure_qapp()
    from annolid.gui.widgets.extract_frame_dialog import ExtractFrameDialog
    from annolid.gui.widgets.keypoint_sequencer import KeypointSequencerWidget

    seq = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict(
            {
                "keypoints": ["nose", "left_ear", "tail_base"],
                "instances": ["mouse_1", "mouse_2"],
            }
        )
        seq.set_pose_schema(schema)
        seq.enable_checkbox.setChecked(True)
        seq.instance_combo.setCurrentIndex(seq.instance_combo.findData("mouse_1"))
        for name in ["nose", "left_ear", "tail_base"]:
            row = seq.keypoint_order().index(name)
            seq.keypoint_list.item(row).setSelected(True)
        seq._lock_selected_keypoints()

        # Skip absent left_ear and step back.
        assert seq.consume_next_label() == "mouse_1_nose"
        assert seq.skip_current_keypoint() == "left_ear"
        assert seq.current_next_keypoint() == "tail_base"
        assert seq.previous_keypoint() == "left_ear"
        assert seq.skip_current_keypoint() == "left_ear"

        # Auto-advance to mouse_2 after finishing tail_base.
        seq.auto_advance_instance_checkbox.setChecked(True)
        assert seq.consume_next_label() == "mouse_1_tail_base"
        assert seq.instance_combo.currentData() == "mouse_2"
        assert seq.consume_next_label() == "mouse_2_nose"
    finally:
        seq.close()

    dlg = ExtractFrameDialog()
    try:
        assert dlg.slider.value() == 100
        dlg.radio_btn_kmeans.setChecked(True)
        assert dlg.algo == "kmeans"
        dlg.radio_btn_uniform.setChecked(True)
        assert dlg.algo == "uniform"
        # Clearing start seconds should not pop a modal dialog.
        dlg.startSecondsLineEdit.setText("")
        assert dlg.start_sconds is None
    finally:
        dlg.close()


def test_gui_canvas_pose_modes_context_menu_and_workbench(tmp_path: Path) -> None:
    app = _ensure_qapp()
    from qtpy import QtCore, QtGui, QtWidgets
    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas
    from annolid.gui.widgets.canvas_context_menu import build_canvas_context_menu
    from annolid.gui.widgets.pose_workbench_dialog import PoseWorkbenchDialog

    canvas = Canvas()
    try:
        schema = _make_mouse_schema()
        canvas.setPoseSchema(schema)
        canvas.setShowPoseEdges(True)

        s_hi = Shape(label="nose", shape_type="point")
        s_hi.addPoint(QtCore.QPointF(20, 20))
        s_hi.other_data = {"score": 0.90}

        s_lo = Shape(label="tail_base", shape_type="point")
        s_lo.addPoint(QtCore.QPointF(60, 20))
        s_lo.other_data = {"score": 0.25}

        canvas.shapes = [s_hi, s_lo]
        canvas.visible = {id(s_hi): True, id(s_lo): True}

        # Confidence threshold hides s_lo (score 0.25 < 0.50).
        canvas.setPoseConfidenceThreshold(0.50)
        assert canvas.isVisible(s_hi) is True
        assert canvas.isVisible(s_lo) is False
        canvas.setPoseConfidenceThreshold(0.0)
        assert canvas.isVisible(s_lo) is True

        # Render in confidence and bodypart color modes + on-canvas pose labels.
        canvas.setShowPoseLabels(True)
        assert Shape.show_pose_labels is True
        img = QtGui.QImage(120, 80, QtGui.QImage.Format_ARGB32)
        for mode in ("confidence", "bodypart", "instance"):
            canvas.setPoseColorMode(mode)
            painter = QtGui.QPainter(img)
            try:
                canvas._draw_pose_edges(painter)
                s_hi.paint(painter, 120, 80)
            finally:
                painter.end()
        canvas.setShowPoseLabels(False)

        # Context menu exposes keypoint visibility actions when a point shape is selected.
        canvas.selectedShapes = [s_hi]
        dummy_win = SimpleNamespace(
            actions=None,
            toggle_keypoint_visibility_action=QtWidgets.QAction(
                "Toggle Keypoint Visibility"
            ),
            mark_keypoint_visible_action=QtWidgets.QAction("Mark Keypoint Visible"),
            mark_keypoint_occluded_action=QtWidgets.QAction("Mark Keypoint Occluded"),
            open_pose_workbench=lambda **_kw: None,
        )
        menu = build_canvas_context_menu(canvas, dummy_win)
        texts = [a.text() for a in menu.actions()]
        assert "Toggle Keypoint Visibility" in texts
        assert "Mark Keypoint Occluded" in texts
        assert "Swap Pose Instances Forward…" in texts

        # PoseWorkbenchDialog syncs canvas controls & emits jumpToFrameRequested on double-click.
        owner = SimpleNamespace(
            canvas=canvas,
            video_file=None,
            current_frame_number=12,
            pose_schema_path=None,
        )
        wb_dlg = PoseWorkbenchDialog(owner_window=owner)
        try:
            assert wb_dlg.workbench.tabs.count() == 5
            assert wb_dlg.workbench.swap_start_spin.value() == 12
            wb_dlg.workbench.canvas_color_combo.setCurrentIndex(
                wb_dlg.workbench.canvas_color_combo.findData("confidence")
            )
            assert canvas.poseColorMode() == "confidence"
            wb_dlg.workbench.canvas_pcutoff_spin.setValue(0.45)
            assert canvas.poseConfidenceThreshold() == pytest.approx(0.45)

            jumped: list[int] = []
            wb_dlg.jumpToFrameRequested.connect(jumped.append)
            wb_dlg.workbench.eval_worst_table.setRowCount(1)
            wb_dlg.workbench.eval_worst_table.setItem(
                0, 0, QtWidgets.QTableWidgetItem("42")
            )
            wb_dlg.workbench._emit_jump_from_table(wb_dlg.workbench.eval_worst_table, 0)
            app.processEvents()
            assert jumped == [42]
        finally:
            wb_dlg.close()
    finally:
        canvas.close()


# ---------------------------------------------------------------------------
# 9. Residual Risk Fixes: DLC H5 Export, Keypoint R-CNN Augment/Video,
#    DINO-KPSeg Sub-Pixel/Multi-Animal Eval, Whole-Skeleton Canvas & 2D Layout
# ---------------------------------------------------------------------------
def test_deeplabcut_h5_analysis_and_training_export_round_trip(
    tmp_path: Path, capsys
) -> None:
    ds_multi = _make_two_mouse_dataset(num_frames=4, inject_noise=False)
    lm_dir = tmp_path / "lm_multi"
    write_pose_to_labelme_dir(ds_multi, lm_dir, write_pose_schema=True)

    # 1. Multi-animal analysis .h5 export + read back via pure-h5py reader.
    h5_out = tmp_path / "multi_dlc.h5"
    res = export_pose_dataset(
        lm_dir, h5_out, target_format="deeplabcut-h5", scorer="lab"
    )
    assert res["target_format"] == "deeplabcut_h5"
    assert h5_out.is_file()
    reloaded = read_pose_dataset(h5_out)
    assert reloaded.source_format == "deeplabcut_h5"
    assert reloaded.is_multi_animal() is True
    assert len(reloaded.frames) == 4
    assert reloaded.frames[0].instances[0].keypoints["nose"].x == pytest.approx(55.0)

    # 2. Single-animal analysis .h5 export via CLI `export-pose --format deeplabcut-h5`.
    h5_cli = tmp_path / "cli_single.h5"
    rc = engine_cli.main(
        [
            "export-pose",
            str(lm_dir),
            "--format",
            "deeplabcut-h5",
            "-o",
            str(h5_cli),
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["target_format"] == "deeplabcut_h5"
    assert h5_cli.is_file()

    # 3. DLC training project writes BOTH CollectedData_<scorer>.csv AND .h5!
    proj_dir = tmp_path / "dlc_proj_h5"
    proj_res = export_pose_dataset(
        lm_dir,
        proj_dir,
        target_format="deeplabcut-project",
        scorer="lab",
        video_name="v1",
    )
    assert proj_res["collected_data_h5"] is not None
    h5_train_file = Path(proj_res["collected_data_h5"])
    assert h5_train_file.is_file()
    # Read directly from the CollectedData .h5 file (not the CSV) to confirm parity.
    loaded_h5_train = read_pose_dataset(
        h5_train_file, source_format="deeplabcut_training"
    )
    assert len(loaded_h5_train.frames) == 4
    assert set(loaded_h5_train.resolved_keypoints()) == set(
        ds_multi.resolved_keypoints()
    )


def test_keypoint_rcnn_pose_augmentation_and_video_folder_inference(
    tmp_path: Path,
) -> None:
    import torch
    from PIL import Image
    from torchvision import tv_tensors

    from annolid.engine.models.keypoint_rcnn import predict_keypoint_rcnn_source
    from annolid.segmentation.maskrcnn.coco_dataset import KeypointAugmentTransform

    # 1. Pose-aware horizontal flip swaps left_ear <-> right_ear and mirrors x -> W - x!
    pil_img = Image.new("RGB", (100, 80), color=(64, 128, 192))
    kps = torch.tensor(
        [
            [
                [30.0, 20.0, 2.0],  # nose
                [20.0, 25.0, 2.0],  # left_ear
                [40.0, 25.0, 2.0],  # right_ear
                [0.0, 0.0, 0.0],  # tail_base (missing)
            ]
        ],
        dtype=torch.float32,
    )
    target = {
        "boxes": tv_tensors.BoundingBoxes(
            torch.tensor([[15.0, 15.0, 45.0, 35.0]], dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(80, 100),
        ),
        "labels": torch.tensor([1], dtype=torch.int64),
        "keypoints": kps,
        "area": torch.tensor([600.0], dtype=torch.float32),
    }

    aug_flip = KeypointAugmentTransform(
        keypoint_names=["nose", "left_ear", "right_ear", "tail_base"],
        hflip_prob=1.0,
        affine_prob=0.0,
        color_jitter=False,
    )
    assert aug_flip.flip_idx == [0, 2, 1, 3]
    img_f, tgt_f = aug_flip(pil_img, target)
    assert img_f.shape == (3, 80, 100)
    # nose x: 100 - 30 = 70; new left_ear comes from old right_ear (x=40 -> 60);
    # new right_ear comes from old left_ear (x=20 -> 80); missing tail_base stays (0, 0, 0).
    assert tgt_f["keypoints"][0, 0].tolist() == pytest.approx([70.0, 20.0, 2.0])
    assert tgt_f["keypoints"][0, 1].tolist() == pytest.approx([60.0, 25.0, 2.0])
    assert tgt_f["keypoints"][0, 2].tolist() == pytest.approx([80.0, 25.0, 2.0])
    assert tgt_f["keypoints"][0, 3].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert tgt_f["boxes"].as_subclass(torch.Tensor)[0].tolist() == pytest.approx(
        [55.0, 15.0, 85.0, 35.0]
    )

    # 2. Pose-aware affine transform keeps shape & finite coordinates.
    aug_aff = KeypointAugmentTransform(
        keypoint_names=["nose", "left_ear", "right_ear", "tail_base"],
        hflip_prob=0.0,
        affine_prob=1.0,
        rotation_deg=15.0,
        scale_range=(0.9, 1.1),
        color_jitter=True,
    )
    img_a, tgt_a = aug_aff(pil_img, target)
    assert img_a.shape == (3, 80, 100)
    assert tgt_a["keypoints"].shape == (1, 4, 3)

    # 3. Batched video + folder inference via predict_keypoint_rcnn_source with stub model.
    class _StubKPRCNN:
        def __call__(self, tensors):
            outs = []
            for _t in tensors:
                outs.append(
                    {
                        "boxes": torch.tensor([[20.0, 20.0, 80.0, 70.0]]),
                        "scores": torch.tensor([0.92]),
                        "keypoints": torch.tensor(
                            [[[60.0, 40.0, 1.0], [50.0, 30.0, 1.0], [50.0, 50.0, 1.0]]]
                        ),
                        "keypoints_scores": torch.tensor([[0.95, 0.90, 0.88]]),
                    }
                )
            return outs

    vid = _write_synthetic_video(tmp_path / "kp_vid.mp4", num_frames=5)
    out_vid = tmp_path / "kp_labeled.mp4"
    vid_summary = predict_keypoint_rcnn_source(
        model=_StubKPRCNN(),
        source=vid,
        keypoint_names=["nose", "left_ear", "right_ear"],
        output_dir=tmp_path / "kp_vid_preds",
        batch_size=2,
        save_video=out_vid,
    )
    assert vid_summary["frames_processed"] == 5
    assert vid_summary["instances_detected"] == 5
    assert Path(vid_summary["pose_schema"]).is_file()
    assert out_vid.is_file() and out_vid.stat().st_size > 0
    # Check boolean-only LabelMe flags contract!
    sample_json = json.loads(
        (tmp_path / "kp_vid_preds" / "kp_vid_000000000.json").read_text()
    )
    pt_shapes = [s for s in sample_json["shapes"] if s["shape_type"] == "point"]
    assert len(pt_shapes) == 3
    assert all(isinstance(v, bool) for v in pt_shapes[0]["flags"].values())
    assert pt_shapes[0]["score"] == pytest.approx(0.95)


def test_dino_kpseg_subpixel_parabolic_refinement_and_multi_animal_eval() -> None:
    import torch

    from annolid.segmentation.dino_kpseg.eval import (
        DinoKPSEGEvalAccumulator,
        _predict_keypoints,
    )
    from annolid.segmentation.dino_kpseg.predictor import DinoKPSEGPredictor

    # Construct a 1-channel 5x5 heatmap peaking at (x=2, y=2) skewed right (+x) and down (+y).
    hm = torch.zeros((1, 5, 5), dtype=torch.float32)
    hm[0, 2, 2] = 0.90
    hm[0, 2, 1] = 0.20
    hm[0, 2, 3] = 0.60  # higher right neighbor -> positive dx
    hm[0, 1, 2] = 0.30
    hm[0, 3, 2] = 0.70  # higher bottom neighbor -> positive dy

    raw_xy = DinoKPSEGPredictor._argmax_coords(hm, patch_size=16, subpixel_refine=False)
    ref_pred_xy = DinoKPSEGPredictor._argmax_coords(
        hm, patch_size=16, subpixel_refine=True
    )
    ref_eval_xy = _predict_keypoints(hm, patch_size=16, subpixel_refine=True)

    assert raw_xy[0] == pytest.approx((40.0, 40.0))
    assert ref_pred_xy[0] == pytest.approx(ref_eval_xy[0], abs=1e-5)
    assert ref_pred_xy[0][0] > 40.0 and ref_pred_xy[0][1] > 40.0

    # Multi-animal Hungarian evaluation + L/R swap rate + RMSE/percentiles.
    acc = DinoKPSEGEvalAccumulator(
        kpt_count=2, thresholds_px=[4.0, 8.0], keypoint_names=["left_ear", "right_ear"]
    )
    gt_a = np.array([[0.20, 0.20, 2.0], [0.30, 0.20, 2.0]], dtype=np.float32)
    gt_b = np.array([[0.70, 0.70, 2.0], [0.80, 0.70, 2.0]], dtype=np.float32)
    # Predictions given in reverse order [B, A], with A's left/right ears swapped!
    pred_b = [(70.0, 70.0), (80.0, 70.0)]
    pred_a_swapped = [(30.0, 20.0), (20.0, 20.0)]
    acc.update_multi_instance(
        pred_instances_xy=[pred_b, pred_a_swapped],
        gt_instances=[gt_a, gt_b],
        image_hw=(100, 100),
        lr_pairs=[(0, 1)],
    )
    summary = acc.summary()
    assert summary["keypoints_visible_total"] == 4
    assert summary["swap_pairs_total"] == 2
    assert summary["swap_pairs_swapped"] == 1
    assert summary["swap_rate"] == pytest.approx(0.5)
    assert summary["rmse_px"] is not None and summary["rmse_px"] > 0.0
    assert summary["p95_error_px"] is not None


def test_gui_whole_skeleton_select_transform_and_draggable_2d_layout(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    from PIL import Image
    from qtpy import QtCore

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas
    from annolid.gui.widgets.canvas_context_menu import build_canvas_context_menu
    from annolid.gui.widgets.keypoint_sequencer import KeypointSequencerWidget

    # 1. Canvas whole-skeleton selection & rigid/similarity transform.
    canvas = Canvas()
    try:
        schema = _make_mouse_schema()
        canvas.setPoseSchema(schema)
        canvas.setShowPoseEdges(True)

        p_a1 = Shape(label="nose", shape_type="point", group_id=1)
        p_a1.addPoint(QtCore.QPointF(40.0, 20.0))
        p_a1.other_data = {"instance_label": "mouse_1"}
        p_a2 = Shape(label="tail_base", shape_type="point", group_id=1)
        p_a2.addPoint(QtCore.QPointF(20.0, 20.0))
        p_a2.other_data = {"instance_label": "mouse_1"}

        p_b1 = Shape(label="nose", shape_type="point", group_id=2)
        p_b1.addPoint(QtCore.QPointF(100.0, 80.0))
        p_b1.other_data = {"instance_label": "mouse_2"}

        canvas.shapes = [p_a1, p_a2, p_b1]
        canvas.visible = {id(s): True for s in canvas.shapes}

        selected = canvas.selectPoseInstanceShapes(p_a1)
        assert {id(s) for s in selected} == {id(p_a1), id(p_a2)}

        # Rotate 90 deg + translate (+10, 0) around centroid (30, 20):
        # (40, 20) -> (40, 30); (20, 20) -> (40, 10)
        moved_signals: list[bool] = []
        canvas.shapeMoved.connect(lambda: moved_signals.append(True))
        n_changed = canvas.transformSelectedPoseShapes(
            dx=10.0, dy=0.0, angle_deg=90.0, scale=1.0
        )
        assert n_changed == 2
        assert len(moved_signals) == 1
        assert (p_a1.points[0].x(), p_a1.points[0].y()) == pytest.approx(
            (40.0, 30.0), abs=1e-3
        )
        assert (p_a2.points[0].x(), p_a2.points[0].y()) == pytest.approx(
            (40.0, 10.0), abs=1e-3
        )
        # Animal 2 must remain completely untouched!
        assert (p_b1.points[0].x(), p_b1.points[0].y()) == pytest.approx((100.0, 80.0))

        # Context menu stays flat (no nested submenus) with icons and exposes
        # whole-instance selection + rigid/similarity pose transform actions.
        dummy_win = SimpleNamespace(actions=None)
        menu = build_canvas_context_menu(canvas, dummy_win)
        non_sep_actions = [a for a in menu.actions() if not a.isSeparator()]
        assert all(a.menu() is None for a in non_sep_actions)
        texts = [a.text() for a in non_sep_actions]
        assert "Select Whole Pose Instance" in texts
        assert "Rotate Pose Left (-15°)" in texts
        assert "Rotate Pose Right (+15°)" in texts
        assert "Scale Pose Up (+10%)" in texts
        assert "Scale Pose Down (-10%)" in texts
    finally:
        canvas.close()

    # 2. Interactive draggable 2D skeleton layout editor & persistence in PoseSchema.
    seq = KeypointSequencerWidget()
    try:
        seq.set_pose_schema(_make_mouse_schema())
        seq.preview.set_custom_positions({"nose": [0.75, 0.25]})
        pts = seq.preview._layout_points(seq.keypoint_order())  # noqa: SLF001
        assert (pts["nose"].x(), pts["nose"].y()) == pytest.approx((0.75, 0.25))

        captured = seq._capture_sequencer_settings()  # noqa: SLF001
        assert captured["preview_layout"]["nose"] == [0.75, 0.25]

        ref_img = tmp_path / "ref_mouse.png"
        Image.new("RGB", (64, 64), color=(80, 120, 160)).save(ref_img)
        assert seq.preview.set_background_image(ref_img) is True
        assert seq.preview._background_pixmap is not None  # noqa: SLF001

        seq.preview.reset_layout()
        assert seq.preview.custom_positions() == {}
        assert seq.preview._background_pixmap is None  # noqa: SLF001
    finally:
        seq.close()
