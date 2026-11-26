from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from annolid.sim2real import (
    RAT_COAT_PATTERNS,
    RAT_KEYPOINT_NAMES,
    SimDatasetConfig,
    SimIterationEvaluation,
    analyze_example_videos,
    build_default_sim_config,
    generate_sim_dataset,
    suggest_next_iteration,
)
from annolid.sim2real.cli import build_parser


def _make_rat_image_folder(
    folder: Path, name_hint: str = "long_evans_rat_treadmill"
) -> Path:
    img_dir = folder / name_hint
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        frame = np.full((240, 320, 3), 205, dtype=np.uint8)
        cx = 110 + i * 25
        cv2.ellipse(frame, (cx, 125), (42, 20), 0, 0, 360, (42, 42, 46), -1)
        cv2.imwrite(str(img_dir / f"frame_{i:03d}.png"), frame)
    return img_dir


def test_sim2real_analyzer_detects_rat_and_builds_default_config(
    tmp_path: Path,
) -> None:
    vid_dir = _make_rat_image_folder(tmp_path, "long_evans_rat_bumpy_side")
    profile = analyze_example_videos(
        [vid_dir], species_hint="auto", output_dir=tmp_path / "env"
    )
    assert profile.species == "rat"
    assert profile.recommended_arena == "bumpy_terrain"
    assert profile.recommended_camera_view == "track_right"
    assert profile.recommended_coat_pair in RAT_COAT_PATTERNS

    cfg = build_default_sim_config(profile, task="yolo_pose", frames_per_scenario=6)
    assert cfg.species == "rat"
    assert cfg.rat_coat == profile.recommended_coat_pair
    assert cfg.rat_terrain_bump_cm > 0.0


def test_sim2real_rat_procedural_fallback_generates_pose_and_seg_datasets(
    tmp_path: Path, monkeypatch
) -> None:
    # Force procedural fallback cleanly so test is fast and headless-safe.
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: None if name == "playwright" else True
    )

    cfg = SimDatasetConfig(
        species="rat",
        task="yolo_pose",
        arena="treadmill",
        camera_view="track_right",
        rat_coat="cinnamon",
        rat_interface="flexor_extensor",
        scenarios=["trot", "walk"],
        frames_per_scenario=4,
        width=320,
        height=240,
        allow_fallback=True,
    )
    artifact = generate_sim_dataset(cfg, tmp_path / "rat_pose_ds")
    assert artifact.species == "rat"
    assert artifact.renderer == "procedural_rat_fallback"
    assert artifact.total_frames == 8
    assert artifact.class_names == ["rat"]

    data_yaml = yaml.safe_load(
        Path(artifact.data_yaml_path).read_text(encoding="utf-8")
    )
    assert data_yaml["names"] == {0: "rat"}
    assert data_yaml["kpt_shape"] == [len(RAT_KEYPOINT_NAMES), 3]
    assert len(data_yaml["flip_idx"]) == len(RAT_KEYPOINT_NAMES)

    # LabelMe files contain rat polygon and instance-prefixed keypoints
    lm_files = sorted(Path(artifact.labelme_dir).glob("*.json"))
    assert len(lm_files) == 8
    sample = json.loads(lm_files[0].read_text(encoding="utf-8"))
    labels = {sh["label"] for sh in sample["shapes"]}
    assert "rat_1" in labels
    assert "rat_1_nose" in labels
    assert "rat_1_left_forepaw" in labels


def test_sim2real_rat_advisor_and_cli_parser() -> None:
    cfg = SimDatasetConfig(species="rat", scenarios=["trot"])
    ev = SimIterationEvaluation(
        evaluated_frames=20,
        frames_with_detections=16,
        detection_coverage=0.80,
        expected_subject_count=1,
        mean_detections_per_frame=0.4,
        subject_count_match_rate=0.4,
        mean_confidence=0.65,
        temporal_iou_stability=0.70,
        sim2real_score=0.55,
    )
    adj = suggest_next_iteration(current_config=cfg, evaluation=ev)
    assert len(adj.suggested_sim_config["scenarios"]) > 1

    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--project",
            "/tmp/p",
            "--species",
            "rat",
            "--arena",
            "bumpy_terrain",
            "--rat-coat",
            "gray",
            "--rat-interface",
            "position_stiffness",
            "--rat-terrain-bump-cm",
            "2.5",
        ]
    )
    assert args.species == "rat"
    assert args.rat_coat == "gray"
    assert args.rat_interface == "position_stiffness"
    assert args.rat_terrain_bump_cm == 2.5


def test_sim2real_dialog_supports_rat_species_selection(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    from qtpy import QtWidgets

    from annolid.gui.widgets.sim2real_dialog import Sim2RealStudioDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    dlg = Sim2RealStudioDialog(project_dir=tmp_path / "proj")
    try:
        rat_idx = dlg.sim_species_combo.findData("rat")
        assert rat_idx >= 0
        dlg.sim_species_combo.setCurrentIndex(rat_idx)
        cfg = dlg.build_sim_config_from_ui()
        assert cfg.species == "rat"
        assert cfg.rat_coat in RAT_COAT_PATTERNS
        assert cfg.arena in {
            "treadmill",
            "open_field",
            "circular_open_field",
            "bumpy_terrain",
            "elevated_plus_maze",
        }
        assert len(cfg.scenarios) >= 1
    finally:
        dlg.close()
        dlg.deleteLater()
