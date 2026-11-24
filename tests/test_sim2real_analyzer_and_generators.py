from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from annolid.sim2real import (
    SimDatasetConfig,
    analyze_example_videos,
    apply_environment_photometry,
    build_default_sim_config,
    composite_foreground_onto_background,
    generate_sim_dataset,
)


def _write_synthetic_video(
    path: Path,
    *,
    width: int = 320,
    height: int = 240,
    n_frames: int = 12,
    bg_bgr: tuple[int, int, int] = (180, 180, 180),
    subject_bgr: tuple[int, int, int] = (35, 35, 35),
    subject_radius: int = 18,
    n_subjects: int = 2,
    circular_arena: bool = False,
    wire_bars: bool = False,
) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    try:
        for i in range(n_frames):
            frame = np.full((height, width, 3), bg_bgr, dtype=np.uint8)
            if circular_arena:
                cv2.circle(
                    frame,
                    (width // 2, height // 2),
                    int(min(width, height) * 0.44),
                    (60, 60, 60),
                    4,
                )
            for s in range(n_subjects):
                ang = (i / max(1, n_frames)) * 2.0 * math.pi + s * 2.4
                cx = int(width * (0.35 + 0.30 * (s % 2)) + 28 * math.cos(ang))
                cy = int(height * 0.50 + 24 * math.sin(ang))
                cv2.ellipse(
                    frame,
                    (cx, cy),
                    (subject_radius, max(3, int(subject_radius * 0.55))),
                    math.degrees(ang),
                    0,
                    360,
                    subject_bgr,
                    -1,
                )
            if wire_bars:
                for y in range(25, height - 15, 28):
                    cv2.line(frame, (10, y), (width - 10, y), (20, 20, 20), 2)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_analyze_dark_ir_mouse_video_and_photometry(tmp_path: Path) -> None:
    vid = _write_synthetic_video(
        tmp_path / "mouse_ir_night_cage.mp4",
        bg_bgr=(46, 46, 46),
        subject_bgr=(190, 190, 190),
        subject_radius=24,
        n_subjects=2,
        wire_bars=True,
    )
    analysis_dir = tmp_path / "analysis"
    profile = analyze_example_videos(
        [vid], species_hint="auto", output_dir=analysis_dir
    )

    assert profile.species == "mouse"
    assert profile.is_grayscale is True
    assert profile.recommended_lighting == "ir_night"
    assert profile.recommended_arena == "home_cage"
    assert profile.recommended_occluder == "wire_bars"
    assert Path(profile.reference_background_path).is_file()
    assert Path(profile.sample_montage_path).is_file()

    cfg = build_default_sim_config(profile, frames_per_scenario=8)
    assert cfg.species == "mouse"
    assert cfg.lighting == "ir_night"
    assert cfg.occluder == "wire_bars"

    # Photometric transfer should push bright color synthetic image toward dark monochrome profile
    bright_sim = np.full((120, 160, 3), (200, 140, 80), dtype=np.uint8)
    matched = apply_environment_photometry(
        bright_sim, profile, strength=0.9, domain_randomization=0.0
    )
    assert matched.shape == bright_sim.shape
    assert matched.dtype == np.uint8
    assert float(matched.mean()) < float(bright_sim.mean()) - 40.0

    # Background compositing preserves exact shape and blends foreground cleanly
    fg_mask = np.zeros((120, 160), dtype=bool)
    fg_mask[40:80, 50:110] = True
    bg = np.full((120, 160, 3), (30, 30, 30), dtype=np.uint8)
    comp = composite_foreground_onto_background(bright_sim, fg_mask, bg)
    assert comp.shape == bright_sim.shape
    assert float(comp[60, 80].mean()) > float(comp[5, 5].mean()) + 30.0


def test_analyze_bright_fly_circular_arena(tmp_path: Path) -> None:
    vid = _write_synthetic_video(
        tmp_path / "drosophila_walk_arena.mp4",
        bg_bgr=(220, 224, 228),
        subject_bgr=(35, 38, 42),
        subject_radius=6,
        n_subjects=2,
        circular_arena=True,
    )
    profile = analyze_example_videos(
        [vid], species_hint="auto", output_dir=tmp_path / "out"
    )
    assert profile.species == "fly"
    assert profile.recommended_arena == "circular_open_field"
    assert profile.recommended_camera_view in {
        "walker/track1",
        "walker/hero",
        "top_camera",
    }

    cfg = build_default_sim_config(profile, frames_per_scenario=6)
    assert cfg.species == "fly"
    assert cfg.composite_on_real_background is True


def test_decode_threejs_four_tuple_rle_segmentation() -> None:
    from annolid.sim2real.generators import _decode_segmentation_rle

    # 8x8 image: solid 3-row block (rows 2..4 = 24 pixels) with track=1, fine=2, coarse=1
    payload = {
        "width": 8,
        "height": 8,
        "rle": [16, 0, 0, 0, 24, 1, 2, 1, 24, 0, 0, 0],
    }
    track_map, fine_map, coarse_map = _decode_segmentation_rle(payload)
    assert track_map.shape == (8, 8)
    assert int(np.count_nonzero(track_map == 1)) == 24
    assert int(np.count_nonzero(fine_map == 2)) == 24
    assert int(np.count_nonzero(coarse_map == 1)) == 24


def test_generate_mouse_and_fly_datasets_all_tasks_and_real_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prepare tiny real labeled LabelMe directory to test sim+real dataset mixing
    real_dir = tmp_path / "real_labels"
    real_dir.mkdir()
    real_img = np.full((240, 320, 3), 140, dtype=np.uint8)
    cv2.imwrite(str(real_dir / "real_001.jpg"), real_img)
    (real_dir / "real_001.json").write_text(
        json.dumps(
            {
                "version": "5.0.1",
                "flags": {},
                "shapes": [
                    {
                        "label": "mouse_1",
                        "points": [[60, 60], [140, 60], [140, 120], [60, 120]],
                        "group_id": 1,
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": "real_001.jpg",
                "imageData": None,
                "imageHeight": 240,
                "imageWidth": 320,
            }
        ),
        encoding="utf-8",
    )

    # Force deterministic offline fallback path for fast CI unit tests
    import annolid.sim2real.generators as gen_mod

    monkeypatch.setattr(gen_mod.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(gen_mod, "_find_flybody_python", lambda: None)

    # Without allow_fallback, both species must fail explicitly (never silently substitute!)
    with pytest.raises(RuntimeError, match="allow_fallback=True"):
        generate_sim_dataset(
            SimDatasetConfig(species="mouse", allow_fallback=False),
            tmp_path / "mouse_fail",
        )
    with pytest.raises(RuntimeError, match="allow_fallback=True"):
        generate_sim_dataset(
            SimDatasetConfig(species="fly", allow_fallback=False),
            tmp_path / "fly_fail",
        )

    # 1. Mouse YOLO-Seg (parts level + mixed real frames)
    mouse_cfg = SimDatasetConfig(
        species="mouse",
        task="yolo_seg",
        export_level="parts",
        scenarios=["arena_explore", "social_huddle"],
        frames_per_scenario=4,
        width=320,
        height=240,
        real_labeled_dir=str(real_dir),
        real_oversample_factor=2,
        allow_fallback=True,
    )
    mouse_art = generate_sim_dataset(mouse_cfg, tmp_path / "mouse_seg_parts")
    assert mouse_art.renderer == "procedural_mouse_fallback"
    assert mouse_art.total_frames == 10  # 2*4 sim + 2 oversampled real
    assert Path(mouse_art.data_yaml_path).is_file()
    data_yaml = yaml.safe_load(
        Path(mouse_art.data_yaml_path).read_text(encoding="utf-8")
    )
    assert data_yaml["path"] == "."
    assert data_yaml["train"] == "images/train"
    assert data_yaml["val"] == "images/val"
    assert set(["head", "torso", "limbs", "tail"]).issubset(
        set(data_yaml["names"].values())
    )
    assert len(mouse_art.preview_images) >= 1

    # 2. Fly YOLO-Pose (11 anatomical keypoints + flip_idx)
    fly_cfg = SimDatasetConfig(
        species="fly",
        task="yolo_pose",
        scenarios=["walk_imitation", "grooming"],
        frames_per_scenario=4,
        width=320,
        height=240,
        fly_subject_count=2,
        allow_fallback=True,
    )
    fly_art = generate_sim_dataset(fly_cfg, tmp_path / "fly_pose")
    assert fly_art.renderer == "procedural_fly_fallback"
    assert fly_art.total_frames == 8
    fly_yaml = yaml.safe_load(Path(fly_art.data_yaml_path).read_text(encoding="utf-8"))
    assert fly_yaml["kpt_shape"] == [11, 3]
    assert len(fly_yaml["flip_idx"]) == 11
    train_lbls = list(
        (Path(fly_art.data_yaml_path).parent / "labels" / "train").glob("*.txt")
    )
    assert train_lbls
    first_line = train_lbls[0].read_text(encoding="utf-8").strip().splitlines()[0]
    # class + 4 bbox + 11*3 keypoints = 38 tokens
    assert len(first_line.split()) == 38
