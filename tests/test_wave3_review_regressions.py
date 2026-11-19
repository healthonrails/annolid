from __future__ import annotations

import json
import math
import subprocess
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from annolid.annotation.batch_relabel import _write_json_atomic
from annolid.data.videos import extract_frames, key_frames
from annolid.datasets.builders.label_index_yolo import build_yolo_from_label_index
from annolid.gui.workers import FrameExtractorWorker, PredictionWorker, TrackAllWorker
from annolid.postprocessing.identity_governor import run_identity_governor
from annolid.postprocessing.temporal_identity_repair import run_temporal_identity_repair
from annolid.postprocessing.video_timestamp_annotator import (
    annotate_csv,
    extract_frame_timestamps,
    match_video_for_csv,
)
from annolid.postprocessing.zone_schema import build_zone_shape
from annolid.utils.annotation_store import AnnotationStore, load_labelme_json


def _rect_pts(cx: float, cy: float, size: float = 10.0) -> list[list[float]]:
    half = size / 2.0
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]


def _whole_animal_shape(label: str, group_id: int, cx: float, cy: float) -> dict:
    return {
        "label": label,
        "instance_label": label,
        "shape_type": "polygon",
        "points": _rect_pts(cx, cy),
        "group_id": int(group_id),
        "flags": {
            "instance_label": label,
            "group_id": int(group_id),
        },
    }


def _nose_part_shape(parent: str, group_id: int, x: float, y: float) -> dict:
    return {
        "label": f"{parent}:nose",
        "parent_instance": parent,
        "instance_label": parent,
        "shape_type": "polygon",
        "points": _rect_pts(x, y, size=3.0),
        "group_id": int(group_id),
        "flags": {
            "semantic_type": "body_part",
            "parent_instance": parent,
            "body_part": "nose",
            "instance_label": parent,
        },
    }


def _write_synthetic_video(
    path: Path, bgr_color: tuple[int, int, int], n_frames: int = 6
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    assert writer.isOpened()
    try:
        for idx in range(n_frames):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :] = bgr_color
            frame[10 : 10 + (idx % 5) + 4, 10:20] = (255, 255, 255)
            writer.write(frame)
    finally:
        writer.release()


def test_identity_governor_repairs_annotation_store_preserves_int_group_id_and_swaps_child_parts(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    # Frame 0 is a manual seed JSON (canonical: alpha on right x=85, beta on left x=15);
    # frames 1-2 live exclusively in AnnotationStore with swapped labels & group_ids.
    seed_payload = {
        "version": "5.0.1",
        "shapes": [
            _whole_animal_shape("alpha", 1, 85.0, 50.0),
            _nose_part_shape("alpha", 1, 86.0, 50.0),
            _whole_animal_shape("beta", 2, 15.0, 50.0),
            _nose_part_shape("beta", 2, 16.0, 50.0),
        ],
        "imagePath": "session_000000000.jpg",
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 100,
    }
    _write_json_atomic(session_dir / "session_000000000.json", seed_payload)

    store = AnnotationStore(session_dir / "session_annotations.ndjson")
    for frame_idx in (1, 2):
        # Swapped labels & group_ids in AnnotationStore (beta placed on right where alpha belongs, alpha on left).
        store.append_frame(
            {
                "frame": frame_idx,
                "version": "annolid",
                "shapes": [
                    _whole_animal_shape("beta", 2, 85.0, 50.0),
                    _nose_part_shape("beta", 2, 86.0, 50.0),
                    _whole_animal_shape("alpha", 1, 15.0, 50.0),
                    _nose_part_shape("alpha", 1, 16.0, 50.0),
                ],
                "imagePath": f"session_{frame_idx:09d}.jpg",
                "imageHeight": 100,
                "imageWidth": 100,
            }
        )

    zone_path = tmp_path / "zones.json"
    zone_path.write_text(
        json.dumps(
            {
                "shapes": [
                    build_zone_shape(
                        "left_lane",
                        [[0, 0], [30, 100]],
                        shape_type="rectangle",
                        zone_kind="chamber",
                    ),
                    build_zone_shape(
                        "right_lane",
                        [[70, 0], [100, 100]],
                        shape_type="rectangle",
                        zone_kind="chamber",
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )

    policy = {
        "rules": [
            {
                "name": "alpha_on_right",
                "assign_label": "alpha",
                "conditions": [
                    {"metric": "zone.inside.right_lane", "op": "eq", "value": True}
                ],
            },
            {
                "name": "beta_on_left",
                "assign_label": "beta",
                "conditions": [
                    {"metric": "zone.inside.left_lane", "op": "eq", "value": True}
                ],
            },
        ],
        "canonical_track_ids": {"alpha": "1", "beta": "2"},
    }

    res = run_identity_governor(
        annotation_dir=session_dir,
        policy=policy,
        zone_file=zone_path,
        apply_changes=True,
    )
    assert res.scanned_files == 3
    assert res.updated_files == 2
    assert not (session_dir / "session_000000001.json").exists()

    repaired_f1 = load_labelme_json(session_dir / "session_000000001.json")
    shapes_f1 = repaired_f1["shapes"]
    # Shape 0 was at x=85 (right_lane) -> repaired to alpha with int group_id == 1
    assert shapes_f1[0]["label"] == "alpha"
    assert shapes_f1[0]["group_id"] == 1
    assert isinstance(shapes_f1[0]["group_id"], int)
    # Child body parts swapped simultaneously without colliding!
    assert shapes_f1[1]["label"] == "alpha:nose"
    assert shapes_f1[1]["parent_instance"] == "alpha"
    assert shapes_f1[3]["label"] == "beta:nose"
    assert shapes_f1[3]["parent_instance"] == "beta"


def test_temporal_identity_repair_updates_annotation_store_records_atomically(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "cutie_session"
    session_dir.mkdir()
    store = AnnotationStore(session_dir / "cutie_session_annotations.ndjson")

    # Frame 0: alpha at (20, 50), beta at (80, 50)
    store.append_frame(
        {
            "frame": 0,
            "version": "annolid",
            "shapes": [
                {
                    "label": "alpha",
                    "shape_type": "polygon",
                    "points": _rect_pts(20.0, 50.0),
                },
                {
                    "label": "beta",
                    "shape_type": "polygon",
                    "points": _rect_pts(80.0, 50.0),
                },
            ],
            "imagePath": "cutie_session_000000000.jpg",
        }
    )
    # Frame 1: swapped labels at nearby locations (22, 50) and (78, 50)
    store.append_frame(
        {
            "frame": 1,
            "version": "annolid",
            "shapes": [
                {
                    "label": "beta",
                    "shape_type": "polygon",
                    "points": _rect_pts(22.0, 50.0),
                },
                {
                    "label": "alpha",
                    "shape_type": "polygon",
                    "points": _rect_pts(78.0, 50.0),
                },
            ],
            "imagePath": "cutie_session_000000001.jpg",
        }
    )

    result = run_temporal_identity_repair(
        session_dir,
        expected_instance_count=2,
        max_match_distance=40.0,
        apply_changes=True,
    )
    assert result.scanned_files == 2
    assert result.updated_files == 1
    assert not (session_dir / "cutie_session_000000001.json").exists()

    repaired = load_labelme_json(session_dir / "cutie_session_000000001.json")
    assert [s["label"] for s in repaired["shapes"]] == ["alpha", "beta"]


def test_key_frames_preserves_rgb_colors_and_extract_frames_skips_headless_highgui(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "red_clip.avi"
    # Pure red in OpenCV BGR = (0, 0, 255)
    _write_synthetic_video(video_path, bgr_color=(0, 0, 255), n_frames=5)

    monkeypatch.setattr("annolid.data.videos.get_keyframe_timestamps", lambda _p: [])
    monkeypatch.setattr(
        cv2,
        "waitKey",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cv2.waitKey must not run when show_flow=False")
        ),
    )

    kf_dir = key_frames(str(video_path), out_dir=tmp_path / "kf_out")
    saved_pngs = sorted(kf_dir.glob("*.png"))
    assert saved_pngs
    with Image.open(saved_pngs[0]) as img:
        rgb_arr = np.asarray(img.convert("RGB"))
    # Corner pixel (0, 0) must remain red (R > 180, B < 60), not swapped to blue.
    r_val, _g_val, b_val = (int(v) for v in rgb_arr[0, 0])
    assert r_val > 180
    assert b_val < 60

    # extract_frames(..., algo='keyframes') returns after key_frames without falling through
    events = list(
        extract_frames(
            str(video_path),
            num_frames=2,
            out_dir=str(tmp_path / "ef_kf"),
            algo="keyframes",
            show_flow=False,
            prediction=False,
        )
    )
    assert len(events) == 1
    assert events[0][0] == 100

    # extract_frames(..., algo='random', show_flow=False) also never touches cv2.waitKey
    flowless_events = list(
        extract_frames(
            str(video_path),
            num_frames=2,
            out_dir=str(tmp_path / "ef_rand"),
            algo="random",
            show_flow=False,
            prediction=False,
        )
    )
    assert flowless_events[-1][0] == 100


def test_frame_extractor_worker_is_linear_and_track_all_worker_forwards_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vid_dir = tmp_path / "videos"
    vid_dir.mkdir()
    v1 = vid_dir / "a.avi"
    v2 = vid_dir / "b.mkv"
    v3_unselected = vid_dir / "c_ignore.avi"
    for vp in (v1, v2, v3_unselected):
        vp.write_bytes(b"stub")

    extracted_calls: list[str] = []
    monkeypatch.setattr(
        "annolid.gui.workers.extract_frames_from_video",
        lambda video_path, output_folder, num_frames=5: extracted_calls.append(
            Path(video_path).name
        )
        or [],
    )
    fe_worker = FrameExtractorWorker(
        [str(v1), str(v2)],
        output_folder=str(tmp_path / "extracted"),
        num_frames=3,
    )
    fe_worker.run()
    assert extracted_calls == ["a.avi", "b.mkv"]

    # TrackAllWorker initializes logger before validation, accepts .mkv, and forwards stop()
    ta_worker = TrackAllWorker([str(v2), str(tmp_path / "nonexistent.mp4")])
    assert ta_worker.video_paths == [str(v2)]
    pred_worker = PredictionWorker()
    ta_worker._active_pred_worker = pred_worker
    assert not pred_worker.is_stopped()
    ta_worker.stop()
    assert pred_worker.is_stopped()


def test_video_timestamp_annotator_prefix_matching_zero_pts_and_nan_safety(
    tmp_path: Path,
    monkeypatch,
) -> None:
    v1 = tmp_path / "trial_1.mp4"
    v10 = tmp_path / "trial_10.mp4"
    csv10 = tmp_path / "trial_10_tracked.csv"
    assert match_video_for_csv(csv10, [v1, v10]) == v10

    fake_ffprobe_json = json.dumps(
        {
            "frames": [
                {"pts_time": 0.0},
                {"pts_time": None, "pkt_pts_time": None},
                {"pts_time": "0.250"},
            ]
        }
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_kw: subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout=fake_ffprobe_json, stderr=""
        ),
    )
    monkeypatch.setattr(
        "annolid.postprocessing.video_timestamp_annotator.run_gap_analysis_for_video",
        lambda _vp: True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = extract_frame_timestamps(v10)
    assert len(ts) == 3
    assert ts[0] == 0.0
    assert math.isnan(ts[1])
    assert ts[2] == 0.25

    pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 1.0, "cy": 2.0},
            {
                "frame_number": float("nan"),
                "instance_name": "mouse",
                "cx": 1.0,
                "cy": 2.0,
            },
            {"frame_number": 2, "instance_name": "mouse", "cx": 3.0, "cy": 4.0},
        ]
    ).to_csv(csv10, index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        annotate_csv(csv10, v10)
    annotated = pd.read_csv(csv10)
    assert annotated.loc[0, "real_timestamp_sec"] == 0.0
    assert pd.isna(annotated.loc[1, "real_timestamp_sec"])
    assert annotated.loc[2, "real_timestamp_sec"] == 0.25


def test_build_yolo_from_label_index_applies_exif_transpose_and_leaves_no_orphan_png(
    tmp_path: Path,
) -> None:
    ds_dir = tmp_path / "raw_ds"
    ds_dir.mkdir()

    # Raw 40x80 JPEG with EXIF Orientation=6 (90 CW -> transposed display size is 80x40)
    jpg_path = ds_dir / "rotated.jpg"
    raw_img = Image.new("RGB", (40, 80), color=(40, 90, 140))
    exif = raw_img.getexif()
    exif[274] = 6  # 0x0112 Orientation = Rotate 90 CW
    raw_img.save(jpg_path, exif=exif)

    valid_json = ds_dir / "rotated.json"
    valid_json.write_text(
        json.dumps(
            {
                "version": "5.0.0",
                "flags": {},
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [
                            [10.0, 5.0],
                            [70.0, 5.0],
                            [70.0, 35.0],
                            [10.0, 35.0],
                        ],
                    }
                ],
                "imagePath": "rotated.jpg",
                "imageData": None,
                "imageHeight": 40,
                "imageWidth": 80,
            }
        ),
        encoding="utf-8",
    )

    corrupt_img = ds_dir / "corrupt.jpg"
    Image.new("RGB", (32, 32), color=(10, 10, 10)).save(corrupt_img)
    corrupt_json = ds_dir / "corrupt.json"
    corrupt_json.write_text("{not valid json", encoding="utf-8")

    index_file = ds_dir / "label_index.jsonl"
    index_file.write_text(
        "\n".join(
            [
                json.dumps({"image_path": str(jpg_path), "json_path": str(valid_json)}),
                json.dumps(
                    {"image_path": str(corrupt_img), "json_path": str(corrupt_json)}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Note: is_labeled_labelme_json skips unreadable JSONs; test list-JSON payload too so
    # iter_indexed_pairs yields it if shapes-check is bypassed with include_empty=True.
    list_json = ds_dir / "list_payload.json"
    list_img = ds_dir / "list_payload.jpg"
    Image.new("RGB", (32, 32), color=(10, 10, 10)).save(list_img)
    list_json.write_text("[]", encoding="utf-8")
    with index_file.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"image_path": str(list_img), "json_path": str(list_json)})
            + "\n"
        )

    summary = build_yolo_from_label_index(
        index_file=index_file,
        output_dir=tmp_path / "yolo_out",
        dataset_name="YOLO_exif",
        val_size=0.0,
        test_size=0.0,
        task="segmentation",
        include_empty=True,
        keep_staging=True,
    )
    assert summary["status"] == "ok"
    staging_dir = Path(str(summary["staging_dir"]))
    staged_pngs = list(staging_dir.glob("*.png"))
    assert len(staged_pngs) == 1
    assert "rotated__" in staged_pngs[0].name
    with Image.open(staged_pngs[0]) as staged_im:
        assert staged_im.size == (80, 40)
