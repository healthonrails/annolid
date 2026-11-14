from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from annolid.annotation.pose_schema import PoseSchema
from annolid.engine import cli as engine_cli
from annolid.postprocessing.track_qc import evaluate_track_qc
from annolid.postprocessing.zone_schema import build_zone_shape


def _box_poly(
    cx: float, cy: float, w: float = 20.0, h: float = 20.0
) -> List[List[float]]:
    hw, hh = w / 2.0, h / 2.0
    return [
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ]


def _poly_shape(
    label: str,
    cx: float,
    cy: float,
    *,
    w: float = 20.0,
    h: float = 20.0,
    score: float | None = None,
    track_id: str | None = None,
) -> Dict[str, Any]:
    shape: Dict[str, Any] = {
        "label": label,
        "shape_type": "polygon",
        "points": _box_poly(cx, cy, w=w, h=h),
        "flags": {},
    }
    if score is not None:
        shape["score"] = score
    if track_id is not None:
        shape["track_id"] = track_id
        shape["tracking_id"] = track_id
        shape["group_id"] = track_id
        shape["instance_label"] = label
        shape["flags"] = {
            "instance_label": label,
            "track_id": track_id,
            "tracking_id": track_id,
            "group_id": track_id,
        }
    return shape


def _point_shape(label: str, x: float, y: float) -> Dict[str, Any]:
    return {
        "label": label,
        "shape_type": "point",
        "points": [[x, y]],
        "flags": {},
    }


def _write_labelme_frame(
    folder: Path, frame_idx: int, shapes: List[Dict[str, Any]], prefix: str = "rec"
) -> Path:
    path = folder / f"{prefix}_{frame_idx:09d}.json"
    payload = {
        "version": "5.0.1",
        "shapes": shapes,
        "imagePath": f"{prefix}_{frame_idx:09d}.png",
        "imageData": None,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_evaluate_track_qc_detects_anomalies_and_prioritizes_swap_bout(
    tmp_path: Path,
) -> None:
    ann_dir = tmp_path / "multi_mouse_session"
    ann_dir.mkdir()

    pose = PoseSchema(
        keypoints=["nose", "tail_base"],
        edges=[("nose", "tail_base")],
        instances=["mouse_1", "mouse_2"],
    )

    # Side-by-side trajectories moving right (+4 px/frame):
    # mouse_1 in upper lane (y=40), mouse_2 in lower lane (y=72); at f=4 they brush close
    # with overlapping masks and then swap identities from f=5 onward!
    for f in range(18):
        x_pos = 30.0 + 4.0 * f
        m1_xy = (x_pos, 40.0) if f < 5 else (x_pos, 72.0)
        m2_xy = (x_pos, 72.0) if f < 5 else (x_pos, 40.0)
        shapes: List[Dict[str, Any]] = []

        if f == 4:
            # Close-contact brush with mask overlap (IoU = 0.33 > 0.25) + low confidence on mouse_1
            shapes.append(
                _poly_shape("mouse_1", x_pos, 45.0, w=32.0, h=32.0, score=0.18)
            )
            shapes.append(
                _poly_shape("mouse_2", x_pos, 61.0, w=32.0, h=32.0, score=0.92)
            )
        elif f == 9:
            # Missing mouse_2 for 1 frame (occlusion gap)
            shapes.append(_poly_shape("mouse_1", m1_xy[0], m1_xy[1], score=0.95))
        elif f == 12:
            # Duplicate mouse_1 + 4x area jump on mouse_2 (40x40 = 1600 vs 20x20 = 400)
            shapes.append(_poly_shape("mouse_1", m1_xy[0], m1_xy[1], score=0.95))
            shapes.append(_poly_shape("mouse_1", m1_xy[0] + 2.0, m1_xy[1], score=0.55))
            shapes.append(
                _poly_shape("mouse_2", m2_xy[0], m2_xy[1], w=40.0, h=40.0, score=0.95)
            )
        elif f == 17:
            # Moderate velocity spike (60px -> warning/error, lower score than swap+overlap cluster)
            shapes.append(_poly_shape("mouse_1", m1_xy[0] + 60.0, m1_xy[1], score=0.95))
            shapes.append(_poly_shape("mouse_2", m2_xy[0], m2_xy[1], score=0.95))
        else:
            shapes.append(_poly_shape("mouse_1", m1_xy[0], m1_xy[1], score=0.95))
            shapes.append(_poly_shape("mouse_2", m2_xy[0], m2_xy[1], score=0.95))

        # Add keypoints: normal 20px bone length except frame 14 where tail_base stretches to 80px (4.0x)
        tail_offset = 80.0 if f == 14 else 20.0
        shapes.append(_point_shape("mouse_1_nose", m1_xy[0], m1_xy[1]))
        shapes.append(
            _point_shape("mouse_1_tail_base", m1_xy[0] - tail_offset, m1_xy[1])
        )
        _write_labelme_frame(ann_dir, f, shapes)

    report = evaluate_track_qc(
        ann_dir,
        pose_schema=pose,
        expected_instances=2,
        bout_merge_gap_frames=1,
    )

    codes = {issue.code for issue in report.issues}
    assert "identity_swap_suspect" in codes
    assert "duplicate_instance" in codes
    assert "spatial_overlap" in codes
    assert "anatomical_edge_outlier" in codes
    assert "velocity_spike" in codes
    assert "area_jump" in codes
    assert "missing_instance_gap" in codes
    assert "low_confidence" in codes

    assert report.has_errors is True
    assert len(report.review_queue) >= 3
    top_bout = report.review_queue[0]
    assert top_bout.rank == 1
    assert "identity_swap_suspect" in top_bout.issue_codes
    assert top_bout.recommended_action == "verify_identity_or_swap"
    assert 4 <= top_bout.anchor_frame <= 6


def test_evaluate_track_qc_csv_disambiguates_keypoints_and_tracked_centroids(
    tmp_path: Path,
) -> None:
    pose = PoseSchema(
        keypoints=["nose", "tail_base"],
        edges=[("nose", "tail_base")],
        instances=["mouse_1", "mouse_2"],
    )

    tracking_csv = tmp_path / "video_tracking.csv"
    with tracking_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame_number",
                "x1",
                "y1",
                "x2",
                "y2",
                "instance_name",
                "class_score",
                "cx",
                "cy",
            ]
        )
        for f in range(5):
            # Bounding-box rows for two mice
            writer.writerow([f, 10 + f, 10, 30 + f, 30, "mouse_1", 0.95, 20 + f, 20])
            writer.writerow([f, 80 - f, 10, 100 - f, 30, "mouse_2", 0.94, 90 - f, 20])
            # Point-like keypoint rows exported by labelme2csv (x1 == x2, y1 == y2)
            stretch = 75.0 if f == 3 else 15.0
            writer.writerow(
                [f, 20 + f, 20, 20 + f, 20, "mouse_1_nose", 0.9, 20 + f, 20]
            )
            writer.writerow(
                [
                    f,
                    20 + f - stretch,
                    20,
                    20 + f - stretch,
                    20,
                    "mouse_1_tail_base",
                    0.9,
                    20 + f - stretch,
                    20,
                ]
            )

    report = evaluate_track_qc(tracking_csv, pose_schema=pose, expected_instances=2)
    # Keypoint rows must NOT be counted as extra animal instances!
    assert [inst.instance for inst in report.instances] == ["mouse_1", "mouse_2"]
    codes = {issue.code for issue in report.issues}
    assert "count_mismatch" not in codes
    assert "anatomical_edge_outlier" in codes

    # Also test centroid-only *_tracked.csv without x1/y1/x2/y2 columns
    tracked_csv = tmp_path / "video_tracked.csv"
    with tracked_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_number", "instance_name", "cx", "cy"])
        for f in range(4):
            cx1 = 200.0 if f == 3 else float(10 + 2 * f)
            writer.writerow([f, "mouse_1", cx1, 50.0])
            writer.writerow([f, "mouse_2", float(100 + 2 * f), 80.0])

    tracked_report = evaluate_track_qc(tracked_csv)
    assert tracked_report.source_kind == "tracking_csv"
    assert any(i.code == "velocity_spike" for i in tracked_report.issues)


def test_track_qc_cli_outputs_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_dir = tmp_path / "clean_tracks"
    clean_dir.mkdir()
    for f in range(5):
        _write_labelme_frame(
            clean_dir,
            f,
            [
                _poly_shape("mouse_1", 20.0 + 2.0 * f, 30.0, score=0.95),
                _poly_shape("mouse_2", 120.0 + 2.0 * f, 80.0, score=0.96),
            ],
        )

    out_json = tmp_path / "qc.json"
    issues_csv = tmp_path / "issues.csv"
    review_csv = tmp_path / "review.csv"

    rc = engine_cli.main(
        [
            "track-qc",
            str(clean_dir),
            "--expected-instances",
            "2",
            "--strict",
            "--json",
            "--out",
            str(out_json),
            "--issues-csv",
            str(issues_csv),
            "--review-csv",
            str(review_csv),
        ]
    )
    assert rc == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["error_count"] == 0
    assert stdout_payload["warning_count"] == 0
    assert stdout_payload["provenance"]["schema"] == "track_qc/v1"
    assert out_json.exists()
    assert issues_csv.exists()
    assert review_csv.exists()

    # Introduce a missing-frame warning and verify --strict returns exit code 2 vs non-strict 0
    (clean_dir / "rec_000000002.json").write_text(
        json.dumps(
            {
                "version": "5.0.1",
                "shapes": [_poly_shape("mouse_1", 24.0, 30.0, score=0.95)],
                "imagePath": "rec_000000002.png",
                "imageData": None,
            }
        ),
        encoding="utf-8",
    )
    assert engine_cli.main(["track-qc", str(clean_dir)]) == 0
    capsys.readouterr()
    assert engine_cli.main(["track-qc", str(clean_dir), "--strict"]) == 2
    capsys.readouterr()


def test_repair_tracks_cli_temporal_dry_run_and_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session_dir = tmp_path / "temporal_swap"
    session_dir.mkdir()

    # Frames 0..1: mouse_1 at left (20, 50), mouse_2 at right (120, 50)
    # Frames 2..3: labels accidentally swapped while animals stayed in place!
    _write_labelme_frame(
        session_dir,
        0,
        [_poly_shape("mouse_1", 20.0, 50.0), _poly_shape("mouse_2", 120.0, 50.0)],
    )
    _write_labelme_frame(
        session_dir,
        1,
        [_poly_shape("mouse_1", 22.0, 50.0), _poly_shape("mouse_2", 118.0, 50.0)],
    )
    frame2_path = _write_labelme_frame(
        session_dir,
        2,
        [_poly_shape("mouse_2", 24.0, 50.0), _poly_shape("mouse_1", 116.0, 50.0)],
    )
    _write_labelme_frame(
        session_dir,
        3,
        [_poly_shape("mouse_2", 26.0, 50.0), _poly_shape("mouse_1", 114.0, 50.0)],
    )
    before_frame2 = frame2_path.read_text(encoding="utf-8")

    # 1. Default dry-run MUST NOT mutate frame JSON files.
    rc_dry = engine_cli.main(
        [
            "repair-tracks",
            str(session_dir),
            "--expected-instances",
            "2",
            "--json",
        ]
    )
    assert rc_dry == 0
    dry_summary = json.loads(capsys.readouterr().out)
    assert dry_summary["dry_run"] is True
    assert dry_summary["proposed_correction_count"] >= 2
    assert dry_summary["updated_files"] == 0
    assert frame2_path.read_text(encoding="utf-8") == before_frame2

    # 2. Explicit --apply rewrites the swapped labels and records provenance.
    summary_out = tmp_path / "repair_summary.json"
    report_out = tmp_path / "temporal_audit.json"
    rc_apply = engine_cli.main(
        [
            "repair-tracks",
            str(session_dir),
            "--expected-instances",
            "2",
            "--apply",
            "--report-out",
            str(report_out),
            "--json-out",
            str(summary_out),
        ]
    )
    assert rc_apply == 0
    capsys.readouterr()

    repaired_f2 = json.loads(frame2_path.read_text(encoding="utf-8"))
    by_label = {
        s["label"]: sum(pt[0] for pt in s["points"]) / len(s["points"])
        for s in repaired_f2["shapes"]
    }
    assert by_label["mouse_1"] == pytest.approx(24.0)
    assert by_label["mouse_2"] == pytest.approx(116.0)

    audit_payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert audit_payload["provenance"]["schema"] == "track_repair/v1"
    assert audit_payload["provenance"]["apply_changes"] is True
    assert summary_out.exists()


def test_repair_tracks_cli_governor_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session_dir = tmp_path / "gov_session"
    session_dir.mkdir()

    _write_labelme_frame(
        session_dir,
        0,
        [
            _poly_shape("alpha", 15.0, 50.0, w=10.0, h=10.0, track_id="1"),
            _poly_shape("beta", 50.0, 50.0, w=10.0, h=10.0, track_id="2"),
        ],
    )
    _write_labelme_frame(
        session_dir,
        1,
        [
            _poly_shape("beta", 85.0, 50.0, w=10.0, h=10.0, track_id="2"),
            _poly_shape("alpha", 15.0, 50.0, w=10.0, h=10.0, track_id="1"),
        ],
    )

    zones_path = tmp_path / "zones.json"
    zones_path.write_text(
        json.dumps(
            {
                "shapes": [
                    build_zone_shape(
                        "left_zone",
                        [[0, 0], [30, 100]],
                        shape_type="rectangle",
                        zone_kind="chamber",
                    ),
                    build_zone_shape(
                        "right_zone",
                        [[70, 0], [100, 100]],
                        shape_type="rectangle",
                        zone_kind="chamber",
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "metric_aliases": {
                    "in_left": "zone.inside.left_zone",
                    "in_right": "zone.inside.right_zone",
                },
                "rules": [
                    {
                        "name": "alpha_in_right",
                        "assign_label": "alpha",
                        "conditions": [
                            {"metric": "in_right", "op": "eq", "value": True}
                        ],
                    },
                    {
                        "name": "beta_in_left",
                        "assign_label": "beta",
                        "conditions": [
                            {"metric": "in_left", "op": "eq", "value": True}
                        ],
                    },
                ],
                "canonical_track_ids": {"alpha": "1", "beta": "2"},
            }
        ),
        encoding="utf-8",
    )

    # Missing --policy in governor mode must fail cleanly with exit code 2.
    assert (
        engine_cli.main(["repair-tracks", str(session_dir), "--mode", "governor"]) == 2
    )
    capsys.readouterr()

    rc = engine_cli.main(
        [
            "repair-tracks",
            str(session_dir),
            "--mode",
            "governor",
            "--policy",
            str(policy_path),
            "--zones",
            str(zones_path),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "governor"
    assert payload["dry_run"] is True
    assert payload["proposed_correction_count"] == 2
