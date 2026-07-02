from __future__ import annotations

import json
import math
from pathlib import Path

from annolid.postprocessing.identity_governor import run_identity_governor
from annolid.postprocessing.temporal_identity_repair import (
    run_temporal_identity_repair,
)
from annolid.postprocessing.zone_schema import build_zone_shape


def _rect(cx: float, cy: float, size: float = 10.0) -> list[list[float]]:
    half = size / 2.0
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]


def _oriented_rect(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_degrees: float,
) -> list[list[float]]:
    radians = math.radians(angle_degrees)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    corners = [
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    ]
    return [
        [cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a] for x, y in corners
    ]


def _shape(label: str, track_id: int, cx: float, cy: float) -> dict:
    track_token = str(track_id)
    return {
        "label": label,
        "instance_label": label,
        "shape_type": "polygon",
        "points": _rect(cx, cy),
        "track_id": track_token,
        "tracking_id": track_token,
        "instance_id": track_token,
        "group_id": track_token,
        "flags": {
            "instance_label": label,
            "track_id": track_token,
            "tracking_id": track_token,
            "instance_id": track_token,
            "group_id": track_token,
        },
    }


def _write_frame(
    root: Path,
    frame_number: int,
    left_label: str,
    left_track: int,
    left_xy: tuple[float, float],
    right_label: str,
    right_track: int,
    right_xy: tuple[float, float],
) -> None:
    payload = {
        "version": "5.0.1",
        "shapes": [
            _shape(left_label, left_track, left_xy[0], left_xy[1]),
            _shape(right_label, right_track, right_xy[0], right_xy[1]),
        ],
        "imagePath": "",
        "imageData": None,
    }
    path = root / f"session_{frame_number:09d}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _make_test_session(tmp_path: Path) -> tuple[Path, Path]:
    annotation_dir = tmp_path / "session"
    annotation_dir.mkdir()

    # Frames 0-1: canonical identity assignment (track 1=alpha, track 2=beta).
    _write_frame(annotation_dir, 0, "alpha", 1, (15, 50), "beta", 2, (50, 50))
    _write_frame(annotation_dir, 1, "alpha", 1, (18, 50), "beta", 2, (52, 50))
    # Frames 2-3: wrestling/ambiguity region with swapped labels and IDs.
    _write_frame(annotation_dir, 2, "beta", 2, (45, 50), "alpha", 1, (47, 50))
    _write_frame(annotation_dir, 3, "beta", 2, (46, 50), "alpha", 1, (48, 50))
    # Frames 4-5: animals separate; geometric evidence is decisive.
    _write_frame(annotation_dir, 4, "beta", 2, (85, 50), "alpha", 1, (15, 50))
    _write_frame(annotation_dir, 5, "beta", 2, (88, 50), "alpha", 1, (12, 50))

    zone_path = tmp_path / "zones.json"
    zone_payload = {
        "shapes": [
            build_zone_shape(
                "left_zone",
                [[0, 0], [30, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                description="left lane",
            ),
            build_zone_shape(
                "right_zone",
                [[70, 0], [100, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                description="right lane",
            ),
        ]
    }
    zone_path.write_text(
        json.dumps(zone_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return annotation_dir, zone_path


def _policy() -> dict:
    return {
        "metric_aliases": {
            "in_left": "zone.inside.left_zone",
            "in_right": "zone.inside.right_zone",
            "dist_to_track_1": "distance.to_track.1",
            "dist_to_track_2": "distance.to_track.2",
            "nearest": "distance.nearest",
            "shape_area": "area",
        },
        "rules": [
            {
                "name": "alpha_when_right_far_from_track1",
                "assign_label": "alpha",
                "conditions": [
                    {"metric": "in_right", "op": "eq", "value": True},
                    {"metric": "dist_to_track_1", "op": "gte", "value": 50.0},
                    {"metric": "shape_area", "op": "gte", "value": 80.0},
                ],
            },
            {
                "name": "beta_when_left_far_from_track2",
                "assign_label": "beta",
                "conditions": [
                    {"metric": "in_left", "op": "eq", "value": True},
                    {"metric": "dist_to_track_2", "op": "gte", "value": 50.0},
                    {"metric": "shape_area", "op": "gte", "value": 80.0},
                ],
            },
        ],
        "ambiguity_conditions": [
            {"metric": "nearest", "op": "lte", "value": 5.0},
        ],
        "canonical_track_ids": {"alpha": "1", "beta": "2"},
    }


def _frame_shapes(annotation_dir: Path, frame_number: int) -> list[dict]:
    path = annotation_dir / f"session_{frame_number:09d}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["shapes"])


def test_identity_governor_dry_run_reports_swap(tmp_path: Path):
    annotation_dir, zone_path = _make_test_session(tmp_path)
    result = run_identity_governor(
        annotation_dir=annotation_dir,
        policy=_policy(),
        zone_file=zone_path,
        apply_changes=False,
    )

    assert result.dry_run is True
    assert result.updated_files == 0
    assert result.updated_shapes == 0
    assert len(result.proposed_corrections) == 2
    assert result.report_path.exists()

    frame2 = _frame_shapes(annotation_dir, 2)
    labels = sorted(shape["instance_label"] for shape in frame2)
    assert labels == ["alpha", "beta"]
    # Dry-run must not rewrite swapped frame content.
    assert frame2[0]["track_id"] == "2"
    assert frame2[1]["track_id"] == "1"


def test_identity_governor_apply_repairs_labels_and_ids(tmp_path: Path):
    annotation_dir, zone_path = _make_test_session(tmp_path)
    result = run_identity_governor(
        annotation_dir=annotation_dir,
        policy=_policy(),
        zone_file=zone_path,
        apply_changes=True,
    )

    assert result.dry_run is False
    assert result.updated_files >= 4
    assert result.updated_shapes >= 8

    for frame_number in (2, 3, 4, 5):
        shapes = _frame_shapes(annotation_dir, frame_number)
        by_track = {str(shape["track_id"]): shape for shape in shapes}
        assert set(by_track.keys()) == {"1", "2"}
        assert by_track["1"]["instance_label"] == "alpha"
        assert by_track["2"]["instance_label"] == "beta"
        assert str(by_track["1"]["group_id"]) == "1"
        assert str(by_track["2"]["group_id"]) == "2"
        assert by_track["1"]["flags"]["instance_label"] == "alpha"
        assert by_track["2"]["flags"]["instance_label"] == "beta"


def test_identity_governor_supports_aggregate_zone_metrics(tmp_path: Path):
    annotation_dir = tmp_path / "session"
    annotation_dir.mkdir()
    for frame_number in (0, 1):
        payload = {
            "version": "5.0.1",
            "shapes": [
                _shape("beta", 1, 10 + frame_number, 50),
                _shape("alpha", 2, 80, 50),
            ],
            "imagePath": "",
            "imageData": None,
        }
        (annotation_dir / f"session_{frame_number:09d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    zone_path = tmp_path / "zones.json"
    zone_payload = {
        "shapes": [
            build_zone_shape(
                "left_stim_chamber",
                [[0, 0], [30, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                occupant_role="stim",
                access_state="open",
            ),
            build_zone_shape(
                "connector_tube",
                [[40, 0], [60, 100]],
                shape_type="rectangle",
                zone_kind="connector_tube",
                occupant_role="neutral",
                access_state="open",
                extra_flags={"neutral_zone": True},
            ),
        ]
    }
    zone_path.write_text(
        json.dumps(zone_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    policy = {
        "metric_aliases": {
            "in_stim_chamber": "zone.inside.stim_chamber",
            "in_neutral_transit": "zone.inside.neutral_transit",
        },
        "rules": [
            {
                "name": "alpha_when_in_stim_chamber",
                "assign_label": "alpha",
                "conditions": [
                    {"metric": "in_stim_chamber", "op": "eq", "value": True},
                    {"metric": "in_neutral_transit", "op": "eq", "value": False},
                ],
                "apply_to_track_ids": ["1"],
            }
        ],
        "interesting_track_ids": ["1"],
    }

    result = run_identity_governor(
        annotation_dir=annotation_dir,
        policy=policy,
        zone_file=zone_path,
        apply_changes=False,
    )

    assert len(result.proposed_corrections) == 1
    correction = result.proposed_corrections[0]
    assert correction.track_id == "1"
    assert correction.corrected_label == "alpha"


def _label_only_shape(
    label: str,
    cx: float,
    cy: float,
    *,
    points: list[list[float]] | None = None,
    note: str = "",
) -> dict:
    payload = {
        "label": label,
        "shape_type": "polygon",
        "points": points or _rect(cx, cy),
        "description": (
            f"motion_index: 0.0; note: {note}" if note else "motion_index: 0.0"
        ),
        "annotation_source": "cutie_vos",
    }
    if note:
        payload["other_data"] = {
            "note": note,
            "annotation_source": "cutie_vos",
        }
    return {
        **payload,
    }


def _write_label_only_frame(
    root: Path,
    frame_number: int,
    left_label: str,
    left_xy: tuple[float, float],
    right_label: str,
    right_xy: tuple[float, float],
) -> None:
    payload = {
        "version": "5.0.1",
        "shapes": [
            _label_only_shape(left_label, left_xy[0], left_xy[1]),
            _label_only_shape(right_label, right_xy[0], right_xy[1]),
        ],
        "imagePath": "",
        "imageData": None,
    }
    (root / f"session_{frame_number:09d}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_label_only_shapes_frame(
    root: Path,
    frame_number: int,
    shapes: list[dict],
) -> None:
    payload = {
        "version": "5.0.1",
        "shapes": shapes,
        "imagePath": "",
        "imageData": None,
    }
    (root / f"session_{frame_number:09d}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_temporal_identity_repair_corrects_label_only_cutie_switch(
    tmp_path: Path,
) -> None:
    annotation_dir = tmp_path / "cutie"
    annotation_dir.mkdir()
    _write_label_only_frame(
        annotation_dir,
        0,
        "mouse_1",
        (10, 50),
        "mouse_2",
        (90, 50),
    )
    _write_label_only_frame(
        annotation_dir,
        1,
        "mouse_1",
        (20, 50),
        "mouse_2",
        (80, 50),
    )
    _write_label_only_frame(
        annotation_dir,
        2,
        "mouse_2",
        (30, 50),
        "mouse_1",
        (70, 50),
    )

    dry_run = run_temporal_identity_repair(
        annotation_dir=annotation_dir,
        expected_instance_count=2,
        max_match_distance=35,
        apply_changes=False,
    )

    assert dry_run.dry_run is True
    assert dry_run.updated_files == 0
    assert dry_run.updated_shapes == 0
    assert len(dry_run.proposed_corrections) == 2
    assert dry_run.report_path.exists()
    assert [shape["label"] for shape in _frame_shapes(annotation_dir, 2)] == [
        "mouse_2",
        "mouse_1",
    ]

    applied = run_temporal_identity_repair(
        annotation_dir=annotation_dir,
        expected_instance_count=2,
        max_match_distance=35,
        apply_changes=True,
    )

    assert applied.dry_run is False
    assert applied.updated_files == 1
    assert applied.updated_shapes == 2
    repaired = _frame_shapes(annotation_dir, 2)
    assert [shape["label"] for shape in repaired] == ["mouse_1", "mouse_2"]
    assert repaired[0]["flags"]["annolid_correction"] == (
        "temporal_identity_switch_corrected"
    )
    assert repaired[1]["flags"]["track_id"] == "mouse_2"


def test_temporal_identity_repair_handles_cutie_recovery_duplicate_ids(
    tmp_path: Path,
) -> None:
    annotation_dir = tmp_path / "cutie_recovery"
    annotation_dir.mkdir()
    _write_label_only_frame(
        annotation_dir,
        0,
        "mouse_1",
        (10, 50),
        "mouse_2",
        (90, 50),
    )
    _write_label_only_frame(
        annotation_dir,
        1,
        "mouse_1",
        (20, 50),
        "mouse_2",
        (80, 50),
    )
    _write_label_only_shapes_frame(
        annotation_dir,
        2,
        [
            _label_only_shape("mouse_2", 70, 50),
        ],
    )
    _write_label_only_shapes_frame(
        annotation_dir,
        3,
        [
            _label_only_shape(
                "mouse_2",
                40,
                50,
                note="recovered_from_nearest_previous_complete_frame",
            ),
            _label_only_shape("mouse_2", 60, 50),
        ],
    )

    result = run_temporal_identity_repair(
        annotation_dir=annotation_dir,
        expected_instance_count=2,
        max_match_distance=35,
        apply_changes=True,
    )

    assert result.updated_files == 2
    assert result.updated_shapes == 2
    frame2 = _frame_shapes(annotation_dir, 2)
    assert sorted(shape["label"] for shape in frame2) == ["mouse_1", "mouse_2"]
    filled = [shape for shape in frame2 if shape["label"] == "mouse_1"][0]
    assert filled["flags"]["annolid_correction"] == "occlusion_gap_carried"
    frame3 = _frame_shapes(annotation_dir, 3)
    assert [shape["label"] for shape in frame3] == ["mouse_1", "mouse_2"]
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["quality_event_counts"]["missing_id_before_repair"] == 2
    assert report["quality_event_counts"]["duplicate_id_before_repair"] == 1
    assert report["quality_event_counts"]["cutie_recovery_note"] == 1
    assert report["problematic_prediction_frames"] == 2


def _write_oriented_label_frame(
    root: Path,
    frame_number: int,
    first_label: str,
    first_xy: tuple[float, float],
    first_angle: float,
    second_label: str,
    second_xy: tuple[float, float],
    second_angle: float,
) -> None:
    payload = {
        "version": "5.0.1",
        "shapes": [
            _label_only_shape(
                first_label,
                first_xy[0],
                first_xy[1],
                points=_oriented_rect(
                    first_xy[0],
                    first_xy[1],
                    width=8.0,
                    height=22.0,
                    angle_degrees=first_angle,
                ),
            ),
            _label_only_shape(
                second_label,
                second_xy[0],
                second_xy[1],
                points=_oriented_rect(
                    second_xy[0],
                    second_xy[1],
                    width=8.0,
                    height=22.0,
                    angle_degrees=second_angle,
                ),
            ),
        ],
        "imagePath": "",
        "imageData": None,
    }
    (root / f"session_{frame_number:09d}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_temporal_identity_repair_uses_velocity_during_crossing(
    tmp_path: Path,
) -> None:
    annotation_dir = tmp_path / "cutie_crossing"
    annotation_dir.mkdir()
    _write_oriented_label_frame(
        annotation_dir,
        0,
        "mouse_1",
        (10, 50),
        90,
        "mouse_2",
        (90, 50),
        0,
    )
    _write_oriented_label_frame(
        annotation_dir,
        1,
        "mouse_1",
        (40, 50),
        90,
        "mouse_2",
        (60, 50),
        0,
    )
    # Nearest-centroid matching would keep this swapped frame unchanged:
    # x=70 is closer to mouse_2's last position, but mouse_1's velocity predicts it.
    _write_oriented_label_frame(
        annotation_dir,
        2,
        "mouse_2",
        (70, 50),
        90,
        "mouse_1",
        (30, 50),
        0,
    )

    result = run_temporal_identity_repair(
        annotation_dir=annotation_dir,
        expected_instance_count=2,
        max_match_distance=45,
        apply_changes=True,
    )

    assert result.updated_files == 1
    assert result.updated_shapes == 2
    repaired = _frame_shapes(annotation_dir, 2)
    assert [shape["label"] for shape in repaired] == ["mouse_1", "mouse_2"]
    assert repaired[0]["flags"]["track_id"] == "mouse_1"
    assert repaired[1]["flags"]["track_id"] == "mouse_2"
