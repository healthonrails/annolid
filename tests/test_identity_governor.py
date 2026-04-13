from __future__ import annotations

import json
from pathlib import Path

from annolid.postprocessing.identity_governor import run_identity_governor
from annolid.postprocessing.zone_schema import build_zone_shape


def _rect(cx: float, cy: float, size: float = 10.0) -> list[list[float]]:
    half = size / 2.0
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
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
