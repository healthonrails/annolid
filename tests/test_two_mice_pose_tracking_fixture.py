import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageStat

from annolid.segmentation.dino_kpseg.data import (
    load_coco_pose_spec,
    load_yolo_pose_spec,
    materialize_coco_pose_as_yolo,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "two_mice_pose_tracking_tiny"
EXPECTED_KEYPOINTS = [
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "spine_1",
    "spine_2",
    "tail_base",
    "left_forepaw",
    "right_forepaw",
    "left_hindpaw",
    "right_hindpaw",
    "tail_mid",
    "tail_tip",
]


def _load_annotation(name: str) -> dict:
    path = FIXTURE_ROOT / "annotations" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_mice_pose_fixture_has_valid_temporal_coco_contract():
    sequence = _load_annotation("sequence.json")

    assert sequence["categories"][0]["keypoints"] == EXPECTED_KEYPOINTS
    assert len(sequence["categories"][0]["skeleton"]) == 13
    assert sequence["videos"] == [
        {
            "id": 1,
            "file_name": "two_mice_synthetic",
            "fps": 5,
            "num_frames": 12,
            "width": 640,
            "height": 480,
        }
    ]
    assert [image["frame_id"] for image in sequence["images"]] == list(range(12))
    assert len(sequence["annotations"]) == 24

    annotations_by_image = defaultdict(list)
    for annotation in sequence["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    for image in sequence["images"]:
        assert image["video_id"] == 1
        assert image["subject_overlap"] is False
        assert image["minimum_subject_clearance"] >= -0.01
        assert image["tail_overlap"] is False
        assert image["minimum_tail_clearance"] >= -0.001
        assert image["maximum_foot_slip"] <= 0.075
        assert image["maximum_paw_ground_error"] <= 0.1
        frame_annotations = annotations_by_image[image["id"]]
        assert {annotation["track_id"] for annotation in frame_annotations} == {1, 2}
        for annotation in frame_annotations:
            assert len(annotation["keypoints"]) == len(EXPECTED_KEYPOINTS) * 3
            points = list(zip(*[iter(annotation["keypoints"])] * 3))
            assert annotation["num_keypoints"] == sum(v > 0 for _, _, v in points)
            for x, y, visibility in points:
                assert visibility in {0, 2}
                if visibility > 0:
                    assert 0 <= x < image["width"]
                    assert 0 <= y < image["height"]
                else:
                    assert (x, y) == (0, 0)
            x, y, width, height = annotation["bbox"]
            assert 0 <= x <= image["width"]
            assert 0 <= y <= image["height"]
            assert width > 0 and height > 0
            assert x + width <= image["width"] + 1e-6
            assert y + height <= image["height"] + 1e-6
            assert math.isclose(annotation["area"], width * height, abs_tol=1e-3)


def test_two_mice_pose_fixture_images_are_distinct_and_tracks_move():
    sequence = _load_annotation("sequence.json")
    digests = set()
    for image_record in sequence["images"]:
        image_path = FIXTURE_ROOT / image_record["file_name"]
        assert image_path.is_file()
        digests.add(hashlib.sha256(image_path.read_bytes()).hexdigest())
        with Image.open(image_path) as image:
            assert image.size == (image_record["width"], image_record["height"])
            assert image.format == "JPEG"
            assert ImageStat.Stat(image.convert("L")).var[0] > 25
    assert len(digests) == len(sequence["images"])

    tracks = defaultdict(list)
    for annotation in sequence["annotations"]:
        tracks[annotation["track_id"]].append(annotation)
    assert set(tracks) == {1, 2}
    for track_annotations in tracks.values():
        track_annotations.sort(key=lambda annotation: annotation["image_id"])
        first_nose = track_annotations[0]["keypoints"][:2]
        last_nose = track_annotations[-1]["keypoints"][:2]
        assert math.dist(first_nose, last_nose) > 20


def test_two_mice_pose_fixture_train_val_split_matches_sequence():
    sequence = _load_annotation("sequence.json")
    train = _load_annotation("train.json")
    val = _load_annotation("val.json")

    assert [image["frame_id"] for image in train["images"]] == list(range(8))
    assert [image["frame_id"] for image in val["images"]] == list(range(8, 12))
    split_image_ids = {image["id"] for image in train["images"] + val["images"]}
    assert split_image_ids == {image["id"] for image in sequence["images"]}
    assert len(train["annotations"]) == 16
    assert len(val["annotations"]) == 8


def test_two_mice_pose_fixture_loads_through_annolid_pose_pipeline(tmp_path: Path):
    spec = load_coco_pose_spec(FIXTURE_ROOT / "coco_spec.yaml")

    assert spec.kpt_count == len(EXPECTED_KEYPOINTS)
    assert spec.kpt_dims == 3
    assert spec.keypoint_names == EXPECTED_KEYPOINTS

    yolo_yaml = materialize_coco_pose_as_yolo(
        spec=spec, output_dir=tmp_path / "yolo_pose"
    )
    yolo_spec = load_yolo_pose_spec(yolo_yaml)

    assert len(yolo_spec.train_images) == 8
    assert len(yolo_spec.val_images) == 4
    for label_path in (tmp_path / "yolo_pose" / "labels").rglob("*.txt"):
        assert len(label_path.read_text(encoding="utf-8").splitlines()) == 2
