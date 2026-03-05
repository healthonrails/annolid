from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from annolid.annotation import labelme2coco


def _write_labelme_sample(root: Path, stem: str, offset: int) -> None:
    image_path = root / f"{stem}.png"
    Image.new("RGB", (64, 48), color=(0, 0, 0)).save(image_path)
    payload = {
        "version": "5.5.0",
        "imagePath": image_path.name,
        "imageHeight": 48,
        "imageWidth": 64,
        "shapes": [
            {
                "label": "mouse",
                "shape_type": "polygon",
                "points": [
                    [10 + offset, 10],
                    [26 + offset, 10],
                    [26 + offset, 26],
                    [10 + offset, 26],
                ],
                "group_id": 1,
            }
        ],
    }
    (root / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_labelme_pose_sample(root: Path, stem: str, offset: int) -> None:
    image_path = root / f"{stem}.png"
    Image.new("RGB", (96, 80), color=(0, 0, 0)).save(image_path)
    payload = {
        "version": "5.5.0",
        "imagePath": image_path.name,
        "imageHeight": 80,
        "imageWidth": 96,
        "shapes": [
            {
                "label": "superanimal",
                "shape_type": "rectangle",
                "points": [[10 + offset, 8], [60 + offset, 48]],
                "group_id": 7,
            },
            {
                "label": "nose",
                "shape_type": "point",
                "points": [[20 + offset, 18]],
                "group_id": 7,
            },
            {
                "label": "tail_base",
                "shape_type": "point",
                "points": [[50 + offset, 40]],
                "group_id": 7,
            },
        ],
    }
    (root / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_labelme2coco_emits_standard_fields(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "labelme"
    out = tmp_path / "coco_out"
    src.mkdir()

    _write_labelme_sample(src, "frame_0001", offset=0)
    _write_labelme_sample(src, "frame_0002", offset=4)

    progress = list(
        labelme2coco.convert(
            str(src),
            str(out),
            labels_file=None,
            train_valid_split=0.5,
        )
    )

    assert progress
    assert progress[-1][0] == 100

    train_json = out / "train" / "annotations.json"
    valid_json = out / "valid" / "annotations.json"
    assert train_json.exists()
    assert valid_json.exists()
    assert (out / "annotations_train.json").exists()
    assert (out / "annotations_valid.json").exists()
    assert (out / "data.yaml").exists()

    train = json.loads(train_json.read_text(encoding="utf-8"))
    valid = json.loads(valid_json.read_text(encoding="utf-8"))

    for payload in (train, valid):
        assert "images" in payload
        assert "annotations" in payload
        assert "categories" in payload
        assert payload["categories"][0]["id"] == 1
        assert payload["categories"][0]["name"] == "mouse"

    assert len(train["images"]) == 1
    assert len(valid["images"]) == 1

    for ann in train["annotations"] + valid["annotations"]:
        assert isinstance(ann["id"], int)
        assert isinstance(ann["image_id"], int)
        assert ann["category_id"] == 1
        assert ann["iscrowd"] == 0
        assert isinstance(ann["segmentation"], list)
        assert ann["segmentation"]
        assert isinstance(ann["bbox"], list)
        assert len(ann["bbox"]) == 4
        assert ann["area"] > 0


def test_labelme2coco_keypoints_mode_emits_pose_schema(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "labelme_pose"
    out = tmp_path / "coco_pose_out"
    src.mkdir()

    _write_labelme_pose_sample(src, "frame_pose_0001", offset=0)
    _write_labelme_pose_sample(src, "frame_pose_0002", offset=6)

    list(
        labelme2coco.convert(
            str(src),
            str(out),
            labels_file=None,
            train_valid_split=0.5,
            output_mode="keypoints",
        )
    )

    train = json.loads((out / "train" / "annotations.json").read_text(encoding="utf-8"))
    valid = json.loads((out / "valid" / "annotations.json").read_text(encoding="utf-8"))

    for payload in (train, valid):
        cats = payload["categories"]
        assert len(cats) == 1
        assert cats[0]["name"] == "superanimal"
        assert cats[0]["keypoints"] == ["nose", "tail_base"]

        anns = payload["annotations"]
        assert len(anns) == 1
        ann = anns[0]
        assert ann["category_id"] == 1
        assert ann["num_keypoints"] == 2
        assert len(ann["keypoints"]) == 6
        # v flags are the 3rd component in each triplet.
        assert ann["keypoints"][2] == 2.0
        assert ann["keypoints"][5] == 2.0
        assert isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4
        assert ann["area"] > 0


def _write_labelme_keypoints_only_sample(root: Path, stem: str) -> None:
    """LabelMe sample with ONLY keypoint shapes (no polygon/rectangle).

    Exercises the keypoints-only fallback path where bbox/area/segmentation
    must be synthesised purely from keypoint coordinates.
    """
    image_path = root / f"{stem}.png"
    Image.new("RGB", (200, 160), color=(0, 0, 0)).save(image_path)
    payload = {
        "version": "5.5.0",
        "imagePath": image_path.name,
        "imageHeight": 160,
        "imageWidth": 200,
        "shapes": [
            {
                "label": "nose",
                "shape_type": "point",
                "points": [[80, 60]],
                "group_id": 1,
            },
            {
                "label": "tail_base",
                "shape_type": "point",
                "points": [[120, 100]],
                "group_id": 1,
            },
        ],
    }
    (root / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_keypoints_only_fallback_area_reasonable(tmp_path: Path) -> None:
    """Area and segmentation must be non-trivial when only keypoints are present.

    Regression for bugs where pad=2.0 produced a ~4x4 px area (<= 16 px^2)
    and segmentation was left as an empty list.
    """
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "labelme_kpts_only"
    out = tmp_path / "coco_kpts_only_out"
    src.mkdir()

    _write_labelme_keypoints_only_sample(src, "frame_kpts_0001")
    _write_labelme_keypoints_only_sample(src, "frame_kpts_0002")

    list(
        labelme2coco.convert(
            str(src),
            str(out),
            labels_file=None,
            train_valid_split=0.5,
            output_mode="keypoints",
        )
    )

    train = json.loads((out / "train" / "annotations.json").read_text(encoding="utf-8"))
    valid = json.loads((out / "valid" / "annotations.json").read_text(encoding="utf-8"))

    for payload in (train, valid):
        anns = payload["annotations"]
        assert len(anns) == 1, "Expected one annotation per image"
        ann = anns[0]

        # Area must be much larger than the old 4x4=16 px^2 fallback.
        assert ann["area"] > 100, (
            f"Fallback area is unrealistically small: {ann['area']}"
        )

        # Both bbox dimensions must be at least 10 pixels.
        _x0, _y0, bw, bh = ann["bbox"]
        assert bw >= 10, f"Fallback bbox width too narrow: {bw}"
        assert bh >= 10, f"Fallback bbox height too short: {bh}"

        # Segmentation must NOT be an empty list.
        assert ann["segmentation"], (
            "segmentation must be non-empty for keypoints-only groups"
        )
        seg = ann["segmentation"][0]
        assert len(seg) >= 6, (
            "Segmentation polygon must have at least 3 vertices (6 coords)"
        )

        # Keypoints sanity.
        assert ann["num_keypoints"] == 2
        assert len(ann["keypoints"]) == 6


# ---------------------------------------------------------------------------
# Helpers for detailed coordinate tests
# ---------------------------------------------------------------------------


def _make_image(root: Path, stem: str, w: int = 200, h: int = 160) -> None:
    Image.new("RGB", (w, h), color=(10, 10, 10)).save(root / f"{stem}.png")


def _write_json(root: Path, stem: str, payload: dict) -> None:
    (root / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def _run_keypoints_convert(src: Path, out: Path) -> tuple[dict, dict]:
    """Run keypoints conversion and return (train_data, valid_data)."""
    list(
        labelme2coco.convert(
            str(src),
            str(out),
            labels_file=None,
            train_valid_split=0.5,
            output_mode="keypoints",
        )
    )
    train = json.loads((out / "train" / "annotations.json").read_text())
    valid = json.loads((out / "valid" / "annotations.json").read_text())
    return train, valid


# ---------------------------------------------------------------------------
# Test 1: Exact coordinate placement in the keypoints array
# ---------------------------------------------------------------------------


def test_keypoints_xy_values_placed_at_correct_index(tmp_path: Path) -> None:
    """x,y of each point label must land at the correct triplet index.

    COCO stores keypoints as a flat list:
        [kp0_x, kp0_y, kp0_v, kp1_x, kp1_y, kp1_v, ...]
    where index is determined by the sorted keypoint_names list.
    For ["nose", "tail_base"] (alphabetical), nose → index 0, tail_base → index 1.
    """
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    # Deterministic, easy-to-verify coordinates.
    nose_xy = [30.0, 20.0]
    tail_xy = [90.0, 70.0]

    for stem in ("frameA", "frameB"):
        _make_image(src, stem)
        _write_json(
            src,
            stem,
            {
                "version": "5.5.0",
                "imagePath": f"{stem}.png",
                "imageHeight": 160,
                "imageWidth": 200,
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "rectangle",
                        "points": [[10, 10], [150, 130]],
                        "group_id": 1,
                    },
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "points": [nose_xy],
                        "group_id": 1,
                    },
                    {
                        "label": "tail_base",
                        "shape_type": "point",
                        "points": [tail_xy],
                        "group_id": 1,
                    },
                ],
            },
        )

    train, valid = _run_keypoints_convert(src, out)

    for ds in (train, valid):
        cats = ds["categories"]
        # Category must expose the keypoints list in sorted order.
        assert cats[0]["keypoints"] == ["nose", "tail_base"], cats[0]["keypoints"]

        for ann in ds["annotations"]:
            kp = ann["keypoints"]
            assert len(kp) == 6, f"Expected 6 values, got {len(kp)}"
            # nose is index 0 → triplet at positions [0,1,2]
            assert kp[0] == nose_xy[0], f"nose x wrong: {kp[0]}"
            assert kp[1] == nose_xy[1], f"nose y wrong: {kp[1]}"
            assert kp[2] == 2.0, f"nose visibility wrong: {kp[2]}"
            # tail_base is index 1 → triplet at positions [3,4,5]
            assert kp[3] == tail_xy[0], f"tail_base x wrong: {kp[3]}"
            assert kp[4] == tail_xy[1], f"tail_base y wrong: {kp[4]}"
            assert kp[5] == 2.0, f"tail_base visibility wrong: {kp[5]}"


# ---------------------------------------------------------------------------
# Test 2: Multiple instances in one image (different group_ids)
# ---------------------------------------------------------------------------


def test_keypoints_multi_instance_per_image(tmp_path: Path) -> None:
    """Two instances (group_id 1 and 2) in one image → two separate annotations."""
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    for stem in ("frameA", "frameB"):
        _make_image(src, stem)
        _write_json(
            src,
            stem,
            {
                "version": "5.5.0",
                "imagePath": f"{stem}.png",
                "imageHeight": 160,
                "imageWidth": 200,
                "shapes": [
                    # Instance 1
                    {
                        "label": "animal",
                        "shape_type": "rectangle",
                        "points": [[5, 5], [80, 80]],
                        "group_id": 1,
                    },
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "points": [[20.0, 15.0]],
                        "group_id": 1,
                    },
                    {
                        "label": "tail_base",
                        "shape_type": "point",
                        "points": [[60.0, 60.0]],
                        "group_id": 1,
                    },
                    # Instance 2
                    {
                        "label": "animal",
                        "shape_type": "rectangle",
                        "points": [[100, 5], [190, 80]],
                        "group_id": 2,
                    },
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "points": [[120.0, 15.0]],
                        "group_id": 2,
                    },
                    {
                        "label": "tail_base",
                        "shape_type": "point",
                        "points": [[170.0, 60.0]],
                        "group_id": 2,
                    },
                ],
            },
        )

    train, valid = _run_keypoints_convert(src, out)

    for ds in (train, valid):
        anns = ds["annotations"]
        assert len(anns) == 2, f"Expected 2 annotations (2 instances), got {len(anns)}"
        for ann in anns:
            assert ann["num_keypoints"] == 2
            assert len(ann["keypoints"]) == 6
            # All visible.
            assert ann["keypoints"][2] == 2.0
            assert ann["keypoints"][5] == 2.0
            assert ann["area"] > 0

        # The two annotations must have different keypoint coordinates.
        kp_sets = {(ann["keypoints"][0], ann["keypoints"][3]) for ann in anns}
        assert len(kp_sets) == 2, "Two instances must have distinct nose x-coordinates"


# ---------------------------------------------------------------------------
# Test 3: Unlabeled / missing keypoints remain zero with visibility=0
# ---------------------------------------------------------------------------


def test_keypoints_missing_label_stays_zero_invisible(tmp_path: Path) -> None:
    """If only one of two keypoints is annotated, the missing one must be [0,0,0]."""
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    # frameA: only "nose" is labeled — this is the annotation under test.
    _make_image(src, "frameA")
    _write_json(
        src,
        "frameA",
        {
            "version": "5.5.0",
            "imagePath": "frameA.png",
            "imageHeight": 160,
            "imageWidth": 200,
            "shapes": [
                {
                    "label": "animal",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [100, 100]],
                    "group_id": 1,
                },
                {
                    "label": "nose",
                    "shape_type": "point",
                    "points": [[50.0, 40.0]],
                    "group_id": 1,
                },
            ],
        },
    )

    # frameB: has both "nose" AND "tail_base" so the global schema includes both.
    # Without this file the converter would never discover "tail_base" and the
    # category keypoints list would only contain ["nose"].
    _make_image(src, "frameB")
    _write_json(
        src,
        "frameB",
        {
            "version": "5.5.0",
            "imagePath": "frameB.png",
            "imageHeight": 160,
            "imageWidth": 200,
            "shapes": [
                {
                    "label": "animal",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [100, 100]],
                    "group_id": 1,
                },
                {
                    "label": "nose",
                    "shape_type": "point",
                    "points": [[55.0, 45.0]],
                    "group_id": 1,
                },
                {
                    "label": "tail_base",
                    "shape_type": "point",
                    "points": [[80.0, 70.0]],
                    "group_id": 1,
                },
            ],
        },
    )

    train, valid = _run_keypoints_convert(src, out)

    # Both splits share the same category schema (built from all files).
    for ds in (train, valid):
        cats = ds["categories"]
        kp_names = cats[0]["keypoints"]  # sorted: ["nose", "tail_base"]
        assert "nose" in kp_names and "tail_base" in kp_names, kp_names

    # Find the annotation from frameA (the nose-only instance) across both splits.
    all_annotations_by_image: dict = {}
    for ds_label, ds in (("train", train), ("valid", valid)):
        img_id_to_name = {img["id"]: img["file_name"] for img in ds["images"]}
        for ann in ds["annotations"]:
            fname = img_id_to_name.get(ann["image_id"], "")
            if "frameA" in fname:
                all_annotations_by_image[ds_label] = (
                    ann,
                    ds["categories"][0]["keypoints"],
                )

    assert all_annotations_by_image, "frameA annotation not found in either split"

    for _split, (ann, kp_names) in all_annotations_by_image.items():
        nose_idx = kp_names.index("nose")
        tail_idx = kp_names.index("tail_base")
        kp = ann["keypoints"]
        assert len(kp) == 6
        # nose should be visible (v=2)
        assert kp[nose_idx * 3 + 2] == 2.0, f"nose visibility: {kp[nose_idx * 3 + 2]}"
        # tail_base was not annotated → must stay [0, 0, 0]
        assert kp[tail_idx * 3] == 0.0
        assert kp[tail_idx * 3 + 1] == 0.0
        assert kp[tail_idx * 3 + 2] == 0.0
        assert ann["num_keypoints"] == 1  # only one visible


# ---------------------------------------------------------------------------
# Test 4: Single keypoint (degenerate case — one point, no polygon)
# ---------------------------------------------------------------------------


def test_keypoints_single_point_no_polygon_produces_valid_annotation(
    tmp_path: Path,
) -> None:
    """A group with a single point and no instance polygon must still produce
    a well-formed COCO annotation with non-zero area and non-empty segmentation.
    """
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    for stem in ("frameA", "frameB"):
        _make_image(src, stem, w=300, h=200)
        _write_json(
            src,
            stem,
            {
                "version": "5.5.0",
                "imagePath": f"{stem}.png",
                "imageHeight": 200,
                "imageWidth": 300,
                "shapes": [
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "points": [[150.0, 100.0]],
                        "group_id": 1,
                    },
                ],
            },
        )

    train, valid = _run_keypoints_convert(src, out)

    for ds in (train, valid):
        for ann in ds["annotations"]:
            assert ann["area"] > 0
            assert ann["segmentation"], "segmentation must not be empty"
            assert ann["bbox"][2] >= 10 and ann["bbox"][3] >= 10
            assert ann["num_keypoints"] == 1


# ---------------------------------------------------------------------------
# Test 5: Out-of-bounds keypoints are clamped but still marked visible
# ---------------------------------------------------------------------------


def test_keypoints_out_of_bounds_are_clamped_visible(tmp_path: Path) -> None:
    """Keypoints outside the image bounds must be clamped to [0..w-1, 0..h-1]
    and still carry visibility=2 (labeled and visible per COCO convention).
    """
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    W, H = 100, 80

    for stem in ("frameA", "frameB"):
        _make_image(src, stem, w=W, h=H)
        _write_json(
            src,
            stem,
            {
                "version": "5.5.0",
                "imagePath": f"{stem}.png",
                "imageHeight": H,
                "imageWidth": W,
                "shapes": [
                    {
                        "label": "animal",
                        "shape_type": "rectangle",
                        "points": [[0, 0], [99, 79]],
                        "group_id": 1,
                    },
                    # nose at (-5, -5) — outside top-left
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "points": [[-5.0, -5.0]],
                        "group_id": 1,
                    },
                    # tail_base at (200, 200) — outside bottom-right
                    {
                        "label": "tail_base",
                        "shape_type": "point",
                        "points": [[200.0, 200.0]],
                        "group_id": 1,
                    },
                ],
            },
        )

    train, valid = _run_keypoints_convert(src, out)

    for ds in (train, valid):
        kp_names = ds["categories"][0]["keypoints"]
        nose_idx = kp_names.index("nose")
        tail_idx = kp_names.index("tail_base")

        for ann in ds["annotations"]:
            kp = ann["keypoints"]
            # Clamped coordinates must be within image bounds.
            assert 0.0 <= kp[nose_idx * 3] < W
            assert 0.0 <= kp[nose_idx * 3 + 1] < H
            assert 0.0 <= kp[tail_idx * 3] < W
            assert 0.0 <= kp[tail_idx * 3 + 1] < H
            # Must still be marked as visible.
            assert kp[nose_idx * 3 + 2] == 2.0
            assert kp[tail_idx * 3 + 2] == 2.0
