from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon

from annolid.annotation.keypoint_visibility import KeypointVisibility
from annolid.engine.cli import main as cli_main
from annolid.postprocessing.track_qc import evaluate_track_qc
from annolid.segmentation.body_parts import (
    BodyPartSegmentationConfig,
    InstanceBodyPartSegmenter,
    parse_frame_instances_and_seed_parts,
)
from annolid.utils.body_part_shapes import is_body_part_shape_payload


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "two_mice_body_part_tiny"

COARSE_ID_TO_NAME = {
    1: "head",
    2: "torso",
    3: "limbs",
    4: "tail",
}

FINE_ID_TO_KEYPOINT_FINE_LABEL = {
    1: "head",
    2: "head",
    3: "head",
    4: "head",
    5: "torso",
    6: "torso",
    7: "left_forelimb",
    8: "right_forelimb",
    9: "left_hindlimb",
    10: "right_hindlimb",
    11: "tail",
}


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.count_nonzero(mask_a & mask_b))
    union = int(np.count_nonzero(mask_a | mask_b))
    return float(inter) / float(max(1, union))


def _mask_to_largest_poly_pts(mask: np.ndarray) -> list[list[float]]:
    cnts, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, 0.75, True)
    pts = approx if len(approx) >= 3 else largest
    return [[float(pt[0][0]), float(pt[0][1])] for pt in pts]


def _poly_pts_to_mask(points: list[list[float]], height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    arr = np.rint(np.asarray(points, dtype=np.float64)[:, :2]).astype(np.int32)
    cv2.fillPoly(canvas, [arr], 1)
    return canvas.astype(bool)


def _load_manifest() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


def _load_fixture_frame(
    record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    bgr = cv2.imread(str(FIXTURE_DIR / record["image_file"]), cv2.IMREAD_COLOR)
    assert bgr is not None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    id_bgr = cv2.imread(str(FIXTURE_DIR / record["id_mask_file"]), cv2.IMREAD_UNCHANGED)
    assert id_bgr is not None and id_bgr.ndim == 3
    coarse_map = id_bgr[:, :, 0]
    fine_map = id_bgr[:, :, 1]
    track_map = id_bgr[:, :, 2]
    lm_payload = json.loads(
        (FIXTURE_DIR / record["labelme_file"]).read_text(encoding="utf-8")
    )
    return rgb, track_map, fine_map, coarse_map, lm_payload


def _visible_keypoints_from_labelme(
    lm_payload: dict[str, Any],
) -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for sh in lm_payload.get("shapes", []):
        if sh.get("shape_type") != "point":
            continue
        vis = int(sh.get("kp_visibility", 2))
        flags = sh.get("flags") or {}
        if flags.get("kp_visible") is False or vis < int(KeypointVisibility.VISIBLE):
            continue
        inst = str(sh.get("instance_label") or f"mouse_{sh.get('group_id')}")
        lbl = str(sh.get("display_label") or sh.get("label"))
        pt = sh["points"][0]
        out.setdefault(inst, {})[lbl] = (float(pt[0]), float(pt[1]))
    return out


def test_two_mice_3d_fixture_integrity_and_occlusion_visibility_filtering():
    manifest = _load_manifest()
    assert manifest["schema_version"] == 1
    assert (manifest["width"], manifest["height"]) == (320, 240)
    assert len(manifest["keypoint_names"]) == 13
    assert len(manifest["frames"]) == 8

    # Ensure zero machine-local paths leaked into fixture metadata or LabelMe JSONs.
    manifest_raw = (FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    assert "/Users/" not in manifest_raw and "/home/" not in manifest_raw

    occluded_kp_seen = False
    visible_paw_seen = False

    for rec in manifest["frames"]:
        rgb, track_map, fine_map, coarse_map, lm_payload = _load_fixture_frame(rec)
        assert rgb.shape == (240, 320, 3)
        assert set(np.unique(track_map).tolist()) == {0, 1, 2}
        assert not np.any((track_map == 0) & (coarse_map > 0))
        assert not np.any((track_map == 0) & (fine_map > 0))
        assert np.all(coarse_map[track_map > 0] > 0)
        assert np.all(fine_map[track_map > 0] > 0)

        lm_raw = (FIXTURE_DIR / rec["labelme_file"]).read_text(encoding="utf-8")
        assert "/Users/" not in lm_raw and "/home/" not in lm_raw

        # Verify all per-shape flags remain strictly boolean for Qt LabelMe compatibility.
        for sh in lm_payload["shapes"]:
            for flag_v in (sh.get("flags") or {}).values():
                assert isinstance(flag_v, bool)
            if sh.get("shape_type") == "point":
                if int(sh.get("kp_visibility", 2)) == int(KeypointVisibility.OCCLUDED):
                    occluded_kp_seen = True
                if "paw" in str(sh.get("label", "")) and int(
                    sh.get("kp_visibility", 0)
                ) == int(KeypointVisibility.VISIBLE):
                    visible_paw_seen = True

        # Video processor parser must filter out occluded (kp_visibility == 1) keypoints automatically.
        _pm, _pp, _pg, _sp, parsed_kps = parse_frame_instances_and_seed_parts(
            lm_payload, default_height=240, default_width=320
        )
        expected_vis = _visible_keypoints_from_labelme(lm_payload)
        assert parsed_kps == expected_vis

    assert occluded_kp_seen, "Expected 3D surface occlusion to mark occluded keypoints"
    assert visible_paw_seen, "Expected visible 3D limb/paw keypoints in fixture"


def test_two_mice_keypoint_guided_coarse_and_fine_against_3d_ground_truth():
    manifest = _load_manifest()
    coarse_seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="keypoint_guided",
            feature_backend="fast",
            keypoint_granularity="coarse",
        )
    )

    coarse_ious: dict[str, list[float]] = {v: [] for v in COARSE_ID_TO_NAME.values()}
    polarity_checks = 0

    for rec in manifest["frames"]:
        # Reset temporal state when switching between distinct scenarios in the fixture.
        if rec["index"] in {0, 4, 6}:
            coarse_seg.reset_temporal_state()

        rgb, track_map, _fine_map, coarse_map, lm_payload = _load_fixture_frame(rec)
        H, W = track_map.shape
        parent_masks = {"mouse_1": track_map == 1, "mouse_2": track_map == 2}
        kps = _visible_keypoints_from_labelme(lm_payload)
        preds = coarse_seg.segment_frame(rgb, parent_masks, keypoints_by_instance=kps)

        for inst_lbl, pmask in parent_masks.items():
            inst_preds = preds[inst_lbl]
            pred_masks = {k: p.to_full_mask(H, W) for k, p in inst_preds.items()}

            # 1. Hard geometric invariants: strict containment + zero pixel/polygon overlap.
            accum = np.zeros_like(pmask, dtype=bool)
            polys = [Polygon(p.polygon) for p in inst_preds.values()]
            for p_m in pred_masks.values():
                assert not np.any(p_m & (~pmask))
                assert not np.any(accum & p_m)
                accum |= p_m
            for idx_a, poly_a in enumerate(polys):
                assert poly_a.is_valid
                for poly_b in polys[idx_a + 1 :]:
                    assert poly_a.intersection(poly_b).area < 1e-2

            # 2. Per-part IoU against WebGL 3D mesh ground truth.
            gt_parts = {
                pname: (pmask & (coarse_map == cid))
                for cid, pname in COARSE_ID_TO_NAME.items()
                if int(np.count_nonzero(pmask & (coarse_map == cid))) >= 6
            }
            for pname, gt_m in gt_parts.items():
                pr_m = pred_masks.get(pname, np.zeros_like(gt_m, dtype=bool))
                coarse_ious[pname].append(_compute_iou(pr_m, gt_m))

            # 3. Head-to-tail anterior-posterior polarity alignment.
            if (
                "head" in gt_parts
                and "tail" in gt_parts
                and "head" in pred_masks
                and "tail" in pred_masks
            ):
                gy_h, gx_h = np.nonzero(gt_parts["head"])
                gy_t, gx_t = np.nonzero(gt_parts["tail"])
                py_h, px_h = np.nonzero(pred_masks["head"])
                py_t, px_t = np.nonzero(pred_masks["tail"])
                gt_vec = (
                    float(gx_h.mean() - gx_t.mean()),
                    float(gy_h.mean() - gy_t.mean()),
                )
                pr_vec = (
                    float(px_h.mean() - px_t.mean()),
                    float(py_h.mean() - py_t.mean()),
                )
                if math.hypot(*gt_vec) > 4.0 and math.hypot(*pr_vec) > 4.0:
                    polarity_checks += 1
                    assert gt_vec[0] * pr_vec[0] + gt_vec[1] * pr_vec[1] > 0.0

    assert polarity_checks >= 10
    assert float(np.mean(coarse_ious["torso"])) >= 0.88
    assert float(np.mean(coarse_ious["head"])) >= 0.68
    assert float(np.mean(coarse_ious["tail"])) >= 0.68
    assert float(np.mean(coarse_ious["limbs"])) >= 0.30

    # Fine granularity on oblique 3D rearing frames verifies left/right fore/hind limb segmentation.
    fine_seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="keypoint_guided",
            feature_backend="fast",
            keypoint_granularity="fine",
        )
    )
    fine_limb_ious: list[float] = []
    seen_limb_labels: set[str] = set()
    for rec in manifest["frames"]:
        if rec["scenario"] != "rearing_oblique":
            continue
        rgb, track_map, fine_map, _, lm_payload = _load_fixture_frame(rec)
        H, W = track_map.shape
        parent_masks = {"mouse_1": track_map == 1, "mouse_2": track_map == 2}
        kps = _visible_keypoints_from_labelme(lm_payload)
        f_preds = fine_seg.segment_frame(rgb, parent_masks, keypoints_by_instance=kps)
        for inst_lbl, pmask in parent_masks.items():
            pmap = {
                k: p.to_full_mask(H, W) for k, p in f_preds.get(inst_lbl, {}).items()
            }
            gt_grouped: dict[str, np.ndarray] = {}
            for fid, flabel in FINE_ID_TO_KEYPOINT_FINE_LABEL.items():
                gt_grouped.setdefault(flabel, np.zeros_like(pmask, dtype=bool))
                gt_grouped[flabel] |= pmask & (fine_map == fid)
            for flabel, gt_m in gt_grouped.items():
                if (
                    "limb" in flabel
                    and int(np.count_nonzero(gt_m)) >= 5
                    and flabel in pmap
                ):
                    seen_limb_labels.add(flabel)
                    fine_limb_ious.append(_compute_iou(pmap[flabel], gt_m))

    assert {"right_forelimb", "left_hindlimb", "right_hindlimb"}.issubset(
        seen_limb_labels
    )
    assert float(np.mean(fine_limb_ious)) >= 0.50


def test_two_mice_seeded_propagation_cross_strain_sharing_partial_and_unsupervised():
    manifest = _load_manifest()
    huddle_records = [r for r in manifest["frames"] if r["scenario"] == "social_huddle"]
    assert len(huddle_records) == 4

    rgb0, tr0, _, c0, _ = _load_fixture_frame(huddle_records[0])
    H, W = tr0.shape
    parent0 = {"mouse_1": tr0 == 1, "mouse_2": tr0 == 2}

    # 1. Seed ONLY black C57BL/6 mouse_1 on frame 0; propagate temporally and transfer zero-shot to white BALB/c mouse_2!
    axial_seed_m1 = {
        "mouse_1": {
            "head": parent0["mouse_1"] & (c0 == 1),
            "torso": parent0["mouse_1"] & np.isin(c0, [2, 3]),
            "tail": parent0["mouse_1"] & (c0 == 4),
        }
    }
    seeded_seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="seeded",
            feature_backend="fast",
            share_seed_across_instances=True,
            partition_full_instance=True,
        )
    )
    assert seeded_seg.fit_seed_frame(rgb0, parent0, axial_seed_m1) == 3

    m1_ious: list[float] = []
    m2_cross_ious: list[float] = []
    for rec in huddle_records[1:]:
        rgb_f, tr_f, _, c_f, _ = _load_fixture_frame(rec)
        p_masks = {"mouse_1": tr_f == 1, "mouse_2": tr_f == 2}
        preds = seeded_seg.segment_frame(rgb_f, p_masks)
        for inst_lbl, pmask in p_masks.items():
            pmap = {k: p.to_full_mask(H, W) for k, p in preds[inst_lbl].items()}
            assert set(pmap.keys()) == {"head", "torso", "tail"}
            for pname, gt_m in (
                ("head", pmask & (c_f == 1)),
                ("torso", pmask & np.isin(c_f, [2, 3])),
                ("tail", pmask & (c_f == 4)),
            ):
                if int(np.count_nonzero(gt_m)) < 6:
                    continue
                iou_v = _compute_iou(pmap[pname], gt_m)
                if inst_lbl == "mouse_1":
                    m1_ious.append(iou_v)
                else:
                    m2_cross_ious.append(iou_v)

    assert float(np.mean(m1_ious)) >= 0.72
    assert float(np.mean(m2_cross_ious)) >= 0.58

    # 2. Partial seeding (head only) keeps unseeded torso+tail unassigned without runaway area growth.
    partial_seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(mode="seeded", feature_backend="fast")
    )
    partial_seg.fit_seed_frame(
        rgb0,
        {"mouse_1": parent0["mouse_1"]},
        {"mouse_1": {"head": axial_seed_m1["mouse_1"]["head"]}},
    )
    rgb1, tr1, _, c1, _ = _load_fixture_frame(huddle_records[1])
    p_head = partial_seg.segment_frame(rgb1, {"mouse_1": tr1 == 1})["mouse_1"]
    assert set(p_head.keys()) == {"head"}
    assert 0.06 < p_head["head"].area_ratio < 0.38
    assert (
        _compute_iou(p_head["head"].to_full_mask(H, W), (tr1 == 1) & (c1 == 1)) >= 0.60
    )

    # 3. Unsupervised 3-part mode segments both mice cleanly across all 4 frames with zero seeds or keypoints.
    unsup_seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="unsupervised",
            feature_backend="fast",
            num_parts=3,
            part_names=("head", "torso", "tail"),
        )
    )
    for rec in huddle_records:
        rgb_f, tr_f, _, _, _ = _load_fixture_frame(rec)
        u_preds = unsup_seg.segment_frame(
            rgb_f, {"mouse_1": tr_f == 1, "mouse_2": tr_f == 2}
        )
        assert len(u_preds["mouse_1"]) == 3
        assert len(u_preds["mouse_2"]) == 3


def test_two_mice_huddle_video_merged_parent_recovery_cli_csv_and_track_qc(
    tmp_path: Path, capsys
):
    manifest = _load_manifest()
    huddle_records = [r for r in manifest["frames"] if r["scenario"] == "social_huddle"]
    H, W = int(manifest["height"]), int(manifest["width"])

    video_path = tmp_path / "social_huddle.mp4"
    ann_dir = tmp_path / "social_huddle"
    ann_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (W, H),
    )
    assert writer.isOpened()

    corrupted_frames = {1, 2}
    gt_track_by_frame: dict[int, np.ndarray] = {}

    try:
        for fid, rec in enumerate(huddle_records):
            rgb, track_map, _, _, lm_payload = _load_fixture_frame(rec)
            gt_track_by_frame[fid] = track_map
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

            shapes: list[dict[str, Any]] = []
            if fid in corrupted_frames:
                # Bridge mouse_1 + mouse_2 into ONE bloated mouse_1 polygon and drop mouse_2's parent polygon!
                bloated = ((track_map == 1) | (track_map == 2)).astype(np.uint8) * 255
                y1, x1 = np.nonzero(track_map == 1)
                y2, x2 = np.nonzero(track_map == 2)
                c1 = (int(round(x1.mean())), int(round(y1.mean())))
                c2 = (int(round(x2.mean())), int(round(y2.mean())))
                cv2.line(bloated, c1, c2, 255, thickness=14)
                merged_poly = _mask_to_largest_poly_pts(bloated > 0)
                shapes.append(
                    {
                        "label": "mouse_1",
                        "points": merged_poly,
                        "group_id": 1,
                        "shape_type": "polygon",
                        "flags": {},
                        "visible": True,
                    }
                )
                # Keep keypoints on fid=1 (Trigger 1: keypoints inside merged blob) and drop ALL keypoints on fid=2
                # so BOTH Trigger 1 and Trigger 2 (area surge + motion-tracked missing neighbor) are verified!
                if fid == 1:
                    for sh in lm_payload["shapes"]:
                        if sh.get("shape_type") == "point":
                            shapes.append(sh)
            else:
                for sh in lm_payload["shapes"]:
                    if not is_body_part_shape_payload(sh):
                        shapes.append(sh)
                if fid == 0:
                    # Include manual seed body parts + companion PNG so frame 0 acts as immutable seed frame.
                    for sh in lm_payload["shapes"]:
                        if is_body_part_shape_payload(sh):
                            shapes.append(sh)
                    cv2.imwrite(str(ann_dir / f"social_huddle_{fid:09d}.png"), bgr)

            (ann_dir / f"social_huddle_{fid:09d}.json").write_text(
                json.dumps(
                    {
                        "version": "5.4.1",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": f"social_huddle_{fid:09d}.png",
                        "imageData": None,
                        "imageHeight": H,
                        "imageWidth": W,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        writer.release()

    exit_code = cli_main(
        [
            "segment-body-parts",
            str(video_path),
            "--annotations",
            str(ann_dir),
            "--mode",
            "auto",
            "--backbone",
            "fast",
            "--export-csv",
            "--json",
        ]
    )
    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["frames_processed"] == 4
    assert report["merged_parent_splits"] == 2
    assert report["recovered_parent_instances"] == 2

    # Verify recovered mouse_1 and mouse_2 parent polygons match 3D ground-truth track masks!
    for fid in sorted(corrupted_frames):
        repaired = json.loads(
            (ann_dir / f"social_huddle_{fid:09d}.json").read_text(encoding="utf-8")
        )
        parent_polys = {
            sh["label"]: sh["points"]
            for sh in repaired["shapes"]
            if sh.get("shape_type") == "polygon" and not is_body_part_shape_payload(sh)
        }
        assert set(parent_polys.keys()) == {"mouse_1", "mouse_2"}
        gt_tr = gt_track_by_frame[fid]
        m1_mask = _poly_pts_to_mask(parent_polys["mouse_1"], H, W)
        m2_mask = _poly_pts_to_mask(parent_polys["mouse_2"], H, W)
        assert _compute_iou(m1_mask, gt_tr == 1) >= 0.80
        assert _compute_iou(m2_mask, gt_tr == 2) >= 0.72

    # Verify CSV contains both whole-animal and '<instance>:<part>' rows.
    csv_path = Path(report["tracking_csv"])
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as fh:
        csv_names = {r["instance_name"] for r in csv.DictReader(fh)}
    assert {"mouse_1", "mouse_2", "mouse_1:head", "mouse_2:torso"}.issubset(csv_names)

    # Verify Track QC passes with zero missing-parent or geometry violations!
    qc = evaluate_track_qc(ann_dir, expected_instances=2)
    assert len(qc.instances) == 2
    bad_codes = {
        "body_part_missing_parent",
        "body_part_outside_instance",
        "body_part_overlap",
        "duplicate_instance",
        "count_mismatch",
    }
    assert not any(iss.code in bad_codes for iss in qc.issues)


def test_two_mice_oblique_occluded_tail_and_cross_posture_seeded_transfer():
    manifest = _load_manifest()
    rearing_records = [
        r for r in manifest["frames"] if r["scenario"] == "rearing_oblique"
    ]
    curled_records = [
        r for r in manifest["frames"] if r["scenario"] == "curled_grooming"
    ]
    assert len(rearing_records) == 2
    assert len(curled_records) == 2

    # 1. On rearing_oblique f02 (mouse_2 distal tail clipped at screen edge, only tail_base visible)
    # and f04 (mouse_2 turning obliquely away with planted forepaws), verify zero rump-carving and
    # pectoral chest containment!
    kp_seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="keypoint_guided",
            feature_backend="fast",
            keypoint_granularity="coarse",
        )
    )
    rgb_r2, tr_r2, _, c_r2, lm_r2 = _load_fixture_frame(rearing_records[0])
    H, W = tr_r2.shape
    preds_r2 = kp_seg.segment_frame(
        rgb_r2,
        {"mouse_1": tr_r2 == 1, "mouse_2": tr_r2 == 2},
        keypoints_by_instance=_visible_keypoints_from_labelme(lm_r2),
    )
    m2_r2 = {k: p.to_full_mask(H, W) for k, p in preds_r2["mouse_2"].items()}
    # mouse_2 on f02 has <= 10 px of clipped tail; tail_base must NOT carve hundreds of rump pixels!
    pr_tail_r2 = int(np.count_nonzero(m2_r2.get("tail", np.zeros((H, W), dtype=bool))))
    assert pr_tail_r2 <= 30
    assert _compute_iou(m2_r2["head"], (tr_r2 == 2) & (c_r2 == 1)) >= 0.84
    assert _compute_iou(m2_r2["torso"], (tr_r2 == 2) & (c_r2 == 2)) >= 0.90
    assert _compute_iou(m2_r2["limbs"], (tr_r2 == 2) & (c_r2 == 3)) >= 0.48

    rgb_r4, tr_r4, _, c_r4, lm_r4 = _load_fixture_frame(rearing_records[1])
    preds_r4 = kp_seg.segment_frame(
        rgb_r4,
        {"mouse_1": tr_r4 == 1, "mouse_2": tr_r4 == 2},
        keypoints_by_instance=_visible_keypoints_from_labelme(lm_r4),
    )
    m2_r4 = {k: p.to_full_mask(H, W) for k, p in preds_r4["mouse_2"].items()}
    assert _compute_iou(m2_r4["head"], (tr_r4 == 2) & (c_r4 == 1)) >= 0.50
    assert _compute_iou(m2_r4["torso"], (tr_r4 == 2) & (c_r4 == 2)) >= 0.84
    assert _compute_iou(m2_r4["limbs"], (tr_r4 == 2) & (c_r4 == 3)) >= 0.45

    # 2. Cross-posture, cross-strain seeded transfer: curled black mouse_1 -> elongated white mouse_2!
    rgb_c0, tr_c0, _, c_c0, _ = _load_fixture_frame(curled_records[0])
    p_c0 = {"mouse_1": tr_c0 == 1, "mouse_2": tr_c0 == 2}
    seed_curled_m1 = {
        "mouse_1": {
            "head": p_c0["mouse_1"] & (c_c0 == 1),
            "torso": p_c0["mouse_1"] & np.isin(c_c0, [2, 3]),
            "tail": p_c0["mouse_1"] & (c_c0 == 4),
        }
    }
    seeded_seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="seeded",
            feature_backend="fast",
            share_seed_across_instances=True,
            partition_full_instance=True,
        )
    )
    seeded_seg.fit_seed_frame(rgb_c0, p_c0, seed_curled_m1)

    rgb_c3, tr_c3, _, c_c3, _ = _load_fixture_frame(curled_records[1])
    p_c3 = {"mouse_1": tr_c3 == 1, "mouse_2": tr_c3 == 2}
    preds_c3 = seeded_seg.segment_frame(rgb_c3, p_c3)
    m2_c3 = {k: p.to_full_mask(H, W) for k, p in preds_c3["mouse_2"].items()}
    m2_ious = [
        _compute_iou(m2_c3["head"], p_c3["mouse_2"] & (c_c3 == 1)),
        _compute_iou(m2_c3["torso"], p_c3["mouse_2"] & np.isin(c_c3, [2, 3])),
        _compute_iou(m2_c3["tail"], p_c3["mouse_2"] & (c_c3 == 4)),
    ]
    assert float(np.mean(m2_ious)) >= 0.62
