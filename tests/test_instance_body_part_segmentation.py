from __future__ import annotations

import builtins
import csv
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from annolid.annotation.polygon_constraints import resolve_polygon_shape_conflicts
from annolid.engine.cli import main as cli_main
from annolid.gui.models_registry import MODEL_REGISTRY, validate_model_registry_entries
from annolid.gui.shape import Shape
from annolid.postprocessing.temporal_identity_repair import run_temporal_identity_repair
from annolid.postprocessing.track_qc import evaluate_track_qc
from annolid.segmentation.body_parts import (
    BodyPartSegmentationConfig,
    InstanceBodyPartSegmenter,
    InstanceBodyPartVideoProcessor,
    compute_masked_geodesic_distances,
)
from annolid.tracker.processor_registry import resolve_tracking_processor_class
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.dino_kpseg_annotations import DinoKPSEGAnnotationParser
from annolid.utils.body_part_shapes import (
    build_body_part_flags,
    build_body_part_metadata,
    build_body_part_shape_dict,
    infer_part_from_keypoint_name,
    is_body_part_shape_payload,
)


def _draw_synthetic_animal(
    height: int = 128,
    width: int = 160,
    *,
    center: tuple[int, int] = (80, 64),
    facing_right: bool = True,
) -> tuple[
    np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, tuple[float, float]]
]:
    """Render a deterministic 3-part synthetic rodent (head, torso, tail) + masks & keypoints."""
    img = np.full((height, width, 3), 235, dtype=np.uint8)
    cx, cy = center
    dir_sign = 1 if facing_right else -1

    torso_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(torso_mask, (cx, cy), (26, 16), 0, 0, 360, 1, -1)

    head_cx = cx + dir_sign * 28
    head_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(head_mask, (head_cx, cy), (14, 11), 0, 0, 360, 1, -1)

    tail_mask = np.zeros((height, width), dtype=np.uint8)
    t_start = (cx - dir_sign * 24, cy)
    t_end = (cx - dir_sign * 54, cy)
    cv2.line(tail_mask, t_start, t_end, 1, thickness=6)

    head_b = head_mask.astype(bool)
    torso_b = torso_mask.astype(bool) & ~head_b
    tail_b = tail_mask.astype(bool) & ~torso_b & ~head_b
    inst_b = head_b | torso_b | tail_b

    # Distinct visual appearance per anatomical region:
    # head = dark charcoal/brown, torso = medium warm gray, tail = bright pink/tan.
    img[torso_b] = np.array([125, 120, 115], dtype=np.uint8)
    img[head_b] = np.array([35, 30, 45], dtype=np.uint8)
    img[tail_b] = np.array([220, 145, 150], dtype=np.uint8)

    keypoints = {
        "nose": (float(cx + dir_sign * 39), float(cy)),
        "spine": (float(cx), float(cy)),
        "tail_base": (float(cx - dir_sign * 25), float(cy)),
        "tail_tip": (float(cx - dir_sign * 52), float(cy)),
    }
    return (
        img,
        inst_b,
        {"head": head_b, "torso": torso_b, "tail": tail_b},
        keypoints,
    )


def _mask_to_poly_pts(mask: np.ndarray) -> list[list[float]]:
    cnts, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest = max(cnts, key=cv2.contourArea)
    return [[float(pt[0][0]), float(pt[0][1])] for pt in largest]


def test_seeded_parts_strictly_contained_and_disjoint():
    frame0, inst0, parts0, _kps0 = _draw_synthetic_animal(
        center=(76, 64), facing_right=True
    )
    frame1, inst1, gt_parts1, _kps1 = _draw_synthetic_animal(
        center=(86, 68), facing_right=True
    )

    cfg = BodyPartSegmentationConfig(
        mode="seeded",
        feature_backend="fast",
        partition_full_instance=True,
    )
    segmenter = InstanceBodyPartSegmenter(cfg)
    built = segmenter.fit_seed_frame(
        frame0,
        {"mouse_1": inst0},
        {"mouse_1": parts0},
    )
    assert built == 3

    preds = segmenter.segment_frame(frame1, {"mouse_1": inst1})["mouse_1"]
    assert set(preds.keys()) == {"head", "torso", "tail"}

    H, W = inst1.shape
    full_masks = {name: p.to_full_mask(H, W) for name, p in preds.items()}

    # 1. Hard containment inside parent mask and full coverage.
    union_parts = np.zeros_like(inst1, dtype=bool)
    for name, pmask in full_masks.items():
        assert not np.any(pmask & ~inst1), f"{name} leaked outside parent mask"
        union_parts |= pmask
    np.testing.assert_array_equal(union_parts, inst1)

    # 2. Pairwise pixel and vector polygon mutual exclusion.
    names = list(full_masks.keys())
    polys = {name: Polygon(preds[name].polygon) for name in names}
    parent_poly = Polygon(_mask_to_poly_pts(inst1)).buffer(0.75)
    for idx, n1 in enumerate(names):
        assert polys[n1].is_valid
        assert parent_poly.covers(polys[n1])
        for n2 in names[idx + 1 :]:
            assert not np.any(full_masks[n1] & full_masks[n2])
            assert polys[n1].intersection(polys[n2]).area < 1e-2

    # 3. High IoU accuracy against ground-truth shifted parts.
    for name in ("head", "torso", "tail"):
        inter = np.count_nonzero(full_masks[name] & gt_parts1[name])
        union = np.count_nonzero(full_masks[name] | gt_parts1[name])
        assert inter / max(1, union) > 0.72, f"Low IoU for {name}"


def test_partial_part_seeding_preserves_unassigned_body_remainder():
    frame0, inst0, parts0, _ = _draw_synthetic_animal(
        center=(80, 64), facing_right=True
    )
    frame1, inst1, gt_parts1, _ = _draw_synthetic_animal(
        center=(84, 64), facing_right=True
    )

    # Seed ONLY "head" (~20% of animal area); torso + tail remain unseeded!
    segmenter = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(mode="seeded", feature_backend="fast")
    )
    segmenter.fit_seed_frame(
        frame0,
        {"mouse_1": inst0},
        {"mouse_1": {"head": parts0["head"]}},
    )
    preds = segmenter.segment_frame(frame1, {"mouse_1": inst1})["mouse_1"]

    assert set(preds.keys()) == {"head"}
    head_pred = preds["head"]
    assert 0.08 < head_pred.area_ratio < 0.42
    head_mask = head_pred.to_full_mask(*inst1.shape)
    inter = np.count_nonzero(head_mask & gt_parts1["head"])
    union = np.count_nonzero(head_mask | gt_parts1["head"])
    assert inter / max(1, union) > 0.70


def test_180_degree_turn_preserves_head_and_tail_polarity():
    frame0, inst0, parts0, _kps0 = _draw_synthetic_animal(
        center=(80, 64), facing_right=True
    )
    # Instantaneous 180-degree flip on frame 1 (animal now faces left!).
    frame1, inst1, gt_parts1, kps1 = _draw_synthetic_animal(
        center=(80, 64), facing_right=False
    )

    # Case A: Appearance + canonical bidirectional orientation hypothesis (no keypoints).
    seg_no_kp = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(mode="seeded", feature_backend="fast")
    )
    seg_no_kp.fit_seed_frame(frame0, {"mouse_1": inst0}, {"mouse_1": parts0})
    preds_no_kp = seg_no_kp.segment_frame(frame1, {"mouse_1": inst1})["mouse_1"]

    for name in ("head", "tail"):
        pm = preds_no_kp[name].to_full_mask(*inst1.shape)
        iou = np.count_nonzero(pm & gt_parts1[name]) / max(
            1, np.count_nonzero(pm | gt_parts1[name])
        )
        assert iou > 0.65, (
            f"180-deg flip swapped {name} without keypoints (IoU={iou:.2f})"
        )

    # Case B: Keypoint-locked canonical polarity.
    seg_kp = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(mode="keypoint_guided", feature_backend="fast")
    )
    preds_kp = seg_kp.segment_frame(
        frame1,
        {"mouse_1": inst1},
        keypoints_by_instance={"mouse_1": kps1},
    )["mouse_1"]
    head_xs = [pt[0] for pt in preds_kp["head"].polygon]
    tail_xs = [pt[0] for pt in preds_kp["tail"].polygon]
    # Facing left -> head is on the left (smaller x) and tail is on the right (larger x).
    assert float(np.mean(head_xs)) < float(np.mean(tail_xs))


def test_multi_animal_touching_instances_zero_bleed_and_shared_seeds():
    f_top, m_top, p_top, _ = _draw_synthetic_animal(
        height=140, width=180, center=(90, 50), facing_right=True
    )
    f_bot, m_bot, p_bot, _ = _draw_synthetic_animal(
        height=140, width=180, center=(90, 78), facing_right=False
    )
    composite = np.full((140, 180, 3), 235, dtype=np.uint8)
    composite[m_top] = f_top[m_top]
    composite[m_bot] = f_bot[m_bot]

    segmenter = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="seeded",
            feature_backend="fast",
            share_seed_across_instances=True,
        )
    )
    # Seed ONLY mouse_0; mouse_1 should inherit shared part prototypes automatically!
    segmenter.fit_seed_frame(
        composite,
        {"mouse_0": m_top, "mouse_1": m_bot},
        {"mouse_0": p_top},
    )
    preds = segmenter.segment_frame(
        composite,
        {"mouse_0": m_top, "mouse_1": m_bot},
    )
    assert set(preds.keys()) == {"mouse_0", "mouse_1"}
    assert set(preds["mouse_0"].keys()) == {"head", "torso", "tail"}
    assert set(preds["mouse_1"].keys()) == {"head", "torso", "tail"}

    H, W = m_top.shape
    m0_union = np.zeros((H, W), dtype=bool)
    m1_union = np.zeros((H, W), dtype=bool)
    for p in preds["mouse_0"].values():
        m0_union |= p.to_full_mask(H, W)
    for p in preds["mouse_1"].values():
        m1_union |= p.to_full_mask(H, W)

    # Zero cross-animal pixel bleed even at social-contact boundary!
    assert not np.any(m0_union & m1_union)
    assert not np.any(m0_union & ~m_top)
    assert not np.any(m1_union & ~p_bot["head"] & ~p_bot["torso"] & ~p_bot["tail"])


def test_keypoint_guided_masked_geodesic_prevents_curled_tail_head_jump(monkeypatch):
    """Curled U-shaped body where tail tip sits 4 px from nose across an air gap."""
    H, W = 96, 96
    domain = np.zeros((H, W), dtype=bool)
    # Top arm (head, y=20..30, x=20..70)
    domain[20:30, 20:70] = True
    # Right spine bend (torso, y=20..76, x=60..72)
    domain[20:76, 60:72] = True
    # Bottom arm curling back to x=20, then rising up to y=34 (only 4px air gap below y=30!)
    domain[66:76, 20:70] = True
    domain[34:76, 20:28] = True

    nose_seed = np.zeros((H, W), dtype=bool)
    nose_seed[25, 24] = True
    tail_seed = np.zeros((H, W), dtype=bool)
    tail_seed[36, 24] = (
        True  # Euclidean distance to nose is 11 px; geodesic is >120 px!
    )

    # Test 1: Default (scipy if installed) geodesic distance.
    dists = compute_masked_geodesic_distances([nose_seed, tail_seed], domain)
    assert dists[0, 25, 45] < dists[1, 25, 45]  # Top arm belongs to head
    assert dists[1, 42, 24] < dists[0, 42, 24]  # Curled tip belongs to tail
    assert not np.isfinite(dists[0, 32, 24])  # Air gap is unreachable

    # Test 2: Pure-NumPy Bellman-Ford fallback when scipy is unavailable.
    real_import = builtins.__import__

    def _no_scipy_import(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("Simulated lean install without scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_scipy_import)
    dists_np = compute_masked_geodesic_distances([nose_seed, tail_seed], domain)
    assert dists_np[0, 25, 45] < dists_np[1, 25, 45]
    assert dists_np[1, 42, 24] < dists_np[0, 42, 24]
    assert not np.isfinite(dists_np[0, 32, 24])

    # End-to-end keypoint_guided segmentation on the curled U-shaped animal.
    img = np.full((H, W, 3), 200, dtype=np.uint8)
    img[domain] = (120, 120, 120)
    seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(mode="keypoint_guided", feature_backend="fast")
    )
    preds = seg.segment_instance(
        img,
        "curled_mouse",
        domain,
        keypoints={
            "nose": (24.0, 25.0),
            "spine": (66.0, 48.0),
            "tail_tip": (24.0, 36.0),
        },
    )
    head_m = preds["head"].to_full_mask(H, W)
    tail_m = preds["tail"].to_full_mask(H, W)
    assert head_m[25, 24] and not tail_m[25, 24]
    assert tail_m[36, 24] and not head_m[36, 24]


def test_unsupervised_mode_segments_consistent_parts_across_frames():
    f0, m0, _, _ = _draw_synthetic_animal(center=(74, 64), facing_right=True)
    f1, m1, _, _ = _draw_synthetic_animal(center=(86, 66), facing_right=True)

    seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="unsupervised",
            num_parts=3,
            feature_backend="fast",
        )
    )
    p0 = seg.segment_frame(f0, {"mouse_1": m0})["mouse_1"]
    p1 = seg.segment_frame(f1, {"mouse_1": m1})["mouse_1"]
    assert set(p0.keys()) == {"head", "torso", "tail"}
    assert set(p1.keys()) == {"head", "torso", "tail"}

    # Verify ordering along x-axis stays consistent between frame 0 and frame 1.
    order_0 = sorted(p0.keys(), key=lambda k: np.mean([pt[0] for pt in p0[k].polygon]))
    order_1 = sorted(p1.keys(), key=lambda k: np.mean([pt[0] for pt in p1[k].polygon]))
    assert order_0 == order_1


def test_dino_and_hybrid_feature_backends_with_mock_extractor():
    class _MockDinoExtractor:
        def __init__(self, multi_layer: bool = False):
            self.patch_size = 8
            self.multi_layer = multi_layer

        def extract(self, pil_img, return_layer="all", normalize=True):
            _ = return_layer, normalize
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            gh = max(4, arr.shape[0] // 8)
            gw = max(4, arr.shape[1] // 8)
            small = cv2.resize(arr, (gw, gh), interpolation=cv2.INTER_AREA)
            chw = np.transpose(small, (2, 0, 1))
            norms = np.linalg.norm(chw, axis=0, keepdims=True)
            normed = chw / np.maximum(norms, 1e-6)
            if self.multi_layer:
                return np.stack([0.9 * normed, normed, 1.05 * normed], axis=0)
            return normed

    f0, m0, parts0, _ = _draw_synthetic_animal(center=(80, 64), facing_right=True)
    f1, m1, gt1, _ = _draw_synthetic_animal(center=(88, 64), facing_right=True)

    for backend, multi_layer in (("dino", False), ("dino", True), ("hybrid", True)):
        seg = InstanceBodyPartSegmenter(
            BodyPartSegmentationConfig(mode="seeded", feature_backend=backend),
            extractor=_MockDinoExtractor(multi_layer=multi_layer),
        )
        seg.fit_seed_frame(f0, {"mouse_1": m0}, {"mouse_1": parts0})
        preds = seg.segment_frame(f1, {"mouse_1": m1})["mouse_1"]
        assert set(preds.keys()) == {"head", "torso", "tail"}
        head_m = preds["head"].to_full_mask(*m1.shape)
        iou = np.count_nonzero(head_m & gt1["head"]) / max(
            1, np.count_nonzero(head_m | gt1["head"])
        )
        assert iou > 0.65


def test_keypoint_lexical_inference_granularities():
    assert infer_part_from_keypoint_name("left_ear", granularity="coarse") == "head"
    assert infer_part_from_keypoint_name("tail_base", granularity="coarse") == "tail"
    assert (
        infer_part_from_keypoint_name("left_fore_paw", granularity="fine")
        == "left_forelimb"
    )
    assert (
        infer_part_from_keypoint_name("right_wing_tip", granularity="fine")
        == "right_wing"
    )
    assert infer_part_from_keypoint_name("nose", granularity="per_keypoint") == "nose"


def test_hierarchy_aware_polygon_conflict_resolution():
    parent = Shape(label="mouse_1", shape_type="polygon", group_id=1)
    parent.points = [[10.0, 10.0], [90.0, 10.0], [90.0, 50.0], [10.0, 50.0]]

    # Head extends slightly outside parent at x=5..45, overlaps torso at x=35..45.
    head = Shape(
        label="head",
        shape_type="polygon",
        group_id=1,
        flags=build_body_part_flags(),
    )
    head.other_data = build_body_part_metadata(
        parent_instance="mouse_1", part_label="head"
    )
    head.points = [[5.0, 12.0], [45.0, 12.0], [45.0, 48.0], [5.0, 48.0]]

    torso = Shape(
        label="torso",
        shape_type="polygon",
        group_id=1,
        flags=build_body_part_flags(),
    )
    torso.other_data = build_body_part_metadata(
        parent_instance="mouse_1", part_label="torso"
    )
    torso.points = [[35.0, 12.0], [88.0, 12.0], [88.0, 48.0], [35.0, 48.0]]

    res = resolve_polygon_shape_conflicts(
        [parent, head, torso], priorities=[1.0, 0.9, 0.8]
    )
    assert len(res.shapes) == 3

    p_poly = Polygon(res.shapes[0].points)
    h_poly = Polygon(res.shapes[1].points)
    t_poly = Polygon(res.shapes[2].points)

    # Parent must remain un-carved (area == 80 * 40 == 3200)!
    assert pytest.approx(p_poly.area, rel=1e-3) == 3200.0
    # Child parts must be clipped inside parent and disjoint from each other!
    assert p_poly.buffer(1e-2).covers(h_poly)
    assert p_poly.buffer(1e-2).covers(t_poly)
    assert h_poly.intersection(t_poly).area < 1e-2


def test_video_processor_cli_and_downstream_compatibility(tmp_path: Path, capsys):
    video_path = tmp_path / "social_mice.mp4"
    result_dir = tmp_path / "social_mice"
    result_dir.mkdir(parents=True, exist_ok=True)

    H, W = 140, 180
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (W, H),
    )
    assert writer.isOpened()

    for frame_idx in range(3):
        shift = frame_idx * 4
        f_a, m_a, parts_a, kps_a = _draw_synthetic_animal(
            height=H, width=W, center=(76 + shift, 44), facing_right=True
        )
        f_b, m_b, _parts_b, kps_b = _draw_synthetic_animal(
            height=H, width=W, center=(96 - shift, 98), facing_right=False
        )
        canvas_rgb = np.full((H, W, 3), 235, dtype=np.uint8)
        canvas_rgb[m_a] = f_a[m_a]
        canvas_rgb[m_b] = f_b[m_b]
        writer.write(cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))

        shapes = [
            {
                "label": "mouse_1",
                "group_id": 1,
                "shape_type": "polygon",
                "points": _mask_to_poly_pts(m_a),
                "flags": {},
            },
            {
                "label": "mouse_2",
                "group_id": 2,
                "shape_type": "polygon",
                "points": _mask_to_poly_pts(m_b),
                "flags": {},
            },
            {
                "label": "nose",
                "group_id": 1,
                "shape_type": "point",
                "points": [[kps_a["nose"][0], kps_a["nose"][1]]],
                "flags": {},
            },
            {
                "label": "nose",
                "group_id": 2,
                "shape_type": "point",
                "points": [[kps_b["nose"][0], kps_b["nose"][1]]],
                "flags": {},
            },
        ]
        # On frame 0, draw manual nested part polygons inside mouse_1 WITHOUT custom flags
        # to verify automatic nested manual seed discovery!
        if frame_idx == 0:
            for pname, pmask in parts_a.items():
                shapes.append(
                    {
                        "label": pname,
                        "group_id": 1,
                        "shape_type": "polygon",
                        "points": _mask_to_poly_pts(pmask),
                        "flags": {},
                    }
                )
            png_path = result_dir / f"social_mice_{frame_idx:09d}.png"
            cv2.imwrite(str(png_path), cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))

        json_path = result_dir / f"social_mice_{frame_idx:09d}.json"
        json_path.write_text(
            json.dumps(
                {
                    "version": "5.3.1",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": f"social_mice_{frame_idx:09d}.png",
                    "imageData": None,
                    "imageHeight": H,
                    "imageWidth": W,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    writer.release()

    # 1. Registry & GUI ModelConfig checks.
    assert (
        resolve_tracking_processor_class("body_part_seg")
        is InstanceBodyPartVideoProcessor
    )
    valid_reg, reg_errors, _ = validate_model_registry_entries(MODEL_REGISTRY)
    assert valid_reg, reg_errors

    # 2. Run CLI `annolid-run segment-body-parts` with CSV export & JSON stdout!
    report_out = tmp_path / "seg_report.json"
    exit_code = cli_main(
        [
            "segment-body-parts",
            str(video_path),
            "--annotations",
            str(result_dir),
            "--mode",
            "auto",
            "--backbone",
            "fast",
            "--export-csv",
            "--out",
            str(report_out),
            "--json",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    cli_report = json.loads(captured.out)
    assert cli_report["frames_processed"] == 3
    assert cli_report["seed_frames_fitted"] == 1
    assert set(cli_report["part_counts"].keys()) == {"head", "torso", "tail"}
    assert report_out.exists()

    # 3. Check AnnotationAdapter round-trip preserves parent instances AND child body_parts!
    adapter = AnnotationAdapter(image_height=H, image_width=W, persist_json=True)
    reg_f1 = adapter.read_annotation(result_dir / "social_mice_000000001.json")
    assert set(reg_f1.instances.keys()) == {"mouse_1", "mouse_2"}
    assert set(reg_f1.instances["mouse_1"].body_parts.keys()) == {
        "head",
        "torso",
        "tail",
    }
    assert set(reg_f1.instances["mouse_2"].body_parts.keys()) == {
        "head",
        "torso",
        "tail",
    }

    roundtrip_dir = tmp_path / "roundtrip"
    rt_path = adapter.write_annotation(
        frame_number=1, registry=reg_f1, output_dir=roundtrip_dir
    )
    rt_payload = json.loads(rt_path.read_text(encoding="utf-8"))
    part_shapes_rt = [s for s in rt_payload["shapes"] if is_body_part_shape_payload(s)]
    assert len(part_shapes_rt) == 6
    # Ensure per-shape flags remain strictly boolean for Qt LabelMe safety!
    for s in rt_payload["shapes"]:
        for flag_v in (s.get("flags") or {}).values():
            assert isinstance(flag_v, bool)

    # 4. DinoKPSEGAnnotationParser ignores child body-part polygons cleanly.
    kpseg_parser = DinoKPSEGAnnotationParser(
        image_height=H, image_width=W, adapter=adapter
    )
    manual_parsed = kpseg_parser.read_manual_annotation(
        1, result_dir / "social_mice_000000001.json"
    )
    assert len(manual_parsed.registry.instances) == 2

    # 5. Tracking CSV contains both whole animals and hierarchical '<instance>:<part>' rows!
    csv_path = Path(cli_report["tracking_csv"])
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    names_in_csv = {r["instance_name"] for r in rows}
    assert {"mouse_1", "mouse_2", "mouse_1:head", "mouse_2:tail"}.issubset(names_in_csv)

    # 6. Track QC on BOTH LabelMe directory and tracking CSV sees 2 animals with zero false errors!
    qc_dir = evaluate_track_qc(result_dir, expected_instances=2)
    assert qc_dir.expected_instances == 2
    assert len(qc_dir.instances) == 2
    assert not any(
        issue.code in {"duplicate_instance", "count_mismatch", "spatial_overlap"}
        for issue in qc_dir.issues
    )

    qc_csv = evaluate_track_qc(csv_path, expected_instances=2)
    assert len(qc_csv.instances) == 2
    assert not any(
        issue.code in {"duplicate_instance", "count_mismatch", "spatial_overlap"}
        for issue in qc_csv.issues
    )

    # 7. Temporal identity repair ignores child body parts cleanly.
    repair_res = run_temporal_identity_repair(
        result_dir, expected_instance_count=2, apply_changes=False
    )
    assert repair_res.scanned_files == 3
    assert repair_res.updated_files == 0

    # 8. Malformed out-of-bounds body part triggers dedicated `body_part_outside_instance` QC warning!
    f2_path = result_dir / "social_mice_000000002.json"
    f2_data = json.loads(f2_path.read_text(encoding="utf-8"))
    f2_data["shapes"].append(
        build_body_part_shape_dict(
            points=[[2.0, 2.0], [22.0, 2.0], [22.0, 22.0], [2.0, 22.0]],
            parent_instance="mouse_1",
            part_label="rogue_part",
            group_id=1,
        )
    )
    f2_path.write_text(json.dumps(f2_data, indent=2), encoding="utf-8")
    qc_rogue = evaluate_track_qc(result_dir, expected_instances=2)
    assert any(i.code == "body_part_outside_instance" for i in qc_rogue.issues)

    # 9. Cooperative stop_event halts video processor immediately.
    stop_ev = threading.Event()
    stop_ev.set()
    proc = InstanceBodyPartVideoProcessor(
        str(video_path),
        results_folder=str(result_dir),
        stop_event=stop_ev,
    )
    msg = proc.process_video_frames()
    proc.cleanup()
    assert "stopped" in msg.lower()


def test_registry_and_gui_wiring_for_body_part_segmentation():
    from annolid.gui.mixins.model_identity_mixin import ModelIdentityMixin
    from annolid.gui.mixins.prediction_execution_mixin import PredictionExecutionMixin
    from annolid.segmentation.cutie_vos.runtime import (
        resolve_tracking_video_processor_class,
    )
    from annolid.tracker.processor_registry import BODY_PART_SEG_BACKEND
    from annolid.tracking.configuration import CutieDinoTrackerConfig

    assert BODY_PART_SEG_BACKEND == "body_part_seg"
    resolved = resolve_tracking_video_processor_class("body_part_seg")
    assert resolved is InstanceBodyPartVideoProcessor

    matches = [
        m for m in MODEL_REGISTRY if m.display_name == "Instance Body Part Segmentation"
    ]
    assert len(matches) == 1
    cfg = matches[0]
    assert cfg.identifier == "body_part_seg"
    assert cfg.weight_file == "INSTANCE_BODY_PART_SEG"
    assert validate_model_registry_entries([cfg]) == (True, [], [])

    class _Host(PredictionExecutionMixin, ModelIdentityMixin):
        pass

    host = _Host()
    assert host._is_body_part_seg_model("body_part_seg", "INSTANCE_BODY_PART_SEG")
    assert not host._is_body_part_seg_model("Cutie", "cutie-base-mega.pth")

    selected_model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    host.tracker_runtime_config = CutieDinoTrackerConfig(
        patch_model_name=selected_model
    )
    assert host._dino_model_kwargs_for_tracking_processor(
        "body_part_seg", "INSTANCE_BODY_PART_SEG"
    ) == {
        "patch_model_name": selected_model,
        "dinov3_model_name": selected_model,
    }


def _draw_circular_animal(
    height: int = 128,
    width: int = 128,
    center: tuple[int, int] = (64, 64),
    radius: int = 24,
    heading_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Draw a perfect aspect-ratio 1.00 circular animal rotated at ``heading_deg``."""
    frame = np.full((height, width, 3), 232, dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dx = xx - center[0]
    dy = yy - center[1]
    inst_mask = (dx * dx + dy * dy) <= (radius * radius)

    rad = np.deg2rad(heading_deg)
    proj = (dx * np.cos(rad) + dy * np.sin(rad)) / float(radius)

    head_gt = inst_mask & (proj >= 0.30)
    tail_gt = inst_mask & (proj <= -0.30)
    torso_gt = inst_mask & ~head_gt & ~tail_gt

    # Distinct anatomical coat colors: dark snout/head cap, warm brown torso, pale tail-base.
    frame[torso_gt] = (145, 102, 68)
    frame[head_gt] = (32, 34, 42)
    frame[tail_gt] = (218, 198, 178)
    return frame, inst_mask, {"head": head_gt, "torso": torso_gt, "tail": tail_gt}


def test_unseeded_circular_posture_tracks_arbitrary_90_and_180_rotations():
    f0, m0, gt0 = _draw_circular_animal(heading_deg=0.0)
    f90, m90, gt90 = _draw_circular_animal(heading_deg=90.0)
    f180, m180, gt180 = _draw_circular_animal(heading_deg=180.0)

    seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="unsupervised",
            num_parts=3,
            feature_backend="fast",
        )
    )
    p0 = seg.segment_frame(f0, {"mouse_1": m0})["mouse_1"]
    p90 = seg.segment_frame(f90, {"mouse_1": m90})["mouse_1"]
    p180 = seg.segment_frame(f180, {"mouse_1": m180})["mouse_1"]

    # Determine which end label ("head" vs "tail") anchored the dark anterior pole on cold-start f0,
    # then verify it rotates by +90 deg on f90 and +180 deg on f180 despite aspect_ratio == 1.00!
    head_mask_0 = p0["head"].to_full_mask(*m0.shape)
    ant_label = (
        "head"
        if np.count_nonzero(head_mask_0 & gt0["head"])
        >= np.count_nonzero(head_mask_0 & gt0["tail"])
        else "tail"
    )
    for preds, gt in ((p0, gt0), (p90, gt90), (p180, gt180)):
        pred_ant = preds[ant_label].to_full_mask(*m0.shape)
        iou = np.count_nonzero(pred_ant & gt["head"]) / max(
            1, np.count_nonzero(pred_ant | gt["head"])
        )
        assert iou > 0.55


def test_unseeded_curled_u_shape_uses_intrinsic_geodesic_spine_without_keypoints():
    from annolid.segmentation.body_parts import estimate_body_frame_pose

    H, W = 96, 96
    frame = np.full((H, W, 3), 236, dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    # U-turn body: top head arm (thick, dark), left spine bend (brown), bottom tail arm (thin, pale).
    mask[18:30, 22:76] = True  # thicker top arm (12 px)
    mask[18:72, 22:34] = True  # left spine bend (12 px)
    mask[64:72, 22:76] = True  # thinner bottom tail arm (8 px)

    top_head = np.zeros((H, W), dtype=bool)
    top_head[18:30, 46:76] = True
    bot_tail = np.zeros((H, W), dtype=bool)
    bot_tail[64:72, 46:76] = True
    bend_torso = mask & ~top_head & ~bot_tail

    frame[bend_torso] = (142, 98, 66)
    frame[top_head] = (34, 36, 44)
    frame[bot_tail] = (216, 196, 176)

    pose = estimate_body_frame_pose(mask)
    assert pose.tortuosity > 1.35

    seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="unsupervised",
            num_parts=3,
            feature_backend="fast",
        )
    )
    preds = seg.segment_frame(frame, {"mouse_1": mask})["mouse_1"]
    assert set(preds.keys()) == {"head", "torso", "tail"}

    head_m = preds["head"].to_full_mask(H, W)
    tail_m = preds["tail"].to_full_mask(H, W)
    # Head must concentrate on the thicker top arm and NOT jump across the air gap onto the bottom arm!
    assert np.count_nonzero(head_m & top_head) > 4 * np.count_nonzero(head_m & bot_tail)
    assert np.count_nonzero(tail_m & bot_tail) > 4 * np.count_nonzero(tail_m & top_head)


def test_merged_parent_mask_splits_huddling_animals_and_recovers_missing_partner(
    tmp_path: Path,
):
    video_path = tmp_path / "huddle.mp4"
    result_dir = tmp_path / "huddle"
    result_dir.mkdir(parents=True, exist_ok=True)

    H, W = 140, 180
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (W, H),
    )
    assert writer.isOpened()

    # Frames 0 & 1: mouse_1 (upper) and mouse_2 (lower) approach;
    # Frame 2: they touch/huddle and upstream tracker emits ONE merged mouse_1 polygon (mouse_2 dropped!);
    # Frame 3: they separate again.
    centers = [
        ((88, 42), (88, 98)),
        ((88, 48), (88, 92)),
        ((88, 56), (88, 82)),  # touching/overlapping vertical contact!
        ((88, 46), (88, 94)),
    ]
    for f_idx, (c_a, c_b) in enumerate(centers):
        f_a, m_a, _pa, _ka = _draw_synthetic_animal(
            height=H, width=W, center=c_a, facing_right=True
        )
        f_b, m_b, _pb, _kb = _draw_synthetic_animal(
            height=H, width=W, center=c_b, facing_right=False
        )
        # Make mouse_2 coat slightly lighter tan for realistic inter-animal boundary contrast.
        f_b_adj = np.clip(f_b.astype(np.int32) + 28, 0, 255).astype(np.uint8)
        canvas = np.full((H, W, 3), 236, dtype=np.uint8)
        canvas[m_a] = f_a[m_a]
        canvas[m_b] = f_b_adj[m_b]
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        if f_idx == 2:
            merged = m_a | m_b
            # Bridge any tiny 1-2 px gap between the two huddling silhouettes so it is one connected blob.
            merged = (
                cv2.morphologyEx(
                    merged.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    np.ones((9, 9), dtype=np.uint8),
                )
                > 0
            )
            shapes = [
                {
                    "label": "mouse_1",
                    "group_id": 1,
                    "shape_type": "polygon",
                    "points": _mask_to_poly_pts(merged),
                    "flags": {},
                }
            ]
        else:
            shapes = [
                {
                    "label": "mouse_1",
                    "group_id": 1,
                    "shape_type": "polygon",
                    "points": _mask_to_poly_pts(m_a),
                    "flags": {},
                },
                {
                    "label": "mouse_2",
                    "group_id": 2,
                    "shape_type": "polygon",
                    "points": _mask_to_poly_pts(m_b),
                    "flags": {},
                },
            ]

        frame_json = result_dir / f"huddle_{f_idx:09d}.json"
        frame_json.write_text(
            json.dumps(
                {
                    "version": "5.0.1",
                    "imagePath": f"huddle_{f_idx:09d}.png",
                    "imageData": None,
                    "imageHeight": H,
                    "imageWidth": W,
                    "shapes": shapes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    writer.release()

    proc = InstanceBodyPartVideoProcessor(
        str(video_path),
        results_folder=str(result_dir),
        body_part_config=BodyPartSegmentationConfig(
            mode="unsupervised",
            num_parts=3,
            feature_backend="fast",
            split_merged_parent_masks=True,
            recover_missing_merged_instances=True,
        ),
    )
    proc.process_video_frames()
    summary = dict(proc.last_summary)
    proc.cleanup()

    assert summary["merged_parent_splits"] == 1
    assert summary["recovered_parent_instances"] == 1

    f2_saved = json.loads((result_dir / "huddle_000000002.json").read_text("utf-8"))
    whole_animals_f2 = [
        s for s in f2_saved["shapes"] if not is_body_part_shape_payload(s)
    ]
    assert {s["label"] for s in whole_animals_f2} == {"mouse_1", "mouse_2"}
    rec_m2 = next(s for s in whole_animals_f2 if s["label"] == "mouse_2")
    assert rec_m2["flags"] == {"recovered_merged_parent": True}

    qc = evaluate_track_qc(result_dir, expected_instances=2)
    assert len(qc.instances) == 2
    assert not any(
        issue.code
        in {
            "count_mismatch",
            "spatial_overlap",
            "body_part_outside_instance",
            "body_part_missing_parent",
        }
        for issue in qc.issues
    )


def test_anomalous_parent_area_freezes_ema_and_qc_flags_missing_parent(tmp_path: Path):
    from annolid.utils.body_part_shapes import build_body_part_shape_dict

    f0, m0, parts0, _ = _draw_synthetic_animal(center=(80, 64), facing_right=True)
    seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="seeded",
            feature_backend="fast",
            ema_alpha=0.35,
            ema_min_confidence=0.1,
        )
    )
    seg.fit_seed_frame(f0, {"mouse_1": m0}, {"mouse_1": parts0})
    seg.segment_frame(f0, {"mouse_1": m0})
    proto_before = seg._instance_prototypes["mouse_1"][0].working_prototype.copy()

    # Simulate an unsplit 2.2x shadow/bedding bloat on a single-animal track with altered RGB.
    bloated = (
        cv2.dilate(m0.astype(np.uint8), np.ones((19, 19), dtype=np.uint8), iterations=1)
        > 0
    )
    corrupted_rgb = np.clip(255 - f0, 0, 255).astype(np.uint8)
    seg.segment_frame(corrupted_rgb, {"mouse_1": bloated})
    assert "mouse_1" in seg.last_frame_diagnostics["anomalous_area_instances"]
    proto_after = seg._instance_prototypes["mouse_1"][0].working_prototype
    np.testing.assert_allclose(proto_before, proto_after, atol=1e-6)

    # Verify TrackQC flags an orphan body part whose parent instance is missing on that frame.
    orphan_dir = tmp_path / "orphan_qc"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    orphan_json = orphan_dir / "orphan_000000000.json"
    orphan_json.write_text(
        json.dumps(
            {
                "version": "5.0.1",
                "imagePath": "orphan_000000000.png",
                "imageData": None,
                "imageHeight": 128,
                "imageWidth": 160,
                "shapes": [
                    {
                        "label": "mouse_1",
                        "shape_type": "polygon",
                        "points": _mask_to_poly_pts(m0),
                        "flags": {},
                    },
                    build_body_part_shape_dict(
                        points=[[10.0, 10.0], [24.0, 10.0], [24.0, 24.0], [10.0, 24.0]],
                        parent_instance="ghost_mouse",
                        part_label="head",
                    ),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    qc = evaluate_track_qc(orphan_dir, expected_instances=1)
    assert any(issue.code == "body_part_missing_parent" for issue in qc.issues)
