from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from annolid.colab.monitor import load_metrics_from_results_csv

from .models import (
    FLY_BEHAVIORS,
    MOUSE_SCENARIOS,
    RAT_SCENARIOS,
    Sim2RealIterationRecord,
    SimDatasetConfig,
    SimIterationAdjustment,
    SimIterationEvaluation,
    VideoEnvironmentProfile,
)

_FRAME_NUM_RE = re.compile(r"(\d+)(?!.*\d)")


def _shape_bbox(shape: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    pts = shape.get("points") or []
    if not pts:
        return None
    stype = str(shape.get("shape_type") or "polygon").lower()
    if stype == "point":
        return None
    xs = [float(p[0]) for p in pts if len(p) >= 2]
    ys = [float(p[1]) for p in pts if len(p) >= 2]
    if not xs or not ys:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0 + 1.0 or y1 <= y0 + 1.0:
        return None
    return x0, y0, x1, y1


def _bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _bbox_area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def _extract_frame_index(path: Path, fallback: int) -> int:
    m = _FRAME_NUM_RE.search(path.stem)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return fallback


def _extract_shape_score(shape: Mapping[str, Any]) -> Optional[float]:
    for key in ("score", "confidence", "conf"):
        val = shape.get(key)
        if val is not None:
            try:
                f = float(val)
                if 0.0 <= f <= 1.0:
                    return f
            except (TypeError, ValueError):
                pass
    flags = shape.get("flags") or {}
    if isinstance(flags, Mapping):
        for key in ("score", "confidence", "conf"):
            val = flags.get(key)
            if val is not None:
                try:
                    f = float(val)
                    if 0.0 <= f <= 1.0:
                        return f
                except (TypeError, ValueError):
                    pass
    return None


def _collect_labelme_jsons(pred_dir: Path) -> list[Path]:
    if not pred_dir.exists():
        return []
    if pred_dir.is_file() and pred_dir.suffix.lower() == ".json":
        return [pred_dir]
    jsons = sorted(
        p
        for p in pred_dir.rglob("*.json")
        if p.is_file()
        and p.name
        not in {
            "dataset_manifest.json",
            "sim2real_state.json",
            "sim2real_report.json",
            "evaluation_summary.json",
        }
    )
    # Filter to files that look like LabelMe payloads (have "shapes" list).
    valid: list[Path] = []
    for jf in jsons:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("shapes"), list):
                valid.append(jf)
        except Exception:
            continue
    return valid


def _find_results_csv(training_output_dir: Optional[str | Path]) -> Optional[Path]:
    if not training_output_dir:
        return None
    root = Path(training_output_dir).expanduser()
    if not root.exists():
        return None
    candidates = sorted(root.rglob("results.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _read_video_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    if not video_path.is_file():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _render_prediction_overlay_sheet(
    sample_items: Sequence[tuple[np.ndarray, list[dict[str, Any]], str]],
    output_path: Path,
) -> str:
    if not sample_items:
        return ""
    palette = [
        (72, 195, 255),
        (90, 230, 130),
        (255, 165, 80),
        (235, 120, 255),
    ]
    tiles: list[np.ndarray] = []
    tw, th = 360, 270
    for bgr, shapes, caption in sample_items[:4]:
        vis = bgr.copy()
        overlay = bgr.copy()
        for idx, sh in enumerate(shapes):
            color = palette[idx % len(palette)]
            pts = sh.get("points") or []
            stype = str(sh.get("shape_type") or "polygon").lower()
            if stype == "polygon" and len(pts) >= 3:
                arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [arr], color)
                cv2.polylines(vis, [arr], True, color, 2, cv2.LINE_AA)
            elif stype == "rectangle" and len(pts) >= 2:
                p0 = (int(round(pts[0][0])), int(round(pts[0][1])))
                p1 = (int(round(pts[1][0])), int(round(pts[1][1])))
                cv2.rectangle(vis, p0, p1, color, 2)
            elif stype == "point" and len(pts) >= 1:
                cv2.circle(
                    vis,
                    (int(round(pts[0][0])), int(round(pts[0][1]))),
                    3,
                    (255, 255, 255),
                    -1,
                )
            lbl = str(sh.get("label") or "")
            sc = _extract_shape_score(sh)
            if pts and lbl:
                tag = f"{lbl} {sc:.2f}" if sc is not None else lbl
                cv2.putText(
                    vis,
                    tag,
                    (int(round(pts[0][0])), max(16, int(round(pts[0][1])) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        blended = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0.0)
        tile = cv2.resize(blended, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tw, 26), (18, 22, 30), -1)
        cv2.putText(
            tile,
            caption[:46],
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 244, 252),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if len(tiles) == 1:
        sheet = tiles[0]
    elif len(tiles) == 2:
        sheet = np.concatenate(tiles, axis=1)
    else:
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate(
            [
                np.concatenate(tiles[:2], axis=1),
                np.concatenate(tiles[2:4], axis=1),
            ],
            axis=0,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return str(output_path)


def _evaluate_supervised_pair(
    pred_shapes_by_stem: Mapping[str, list[dict[str, Any]]],
    gt_dir: Path,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    gt_files = _collect_labelme_jsons(gt_dir)
    if not gt_files:
        return None, None, None, None
    matched_ious: list[float] = []
    tp = 0
    fp = 0
    fn = 0
    for jf in gt_files:
        gt_data = json.loads(jf.read_text(encoding="utf-8"))
        gt_boxes = [
            b
            for sh in (gt_data.get("shapes") or [])
            if (b := _shape_bbox(sh)) is not None
        ]
        pred_shapes = pred_shapes_by_stem.get(jf.stem) or []
        pred_boxes = [b for sh in pred_shapes if (b := _shape_bbox(sh)) is not None]
        if not gt_boxes and not pred_boxes:
            continue
        used_pred: set[int] = set()
        for gb in gt_boxes:
            best_iou = 0.0
            best_j = -1
            for j, pb in enumerate(pred_boxes):
                if j in used_pred:
                    continue
                iou = _bbox_iou(gb, pb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= 0.5:
                used_pred.add(best_j)
                matched_ious.append(best_iou)
                tp += 1
            else:
                fn += 1
        fp += max(0, len(pred_boxes) - len(used_pred))

    if tp + fp + fn == 0:
        return None, None, None, None
    prec = tp / float(max(1, tp + fp))
    rec = tp / float(max(1, tp + fn))
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    miou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return round(miou, 4), round(prec, 4), round(rec, 4), round(f1, 4)


def evaluate_iteration_outputs(
    *,
    example_videos: Sequence[str | Path],
    prediction_dirs: Sequence[str | Path],
    profile: Optional[VideoEnvironmentProfile] = None,
    training_output_dir: Optional[str | Path] = None,
    ground_truth_labelme_dir: Optional[str | Path] = None,
    overlay_output_path: Optional[str | Path] = None,
    expected_subject_count: Optional[int] = None,
) -> SimIterationEvaluation:
    """Evaluate a trained iteration using training metrics + real example video predictions."""
    ev = SimIterationEvaluation()
    exp_count = int(
        expected_subject_count
        if expected_subject_count is not None
        else (profile.estimated_subject_count if profile is not None else 1)
    )
    ev.expected_subject_count = max(1, exp_count)

    # 1. Extract training validation metrics from results.csv if available.
    results_csv = _find_results_csv(training_output_dir)
    if results_csv is not None:
        pts = load_metrics_from_results_csv(results_csv)
        if pts:
            best_pt = max(
                pts,
                key=lambda p: (
                    p.mask_map50
                    if p.mask_map50 is not None
                    else (
                        p.pose_map50
                        if p.pose_map50 is not None
                        else (p.map50 if p.map50 is not None else -1.0)
                    )
                ),
            )
            ev.best_train_epoch = int(best_pt.step)
            ev.val_box_map50 = (
                round(float(best_pt.map50), 4) if best_pt.map50 is not None else None
            )
            ev.val_mask_map50 = (
                round(float(best_pt.mask_map50), 4)
                if best_pt.mask_map50 is not None
                else None
            )
            ev.val_pose_map50 = (
                round(float(best_pt.pose_map50), 4)
                if best_pt.pose_map50 is not None
                else None
            )
            last_pt = pts[-1]
            ev.final_train_loss = (
                round(float(last_pt.train_loss), 4)
                if last_pt.train_loss is not None
                else None
            )
            ev.final_val_loss = (
                round(float(last_pt.val_loss), 4)
                if last_pt.val_loss is not None
                else None
            )

    # 2. Evaluate real-video LabelMe predictions.
    all_scores: list[float] = []
    det_counts: list[int] = []
    temporal_ious: list[float] = []
    area_series: list[float] = []
    class_counts: dict[str, int] = {}
    pred_shapes_by_stem: dict[str, list[dict[str, Any]]] = {}
    overlay_samples: list[tuple[np.ndarray, list[dict[str, Any]], str]] = []
    per_video: list[dict[str, Any]] = []

    resolved_videos = [Path(v).expanduser().resolve() for v in example_videos]
    resolved_pred_dirs = [Path(d).expanduser().resolve() for d in prediction_dirs]

    for v_idx, pred_dir in enumerate(resolved_pred_dirs):
        src_video = resolved_videos[v_idx] if v_idx < len(resolved_videos) else None
        json_files = _collect_labelme_jsons(pred_dir)
        json_files.sort(key=lambda p: (_extract_frame_index(p, 0), p.name))

        prev_boxes: list[tuple[float, float, float, float]] = []
        vid_frames = 0
        vid_nonempty = 0
        vid_scores: list[float] = []
        vid_counts: list[int] = []

        sample_pick = set(
            np.linspace(
                0, max(0, len(json_files) - 1), num=min(2, len(json_files)), dtype=int
            ).tolist()
            if json_files
            else []
        )

        for j_idx, jf in enumerate(json_files):
            payload = json.loads(jf.read_text(encoding="utf-8"))
            shapes = [
                sh for sh in (payload.get("shapes") or []) if isinstance(sh, Mapping)
            ]
            pred_shapes_by_stem[jf.stem] = shapes
            boxes: list[tuple[float, float, float, float]] = []
            frame_scores: list[float] = []
            for sh in shapes:
                lbl = str(sh.get("label") or "object")
                class_counts[lbl] = class_counts.get(lbl, 0) + 1
                box = _shape_bbox(sh)
                if box is not None:
                    boxes.append(box)
                sc = _extract_shape_score(sh)
                if sc is not None:
                    frame_scores.append(sc)
                    all_scores.append(sc)
                    vid_scores.append(sc)

            f_num = _extract_frame_index(jf, j_idx)
            vid_frames += 1
            ev.evaluated_frames += 1
            n_det = len(boxes) if boxes else len(shapes)
            det_counts.append(n_det)
            vid_counts.append(n_det)

            if n_det > 0:
                vid_nonempty += 1
                ev.frames_with_detections += 1
                area_series.append(
                    sum(_bbox_area(b) for b in boxes) / max(1, len(boxes))
                )
            else:
                ev.empty_frames.append(f_num)

            if frame_scores and float(np.mean(frame_scores)) < 0.45:
                ev.low_confidence_frames.append(f_num)

            # Frame-to-frame temporal continuity via greedy box IoU matching.
            if prev_boxes and boxes:
                best_matches: list[float] = []
                for cur in boxes:
                    best_matches.append(max(_bbox_iou(cur, prv) for prv in prev_boxes))
                if best_matches:
                    temporal_ious.append(float(np.mean(best_matches)))
            prev_boxes = boxes

            # Collect overlay samples for the report contact sheet.
            if j_idx in sample_pick and len(overlay_samples) < 4:
                img_bgr = None
                img_name = str(payload.get("imagePath") or "")
                if img_name:
                    cand = jf.parent / Path(img_name).name
                    if cand.is_file():
                        img_bgr = cv2.imread(str(cand), cv2.IMREAD_COLOR)
                if img_bgr is None and src_video is not None:
                    img_bgr = _read_video_frame(src_video, f_num)
                if img_bgr is not None:
                    caption = f"{src_video.name if src_video else pred_dir.name} | f{f_num} ({n_det} det)"
                    overlay_samples.append((img_bgr, shapes, caption))

        per_video.append(
            {
                "video": str(src_video) if src_video else str(pred_dir),
                "prediction_dir": str(pred_dir),
                "evaluated_frames": vid_frames,
                "detection_coverage": round(
                    vid_nonempty / float(max(1, vid_frames)), 4
                ),
                "mean_confidence": round(float(np.mean(vid_scores)), 4)
                if vid_scores
                else 0.0,
                "mean_detections_per_frame": round(float(np.mean(vid_counts)), 3)
                if vid_counts
                else 0.0,
            }
        )

    ev.per_video_summaries = per_video
    ev.class_detection_counts = class_counts

    if ev.evaluated_frames > 0:
        ev.detection_coverage = round(
            ev.frames_with_detections / float(ev.evaluated_frames), 4
        )
        ev.mean_detections_per_frame = round(float(np.mean(det_counts)), 3)
        # Match rate allows exact count or within +/-1 when expected_subject_count >= 3.
        tol = 0 if ev.expected_subject_count <= 2 else 1
        matches = sum(
            1 for c in det_counts if abs(c - ev.expected_subject_count) <= tol
        )
        ev.subject_count_match_rate = round(matches / float(len(det_counts)), 4)

    if all_scores:
        ev.mean_confidence = round(float(np.mean(all_scores)), 4)
        ev.p10_confidence = round(float(np.percentile(all_scores, 10)), 4)
    elif ev.frames_with_detections > 0:
        # Prediction JSONs without per-shape scores get neutral confidence proxy.
        ev.mean_confidence = 0.65
        ev.p10_confidence = 0.55
        ev.notes.append(
            "Prediction LabelMe JSONs did not include per-shape `score` fields; used neutral confidence proxy."
        )

    if temporal_ious:
        ev.temporal_iou_stability = round(float(np.mean(temporal_ious)), 4)
    if len(area_series) >= 2 and float(np.mean(area_series)) > 1.0:
        cv_area = float(np.std(area_series) / np.mean(area_series))
        ev.area_stability = round(float(np.clip(1.0 - min(1.0, cv_area), 0.0, 1.0)), 4)
    elif area_series:
        ev.area_stability = 1.0

    # 3. Optional supervised metrics if labeled validation LabelMe frames exist.
    if ground_truth_labelme_dir:
        gt_path = Path(ground_truth_labelme_dir).expanduser().resolve()
        miou, prec, rec, f1 = _evaluate_supervised_pair(pred_shapes_by_stem, gt_path)
        ev.supervised_miou = miou
        ev.supervised_precision = prec
        ev.supervised_recall = rec
        ev.supervised_f1 = f1

    # 4. Transparent composite Sim-to-Real score in [0, 1].
    val_metric = (
        ev.val_mask_map50
        if ev.val_mask_map50 is not None
        else (
            ev.val_pose_map50
            if ev.val_pose_map50 is not None
            else (ev.val_box_map50 if ev.val_box_map50 is not None else 0.0)
        )
    )
    if ev.supervised_f1 is not None and ev.supervised_miou is not None:
        score = (
            0.35 * ev.supervised_f1
            + 0.25 * ev.supervised_miou
            + 0.20 * ev.detection_coverage
            + 0.10 * ev.temporal_iou_stability
            + 0.10 * val_metric
        )
    elif ev.evaluated_frames > 0:
        score = (
            0.30 * ev.detection_coverage
            + 0.25 * ev.subject_count_match_rate
            + 0.20 * ev.mean_confidence
            + 0.15 * ev.temporal_iou_stability
            + 0.10 * val_metric
        )
    else:
        score = float(val_metric)
    ev.sim2real_score = round(float(np.clip(score, 0.0, 1.0)), 4)

    if overlay_output_path is not None and overlay_samples:
        ev.overlay_contact_sheet_path = _render_prediction_overlay_sheet(
            overlay_samples, Path(overlay_output_path)
        )

    return ev


def suggest_next_iteration(
    *,
    current_config: SimDatasetConfig,
    evaluation: SimIterationEvaluation,
    profile: Optional[VideoEnvironmentProfile] = None,
    previous_iterations: Sequence[Sim2RealIterationRecord] = (),
    current_best_weights: str = "",
    current_epochs: int = 20,
    current_imgsz: int = 640,
    target_score: float = 0.88,
) -> SimIterationAdjustment:
    """Analyze real-video evaluation signals and propose the next Sim-to-Real iteration."""
    next_cfg = SimDatasetConfig.from_dict(current_config.to_dict())
    next_cfg.seed = int(current_config.seed) + 101
    train_overrides: dict[str, Any] = {}
    if current_best_weights:
        train_overrides["weights"] = current_best_weights

    diagnosis: list[str] = []
    recommendations: list[str] = []

    # Check convergence / plateau
    if (
        evaluation.sim2real_score >= target_score
        and evaluation.detection_coverage >= 0.90
    ):
        diagnosis.append(
            f"Target Sim-to-Real score reached ({evaluation.sim2real_score:.3f} >= {target_score:.2f}) "
            f"with {evaluation.detection_coverage:.1%} real-video detection coverage."
        )
        recommendations.append(
            "Model is performing well on the example videos. Export the report and run full-video inference, "
            "or add 5-15 hand-corrected hard frames for final fine-tuning."
        )
        return SimIterationAdjustment(
            diagnosis=diagnosis,
            recommendations=recommendations,
            suggested_sim_config=next_cfg.to_dict(),
            suggested_train_overrides=train_overrides,
            should_continue=False,
            stop_reason="target_score_reached",
        )

    # 1. Low recall / missing detections on real example video
    if evaluation.evaluated_frames > 0 and evaluation.detection_coverage < 0.78:
        diagnosis.append(
            f"Low real-video detection coverage ({evaluation.detection_coverage:.1%}; "
            f"{len(evaluation.empty_frames)} empty frame(s)). Indicates sim-to-real appearance gap."
        )
        next_cfg.match_example_photometry = True
        next_cfg.photometry_strength = round(
            min(0.95, next_cfg.photometry_strength + 0.12), 2
        )
        next_cfg.domain_randomization = round(
            min(0.65, next_cfg.domain_randomization + 0.12), 2
        )
        if profile is not None and profile.reference_background_path:
            next_cfg.composite_on_real_background = True
            recommendations.append(
                "Enable compositing rendered 3D subjects onto the extracted real arena background "
                "and raise photometry matching + domain randomization."
            )
        else:
            recommendations.append(
                "Increase photometry matching strength and domain randomization to bridge visual gap."
            )
        next_cfg.frames_per_scenario = min(
            120, int(round(next_cfg.frames_per_scenario * 1.4))
        )
        train_overrides["epochs"] = min(80, max(int(current_epochs) + 10, 25))
        recommendations.append(
            f"Expand synthetic frames per scenario to {next_cfg.frames_per_scenario} and fine-tune for "
            f"{train_overrides['epochs']} epochs from previous best.pt."
        )

    # 2. Under-counting or over-counting subjects (occlusion / false positives)
    if evaluation.evaluated_frames > 0 and evaluation.subject_count_match_rate < 0.72:
        if (
            evaluation.mean_detections_per_frame
            < evaluation.expected_subject_count - 0.35
        ):
            diagnosis.append(
                f"Under-detecting subjects (mean {evaluation.mean_detections_per_frame:.2f} vs "
                f"expected {evaluation.expected_subject_count}). Likely contact occlusion or dark/small instances."
            )
            if next_cfg.species == "mouse":
                for scen in ("social_huddle", "curled_groom", "calms_corner_huddle"):
                    if scen not in next_cfg.scenarios and scen in MOUSE_SCENARIOS:
                        next_cfg.scenarios.append(scen)
                        break
                recommendations.append(
                    f"Add close-contact scenario to dataset (`{next_cfg.scenarios}`) to improve multi-mouse separation."
                )
            elif next_cfg.species == "rat":
                for scen in ("gait_transition", "bumpy_terrain", "bound", "walk"):
                    if scen not in next_cfg.scenarios and scen in RAT_SCENARIOS:
                        next_cfg.scenarios.append(scen)
                        break
                recommendations.append(
                    f"Add varied Rat NCAP locomotion scenario (`{next_cfg.scenarios}`) to expand posture diversity."
                )
            else:
                for beh in ("grooming", "forage_walk", "backward_walk"):
                    if beh not in next_cfg.scenarios and beh in FLY_BEHAVIORS:
                        next_cfg.scenarios.append(beh)
                        break
                recommendations.append(
                    f"Add varied FlyBody posture behavior (`{next_cfg.scenarios}`) and raise resolution."
                )
            if current_imgsz < 800:
                train_overrides["imgsz"] = 768 if current_imgsz <= 640 else 960
                recommendations.append(
                    f"Increase training `imgsz` to {train_overrides['imgsz']} for clearer small-subject boundaries."
                )
        elif (
            evaluation.mean_detections_per_frame
            > evaluation.expected_subject_count + 0.65
        ):
            diagnosis.append(
                f"Over-detecting false positives (mean {evaluation.mean_detections_per_frame:.2f} vs "
                f"expected {evaluation.expected_subject_count})."
            )
            next_cfg.domain_randomization = round(
                min(0.7, next_cfg.domain_randomization + 0.15), 2
            )
            if profile is not None and profile.reference_background_path:
                next_cfg.composite_on_real_background = True
            train_overrides["conf"] = 0.35
            recommendations.append(
                "Composite onto real arena background to suppress static arena false positives and test with `conf=0.35`."
            )

    # 3. Low confidence or temporal flicker
    if evaluation.evaluated_frames > 0 and (
        evaluation.mean_confidence < 0.58 or evaluation.temporal_iou_stability < 0.55
    ):
        diagnosis.append(
            f"Unstable or low-confidence predictions (mean conf={evaluation.mean_confidence:.2f}, "
            f"temporal IoU={evaluation.temporal_iou_stability:.2f})."
        )
        next_cfg.frames_per_scenario = min(
            140,
            max(
                next_cfg.frames_per_scenario + 12,
                int(next_cfg.frames_per_scenario * 1.25),
            ),
        )
        train_overrides.setdefault("epochs", min(80, int(current_epochs) + 10))
        recommendations.append(
            "Increase training diversity (new seed + more frames per scenario) and continue fine-tuning from previous weights."
        )

    # 4. Synthetic validation mAP itself is low -> model underfitted on synthetic dataset
    val_map = (
        evaluation.val_mask_map50
        if evaluation.val_mask_map50 is not None
        else (
            evaluation.val_pose_map50
            if evaluation.val_pose_map50 is not None
            else evaluation.val_box_map50
        )
    )
    if val_map is not None and val_map < 0.65:
        diagnosis.append(
            f"Synthetic validation mAP@50 is still low ({val_map:.3f}); model has not converged yet."
        )
        train_overrides["epochs"] = min(100, max(int(current_epochs) + 15, 30))
        recommendations.append(
            f"Train longer ({train_overrides['epochs']} epochs) before increasing domain randomization further."
        )

    # Check if two consecutive iterations plateaued with negligible gain.
    if len(previous_iterations) >= 2:
        prev_scores = [
            float((it.evaluation or {}).get("sim2real_score", 0.0))
            for it in previous_iterations[-2:]
            if isinstance(it.evaluation, Mapping)
        ]
        if (
            len(prev_scores) == 2
            and abs(evaluation.sim2real_score - prev_scores[-1]) < 0.01
            and abs(prev_scores[-1] - prev_scores[-2]) < 0.01
        ):
            diagnosis.append(
                "Sim-to-Real score plateaued across 3 consecutive iterations (< 0.01 gain)."
            )
            recommendations.append(
                "Consider annotating 5-10 hard frames (see `low_confidence_frames`) in Annolid and setting `real_labeled_dir` for hybrid sim+real fine-tuning."
            )

    if not diagnosis:
        diagnosis.append(
            f"Iteration scored {evaluation.sim2real_score:.3f} (coverage={evaluation.detection_coverage:.1%}, "
            f"conf={evaluation.mean_confidence:.2f}, temporal IoU={evaluation.temporal_iou_stability:.2f})."
        )
        next_cfg.frames_per_scenario = min(120, next_cfg.frames_per_scenario + 10)
        next_cfg.domain_randomization = round(
            min(0.6, next_cfg.domain_randomization + 0.08), 2
        )
        train_overrides.setdefault("epochs", min(60, int(current_epochs) + 5))
        recommendations.append(
            "Fine-tune another round from current best.pt with expanded seeded diversity to further improve generalization."
        )

    return SimIterationAdjustment(
        diagnosis=diagnosis,
        recommendations=recommendations,
        suggested_sim_config=next_cfg.to_dict(),
        suggested_train_overrides=train_overrides,
        should_continue=True,
    )
