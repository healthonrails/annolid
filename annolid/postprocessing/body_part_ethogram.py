"""Anatomical body-part kinematics, directed social-contact ethogram, and uncertainty ranking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import Polygon

from annolid.utils.annotation_store import (
    AnnotationStore,
    AnnotationStoreError,
    load_labelme_json,
)
from annolid.utils.body_part_shapes import (
    extract_body_part_label,
    extract_body_part_parent_instance,
    infer_part_from_keypoint_name,
    is_body_part_shape_payload,
)
from annolid.utils.zone_shapes import is_zone_shape_payload

__all__ = [
    "BodyPartEthogramReport",
    "SocialContactBout",
    "analyze_body_part_ethogram",
    "export_body_part_ethogram_csvs",
]


@dataclass(frozen=True, slots=True)
class SocialContactBout:
    """Contiguous behavioral contact bout between two tracked animals."""

    actor: str
    recipient: str
    behavior: str
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_s: float
    min_distance_px: float


@dataclass(slots=True)
class BodyPartEthogramReport:
    """Structured output of part-level kinematics, social contact bouts, and QC uncertainty."""

    kinematics_df: pd.DataFrame
    contacts_df: pd.DataFrame
    bouts: List[SocialContactBout]
    uncertain_frames: List[int]
    fps: float

    def to_summary_dict(self) -> Dict[str, Any]:
        bout_counts: Dict[str, int] = defaultdict(int)
        bout_seconds: Dict[str, float] = defaultdict(float)
        for b in self.bouts:
            bout_counts[b.behavior] += 1
            bout_seconds[b.behavior] += float(b.duration_s)
        posture_counts: Dict[str, int] = {}
        if (
            not self.kinematics_df.empty
            and "posture_state" in self.kinematics_df.columns
        ):
            vc = self.kinematics_df["posture_state"].value_counts()
            posture_counts = {str(k): int(v) for k, v in sorted(vc.items())}
        return {
            "fps": float(self.fps),
            "frames_analyzed": (
                int(self.kinematics_df["frame_number"].nunique())
                if not self.kinematics_df.empty
                else 0
            ),
            "instances": (
                sorted(self.kinematics_df["instance_name"].unique().tolist())
                if not self.kinematics_df.empty
                else []
            ),
            "posture_frame_counts": posture_counts,
            "total_bouts": len(self.bouts),
            "bout_counts_by_behavior": dict(sorted(bout_counts.items())),
            "bout_counts_by_type": dict(sorted(bout_counts.items())),
            "bout_seconds_by_behavior": {
                k: round(v, 4) for k, v in sorted(bout_seconds.items())
            },
            "bouts": [asdict(b) for b in self.bouts],
            "uncertain_frames": list(self.uncertain_frames),
        }


def _wrap_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return -180.0 if wrapped == 180.0 else wrapped


def _valid_polygon(points: Sequence[Sequence[float]]) -> Optional[Polygon]:
    if len(points) < 3:
        return None
    coords = [(float(pt[0]), float(pt[1])) for pt in points if len(pt) >= 2]
    if len(coords) < 3:
        return None
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            return None
        if isinstance(poly, Polygon):
            return poly if poly.area > 0.0 else None
        polys = [
            g
            for g in getattr(poly, "geoms", ())
            if isinstance(g, Polygon) and g.area > 0
        ]
        return max(polys, key=lambda p: p.area) if polys else None
    except Exception:
        return None


def _geometry_aspect_and_span(geom: Any) -> Tuple[float, float]:
    """Return ``(aspect_ratio, major_axis_span_px)`` for a ``Polygon`` or ``MultiPolygon``."""
    sub_polys = (
        [geom]
        if isinstance(geom, Polygon)
        else [g for g in getattr(geom, "geoms", ()) if isinstance(g, Polygon)]
    )
    if not sub_polys:
        return 1.0, 1.0
    coord_blocks = [
        np.asarray(p.exterior.coords[:-1], dtype=np.float64)
        for p in sub_polys
        if len(p.exterior.coords) >= 4
    ]
    if not coord_blocks:
        return 1.0, 1.0
    coords = np.vstack(coord_blocks)
    if coords.shape[0] < 3:
        return 1.0, 1.0
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / float(coords.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    lam_max = max(float(eigvals[-1]), 1e-4)
    lam_min = max(float(eigvals[0]), 1e-4)
    major_vec = eigvecs[:, -1]
    proj = centered @ major_vec
    major_span = max(1.0, float(proj.max() - proj.min()))
    return float(math.sqrt(lam_max / lam_min)), major_span


def _iter_frame_payloads(
    source: Path | str | Sequence[Tuple[int, Mapping[str, Any]]],
) -> List[Tuple[int, Mapping[str, Any]]]:
    if not isinstance(source, (Path, str)):
        return sorted(
            [(int(fnum), payload) for fnum, payload in source],
            key=lambda item: item[0],
        )

    folder = Path(source).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {folder}")

    frame_paths: Dict[int, Path] = {}
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".json"):
            continue
        m = re.search(r"(\d+)(?=\.json$)", name)
        if m is None:
            continue
        frame_paths[int(m.group(1))] = folder / name

    stem = folder.name
    store = AnnotationStore.for_frame_path(folder / f"{stem}_000000000.json")
    if store.store_path.exists():
        for fnum in store.iter_frames():
            frame_paths.setdefault(int(fnum), folder / f"{stem}_{int(fnum):09d}.json")

    loaded: List[Tuple[int, Mapping[str, Any]]] = []
    for fnum, fpath in sorted(frame_paths.items()):
        try:
            payload = load_labelme_json(fpath)
        except (FileNotFoundError, AnnotationStoreError, ValueError):
            continue
        if isinstance(payload, Mapping):
            loaded.append((fnum, payload))
    return loaded


def _normalize_camera_view_mode(camera_view: Optional[str]) -> str:
    raw = str(camera_view or "auto").strip().lower()
    if raw in {"top", "top-close", "top_down", "top-down", "overhead"}:
        return "top"
    if raw in {"oblique", "oblique-close", "side", "lateral", "front"}:
        return "oblique"
    if raw in {"bottom", "bottom-up", "bottom_up", "ventral"}:
        return "bottom"
    return "auto"


def analyze_body_part_ethogram(
    source: Path | str | Sequence[Tuple[int, Mapping[str, Any]]],
    *,
    fps: float = 30.0,
    contact_margin_px: float = 6.0,
    min_bout_frames: int = 1,
    max_uncertain_frames: int = 10,
    camera_view: str = "auto",
) -> BodyPartEthogramReport:
    """Compute anatomical body-part kinematics, directed social-contact bouts, and QC rankings."""
    eff_fps = max(1e-3, float(fps))
    margin = max(0.0, float(contact_margin_px))
    min_len = max(1, int(min_bout_frames))
    view_mode = _normalize_camera_view_mode(camera_view)
    frames = _iter_frame_payloads(source)

    kin_rows: List[Dict[str, Any]] = []
    raw_posture_cues: List[Dict[str, float]] = []
    contact_rows: List[Dict[str, Any]] = []
    uncertainty_candidates: List[Tuple[float, int]] = []
    prev_head_ratio: Dict[str, float] = {}

    for fnum, payload in frames:
        shapes = payload.get("shapes") or []
        if not isinstance(shapes, Sequence):
            continue

        parents: Dict[str, Any] = {}
        gid_to_parent: Dict[str, str] = {}
        raw_parts: List[Tuple[Mapping[str, Any], Polygon]] = []
        raw_points: List[Mapping[str, Any]] = []

        for shape in shapes:
            if not isinstance(shape, Mapping) or is_zone_shape_payload(shape):
                continue
            stype = str(shape.get("shape_type") or "polygon").strip().lower()
            if stype == "point":
                raw_points.append(shape)
                continue
            if stype != "polygon":
                continue
            poly = _valid_polygon(shape.get("points") or [])
            if poly is None:
                continue
            if is_body_part_shape_payload(shape):
                raw_parts.append((shape, poly))
                continue
            inst = str(shape.get("instance_label") or shape.get("label") or "").strip()
            if not inst:
                continue
            if inst in parents:
                parents[inst] = make_valid(parents[inst].union(poly))
            else:
                parents[inst] = poly
            gid = shape.get("group_id")
            if gid not in (None, ""):
                gid_to_parent[str(gid).strip()] = inst

        if not parents:
            continue

        # Map visible pose keypoints per animal instance when available.
        kps_by_inst: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
        for pt_sh in raw_points:
            pts = pt_sh.get("points") or []
            if not pts or len(pts[0]) < 2:
                continue
            flags = pt_sh.get("flags") or {}
            if (
                flags.get("kp_visible") is False
                or int(pt_sh.get("kp_visibility", 2)) <= 0
            ):
                continue
            owner_kp = str(pt_sh.get("instance_label") or "").strip()
            if not owner_kp or owner_kp not in parents:
                gid = pt_sh.get("group_id")
                if gid not in (None, "") and str(gid).strip() in gid_to_parent:
                    owner_kp = gid_to_parent[str(gid).strip()]
            if owner_kp in parents:
                kp_lbl = (
                    str(pt_sh.get("display_label") or pt_sh.get("label") or "")
                    .strip()
                    .lower()
                )
                if kp_lbl:
                    kps_by_inst[owner_kp][kp_lbl] = (float(pts[0][0]), float(pts[0][1]))

        # Group coarse anatomical part polygons & confidences per parent instance.
        parts_by_inst: Dict[str, Dict[str, List[Polygon]]] = defaultdict(
            lambda: defaultdict(list)
        )
        confs_by_inst: Dict[str, List[float]] = defaultdict(list)

        for shape, poly in raw_parts:
            owner = extract_body_part_parent_instance(shape)
            if not owner or owner not in parents:
                gid = shape.get("group_id")
                if gid not in (None, "") and str(gid).strip() in gid_to_parent:
                    owner = gid_to_parent[str(gid).strip()]
                else:
                    owner = max(
                        parents.keys(),
                        key=lambda k: parents[k].intersection(poly).area,
                    )
            raw_plbl = extract_body_part_label(shape) or "torso"
            coarse = infer_part_from_keypoint_name(raw_plbl, granularity="coarse")
            parts_by_inst[owner][coarse].append(poly)
            score_raw = shape.get("score")
            if isinstance(score_raw, (int, float)) and math.isfinite(float(score_raw)):
                confs_by_inst[owner].append(float(score_raw))

        time_s = float(fnum) / eff_fps
        frame_max_unc = 0.0

        merged_coarse_geom: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for inst in sorted(parents.keys()):
            p_poly = parents[inst]
            p_area = max(1e-3, float(p_poly.area))
            elong, major_span_px = _geometry_aspect_and_span(p_poly)

            part_areas: Dict[str, float] = {}
            for cname in ("head", "torso", "tail", "limbs"):
                plist = parts_by_inst[inst].get(cname, [])
                if plist:
                    u_geom = plist[0]
                    for extra in plist[1:]:
                        u_geom = make_valid(u_geom.union(extra))
                    merged_coarse_geom[inst][cname] = u_geom
                    part_areas[cname] = float(u_geom.area)
                else:
                    part_areas[cname] = 0.0

            h_ratio = part_areas["head"] / p_area
            t_ratio = part_areas["torso"] / p_area
            tl_ratio = part_areas["tail"] / p_area
            lm_ratio = part_areas["limbs"] / p_area

            head_g = merged_coarse_geom[inst].get("head")
            torso_g = merged_coarse_geom[inst].get("torso")
            tail_g = merged_coarse_geom[inst].get("tail")

            head_yaw = float("nan")
            body_yaw = float("nan")
            deflection = float("nan")
            upward_pitch_sin = 0.0
            head_tail_span_ratio = float("nan")
            trunk_elong = elong
            trunk_span = major_span_px
            head_torso_reach_ratio = 0.55
            if head_g is not None and torso_g is not None:
                hc = head_g.centroid
                tc = torso_g.centroid
                dx_ht = float(hc.x - tc.x)
                dy_ht = float(hc.y - tc.y)
                dist_ht = math.hypot(dx_ht, dy_ht)
                trunk_geom = make_valid(head_g.union(torso_g))
                trunk_elong, trunk_span = _geometry_aspect_and_span(trunk_geom)
                head_torso_reach_ratio = float(
                    np.clip(dist_ht / max(1.0, trunk_span), 0.0, 1.0)
                )
                head_yaw = math.degrees(math.atan2(dy_ht, dx_ht))
                if dist_ht > 1e-3:
                    # In image coordinates (+y downward), -dy_ht > 0 indicates upward head pitch.
                    upward_pitch_sin = float(np.clip(-dy_ht / dist_ht, 0.0, 1.0))
                if tail_g is not None:
                    tlc = tail_g.centroid
                    body_yaw = math.degrees(math.atan2(tc.y - tlc.y, tc.x - tlc.x))
                    ht_dist = math.hypot(float(hc.x - tlc.x), float(hc.y - tlc.y))
                    head_tail_span_ratio = float(
                        np.clip(ht_dist / max(1.0, major_span_px), 0.0, 1.5)
                    )
                else:
                    pc = p_poly.centroid
                    body_yaw = math.degrees(math.atan2(hc.y - pc.y, hc.x - pc.x))
                deflection = _wrap_deg(head_yaw - body_yaw)

            # Bipedal vertical stance & spinal flexion cues from keypoints when present.
            kp_map_i = kps_by_inst.get(inst) or {}
            kp_spinal_bend_deg = 0.0
            rostral_kp = (
                kp_map_i.get("nose")
                or kp_map_i.get("snout")
                or (
                    (float(head_g.centroid.x), float(head_g.centroid.y))
                    if head_g is not None
                    else None
                )
            )
            cerv_kp = kp_map_i.get("neck") or kp_map_i.get("spine_1")
            caud_kp = (
                kp_map_i.get("tail_base")
                or kp_map_i.get("tailbase")
                or kp_map_i.get("spine_2")
            )
            if rostral_kp is not None and cerv_kp is not None and caud_kp is not None:
                v_head = (rostral_kp[0] - cerv_kp[0], rostral_kp[1] - cerv_kp[1])
                v_trunk = (cerv_kp[0] - caud_kp[0], cerv_kp[1] - caud_kp[1])
                if math.hypot(*v_head) > 2.0 and math.hypot(*v_trunk) > 2.0:
                    ang_h = math.degrees(math.atan2(v_head[1], v_head[0]))
                    ang_t = math.degrees(math.atan2(v_trunk[1], v_trunk[0]))
                    kp_spinal_bend_deg = abs(_wrap_deg(ang_h - ang_t))
                    if not math.isfinite(deflection) or kp_spinal_bend_deg > abs(
                        deflection
                    ):
                        deflection = _wrap_deg(ang_h - ang_t)
            ant_ys = [
                pt[1]
                for k_n, pt in kp_map_i.items()
                if any(
                    tok in k_n
                    for tok in ("nose", "snout", "ear", "forepaw", "forelimb")
                )
            ]
            hind_ys = [
                pt[1]
                for k_n, pt in kp_map_i.items()
                if any(tok in k_n for tok in ("hindpaw", "hindlimb", "tail_base"))
            ]
            kp_bipedal_lift = 0.0
            if ant_ys and hind_ys:
                kp_bipedal_lift = float(
                    np.clip(
                        (float(np.mean(hind_ys)) - float(np.mean(ant_ys)))
                        / max(6.0, trunk_span),
                        0.0,
                        1.0,
                    )
                )

            c_vals = confs_by_inst.get(inst) or [1.0]
            mean_c = float(np.mean(c_vals))
            min_c = float(np.min(c_vals))

            inst_unc = 1.0 - (0.55 * min_c + 0.45 * mean_c)
            if head_g is None or torso_g is None:
                inst_unc += 0.35
            if (
                inst in prev_head_ratio
                and h_ratio > 0.02
                and prev_head_ratio[inst] > 0.02
            ):
                jump = max(
                    h_ratio / prev_head_ratio[inst], prev_head_ratio[inst] / h_ratio
                )
                if jump > 2.0:
                    inst_unc += min(0.35, 0.18 * (jump - 2.0))
            if h_ratio > 0.0:
                prev_head_ratio[inst] = h_ratio
            frame_max_unc = max(frame_max_unc, float(np.clip(inst_unc, 0.0, 1.0)))

            raw_posture_cues.append(
                {
                    "upward_pitch_sin": upward_pitch_sin,
                    "kp_bipedal_lift": kp_bipedal_lift,
                    "head_tail_span_ratio": head_tail_span_ratio,
                    "trunk_elong": trunk_elong,
                    "head_torso_reach_ratio": head_torso_reach_ratio,
                    "kp_spinal_bend_deg": kp_spinal_bend_deg,
                }
            )
            kin_rows.append(
                {
                    "frame_number": int(fnum),
                    "time_s": round(time_s, 4),
                    "instance_name": inst,
                    "whole_area_px": round(p_area, 2),
                    "elongation_ratio": round(elong, 4),
                    "head_area_ratio": round(h_ratio, 4),
                    "torso_area_ratio": round(t_ratio, 4),
                    "tail_area_ratio": round(tl_ratio, 4),
                    "limbs_area_ratio": round(lm_ratio, 4),
                    "head_yaw_deg": round(head_yaw, 2)
                    if math.isfinite(head_yaw)
                    else np.nan,
                    "body_yaw_deg": round(body_yaw, 2)
                    if math.isfinite(body_yaw)
                    else np.nan,
                    "head_body_deflection_deg": (
                        round(deflection, 2) if math.isfinite(deflection) else np.nan
                    ),
                    "head_tail_span_ratio": (
                        round(head_tail_span_ratio, 4)
                        if math.isfinite(head_tail_span_ratio)
                        else np.nan
                    ),
                    "curl_groom_index": 0.0,
                    "rearing_index": 0.0,
                    "posture_state": "locomotion",
                    "mean_part_confidence": round(mean_c, 4),
                    "min_part_confidence": round(min_c, 4),
                }
            )

        if frame_max_unc >= 0.22:
            uncertainty_candidates.append((frame_max_unc, int(fnum)))

        # Directed dyadic contact measurements across all ordered pairs (actor != recipient).
        inst_list = sorted(parents.keys())
        for actor in inst_list:
            for recipient in inst_list:
                if actor == recipient:
                    continue
                a_head = merged_coarse_geom[actor].get("head")
                a_torso = merged_coarse_geom[actor].get("torso") or parents[actor]
                b_head = merged_coarse_geom[recipient].get("head")
                b_torso = (
                    merged_coarse_geom[recipient].get("torso") or parents[recipient]
                )
                b_tail = merged_coarse_geom[recipient].get("tail")

                d_hh = (
                    float(a_head.distance(b_head))
                    if (a_head is not None and b_head is not None)
                    else float("inf")
                )
                if a_head is not None and b_tail is not None:
                    d_ht = float(a_head.distance(b_tail))
                elif a_head is not None and b_torso is not None:
                    d_ht = float(a_head.distance(b_torso))
                else:
                    d_ht = float("inf")
                d_tt = float(a_torso.distance(b_torso))

                nose_to_nose = d_hh <= margin
                anogenital = (
                    a_head is not None
                    and d_ht <= margin
                    and (b_tail is not None or d_ht + 2.0 < d_hh)
                    and not nose_to_nose
                )
                huddle = d_tt <= margin

                contact_rows.append(
                    {
                        "frame_number": int(fnum),
                        "time_s": round(time_s, 4),
                        "actor": actor,
                        "recipient": recipient,
                        "head_to_head_px": round(d_hh, 3)
                        if math.isfinite(d_hh)
                        else np.nan,
                        "head_to_tail_px": round(d_ht, 3)
                        if math.isfinite(d_ht)
                        else np.nan,
                        "torso_to_torso_px": round(d_tt, 3),
                        "nose_to_nose": bool(nose_to_nose),
                        "anogenital_sniff": bool(anogenital),
                        "side_by_side_huddle": bool(huddle),
                    }
                )

    if kin_rows:
        areas_by_inst: Dict[str, List[float]] = defaultdict(list)
        elongs_by_inst: Dict[str, List[float]] = defaultdict(list)
        all_areas: List[float] = []
        for row in kin_rows:
            a_val = float(row["whole_area_px"])
            e_val = float(row["elongation_ratio"])
            areas_by_inst[str(row["instance_name"])].append(a_val)
            elongs_by_inst[str(row["instance_name"])].append(e_val)
            all_areas.append(a_val)

        cohort_ref_area = float(np.percentile(all_areas, 80)) if all_areas else 1.0
        ref_area_by_inst: Dict[str, float] = {}
        ref_elong_by_inst: Dict[str, float] = {}
        for inst_k, a_list in areas_by_inst.items():
            inst_p80_a = float(np.percentile(a_list, 80))
            # If an animal stays reared throughout a short clip while its partner walks horizontally,
            # incorporate cohort open-posture scale so cos(theta) foreshortening remains calibrated.
            ref_area_by_inst[inst_k] = max(inst_p80_a, 0.85 * cohort_ref_area)
            ref_elong_by_inst[inst_k] = max(
                1.95, float(np.percentile(elongs_by_inst[inst_k], 80))
            )

        for row, cues in zip(kin_rows, raw_posture_cues):
            inst_k = str(row["instance_name"])
            area_v = float(row["whole_area_px"])
            elong_v = float(row["elongation_ratio"])
            lm_v = float(row["limbs_area_ratio"])
            defl_v = float(row["head_body_deflection_deg"])
            defl_abs = abs(defl_v) if math.isfinite(defl_v) else 0.0
            span_r = float(cues["head_tail_span_ratio"])
            up_sin = float(cues["upward_pitch_sin"])
            kp_lift = float(cues["kp_bipedal_lift"])

            ref_a = max(1.0, ref_area_by_inst[inst_k])
            ref_e = max(1.5, ref_elong_by_inst[inst_k])
            area_contract = float(
                np.clip((ref_a - area_v) / max(1.0, 0.36 * ref_a), 0.0, 1.0)
            )
            elong_contract = float(
                np.clip((ref_e - elong_v) / max(0.50, 0.42 * ref_e), 0.0, 1.0)
            )

            # 1. Lateral spinal flexion / C-curl grooming score (handles trailing tail via trunk_elong & spinal bend).
            eff_bend = max(defl_abs, float(cues["kp_spinal_bend_deg"]))
            bend_score = float(np.clip((eff_bend - 26.0) / 52.0, 0.0, 1.0))
            fold_score = (
                float(np.clip((0.66 - span_r) / 0.34, 0.0, 1.0))
                if math.isfinite(span_r)
                else 0.0
            )
            trunk_ball_score = (
                float(np.clip((1.68 - float(cues["trunk_elong"])) / 0.42, 0.0, 1.0))
                * float(
                    np.clip(
                        (0.46 - float(cues["head_torso_reach_ratio"])) / 0.22,
                        0.25,
                        1.0,
                    )
                )
                * float(np.clip(1.0 - 1.5 * area_contract, 0.0, 1.0))
            )
            if view_mode in {"oblique", "side"} or lm_v >= 0.035:
                # Upright bipedal pitch foreshortens dorsal head-tail distance without lateral C-curl.
                upright_damp = float(
                    np.clip(max(up_sin - 0.18, kp_lift * 1.40) / 0.40, 0.0, 1.0)
                )
                bend_score *= 1.0 - 0.78 * upright_damp
                fold_score *= 1.0 - 0.82 * upright_damp
                trunk_ball_score *= 1.0 - 0.85 * upright_damp
            curl_idx = float(
                np.clip(
                    max(
                        bend_score,
                        0.72 * fold_score + 0.35 * bend_score,
                        0.75 * trunk_ball_score + 0.35 * bend_score,
                    ),
                    0.0,
                    1.0,
                )
            )

            # 2. Curl-suppressed upright rearing score combining projective cos(theta) foreshortening
            #    and oblique/lateral anti-gravity vertical alignment.
            axial_gate = float(np.clip(1.0 - 0.85 * curl_idx, 0.05, 1.0))
            top_rear = axial_gate * (0.58 * area_contract + 0.42 * elong_contract)
            if (
                area_contract < 0.05
                and elong_v < 1.48
                and defl_abs < 22.0
                and curl_idx < 0.15
            ):
                top_rear = max(top_rear, 0.45 * elong_contract)

            bipedal_gate = float(
                np.clip(
                    max(
                        np.clip((lm_v - 0.03) / 0.07, 0.0, 1.0),
                        np.clip(kp_lift / 0.35, 0.0, 1.0),
                        0.85 * elong_contract if view_mode == "oblique" else 0.0,
                    ),
                    0.0,
                    1.0,
                )
            )
            oblique_rear = (
                (1.0 - 0.70 * bend_score)
                * max(up_sin, kp_lift)
                * (0.35 + 0.65 * bipedal_gate)
                * float(np.clip((2.45 - elong_v) / 0.95, 0.25, 1.0))
            )

            if view_mode == "top":
                rear_idx = top_rear
            elif view_mode == "oblique":
                rear_idx = max(oblique_rear, 0.65 * top_rear)
            elif view_mode == "bottom":
                rear_idx = axial_gate * (0.70 * area_contract + 0.30 * elong_contract)
            else:
                rear_idx = max(top_rear, oblique_rear)
            rear_idx = float(np.clip(rear_idx, 0.0, 1.0))

            if rear_idx >= 0.46 and rear_idx >= curl_idx:
                p_state = "rearing"
            elif curl_idx >= 0.46:
                p_state = "curled_grooming"
            elif elong_v < 1.65:
                p_state = "stationary_crouch"
            else:
                p_state = "locomotion"

            row["curl_groom_index"] = round(curl_idx, 4)
            row["rearing_index"] = round(rear_idx, 4)
            row["posture_state"] = p_state

    kinematics_df = pd.DataFrame(kin_rows)
    contacts_df = pd.DataFrame(contact_rows)

    bouts: List[SocialContactBout] = []
    if not contacts_df.empty:
        behavior_specs = (
            ("nose_to_nose", "head_to_head_px", True),
            ("anogenital_sniff", "head_to_tail_px", False),
            ("side_by_side_huddle", "torso_to_torso_px", True),
        )
        for (actor, recipient), pair_df in contacts_df.groupby(
            ["actor", "recipient"], sort=True
        ):
            ordered_p = pair_df.sort_values("frame_number")
            fnums = ordered_p["frame_number"].to_numpy(dtype=np.int64)
            for beh_col, dist_col, symmetric in behavior_specs:
                if symmetric and str(actor) > str(recipient):
                    continue
                flags = ordered_p[beh_col].to_numpy(dtype=bool)
                dists = ordered_p[dist_col].to_numpy(dtype=np.float64)
                run_start = -1
                run_end = -1
                run_len = 0
                run_min_d = float("inf")
                for idx_r, active in enumerate(flags):
                    fn = int(fnums[idx_r])
                    dv = float(dists[idx_r]) if math.isfinite(dists[idx_r]) else 0.0
                    if active and (run_start < 0 or fn - run_end <= 2):
                        if run_start < 0:
                            run_start = fn
                        run_end = fn
                        run_len += 1
                        run_min_d = min(run_min_d, dv)
                    else:
                        if run_start >= 0 and run_len >= min_len:
                            dur_f = int(run_end - run_start + 1)
                            bouts.append(
                                SocialContactBout(
                                    actor=str(actor),
                                    recipient=str(recipient),
                                    behavior=beh_col,
                                    start_frame=run_start,
                                    end_frame=run_end,
                                    duration_frames=dur_f,
                                    duration_s=round(float(dur_f) / eff_fps, 4),
                                    min_distance_px=round(run_min_d, 3),
                                )
                            )
                        if active:
                            run_start = fn
                            run_end = fn
                            run_len = 1
                            run_min_d = dv
                        else:
                            run_start = -1
                            run_end = -1
                            run_len = 0
                            run_min_d = float("inf")
                if run_start >= 0 and run_len >= min_len:
                    dur_f = int(run_end - run_start + 1)
                    bouts.append(
                        SocialContactBout(
                            actor=str(actor),
                            recipient=str(recipient),
                            behavior=beh_col,
                            start_frame=run_start,
                            end_frame=run_end,
                            duration_frames=dur_f,
                            duration_s=round(float(dur_f) / eff_fps, 4),
                            min_distance_px=round(run_min_d, 3),
                        )
                    )

    bouts.sort(key=lambda b: (b.start_frame, b.behavior, b.actor, b.recipient))
    uncertainty_candidates.sort(key=lambda item: (-item[0], item[1]))
    uncertain_frames = [
        fn for _u, fn in uncertainty_candidates[: max(1, int(max_uncertain_frames))]
    ]

    return BodyPartEthogramReport(
        kinematics_df=kinematics_df,
        contacts_df=contacts_df,
        bouts=bouts,
        uncertain_frames=uncertain_frames,
        fps=eff_fps,
    )


def export_body_part_ethogram_csvs(
    report: BodyPartEthogramReport,
    output_dir: Path | str,
    *,
    stem: str = "body_part",
) -> Dict[str, Any]:
    """Export kinematics, directed contact, and bout CSVs plus JSON summary."""
    from annolid.utils.atomic_io import atomic_write_json

    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    clean_stem = str(stem or "body_part").strip() or "body_part"

    kin_csv = out_root / f"{clean_stem}_body_part_kinematics.csv"
    con_csv = out_root / f"{clean_stem}_body_part_contacts.csv"
    bouts_csv = out_root / f"{clean_stem}_body_part_bouts.csv"
    eth_json = out_root / f"{clean_stem}_body_part_ethogram.json"

    report.kinematics_df.to_csv(kin_csv, index=False)
    report.contacts_df.to_csv(con_csv, index=False)
    bouts_df = pd.DataFrame(
        [asdict(b) for b in report.bouts],
        columns=[
            "actor",
            "recipient",
            "behavior",
            "start_frame",
            "end_frame",
            "duration_frames",
            "duration_s",
            "min_distance_px",
        ],
    )
    bouts_df.to_csv(bouts_csv, index=False)

    summary = report.to_summary_dict()
    summary["kinematics_csv"] = str(kin_csv)
    summary["contacts_csv"] = str(con_csv)
    summary["bouts_csv"] = str(bouts_csv)
    summary["ethogram_json"] = str(eth_json)
    atomic_write_json(eth_json, summary, indent=2, ensure_ascii=False)
    return summary
