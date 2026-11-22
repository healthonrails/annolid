from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from annolid.utils.logger import logger


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            yield payload


def _point_from_shape(shape: Dict[str, Any]) -> Tuple[float, float, float] | None:
    if str(shape.get("shape_type") or "").lower() != "point":
        return None
    points = shape.get("points")
    if not isinstance(points, list) or not points:
        return None
    first = points[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        return None
    try:
        return float(first[0]), float(first[1]), 0.0
    except (TypeError, ValueError):
        return None


def _extract_frame_points(
    record: Dict[str, Any],
) -> Dict[str, Tuple[float, float, float]]:
    simulation = (record.get("otherData") or {}).get("simulation") or {}
    state = simulation.get("state") or {}
    site_targets = state.get("site_targets") or {}
    points: Dict[str, Tuple[float, float, float]] = {}
    if isinstance(site_targets, dict):
        for label, coords in site_targets.items():
            if not isinstance(coords, (list, tuple)) or len(coords) < 3:
                continue
            try:
                points[str(label)] = (
                    float(coords[0]),
                    float(coords[1]),
                    float(coords[2]),
                )
            except (TypeError, ValueError):
                continue
    if points:
        return points

    for shape in record.get("shapes") or []:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("display_label") or shape.get("label") or "").strip()
        if not label:
            continue
        point = _point_from_shape(shape)
        if point is None:
            continue
        points[label] = point
    return points


def _extract_edges(metadata: Dict[str, Any]) -> List[List[str]]:
    candidate = metadata.get("viewer_edges") or metadata.get("skeleton_edges") or []
    if not isinstance(candidate, list):
        return []
    edges: List[List[str]] = []
    for item in candidate:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        left = str(item[0]).strip()
        right = str(item[1]).strip()
        if left and right:
            edges.append([left, right])
    return edges


def _infer_edges_from_labels(labels: List[str]) -> List[List[str]]:
    by_norm = {str(label).strip().lower(): str(label) for label in labels}
    edges: List[List[str]] = []

    def add(left_candidates: List[str], right_candidates: List[str]) -> None:
        left = next(
            (by_norm[name] for name in left_candidates if name in by_norm), None
        )
        right = next(
            (by_norm[name] for name in right_candidates if name in by_norm), None
        )
        if left and right and left != right:
            edge = [left, right]
            if edge not in edges and edge[::-1] not in edges:
                edges.append(edge)

    add(["nose", "head", "head_site"], ["thorax", "thorax_site"])
    add(["thorax", "thorax_site"], ["abdomen", "abdomen_tip", "abdomen_tip_site"])
    add(["head", "head_site"], ["left_antenna", "left_antenna_site", "antenna_left"])
    add(["head", "head_site"], ["right_antenna", "right_antenna_site", "antenna_right"])
    for seg, prefix in (
        ("t1_left", "left_front"),
        ("t1_right", "right_front"),
        ("t2_left", "left_middle"),
        ("t2_right", "right_middle"),
        ("t3_left", "left_hind"),
        ("t3_right", "right_hind"),
    ):
        add(
            ["thorax", "thorax_site"],
            [
                f"tarsus_{seg}",
                f"{prefix}_leg_tip",
                f"{prefix}_tarsus_site",
                f"claw_{seg}",
            ],
        )
        add([f"tarsus_{seg}", f"{prefix}_tarsus_site"], [f"claw_{seg}"])
    if not edges and len(labels) >= 2:
        from annolid.annotation.pose_schema import infer_pose_skeleton_edges

        edges = [[a, b] for a, b in infer_pose_skeleton_edges(labels, preset="auto")]
    return edges


def build_simulation_view_payload(path: str | Path) -> Dict[str, Any]:
    source = Path(path).expanduser()
    frames: List[Dict[str, Any]] = []
    labels: List[str] = []
    label_seen: set[str] = set()
    adapter_name = ""
    run_metadata: Dict[str, Any] = {}
    mapping_metadata: Dict[str, Any] = {}
    title = source.stem

    for record in _iter_records(source):
        simulation = (record.get("otherData") or {}).get("simulation") or {}
        adapter_name = str(simulation.get("adapter") or adapter_name or "").strip()
        run_metadata = dict(simulation.get("run_metadata") or run_metadata or {})
        mapping_metadata = dict(
            simulation.get("mapping_metadata") or mapping_metadata or {}
        )
        if not title:
            title = str(
                record.get("video_name") or record.get("videoName") or source.stem
            )

        point_map = _extract_frame_points(record)
        frame_points: List[Dict[str, Any]] = []
        for label, coords in point_map.items():
            frame_points.append(
                {
                    "label": label,
                    "x": coords[0],
                    "y": coords[1],
                    "z": coords[2],
                }
            )
            if label not in label_seen:
                label_seen.add(label)
                labels.append(label)

        state = dict(simulation.get("state") or {})
        diagnostics = dict(simulation.get("diagnostics") or {})
        frame_entry: Dict[str, Any] = {
            "frame_index": int(record.get("frame_index") or record.get("frame") or 0),
            "timestamp_sec": record.get("timestamp_sec"),
            "points": frame_points,
            "qpos": list(state.get("qpos") or []),
            "diagnostics": diagnostics,
            "dry_run": bool(state.get("dry_run", False)),
        }
        if isinstance(state.get("body_poses"), dict):
            frame_entry["body_poses"] = dict(state["body_poses"])
        if isinstance(state.get("model_pose"), dict):
            frame_entry["model_pose"] = dict(state["model_pose"])
        frames.append(frame_entry)

    metadata = dict(mapping_metadata.get("metadata") or {})
    coordinate_system = dict(mapping_metadata.get("coordinate_system") or {})
    explicit_edges = _extract_edges(metadata)
    return {
        "kind": "annolid-simulation-v1",
        "title": title or source.stem,
        "adapter": adapter_name,
        "labels": labels,
        "edges": explicit_edges or _infer_edges_from_labels(labels),
        "metadata": {
            "run_metadata": run_metadata,
            "mapping_metadata": mapping_metadata,
            "coordinate_system": coordinate_system,
        },
        "frames": frames,
    }


def _attach_flybody_viewer_assets(
    payload: Dict[str, Any], target_dir: Path
) -> Dict[str, Any]:
    adapter = str(payload.get("adapter") or "").strip().lower()
    if not adapter.startswith("flybody"):
        return payload
    frames = payload.get("frames") or []
    has_body_poses = any(
        isinstance(frame, dict) and bool(frame.get("body_poses")) for frame in frames
    )
    has_model_pose = any(
        isinstance(frame, dict) and bool(frame.get("model_pose")) for frame in frames
    )
    if not has_body_poses and not has_model_pose:
        return payload
    try:
        from annolid.gui.threejs_examples import (
            attach_flybody_floor,
            attach_flybody_mesh,
            attach_flybody_mesh_parts,
        )
    except Exception:
        return payload

    enriched = dict(payload)
    if has_body_poses:
        enriched = attach_flybody_mesh_parts(enriched, target_dir)
    if not enriched.get("mesh"):
        enriched = attach_flybody_mesh(enriched, target_dir)
    enriched = attach_flybody_floor(enriched)
    floor = (enriched.get("environment") or {}).get("floor") or {}
    model_scale = float(
        ((frames or [{}])[0].get("model_pose") or {}).get("scale") or 7.5
    )
    if isinstance(floor.get("position"), list) and len(floor["position"]) >= 3:
        floor_pos = floor["position"]
        normalized_floor = dict(floor)
        normalized_floor["position"] = [
            0.0,
            float(floor_pos[2]) * model_scale,
            0.0,
        ]
        enriched["environment"] = dict(enriched.get("environment") or {})
        enriched["environment"]["floor"] = normalized_floor
    if enriched.get("mesh"):
        enriched.setdefault("display", {})
        enriched["display"].update(
            {
                "show_points": False,
                "show_labels": False,
                "show_edges": False,
                "show_trails": False,
            }
        )
    return enriched


def export_simulation_view_payload(
    source_path: str | Path, *, out_dir: str | Path | None = None
) -> Path:
    source = Path(source_path).expanduser()
    target_dir = (
        Path(out_dir).expanduser()
        if out_dir is not None
        else Path(tempfile.gettempdir()) / "annolid_simulation_viewer"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{source.stem}.simulation-view.json"
    if out_path.exists():
        try:
            if out_path.stat().st_mtime_ns >= source.stat().st_mtime_ns:
                logger.info("Reusing cached simulation viewer payload: %s", out_path)
                return out_path
        except OSError:
            pass
    started = time.perf_counter()
    payload = build_simulation_view_payload(source)
    payload = _attach_flybody_viewer_assets(payload, target_dir)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    logger.info(
        "Exported simulation viewer payload to %s in %.1fms",
        out_path,
        (time.perf_counter() - started) * 1000.0,
    )
    return out_path
