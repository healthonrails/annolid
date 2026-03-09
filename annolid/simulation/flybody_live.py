from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def probe_flybody_runtime() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": __import__("sys").executable,
        "ready": False,
    }
    try:
        importlib.import_module("flybody")
        importlib.import_module("dm_control")
        importlib.import_module("mujoco")
        fly_envs = importlib.import_module("flybody.fly_envs")
        env = getattr(fly_envs, "walk_imitation")()
        payload["environment_factory"] = "flybody.fly_envs:walk_imitation"
        payload["action_size"] = int(np.prod(env.action_spec().shape, dtype=int))
        payload["ready"] = True
    except Exception as exc:
        payload["error"] = f"{exc.__class__.__name__}: {exc}"
    return payload


def _candidate_site_names(physics: Any) -> list[str]:
    axis = physics.named.model.site_pos.axes.row
    names = []
    for attr in ("_names_to_indices", "_names_to_offsets", "_names_to_slices"):
        mapping = getattr(axis, attr, None)
        if isinstance(mapping, dict):
            names.extend(str(name) for name in mapping.keys())
    seen: set[str] = set()
    selected: list[str] = []
    preferred_terms = (
        "head_site",
        "antenna",
        "thorax_site",
        "abdomen_tip_site",
        "tarsus",
        "claw",
    )
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        stripped = name.split("/", 1)[-1]
        if any(term in stripped for term in preferred_terms):
            selected.append(name)
    return selected


def _candidate_body_names(physics: Any) -> list[str]:
    axis = physics.named.data.xpos.axes.row
    names = []
    for attr in ("_names_to_indices", "_names_to_offsets", "_names_to_slices"):
        mapping = getattr(axis, attr, None)
        if isinstance(mapping, dict):
            names.extend(str(name) for name in mapping.keys())
    seen: set[str] = set()
    selected: list[str] = []
    preferred_terms = (
        "thorax",
        "head",
        "antenna",
        "wing_left",
        "wing_right",
        "abdomen",
        "coxa_",
        "femur_",
        "tibia_",
        "tarsus_",
        "tarsal_claw_",
    )
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        stripped = name.split("/", 1)[-1]
        if any(term in stripped for term in preferred_terms):
            selected.append(name)
    return selected


def _label_for_site(name: str) -> str:
    return str(name or "").split("/", 1)[-1]


def _edges_for_labels(labels: list[str]) -> list[list[str]]:
    edges: list[list[str]] = []

    def lookup(options: list[str]) -> str | None:
        for candidate in options:
            for label in labels:
                if label == candidate:
                    return label
        return None

    def add(left: list[str], right: list[str]) -> None:
        a = lookup(left)
        b = lookup(right)
        if a and b and a != b and [a, b] not in edges and [b, a] not in edges:
            edges.append([a, b])

    add(["head_site"], ["thorax_site"])
    add(["thorax_site"], ["abdomen_tip_site"])
    add(["head_site"], ["left_antenna_site"])
    add(["head_site"], ["right_antenna_site"])
    for prefix in (
        "left_front",
        "right_front",
        "left_middle",
        "right_middle",
        "left_hind",
        "right_hind",
    ):
        add(["thorax_site"], [f"{prefix}_tarsus_site", f"{prefix}_tarsal_claw_site"])
    return edges


def _yaw_from_points(head: np.ndarray | None, thorax: np.ndarray | None) -> float:
    if head is None or thorax is None:
        return 0.0
    vec = np.asarray(head) - np.asarray(thorax)
    return float(math.atan2(vec[1], vec[0]))


def generate_live_rollout_payload(
    *,
    steps: int = 180,
    seed: int = 7,
    title: str = "FlyBody Live Rollout",
) -> dict[str, Any]:
    fly_envs = importlib.import_module("flybody.fly_envs")
    env = getattr(fly_envs, "walk_imitation")()
    timestep = env.reset()
    _ = timestep
    physics = getattr(env, "physics", env)
    action_spec = env.action_spec()
    minimum = np.asarray(action_spec.minimum, dtype=float)
    maximum = np.asarray(action_spec.maximum, dtype=float)
    center = (minimum + maximum) * 0.5
    span = (maximum - minimum) * 0.35
    site_names = _candidate_site_names(physics)
    body_names = _candidate_body_names(physics)
    labels = [_label_for_site(name) for name in site_names]
    frames: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    phase_offsets = rng.uniform(0.0, math.tau, size=action_spec.shape[0])

    for index in range(int(steps)):
        data = physics.named.data
        points = []
        point_map: dict[str, np.ndarray] = {}
        for site_name in site_names:
            coords = np.asarray(data.site_xpos[site_name], dtype=float)
            label = _label_for_site(site_name)
            point_map[label] = coords
            points.append(
                {
                    "label": label,
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "z": float(coords[2]),
                }
            )

        thorax = point_map.get("thorax_site")
        head = point_map.get("head_site")
        model_pos = thorax if thorax is not None else np.zeros(3, dtype=float)
        qpos = np.asarray(physics.data.qpos, dtype=float)
        body_poses = {}
        for body_name in body_names:
            label = _label_for_site(body_name)
            position = np.asarray(data.xpos[body_name], dtype=float)
            quaternion = np.asarray(data.xquat[body_name], dtype=float)
            body_poses[label] = {
                "position": [
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                ],
                "quaternion": [
                    float(quaternion[0]),
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                ],
            }
        frames.append(
            {
                "frame_index": index,
                "timestamp_sec": round(index / 60.0, 4),
                "points": points,
                "qpos": qpos.tolist(),
                "diagnostics": {"mode": "live_rollout"},
                "dry_run": False,
                "body_poses": body_poses,
                "model_pose": {
                    "position": [
                        float(model_pos[0]),
                        float(model_pos[1]),
                        float(model_pos[2]),
                    ],
                    "rotation": [0.0, 0.0, _yaw_from_points(head, thorax)],
                    "scale": 7.5,
                },
            }
        )
        phase = (index + 1) / 18.0
        action = center + span * np.sin(phase + phase_offsets)
        action = np.clip(action, minimum, maximum)
        env.step(action.astype(action_spec.dtype))

    return {
        "kind": "annolid-simulation-v1",
        "title": title,
        "adapter": "flybody-live",
        "labels": labels,
        "edges": _edges_for_labels(labels),
        "metadata": {
            "run_metadata": {"source": "live_rollout", "steps": int(steps)},
            "mapping_metadata": {"metadata": {"template": "flybody-live"}},
            "coordinate_system": {"units": "meters"},
        },
        "frames": frames,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe or generate FlyBody live rollout payloads."
    )
    parser.add_argument(
        "--probe", action="store_true", help="Probe runtime readiness and print JSON."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument(
        "--out", type=str, help="Output JSON path for live rollout payload."
    )
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.probe:
        payload = probe_flybody_runtime()
        if args.json:
            print(json.dumps(payload))
        else:
            print(payload)
        return 0 if payload.get("ready") else 1
    if not args.out:
        raise SystemExit("--out is required unless --probe is used")
    payload = generate_live_rollout_payload(steps=args.steps, seed=args.seed)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
