from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from annolid.simulation.fly_brain import (
    BEHAVIOR_ENV_FACTORY_NAMES,
    CONNECTOME_RELEASE_METADATA,
    SUPPORTED_BRAIN_DYNAMICS,
    SUPPORTED_FLYBODY_BEHAVIORS,
    FlyBrainController,
    build_canonical_fly_connectome,
    build_spiking_fly_connectome,
    ensure_calibrated_wingbeat_path,
    load_connectome_graph,
)


LIVE_PAYLOAD_VERSION = 3


def probe_flybody_runtime() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": __import__("sys").executable,
        "ready": False,
        "connectome_id": CONNECTOME_RELEASE_METADATA["id"],
        "supported_behaviors": list(SUPPORTED_FLYBODY_BEHAVIORS),
    }
    env = None
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
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
    return payload


def _call_factory_with_supported_kwargs(factory: Any, **preferred_kwargs: Any) -> Any:
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()
    accepted: dict[str, Any] = {}
    has_var_kw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    for key, value in preferred_kwargs.items():
        if has_var_kw or key in sig.parameters:
            accepted[key] = value
    return factory(**accepted)


def _build_behavior_env(
    behavior: str, *, wpg_pattern_path: str | Path | None = None
) -> Any:
    behavior_key = str(behavior)
    if behavior_key not in BEHAVIOR_ENV_FACTORY_NAMES:
        raise ValueError(f"Unsupported FlyBody behavior: {behavior}")
    fly_envs = importlib.import_module("flybody.fly_envs")
    factory_attr = BEHAVIOR_ENV_FACTORY_NAMES[behavior_key]
    factory = getattr(fly_envs, factory_attr, None)
    if not callable(factory):
        raise ValueError(
            f"FlyBody environment factory {factory_attr!r} not available for {behavior_key!r}"
        )
    if behavior_key == "walk_imitation":
        return _call_factory_with_supported_kwargs(factory, terminal_com_dist=10.0)
    if behavior_key == "flight_imitation":
        resolved_wpg = str(
            Path(wpg_pattern_path).expanduser()
            if wpg_pattern_path
            else ensure_calibrated_wingbeat_path()
        )
        return _call_factory_with_supported_kwargs(
            factory,
            wpg_pattern_path=resolved_wpg,
            terminal_com_dist=50.0,
        )
    if behavior_key == "vision_guided_flight":
        resolved_wpg = str(
            Path(wpg_pattern_path).expanduser()
            if wpg_pattern_path
            else ensure_calibrated_wingbeat_path()
        )
        return _call_factory_with_supported_kwargs(
            factory,
            wpg_pattern_path=resolved_wpg,
        )
    if behavior_key == "escape_takeoff":
        return _call_factory_with_supported_kwargs(
            factory, disable_wings=False, time_limit=10.0
        )
    if behavior_key in ("forage_walk", "backward_walk", "grooming", "template_task"):
        return _call_factory_with_supported_kwargs(factory, time_limit=10.0)
    return factory()


def _extend_inference_trajectory_if_needed(
    env: Any, *, behavior: str, min_control_steps: int
) -> None:
    """Extend synthetic Inference*TrajectoryLoader horizons so rollouts do not teleport mid-clip."""
    task = getattr(env, "task", None)
    traj_gen = getattr(task, "_traj_generator", None)
    set_next = getattr(traj_gen, "set_next_trajectory", None)
    if not callable(set_next):
        return
    cls_name = type(traj_gen).__name__
    if cls_name not in (
        "InferenceWalkingTrajectoryLoader",
        "InferenceFlightTrajectoryLoader",
    ):
        return
    try:
        loaders = importlib.import_module("flybody.tasks.trajectory_loaders")
        const_traj = getattr(loaders, "constant_speed_trajectory", None)
        if not callable(const_traj):
            return
        dt_fn = getattr(env, "control_timestep", None)
        dt = float(dt_fn()) if callable(dt_fn) else 0.002
        target_steps = max(2500, int(min_control_steps) + 160)
        if (
            cls_name == "InferenceFlightTrajectoryLoader"
            or str(behavior) == "flight_imitation"
        ):
            qpos, qvel = const_traj(
                n_steps=target_steps,
                speed=20.0,
                init_pos=(0.0, 0.0, 1.0),
                body_rot_angle_y=-47.5,
                control_timestep=dt,
            )
        else:
            qpos, qvel = const_traj(
                n_steps=target_steps,
                speed=2.0,
                init_pos=(0.0, 0.0, 0.1278),
                control_timestep=dt,
            )
        set_next(qpos, qvel)
        needed_sec = float(target_steps) * dt
        if (
            hasattr(env, "_time_limit")
            and float(getattr(env, "_time_limit")) < needed_sec
        ):
            setattr(env, "_time_limit", needed_sec)
    except Exception:
        return


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
        "head",
        "antenna",
        "thorax",
        "abdomen",
        "tarsus",
        "claw",
    )
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if not str(name).startswith("walker/"):
            continue
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
        "rostrum",
        "haustellum",
        "labrum",
        "antenna",
        "wing_left",
        "wing_right",
        "haltere",
        "abdomen",
        "coxa_",
        "femur_",
        "tibia_",
        "tarsus",
        "claw_",
        "tarsal_claw_",
    )
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if not str(name).startswith("walker/"):
            continue
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

    add(["head", "head_site"], ["thorax", "thorax_site"])
    add(["thorax", "thorax_site"], ["abdomen_tip_site", "abdomen"])
    add(["head", "head_site"], ["left_antenna_site", "antenna_left"])
    add(["head", "head_site"], ["right_antenna_site", "antenna_right"])

    for seg, legacy in (
        ("T1_left", "left_front"),
        ("T1_right", "right_front"),
        ("T2_left", "left_middle"),
        ("T2_right", "right_middle"),
        ("T3_left", "left_hind"),
        ("T3_right", "right_hind"),
    ):
        add(
            ["thorax", "thorax_site"],
            [
                f"tarsus_{seg}",
                f"{legacy}_tarsus_site",
                f"claw_{seg}",
                f"{legacy}_tarsal_claw_site",
            ],
        )
        add(
            [f"tarsus_{seg}", f"{legacy}_tarsus_site"],
            [f"claw_{seg}", f"{legacy}_tarsal_claw_site"],
        )
    return edges


def _yaw_from_points(head: np.ndarray | None, thorax: np.ndarray | None) -> float:
    if head is None or thorax is None:
        return 0.0
    vec = np.asarray(head) - np.asarray(thorax)
    return float(math.atan2(vec[1], vec[0]))


def _yaw_from_root_quaternion(
    quaternion_wxyz: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape[0] < 4:
        return 0.0
    w, x, y, z = q[:4]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _reference_root_qpos(task: Any, index: int) -> np.ndarray | None:
    ref = getattr(task, "_ref_qpos", None)
    if (
        isinstance(ref, np.ndarray)
        and ref.ndim == 2
        and ref.shape[0] > 0
        and ref.shape[1] >= 7
    ):
        safe_index = min(max(int(index), 0), ref.shape[0] - 1)
        return np.asarray(ref[safe_index, :7], dtype=float)
    return None


def _sample_repo_style_action(action_spec: Any, rng: np.random.Generator) -> np.ndarray:
    """Sample seeded Gaussian random actions clipped to spec (legacy debug mode)."""
    action = rng.normal(size=action_spec.shape).astype(action_spec.dtype)
    minimum = np.asarray(action_spec.minimum, dtype=float)
    maximum = np.asarray(action_spec.maximum, dtype=float)
    return np.clip(action, minimum, maximum).astype(action_spec.dtype)


def _thorax_pose(physics: Any) -> tuple[np.ndarray, np.ndarray]:
    data = physics.named.data
    xpos_axis = getattr(getattr(data, "xpos", None), "axes", None)
    row_axis = getattr(xpos_axis, "row", None)
    has_thorax = False
    for attr in ("_names_to_indices", "_names_to_offsets", "_names_to_slices"):
        mapping = getattr(row_axis, attr, None)
        if isinstance(mapping, dict) and "walker/thorax" in mapping:
            has_thorax = True
            break
    if has_thorax:
        pos = np.asarray(data.xpos["walker/thorax"], dtype=float).copy()
        quat = np.asarray(data.xquat["walker/thorax"], dtype=float).copy()
        return pos, quat
    qpos = np.asarray(physics.data.qpos, dtype=float)
    pos = qpos[:3].copy() if qpos.size >= 3 else np.zeros(3, dtype=float)
    quat = (
        qpos[3:7].copy()
        if qpos.size >= 7
        else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    )
    return pos, quat


def _default_substeps_per_frame(
    env: Any, substeps_per_frame: int | None
) -> tuple[int, float]:
    dt_fn = getattr(env, "control_timestep", None)
    if not callable(dt_fn):
        return max(1, int(substeps_per_frame or 1)), 1.0 / 60.0
    dt = max(1e-5, float(dt_fn()))
    if substeps_per_frame is not None:
        return max(1, int(substeps_per_frame)), dt
    if dt <= 0.0005:
        return 8, dt
    return 6, dt


def generate_live_rollout_payload(
    *,
    steps: int = 180,
    seed: int = 7,
    title: str = "FlyBody Live Rollout",
    behavior: str = "walk_imitation",
    policy_dir: str | None = None,
    controller: str = "connectome",
    dynamics: str = "spiking",
    connectome_path: str | Path | None = None,
    wpg_pattern_path: str | Path | None = None,
    substeps_per_frame: int | None = None,
) -> dict[str, Any]:
    controller_mode = str(controller or "connectome").strip().lower()
    if controller_mode not in ("connectome", "random"):
        raise ValueError(
            f"Unsupported FlyBody controller {controller!r}; expected 'connectome' or 'random'"
        )
    dynamics_mode = str(dynamics or "spiking").strip().lower()
    if dynamics_mode not in SUPPORTED_BRAIN_DYNAMICS:
        raise ValueError(
            f"Unsupported brain dynamics {dynamics!r}; expected one of {SUPPORTED_BRAIN_DYNAMICS}"
        )

    env = (
        _build_behavior_env(behavior, wpg_pattern_path=wpg_pattern_path)
        if wpg_pattern_path is not None
        else _build_behavior_env(behavior)
    )
    try:
        substeps_hint = int(substeps_per_frame or 8)
        _extend_inference_trajectory_if_needed(
            env,
            behavior=behavior,
            min_control_steps=max(1, int(steps)) * substeps_hint,
        )
        timestep = env.reset()
        substeps, control_dt = _default_substeps_per_frame(env, substeps_per_frame)

        policy = None
        active_controller_name = controller_mode
        if policy_dir:
            try:
                tf = importlib.import_module("tensorflow")
                utils_tf = importlib.import_module("flybody.agents.utils_tf")
                loaded_policy = tf.saved_model.load(policy_dir)
                policy = utils_tf.TestPolicyWrapper(loaded_policy)
                active_controller_name = "saved_policy"
            except Exception as exc:
                print(f"Warning: Failed to load policy from {policy_dir}: {exc}")

        action_spec = env.action_spec()
        connectome = None
        brain_controller: FlyBrainController | None = None
        if policy is None and controller_mode == "connectome":
            if connectome_path:
                connectome = load_connectome_graph(connectome_path)
            elif dynamics_mode == "spiking":
                connectome = build_spiking_fly_connectome(seed=seed)
            else:
                connectome = build_canonical_fly_connectome()
            brain_controller = FlyBrainController(
                action_spec,
                behavior=behavior,
                connectome=connectome,
                dynamics=dynamics_mode,
                seed=seed,
                control_timestep=control_dt,
            )

        def _physics() -> Any:
            return getattr(env, "physics", env)

        physics = _physics()
        initial_thorax_pos, _initial_thorax_quat = _thorax_pose(physics)
        is_aerial = str(behavior) in ("flight_imitation", "vision_guided_flight")
        # Keep aerial flights elevated above the floor plane while centering X/Y at origin
        world_origin = np.asarray(
            [
                float(initial_thorax_pos[0]),
                float(initial_thorax_pos[1]),
                float(initial_thorax_pos[2] - (0.28 if is_aerial else 0.0)),
            ],
            dtype=float,
        )

        site_names = _candidate_site_names(physics)
        body_names = _candidate_body_names(physics)
        labels = [_label_for_site(name) for name in site_names]
        frames: list[dict[str, Any]] = []
        rng = np.random.default_rng(seed)
        sim_time_sec = 0.0

        for index in range(int(steps)):
            # Step closed-loop brain + body physics before recording frame > 0
            if index > 0:
                for _ in range(substeps):
                    obs = getattr(timestep, "observation", {}) or {}
                    if policy is not None:
                        action = policy(obs)
                    elif brain_controller is not None:
                        action = brain_controller.step(obs, dt=control_dt)
                    else:
                        action = _sample_repo_style_action(action_spec, rng)
                    timestep = env.step(action)
                    sim_time_sec += control_dt
                    last_fn = getattr(timestep, "last", None)
                    if callable(last_fn) and last_fn():
                        timestep = env.reset()
                        if brain_controller is not None:
                            brain_controller.reset()
            elif brain_controller is not None:
                # Equilibrate sensory->DN->VNC firing rates on initial observation at t=0
                obs0 = getattr(timestep, "observation", {}) or {}
                brain_controller.warmup(obs0, duration_sec=0.04)

            physics = _physics()
            data = physics.named.data
            points = []
            for site_name in site_names:
                coords = (
                    np.asarray(data.site_xpos[site_name], dtype=float) - world_origin
                )
                label = _label_for_site(site_name)
                points.append(
                    {
                        "label": label,
                        "x": float(coords[0]),
                        "y": float(coords[1]),
                        "z": float(coords[2]),
                    }
                )

            qpos = np.asarray(physics.data.qpos, dtype=float)
            thorax_pos, thorax_quat = _thorax_pose(physics)
            model_pos = thorax_pos - world_origin

            body_poses = {}
            for body_name in body_names:
                label = _label_for_site(body_name)
                position = np.asarray(data.xpos[body_name], dtype=float) - world_origin
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

            frame_diagnostics: dict[str, Any] = {
                "mode": "live_rollout",
                "controller": active_controller_name,
            }
            if brain_controller is not None:
                frame_diagnostics["brain"] = brain_controller.diagnostics()

            frames.append(
                {
                    "frame_index": index,
                    "timestamp_sec": round(sim_time_sec, 5),
                    "points": points,
                    "qpos": qpos.tolist(),
                    "diagnostics": frame_diagnostics,
                    "dry_run": False,
                    "body_poses": body_poses,
                    "model_pose": {
                        "position": [
                            float(model_pos[0]),
                            float(model_pos[1]),
                            float(model_pos[2]),
                        ],
                        "rotation": [0.0, 0.0, _yaw_from_root_quaternion(thorax_quat)],
                        "scale": 7.5,
                    },
                }
            )

        run_metadata: dict[str, Any] = {
            "source": "live_rollout",
            "steps": int(steps),
            "substeps_per_frame": int(substeps),
            "control_timestep_sec": float(control_dt),
            "behavior": behavior,
            "controller": active_controller_name,
            "dynamics": dynamics_mode if brain_controller is not None else None,
            "payload_version": LIVE_PAYLOAD_VERSION,
        }
        if connectome is not None:
            run_metadata["connectome"] = connectome.summary()

        return {
            "kind": "annolid-simulation-v1",
            "title": title,
            "adapter": "flybody-live",
            "labels": labels,
            "edges": _edges_for_labels(labels),
            "metadata": {
                "run_metadata": run_metadata,
                "mapping_metadata": {
                    "metadata": {
                        "template": "flybody-live",
                        "behavior": behavior,
                        "controller": active_controller_name,
                    }
                },
                "coordinate_system": {"units": "meters"},
            },
            "frames": frames,
        }
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


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
    parser.add_argument(
        "--behavior",
        type=str,
        default="walk_imitation",
        choices=SUPPORTED_FLYBODY_BEHAVIORS,
    )
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--controller",
        type=str,
        default="connectome",
        choices=("connectome", "random"),
        help="Neural controller driving FlyBody ('connectome' uses Google MaleCNS/FlyWire wiring).",
    )
    parser.add_argument(
        "--dynamics",
        type=str,
        default="spiking",
        choices=SUPPORTED_BRAIN_DYNAMICS,
        help="Neural dynamics mode ('spiking' LIF individual neurons or 'rate' population dynamics).",
    )
    parser.add_argument(
        "--connectome-path",
        type=str,
        default=None,
        help="Optional path to custom MaleCNS/FlyWire connectome (.json, .csv, .tsv, .parquet, .feather, or .npz).",
    )
    parser.add_argument(
        "--wpg-pattern-path",
        type=str,
        default=None,
        help="Optional wingbeat cycle .npy path for flight tasks (defaults to calibrated Drosophila pattern).",
    )
    parser.add_argument(
        "--policy_dir",
        type=str,
        default=None,
        help="Path to a directory containing a pre-trained flybody policy.",
    )
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
    payload = generate_live_rollout_payload(
        steps=args.steps,
        seed=args.seed,
        behavior=args.behavior,
        title=f"FlyBody Connectome Rollout ({args.behavior})",
        policy_dir=args.policy_dir,
        controller=args.controller,
        dynamics=args.dynamics,
        connectome_path=args.connectome_path,
        wpg_pattern_path=args.wpg_pattern_path,
    )
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
