from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from annolid.simulation.fly_brain import (
    BEHAVIOR_ENV_FACTORY_NAMES,
    FlyBrainController,
    build_canonical_fly_connectome,
    build_spiking_fly_connectome,
    ensure_calibrated_wingbeat_path,
    load_connectome_graph,
)
from annolid.simulation.flybody_live import (
    _candidate_body_names,
    _candidate_site_names,
    _extend_inference_trajectory_if_needed,
    _label_for_site,
    _yaw_from_root_quaternion,
)
from annolid.simulation.types import (
    Pose2DFrame,
    Pose3DFrame,
    SimulationAdapter,
    SimulationFrameResult,
    SimulationRunResult,
)


def _resolve_dotted_callable(path: str) -> Callable[..., Any]:
    module_name, sep, attr_name = str(path or "").partition(":")
    if not module_name or not sep or not attr_name:
        raise ValueError(
            "Expected dotted callable in the form 'module.submodule:attribute'"
        )
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name, None)
    if not callable(value):
        raise AttributeError(f"Callable not found: {path}")
    return value


def _synthetic_action_spec(behavior: str) -> Any:
    is_flight = str(behavior) in ("flight_imitation", "vision_guided_flight")
    if is_flight:
        names = (
            "head_abduct",
            "head_twist",
            "head",
            "wing_yaw_left",
            "wing_roll_left",
            "wing_pitch_left",
            "wing_yaw_right",
            "wing_roll_right",
            "wing_pitch_right",
            "abdomen_abduct",
            "abdomen",
            "user_0",
        )
        minimum = np.full(len(names), -1.0, dtype=np.float64)
        maximum = np.full(len(names), 1.0, dtype=np.float64)
    else:
        legs = (
            "T1_left",
            "T1_right",
            "T2_left",
            "T2_right",
            "T3_left",
            "T3_right",
        )
        names_list = [f"adhere_claw_{leg}" for leg in legs]
        names_list.extend(
            ["head_abduct", "head_twist", "head", "abdomen_abduct", "abdomen"]
        )
        for leg in legs:
            names_list.extend(
                [
                    f"coxa_abduct_{leg}",
                    f"coxa_twist_{leg}",
                    f"coxa_{leg}",
                    f"femur_twist_{leg}",
                    f"femur_{leg}",
                    f"tibia_{leg}",
                    f"tarsus_{leg}",
                    f"tarsus2_{leg}",
                ]
            )
        names = tuple(names_list)
        minimum = np.concatenate(
            [
                np.zeros(6, dtype=np.float64),
                np.full(len(names) - 6, -1.5, dtype=np.float64),
            ]
        )
        maximum = np.concatenate(
            [
                np.ones(6, dtype=np.float64),
                np.full(len(names) - 6, 2.0, dtype=np.float64),
            ]
        )
    return SimpleNamespace(
        name="\t".join(names),
        shape=(len(names),),
        dtype=np.float64,
        minimum=minimum,
        maximum=maximum,
    )


def _dry_run_site_targets(
    action: np.ndarray,
    idx: int,
    dt: float,
    behavior: str,
) -> Dict[str, list[float]]:
    is_flight = str(behavior) in ("flight_imitation", "vision_guided_flight")
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    t = float(idx) * float(dt)
    base_z = 0.15 if is_flight else 0.0
    dx = (0.8 if is_flight else 0.18) * t
    swing_a = float(arr[6]) * 0.02 if arr.size > 6 else 0.0
    swing_b = float(arr[14]) * 0.02 if arr.size > 14 else -swing_a
    wing_l = float(arr[3]) * 0.04 if arr.size > 3 else 0.0
    wing_r = float(arr[6 if is_flight else 4]) * 0.04 if arr.size > 6 else wing_l
    return {
        "head": [dx + 0.095, 0.0, base_z + 0.015],
        "thorax": [dx, 0.0, base_z],
        "abdomen": [dx - 0.145, 0.0, base_z - 0.015],
        "tarsus_T1_left": [dx + 0.075 + swing_a, 0.075, base_z - 0.115],
        "tarsus_T1_right": [dx + 0.075 + swing_b, -0.075, base_z - 0.115],
        "tarsus_T2_left": [dx - 0.010 + swing_b, 0.095, base_z - 0.120],
        "tarsus_T2_right": [dx - 0.010 + swing_a, -0.095, base_z - 0.120],
        "tarsus_T3_left": [dx - 0.105 + swing_a, 0.085, base_z - 0.125],
        "tarsus_T3_right": [dx - 0.105 + swing_b, -0.085, base_z - 0.125],
        "wing_left": [dx - 0.05, 0.11, base_z + 0.04 + wing_l],
        "wing_right": [dx - 0.05, -0.11, base_z + 0.04 + wing_r],
    }


_SITE_NAME_ALIASES: Dict[str, str] = {
    "head_site": "head",
    "thorax_site": "thorax",
    "left_front_tarsus_site": "tarsus_T1_left",
    "right_front_tarsus_site": "tarsus_T1_right",
    "left_middle_tarsus_site": "tarsus_T2_left",
    "right_middle_tarsus_site": "tarsus_T2_right",
    "left_hind_tarsus_site": "tarsus_T3_left",
    "right_hind_tarsus_site": "tarsus_T3_right",
    "left_front_tarsal_claw_site": "claw_T1_left",
    "right_front_tarsal_claw_site": "claw_T1_right",
    "left_middle_tarsal_claw_site": "claw_T2_left",
    "right_middle_tarsal_claw_site": "claw_T2_right",
    "left_hind_tarsal_claw_site": "claw_T3_left",
    "right_hind_tarsal_claw_site": "claw_T3_right",
}


class FlyBodyAdapter(SimulationAdapter):
    """FlyBody-backed adapter supporting IK pose fitting and connectome rollouts."""

    name = "flybody"

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._env_factory: Callable[..., Any] | None = None
        self._ik_solver: Callable[..., Any] | None = None
        self._env: Any = None
        self._uses_default_env_factory: bool = False
        self._env_behavior: str | None = None

    def configure(self, config: Mapping[str, Any]) -> None:
        self.close()
        self._config = dict(config)
        self._env_factory = None
        self._ik_solver = None
        self._uses_default_env_factory = False
        self._env_behavior = None

    def close(self) -> None:
        if self._env is not None:
            close_fn = getattr(self._env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            self._env = None

    def __enter__(self) -> FlyBodyAdapter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def initialize(self) -> None:
        if bool(self._config.get("dry_run")):
            return None
        behavior = str(self._config.get("behavior") or "walk_imitation")
        default_factory_attr = BEHAVIOR_ENV_FACTORY_NAMES.get(
            behavior, "walk_imitation"
        )
        explicit_factory = self._config.get("environment_factory")
        env_factory_path = str(
            explicit_factory or f"flybody.fly_envs:{default_factory_attr}"
        )
        ik_solver_path = str(
            self._config.get("ik_function")
            or "flybody.inverse_kinematics:qpos_from_site_xpos"
        )
        try:
            self._env_factory = _resolve_dotted_callable(env_factory_path)
            self._ik_solver = _resolve_dotted_callable(ik_solver_path)
        except Exception as exc:
            raise RuntimeError(
                "FlyBody integration requires the optional FlyBody stack "
                "(including dm_control / MuJoCo) and compatible import paths. "
                "Use --dry-run to validate IO and mapping without those dependencies."
            ) from exc
        self._uses_default_env_factory = not bool(explicit_factory)
        self._env_behavior = behavior
        self.close()
        return None

    def fit_2d(self, observations: Sequence[Pose2DFrame]) -> SimulationRunResult:
        default_z = float(self._config.get("default_z", 0.0))
        observations_3d = []
        for frame in observations:
            points = {
                label: (point[0], point[1], default_z)
                for label, point in frame.points.items()
            }
            observations_3d.append(
                Pose3DFrame(
                    frame_index=frame.frame_index,
                    video_name=frame.video_name,
                    timestamp_sec=frame.timestamp_sec,
                    points=points,
                    scores=dict(frame.scores),
                    source_record=dict(frame.source_record),
                )
            )
        return self.fit_3d(observations_3d)

    def fit_3d(self, observations: Sequence[Pose3DFrame]) -> SimulationRunResult:
        mapping = dict(self._config.get("keypoint_to_site") or {})
        dry_run = bool(self._config.get("dry_run"))
        frames = []
        behavior = str(self._config.get("behavior") or "walk_imitation")
        world_origin: np.ndarray | None = None
        if not dry_run and (self._env_factory is None or self._ik_solver is None):
            self.initialize()
        for frame in observations:
            site_targets = {
                mapping.get(label, label): [point[0], point[1], point[2]]
                for label, point in frame.points.items()
            }
            if dry_run:
                state: Dict[str, Any] = {
                    "backend": self.name,
                    "dry_run": True,
                    "site_targets": site_targets,
                }
                diagnostics: Dict[str, Any] = {
                    "mode": "dry_run",
                    "mapped_sites": len(site_targets),
                }
            else:
                ik_out = self._run_ik(site_targets)
                qpos_payload, ik_meta = self._unpack_ik_output(ik_out)
                env = self._get_env(behavior=behavior)
                physics = getattr(env, "physics", env)
                if (
                    isinstance(qpos_payload, list)
                    and hasattr(physics, "data")
                    and hasattr(physics.data, "qpos")
                ):
                    arr = np.asarray(qpos_payload, dtype=np.float64).reshape(-1)
                    n = min(arr.size, physics.data.qpos.size)
                    physics.data.qpos[:n] = arr[:n]
                    if hasattr(physics, "forward"):
                        physics.forward()
                world_origin, _, body_poses, model_pose = (
                    self._extract_kinematic_snapshot(
                        physics,
                        behavior=behavior,
                        world_origin=world_origin,
                    )
                )
                state = {
                    "backend": self.name,
                    "dry_run": False,
                    "site_targets": site_targets,
                    "qpos": qpos_payload,
                }
                if body_poses:
                    state["body_poses"] = body_poses
                if model_pose is not None:
                    state["model_pose"] = model_pose
                diagnostics = {
                    "mode": "ik_fit",
                    "mapped_sites": len(site_targets),
                    **ik_meta,
                }
            frames.append(
                SimulationFrameResult(
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    state=state,
                    diagnostics=diagnostics,
                )
            )
        return SimulationRunResult(
            frames=frames,
            metadata={
                "backend": self.name,
                "frames": len(frames),
                "dry_run": dry_run,
            },
        )

    def rollout(
        self, initial_state: Mapping[str, Any], steps: int
    ) -> SimulationRunResult:
        """Roll out closed-loop FlyBody behavior controlled by the Drosophila connectome."""
        dry_run = bool(self._config.get("dry_run"))
        behavior = str(
            initial_state.get("behavior")
            or self._config.get("behavior")
            or "walk_imitation"
        )
        dynamics = str(
            initial_state.get("dynamics") or self._config.get("dynamics") or "spiking"
        )
        seed = int(initial_state.get("seed", self._config.get("seed", 7)))
        connectome_path = initial_state.get(
            "connectome_path", self._config.get("connectome_path")
        )
        if connectome_path:
            connectome = load_connectome_graph(connectome_path)
        elif dynamics == "spiking":
            connectome = build_spiking_fly_connectome(seed=seed)
        else:
            connectome = build_canonical_fly_connectome()
        step_count = max(0, int(steps))
        frames: list[SimulationFrameResult] = []

        if dry_run:
            spec = _synthetic_action_spec(behavior)
            dt = (
                0.0002
                if behavior in ("flight_imitation", "vision_guided_flight")
                else 0.002
            )
            controller = FlyBrainController(
                spec,
                behavior=behavior,
                connectome=connectome,
                dynamics=dynamics,
                seed=seed,
                control_timestep=dt,
            )
            controller.warmup({}, duration_sec=0.04)
            for idx in range(step_count):
                action = controller.step({}, dt=dt)
                frames.append(
                    SimulationFrameResult(
                        frame_index=idx,
                        timestamp_sec=round(idx * dt, 5),
                        state={
                            "backend": self.name,
                            "dry_run": True,
                            "action": self._to_jsonable(action),
                            "site_targets": _dry_run_site_targets(
                                action, idx, dt, behavior
                            ),
                        },
                        diagnostics={
                            "mode": "connectome_rollout_dry_run",
                            "brain": controller.diagnostics(),
                        },
                    )
                )
            return SimulationRunResult(
                frames=frames,
                metadata={
                    "backend": self.name,
                    "frames": len(frames),
                    "dry_run": True,
                    "behavior": behavior,
                    "dynamics": dynamics,
                    "connectome": connectome.summary(),
                },
            )

        if self._env_factory is None:
            self.initialize()
        env = self._get_env(behavior=behavior)
        _extend_inference_trajectory_if_needed(
            env,
            behavior=behavior,
            min_control_steps=step_count + 40,
        )
        timestep = env.reset()
        init_qpos = initial_state.get("qpos")
        if init_qpos is not None:
            physics = getattr(env, "physics", env)
            arr = np.asarray(init_qpos, dtype=np.float64).reshape(-1)
            n = min(arr.size, physics.data.qpos.size)
            physics.data.qpos[:n] = arr[:n]
            if hasattr(physics, "forward"):
                physics.forward()

        dt_fn = getattr(env, "control_timestep", None)
        dt = float(dt_fn()) if callable(dt_fn) else 0.002
        controller = FlyBrainController(
            env.action_spec(),
            behavior=behavior,
            connectome=connectome,
            dynamics=dynamics,
            seed=seed,
            control_timestep=dt,
        )
        controller.warmup(getattr(timestep, "observation", {}) or {}, duration_sec=0.04)

        world_origin: np.ndarray | None = None
        for idx in range(step_count):
            obs = getattr(timestep, "observation", {}) or {}
            action = controller.step(obs, dt=dt)
            physics = getattr(env, "physics", env)
            qpos = np.asarray(physics.data.qpos, dtype=float)
            world_origin, site_targets, body_poses, model_pose = (
                self._extract_kinematic_snapshot(
                    physics,
                    behavior=behavior,
                    world_origin=world_origin,
                )
            )
            frame_state: Dict[str, Any] = {
                "backend": self.name,
                "dry_run": False,
                "qpos": qpos.tolist(),
                "action": self._to_jsonable(action),
            }
            if site_targets:
                frame_state["site_targets"] = site_targets
            if body_poses:
                frame_state["body_poses"] = body_poses
            if model_pose is not None:
                frame_state["model_pose"] = model_pose
            frames.append(
                SimulationFrameResult(
                    frame_index=idx,
                    timestamp_sec=round(idx * dt, 5),
                    state=frame_state,
                    diagnostics={
                        "mode": "connectome_rollout",
                        "brain": controller.diagnostics(),
                    },
                )
            )
            timestep = env.step(action)
            last_fn = getattr(timestep, "last", None)
            if callable(last_fn) and last_fn():
                timestep = env.reset()

        return SimulationRunResult(
            frames=frames,
            metadata={
                "backend": self.name,
                "frames": len(frames),
                "dry_run": False,
                "behavior": behavior,
                "dynamics": dynamics,
                "connectome": connectome.summary(),
            },
        )

    def _run_ik(self, site_targets: Dict[str, list[float]]) -> Any:
        if self._env_factory is None or self._ik_solver is None:
            raise RuntimeError("FlyBody adapter is not initialized")
        env = self._get_env()
        physics = getattr(env, "physics", env)
        requested_sites = list(site_targets.keys())
        site_names = self._resolve_site_names(physics, requested_sites)
        target_xpos = np.asarray(
            [site_targets[req] for req in requested_sites],
            dtype=float,
        )
        joint_names = self._resolve_joint_names(env, physics, site_names)
        return self._ik_solver(
            physics=physics,
            site_names=site_names,
            target_xpos=target_xpos,
            joint_names=joint_names,
            **dict(self._config.get("ik_kwargs") or {}),
        )

    @classmethod
    def _unpack_ik_output(cls, ik_out: Any) -> tuple[Any, Dict[str, Any]]:
        if not isinstance(ik_out, dict) and hasattr(ik_out, "qpos"):
            meta: Dict[str, Any] = {}
            if hasattr(ik_out, "err_norm"):
                meta["ik_err_norm"] = float(getattr(ik_out, "err_norm"))
            if hasattr(ik_out, "steps"):
                meta["ik_steps"] = int(getattr(ik_out, "steps"))
            if hasattr(ik_out, "success"):
                meta["ik_success"] = bool(getattr(ik_out, "success"))
            return cls._to_jsonable(getattr(ik_out, "qpos")), meta
        return cls._to_jsonable(ik_out), {}

    def _get_env(self, behavior: str | None = None) -> Any:
        active_behavior = str(
            behavior or self._config.get("behavior") or "walk_imitation"
        )
        if self._uses_default_env_factory and self._env_behavior != active_behavior:
            self.close()
            default_factory_attr = BEHAVIOR_ENV_FACTORY_NAMES.get(
                active_behavior, "walk_imitation"
            )
            self._env_factory = _resolve_dotted_callable(
                f"flybody.fly_envs:{default_factory_attr}"
            )
            self._env_behavior = active_behavior
        if self._env is None:
            if self._env_factory is None:
                raise RuntimeError("FlyBody adapter environment factory is unset")
            kwargs: Dict[str, Any] = {}
            try:
                sig = inspect.signature(self._env_factory)
                params = sig.parameters
            except (TypeError, ValueError):
                params = {}
            if active_behavior in ("flight_imitation", "vision_guided_flight"):
                wpg = (
                    self._config.get("wpg_pattern_path")
                    or ensure_calibrated_wingbeat_path()
                )
                if "wpg_pattern_path" in params:
                    kwargs["wpg_pattern_path"] = str(wpg)
            elif active_behavior == "escape_takeoff" and "disable_wings" in params:
                kwargs["disable_wings"] = False
            self._env = self._env_factory(**kwargs)
        return self._env

    @staticmethod
    def _extract_kinematic_snapshot(
        physics: Any,
        *,
        behavior: str,
        world_origin: np.ndarray | None,
    ) -> tuple[
        np.ndarray | None,
        Dict[str, list[float]],
        Dict[str, Dict[str, list[float]]],
        Dict[str, Any] | None,
    ]:
        named = getattr(physics, "named", None)
        data = getattr(named, "data", None)
        if data is None or not (hasattr(data, "xpos") and hasattr(data, "xquat")):
            return world_origin, {}, {}, None
        try:
            body_names = _candidate_body_names(physics)
        except Exception:
            body_names = []
        if not body_names:
            return world_origin, {}, {}, None
        try:
            site_names = _candidate_site_names(physics)
        except Exception:
            site_names = []

        if "walker/thorax" in body_names:
            thorax_pos = np.asarray(data.xpos["walker/thorax"], dtype=float).copy()
            thorax_quat = np.asarray(data.xquat["walker/thorax"], dtype=float).copy()
        else:
            thorax_pos = np.asarray(data.xpos[body_names[0]], dtype=float).copy()
            thorax_quat = np.asarray(data.xquat[body_names[0]], dtype=float).copy()

        if world_origin is None:
            is_flight = str(behavior) in ("flight_imitation", "vision_guided_flight")
            world_origin = np.array(
                [
                    float(thorax_pos[0]),
                    float(thorax_pos[1]),
                    float(thorax_pos[2]) - 0.15 if is_flight else 0.0,
                ],
                dtype=float,
            )

        site_targets: Dict[str, list[float]] = {}
        if hasattr(data, "site_xpos"):
            for site_name in site_names:
                coords = (
                    np.asarray(data.site_xpos[site_name], dtype=float) - world_origin
                )
                site_targets[_label_for_site(site_name)] = [
                    float(coords[0]),
                    float(coords[1]),
                    float(coords[2]),
                ]

        body_poses: Dict[str, Dict[str, list[float]]] = {}
        for body_name in body_names:
            pos = np.asarray(data.xpos[body_name], dtype=float) - world_origin
            quat = np.asarray(data.xquat[body_name], dtype=float)
            body_poses[_label_for_site(body_name)] = {
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "quaternion": [
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                ],
            }

        model_pos = thorax_pos - world_origin
        model_pose = {
            "position": [float(model_pos[0]), float(model_pos[1]), float(model_pos[2])],
            "rotation": [0.0, 0.0, _yaw_from_root_quaternion(thorax_quat)],
            "scale": 7.5,
        }
        return world_origin, site_targets, body_poses, model_pose

    def _resolve_joint_names(
        self,
        env: Any,
        physics: Any,
        site_names: Sequence[str],
    ) -> list[str]:
        mapped_joint_names = self._mapped_joint_names_for_sites(physics, site_names)
        if mapped_joint_names:
            return mapped_joint_names
        walker = getattr(getattr(env, "task", None), "_walker", None)
        mjcf_model = getattr(walker, "mjcf_model", None)
        if mjcf_model is None or not hasattr(mjcf_model, "find_all"):
            explicit = self._explicit_joint_names()
            if explicit:
                return explicit
            raise RuntimeError(
                "Unable to resolve FlyBody joint names from mapping or walker model."
            )
        joints = []
        for joint in mjcf_model.find_all("joint"):
            name = str(getattr(joint, "name", "") or "").strip()
            if name and name != "free":
                resolved = self._resolve_runtime_name(
                    physics.named.model.dof_jntid.axes.row,
                    name,
                )
                if resolved:
                    joints.append(resolved)
        if joints:
            return joints
        explicit = self._explicit_joint_names()
        if explicit:
            return explicit
        raise RuntimeError(
            "FlyBody walker model did not expose any usable joint names."
        )

    def _mapped_joint_names_for_sites(
        self, physics: Any, site_names: Sequence[str]
    ) -> list[str]:
        site_to_joint = dict(self._config.get("site_to_joint") or {})
        names = []
        seen = set()
        for site_name in site_names:
            joint_name = str(
                site_to_joint.get(site_name)
                or site_to_joint.get(self._strip_runtime_prefix(site_name))
                or ""
            ).strip()
            resolved = self._resolve_runtime_name(
                physics.named.model.dof_jntid.axes.row,
                joint_name,
            )
            if resolved and resolved not in seen:
                seen.add(resolved)
                names.append(resolved)
        return names

    def _explicit_joint_names(self) -> list[str]:
        names = []
        seen = set()
        for item in self._config.get("joint_names") or ():
            label = str(item or "").strip()
            if label and label not in seen:
                seen.add(label)
                names.append(label)
        return names

    def _resolve_site_names(self, physics: Any, site_names: Sequence[str]) -> list[str]:
        resolved = []
        axis = physics.named.model.site_pos.axes.row
        for site_name in site_names:
            name = self._resolve_runtime_name(axis, site_name)
            if not name:
                stripped = self._strip_runtime_prefix(site_name)
                alias = _SITE_NAME_ALIASES.get(stripped)
                if alias:
                    name = self._resolve_runtime_name(axis, alias)
            if not name:
                raise RuntimeError(f"Unknown FlyBody site name: {site_name!r}")
            resolved.append(name)
        return resolved

    @staticmethod
    def _strip_runtime_prefix(name: str) -> str:
        text = str(name or "").strip()
        if "/" in text:
            return text.split("/", 1)[1]
        return text

    @staticmethod
    def _resolve_runtime_name(axis: Any, name: str) -> str | None:
        text = str(name or "").strip()
        if not text:
            return None
        for attr in ("_names_to_indices", "_names_to_offsets", "_names_to_slices"):
            mapping = getattr(axis, attr, None)
            if isinstance(mapping, dict):
                if text in mapping:
                    return text
                prefixed = f"walker/{text}"
                if prefixed in mapping:
                    return prefixed
        return None

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, tuple):
            return [FlyBodyAdapter._to_jsonable(item) for item in value]
        if isinstance(value, list):
            return [FlyBodyAdapter._to_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): FlyBodyAdapter._to_jsonable(item)
                for key, item in value.items()
            }
        return value
