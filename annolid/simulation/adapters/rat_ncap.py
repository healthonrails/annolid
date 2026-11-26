from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from annolid.simulation.biomechanics import (
    solve_rat_v2_ik_sequence,
)
from annolid.simulation.rat_ncap import (
    RAT_NCAP_METADATA,
    RAT_V2_KEYPOINT_TO_SITE,
    RAT_V2_SKELETON_EDGES,
    RatNCAPConfig,
    RatNCAPController,
    default_speed_for_behavior,
    normalize_rat_coat,
    normalize_rat_interface,
)
from annolid.simulation.types import (
    Pose2DFrame,
    Pose3DFrame,
    SimulationAdapter,
    SimulationFrameResult,
    SimulationRunResult,
)


class RatNCAPAdapter(SimulationAdapter):
    """SimulationAdapter for Rat v2 / Rat NCAP fitting and closed-loop spinal rollouts.

    Works out-of-the-box with zero heavy runtime dependencies using Annolid's
    pure-NumPy Rat NCAP implementation, and can optionally load a local
    `rat-ncap` MuJoCo `rat.xml` model when `mujoco` + `rat.xml` are available
    and `use_mujoco=True` is requested.
    """

    name = "rat_ncap"

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._initialized: bool = False
        self._mujoco_model: Any = None
        self._mujoco_data: Any = None
        self._rat_xml_path: str | None = None

    def configure(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self._initialized = False

    def initialize(self) -> None:
        self._mujoco_model = None
        self._mujoco_data = None
        self._rat_xml_path = None
        use_mujoco = bool(self._config.get("use_mujoco", False)) and not bool(
            self._config.get("dry_run", False)
        )
        if use_mujoco:
            self._try_initialize_mujoco()
        self._initialized = True

    def _try_initialize_mujoco(self) -> None:
        try:
            import mujoco  # type: ignore

            from annolid.gui.rat_ncap_support import resolve_rat_xml_path

            xml_path = self._config.get("rat_xml_path") or resolve_rat_xml_path()
            if xml_path is not None:
                self._mujoco_model = mujoco.MjModel.from_xml_path(str(xml_path))
                self._mujoco_data = mujoco.MjData(self._mujoco_model)
                self._rat_xml_path = str(xml_path)
        except Exception:
            self._mujoco_model = None
            self._mujoco_data = None
            self._rat_xml_path = None

    def close(self) -> None:
        self._mujoco_model = None
        self._mujoco_data = None
        self._initialized = False

    def _map_points_to_sites(
        self,
        points: Mapping[str, Sequence[float]],
        *,
        default_z: float = 0.0,
    ) -> Dict[str, list[float]]:
        custom_map = dict(self._config.get("keypoint_to_site") or {})
        mapped: Dict[str, list[float]] = {}
        for kp_name, coords in points.items():
            if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                continue
            site_name = str(
                custom_map.get(kp_name)
                or RAT_V2_KEYPOINT_TO_SITE.get(kp_name)
                or kp_name
            )
            x = float(coords[0])
            y = float(coords[1])
            z = float(coords[2]) if len(coords) >= 3 else float(default_z)
            mapped[site_name] = [x, y, z]
        return mapped

    def _estimate_kinematic_state_from_sites(
        self,
        site_targets: Mapping[str, Sequence[float]],
        *,
        prev_root: np.ndarray | None,
        dt: float,
    ) -> tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
        """Estimate Rat v2 root pose, heading, speed, and basic joint pose from mapped sites."""
        torso = site_targets.get("torso") or site_targets.get("torso_center")
        head = site_targets.get("head") or site_targets.get("nose_site")
        pelvis = site_targets.get("pelvis") or site_targets.get("tail_base_site")

        if torso is not None:
            root = np.asarray(torso[:3], dtype=np.float64)
        elif site_targets:
            pts = np.asarray(list(site_targets.values()), dtype=np.float64)
            root = np.mean(pts, axis=0)
        else:
            root = np.zeros(3, dtype=np.float64)

        yaw = 0.0
        if head is not None and pelvis is not None:
            dx = float(head[0]) - float(pelvis[0])
            dy = float(head[1]) - float(pelvis[1])
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                yaw = math.atan2(dy, dx)
        elif head is not None and torso is not None:
            dx = float(head[0]) - float(torso[0])
            dy = float(head[1]) - float(torso[1])
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                yaw = math.atan2(dy, dx)

        speed = 0.0
        if prev_root is not None and dt > 0.0:
            speed = float(np.linalg.norm(root[:2] - prev_root[:2]) / dt)

        interface = normalize_rat_interface(
            self._config.get("interface", "flexor_extensor")
        )
        state: Dict[str, Any] = {
            "dry_run": bool(self._config.get("dry_run", False)),
            "site_targets": {
                k: [float(v[0]), float(v[1]), float(v[2])]
                for k, v in site_targets.items()
            },
            "root_pos": [round(float(x), 6) for x in root],
            "root_yaw": round(yaw, 5),
            "estimated_speed": round(speed, 5),
            "interface": interface,
            "mujoco_active": self._mujoco_model is not None,
        }
        diagnostics: Dict[str, Any] = {
            "mapped_site_count": len(site_targets),
            "estimated_speed": round(speed, 5),
            "mujoco_active": self._mujoco_model is not None,
        }
        return state, diagnostics, root

    def _fit_observations(
        self,
        observations: Sequence[Pose2DFrame] | Sequence[Pose3DFrame],
        *,
        mode: str,
        default_z: float,
    ) -> SimulationRunResult:
        if not self._initialized:
            self.initialize()
        fps = max(1.0, float(self._config.get("fps", 30.0)))
        dt = 1.0 / fps
        interface = normalize_rat_interface(
            self._config.get("interface", "flexor_extensor")
        )
        coat = normalize_rat_coat(self._config.get("coat", "black_hooded"))

        mapped_per_frame: list[Dict[str, list[float]]] = []
        for obs in observations:
            mapped_per_frame.append(
                self._map_points_to_sites(obs.points, default_z=default_z)
            )

        ik_result = solve_rat_v2_ik_sequence(
            mapped_per_frame,
            dt=dt,
            interface_type=interface,
        )
        ik_frames = ik_result["frames"]
        spinal_info = ik_result["spinal_circuit_inference"]

        frames: list[SimulationFrameResult] = []
        prev_root: np.ndarray | None = None
        for idx, obs in enumerate(observations):
            site_targets = mapped_per_frame[idx]
            state, diagnostics, prev_root = self._estimate_kinematic_state_from_sites(
                site_targets, prev_root=prev_root, dt=dt
            )
            if idx < len(ik_frames):
                ik_f = ik_frames[idx]
                state.update(
                    {
                        "qpos": list(ik_f["qpos"]),
                        "qvel": list(ik_f["qvel"]),
                        "torque": list(ik_f["torque"]),
                        "flx_activation": list(ik_f["flx_activation"]),
                        "ext_activation": list(ik_f["ext_activation"]),
                        "stiffness": list(ik_f["stiffness"]),
                        "root_euler": list(ik_f["root_euler"]),
                        "reconstructed_keypoints_3d": dict(
                            ik_f["reconstructed_keypoints_3d"]
                        ),
                        "gait": str(spinal_info.get("dominant_gait", "stand")),
                        "coat": coat,
                    }
                )
                diagnostics.update(
                    {
                        "ik_rmse_m": float(ik_f["ik_rmse_m"]),
                        "joint_limit_violations": int(ik_f["joint_limit_violations"]),
                        "mechanical_power": dict(ik_f["mechanical_power"]),
                        "paw_clearance_m": dict(ik_f["paw_clearance_m"]),
                    }
                )
            frames.append(
                SimulationFrameResult(
                    frame_index=int(obs.frame_index),
                    timestamp_sec=obs.timestamp_sec,
                    state=state,
                    diagnostics=diagnostics,
                )
            )
        return SimulationRunResult(
            frames=frames,
            metadata={
                "backend": self.name,
                "mode": mode,
                "model": RAT_NCAP_METADATA,
                "interface": interface,
                "coat": coat,
                "ik_summary": ik_result["summary"],
                "inverse_dynamics": ik_result["inverse_dynamics"],
                "spinal_circuit_inference": spinal_info,
                "viewer_edges": [list(edge) for edge in RAT_V2_SKELETON_EDGES],
                "mujoco_xml": self._rat_xml_path,
            },
        )

    def fit_2d(self, observations: Sequence[Pose2DFrame]) -> SimulationRunResult:
        default_z = float(self._config.get("default_z", 0.0))
        return self._fit_observations(observations, mode="fit_2d", default_z=default_z)

    def fit_3d(self, observations: Sequence[Pose3DFrame]) -> SimulationRunResult:
        return self._fit_observations(observations, mode="fit_3d", default_z=0.0)

    def rollout(
        self, initial_state: Mapping[str, Any], steps: int
    ) -> SimulationRunResult:
        if not self._initialized:
            self.initialize()
        behavior = (
            str(initial_state.get("behavior") or self._config.get("behavior") or "trot")
            .strip()
            .lower()
        )
        # Allow FlyBody default behavior string to map cleanly onto Rat trot if user switched backends.
        if behavior in {"walk_imitation", "forage_walk"}:
            behavior = "walk"
        elif behavior in {"flight_imitation", "escape_takeoff", "vision_guided_flight"}:
            behavior = "bound"
        elif behavior == "grooming":
            behavior = "stand"

        interface = normalize_rat_interface(
            initial_state.get("interface")
            or self._config.get("interface")
            or "flexor_extensor"
        )
        coat = normalize_rat_coat(
            initial_state.get("coat") or self._config.get("coat") or "black_hooded"
        )
        raw_speed = initial_state.get("target_speed", self._config.get("target_speed"))
        target_speed = (
            float(raw_speed)
            if raw_speed is not None
            else default_speed_for_behavior(behavior)
        )
        ablation = (
            str(initial_state.get("ablation") or self._config.get("ablation") or "none")
            .strip()
            .lower()
        )
        ablation_scale = float(
            initial_state.get("ablation_scale", self._config.get("ablation_scale", 0.0))
        )
        rg_noise_std = float(
            initial_state.get("rg_noise_std", self._config.get("rg_noise_std", 0.0))
        )
        terrain_bump_cm = float(
            initial_state.get(
                "terrain_bump_cm", self._config.get("terrain_bump_cm", 0.0)
            )
        )
        dt = float(initial_state.get("dt", self._config.get("dt", 0.02)))
        seed = int(initial_state.get("seed", self._config.get("seed", 7)))

        controller = RatNCAPController(
            RatNCAPConfig(
                interface=interface,
                behavior=behavior,
                target_speed=target_speed,
                coat=coat,
                ablation=ablation,
                ablation_scale=ablation_scale,
                rg_noise_std=rg_noise_std,
                terrain_bump_cm=terrain_bump_cm,
                dt=dt,
                seed=seed,
            )
        )

        frames: list[SimulationFrameResult] = []
        gait_counts: Dict[str, int] = {}
        total_steps = max(1, int(steps))
        for frame_idx in range(total_steps):
            step_data = controller.step()
            gait = str(step_data["gait"])
            gait_counts[gait] = gait_counts.get(gait, 0) + 1
            frames.append(
                SimulationFrameResult(
                    frame_index=frame_idx,
                    timestamp_sec=float(step_data["time_sec"]),
                    state={
                        "dry_run": bool(self._config.get("dry_run", False)),
                        "qpos": list(step_data["qpos"]),
                        "qvel": list(step_data["qvel"]),
                        "torque": list(step_data["torque"]),
                        "flx_activation": list(step_data["flx_activation"]),
                        "ext_activation": list(step_data["ext_activation"]),
                        "site_targets": dict(step_data["keypoints_3d"]),
                        "body_poses": dict(step_data["body_poses"]),
                        "model_pose": {
                            "position": list(step_data["root_pos"]),
                            "rotation": list(step_data["root_euler"]),
                            "scale": 10.0,
                        },
                        "root_pos": list(step_data["root_pos"]),
                        "gait": gait,
                        "coat": coat,
                        "interface": interface,
                    },
                    diagnostics={
                        "behavior": behavior,
                        "gait": gait,
                        "target_speed_mps": step_data["target_speed_mps"],
                        "forward_speed_mps": step_data["forward_speed_mps"],
                        "command": step_data["command"],
                        "step_frequency_hz": step_data["step_frequency_hz"],
                        "step_length_m": step_data["step_length_m"],
                        "duty_factor": step_data["duty_factor"],
                        "phase_offsets_rel_fr": step_data["phase_offsets_rel_fr"],
                        "foot_contacts": step_data["foot_contacts"],
                        "mechanical_power": step_data["mechanical_power"],
                        "rg_flx": step_data["rg"]["flx"],
                        "rg_ext": step_data["rg"]["ext"],
                        "ablation": controller.ablation,
                        "ablation_scale": controller.ablation_scale,
                        "rg_noise_std": controller.rg.rg_noise_std,
                    },
                )
            )

        dominant_gait = (
            max(gait_counts.items(), key=lambda kv: kv[1])[0]
            if gait_counts
            else "stand"
        )
        return SimulationRunResult(
            frames=frames,
            metadata={
                "backend": self.name,
                "mode": "rollout",
                "model": RAT_NCAP_METADATA,
                "behavior": behavior,
                "interface": interface,
                "coat": coat,
                "ablation": controller.ablation,
                "ablation_scale": controller.ablation_scale,
                "rg_noise_std": controller.rg.rg_noise_std,
                "terrain_bump_cm": terrain_bump_cm,
                "dominant_gait": dominant_gait,
                "gait_counts": gait_counts,
                "viewer_edges": [list(edge) for edge in RAT_V2_SKELETON_EDGES],
                "mujoco_xml": self._rat_xml_path,
            },
        )
