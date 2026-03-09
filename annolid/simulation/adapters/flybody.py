from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

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


class FlyBodyAdapter(SimulationAdapter):
    """FlyBody-backed adapter with a useful dry-run mode before heavy deps exist."""

    name = "flybody"

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._env_factory: Callable[..., Any] | None = None
        self._ik_solver: Callable[..., Any] | None = None
        self._env: Any = None

    def configure(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self._env = None

    def initialize(self) -> None:
        if bool(self._config.get("dry_run")):
            return None
        env_factory_path = str(
            self._config.get("environment_factory") or "flybody.fly_envs:walk_imitation"
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
        self._env = None
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
        if not dry_run:
            self.initialize()
        for frame in observations:
            site_targets = {
                mapping.get(label, label): [point[0], point[1], point[2]]
                for label, point in frame.points.items()
            }
            if dry_run:
                state = {
                    "backend": self.name,
                    "dry_run": True,
                    "site_targets": site_targets,
                }
                diagnostics = {
                    "mode": "dry_run",
                    "mapped_sites": len(site_targets),
                }
            else:
                qpos = self._run_ik(site_targets)
                state = {
                    "backend": self.name,
                    "dry_run": False,
                    "site_targets": site_targets,
                    "qpos": self._to_jsonable(qpos),
                }
                diagnostics = {
                    "mode": "ik_fit",
                    "mapped_sites": len(site_targets),
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

    def _run_ik(self, site_targets: Dict[str, list[float]]) -> Any:
        if self._env_factory is None or self._ik_solver is None:
            raise RuntimeError("FlyBody adapter is not initialized")
        env = self._get_env()
        physics = getattr(env, "physics", env)
        site_names = self._resolve_site_names(physics, list(site_targets.keys()))
        target_xpos = np.asarray(
            [site_targets[self._strip_runtime_prefix(name)] for name in site_names],
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

    def _get_env(self) -> Any:
        if self._env is None:
            self._env = self._env_factory()
        return self._env

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
        for site_name in site_names:
            name = self._resolve_runtime_name(
                physics.named.model.site_pos.axes.row,
                site_name,
            )
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
