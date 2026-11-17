from __future__ import annotations

from types import SimpleNamespace

from annolid.simulation.adapters.flybody import FlyBodyAdapter
from annolid.simulation.types import Pose3DFrame


class _FakeJoint:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeModel:
    def __init__(self, joint_names: list[str]) -> None:
        self._joint_names = joint_names

    def find_all(self, kind: str):
        if kind != "joint":
            return []
        return [_FakeJoint(name) for name in self._joint_names]


class _FakeRegularAxis:
    def __init__(self, names: list[str]) -> None:
        self._names_to_offsets = {name: idx for idx, name in enumerate(names)}


class _FakeRaggedAxis:
    def __init__(self, names: list[str]) -> None:
        self._names_to_indices = {name: idx for idx, name in enumerate(names)}


def _fake_physics(site_names: list[str], joint_names: list[str]):
    return SimpleNamespace(
        named=SimpleNamespace(
            model=SimpleNamespace(
                site_pos=SimpleNamespace(
                    axes=SimpleNamespace(row=_FakeRegularAxis(site_names))
                ),
                dof_jntid=SimpleNamespace(
                    axes=SimpleNamespace(row=_FakeRaggedAxis(joint_names))
                ),
            )
        )
    )


def test_flybody_adapter_calls_real_signature_and_infers_joint_names() -> None:
    captured = {}
    adapter = FlyBodyAdapter()
    adapter.configure(
        {
            "keypoint_to_site": {"nose": "head_site"},
            "ik_kwargs": {"max_steps": 123},
        }
    )
    physics = _fake_physics(["walker/head_site"], ["walker/joint_a", "walker/joint_b"])
    adapter._env_factory = lambda: SimpleNamespace(
        physics=physics,
        task=SimpleNamespace(
            _walker=SimpleNamespace(
                mjcf_model=_FakeModel(["free", "joint_a", "joint_b"])
            )
        ),
    )
    adapter.initialize = lambda: None

    def _solver(*, physics, site_names, target_xpos, joint_names, **kwargs):
        captured["physics"] = physics
        captured["site_names"] = list(site_names)
        captured["target_xpos"] = target_xpos.tolist()
        captured["joint_names"] = list(joint_names)
        captured["kwargs"] = dict(kwargs)
        return {"qpos": [1, 2, 3]}

    adapter._ik_solver = _solver

    result = adapter.fit_3d(
        [
            Pose3DFrame(
                frame_index=0,
                video_name="demo.mp4",
                points={"nose": (1.0, 2.0, 3.0)},
            )
        ]
    )

    assert captured["physics"] is physics
    assert captured["site_names"] == ["walker/head_site"]
    assert captured["target_xpos"] == [[1.0, 2.0, 3.0]]
    assert captured["joint_names"] == ["walker/joint_a", "walker/joint_b"]
    assert captured["kwargs"]["max_steps"] == 123
    assert result.frames[0].state["qpos"]["qpos"] == [1, 2, 3]


def test_flybody_adapter_prefers_site_to_joint_mapping() -> None:
    captured = {}
    adapter = FlyBodyAdapter()
    adapter.configure(
        {
            "keypoint_to_site": {
                "nose": "head_site",
                "thorax": "thorax_site",
            },
            "site_to_joint": {
                "head_site": "neck_joint",
                "thorax_site": "thorax_joint",
            },
        }
    )
    physics = _fake_physics(
        ["walker/head_site", "walker/thorax_site"],
        ["walker/neck_joint", "walker/thorax_joint"],
    )
    adapter._env_factory = lambda: SimpleNamespace(
        physics=physics,
        task=SimpleNamespace(_walker=SimpleNamespace(mjcf_model=_FakeModel(["free"]))),
    )
    adapter.initialize = lambda: None

    def _solver(*, physics, site_names, target_xpos, joint_names, **kwargs):
        captured["joint_names"] = list(joint_names)
        return [0.0]

    adapter._ik_solver = _solver

    adapter.fit_3d(
        [
            Pose3DFrame(
                frame_index=0,
                video_name="demo.mp4",
                points={
                    "nose": (1.0, 2.0, 3.0),
                    "thorax": (4.0, 5.0, 6.0),
                },
            )
        ]
    )

    assert captured["joint_names"] == ["walker/neck_joint", "walker/thorax_joint"]


def test_flybody_adapter_resolves_template_aliases_and_unpacks_ik_result() -> None:
    from collections import namedtuple

    IKResult = namedtuple("IKResult", ["qpos", "err_norm", "steps", "success"])
    captured = {}
    adapter = FlyBodyAdapter()
    adapter.configure(
        {
            "keypoint_to_site": {
                "nose": "head_site",
                "left_front_leg_tip": "left_front_tarsus_site",
            },
        }
    )
    physics = _fake_physics(
        ["walker/head", "walker/tarsus_T1_left"],
        ["walker/head_twist"],
    )
    closed = {"count": 0}
    adapter._env_factory = lambda: SimpleNamespace(
        physics=physics,
        task=SimpleNamespace(
            _walker=SimpleNamespace(mjcf_model=_FakeModel(["free", "head_twist"]))
        ),
        close=lambda: closed.__setitem__("count", closed["count"] + 1),
    )
    adapter.initialize = lambda: None

    def _solver(*, physics, site_names, target_xpos, joint_names, **kwargs):
        captured["site_names"] = list(site_names)
        captured["target_xpos"] = target_xpos.tolist()
        return IKResult(qpos=[0.1, 0.2], err_norm=0.003, steps=14, success=True)

    adapter._ik_solver = _solver
    result = adapter.fit_3d(
        [
            Pose3DFrame(
                frame_index=0,
                video_name="demo.mp4",
                points={
                    "nose": (0.1, 0.0, 0.13),
                    "left_front_leg_tip": (0.09, 0.08, 0.0),
                },
            )
        ]
    )
    adapter.close()

    assert captured["site_names"] == ["walker/head", "walker/tarsus_T1_left"]
    assert result.frames[0].state["qpos"] == [0.1, 0.2]
    assert result.frames[0].diagnostics["ik_steps"] == 14
    assert result.frames[0].diagnostics["ik_success"] is True
    assert closed["count"] == 1


def test_flybody_adapter_dry_run_rollout_uses_connectome_controller() -> None:
    adapter = FlyBodyAdapter()
    adapter.configure({"dry_run": True, "behavior": "grooming"})
    result = adapter.rollout({"seed": 5}, steps=4)

    assert len(result.frames) == 4
    assert result.metadata["connectome"]["id"] == "google-malecns-flywire-cns-v1"
    assert "action" in result.frames[-1].state
    assert "thorax" in result.frames[-1].state["site_targets"]
    assert "brain" in result.frames[-1].diagnostics


def test_flybody_adapter_rollout_extracts_body_poses_and_sites_when_named_data_present() -> (
    None
):
    import numpy as np

    class _NamedTable(dict):
        def __init__(self, values: dict[str, list[float]]) -> None:
            super().__init__({k: np.asarray(v, dtype=float) for k, v in values.items()})
            self.axes = SimpleNamespace(row=_FakeRegularAxis(list(values.keys())))

    xpos = _NamedTable(
        {
            "walker/thorax": [2.0, 3.0, 0.13],
            "walker/head": [2.1, 3.0, 0.15],
        }
    )
    xquat = _NamedTable(
        {
            "walker/thorax": [1.0, 0.0, 0.0, 0.0],
            "walker/head": [1.0, 0.0, 0.0, 0.0],
        }
    )
    site_xpos = _NamedTable(
        {
            "walker/thorax": [2.0, 3.0, 0.13],
            "walker/head": [2.12, 3.0, 0.16],
        }
    )
    physics = SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(14, dtype=float)),
        named=SimpleNamespace(
            model=SimpleNamespace(
                site_pos=SimpleNamespace(
                    axes=SimpleNamespace(
                        row=_FakeRegularAxis(["walker/thorax", "walker/head"])
                    )
                ),
                dof_jntid=SimpleNamespace(
                    axes=SimpleNamespace(row=_FakeRaggedAxis(["walker/head_twist"]))
                ),
            ),
            data=SimpleNamespace(xpos=xpos, xquat=xquat, site_xpos=site_xpos),
        ),
    )
    spec = SimpleNamespace(
        name="adhere_claw_T1_left\thead_twist",
        shape=(2,),
        dtype=np.float64,
        minimum=np.array([0.0, -1.0]),
        maximum=np.array([1.0, 1.0]),
    )
    timestep = SimpleNamespace(observation={}, last=lambda: False)
    adapter = FlyBodyAdapter()
    adapter.configure({"behavior": "walk_imitation", "dynamics": "spiking"})
    adapter._env_factory = lambda **kwargs: SimpleNamespace(
        physics=physics,
        action_spec=lambda: spec,
        control_timestep=lambda: 0.002,
        reset=lambda: timestep,
        step=lambda action: timestep,
    )
    adapter.initialize = lambda: None

    result = adapter.rollout({"seed": 3}, steps=2)

    state0 = result.frames[0].state
    assert state0["body_poses"]["thorax"]["position"] == [0.0, 0.0, 0.13]
    assert state0["site_targets"]["thorax"] == [0.0, 0.0, 0.13]
    assert state0["model_pose"]["scale"] == 7.5
