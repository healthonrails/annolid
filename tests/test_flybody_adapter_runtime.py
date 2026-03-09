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
