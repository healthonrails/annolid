from __future__ import annotations

from types import SimpleNamespace

from annolid.simulation.flybody_live import (
    _candidate_body_names,
    _candidate_site_names,
    _edges_for_labels,
    _yaw_from_root_quaternion,
    generate_live_rollout_payload,
)


class _Axis:
    def __init__(self, names: list[str]) -> None:
        self._names_to_offsets = {name: index for index, name in enumerate(names)}


def _fake_physics(site_names: list[str], body_names: list[str]):
    return SimpleNamespace(
        named=SimpleNamespace(
            model=SimpleNamespace(
                site_pos=SimpleNamespace(axes=SimpleNamespace(row=_Axis(site_names))),
            ),
            data=SimpleNamespace(
                xpos=SimpleNamespace(axes=SimpleNamespace(row=_Axis(body_names))),
            ),
        )
    )


def test_candidate_site_names_only_selects_walker_sites() -> None:
    physics = _fake_physics(
        [
            "walker/tarsus_T1_left",
            "ghost/tarsus_T1_left",
            "walker/claw_T1_left",
            "ghost/claw_T1_left",
        ],
        [],
    )

    names = _candidate_site_names(physics)

    assert names == ["walker/tarsus_T1_left", "walker/claw_T1_left"]


def test_candidate_body_names_only_selects_walker_bodies() -> None:
    physics = _fake_physics(
        [],
        [
            "walker/thorax",
            "ghost/thorax",
            "walker/head",
            "ghost/head",
        ],
    )

    names = _candidate_body_names(physics)

    assert names == ["walker/thorax", "walker/head"]


def test_yaw_from_root_quaternion_uses_z_axis_heading() -> None:
    yaw = _yaw_from_root_quaternion([0.70710678, 0.0, 0.0, 0.70710678])

    assert abs(yaw - 1.57079632679) < 1e-3


def test_edges_for_labels_supports_real_flybody_mujoco_sites() -> None:
    edges = _edges_for_labels(
        ["thorax", "head", "tarsus_T1_left", "claw_T1_left", "tarsus_T3_right"]
    )
    assert ["head", "thorax"] in edges
    assert ["thorax", "tarsus_T1_left"] in edges
    assert ["tarsus_T1_left", "claw_T1_left"] in edges


def test_generate_live_rollout_payload_uses_connectome_and_thorax_frame(
    monkeypatch,
) -> None:
    import numpy as np

    site_names = ["walker/thorax", "walker/head", "walker/tarsus_T1_left"]

    class _NamedTable:
        def __init__(self, values: dict[str, list[float]]) -> None:
            self._values = values
            self.axes = SimpleNamespace(row=_Axis(list(values.keys())))

        def __getitem__(self, key: str):
            return np.asarray(self._values[key], dtype=float)

    # Simulate walk_on_ball where qpos[:4] is a ball quaternion [0.5, ...], NOT thorax XYZ!
    fake_physics = SimpleNamespace(
        data=SimpleNamespace(qpos=np.asarray([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])),
        named=SimpleNamespace(
            model=SimpleNamespace(
                site_pos=SimpleNamespace(axes=SimpleNamespace(row=_Axis(site_names)))
            ),
            data=SimpleNamespace(
                site_xpos=_NamedTable(
                    {
                        "walker/thorax": [0.0, 0.0, 0.128],
                        "walker/head": [0.09, 0.0, 0.14],
                        "walker/tarsus_T1_left": [0.08, 0.07, 0.002],
                    }
                ),
                xpos=_NamedTable(
                    {
                        "walker/thorax": [0.0, 0.0, 0.128],
                        "walker/head": [0.09, 0.0, 0.14],
                        "walker/claw_T1_left": [0.09, 0.08, 0.002],
                    }
                ),
                xquat=_NamedTable(
                    {
                        "walker/thorax": [1.0, 0.0, 0.0, 0.0],
                        "walker/head": [1.0, 0.0, 0.0, 0.0],
                        "walker/claw_T1_left": [1.0, 0.0, 0.0, 0.0],
                    }
                ),
            ),
        ),
    )

    class _FakeEnv:
        physics = fake_physics

        def control_timestep(self) -> float:
            return 0.002

        def action_spec(self):
            return SimpleNamespace(
                name="adhere_claw_T1_left\tcoxa_T1_left\tfemur_T1_left\ttibia_T1_left",
                shape=(4,),
                dtype=np.float64,
                minimum=np.asarray([0.0, -1.0, -1.0, -1.0]),
                maximum=np.asarray([1.0, 1.5, 1.5, 1.5]),
            )

        def reset(self):
            return SimpleNamespace(
                observation={"walker/touch": np.ones(6)},
                last=lambda: False,
            )

        def step(self, action):
            assert action.shape == (4,)
            return SimpleNamespace(
                observation={"walker/touch": np.ones(6)},
                last=lambda: False,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "annolid.simulation.flybody_live._build_behavior_env",
        lambda behavior: _FakeEnv(),
    )

    payload = generate_live_rollout_payload(
        steps=3,
        behavior="walk_on_ball",
        seed=7,
        substeps_per_frame=2,
    )

    assert payload["adapter"] == "flybody-live"
    assert payload["metadata"]["run_metadata"]["controller"] == "connectome"
    assert "connectome" in payload["metadata"]["run_metadata"]
    assert ["head", "thorax"] in payload["edges"]
    frame0 = payload["frames"][0]
    # Thorax pose is anchored at [0, 0, 0] rather than reading ball quaternion qpos[0]=0.5
    assert frame0["model_pose"]["position"] == [0.0, 0.0, 0.0]
    assert "claw_T1_left" in frame0["body_poses"]
    assert frame0["diagnostics"]["brain"]["dominant_circuit"] == "dnp09_forward_walk"
