from __future__ import annotations

from types import SimpleNamespace

from annolid.simulation.flybody_live import (
    _candidate_body_names,
    _candidate_site_names,
    _yaw_from_root_quaternion,
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
