from __future__ import annotations

from pathlib import Path

from annolid.simulation import load_simulation_mapping


def test_checked_flybody_template_loads() -> None:
    template_path = Path("annolid/configs/flybody_template.yaml")

    mapping = load_simulation_mapping(template_path)

    assert mapping.backend == "flybody"
    assert mapping.keypoint_to_site["nose"] == "head_site"
    assert mapping.coordinate_system["units"] == "meters"
    assert mapping.metadata["template"] == "flybody"
