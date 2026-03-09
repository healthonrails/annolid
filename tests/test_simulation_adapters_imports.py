from __future__ import annotations

import importlib
import sys


def test_simulation_adapters_package_does_not_eager_import_flybody_adapter() -> None:
    sys.modules.pop("annolid.simulation.adapters", None)
    sys.modules.pop("annolid.simulation.adapters.flybody", None)

    module = importlib.import_module("annolid.simulation.adapters")

    assert "annolid.simulation.adapters.flybody" not in sys.modules
    assert hasattr(module, "IdentitySimulationAdapter")
    assert "annolid.simulation.adapters.flybody" not in sys.modules
