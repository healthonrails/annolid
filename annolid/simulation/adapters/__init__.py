from __future__ import annotations

from typing import Any

__all__ = ["FlyBodyAdapter", "IdentitySimulationAdapter"]


def __getattr__(name: str) -> Any:
    """
    Lazy adapter exports.

    Keeps adapter package import-light so default Annolid paths do not eagerly
    import optional simulation backend modules.
    """
    if name == "IdentitySimulationAdapter":
        from annolid.simulation.adapters.identity import IdentitySimulationAdapter

        return IdentitySimulationAdapter
    if name == "FlyBodyAdapter":
        from annolid.simulation.adapters.flybody import FlyBodyAdapter

        return FlyBodyAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
