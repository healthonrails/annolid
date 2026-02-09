"""CoWTracker integration for Annolid.

This package provides a wrapper around the CoWTracker dense point tracking model.
"""


def __getattr__(name: str):
    if name == "CoWTrackerProcessor":
        from annolid.tracker.cowtracker.track import CoWTrackerProcessor

        return CoWTrackerProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CoWTrackerProcessor"]
