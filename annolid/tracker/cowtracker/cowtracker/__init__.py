# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CoWTracker: Cost-Volume Free Warping-Based Dense Point Tracking."""


def __getattr__(name):
    """Lazy import to avoid import errors when dependencies are missing."""
    if name == "CoWTracker":
        from cowtracker.models.cowtracker import CoWTracker

        return CoWTracker
    if name == "CoWTrackerWindowed":
        from cowtracker.models.cowtracker_windowed import CoWTrackerWindowed

        return CoWTrackerWindowed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CoWTracker", "CoWTrackerWindowed"]
