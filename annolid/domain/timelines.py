"""Timeline-oriented value objects for behavior analysis."""

from annolid.behavior.time_budget import (
    BehaviorInterval,
    BinnedTimeBudgetRow,
    TimeBudgetRow,
)
from annolid.core.types.frame import FrameRef

__all__ = [
    "BehaviorInterval",
    "BinnedTimeBudgetRow",
    "FrameRef",
    "TimeBudgetRow",
]
