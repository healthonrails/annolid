"""Behavior event types for the domain layer."""

from annolid.behavior.event_utils import normalize_event_label
from annolid.core.types.behavior import BehaviorEvent, BehaviorSpan

__all__ = [
    "BehaviorEvent",
    "BehaviorSpan",
    "normalize_event_label",
]
