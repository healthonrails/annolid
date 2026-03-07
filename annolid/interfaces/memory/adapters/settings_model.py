from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SettingsSnapshot:
    """A structured representation of workspace or project settings at a point in time."""

    id: str
    description: str
    settings: Dict[str, Any]
    timestamp_ms: int
    context: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "settings": self.settings,
            "timestamp_ms": self.timestamp_ms,
            "context": self.context,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SettingsSnapshot":
        return cls(
            id=data.get("id", ""),
            description=data.get("description", ""),
            settings=data.get("settings", {}),
            timestamp_ms=data.get("timestamp_ms", 0),
            context=data.get("context"),
            tags=data.get("tags", []),
        )
