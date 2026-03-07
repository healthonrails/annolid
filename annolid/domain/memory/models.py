from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class MemoryRecord:
    text: str
    scope: str = "global"
    category: str = "other"
    source: str = "system"
    importance: float = 0.5
    timestamp_ms: Optional[int] = None
    token_count: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryHit:
    id: str
    text: str
    score: float
    scope: str
    category: str
    source: str
    importance: float
    timestamp_ms: int
    tags: List[str]
    metadata: dict[str, Any]
