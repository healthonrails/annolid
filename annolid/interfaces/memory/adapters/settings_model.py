from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time


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


@dataclass
class SettingsProfile:
    """Structured settings profile for reusable workflow configuration."""

    name: str
    workflow: str
    settings: Dict[str, Any]
    id: str = ""
    workspace_id: Optional[str] = None
    project_id: Optional[str] = None
    model_name: Optional[str] = None
    output_dir: Optional[str] = None
    export_format: Optional[str] = None
    ui_preset: Optional[str] = None
    created_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    tags: list[str] = field(default_factory=list)
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "workflow": self.workflow,
            "workspace_id": self.workspace_id,
            "project_id": self.project_id,
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "export_format": self.export_format,
            "ui_preset": self.ui_preset,
            "settings": self.settings,
            "created_ms": self.created_ms,
            "updated_ms": self.updated_ms,
            "tags": self.tags,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SettingsProfile":
        profile = cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            workflow=data.get("workflow", ""),
            workspace_id=data.get("workspace_id"),
            project_id=data.get("project_id"),
            model_name=data.get("model_name"),
            output_dir=data.get("output_dir"),
            export_format=data.get("export_format"),
            ui_preset=data.get("ui_preset"),
            settings=data.get("settings", {}),
            created_ms=int(data.get("created_ms", 0) or 0),
            updated_ms=int(data.get("updated_ms", 0) or 0),
            tags=list(data.get("tags", []) or []),
            context=data.get("context"),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("SettingsProfile name must not be empty.")
        if not self.workflow.strip():
            raise ValueError("SettingsProfile workflow must not be empty.")
        if not isinstance(self.settings, dict):
            raise ValueError("SettingsProfile settings must be a dictionary.")
