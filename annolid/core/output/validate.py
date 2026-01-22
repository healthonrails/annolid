from __future__ import annotations

from dataclasses import dataclass
import json
import importlib.resources as resources
from typing import Any, Dict, Optional


class AgentOutputValidationError(ValueError):
    pass


@dataclass(frozen=True)
class AgentOutputSchema:
    schema: Dict[str, Any]


_SCHEMA_CACHE: Optional[AgentOutputSchema] = None


def load_agent_output_schema() -> AgentOutputSchema:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    schema_path = resources.files("annolid.core.output.schema").joinpath(
        "agent_output.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    _SCHEMA_CACHE = AgentOutputSchema(schema=schema)
    return _SCHEMA_CACHE


def validate_agent_record(record: Dict[str, Any]) -> None:
    """Validate a single NDJSON record against the agent output schema."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "jsonschema is required to validate agent outputs. Install it with: pip install jsonschema"
        ) from exc
    try:
        jsonschema.validate(instance=record, schema=load_agent_output_schema().schema)
    except jsonschema.ValidationError as exc:
        raise AgentOutputValidationError(str(exc)) from exc
