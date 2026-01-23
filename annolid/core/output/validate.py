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
_VALIDATOR_CACHE: Any = None


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


def _format_validation_error(exc: Any) -> str:
    message = getattr(exc, "message", None) or str(exc)
    path = getattr(exc, "absolute_path", None)
    if path:
        pointer = "/" + "/".join(str(part) for part in path)
    else:
        pointer = "$"
    validator = getattr(exc, "validator", None)
    if validator:
        return f"{message} (at {pointer}; validator={validator})"
    return f"{message} (at {pointer})"


def validate_agent_record(record: Dict[str, Any]) -> None:
    """Validate a single NDJSON record against the agent output schema."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "jsonschema is required to validate agent outputs. Install it with: pip install jsonschema"
        ) from exc
    try:
        global _VALIDATOR_CACHE
        if _VALIDATOR_CACHE is None:
            schema = load_agent_output_schema().schema
            _VALIDATOR_CACHE = jsonschema.Draft202012Validator(
                schema, format_checker=jsonschema.FormatChecker()
            )
        first_error = next(_VALIDATOR_CACHE.iter_errors(record), None)
        if first_error is not None:
            raise AgentOutputValidationError(_format_validation_error(first_error))
    except jsonschema.ValidationError as exc:
        raise AgentOutputValidationError(_format_validation_error(exc)) from exc
