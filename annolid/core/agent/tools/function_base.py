from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from typing import Any


class FunctionTool(ABC):
    """Nanobot-style function tool contract."""

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def aliases(self) -> tuple[str, ...]:
        """Optional legacy/compatibility aliases resolved by FunctionToolRegistry."""
        return ()

    @property
    def read_only(self) -> bool:
        """Whether this tool is side-effect-free and safe for parallel execution."""
        return False

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, "")

    def _validate(self, val: Any, schema: dict[str, Any], path: str) -> list[str]:
        t = schema.get("type")
        label = path or "parameter"
        if isinstance(t, list):
            allowed_types = [str(item) for item in t if isinstance(item, str)]
            if val is None:
                if "null" in allowed_types:
                    return []
                return [f"{label} should be one of {allowed_types}"]
            matched_type: str | None = None
            for candidate in allowed_types:
                if candidate == "null":
                    continue
                if candidate in ("integer", "number") and isinstance(val, bool):
                    continue
                expected = self._TYPE_MAP.get(candidate)
                if expected is not None and isinstance(val, expected):
                    matched_type = candidate
                    break
            if matched_type is None:
                return [f"{label} should be one of {allowed_types}"]
            t = matched_type
        if t == "null":
            return [] if val is None else [f"{label} should be null"]
        if t in ("integer", "number") and isinstance(val, bool):
            return [f"{label} should be {t}"]
        if (
            isinstance(t, str)
            and t in self._TYPE_MAP
            and not isinstance(val, self._TYPE_MAP[t])
        ):
            return [f"{label} should be {t}"]

        errors: list[str] = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if t in ("integer", "number"):
            if isinstance(val, float) and not math.isfinite(val):
                errors.append(f"{label} must be a finite number")
            else:
                if "minimum" in schema and val < schema["minimum"]:
                    errors.append(f"{label} must be >= {schema['minimum']}")
                if "exclusiveMinimum" in schema and val <= schema["exclusiveMinimum"]:
                    errors.append(f"{label} must be > {schema['exclusiveMinimum']}")
                if "maximum" in schema and val > schema["maximum"]:
                    errors.append(f"{label} must be <= {schema['maximum']}")
                if "exclusiveMaximum" in schema and val >= schema["exclusiveMaximum"]:
                    errors.append(f"{label} must be < {schema['exclusiveMaximum']}")
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
            pattern = schema.get("pattern")
            if isinstance(pattern, str) and pattern:
                try:
                    if re.search(pattern, val) is None:
                        errors.append(f"{label} must match pattern {pattern!r}")
                except re.error:
                    pass
        if t == "object":
            props = schema.get("properties", {})
            if not isinstance(props, dict):
                props = {}
            pattern_props = schema.get("patternProperties")
            if not isinstance(pattern_props, dict):
                pattern_props = {}
            additional_props = schema.get("additionalProperties", True)
            for key in schema.get("required", []):
                if key not in val:
                    errors.append(
                        f"missing required {path + '.' + key if path else key}"
                    )
            for key, value in val.items():
                child_path = path + "." + key if path else key
                if key in props:
                    errors.extend(self._validate(value, props[key], child_path))
                    continue
                matched_pattern = False
                for prop_pattern, prop_schema in pattern_props.items():
                    if not isinstance(prop_pattern, str) or not isinstance(
                        prop_schema, dict
                    ):
                        continue
                    try:
                        if re.search(prop_pattern, str(key)) is not None:
                            matched_pattern = True
                            errors.extend(
                                self._validate(value, prop_schema, child_path)
                            )
                    except re.error:
                        continue
                if matched_pattern:
                    continue
                if additional_props is False:
                    errors.append(f"unexpected parameter {child_path}")
                elif isinstance(additional_props, dict):
                    errors.extend(self._validate(value, additional_props, child_path))
        if t == "array":
            if "minItems" in schema and len(val) < schema["minItems"]:
                errors.append(
                    f"{label} must contain at least {schema['minItems']} items"
                )
            if "maxItems" in schema and len(val) > schema["maxItems"]:
                errors.append(
                    f"{label} must contain at most {schema['maxItems']} items"
                )
            if schema.get("uniqueItems") is True:
                seen_items: set[str] = set()
                for item in val:
                    try:
                        fingerprint = json.dumps(item, sort_keys=True, default=repr)
                    except Exception:
                        fingerprint = repr(item)
                    if fingerprint in seen_items:
                        errors.append(f"{label} items must be unique")
                        break
                    seen_items.add(fingerprint)
            if "items" in schema and isinstance(schema["items"], dict):
                for i, item in enumerate(val):
                    idx_path = f"{path}[{i}]" if path else f"[{i}]"
                    errors.extend(self._validate(item, schema["items"], idx_path))
        return errors

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
