from __future__ import annotations

"""
Lightweight JSON-schema-like validator to avoid external dependencies.

Supported keywords (subset):
- type: "object" | "array" | "string" | "number" | "integer" | "boolean"
- required: [list of property names]
- properties: {name: schema}
- items: schema (for arrays)
"""

from typing import Any, Dict, List


class SchemaError(Exception):
    pass


class ValidationError(Exception):
    pass


def _type_ok(val: Any, t: str) -> bool:
    if t == "object":
        return isinstance(val, dict)
    if t == "array":
        return isinstance(val, list)
    if t == "string":
        return isinstance(val, str)
    if t == "number":
        return isinstance(val, (int, float))
    if t == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if t == "boolean":
        return isinstance(val, bool)
    return False


def validate(instance: Any, schema: Dict[str, Any], path: str = "$") -> None:
    stype = schema.get("type")
    if stype and not _type_ok(instance, stype):
        raise ValidationError(f"{path}: expected {stype}, got {type(instance).__name__}")

    if stype == "object":
        props: Dict[str, Dict[str, Any]] = schema.get("properties", {})
        required: List[str] = schema.get("required", [])
        for req in required:
            if req not in instance:
                raise ValidationError(f"{path}: missing required property '{req}'")
        for k, v in instance.items():
            if k in props:
                validate(v, props[k], path=f"{path}.{k}")
        return

    if stype == "array":
        item_schema = schema.get("items")
        if item_schema:
            for i, elem in enumerate(instance):
                validate(elem, item_schema, path=f"{path}[{i}]")
        return

    # primitives: nothing further

