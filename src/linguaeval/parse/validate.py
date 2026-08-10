"""OutputSpec validator — required fields, light types, constraints."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from linguaeval.core.schema import OutputSpec


def _check_type(value: Any, expected: str) -> bool:
    t = (expected or "").strip().lower()
    if t in {"", "any"}:
        return True
    if t in {"string", "str", "text"}:
        return isinstance(value, str)
    if t in {"number", "float"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t in {"integer", "int"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if t in {"boolean", "bool"}:
        return isinstance(value, bool)
    if t in {"object", "dict"}:
        return isinstance(value, dict)
    if t in {"array", "list"}:
        return isinstance(value, list)
    return True


def validate_parsed(parsed: Dict[str, Any] | None, output: OutputSpec) -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {"errors": []}
    if parsed is None or not isinstance(parsed, dict):
        details["errors"].append("parsed_missing_or_not_object")
        return False, details

    schema = output.schema or {}
    required: List[str] = list(schema.get("required") or [])
    types: Dict[str, str] = dict(schema.get("types") or {})
    # also accept JSON-Schema-like properties.{k}.type
    props = schema.get("properties") or {}
    if isinstance(props, dict):
        for k, spec in props.items():
            if isinstance(spec, dict) and "type" in spec and k not in types:
                types[k] = str(spec["type"])

    for key in required:
        if key not in parsed or parsed[key] is None:
            details["errors"].append(f"missing_required:{key}")

    for key, expected in types.items():
        if key not in parsed:
            continue
        if not _check_type(parsed[key], expected):
            details["errors"].append(f"type_mismatch:{key}:expected_{expected}")

    constraints = output.constraints or {}
    if constraints.get("no_markdown"):
        # if any string value looks like markdown fence / heading noise
        for k, v in parsed.items():
            if isinstance(v, str) and re.search(r"```|^\s*#\s", v):
                details["errors"].append(f"markdown_violation:{k}")

    forbidden = list(constraints.get("forbidden_keys") or schema.get("forbidden") or [])
    for key in forbidden:
        if key in parsed:
            details["errors"].append(f"forbidden_key:{key}")

    ok = len(details["errors"]) == 0
    details["required"] = required
    return ok, details
