"""Minimal JSON-path getter: supports ``$.a.b`` and ``a.b``."""

from __future__ import annotations

from typing import Any, Optional


def normalize_path(path: str) -> str:
    p = (path or "").strip()
    if p.startswith("$."):
        return p[2:]
    if p.startswith("$"):
        return p[1:].lstrip(".")
    return p


def get_by_path(obj: Any, path: str, default: Any = None) -> Any:
    """Resolve dotted path against dict-like objects."""
    if not path:
        return default
    cur: Any = obj
    for key in normalize_path(path).split("."):
        if key == "":
            continue
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def condition_holds(gold_or_parsed: dict, condition: Optional[dict]) -> bool:
    """condition: {field|path, equals}."""
    if not condition:
        return True
    path = condition.get("path") or condition.get("field")
    if path is None:
        return True
    left = get_by_path(gold_or_parsed, str(path), default=None)
    if "equals" in condition:
        return left == condition["equals"]
    return True
