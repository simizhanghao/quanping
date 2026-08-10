"""Simple CI-aware gate engine for paired compare (P1-C)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from linguaeval.core.paths import get_by_path


_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class GateError(ValueError):
    pass


def _as_number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def evaluate_gates(
    context: Dict[str, Any],
    gate_specs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate declarative gates against a metrics context dict.

    Each gate:
      id: optional
      path: dotted path into context
      op: >= | <= | > | < | == | !=
      value: threshold
    """
    results: List[Dict[str, Any]] = []
    overall = "PASS"
    for i, g in enumerate(gate_specs or []):
        gid = str(g.get("id") or f"gate_{i}")
        path = str(g.get("path") or "")
        op = str(g.get("op") or ">=")
        if op not in _OPS:
            raise GateError(f"unsupported gate op={op!r} for {gid}")
        if not path:
            raise GateError(f"gate {gid} missing path")
        raw = get_by_path(context, path, default=None)
        threshold = g.get("value")
        left = _as_number(raw)
        right = _as_number(threshold)
        if left is None or right is None:
            status = "ERROR"
            ok = False
            detail = f"non-numeric compare: left={raw!r} right={threshold!r}"
        else:
            ok = bool(_OPS[op](left, right))
            status = "PASS" if ok else "FAIL"
            detail = f"{left} {op} {right}"
        results.append(
            {
                "id": gid,
                "path": path,
                "op": op,
                "value": threshold,
                "observed": raw,
                "status": status,
                "detail": detail,
            }
        )
    n_error = sum(1 for r in results if r["status"] == "ERROR")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    if n_error:
        overall = "ERROR"
    elif n_fail:
        overall = "FAIL"
    else:
        overall = "PASS"
    return {
        "status": overall if results else "PASS",
        "n_gates": len(results),
        "n_pass": sum(1 for r in results if r["status"] == "PASS"),
        "n_fail": n_fail,
        "n_error": n_error,
        "gates": results,
    }
