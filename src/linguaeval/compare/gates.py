"""CI-aware gate engine with support policy (P1-C/D)."""

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


def _check_requirements(
    requirements: Dict[str, Any],
    context: Dict[str, Any],
) -> Optional[str]:
    """Return reason string if insufficient; else None."""
    if not requirements:
        return None
    support = context.get("support") or {}
    n_samples = support.get("n_samples")
    n_clusters = support.get("n_units")
    if n_clusters is None:
        n_clusters = support.get("n_clusters")

    min_samples = requirements.get("min_samples")
    if min_samples is not None:
        if n_samples is None or int(n_samples) < int(min_samples):
            return f"n_samples={n_samples} < min_samples={min_samples}"

    min_clusters = requirements.get("min_clusters")
    if min_clusters is not None:
        if n_clusters is None or int(n_clusters) < int(min_clusters):
            return f"n_clusters={n_clusters} < min_clusters={min_clusters}"

    # path-level NOT_APPLICABLE: if observed path parent has status
    return None


def evaluate_gates(
    context: Dict[str, Any],
    gate_specs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate declarative gates.

    Statuses per gate: PASS | FAIL | ERROR | INSUFFICIENT_SUPPORT | NOT_APPLICABLE

    Overall:
      ERROR if any ERROR
      FAIL if any FAIL
      INSUFFICIENT_SUPPORT if no FAIL/ERROR but any INSUFFICIENT_SUPPORT
      NOT_APPLICABLE if only N/A (+PASS)
      PASS if all PASS
    """
    results: List[Dict[str, Any]] = []
    for i, g in enumerate(gate_specs or []):
        gid = str(g.get("id") or f"gate_{i}")
        path = str(g.get("path") or "")
        op = str(g.get("op") or ">=")
        if op not in _OPS:
            raise GateError(f"unsupported gate op={op!r} for {gid}")
        if not path:
            raise GateError(f"gate {gid} missing path")

        req_reason = _check_requirements(dict(g.get("requirements") or {}), context)
        if req_reason:
            results.append(
                {
                    "id": gid,
                    "path": path,
                    "op": op,
                    "value": g.get("value"),
                    "observed": None,
                    "status": "INSUFFICIENT_SUPPORT",
                    "detail": req_reason,
                }
            )
            continue

        raw = get_by_path(context, path, default=None)
        # If path points into an applicability block with status
        if isinstance(raw, dict) and "status" in raw and "value" in raw:
            if raw.get("status") == "NOT_APPLICABLE":
                results.append(
                    {
                        "id": gid,
                        "path": path,
                        "op": op,
                        "value": g.get("value"),
                        "observed": raw,
                        "status": "NOT_APPLICABLE",
                        "detail": raw.get("reason") or "metric_not_applicable",
                    }
                )
                continue
            raw = raw.get("value")

        if raw is None:
            # missing statistic (e.g. CI disabled) → insufficient rather than ERROR
            results.append(
                {
                    "id": gid,
                    "path": path,
                    "op": op,
                    "value": g.get("value"),
                    "observed": None,
                    "status": "INSUFFICIENT_SUPPORT",
                    "detail": "observed_path_missing",
                }
            )
            continue

        threshold = g.get("value")
        left = _as_number(raw)
        right = _as_number(threshold)
        if left is None or right is None:
            status = "ERROR"
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
    n_insuff = sum(1 for r in results if r["status"] == "INSUFFICIENT_SUPPORT")
    n_na = sum(1 for r in results if r["status"] == "NOT_APPLICABLE")
    n_pass = sum(1 for r in results if r["status"] == "PASS")

    if n_error:
        overall = "ERROR"
    elif n_fail:
        overall = "FAIL"
    elif n_insuff:
        overall = "INSUFFICIENT_SUPPORT"
    elif n_na and n_pass == 0:
        overall = "NOT_APPLICABLE"
    else:
        overall = "PASS"

    return {
        "status": overall if results else "PASS",
        "n_gates": len(results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_error": n_error,
        "n_insufficient_support": n_insuff,
        "n_not_applicable": n_na,
        "gates": results,
    }
