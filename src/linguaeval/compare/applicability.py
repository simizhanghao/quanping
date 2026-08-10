"""Metric applicability — avoid fake zeros when a metric is undefined (P1-D)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from linguaeval.compare.bootstrap import PairRow
from linguaeval.metrics.classification import (
    _as_bool,
    _round_maybe,
    binary_confusion,
    metrics_from_confusion,
    multiclass_metrics,
)

STATUS_APPLICABLE = "APPLICABLE"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"


def _pack(
    status: str,
    value: Any = None,
    *,
    reason: Optional[str] = None,
    round_digits: Optional[int] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": status, "value": value}
    if reason:
        out["reason"] = reason
    if status == STATUS_APPLICABLE and isinstance(value, float) and round_digits is not None:
        out["value"] = _round_maybe(value, round_digits)
    return out


def binary_support(golds: Sequence[Any], preds: Sequence[Any]) -> Dict[str, int]:
    g_bool: List[bool] = []
    p_bool: List[bool] = []
    for g, p in zip(golds, preds):
        gb = _as_bool(g)
        if gb is None:
            continue
        pb = _as_bool(p)
        if pb is None:
            pb = not gb
        g_bool.append(gb)
        p_bool.append(pb)
    cm = binary_confusion(g_bool, p_bool) if g_bool else {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    pos = cm["TP"] + cm["FN"]
    neg = cm["TN"] + cm["FP"]
    pred_pos = cm["TP"] + cm["FP"]
    return {
        "n": len(g_bool),
        "positive_support": pos,
        "negative_support": neg,
        "predicted_positive": pred_pos,
        **cm,
    }


def metrics_with_applicability(
    rows: List[PairRow],
    indices: Sequence[int],
    *,
    side: str,
    target_type: str,
    metric_names: Sequence[str],
    labels: Optional[List[str]] = None,
    round_digits: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return per-metric APPLICABLE / NOT_APPLICABLE blocks (generic)."""
    golds: List[Any] = []
    preds: List[Any] = []
    for i in indices:
        r = rows[i]
        golds.append(r.gold)
        preds.append(r.baseline_pred if side == "baseline" else r.candidate_pred)

    out: Dict[str, Dict[str, Any]] = {}
    if not golds:
        for m in metric_names:
            out[str(m).lower()] = _pack(STATUS_NOT_APPLICABLE, reason="empty_slice")
        return out

    if target_type == "binary":
        support = binary_support(golds, preds)
        cm = {
            "TP": support["TP"],
            "TN": support["TN"],
            "FP": support["FP"],
            "FN": support["FN"],
        }
        raw = metrics_from_confusion(
            cm, ["precision", "recall", "f1", "f2", "accuracy"], round_digits=None
        )
        neg = support["negative_support"]
        specificity = (cm["TN"] / neg) if neg else None
        fpr = (cm["FP"] / neg) if neg else None

        for m in metric_names:
            key = str(m).lower()
            if key in {"recall", "f1", "f2"}:
                if support["positive_support"] <= 0:
                    out[key] = _pack(STATUS_NOT_APPLICABLE, reason="positive_support=0")
                else:
                    out[key] = _pack(
                        STATUS_APPLICABLE, float(raw[key]), round_digits=round_digits
                    )
            elif key == "precision":
                if support["positive_support"] <= 0:
                    out[key] = _pack(STATUS_NOT_APPLICABLE, reason="positive_support=0")
                else:
                    out[key] = _pack(
                        STATUS_APPLICABLE, float(raw["precision"]), round_digits=round_digits
                    )
            elif key == "accuracy":
                out[key] = _pack(
                    STATUS_APPLICABLE, float(raw["accuracy"]), round_digits=round_digits
                )
            elif key == "specificity":
                if neg <= 0:
                    out[key] = _pack(STATUS_NOT_APPLICABLE, reason="negative_support=0")
                else:
                    out[key] = _pack(
                        STATUS_APPLICABLE, float(specificity), round_digits=round_digits
                    )
            elif key in {"false_positive_rate", "fpr"}:
                if neg <= 0:
                    out[key] = _pack(STATUS_NOT_APPLICABLE, reason="negative_support=0")
                else:
                    out[key] = _pack(
                        STATUS_APPLICABLE, float(fpr), round_digits=round_digits
                    )
            else:
                out[key] = _pack(STATUS_NOT_APPLICABLE, reason=f"unsupported_metric:{key}")
        out["_support"] = {"status": STATUS_APPLICABLE, "value": support}
        return out

    if target_type == "multiclass":
        g_str = ["" if g is None else str(g) for g in golds]
        p_str = ["" if p is None else str(p) for p in preds]
        block = multiclass_metrics(
            g_str, p_str, list(metric_names), labels=labels, round_digits=None
        )
        for m in metric_names:
            key = str(m).lower()
            if key in block and isinstance(block[key], (int, float)):
                out[key] = _pack(
                    STATUS_APPLICABLE, float(block[key]), round_digits=round_digits
                )
            else:
                out[key] = _pack(STATUS_NOT_APPLICABLE, reason=f"unsupported_metric:{key}")
        return out

    n = len(golds)
    em = sum(1 for g, p in zip(golds, preds) if g == p) / n if n else 0.0
    for m in metric_names:
        key = str(m).lower()
        if key in {"exact_match", "accuracy"}:
            out[key] = _pack(STATUS_APPLICABLE, em, round_digits=round_digits)
        else:
            out[key] = _pack(STATUS_NOT_APPLICABLE, reason=f"unsupported_metric:{key}")
    return out


def pair_metric_delta(
    baseline_m: Dict[str, Dict[str, Any]],
    candidate_m: Dict[str, Dict[str, Any]],
    metric_names: Sequence[str],
    *,
    round_digits: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for m in metric_names:
        key = str(m).lower()
        if key.startswith("_"):
            continue
        b = baseline_m.get(key) or {}
        c = candidate_m.get(key) or {}
        if b.get("status") != STATUS_APPLICABLE or c.get("status") != STATUS_APPLICABLE:
            reason = b.get("reason") or c.get("reason") or "side_not_applicable"
            out[key] = {
                "status": STATUS_NOT_APPLICABLE,
                "baseline": b,
                "candidate": c,
                "delta": None,
                "reason": reason,
            }
            continue
        bv = float(b["value"])
        cv = float(c["value"])
        delta = cv - bv
        if round_digits is not None:
            delta = _round_maybe(delta, round_digits)
            bv = _round_maybe(bv, round_digits)
            cv = _round_maybe(cv, round_digits)
        out[key] = {
            "status": STATUS_APPLICABLE,
            "baseline": bv,
            "candidate": cv,
            "delta": delta,
        }
    return out
