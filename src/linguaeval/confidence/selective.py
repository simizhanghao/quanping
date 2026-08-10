"""Selective prediction / Risk-Coverage (P1.5-D).

Rank by confidence (high→auto); abstain on the rest (fallback).
Metrics: RC curve, AURC, Risk@Coverage, Coverage@Risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from linguaeval.core.schema import ConfidenceRecord, SampleRecord, SelectiveSpec

STATUS_AVAILABLE = "AVAILABLE"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"
STATUS_INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"


@dataclass
class SelectiveRow:
    sample_id: str
    confidence: float
    correct: bool
    split_role: str


def _split_role(sample: Optional[SampleRecord]) -> str:
    if sample is None:
        return "test"
    meta = sample.meta or {}
    return str(meta.get("split_role") or meta.get("split") or "test").strip().lower()


def rows_from_records(
    records: Sequence[ConfidenceRecord],
    samples: Sequence[SampleRecord],
) -> List[SelectiveRow]:
    by_id = {s.sample_id: s for s in samples}
    out: List[SelectiveRow] = []
    for r in records:
        if r.status != STATUS_AVAILABLE:
            continue
        if r.confidence is None or r.gold is None or r.prediction is None:
            continue
        s = by_id.get(r.sample_id)
        out.append(
            SelectiveRow(
                sample_id=r.sample_id,
                confidence=float(r.confidence),
                correct=str(r.gold) == str(r.prediction),
                split_role=_split_role(s),
            )
        )
    return out


def filter_split(rows: Sequence[SelectiveRow], evaluate_on: str) -> List[SelectiveRow]:
    role = str(evaluate_on).strip().lower()
    if role in {"", "all", "*"}:
        return list(rows)
    return [r for r in rows if r.split_role == role]


def risk_coverage_curve(rows: Sequence[SelectiveRow]) -> List[Dict[str, Any]]:
    """Sort by confidence desc; accept top-k; risk = error rate among accepted."""
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: (-r.confidence, r.sample_id))
    n = len(ordered)
    errors = 0
    points: List[Dict[str, Any]] = []
    for k, r in enumerate(ordered, start=1):
        if not r.correct:
            errors += 1
        points.append(
            {
                "k": k,
                "coverage": k / n,
                "risk": errors / k,
                "n_accepted": k,
                "n_errors": errors,
                "min_confidence": r.confidence,
            }
        )
    return points


def aurc_from_curve(curve: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Trapezoidal AURC on [0,1] coverage (lower is better)."""
    if not curve:
        return None
    xs = [0.0] + [float(p["coverage"]) for p in curve]
    ys = [float(curve[0]["risk"])] + [float(p["risk"]) for p in curve]
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return float(area)


def risk_at_coverage(curve: Sequence[Dict[str, Any]], coverage: float) -> Optional[float]:
    if not curve:
        return None
    n = curve[-1]["n_accepted"]
    c = min(max(float(coverage), 0.0), 1.0)
    if c <= 0:
        return 0.0
    k = max(1, min(n, int(round(c * n))))
    return float(curve[k - 1]["risk"])


def coverage_at_risk(curve: Sequence[Dict[str, Any]], risk_max: float) -> Optional[float]:
    """Largest coverage with risk <= risk_max; None if none."""
    best: Optional[float] = None
    floor = float(risk_max)
    for p in curve:
        if float(p["risk"]) <= floor + 1e-15:
            best = float(p["coverage"])
    return best


def compute_selective_metrics(
    records: Sequence[ConfidenceRecord],
    samples: Sequence[SampleRecord],
    spec: SelectiveSpec,
    *,
    min_samples: int = 10,
) -> Dict[str, Any]:
    rows_all = rows_from_records(records, samples)
    rows = filter_split(rows_all, spec.evaluate_on)
    base = {
        "target": spec.target,
        "evaluate_on": spec.evaluate_on,
        "n_scored_all": len(rows_all),
        "n_evaluate": len(rows),
        "min_samples": min_samples,
    }
    if not rows:
        return {
            **base,
            "status": STATUS_NOT_AVAILABLE,
            "reason": "confidence_source_unavailable",
            "risk_coverage_curve": [],
            "aurc": None,
            "risk_at_coverage": {},
            "coverage_at_risk": {},
            "full_coverage_risk": None,
            "accuracy_full": None,
        }

    curve = risk_coverage_curve(rows)
    aurc = aurc_from_curve(curve)
    n_correct = sum(1 for r in rows if r.correct)
    acc = n_correct / len(rows)
    rac = {str(c): risk_at_coverage(curve, c) for c in spec.coverage_targets}
    car = {str(r): coverage_at_risk(curve, r) for r in spec.risk_targets}

    status = STATUS_AVAILABLE
    reason = None
    if len(rows) < min_samples:
        status = STATUS_INSUFFICIENT_SUPPORT
        reason = "n_evaluate_below_min_samples"

    return {
        **base,
        "status": status,
        "reason": reason,
        "risk_coverage_curve": curve,
        "aurc": aurc,
        "risk_at_coverage": rac,
        "coverage_at_risk": car,
        "full_coverage_risk": float(curve[-1]["risk"]) if curve else None,
        "accuracy_full": acc,
    }
