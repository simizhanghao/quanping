"""ConfidenceExtractor — ConfidenceSpec → ConfidenceRecord (generic)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from linguaeval.core.paths import get_by_path
from linguaeval.core.schema import (
    ConfidenceRecord,
    ConfidenceSpec,
    PredictionRecord,
    SampleRecord,
    TaskSpec,
)

STATUS_AVAILABLE = "AVAILABLE"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"

_SUPPORTED_TYPES = {"probabilities", "logits", "logprob_margin", "none"}


def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    if not logits:
        return {}
    # stable softmax
    m = max(float(v) for v in logits.values())
    exps = {k: math.exp(float(v) - m) for k, v in logits.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


def _as_float_map(raw: Any) -> Optional[Dict[str, float]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        out: Dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                return None
        return out if out else None
    return None


def _resolve_raw_scores(pred: PredictionRecord, path: str) -> Any:
    from_pred = get_by_path(
        {
            "scores": pred.scores or {},
            "parsed": pred.parsed or {},
            "meta": pred.meta or {},
            "raw_output": pred.raw_output,
        },
        path,
        default=None,
    )
    if from_pred is not None:
        return from_pred
    if path.startswith("scores."):
        return get_by_path(pred.scores or {}, path[len("scores.") :], default=None)
    return get_by_path(pred.scores or {}, path, default=None)


def _normalize_class_scores(
    raw: Any,
    *,
    source_type: str,
    labels: Optional[Sequence[str]] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Return (class_probability_map, error_reason)."""
    if source_type == "none":
        return None, "source_type_none"

    fmap = _as_float_map(raw)
    if fmap is None:
        return None, "confidence_source_unavailable"

    if source_type == "probabilities":
        s = sum(fmap.values())
        if s <= 0:
            return None, "non_positive_probability_mass"
        # renormalize lightly if needed
        probs = {k: v / s for k, v in fmap.items()}
        if labels:
            for lab in labels:
                probs.setdefault(str(lab), 0.0)
        return probs, None

    if source_type == "logits":
        return _softmax(fmap), None

    if source_type == "logprob_margin":
        # interpret values as log-probs; convert via softmax then margin meta later
        return _softmax(fmap), None

    return None, f"unsupported_source_type:{source_type}"


def _scalar_confidence(
    probs: Dict[str, float],
    *,
    prediction: Any,
    source_type: str,
) -> float:
    if source_type == "logprob_margin":
        vals = sorted(probs.values(), reverse=True)
        if len(vals) >= 2:
            # margin on probability space after softmax of logprobs
            return float(vals[0] - vals[1])
        return float(vals[0]) if vals else 0.0
    pred_key = None if prediction is None else str(prediction)
    if pred_key is not None and pred_key in probs:
        return float(probs[pred_key])
    # multiclass default: max prob
    return float(max(probs.values())) if probs else 0.0


def extract_one(
    sample: SampleRecord,
    pred: PredictionRecord,
    *,
    spec: ConfidenceSpec,
    task: TaskSpec,
) -> ConfidenceRecord:
    target = next((t for t in task.targets if t.name == spec.target), None)
    if target is None:
        return ConfidenceRecord(
            sample_id=sample.sample_id,
            target=spec.target,
            status=STATUS_NOT_APPLICABLE,
            reason="target_not_in_task_spec",
            source_type=spec.source.type,
        )

    gold = get_by_path(sample.gold, target.path, default=None)
    pred_path = spec.predicted_path or target.path
    prediction = get_by_path(pred.parsed or {}, pred_path, default=None)

    stype = (spec.source.type or "probabilities").strip()
    if stype not in _SUPPORTED_TYPES:
        return ConfidenceRecord(
            sample_id=sample.sample_id,
            target=spec.target,
            status=STATUS_NOT_APPLICABLE,
            reason=f"unsupported_source_type:{stype}",
            gold=gold,
            prediction=prediction,
            source_type=stype,
        )

    if stype == "none":
        return ConfidenceRecord(
            sample_id=sample.sample_id,
            target=spec.target,
            status=STATUS_NOT_AVAILABLE,
            reason="confidence_source_unavailable",
            gold=gold,
            prediction=prediction,
            source_type=stype,
        )

    raw = _resolve_raw_scores(pred, spec.source.path)
    probs, err = _normalize_class_scores(raw, source_type=stype, labels=spec.labels or target.labels)
    if err or probs is None:
        return ConfidenceRecord(
            sample_id=sample.sample_id,
            target=spec.target,
            status=STATUS_NOT_AVAILABLE,
            reason=err or "confidence_source_unavailable",
            gold=gold,
            prediction=prediction,
            source_type=stype,
        )

    conf = _scalar_confidence(probs, prediction=prediction, source_type=stype)
    return ConfidenceRecord(
        sample_id=sample.sample_id,
        target=spec.target,
        status=STATUS_AVAILABLE,
        reason=None,
        gold=gold,
        prediction=prediction,
        class_scores=probs,
        confidence=conf,
        source_type=stype,
        meta={"n_classes": len(probs)},
    )


def extract_confidence_records(
    samples: List[SampleRecord],
    preds: List[PredictionRecord],
    *,
    spec: ConfidenceSpec,
    task: TaskSpec,
) -> List[ConfidenceRecord]:
    by_id = {p.sample_id: p for p in preds}
    out: List[ConfidenceRecord] = []
    for s in samples:
        p = by_id.get(s.sample_id)
        if p is None:
            out.append(
                ConfidenceRecord(
                    sample_id=s.sample_id,
                    target=spec.target,
                    status=STATUS_NOT_AVAILABLE,
                    reason="missing_prediction",
                    source_type=spec.source.type,
                )
            )
            continue
        out.append(extract_one(s, p, spec=spec, task=task))
    return out


def summarize_confidence(records: Sequence[ConfidenceRecord]) -> Dict[str, Any]:
    counts = {
        STATUS_AVAILABLE: 0,
        STATUS_NOT_AVAILABLE: 0,
        STATUS_NOT_APPLICABLE: 0,
    }
    reasons: Dict[str, int] = {}
    for r in records:
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.reason:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
    n = len(records)
    return {
        "n_records": n,
        "counts": counts,
        "reason_counts": reasons,
        "availability_rate": (counts[STATUS_AVAILABLE] / n) if n else None,
    }
