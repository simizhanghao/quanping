"""Calibration / discrimination metrics on ConfidenceRecords (P1.5-B).

Generic: gold class · predicted class · class_scores · confidence.
No business field names. No sklearn dependency.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from linguaeval.core.schema import ConfidenceRecord

STATUS_AVAILABLE = "AVAILABLE"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUS_INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"


def _block(
    status: str,
    value: Any = None,
    *,
    reason: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": status, "value": value}
    if reason:
        out["reason"] = reason
    out.update(extra)
    return out


def _usable(records: Sequence[ConfidenceRecord]) -> List[ConfidenceRecord]:
    out: List[ConfidenceRecord] = []
    for r in records:
        if r.status != STATUS_AVAILABLE:
            continue
        if r.gold is None or r.class_scores is None or r.confidence is None:
            continue
        out.append(r)
    return out


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Top-label ECE with equal-width bins on [0, 1]."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins: List[Dict[str, Any]] = [
        {"bin": i, "lo": i / n_bins, "hi": (i + 1) / n_bins, "count": 0, "acc": 0.0, "conf": 0.0}
        for i in range(n_bins)
    ]
    n = len(confidences)
    if n == 0:
        return 0.0, bins

    for c, ok in zip(confidences, correct):
        conf = min(max(float(c), 0.0), 1.0)
        idx = min(int(conf * n_bins), n_bins - 1)
        b = bins[idx]
        b["count"] += 1
        b["acc"] += 1.0 if ok else 0.0
        b["conf"] += conf

    ece = 0.0
    for b in bins:
        if b["count"] == 0:
            continue
        acc = b["acc"] / b["count"]
        conf = b["conf"] / b["count"]
        b["acc"] = acc
        b["conf"] = conf
        ece += (b["count"] / n) * abs(acc - conf)
    return float(ece), bins


def multiclass_brier(records: Sequence[ConfidenceRecord]) -> float:
    label_set = set()
    for r in records:
        label_set.add(str(r.gold))
        label_set.update(str(k) for k in (r.class_scores or {}))
    labels = sorted(label_set)
    if not records or not labels:
        return 0.0
    total = 0.0
    for r in records:
        scores = r.class_scores or {}
        g = str(r.gold)
        s = 0.0
        for lab in labels:
            y = 1.0 if lab == g else 0.0
            p = float(scores.get(lab, 0.0))
            s += (p - y) ** 2
        total += s
    return total / len(records)


def mean_nll(records: Sequence[ConfidenceRecord], *, eps: float = 1e-12) -> float:
    if not records:
        return 0.0
    total = 0.0
    for r in records:
        p = float((r.class_scores or {}).get(str(r.gold), 0.0))
        total += -math.log(max(p, eps))
    return total / len(records)


def roc_auc_binary(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """ROC-AUC via Mann–Whitney (tie-aware). None if a class is missing."""
    pos = [float(s) for s, y in zip(scores, labels) if y]
    neg = [float(s) for s, y in zip(scores, labels) if not y]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    # rank all scores; average ranks for ties
    paired = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: x[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based ranks
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(rank for rank, (_, y) in zip(ranks, paired) if y == 1)
    # AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def multiclass_ovr_macro_auc(records: Sequence[ConfidenceRecord]) -> Tuple[Optional[float], Dict[str, Any]]:
    labels = sorted({str(r.gold) for r in records} | {str(k) for r in records for k in (r.class_scores or {})})
    per_class: Dict[str, Any] = {}
    vals: List[float] = []
    for lab in labels:
        scores = [float((r.class_scores or {}).get(lab, 0.0)) for r in records]
        y = [str(r.gold) == lab for r in records]
        auc = roc_auc_binary(scores, y)
        if auc is None:
            per_class[lab] = _block(
                STATUS_NOT_APPLICABLE,
                None,
                reason="single_class_in_labels",
            )
        else:
            per_class[lab] = _block(STATUS_AVAILABLE, auc)
            vals.append(auc)
    if not vals:
        return None, per_class
    return sum(vals) / len(vals), per_class


def compute_calibration_metrics(
    records: Sequence[ConfidenceRecord],
    *,
    n_bins: int = 10,
    min_samples: int = 10,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Aggregate calibration + discrimination metrics with availability statuses."""
    usable = _usable(records)
    n_usable = len(usable)
    base = {
        "n_records": len(records),
        "n_usable": n_usable,
        "n_bins": n_bins,
        "min_samples": min_samples,
    }

    if n_usable == 0:
        reason = "confidence_source_unavailable"
        empty = {
            "ece": _block(STATUS_NOT_AVAILABLE, None, reason=reason),
            "brier": _block(STATUS_NOT_AVAILABLE, None, reason=reason),
            "nll": _block(STATUS_NOT_AVAILABLE, None, reason=reason),
            "auroc_ovr_macro": _block(STATUS_NOT_AVAILABLE, None, reason=reason),
            "accuracy": _block(STATUS_NOT_AVAILABLE, None, reason=reason),
        }
        return {
            **base,
            "status": STATUS_NOT_AVAILABLE,
            "reason": reason,
            "metrics": empty,
        }

    confidences = [float(r.confidence) for r in usable]
    correct = [str(r.gold) == str(r.prediction) for r in usable]
    acc = sum(1 for ok in correct if ok) / n_usable
    ece_val, bins = expected_calibration_error(confidences, correct, n_bins=n_bins)
    brier_val = multiclass_brier(usable)
    nll_val = mean_nll(usable, eps=eps)
    auc_val, auc_per = multiclass_ovr_macro_auc(usable)

    # Brier/NLL/accuracy are defined for n>=1; ECE/AUROC may need min_samples.
    insuff = n_usable < min_samples
    pack_status = STATUS_INSUFFICIENT_SUPPORT if insuff else STATUS_AVAILABLE
    pack_reason = "n_usable_below_min_samples" if insuff else None

    if auc_val is None:
        auc_block = _block(
            STATUS_NOT_APPLICABLE,
            None,
            reason="auroc_undefined_for_label_support",
            per_class=auc_per,
        )
    elif insuff:
        auc_block = _block(
            STATUS_INSUFFICIENT_SUPPORT,
            auc_val,
            reason="n_usable_below_min_samples",
            per_class=auc_per,
        )
    else:
        auc_block = _block(STATUS_AVAILABLE, auc_val, per_class=auc_per)

    metrics = {
        "ece": _block(
            STATUS_INSUFFICIENT_SUPPORT if insuff else STATUS_AVAILABLE,
            ece_val,
            reason=("n_usable_below_min_samples" if insuff else None),
            bins=bins,
        ),
        "brier": _block(STATUS_AVAILABLE, brier_val),
        "nll": _block(STATUS_AVAILABLE, nll_val),
        "auroc_ovr_macro": auc_block,
        "accuracy": _block(STATUS_AVAILABLE, acc),
    }

    out: Dict[str, Any] = {
        **base,
        "status": pack_status,
        "metrics": metrics,
    }
    if pack_reason:
        out["reason"] = pack_reason
    return out
