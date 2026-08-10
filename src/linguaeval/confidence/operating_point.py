"""Generic operating-point / threshold selection (P1.5-C).

Binary engine; multiclass via one-vs-rest (positive_class).
Never optimize on split_role=test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from linguaeval.core.schema import ConfidenceRecord, OperatingPointSpec, SampleRecord

STATUS_AVAILABLE = "AVAILABLE"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"
STATUS_NO_FEASIBLE = "NO_FEASIBLE_OPERATING_POINT"
STATUS_TEST_LEAKAGE = "TEST_LEAKAGE"

_ALLOWED_OPTIMIZE = frozenset({"validation", "calibration"})


class OperatingPointError(ValueError):
    """Hard failure (e.g. test leakage)."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class ScoredBinaryRow:
    sample_id: str
    score: float
    gold_pos: bool
    split_role: str


def _split_role(sample: SampleRecord) -> str:
    meta = sample.meta or {}
    role = meta.get("split_role") or meta.get("split") or "test"
    return str(role).strip().lower()


def _is_positive(gold: Any, positive_class: str) -> bool:
    if gold is None:
        return False
    if isinstance(gold, bool):
        return gold is (positive_class.lower() in {"true", "1", "yes"})
    return str(gold) == str(positive_class)


def _positive_score(rec: ConfidenceRecord, positive_class: str) -> Optional[float]:
    if rec.status != STATUS_AVAILABLE or not rec.class_scores:
        return None
    if positive_class in rec.class_scores:
        return float(rec.class_scores[positive_class])
    # allow bool-ish keys
    for k, v in rec.class_scores.items():
        if str(k) == str(positive_class):
            return float(v)
    return None


def rows_from_records(
    records: Sequence[ConfidenceRecord],
    samples: Sequence[SampleRecord],
    *,
    positive_class: str,
) -> List[ScoredBinaryRow]:
    by_id = {s.sample_id: s for s in samples}
    out: List[ScoredBinaryRow] = []
    for r in records:
        score = _positive_score(r, positive_class)
        if score is None:
            continue
        s = by_id.get(r.sample_id)
        role = _split_role(s) if s is not None else "test"
        out.append(
            ScoredBinaryRow(
                sample_id=r.sample_id,
                score=float(score),
                gold_pos=_is_positive(r.gold, positive_class),
                split_role=role,
            )
        )
    return out


def metrics_at_threshold(rows: Sequence[ScoredBinaryRow], threshold: float, *, beta: float = 1.0) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for r in rows:
        pred = r.score >= threshold
        if pred and r.gold_pos:
            tp += 1
        elif pred and not r.gold_pos:
            fp += 1
        elif (not pred) and r.gold_pos:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    tpr = rec
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    b2 = beta * beta
    if prec == 0.0 and rec == 0.0:
        fbeta = 0.0
        f1 = 0.0
    else:
        fbeta = (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    n = len(rows)
    support_pos = tp + fn
    return {
        "threshold": threshold,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fbeta": fbeta,
        "tpr": tpr,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": n,
        "support_positive": support_pos,
        "predicted_positive": tp + fp,
        "coverage_predicted_positive": ((tp + fp) / n) if n else 0.0,
    }


def candidate_thresholds(rows: Sequence[ScoredBinaryRow]) -> List[float]:
    scores = sorted({round(r.score, 12) for r in rows})
    # include edge: always-negative via +eps above max, and 0.0
    out: List[float] = []
    if scores:
        out.append(scores[0])  # lowest score still can predict that point positive
        for s in scores[1:]:
            out.append(s)
        # threshold above all scores → all negative
        out.append(scores[-1] + 1e-9)
    else:
        out = [0.5]
    # unique sorted
    return sorted(set(out))


def sweep_curve(rows: Sequence[ScoredBinaryRow], *, beta: float = 1.0) -> List[Dict[str, Any]]:
    return [metrics_at_threshold(rows, t, beta=beta) for t in candidate_thresholds(rows)]


def _select_from_curve(
    curve: List[Dict[str, Any]],
    spec: OperatingPointSpec,
) -> Tuple[Optional[Dict[str, Any]], str]:
    mode = spec.mode
    if mode == "best_f1":
        # maximize f1, then precision, then higher threshold
        best = max(curve, key=lambda m: (m["f1"], m["precision"], m["threshold"]))
        return best, STATUS_AVAILABLE
    if mode == "best_fbeta":
        best = max(curve, key=lambda m: (m["fbeta"], m["precision"], m["threshold"]))
        return best, STATUS_AVAILABLE
    if mode == "max_recall_at_precision":
        floor = spec.precision_min
        if floor is None:
            raise ValueError("max_recall_at_precision requires constraint.precision.min")
        feasible = [m for m in curve if m["precision"] + 1e-15 >= float(floor)]
        if not feasible:
            return None, STATUS_NO_FEASIBLE
        best = max(feasible, key=lambda m: (m["recall"], m["precision"], m["threshold"]))
        return best, STATUS_AVAILABLE
    if mode == "max_precision_at_recall":
        floor = spec.recall_min
        if floor is None:
            raise ValueError("max_precision_at_recall requires constraint.recall.min")
        feasible = [m for m in curve if m["recall"] + 1e-15 >= float(floor)]
        if not feasible:
            return None, STATUS_NO_FEASIBLE
        best = max(feasible, key=lambda m: (m["precision"], m["recall"], m["threshold"]))
        return best, STATUS_AVAILABLE
    raise ValueError(f"unsupported operating_point mode: {mode}")


def select_operating_point(
    records: Sequence[ConfidenceRecord],
    samples: Sequence[SampleRecord],
    spec: OperatingPointSpec,
) -> Dict[str, Any]:
    """Select threshold on optimize_on split; evaluate frozen threshold on evaluate_on."""
    opt = str(spec.optimize_on).lower().strip()
    if opt == "test":
        raise OperatingPointError(
            STATUS_TEST_LEAKAGE,
            "optimize_on=test is forbidden (test_leakage)",
        )
    if opt not in _ALLOWED_OPTIMIZE:
        raise OperatingPointError(
            STATUS_TEST_LEAKAGE if opt == "test" else "invalid_optimize_on",
            f"optimize_on must be one of {sorted(_ALLOWED_OPTIMIZE)}, got {opt!r}",
        )

    rows = rows_from_records(records, samples, positive_class=spec.positive_class)
    if not rows:
        return {
            "status": STATUS_NOT_AVAILABLE,
            "reason": "confidence_source_unavailable",
            "target": spec.target,
            "positive_class": spec.positive_class,
            "optimize_on": opt,
            "evaluate_on": spec.evaluate_on,
            "mode": spec.mode,
            "selected": None,
            "test_evaluation": None,
            "threshold_curve": [],
        }

    opt_rows = [r for r in rows if r.split_role == opt]
    eval_role = str(spec.evaluate_on).lower().strip()
    eval_rows = [r for r in rows if r.split_role == eval_role]

    if not opt_rows:
        return {
            "status": STATUS_NOT_AVAILABLE,
            "reason": f"no_rows_on_optimize_split:{opt}",
            "target": spec.target,
            "positive_class": spec.positive_class,
            "optimize_on": opt,
            "evaluate_on": eval_role,
            "mode": spec.mode,
            "selected": None,
            "test_evaluation": None,
            "threshold_curve": [],
            "n_optimize": 0,
            "n_evaluate": len(eval_rows),
            "n_scored": len(rows),
        }

    curve = sweep_curve(opt_rows, beta=spec.beta)
    selected, st = _select_from_curve(curve, spec)
    out: Dict[str, Any] = {
        "status": st,
        "target": spec.target,
        "positive_class": spec.positive_class,
        "optimize_on": opt,
        "evaluate_on": eval_role,
        "mode": spec.mode,
        "beta": spec.beta,
        "constraint": {
            "precision_min": spec.precision_min,
            "recall_min": spec.recall_min,
        },
        "n_scored": len(rows),
        "n_optimize": len(opt_rows),
        "n_evaluate": len(eval_rows),
        "threshold_curve": curve,
        "selected": selected,
        "test_evaluation": None,
    }
    if st == STATUS_NO_FEASIBLE:
        out["reason"] = "no_threshold_satisfies_constraint"
        return out

    assert selected is not None
    thr = float(selected["threshold"])
    if eval_rows:
        out["test_evaluation"] = metrics_at_threshold(eval_rows, thr, beta=spec.beta)
    else:
        out["test_evaluation"] = {
            "status": STATUS_NOT_AVAILABLE,
            "reason": f"no_rows_on_evaluate_split:{eval_role}",
        }
    return out


def build_toy_binary_operating_point_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic binary toy (128 rows): 64 validation + 64 test.

    Validation is constructed so max-recall @ precision>=0.9 selects threshold 0.48
    with recall=1.0 (see tests). Purpose: functional validation, not model ranking.
    """
    val_pos = [round(0.99 - 0.01 * i, 2) for i in range(36)] + [0.58, 0.55, 0.52, 0.48]
    val_neg = [round(0.01 + 0.02 * i, 2) for i in range(20)] + [0.62, 0.60, 0.59, 0.56]
    # test: similar mass, slight shift (frozen eval only)
    test_pos = [round(min(0.99, s + 0.01), 2) for s in val_pos]
    test_neg = [round(max(0.0, s - 0.01), 2) for s in val_neg]

    samples: List[Dict[str, Any]] = []
    preds: List[Dict[str, Any]] = []

    def _add(split: str, gold_pos: bool, score: float, idx: int) -> None:
        sid = f"{split[:3]}_{idx:03d}"
        label = "fraud" if gold_pos else "legit"
        other = 1.0 - float(score)
        samples.append(
            {
                "sample_id": sid,
                "input": {"text": f"toy {sid}"},
                "gold": {"label": label},
                "meta": {"split_role": split, "language": "en"},
            }
        )
        pred_label = "fraud" if score >= 0.5 else "legit"
        preds.append(
            {
                "sample_id": sid,
                "model_id": "toy_op",
                "parsed": {"label": pred_label},
                "scores": {"label": {"fraud": float(score), "legit": float(other)}},
                "format": {"parse_ok": True, "schema_ok": True},
            }
        )

    i = 0
    for s in val_pos:
        _add("validation", True, s, i)
        i += 1
    for s in val_neg:
        _add("validation", False, s, i)
        i += 1
    i = 0
    for s in test_pos:
        _add("test", True, s, i)
        i += 1
    for s in test_neg:
        _add("test", False, s, i)
        i += 1
    return samples, preds
