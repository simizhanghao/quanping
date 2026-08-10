"""D6 context ablation — with_context vs without_context (offline)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from linguaeval.compare.transitions import classify_transition
from linguaeval.core.schema import PredictionRecord, SampleRecord, TaskSpec
from linguaeval.metrics.score_records import build_score_records
from linguaeval.robustness.relations import values_equal


def build_context_ablation_records(
    *,
    samples: Sequence[SampleRecord],
    without_preds: Sequence[PredictionRecord],
    with_preds: Sequence[PredictionRecord],
    task: TaskSpec,
    target: str,
    denominator: str = "semantic",
) -> List[Dict[str, Any]]:
    """Pair without_context vs with_context ScoreRecords for one target."""
    if denominator not in {"semantic", "strict"}:
        raise ValueError("denominator must be semantic|strict")
    if target not in {t.name for t in task.targets}:
        raise KeyError(f"context.target={target!r} not in TaskSpec")

    without_scores = {r.sample_id: r for r in build_score_records(list(samples), list(without_preds), task)}
    with_scores = {r.sample_id: r for r in build_score_records(list(samples), list(with_preds), task)}

    rows: List[Dict[str, Any]] = []
    for s in samples:
        b = without_scores.get(s.sample_id)
        c = with_scores.get(s.sample_id)
        if b is None or c is None:
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "target": target,
                    "applicable": False,
                    "exclusion": "missing_prediction",
                    "conversation": s.conversation,
                }
            )
            continue
        bts = b.targets.get(target)
        cts = c.targets.get(target)
        if bts is None or cts is None or not bts.applicable or not cts.applicable:
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "target": target,
                    "applicable": False,
                    "exclusion": "target_not_applicable",
                    "conversation": s.conversation,
                }
            )
            continue
        format_ok = bool(b.parse_ok and b.schema_ok and c.parse_ok and c.schema_ok)
        if denominator == "semantic" and not format_ok:
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "target": target,
                    "applicable": False,
                    "exclusion": "excluded_format",
                    "conversation": s.conversation,
                }
            )
            continue
        b_ok = bool(bts.correct)
        c_ok = bool(cts.correct)
        flipped = not values_equal(bts.pred, cts.pred)
        rows.append(
            {
                "sample_id": s.sample_id,
                "target": target,
                "applicable": True,
                "without_context": {"pred": bts.pred, "correct": b_ok},
                "with_context": {"pred": cts.pred, "correct": c_ok},
                "prediction_flipped": flipped,
                "transition": classify_transition(b_ok, c_ok),
                "conversation": s.conversation,
            }
        )
    return rows


def aggregate_context_ablation(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    applicable = [r for r in records if r.get("applicable")]
    coverage = {
        "n_records": len(records),
        "n_applicable": len(applicable),
        "n_excluded": len(records) - len(applicable),
        "exclusion_counts": {},
    }
    for r in records:
        ex = r.get("exclusion")
        if ex:
            coverage["exclusion_counts"][ex] = coverage["exclusion_counts"].get(ex, 0) + 1

    if not applicable:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "no_applicable_context_pairs",
            "coverage": coverage,
            "by_target": {},
        }

    by_target: Dict[str, Any] = {}
    for tname in sorted({r["target"] for r in applicable}):
        rows = [r for r in applicable if r["target"] == tname]
        n = len(rows)
        acc_wo = sum(1 for r in rows if r["without_context"]["correct"]) / n
        acc_w = sum(1 for r in rows if r["with_context"]["correct"]) / n
        trans: Dict[str, int] = {}
        for r in rows:
            t = r.get("transition")
            if t:
                trans[t] = trans.get(t, 0) + 1
        by_target[tname] = {
            "n": n,
            "accuracy_without_context": acc_wo,
            "accuracy_with_context": acc_w,
            "delta_accuracy": acc_w - acc_wo,
            "prediction_flip_rate": sum(1 for r in rows if r.get("prediction_flipped")) / n,
            "context_gain_rate": trans.get("gain", 0) / n,
            "context_regression_rate": trans.get("regression", 0) / n,
            "transitions": trans,
        }
    return {
        "status": "AVAILABLE",
        "coverage": coverage,
        "by_target": by_target,
        "modes": ["without_context", "with_context"],
    }
