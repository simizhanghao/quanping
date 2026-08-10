"""D8 self-consistency — repeated predictions on the same sample (offline)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from linguaeval.core.paths import condition_holds, get_by_path
from linguaeval.core.schema import PredictionRecord, SampleRecord, TaskSpec
from linguaeval.robustness.relations import values_equal


def group_replicates(preds: Sequence[PredictionRecord]) -> Dict[str, List[PredictionRecord]]:
    """Group by sample_id; duplicate sample_id lines are replicates."""
    out: Dict[str, List[PredictionRecord]] = defaultdict(list)
    for p in preds:
        out[str(p.sample_id)].append(p)
    return dict(out)


def _target_pred(pred: PredictionRecord, task: TaskSpec, target: str) -> Tuple[Any, bool]:
    tspec = next((t for t in task.targets if t.name == target), None)
    if tspec is None:
        return None, False
    fmt_ok = bool(pred.format.parse_ok and pred.format.schema_ok)
    val = get_by_path(pred.parsed, tspec.path, default=None)
    return val, fmt_ok


def _majority(values: Sequence[Any]) -> Any:
    if not values:
        return None
    keys = [str(x) if x is not None else "null" for x in values]
    counts = Counter(keys)
    # tie-break: highest count, then earliest first-seen key
    best_key, _ = max(
        counts.items(),
        key=lambda kv: (kv[1], -keys.index(kv[0])),
    )
    for x, key in zip(values, keys):
        if key == best_key:
            return x
    return values[0]


def build_consistency_records(
    samples: Sequence[SampleRecord],
    preds: Sequence[PredictionRecord],
    task: TaskSpec,
    *,
    target: str,
    min_replicates: int = 2,
) -> List[Dict[str, Any]]:
    if target not in {t.name for t in task.targets}:
        raise KeyError(f"consistency.target={target!r} not in TaskSpec")
    tspec = next(t for t in task.targets if t.name == target)
    by_id = group_replicates(preds)
    rows: List[Dict[str, Any]] = []
    for s in samples:
        applicable = condition_holds(s.gold, tspec.condition)
        gold = get_by_path(s.gold, tspec.path, default=None)
        reps = by_id.get(s.sample_id) or []
        if not applicable:
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "target": target,
                    "applicable": False,
                    "exclusion": "target_not_applicable",
                    "n_replicates": len(reps),
                }
            )
            continue
        if len(reps) < min_replicates:
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "target": target,
                    "applicable": False,
                    "exclusion": "insufficient_replicates",
                    "n_replicates": len(reps),
                    "gold": gold,
                }
            )
            continue
        vals: List[Any] = []
        format_oks: List[bool] = []
        for p in reps:
            v, ok = _target_pred(p, task, target)
            vals.append(v)
            format_oks.append(ok)
        n = len(vals)
        pair_eq = 0
        pair_n = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_n += 1
                if values_equal(vals[i], vals[j]):
                    pair_eq += 1
        all_agree = all(values_equal(vals[0], vals[k]) for k in range(1, n))
        maj = _majority(vals)
        maj_correct = bool(all(format_oks) and values_equal(maj, gold))
        rows.append(
            {
                "sample_id": s.sample_id,
                "target": target,
                "applicable": True,
                "n_replicates": n,
                "gold": gold,
                "predictions": vals,
                "format_ok_all": all(format_oks),
                "pairwise_agreement": (pair_eq / pair_n) if pair_n else None,
                "all_agree": all_agree,
                "majority_pred": maj,
                "majority_correct": maj_correct,
                "meta": {
                    "replicate_ids": [
                        (p.meta or {}).get("replicate")
                        or (p.meta or {}).get("replicate_id")
                        or i
                        for i, p in enumerate(reps)
                    ]
                },
            }
        )
    return rows


def aggregate_consistency(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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
            "reason": "no_applicable_consistency_rows",
            "coverage": coverage,
            "by_target": {},
        }

    by_target: Dict[str, Any] = {}
    for tname in sorted({r["target"] for r in applicable}):
        rows = [r for r in applicable if r["target"] == tname]
        n = len(rows)
        pairwise = [r["pairwise_agreement"] for r in rows if r.get("pairwise_agreement") is not None]
        by_target[tname] = {
            "n": n,
            "mean_n_replicates": sum(r["n_replicates"] for r in rows) / n,
            "pairwise_agreement_rate": sum(pairwise) / len(pairwise) if pairwise else None,
            "all_agree_rate": sum(1 for r in rows if r.get("all_agree")) / n,
            "majority_accuracy": sum(1 for r in rows if r.get("majority_correct")) / n,
        }
    return {
        "status": "AVAILABLE",
        "coverage": coverage,
        "by_target": by_target,
    }
