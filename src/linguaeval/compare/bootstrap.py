"""Paired / cluster bootstrap for offline compare (P1-B).

- bootstrap_unit=sample: each row is an independent unit
- bootstrap_unit=<field>: rows sharing the same unit id are resampled together
  (cluster bootstrap). Typical: dialogue_id for multi-turn data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import random

from linguaeval.core.schema import ComparisonRecord, SampleRecord, ScoreRecord
from linguaeval.metrics.classification import (
    _as_bool,
    _round_maybe,
    binary_confusion,
    metrics_from_confusion,
    multiclass_metrics,
)


@dataclass
class PairRow:
    sample_id: str
    unit_id: str
    gold: Any
    baseline_pred: Any
    candidate_pred: Any
    transition: Optional[str]  # None if not transition-eligible


def resolve_unit_id(sample: SampleRecord, bootstrap_unit: str) -> str:
    """Map a sample to its bootstrap cluster id (generic; no business names)."""
    key = (bootstrap_unit or "sample").strip()
    if key in {"sample", "sample_id", "none", ""}:
        return sample.sample_id
    conv = sample.conversation or {}
    if key in conv and conv[key] is not None:
        return str(conv[key])
    meta = sample.meta or {}
    if key in meta and meta[key] is not None:
        return str(meta[key])
    # dotted convenience: conversation.dialogue_id already handled via conv
    if key.startswith("meta."):
        return str(meta.get(key[5:]) or sample.sample_id)
    if key.startswith("conversation."):
        return str(conv.get(key[len("conversation.") :]) or sample.sample_id)
    # fall back to sample — never invent business defaults
    return sample.sample_id


def build_cluster_map(unit_ids: Sequence[str]) -> Dict[str, List[int]]:
    """unit_id → row indices (order preserved)."""
    clusters: Dict[str, List[int]] = {}
    for i, uid in enumerate(unit_ids):
        clusters.setdefault(str(uid), []).append(i)
    return clusters


def resample_indices(
    n_rows: int,
    *,
    unit_ids: Optional[Sequence[str]] = None,
    rng: random.Random,
) -> List[int]:
    """Draw a bootstrap sample of row indices.

    If ``unit_ids`` is None or every unit is unique, this is ordinary bootstrap.
    Otherwise resample **clusters** with replacement, then expand to member rows
    (a cluster drawn twice contributes its rows twice).
    """
    if n_rows <= 0:
        return []
    if unit_ids is None:
        return [rng.randrange(n_rows) for _ in range(n_rows)]

    if len(unit_ids) != n_rows:
        raise ValueError("unit_ids length must equal n_rows")

    clusters = build_cluster_map(unit_ids)
    unit_keys = list(clusters.keys())
    # Ordinary bootstrap iff one row per unit
    if len(unit_keys) == n_rows:
        return [rng.randrange(n_rows) for _ in range(n_rows)]

    chosen = [rng.choice(unit_keys) for _ in range(len(unit_keys))]
    out: List[int] = []
    for uk in chosen:
        out.extend(clusters[uk])
    return out


def percentile_ci(
    values: Sequence[float],
    confidence_level: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    xs = sorted(float(v) for v in values)
    alpha = 1.0 - confidence_level
    lo_q = alpha / 2.0
    hi_q = 1.0 - lo_q
    n = len(xs)

    def _q(q: float) -> float:
        if n == 1:
            return xs[0]
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        w = pos - lo
        return xs[lo] * (1 - w) + xs[hi] * w

    return _q(lo_q), _q(hi_q)


def _binary_f_metrics(golds: List[bool], preds: List[bool]) -> Dict[str, float]:
    cm = binary_confusion(golds, preds)
    block = metrics_from_confusion(cm, ["precision", "recall", "f1", "accuracy"], round_digits=None)
    return {
        "precision": float(block["precision"]),
        "recall": float(block["recall"]),
        "f1": float(block["f1"]),
        "accuracy": float(block["accuracy"]),
    }


def _metric_on_indices(
    rows: List[PairRow],
    indices: Sequence[int],
    *,
    side: str,
    target_type: str,
    metric_names: Sequence[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    golds: List[Any] = []
    preds: List[Any] = []
    for i in indices:
        r = rows[i]
        golds.append(r.gold)
        preds.append(r.baseline_pred if side == "baseline" else r.candidate_pred)

    out: Dict[str, float] = {}
    if not golds:
        for m in metric_names:
            out[m] = 0.0
        return out

    if target_type == "binary":
        g_bool = []
        p_bool = []
        for g, p in zip(golds, preds):
            gb = _as_bool(g)
            pb = _as_bool(p)
            if gb is None:
                continue
            if pb is None:
                pb = not gb
            g_bool.append(gb)
            p_bool.append(pb)
        raw = _binary_f_metrics(g_bool, p_bool)
        for m in metric_names:
            if m in raw:
                out[m] = raw[m]
    elif target_type == "multiclass":
        g_str = ["" if g is None else str(g) for g in golds]
        p_str = ["" if p is None else str(p) for p in preds]
        block = multiclass_metrics(
            g_str, p_str, list(metric_names), labels=labels, round_digits=None
        )
        for m in metric_names:
            if m in block and isinstance(block[m], (int, float)):
                out[m] = float(block[m])
    else:
        # text / exact match
        n = len(golds)
        em = sum(1 for g, p in zip(golds, preds) if g == p) / n if n else 0.0
        if "exact_match" in metric_names or "accuracy" in metric_names:
            key = "exact_match" if "exact_match" in metric_names else "accuracy"
            out[key] = em
    return out


def build_pair_rows(
    samples: List[SampleRecord],
    scores_b: List[ScoreRecord],
    scores_c: List[ScoreRecord],
    records: List[ComparisonRecord],
    *,
    target: str,
    bootstrap_unit: str,
    denominator: str,
) -> List[PairRow]:
    """One row per aligned sample that is metric-eligible under denominator."""
    b_map = {s.sample_id: s for s in scores_b}
    c_map = {s.sample_id: s for s in scores_c}
    sample_map = {s.sample_id: s for s in samples}
    rec_map = {r.sample_id: r for r in records}
    rows: List[PairRow] = []
    for sid, sample in sample_map.items():
        b = b_map.get(sid)
        c = c_map.get(sid)
        if b is None or c is None:
            continue
        b_ts = b.targets.get(target)
        c_ts = c.targets.get(target)
        if b_ts is None or c_ts is None or not (b_ts.applicable and c_ts.applicable):
            continue
        b_ok = b.parse_ok and b.schema_ok
        c_ok = c.parse_ok and c.schema_ok
        if denominator == "semantic" and not (b_ok and c_ok):
            continue
        rec = rec_map.get(sid)
        rows.append(
            PairRow(
                sample_id=sid,
                unit_id=resolve_unit_id(sample, bootstrap_unit),
                gold=b_ts.gold,
                baseline_pred=b_ts.pred,
                candidate_pred=c_ts.pred,
                transition=rec.transition if rec else None,
            )
        )
    return rows


def run_paired_bootstrap(
    rows: List[PairRow],
    *,
    target_type: str,
    metric_names: Sequence[str],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    labels: Optional[List[str]] = None,
    round_digits: Optional[int] = None,
) -> Dict[str, Any]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    rng = random.Random(seed)
    unit_ids = [r.unit_id for r in rows]
    clusters = build_cluster_map(unit_ids)
    n_units = len(clusters)
    cluster_mode = n_units < len(rows)

    # point estimates on full set
    idx_all = list(range(len(rows)))
    point_b = _metric_on_indices(
        rows, idx_all, side="baseline", target_type=target_type, metric_names=metric_names, labels=labels
    )
    point_c = _metric_on_indices(
        rows, idx_all, side="candidate", target_type=target_type, metric_names=metric_names, labels=labels
    )

    boot_b: Dict[str, List[float]] = {m: [] for m in metric_names}
    boot_c: Dict[str, List[float]] = {m: [] for m in metric_names}
    boot_d: Dict[str, List[float]] = {m: [] for m in metric_names}
    boot_gain: List[float] = []
    boot_reg: List[float] = []
    boot_net: List[float] = []

    for _ in range(n_bootstrap):
        idxs = resample_indices(len(rows), unit_ids=unit_ids, rng=rng)
        mb = _metric_on_indices(
            rows, idxs, side="baseline", target_type=target_type, metric_names=metric_names, labels=labels
        )
        mc = _metric_on_indices(
            rows, idxs, side="candidate", target_type=target_type, metric_names=metric_names, labels=labels
        )
        for m in metric_names:
            if m not in mb or m not in mc:
                continue
            boot_b[m].append(mb[m])
            boot_c[m].append(mc[m])
            boot_d[m].append(mc[m] - mb[m])
        g = sum(1 for i in idxs if rows[i].transition == "gain")
        r = sum(1 for i in idxs if rows[i].transition == "regression")
        boot_gain.append(float(g))
        boot_reg.append(float(r))
        boot_net.append(float(g - r))

    def _pack(point: float, samples: List[float]) -> Dict[str, Any]:
        lo, hi = percentile_ci(samples, confidence_level)
        return {
            "point": _round_maybe(point, round_digits) if round_digits is not None else point,
            "ci_low": _round_maybe(lo, round_digits) if lo is not None and round_digits is not None else lo,
            "ci_high": _round_maybe(hi, round_digits) if hi is not None and round_digits is not None else hi,
            "n_bootstrap": n_bootstrap,
        }

    metrics_out: Dict[str, Any] = {}
    for m in metric_names:
        if m not in point_b or m not in point_c:
            continue
        metrics_out[m] = {
            "baseline": _pack(point_b[m], boot_b[m]),
            "candidate": _pack(point_c[m], boot_c[m]),
            "delta": _pack(point_c[m] - point_b[m], boot_d[m]),
        }

    return {
        "n_rows": len(rows),
        "n_units": n_units,
        "cluster_mode": cluster_mode,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "seed": seed,
        "metrics": metrics_out,
        "transitions": {
            "gain": _pack(sum(1 for r in rows if r.transition == "gain"), boot_gain),
            "regression": _pack(sum(1 for r in rows if r.transition == "regression"), boot_reg),
            "net_gain": _pack(
                sum(1 for r in rows if r.transition == "gain")
                - sum(1 for r in rows if r.transition == "regression"),
                boot_net,
            ),
        },
    }
