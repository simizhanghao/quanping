"""Shared paired Base↔Candidate comparison kernel (P1).

Used by compare-offline and language-matrix-offline — do not fork a second
regression implementation for language packs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from linguaeval.compare.aggregate import metric_deltas, summarize_transitions
from linguaeval.compare.alignment import AlignmentError, index_by_id, require_strict_alignment
from linguaeval.compare.bootstrap import build_pair_rows, run_paired_bootstrap
from linguaeval.compare.transitions import build_comparison_records
from linguaeval.core.schema import MetricSpec, PredictionRecord, SampleRecord, TaskSpec
from linguaeval.metrics.aggregate import build_business_metrics
from linguaeval.metrics.classification import score_targets
from linguaeval.metrics.score_records import build_score_records

_BOOTSTRAP_METRICS = {
    "precision",
    "recall",
    "f1",
    "accuracy",
    "macro_f1",
    "exact_match",
}


def _resolve_metric_names(
    *,
    target: str,
    metric_spec: MetricSpec,
    report_cfg: Dict[str, Any],
    stats_cfg: Dict[str, Any],
) -> List[str]:
    names = list(stats_cfg.get("metrics") or metric_spec.metrics.get(str(target)) or [])
    names = [m for m in names if str(m).lower() in _BOOTSTRAP_METRICS]
    if not names and report_cfg.get("primary_metric"):
        names = [str(report_cfg["primary_metric"])]
    if not names:
        names = ["accuracy"]
    return names


def compute_paired_comparison(
    samples: Sequence[SampleRecord],
    preds_b: Sequence[PredictionRecord],
    preds_c: Sequence[PredictionRecord],
    *,
    task: TaskSpec,
    metric_spec: MetricSpec,
    target: str,
    denominator: str = "semantic",
    report_cfg: Optional[Dict[str, Any]] = None,
    stats_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Strict sample_id alignment + Gain/Regression + optional paired bootstrap.

    Raises AlignmentError when prediction id sets differ.
    """
    report_cfg = dict(report_cfg or {})
    stats_cfg = dict(stats_cfg or {})
    if denominator not in {"semantic", "strict"}:
        raise ValueError("denominator must be semantic|strict")
    if target not in {t.name for t in task.targets}:
        raise KeyError(f"target={target!r} not in TaskSpec targets")

    align_audit = require_strict_alignment(list(preds_b), list(preds_c))

    sample_ids = {s.sample_id for s in samples}
    pred_ids = {p.sample_id for p in preds_b}
    missing_gold = sorted(pred_ids - sample_ids)
    if missing_gold:
        raise AlignmentError(
            f"samples missing gold for sample_id={missing_gold[0]} "
            f"(count={len(missing_gold)})"
        )

    sample_map = index_by_id(list(samples))
    ordered_ids = [p.sample_id for p in preds_b]
    samples_aligned = [sample_map[i] for i in ordered_ids]
    preds_b_map = index_by_id(list(preds_b))
    preds_c_map = index_by_id(list(preds_c))
    preds_b_ord = [preds_b_map[i] for i in ordered_ids]
    preds_c_ord = [preds_c_map[i] for i in ordered_ids]

    scores_b = build_score_records(samples_aligned, preds_b_ord, task)
    scores_c = build_score_records(samples_aligned, preds_c_ord, task)
    records, tcounts = build_comparison_records(
        scores_b,
        scores_c,
        target=str(target),
        denominator=denominator,
        ordered_ids=ordered_ids,
    )

    scored_b = score_targets(samples_aligned, preds_b_ord, task, metric_spec)
    scored_c = score_targets(samples_aligned, preds_c_ord, task, metric_spec)
    biz_b = build_business_metrics(scored_b, report_cfg=report_cfg)
    biz_c = build_business_metrics(scored_c, report_cfg=report_cfg)
    deltas = metric_deltas(biz_b, biz_c, target=str(target))

    target_spec = next(t for t in task.targets if t.name == target)
    bootstrap_unit = str(stats_cfg.get("bootstrap_unit") or "sample")
    metric_names = _resolve_metric_names(
        target=str(target),
        metric_spec=metric_spec,
        report_cfg=report_cfg,
        stats_cfg=stats_cfg,
    )
    pair_rows = build_pair_rows(
        samples_aligned,
        scores_b,
        scores_c,
        records,
        target=str(target),
        bootstrap_unit=bootstrap_unit,
        denominator=denominator,
    )

    statistics: Optional[Dict[str, Any]] = None
    if bool(stats_cfg.get("enabled", False)):
        statistics = {
            "enabled": True,
            "bootstrap_unit": bootstrap_unit,
            **run_paired_bootstrap(
                pair_rows,
                target_type=target_spec.type,
                metric_names=metric_names,
                n_bootstrap=int(stats_cfg.get("n_bootstrap") or 1000),
                confidence_level=float(
                    stats_cfg.get("confidence_level") or stats_cfg.get("ci") or 0.95
                ),
                seed=int(stats_cfg.get("seed") or 42),
                labels=target_spec.labels,
                round_digits=metric_spec.round_digits,
            ),
        }

    primary_metric = str(report_cfg.get("primary_metric") or metric_names[0])
    metric_path = f"targets.{target}.{primary_metric}"
    primary_delta = (deltas.get("metrics") or {}).get(primary_metric) or {}
    baseline_value = primary_delta.get("baseline")
    candidate_value = primary_delta.get("candidate")
    delta = primary_delta.get("delta")
    if baseline_value is None:
        baseline_value = ((biz_b.get("targets") or {}).get(target) or {}).get(primary_metric)
    if candidate_value is None:
        candidate_value = ((biz_c.get("targets") or {}).get(target) or {}).get(primary_metric)
    if (
        delta is None
        and isinstance(baseline_value, (int, float))
        and isinstance(candidate_value, (int, float))
    ):
        delta = float(candidate_value) - float(baseline_value)

    delta_ci_low = None
    delta_ci_high = None
    if statistics:
        packed = ((statistics.get("metrics") or {}).get(primary_metric) or {}).get("delta") or {}
        delta_ci_low = packed.get("ci_low")
        delta_ci_high = packed.get("ci_high")

    support = {
        "n_samples": len(pair_rows),
        "n_units": (statistics or {}).get("n_units") or len(pair_rows),
        "n_aligned": len(ordered_ids),
        "bootstrap_unit": bootstrap_unit,
        "cluster_mode": bool((statistics or {}).get("cluster_mode")),
    }

    return {
        "status": "AVAILABLE",
        "alignment": align_audit,
        "target": target,
        "denominator": denominator,
        "primary_target": target,
        "primary_metric": primary_metric,
        "metric_path": metric_path,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "delta": delta,
        "delta_ci_low": delta_ci_low,
        "delta_ci_high": delta_ci_high,
        "transitions": summarize_transitions(tcounts),
        "metric_deltas": deltas,
        "statistics": statistics,
        "baseline_business": biz_b,
        "candidate_business": biz_c,
        "baseline_business_view": {
            "primary": biz_b.get("primary"),
            "targets": {str(target): (biz_b.get("targets") or {}).get(target)},
            "schema": biz_b.get("schema"),
        },
        "candidate_business_view": {
            "primary": biz_c.get("primary"),
            "targets": {str(target): (biz_c.get("targets") or {}).get(target)},
            "schema": biz_c.get("schema"),
        },
        "support": support,
        "records": records,
        "pair_rows": pair_rows,
        "samples_aligned": samples_aligned,
        "preds_baseline": preds_b_ord,
        "preds_candidate": preds_c_ord,
        "scores_baseline": scores_b,
        "scores_candidate": scores_c,
    }
