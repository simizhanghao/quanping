"""Build RobustnessRecords from ScoreRecords; aggregate via MetricSpec (P2-C0)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from linguaeval.compare.bootstrap import percentile_ci, resample_indices, resolve_unit_id
from linguaeval.core.schema import (
    MetamorphicRelationSpec,
    MetricSpec,
    PredictionRecord,
    RobustnessRecord,
    SampleRecord,
    TaskSpec,
    VariantRecord,
)
from linguaeval.metrics.classification import score_targets
from linguaeval.metrics.score_records import build_score_records
from linguaeval.robustness.relations import (
    is_valid_for_metrics,
    relation_satisfied,
    transition_label,
    values_equal,
)


def _variant_as_samples(
    samples: Sequence[SampleRecord],
    variants: Sequence[VariantRecord],
) -> List[SampleRecord]:
    by_id = {s.sample_id: s for s in samples}
    out: List[SampleRecord] = []
    for v in variants:
        parent = by_id.get(v.parent_sample_id)
        if parent is None:
            continue
        out.append(
            SampleRecord(
                sample_id=v.variant_id,
                gold=dict(parent.gold),
                input=v.input,
                meta=dict(parent.meta or {}),
                conversation=parent.conversation,
            )
        )
    return out


def build_robustness_records(
    *,
    samples: Sequence[SampleRecord],
    clean_preds: Sequence[PredictionRecord],
    variants: Sequence[VariantRecord],
    variant_preds: Sequence[PredictionRecord],
    task: TaskSpec,
    relation: MetamorphicRelationSpec,
) -> List[RobustnessRecord]:
    """Correctness from ScoreRecord (D1), not a second equality world."""
    clean_scores = {r.sample_id: r for r in build_score_records(list(samples), list(clean_preds), task)}
    var_samples = _variant_as_samples(samples, variants)
    var_scores = {
        r.sample_id: r for r in build_score_records(var_samples, list(variant_preds), task)
    }

    target_names = relation.targets or [t.name for t in task.targets]
    out: List[RobustnessRecord] = []
    for v in variants:
        cscore = clean_scores.get(v.parent_sample_id)
        vscore = var_scores.get(v.variant_id)
        for tname in target_names:
            if tname not in {t.name for t in task.targets}:
                out.append(
                    RobustnessRecord(
                        parent_sample_id=v.parent_sample_id,
                        variant_id=v.variant_id,
                        target=tname,
                        perturbation_id=v.perturbation_id,
                        applicable=False,
                        exclusion="target_not_in_task_spec",
                        semantic_validity=v.semantic_validity,
                        relation_type=relation.type,
                        severity=v.severity,
                    )
                )
                continue
            if not is_valid_for_metrics(v.semantic_validity):
                out.append(
                    RobustnessRecord(
                        parent_sample_id=v.parent_sample_id,
                        variant_id=v.variant_id,
                        target=tname,
                        perturbation_id=v.perturbation_id,
                        applicable=False,
                        exclusion=f"semantic_validity:{v.semantic_validity}",
                        semantic_validity=v.semantic_validity,
                        relation_type=relation.type,
                        severity=v.severity,
                    )
                )
                continue
            if cscore is None or vscore is None:
                out.append(
                    RobustnessRecord(
                        parent_sample_id=v.parent_sample_id,
                        variant_id=v.variant_id,
                        target=tname,
                        perturbation_id=v.perturbation_id,
                        applicable=False,
                        exclusion="missing_score_record",
                        semantic_validity=v.semantic_validity,
                        relation_type=relation.type,
                        severity=v.severity,
                    )
                )
                continue
            cts = cscore.targets.get(tname)
            vts = vscore.targets.get(tname)
            if cts is None or vts is None or not cts.applicable or not vts.applicable:
                out.append(
                    RobustnessRecord(
                        parent_sample_id=v.parent_sample_id,
                        variant_id=v.variant_id,
                        target=tname,
                        perturbation_id=v.perturbation_id,
                        applicable=False,
                        exclusion="target_not_applicable",
                        semantic_validity=v.semantic_validity,
                        relation_type=relation.type,
                        severity=v.severity,
                    )
                )
                continue
            clean_val = cts.pred
            var_val = vts.pred
            flipped = not values_equal(clean_val, var_val)
            sat = relation_satisfied(relation, clean_pred=clean_val, variant_pred=var_val)
            out.append(
                RobustnessRecord(
                    parent_sample_id=v.parent_sample_id,
                    variant_id=v.variant_id,
                    target=tname,
                    perturbation_id=v.perturbation_id,
                    clean_pred=clean_val,
                    variant_pred=var_val,
                    clean_correct=cts.correct,
                    variant_correct=vts.correct,
                    flipped=flipped,
                    relation_type=relation.type,
                    relation_satisfied=sat,
                    transition=transition_label(cts.correct, vts.correct),
                    severity=v.severity,
                    semantic_validity=v.semantic_validity,
                    applicable=True,
                )
            )
    return out


def _bootstrap_rate(
    flags: Sequence[bool],
    unit_ids: Sequence[str],
    *,
    n_boot: int,
    seed: int,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    n = len(flags)
    if n == 0:
        return {"value": None, "ci_low": None, "ci_high": None, "n_boot": 0}
    point = sum(1 for x in flags if x) / n
    rng = random.Random(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = resample_indices(n, unit_ids=unit_ids, rng=rng)
        if not idx:
            continue
        vals.append(sum(1 for i in idx if flags[i]) / len(idx))
    lo, hi = percentile_ci(vals, confidence_level=confidence_level)
    return {
        "value": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": n_boot,
        "confidence_level": confidence_level,
    }


def _metric_deltas(clean: Dict[str, Any], pert: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ct = (clean or {}).get("targets") or {}
    pt = (pert or {}).get("targets") or {}
    for tname, cblock in ct.items():
        pblock = pt.get(tname) or {}
        deltas: Dict[str, Any] = {}
        for k, cv in cblock.items():
            if not isinstance(cv, (int, float)):
                continue
            pv = pblock.get(k)
            if isinstance(pv, (int, float)):
                deltas[k] = float(pv) - float(cv)
        out[tname] = deltas
    return out


def _slice_stats(rows: Sequence[RobustnessRecord]) -> Dict[str, Any]:
    nt = len(rows)
    if nt == 0:
        return {"n": 0}
    flip = sum(1 for r in rows if r.flipped) / nt
    viol = sum(1 for r in rows if r.relation_satisfied is False) / nt
    trans: Dict[str, int] = {}
    for r in rows:
        if r.transition:
            trans[r.transition] = trans.get(r.transition, 0) + 1
    return {
        "n": nt,
        "flip_rate": flip,
        "metamorphic_violation_rate": viol,
        "transitions": trans,
    }


def aggregate_robustness(
    records: Sequence[RobustnessRecord],
    *,
    samples: Optional[Sequence[SampleRecord]] = None,
    clean_preds: Optional[Sequence[PredictionRecord]] = None,
    variants: Optional[Sequence[VariantRecord]] = None,
    variant_preds: Optional[Sequence[PredictionRecord]] = None,
    task: Optional[TaskSpec] = None,
    metric_spec: Optional[MetricSpec] = None,
    bootstrap: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    generated = len(records)
    applicable = [r for r in records if r.applicable]
    n = len(applicable)
    coverage = {
        "n_records": generated,
        "n_applicable": n,
        "n_excluded": generated - n,
        "exclusion_counts": {},
    }
    for r in records:
        if r.exclusion:
            coverage["exclusion_counts"][r.exclusion] = (
                coverage["exclusion_counts"].get(r.exclusion, 0) + 1
            )

    metrics_block: Dict[str, Any] = {}
    if task is not None and metric_spec is not None and samples is not None and clean_preds is not None:
        clean_m = score_targets(list(samples), list(clean_preds), task, metric_spec)
        pert_m = None
        if variants is not None and variant_preds is not None:
            vsamples = _variant_as_samples(samples, [v for v in variants if is_valid_for_metrics(v.semantic_validity)])
            # only preds for valid variants
            valid_ids = {s.sample_id for s in vsamples}
            vpreds = [p for p in variant_preds if p.sample_id in valid_ids]
            pert_m = score_targets(vsamples, vpreds, task, metric_spec)
        metrics_block = {
            "clean": clean_m,
            "perturbed": pert_m,
            "delta": _metric_deltas(clean_m, pert_m or {}),
        }

    if n == 0:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "no_applicable_robustness_pairs",
            "coverage": coverage,
            "metrics": metrics_block,
            "by_target": {},
            "by_perturbation": {},
            "by_severity": {},
        }

    boot_cfg = dict(bootstrap or {})
    do_boot = bool(boot_cfg.get("enabled") or boot_cfg.get("n_boot"))
    n_boot = int(boot_cfg.get("n_boot") or 200)
    boot_seed = int(boot_cfg.get("seed") or 42)
    cluster_path = str(boot_cfg.get("cluster_path") or boot_cfg.get("bootstrap_unit") or "sample")
    sample_by = {s.sample_id: s for s in (samples or [])}

    def _unit_for(parent_id: str) -> str:
        s = sample_by.get(parent_id)
        if s is None:
            return parent_id
        return resolve_unit_id(s, cluster_path)

    by_target: Dict[str, Any] = {}
    for tname in sorted({r.target for r in applicable}):
        rows = [r for r in applicable if r.target == tname]
        nt = len(rows)
        parents: Dict[str, List[RobustnessRecord]] = {}
        for r in rows:
            parents.setdefault(r.parent_sample_id, []).append(r)

        variant_all = 0
        e2e = 0
        for plist in parents.values():
            if not plist:
                continue
            all_var_ok = all(r.variant_correct is True for r in plist)
            all_rel_ok = all(r.relation_satisfied is True for r in plist)
            clean_ok = all(r.clean_correct is True for r in plist)
            if all_var_ok:
                variant_all += 1
            if clean_ok and all_var_ok and all_rel_ok:
                e2e += 1

        block = {
            **_slice_stats(rows),
            "n_parents": len(parents),
            "variant_all_correct_rate": variant_all / len(parents) if parents else None,
            "end_to_end_robust_success_rate": e2e / len(parents) if parents else None,
            # backward-compatible alias (deprecated): old robust_success ≈ variant_all_correct
            "robust_success_rate": variant_all / len(parents) if parents else None,
        }
        # keep accuracy diagnostics from ScoreRecord for smoke
        clean_known = [r for r in rows if r.clean_correct is not None]
        var_known = [r for r in rows if r.variant_correct is not None]
        acc_c = (
            sum(1 for r in clean_known if r.clean_correct) / len(clean_known) if clean_known else None
        )
        acc_v = sum(1 for r in var_known if r.variant_correct) / len(var_known) if var_known else None
        block["accuracy_clean"] = acc_c
        block["accuracy_perturbed"] = acc_v
        block["delta_accuracy"] = (acc_v - acc_c) if (acc_c is not None and acc_v is not None) else None

        if do_boot:
            units = [_unit_for(r.parent_sample_id) for r in rows]
            block["flip_rate_bootstrap"] = _bootstrap_rate(
                [r.flipped for r in rows], units, n_boot=n_boot, seed=boot_seed
            )
            block["violation_rate_bootstrap"] = _bootstrap_rate(
                [r.relation_satisfied is False for r in rows],
                units,
                n_boot=n_boot,
                seed=boot_seed + 1,
            )
        by_target[tname] = block

    by_pert: Dict[str, Any] = {}
    for pid in sorted({r.perturbation_id for r in applicable}):
        by_pert[pid] = _slice_stats([r for r in applicable if r.perturbation_id == pid])

    by_sev: Dict[str, Any] = {}
    for sev in sorted({r.severity for r in applicable}):
        by_sev[str(sev)] = _slice_stats([r for r in applicable if r.severity == sev])

    return {
        "status": "AVAILABLE",
        "coverage": coverage,
        "metrics": metrics_block,
        "by_target": by_target,
        "by_perturbation": by_pert,
        "by_severity": by_sev,
        "bootstrap_cluster_path": cluster_path,
    }
