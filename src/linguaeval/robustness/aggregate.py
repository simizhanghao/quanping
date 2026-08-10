"""Build RobustnessRecords and aggregate Flip / Violation / Drop (P2-A/B)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from linguaeval.compare.bootstrap import percentile_ci, resample_indices
from linguaeval.core.paths import get_by_path
from linguaeval.core.schema import (
    MetamorphicRelationSpec,
    PredictionRecord,
    RobustnessRecord,
    SampleRecord,
    TaskSpec,
    VariantRecord,
)
from linguaeval.robustness.relations import is_valid_for_metrics, relation_satisfied, values_equal


def _pred_value(pred: Optional[PredictionRecord], path: str) -> Any:
    if pred is None:
        return None
    return get_by_path(pred.parsed or {}, path, default=None)


def _gold_value(sample: Optional[SampleRecord], path: str) -> Any:
    if sample is None:
        return None
    return get_by_path(sample.gold or {}, path, default=None)


def build_robustness_records(
    *,
    samples: Sequence[SampleRecord],
    clean_preds: Sequence[PredictionRecord],
    variants: Sequence[VariantRecord],
    variant_preds: Sequence[PredictionRecord],
    task: TaskSpec,
    relation: MetamorphicRelationSpec,
) -> List[RobustnessRecord]:
    by_sample = {s.sample_id: s for s in samples}
    clean_by = {p.sample_id: p for p in clean_preds}
    var_pred_by = {p.sample_id: p for p in variant_preds}

    target_names = relation.targets or [t.name for t in task.targets]
    target_specs = {t.name: t for t in task.targets}

    out: List[RobustnessRecord] = []
    for v in variants:
        parent = by_sample.get(v.parent_sample_id)
        cpred = clean_by.get(v.parent_sample_id)
        vpred = var_pred_by.get(v.variant_id)
        for tname in target_names:
            tspec = target_specs.get(tname)
            if tspec is None:
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
                    )
                )
                continue
            if cpred is None or vpred is None:
                out.append(
                    RobustnessRecord(
                        parent_sample_id=v.parent_sample_id,
                        variant_id=v.variant_id,
                        target=tname,
                        perturbation_id=v.perturbation_id,
                        applicable=False,
                        exclusion="missing_prediction",
                        semantic_validity=v.semantic_validity,
                        relation_type=relation.type,
                    )
                )
                continue

            clean_val = _pred_value(cpred, tspec.path)
            var_val = _pred_value(vpred, tspec.path)
            gold = _gold_value(parent, tspec.path)
            clean_ok = values_equal(gold, clean_val) if gold is not None else None
            var_ok = values_equal(gold, var_val) if gold is not None else None
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
                    clean_correct=clean_ok,
                    variant_correct=var_ok,
                    flipped=flipped,
                    relation_type=relation.type,
                    relation_satisfied=sat,
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
    """Reuse P1 resample_indices / percentile_ci (no duplicated bootstrap core)."""
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


def aggregate_robustness(
    records: Sequence[RobustnessRecord],
    *,
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

    if n == 0:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "no_applicable_robustness_pairs",
            "coverage": coverage,
            "by_target": {},
        }

    boot_cfg = dict(bootstrap or {})
    do_boot = bool(boot_cfg.get("enabled") or boot_cfg.get("n_boot"))
    n_boot = int(boot_cfg.get("n_boot") or 200)
    boot_seed = int(boot_cfg.get("seed") or 42)

    by_target: Dict[str, Any] = {}
    for tname in sorted({r.target for r in applicable}):
        rows = [r for r in applicable if r.target == tname]
        nt = len(rows)
        clean_known = [r for r in rows if r.clean_correct is not None]
        var_known = [r for r in rows if r.variant_correct is not None]
        acc_c = (
            sum(1 for r in clean_known if r.clean_correct) / len(clean_known)
            if clean_known
            else None
        )
        acc_v = (
            sum(1 for r in var_known if r.variant_correct) / len(var_known) if var_known else None
        )
        flip = sum(1 for r in rows if r.flipped) / nt
        viol = sum(1 for r in rows if r.relation_satisfied is False) / nt
        parents: Dict[str, List[RobustnessRecord]] = {}
        for r in rows:
            parents.setdefault(r.parent_sample_id, []).append(r)
        robust_ok = 0
        for plist in parents.values():
            if all(r.variant_correct is not None for r in plist) and all(
                r.variant_correct for r in plist
            ):
                robust_ok += 1
        block: Dict[str, Any] = {
            "n": nt,
            "n_parents": len(parents),
            "accuracy_clean": acc_c,
            "accuracy_perturbed": acc_v,
            "delta_accuracy": (acc_v - acc_c) if (acc_c is not None and acc_v is not None) else None,
            "flip_rate": flip,
            "metamorphic_violation_rate": viol,
            "robust_success_rate": robust_ok / len(parents) if parents else None,
        }
        if do_boot:
            units = [r.parent_sample_id for r in rows]
            block["flip_rate_bootstrap"] = _bootstrap_rate(
                [r.flipped for r in rows],
                units,
                n_boot=n_boot,
                seed=boot_seed,
            )
            block["violation_rate_bootstrap"] = _bootstrap_rate(
                [r.relation_satisfied is False for r in rows],
                units,
                n_boot=n_boot,
                seed=boot_seed + 1,
            )
        by_target[tname] = block

    return {
        "status": "AVAILABLE",
        "coverage": coverage,
        "by_target": by_target,
    }
