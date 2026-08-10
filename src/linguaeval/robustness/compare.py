"""Paired baseline↔candidate robustness compare (P2-D).

Requires a shared VariantRecord set / variant_fingerprint.
Roles are baseline/candidate (display may say Base/SFT).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from linguaeval.core.schema import RobustnessRecord, VariantRecord
from linguaeval.robustness.generate import variant_fingerprint

# Model-level robustness transitions on relation_satisfied (True = OK under metamorphic).
MODEL_ROBUST_TRANSITIONS = (
    "stable_robust",
    "robustness_gain",
    "robustness_regression",
    "both_fragile",
)


class VariantFingerprintError(ValueError):
    """Raised when baseline/candidate are not on the same variant set."""

    def __init__(self, message: str, *, reason: str = "fingerprint_mismatch"):
        super().__init__(message)
        self.reason = reason


def compute_variant_fingerprint(variants: Sequence[VariantRecord]) -> str:
    return variant_fingerprint(variants)


def require_shared_variant_fingerprint(
    *,
    variants: Sequence[VariantRecord],
    expected_fingerprint: Optional[str] = None,
    baseline_fingerprint: Optional[str] = None,
    candidate_fingerprint: Optional[str] = None,
) -> str:
    """Gate: both sides must share the same variant_fingerprint.

    Priority:
    1. fingerprint(variants) is source of truth for this run
    2. if expected_fingerprint set → must match
    3. if baseline/candidate fingerprints provided → must match each other and variants
    """
    fp = compute_variant_fingerprint(variants)
    if expected_fingerprint and str(expected_fingerprint) != fp:
        raise VariantFingerprintError(
            f"variants fingerprint {fp[:12]}… != expected {str(expected_fingerprint)[:12]}…",
            reason="expected_mismatch",
        )
    if baseline_fingerprint and candidate_fingerprint:
        if str(baseline_fingerprint) != str(candidate_fingerprint):
            raise VariantFingerprintError(
                "baseline and candidate variant_fingerprint differ "
                f"({str(baseline_fingerprint)[:12]}… vs {str(candidate_fingerprint)[:12]}…)",
                reason="side_mismatch",
            )
        if str(baseline_fingerprint) != fp:
            raise VariantFingerprintError(
                "side fingerprint does not match shared variants "
                f"({str(baseline_fingerprint)[:12]}… vs {fp[:12]}…)",
                reason="variants_mismatch",
            )
    elif baseline_fingerprint and str(baseline_fingerprint) != fp:
        raise VariantFingerprintError(
            f"baseline fingerprint != shared variants ({str(baseline_fingerprint)[:12]}…)",
            reason="variants_mismatch",
        )
    elif candidate_fingerprint and str(candidate_fingerprint) != fp:
        raise VariantFingerprintError(
            f"candidate fingerprint != shared variants ({str(candidate_fingerprint)[:12]}…)",
            reason="variants_mismatch",
        )
    return fp


def _pair_key(r: RobustnessRecord) -> Tuple[str, str, str]:
    return (r.parent_sample_id, r.variant_id, r.target)


def classify_model_robust_transition(
    baseline_ok: Optional[bool],
    candidate_ok: Optional[bool],
) -> Optional[str]:
    """OK = relation_satisfied is True (invariant held / directional ok)."""
    if baseline_ok is None or candidate_ok is None:
        return None
    if baseline_ok and candidate_ok:
        return "stable_robust"
    if (not baseline_ok) and candidate_ok:
        return "robustness_gain"
    if baseline_ok and (not candidate_ok):
        return "robustness_regression"
    return "both_fragile"


def pair_robustness_records(
    baseline: Sequence[RobustnessRecord],
    candidate: Sequence[RobustnessRecord],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    """Strict-align applicable pairs by (parent, variant_id, target)."""
    b_map = {_pair_key(r): r for r in baseline if r.applicable}
    c_map = {_pair_key(r): r for r in candidate if r.applicable}
    b_keys = set(b_map)
    c_keys = set(c_map)
    audit = {
        "baseline_applicable": len(b_map),
        "candidate_applicable": len(c_map),
        "aligned": len(b_keys & c_keys),
        "baseline_only": sorted(b_keys - c_keys)[:50],
        "candidate_only": sorted(c_keys - b_keys)[:50],
        "baseline_only_count": len(b_keys - c_keys),
        "candidate_only_count": len(c_keys - b_keys),
    }
    if audit["baseline_only_count"] or audit["candidate_only_count"]:
        raise VariantFingerprintError(
            "applicable robustness record keys differ between baseline and candidate "
            f"(baseline_only={audit['baseline_only_count']}, "
            f"candidate_only={audit['candidate_only_count']})",
            reason="record_alignment",
        )

    counts = {k: 0 for k in MODEL_ROBUST_TRANSITIONS}
    rows: List[Dict[str, Any]] = []
    for key in sorted(b_keys):
        br = b_map[key]
        cr = c_map[key]
        trans = classify_model_robust_transition(br.relation_satisfied, cr.relation_satisfied)
        if trans:
            counts[trans] += 1
        rows.append(
            {
                "parent_sample_id": br.parent_sample_id,
                "variant_id": br.variant_id,
                "target": br.target,
                "perturbation_id": br.perturbation_id,
                "baseline": {
                    "flipped": br.flipped,
                    "relation_satisfied": br.relation_satisfied,
                    "clean_correct": br.clean_correct,
                    "variant_correct": br.variant_correct,
                },
                "candidate": {
                    "flipped": cr.flipped,
                    "relation_satisfied": cr.relation_satisfied,
                    "clean_correct": cr.clean_correct,
                    "variant_correct": cr.variant_correct,
                },
                "transition": trans,
            }
        )
    return rows, counts, audit


_RATE_KEYS = (
    "flip_rate",
    "metamorphic_violation_rate",
    "variant_all_correct_rate",
    "end_to_end_robust_success_rate",
    "accuracy_clean",
    "accuracy_perturbed",
    "delta_accuracy",
)


def _delta_block(b: Dict[str, Any], c: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in _RATE_KEYS:
        bv, cv = b.get(k), c.get(k)
        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
            out[k] = float(cv) - float(bv)
        else:
            out[k] = None
    return out


def compare_robustness_metrics(
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Point-estimate deltas: candidate − baseline (negative flip_rate = improvement)."""
    by_target: Dict[str, Any] = {}
    bt = dict(baseline_metrics.get("by_target") or {})
    ct = dict(candidate_metrics.get("by_target") or {})
    for tname in sorted(set(bt) | set(ct)):
        bblock = dict(bt.get(tname) or {})
        cblock = dict(ct.get(tname) or {})
        by_target[tname] = {
            "baseline": {k: bblock.get(k) for k in _RATE_KEYS},
            "candidate": {k: cblock.get(k) for k in _RATE_KEYS},
            "delta": _delta_block(bblock, cblock),
        }
    return {
        "baseline_status": baseline_metrics.get("status"),
        "candidate_status": candidate_metrics.get("status"),
        "by_target": by_target,
    }


def summarize_compare(
    *,
    fingerprint: str,
    metric_compare: Dict[str, Any],
    transition_counts: Dict[str, int],
    alignment_audit: Dict[str, Any],
) -> Dict[str, Any]:
    eligible = sum(transition_counts.values())
    return {
        "status": "AVAILABLE",
        "variant_fingerprint": fingerprint,
        "alignment": alignment_audit,
        "transitions": transition_counts,
        "n_transition_eligible": eligible,
        "metrics_compare": metric_compare,
    }
