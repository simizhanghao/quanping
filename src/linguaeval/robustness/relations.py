"""Metamorphic relation evaluation (P2: invariance; directional reserved)."""

from __future__ import annotations

from typing import Any, Optional

from linguaeval.core.schema import MetamorphicRelationSpec

VALID_FOR_METRICS = frozenset({"VERIFIED", "AUTO_VALIDATED"})


def values_equal(a: Any, b: Any) -> bool:
    """Only for prediction equality (flip); correctness uses ScoreRecord."""
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) is bool(b) if (isinstance(a, bool) and isinstance(b, bool)) else a == b
    return str(a) == str(b)


def relation_satisfied(
    relation: MetamorphicRelationSpec,
    *,
    clean_pred: Any,
    variant_pred: Any,
) -> Optional[bool]:
    rtype = (relation.type or "invariance").lower()
    if rtype == "invariance":
        return values_equal(clean_pred, variant_pred)
    if rtype == "directional":
        return None
    raise ValueError(f"unsupported metamorphic relation type: {rtype}")


def is_valid_for_metrics(semantic_validity: str) -> bool:
    return str(semantic_validity).upper() in VALID_FOR_METRICS


def transition_label(clean_correct: Optional[bool], variant_correct: Optional[bool]) -> Optional[str]:
    if clean_correct is None or variant_correct is None:
        return None
    if clean_correct and variant_correct:
        return "stable_correct"
    if (not clean_correct) and variant_correct:
        return "perturbation_gain"
    if clean_correct and (not variant_correct):
        return "robustness_regression"
    return "stable_wrong"
