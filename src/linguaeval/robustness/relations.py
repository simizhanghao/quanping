"""Metamorphic relation evaluation (P2-A: invariance only)."""

from __future__ import annotations

from typing import Any, Optional

from linguaeval.core.schema import MetamorphicRelationSpec

VALID_FOR_METRICS = frozenset({"VERIFIED", "AUTO_VALIDATED"})


def values_equal(a: Any, b: Any) -> bool:
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
        # Reserved for P2-C+; not implemented in P2-A.
        return None
    raise ValueError(f"unsupported metamorphic relation type: {rtype}")


def is_valid_for_metrics(semantic_validity: str) -> bool:
    return str(semantic_validity).upper() in VALID_FOR_METRICS
