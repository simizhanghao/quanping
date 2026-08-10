"""Strict sample_id alignment for paired comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, TypeVar

T = TypeVar("T")


class AlignmentError(ValueError):
    """Raised when baseline/candidate sample_id sets are not identical."""


@dataclass
class AlignmentAudit:
    baseline_count: int
    candidate_count: int
    aligned_count: int
    baseline_only: List[str]
    candidate_only: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "baseline_count": self.baseline_count,
            "candidate_count": self.candidate_count,
            "aligned_count": self.aligned_count,
            "baseline_only_count": len(self.baseline_only),
            "candidate_only_count": len(self.candidate_only),
            "baseline_only": self.baseline_only[:50],
            "candidate_only": self.candidate_only[:50],
            "baseline_only_truncated": len(self.baseline_only) > 50,
            "candidate_only_truncated": len(self.candidate_only) > 50,
        }


def _ids(items: Iterable[object], attr: str = "sample_id") -> List[str]:
    return [str(getattr(x, attr)) for x in items]


def audit_id_sets(baseline_ids: Sequence[str], candidate_ids: Sequence[str]) -> AlignmentAudit:
    b = set(baseline_ids)
    c = set(candidate_ids)
    return AlignmentAudit(
        baseline_count=len(baseline_ids),
        candidate_count=len(candidate_ids),
        aligned_count=len(b & c),
        baseline_only=sorted(b - c),
        candidate_only=sorted(c - b),
    )


def require_strict_alignment(
    baseline_items: Sequence[T],
    candidate_items: Sequence[T],
    *,
    attr: str = "sample_id",
) -> AlignmentAudit:
    """Require identical sample_id sets; raise AlignmentError otherwise."""
    b_ids = _ids(baseline_items, attr)
    c_ids = _ids(candidate_items, attr)
    if len(b_ids) != len(set(b_ids)):
        raise AlignmentError("baseline has duplicate sample_id values")
    if len(c_ids) != len(set(c_ids)):
        raise AlignmentError("candidate has duplicate sample_id values")
    audit = audit_id_sets(b_ids, c_ids)
    if audit.baseline_only or audit.candidate_only:
        parts = []
        if audit.baseline_only:
            parts.append(
                f"baseline_only={len(audit.baseline_only)} "
                f"(e.g. {audit.baseline_only[0]})"
            )
        if audit.candidate_only:
            parts.append(
                f"candidate_only={len(audit.candidate_only)} "
                f"(e.g. missing on baseline? first={audit.candidate_only[0]})"
            )
            # clearer message when candidate missing an id that baseline has
        if audit.baseline_only and not audit.candidate_only:
            raise AlignmentError(
                "sample sets mismatch: candidate missing sample_id="
                f"{audit.baseline_only[0]} ({'; '.join(parts)})"
            )
        if audit.candidate_only and not audit.baseline_only:
            raise AlignmentError(
                "sample sets mismatch: baseline missing sample_id="
                f"{audit.candidate_only[0]} ({'; '.join(parts)})"
            )
        raise AlignmentError(f"sample sets mismatch: {'; '.join(parts)}")
    return audit


def index_by_id(items: Sequence[T], *, attr: str = "sample_id") -> Dict[str, T]:
    out: Dict[str, T] = {}
    for item in items:
        sid = str(getattr(item, attr))
        if sid in out:
            raise AlignmentError(f"duplicate sample_id={sid}")
        out[sid] = item
    return out
