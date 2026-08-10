"""Transition classification for paired regression."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from linguaeval.core.schema import ComparisonRecord, ScoreRecord, SideScore

TRANSITIONS = ("stable_correct", "gain", "regression", "both_wrong")


def classify_transition(baseline_correct: bool, candidate_correct: bool) -> str:
    if baseline_correct and candidate_correct:
        return "stable_correct"
    if (not baseline_correct) and candidate_correct:
        return "gain"
    if baseline_correct and (not candidate_correct):
        return "regression"
    return "both_wrong"


def _format_ok(score: ScoreRecord) -> bool:
    return bool(score.parse_ok and score.schema_ok)


def build_comparison_records(
    baseline_scores: List[ScoreRecord],
    candidate_scores: List[ScoreRecord],
    *,
    target: str,
    denominator: str = "semantic",
    ordered_ids: Optional[List[str]] = None,
) -> Tuple[List[ComparisonRecord], Dict[str, int]]:
    """Build ComparisonRecords for one target.

    ``denominator``:
      - semantic: both format_ok and applicable → 4-cell
      - strict: applicable → 4-cell (format fail already incorrect)
    """
    if denominator not in {"semantic", "strict"}:
        raise ValueError(f"unsupported denominator={denominator!r}")

    b_map = {s.sample_id: s for s in baseline_scores}
    c_map = {s.sample_id: s for s in candidate_scores}
    ids = ordered_ids or sorted(b_map.keys())

    records: List[ComparisonRecord] = []
    counts = {
        "total_aligned_samples": len(ids),
        "applicable_samples": 0,
        "not_applicable_samples": 0,
        "excluded_format_samples": 0,
        "transition_eligible": 0,
        "stable_correct": 0,
        "gain": 0,
        "regression": 0,
        "both_wrong": 0,
    }

    for sid in ids:
        b = b_map[sid]
        c = c_map[sid]
        b_ts = b.targets.get(target)
        c_ts = c.targets.get(target)
        if b_ts is None or c_ts is None:
            raise KeyError(f"target={target!r} missing on sample_id={sid}")

        # applicable if either side says so; they should match (same gold/condition)
        applicable = bool(b_ts.applicable and c_ts.applicable)
        if not applicable:
            counts["not_applicable_samples"] += 1
            records.append(
                ComparisonRecord(
                    sample_id=sid,
                    target=target,
                    applicable=False,
                    baseline=SideScore(pred=b_ts.pred, correct=b_ts.correct),
                    candidate=SideScore(pred=c_ts.pred, correct=c_ts.correct),
                    transition=None,
                    exclusion="not_applicable",
                )
            )
            continue

        counts["applicable_samples"] += 1
        b_ok = _format_ok(b)
        c_ok = _format_ok(c)

        if denominator == "semantic" and not (b_ok and c_ok):
            counts["excluded_format_samples"] += 1
            records.append(
                ComparisonRecord(
                    sample_id=sid,
                    target=target,
                    applicable=True,
                    baseline=SideScore(pred=b_ts.pred, correct=b_ts.correct),
                    candidate=SideScore(pred=c_ts.pred, correct=c_ts.correct),
                    transition=None,
                    exclusion="excluded_format",
                )
            )
            continue

        b_correct = bool(b_ts.correct)
        c_correct = bool(c_ts.correct)
        transition = classify_transition(b_correct, c_correct)
        counts["transition_eligible"] += 1
        counts[transition] += 1
        records.append(
            ComparisonRecord(
                sample_id=sid,
                target=target,
                applicable=True,
                baseline=SideScore(pred=b_ts.pred, correct=b_correct),
                candidate=SideScore(pred=c_ts.pred, correct=c_correct),
                transition=transition,
                exclusion=None,
            )
        )

    four = (
        counts["stable_correct"]
        + counts["gain"]
        + counts["regression"]
        + counts["both_wrong"]
    )
    if four != counts["transition_eligible"]:
        raise AssertionError("transition 4-cell invariant broken")
    return records, counts
