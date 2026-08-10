"""Build sample-level ScoreRecords from Sample + Prediction + TaskSpec."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from linguaeval.core.paths import condition_holds, get_by_path
from linguaeval.core.schema import (
    PredictionRecord,
    SampleRecord,
    ScoreRecord,
    TargetScore,
    TaskSpec,
)


def _values_equal(gold: Any, pred: Any) -> bool:
    if isinstance(gold, bool) and isinstance(pred, bool):
        return gold == pred
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)) and not isinstance(
        gold, bool
    ) and not isinstance(pred, bool):
        return float(gold) == float(pred)
    return gold == pred


def build_score_records(
    samples: List[SampleRecord],
    preds: List[PredictionRecord],
    task: TaskSpec,
) -> List[ScoreRecord]:
    by_id = {p.sample_id: p for p in preds}
    records: List[ScoreRecord] = []
    for s in samples:
        p = by_id.get(s.sample_id)
        if p is None:
            continue
        target_scores: Dict[str, TargetScore] = {}
        joint_bits: List[bool] = []
        for target in task.targets:
            applicable = condition_holds(s.gold, target.condition)
            gold = get_by_path(s.gold, target.path, default=None)
            pred = get_by_path(p.parsed, target.path, default=None)
            if not applicable:
                target_scores[target.name] = TargetScore(
                    gold=gold, pred=pred, correct=None, applicable=False
                )
                continue
            # format fail → incorrect for sample-level joint/business
            if not (p.format.parse_ok and p.format.schema_ok):
                correct = False
            else:
                correct = _values_equal(gold, pred)
            target_scores[target.name] = TargetScore(
                gold=gold, pred=pred, correct=correct, applicable=True
            )
            joint_bits.append(bool(correct))

        joint_success: Optional[bool]
        if not joint_bits:
            joint_success = None
        else:
            joint_success = all(joint_bits)

        slices = {
            "language": (s.meta or {}).get("language"),
            "domain": (s.meta or {}).get("domain"),
            "split": (s.meta or {}).get("split"),
        }
        if s.conversation:
            slices["dialogue_id"] = s.conversation.get("dialogue_id")
            slices["turn_id"] = s.conversation.get("turn_id")

        records.append(
            ScoreRecord(
                sample_id=s.sample_id,
                model_id=p.model_id,
                targets=target_scores,
                parse_ok=p.format.parse_ok,
                schema_ok=p.format.schema_ok,
                joint_success=joint_success,
                slices={k: v for k, v in slices.items() if v is not None},
            )
        )
    return records
