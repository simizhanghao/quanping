"""Coverage + semantic/strict metric modes.

Definitions (frozen for P0.5-C):

- eligible: all SampleRecords loaded for the run
- with_prediction: samples that have a PredictionRecord
- format_ok: parse_ok AND schema_ok
- coverage_prediction = with_prediction / eligible
- coverage_valid = format_ok / eligible

- semantic: business metrics only on format_ok samples (denominator = valid subset)
- strict: all eligible-with-prediction samples in denominator; format fail => incorrect
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from linguaeval.core.schema import ScoreRecord, TaskSpec, TargetSpec
from linguaeval.metrics.classification import (
    _as_bool,
    _round_maybe,
    binary_confusion,
    metrics_from_confusion,
    multiclass_metrics,
)


def build_coverage(
    *,
    eligible: int,
    with_prediction: int,
    format_ok: int,
    round_digits: Optional[int] = 4,
) -> Dict[str, Any]:
    def _r(x: float) -> float:
        return _round_maybe(x, round_digits)

    return {
        "eligible_samples": eligible,
        "with_prediction": with_prediction,
        "format_ok_samples": format_ok,
        "format_fail_samples": max(with_prediction - format_ok, 0),
        "coverage_prediction": _r(with_prediction / eligible) if eligible else None,
        "coverage_valid": _r(format_ok / eligible) if eligible else None,
        "definitions": {
            "eligible": "all loaded SampleRecords",
            "with_prediction": "samples aligned to a PredictionRecord",
            "format_ok": "parse_ok AND schema_ok",
            "coverage_prediction": "with_prediction / eligible",
            "coverage_valid": "format_ok / eligible",
            "semantic": "metrics on format_ok subset only; report support/denominator",
            "strict": "all with_prediction in denominator; format fail counts as incorrect",
        },
    }


def _filter_scores(scores: List[ScoreRecord], mode: str) -> List[ScoreRecord]:
    if mode == "semantic":
        return [s for s in scores if s.parse_ok and s.schema_ok]
    if mode == "strict":
        return list(scores)
    raise ValueError(f"unknown mode {mode}")


def _target_metrics_from_scores(
    scores: List[ScoreRecord],
    target: TargetSpec,
    wanted: List[str],
    round_digits: Optional[int],
) -> Dict[str, Any]:
    golds: List[Any] = []
    preds: List[Any] = []
    corrects: List[bool] = []
    for s in scores:
        ts = s.targets.get(target.name)
        if ts is None or not ts.applicable:
            continue
        golds.append(ts.gold)
        preds.append(ts.pred)
        corrects.append(bool(ts.correct))

    block: Dict[str, Any] = {
        "type": target.type,
        "path": target.path,
        "support": len(corrects),
        "denominator": len(corrects),
    }
    if not corrects:
        for w in wanted:
            block[w.lower()] = None
        return block

    if target.type == "binary":
        g_bool: List[bool] = []
        p_bool: List[bool] = []
        for s in scores:
            ts = s.targets.get(target.name)
            if ts is None or not ts.applicable:
                continue
            g = _as_bool(ts.gold)
            if g is None:
                continue
            if not (s.parse_ok and s.schema_ok):
                # strict path includes these; treat as wrong prediction
                p = not g
            else:
                p = _as_bool(ts.pred)
                if p is None:
                    p = not g
            g_bool.append(g)
            p_bool.append(p)
        cm = binary_confusion(g_bool, p_bool)
        block.update(metrics_from_confusion(cm, wanted, round_digits))
        block["support"] = len(g_bool)
        block["denominator"] = len(g_bool)
    elif target.type == "multiclass":
        g_str: List[str] = []
        p_str: List[str] = []
        for s in scores:
            ts = s.targets.get(target.name)
            if ts is None or not ts.applicable:
                continue
            g = "" if ts.gold is None else str(ts.gold)
            if not (s.parse_ok and s.schema_ok):
                p = "__FORMAT_FAIL__"
            else:
                p = "" if ts.pred is None else str(ts.pred)
            g_str.append(g)
            p_str.append(p)
        block.update(
            multiclass_metrics(
                g_str, p_str, wanted, labels=target.labels, round_digits=round_digits
            )
        )
        block["support"] = len(g_str)
        block["denominator"] = len(g_str)
    else:
        # text / extraction fields: exact_match from ScoreRecord.correct
        n = len(corrects)
        exact = sum(1 for c in corrects if c) / n if n else 0.0
        block["exact_match"] = _round_maybe(exact, round_digits)
        block["support"] = n
        block["denominator"] = n
    return block


def score_modes_from_score_records(
    scores: List[ScoreRecord],
    task: TaskSpec,
    metrics_wanted: Dict[str, List[str]],
    *,
    round_digits: Optional[int] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"semantic": {"targets": {}}, "strict": {"targets": {}}}
    for mode in ("semantic", "strict"):
        subset = _filter_scores(scores, mode)
        for target in task.targets:
            wanted = metrics_wanted.get(target.name, [])
            if not wanted:
                continue
            out[mode]["targets"][target.name] = _target_metrics_from_scores(
                subset, target, wanted, round_digits
            )
        # joint from score records
        if "joint" in metrics_wanted:
            joint_bits = []
            for s in subset:
                if s.joint_success is None:
                    continue
                # semantic subset already format-ok; strict includes fails (joint_success False)
                joint_bits.append(bool(s.joint_success))
            n = len(joint_bits)
            rate = sum(1 for x in joint_bits if x) / n if n else 0.0
            out[mode]["joint"] = {
                "exact_joint_success": _round_maybe(rate, round_digits),
                "support": n,
                "denominator": n,
            }
    return out


def light_data_audit(samples: List[Dict[str, Any]], scores: List[ScoreRecord]) -> Dict[str, Any]:
    langs = Counter()
    domains = Counter()
    for s in samples:
        meta = s.get("meta") or {}
        langs[str(meta.get("language") or "unknown")] += 1
        domains[str(meta.get("domain") or "unknown")] += 1
    return {
        "sample_count": len(samples),
        "score_record_count": len(scores),
        "language_distribution": dict(langs),
        "domain_distribution": dict(domains),
    }
