"""Aggregate paired comparison metrics and case dumps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from linguaeval.core.schema import ComparisonRecord


def summarize_transitions(counts: Dict[str, int]) -> Dict[str, Any]:
    gain = counts.get("gain", 0)
    reg = counts.get("regression", 0)
    return {
        "total_aligned_samples": counts.get("total_aligned_samples", 0),
        "applicable_samples": counts.get("applicable_samples", 0),
        "not_applicable_samples": counts.get("not_applicable_samples", 0),
        "excluded_format_samples": counts.get("excluded_format_samples", 0),
        "transition_eligible": counts.get("transition_eligible", 0),
        "stable_correct": counts.get("stable_correct", 0),
        "gain": gain,
        "regression": reg,
        "both_wrong": counts.get("both_wrong", 0),
        "net_gain": gain - reg,
    }


def metric_deltas(
    baseline_business: Dict[str, Any],
    candidate_business: Dict[str, Any],
    *,
    target: str,
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Point-estimate deltas for one target (and primary if present)."""
    b_targets = baseline_business.get("targets") or {}
    c_targets = candidate_business.get("targets") or {}
    b_block = dict(b_targets.get(target) or {})
    c_block = dict(c_targets.get(target) or {})
    keys = metric_keys or sorted(
        set(b_block) & set(c_block) & {
            "precision",
            "recall",
            "f1",
            "f2",
            "accuracy",
            "macro_f1",
            "exact_match",
        }
    )
    deltas: Dict[str, Any] = {}
    for k in keys:
        bv, cv = b_block.get(k), c_block.get(k)
        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
            deltas[k] = {
                "baseline": bv,
                "candidate": cv,
                "delta": cv - bv,
            }
    out: Dict[str, Any] = {"target": target, "metrics": deltas}
    # primary convenience if both runs exposed it for this target
    for side_name, biz in ("baseline", baseline_business), ("candidate", candidate_business):
        prim = biz.get("primary") or {}
        if prim.get("target") == target and prim.get("metric"):
            out.setdefault("primary", {})[side_name] = {
                "metric": prim.get("metric"),
                "value": prim.get("value"),
            }
    if "primary" in out and "baseline" in out["primary"] and "candidate" in out["primary"]:
        bm = out["primary"]["baseline"]
        cm = out["primary"]["candidate"]
        if bm.get("metric") == cm.get("metric") and isinstance(bm.get("value"), (int, float)) and isinstance(
            cm.get("value"), (int, float)
        ):
            out["primary"]["delta"] = cm["value"] - bm["value"]
            out["primary"]["metric"] = bm["metric"]
    return out


def write_comparison_jsonl(path: Path, records: List[ComparisonRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def write_transition_cases(
    out_dir: Path,
    records: List[ComparisonRecord],
    *,
    transitions: Optional[List[str]] = None,
) -> Dict[str, str]:
    wanted = transitions or ["gain", "regression", "both_wrong", "stable_correct"]
    paths: Dict[str, str] = {}
    for name in wanted:
        path = out_dir / f"{name}_cases.jsonl"
        subset = [r for r in records if r.transition == name]
        write_comparison_jsonl(path, subset)
        paths[name] = str(path)
    return paths
