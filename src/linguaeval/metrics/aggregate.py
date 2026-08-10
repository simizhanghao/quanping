from __future__ import annotations

from typing import Any, Dict


def build_business_metrics(score_block: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten primary business metrics for report / gate consumers."""
    out: Dict[str, Any] = {
        "task": score_block.get("task"),
        "task_type": score_block.get("task_type"),
        "schema": score_block.get("schema"),
        "targets": score_block.get("targets"),
    }
    if "joint" in score_block:
        out["joint"] = score_block["joint"]
    # convenience top-level for single binary target named decision/n2s/label
    targets = score_block.get("targets") or {}
    for key in ("n2s", "decision", "label"):
        if key in targets and "f1" in targets[key]:
            out["primary"] = {
                "target": key,
                "precision": targets[key].get("precision"),
                "recall": targets[key].get("recall"),
                "f1": targets[key].get("f1"),
                "f2": targets[key].get("f2"),
                "accuracy": targets[key].get("accuracy"),
                "TP": targets[key].get("TP"),
                "TN": targets[key].get("TN"),
                "FP": targets[key].get("FP"),
                "FN": targets[key].get("FN"),
            }
            break
    return out
