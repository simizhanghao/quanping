from __future__ import annotations

from typing import Any, Dict, Optional


def build_business_metrics(
    score_block: Dict[str, Any],
    report_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flatten metrics for report / gate consumers.

    Primary target/metric come **only** from config ``report:`` — never from
    hardcoded business field names.
    """
    report_cfg = report_cfg or {}
    out: Dict[str, Any] = {
        "task": score_block.get("task"),
        "task_type": score_block.get("task_type"),
        "schema": score_block.get("schema"),
        "targets": score_block.get("targets"),
    }
    if "joint" in score_block:
        out["joint"] = score_block["joint"]

    primary_target = report_cfg.get("primary_target")
    primary_metric = report_cfg.get("primary_metric")
    targets = score_block.get("targets") or {}

    if primary_target:
        if primary_target not in targets:
            raise KeyError(
                f"report.primary_target={primary_target!r} not in scored targets "
                f"{sorted(targets)}"
            )
        block = dict(targets[primary_target])
        primary: Dict[str, Any] = {
            "target": primary_target,
            "metric": primary_metric,
            "value": block.get(primary_metric) if primary_metric else None,
        }
        # Expose the target block fields for gates/report without selecting by name.
        for k, v in block.items():
            primary.setdefault(k, v)
        out["primary"] = primary
    return out
