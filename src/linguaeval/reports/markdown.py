from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def write_report_md(path: Path, *, business: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    lines = [
        f"# LinguaEval Report — {business.get('task', 'unknown')}",
        "",
        f"- run_id: `{manifest.get('run_id')}`",
        f"- created_at: `{manifest.get('created_at')}`",
        "",
        "## Schema",
        "",
    ]
    schema = business.get("schema") or {}
    lines.append(f"- eval_sample_count: {schema.get('eval_sample_count')}")
    lines.append(f"- format_match_rate: {schema.get('format_match_rate')}")
    lines.append("")
    if business.get("primary"):
        p = business["primary"]
        lines += [
            "## Primary Business Metrics",
            "",
            f"- target: `{p.get('target')}`",
            f"- metric: `{p.get('metric')}`",
            f"- value: {p.get('value')}",
            "",
        ]
        for k in ("precision", "recall", "f1", "f2", "accuracy", "macro_f1", "exact_match"):
            if k in p and k != p.get("metric"):
                lines.append(f"- {k}: {p[k]}")
        if "TP" in p:
            lines.append(f"- TP/TN/FP/FN: {p.get('TP')}/{p.get('TN')}/{p.get('FP')}/{p.get('FN')}")
        lines.append("")
    lines += ["## Targets", ""]
    for name, block in (business.get("targets") or {}).items():
        lines.append(f"### `{name}` ({block.get('type')})")
        for k in ("precision", "recall", "f1", "f2", "accuracy", "macro_f1", "exact_match"):
            if k in block:
                lines.append(f"- {k}: {block[k]}")
        lines.append("")
    if business.get("joint"):
        lines += [
            "## Joint",
            "",
            f"- exact_joint_success: {business['joint'].get('exact_joint_success')}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
