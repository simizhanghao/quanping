from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def write_compare_report_md(
    path: Path,
    *,
    metrics: Dict[str, Any],
    manifest: Dict[str, Any],
) -> None:
    cmp = metrics.get("compare") or {}
    trans = metrics.get("transitions") or {}
    deltas = (metrics.get("metric_deltas") or {}).get("metrics") or {}
    display = cmp.get("display") or {}
    b_name = display.get("baseline", "baseline")
    c_name = display.get("candidate", "candidate")

    lines = [
        f"# LinguaEval Paired Compare — {cmp.get('target', 'unknown')}",
        "",
        f"- run_id: `{manifest.get('run_id')}`",
        f"- {b_name} → {c_name}",
        f"- denominator: `{cmp.get('denominator')}`",
        f"- target: `{cmp.get('target')}`",
        "",
        "## Transitions",
        "",
        f"- total_aligned: {trans.get('total_aligned_samples')}",
        f"- applicable: {trans.get('applicable_samples')}",
        f"- not_applicable: {trans.get('not_applicable_samples')}",
        f"- excluded_format: {trans.get('excluded_format_samples')}",
        f"- transition_eligible: {trans.get('transition_eligible')}",
        f"- stable_correct: {trans.get('stable_correct')}",
        f"- gain: {trans.get('gain')}",
        f"- regression: {trans.get('regression')}",
        f"- both_wrong: {trans.get('both_wrong')}",
        f"- net_gain: {trans.get('net_gain')}",
        "",
        "## Metric deltas (point estimate)",
        "",
    ]
    for name, block in deltas.items():
        lines.append(
            f"- `{name}`: {block.get('baseline')} → {block.get('candidate')} "
            f"(Δ={block.get('delta')})"
        )
    primary = (metrics.get("metric_deltas") or {}).get("primary")
    if primary and "delta" in primary:
        lines += [
            "",
            f"Primary `{primary.get('metric')}` Δ = {primary.get('delta')}",
        ]
    lines += [
        "",
        "## Provenance",
        "",
        f"- git_sha: `{(manifest.get('provenance') or {}).get('git_sha')}`",
        f"- config_hash: `{(manifest.get('provenance') or {}).get('config_hash')}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
