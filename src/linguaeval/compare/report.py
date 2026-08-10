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

    stats = metrics.get("statistics") or {}
    if stats.get("enabled"):
        lines += [
            "",
            "## Statistics (bootstrap CI)",
            "",
            f"- bootstrap_unit: `{stats.get('bootstrap_unit')}`",
            f"- n_units: {stats.get('n_units')} (rows={stats.get('n_rows')}, "
            f"cluster_mode={stats.get('cluster_mode')})",
            f"- n_bootstrap: {stats.get('n_bootstrap')}",
            f"- confidence_level: {stats.get('confidence_level')}",
            "",
        ]
        for name, block in (stats.get("metrics") or {}).items():
            d = block.get("delta") or {}
            lines.append(
                f"- Δ`{name}`: {d.get('point')}  "
                f"CI=[{d.get('ci_low')}, {d.get('ci_high')}]"
            )
        net = (stats.get("transitions") or {}).get("net_gain") or {}
        if net:
            lines.append(
                f"- net_gain: {net.get('point')}  "
                f"CI=[{net.get('ci_low')}, {net.get('ci_high')}]"
            )

    slices = (metrics.get("slices") or {}).get("slices") or {}
    if slices:
        lines += ["", "## Fixed slices", ""]
        for sname, sblock in slices.items():
            lines.append(f"### `{sname}` (source={sblock.get('source')})")
            for key, vblock in (sblock.get("values") or {}).items():
                # show primary-ish first metric
                mets = vblock.get("metrics") or {}
                m0 = next(iter(mets), None)
                if m0:
                    mb = mets[m0]
                    lines.append(
                        f"- `{key}` n={vblock.get('support')}: "
                        f"{m0} {mb.get('baseline')} → {mb.get('candidate')} "
                        f"(Δ={mb.get('delta')}; net_gain={((vblock.get('transitions') or {}).get('net_gain'))})"
                    )
                else:
                    lines.append(f"- `{key}` n={vblock.get('support')}")
            lines.append("")

    gate = metrics.get("gate") or {}
    if gate:
        lines += ["## Gate", "", f"- status: `{gate.get('status')}`", ""]
        for g in gate.get("gates") or []:
            lines.append(
                f"- `{g.get('id')}`: {g.get('status')} — {g.get('detail')} "
                f"(path=`{g.get('path')}`)"
            )
        lines.append("")

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
