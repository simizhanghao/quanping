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

    cov = business.get("coverage") or {}
    if cov:
        lines += [
            "## Coverage",
            "",
            f"- eligible: {cov.get('eligible_samples')}",
            f"- with_prediction: {cov.get('with_prediction')}",
            f"- format_ok: {cov.get('format_ok_samples')}",
            f"- coverage_prediction: {cov.get('coverage_prediction')}",
            f"- coverage_valid: {cov.get('coverage_valid')}",
            "",
        ]

    if business.get("primary"):
        p = business["primary"]
        lines += [
            "## Primary Business Metrics (semantic default)",
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

    modes = business.get("metrics_by_mode") or {}
    if modes:
        lines += ["## Metrics by Mode", ""]
        for mode in ("semantic", "strict"):
            block = modes.get(mode) or {}
            lines.append(f"### {mode}")
            for name, tblock in (block.get("targets") or {}).items():
                bits = []
                for k in ("exact_match", "f1", "accuracy", "macro_f1", "denominator"):
                    if k in tblock and tblock[k] is not None:
                        bits.append(f"{k}={tblock[k]}")
                lines.append(f"- `{name}`: " + (", ".join(bits) if bits else str(tblock)))
            if block.get("joint"):
                lines.append(
                    f"- joint: {block['joint'].get('exact_joint_success')} "
                    f"(den={block['joint'].get('denominator')})"
                )
            lines.append("")

    lines += ["## Targets (semantic)", ""]
    for name, block in (business.get("targets") or {}).items():
        lines.append(f"### `{name}` ({block.get('type')})")
        for k in ("precision", "recall", "f1", "f2", "accuracy", "macro_f1", "exact_match"):
            if k in block:
                lines.append(f"- {k}: {block[k]}")
        lines.append("")
    if business.get("joint"):
        lines += [
            "## Joint (semantic)",
            "",
            f"- exact_joint_success: {business['joint'].get('exact_joint_success')}",
            "",
        ]

    prov = (manifest.get("provenance") or {})
    if prov:
        lines += [
            "## Provenance",
            "",
            f"- git_sha: `{prov.get('git_sha')}`",
            f"- config_hash: `{prov.get('config_hash')}`",
            f"- dataset_fingerprint: `{prov.get('dataset_fingerprint')}`",
            f"- prediction_fingerprint: `{prov.get('prediction_fingerprint')}`",
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
