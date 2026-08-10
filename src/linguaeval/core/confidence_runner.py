"""Offline confidence extraction + calibration metrics (P1.5-A/B)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.registry import get_adapter
from linguaeval.confidence.extract import extract_confidence_records, summarize_confidence
from linguaeval.confidence.metrics import compute_calibration_metrics
from linguaeval.core.fingerprint import build_provenance
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import ConfidenceSpec, OutputSpec, RunManifest, TaskSpec
from linguaeval.parse.pipeline import apply_output_spec


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve(base: Path, maybe: Optional[str]) -> Optional[Path]:
    if not maybe:
        return None
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def _resolve_out_dir(config_path: Path, out_dir_raw: str) -> Path:
    out_dir = Path(out_dir_raw)
    if out_dir.is_absolute():
        return out_dir
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "src" / "linguaeval").exists():
            return parent / out_dir_raw
    return config_path.parent / out_dir_raw


def _fmt_metric(block: Dict[str, Any]) -> str:
    st = block.get("status")
    if st != "AVAILABLE":
        reason = block.get("reason")
        return f"{st}" + (f" ({reason})" if reason else "")
    val = block.get("value")
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def run_offline_confidence(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    root = config_path.parent

    task_path = _resolve(root, cfg.get("task_spec") or cfg.get("task"))
    if not task_path or not task_path.is_file():
        raise FileNotFoundError(f"task_spec not found: {task_path}")
    task = TaskSpec.from_dict(_load_yaml(task_path))

    conf_path = _resolve(root, cfg.get("confidence_spec"))
    if conf_path and conf_path.is_file():
        conf_cfg = _load_yaml(conf_path)
    else:
        conf_cfg = dict(cfg.get("confidence") or {})
    spec = ConfidenceSpec.from_dict(conf_cfg)

    source = dict(cfg.get("source") or {})
    adapter_name = source.get("adapter") or source.get("type") or "jsonl"
    adapter = get_adapter(str(adapter_name))
    samples, preds = adapter(source, root, cfg)

    output_path = _resolve(root, cfg.get("output_spec") or cfg.get("output"))
    output_spec = OutputSpec.from_dict(
        _load_yaml(output_path) if output_path and output_path.is_file() else {}
    )
    parse_mode = (cfg.get("parse") or {}).get("mode") or "from_parsed"
    preds = apply_output_spec(preds, output_spec, mode=parse_mode)

    records = extract_confidence_records(samples, preds, spec=spec, task=task)
    audit = {
        "target": spec.target,
        "source": {"type": spec.source.type, "path": spec.source.path},
        **summarize_confidence(records),
    }

    cal_cfg = dict(cfg.get("calibration") or {})
    calibration = compute_calibration_metrics(
        records,
        n_bins=int(cal_cfg.get("n_bins") or 10),
        min_samples=int(cal_cfg.get("min_samples") or 10),
    )
    calibration["target"] = spec.target

    out_dir = _resolve_out_dir(
        config_path, cfg.get("output_dir") or "results/07_confidence_offline"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "confidence_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    audit_path = out_dir / "confidence_audit.json"
    write_json(audit_path, audit)

    cal_path = out_dir / "calibration_metrics.json"
    write_json(cal_path, calibration)

    m = calibration.get("metrics") or {}
    report_path = out_dir / "report.md"
    lines = [
        f"# LinguaEval Confidence — `{spec.target}`",
        "",
        f"- source.type: `{spec.source.type}`",
        f"- source.path: `{spec.source.path}`",
        f"- n_records: {audit['n_records']}",
        f"- AVAILABLE: {audit['counts'].get('AVAILABLE', 0)}",
        f"- NOT_AVAILABLE: {audit['counts'].get('NOT_AVAILABLE', 0)}",
        f"- NOT_APPLICABLE: {audit['counts'].get('NOT_APPLICABLE', 0)}",
        f"- availability_rate: {audit.get('availability_rate')}",
        "",
        "## Calibration (P1.5-B)",
        "",
        f"- pack status: `{calibration.get('status')}`",
        f"- n_usable: {calibration.get('n_usable')}",
        f"- ECE: {_fmt_metric(m.get('ece') or {})}",
        f"- Brier: {_fmt_metric(m.get('brier') or {})}",
        f"- NLL: {_fmt_metric(m.get('nll') or {})}",
        f"- AUROC (OVR macro): {_fmt_metric(m.get('auroc_ovr_macro') or {})}",
        f"- accuracy: {_fmt_metric(m.get('accuracy') or {})}",
        "",
    ]
    if calibration.get("status") == "NOT_AVAILABLE":
        lines.append(
            "**Calibration NOT_AVAILABLE** — no usable confidence scores "
            "(expected for free-generation predictions without scores)."
        )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    provenance = build_provenance(
        config_path=config_path,
        cfg=cfg,
        task_path=task_path,
        output_path=output_path,
        metric_path=None,
        sample_dicts=[s.to_dict() for s in samples],
        prediction_dicts=[p.to_dict() for p in preds],
    )
    run_id = cfg.get("run_id") or f"confidence_{uuid4().hex[:8]}"
    manifest = RunManifest(
        run_id=run_id,
        config_path=str(config_path.resolve()),
        packs=list(cfg.get("packs") or ["confidence"]),
        provenance=provenance,
        notes={
            "mode": "offline_confidence",
            "target": spec.target,
            "source_type": spec.source.type,
            "availability_rate": audit.get("availability_rate"),
            "calibration_status": calibration.get("status"),
        },
        artifact_index={
            "confidence_records": str(records_path),
            "confidence_audit": str(audit_path),
            "calibration_metrics": str(cal_path),
            "report": str(report_path),
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return out_dir
