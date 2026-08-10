"""Offline selective prediction / risk-coverage runner (P1.5-D)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.registry import get_adapter
from linguaeval.confidence.extract import extract_confidence_records, summarize_confidence
from linguaeval.confidence.selective import compute_selective_metrics
from linguaeval.core.fingerprint import build_provenance
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import ConfidenceSpec, OutputSpec, RunManifest, SelectiveSpec, TaskSpec
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


def run_offline_selective(config_path: Path) -> Path:
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
    conf_spec = ConfidenceSpec.from_dict(conf_cfg)

    sel_path = _resolve(root, cfg.get("selective_spec"))
    if sel_path and sel_path.is_file():
        sel_cfg = _load_yaml(sel_path)
    else:
        sel_cfg = dict(cfg.get("selective") or {})
    if not sel_cfg.get("target"):
        sel_cfg = {**sel_cfg, "target": conf_spec.target}
    sel_spec = SelectiveSpec.from_dict(sel_cfg)

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

    records = extract_confidence_records(samples, preds, spec=conf_spec, task=task)
    audit = {
        "target": conf_spec.target,
        "source": {"type": conf_spec.source.type, "path": conf_spec.source.path},
        **summarize_confidence(records),
    }

    cal_cfg = dict(cfg.get("selective") or {})
    result = compute_selective_metrics(
        records,
        samples,
        sel_spec,
        min_samples=int(cal_cfg.get("min_samples") or 10),
    )

    out_dir = _resolve_out_dir(config_path, cfg.get("output_dir") or "results/13_selective")
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "confidence_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    write_json(out_dir / "confidence_audit.json", audit)

    curve = result.get("risk_coverage_curve") or []
    write_json(out_dir / "risk_coverage_curve.json", {"n": len(curve), "points": curve})

    slim = {k: v for k, v in result.items() if k != "risk_coverage_curve"}
    write_json(out_dir / "selective_metrics.json", slim)

    lines = [
        f"# LinguaEval Selective Prediction — `{sel_spec.target}`",
        "",
        f"- status: `{result.get('status')}`",
        f"- evaluate_on: `{sel_spec.evaluate_on}`",
        f"- n_evaluate: {result.get('n_evaluate')}",
        f"- AURC: `{result.get('aurc')}`",
        f"- full_coverage_risk: `{result.get('full_coverage_risk')}`",
        f"- accuracy_full: `{result.get('accuracy_full')}`",
        "",
        "## Risk@Coverage",
        "",
    ]
    for k, v in (result.get("risk_at_coverage") or {}).items():
        lines.append(f"- @{k}: `{v}`")
    lines += ["", "## Coverage@Risk", ""]
    for k, v in (result.get("coverage_at_risk") or {}).items():
        lines.append(f"- risk≤{k}: `{v}`")
    lines.append("")
    if result.get("status") == "NOT_AVAILABLE":
        lines += ["**NOT_AVAILABLE** — no usable confidence for selective prediction.", ""]

    report_path = out_dir / "report.md"
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
    run_id = cfg.get("run_id") or f"selective_{uuid4().hex[:8]}"
    manifest = RunManifest(
        run_id=run_id,
        config_path=str(config_path.resolve()),
        packs=list(cfg.get("packs") or ["selective"]),
        provenance=provenance,
        notes={
            "mode": "offline_selective",
            "status": result.get("status"),
            "aurc": result.get("aurc"),
        },
        artifact_index={
            "confidence_records": str(records_path),
            "confidence_audit": str(out_dir / "confidence_audit.json"),
            "risk_coverage_curve": str(out_dir / "risk_coverage_curve.json"),
            "selective_metrics": str(out_dir / "selective_metrics.json"),
            "report": str(report_path),
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return out_dir
