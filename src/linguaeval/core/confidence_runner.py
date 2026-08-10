"""Offline confidence extraction runner (P1.5-A) — no calibration metrics yet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.registry import get_adapter
from linguaeval.confidence.extract import extract_confidence_records, summarize_confidence
from linguaeval.core.fingerprint import build_provenance
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import ConfidenceSpec, RunManifest, TaskSpec
from linguaeval.parse.pipeline import apply_output_spec
from linguaeval.core.schema import OutputSpec


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
        "P1.5-A extracts confidence only; ECE/Brier/thresholds are later slices.",
        "",
    ]
    if audit["counts"].get("NOT_AVAILABLE", 0) == audit["n_records"] and audit["n_records"]:
        lines.append(
            "**All records NOT_AVAILABLE** — confidence source missing "
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
        },
        artifact_index={
            "confidence_records": str(records_path),
            "confidence_audit": str(audit_path),
            "report": str(report_path),
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    write_json(out_dir / "confidence_audit.json", audit)
    return out_dir
