"""Offline self-consistency evaluation (D8 / P2-E)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.jsonl_samples import load_predictions_jsonl, load_samples_jsonl
from linguaeval.consistency.aggregate import aggregate_consistency, build_consistency_records
from linguaeval.core.fingerprint import build_provenance
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import OutputSpec, RunManifest, TaskSpec
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


def run_offline_consistency(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    root = config_path.parent

    task_path = _resolve(root, cfg.get("task_spec") or cfg.get("task"))
    if not task_path or not task_path.is_file():
        raise FileNotFoundError(f"task_spec not found: {task_path}")
    task = TaskSpec.from_dict(_load_yaml(task_path))

    cons = dict(cfg.get("consistency") or {})
    target = cons.get("target") or cfg.get("target")
    if not target:
        raise ValueError("consistency.target is required")
    min_reps = int(cons.get("min_replicates") or 2)

    source = dict(cfg.get("source") or {})
    samples_path = _resolve(root, source.get("samples"))
    preds_path = _resolve(root, source.get("predictions"))
    if not samples_path or not samples_path.is_file():
        raise FileNotFoundError(f"samples not found: {samples_path}")
    if not preds_path or not preds_path.is_file():
        raise FileNotFoundError(f"predictions not found: {preds_path}")

    samples = load_samples_jsonl(samples_path)
    preds = load_predictions_jsonl(preds_path)

    output_path = _resolve(root, cfg.get("output_spec") or cfg.get("output"))
    output_spec = OutputSpec.from_dict(
        _load_yaml(output_path) if output_path and output_path.is_file() else {}
    )
    parse_mode = (cfg.get("parse") or {}).get("mode") or "from_parsed"
    preds = apply_output_spec(preds, output_spec, mode=parse_mode)

    records = build_consistency_records(
        samples, preds, task, target=str(target), min_replicates=min_reps
    )
    metrics = aggregate_consistency(records)
    metrics["consistency"] = {"target": target, "min_replicates": min_reps}

    out_dir = _resolve_out_dir(config_path, cfg.get("output_dir") or "results/19_consistency")
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "consistency_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out_dir / "consistency_metrics.json", metrics)

    lines = [
        "# LinguaEval Consistency (D8)",
        "",
        f"- status: `{metrics.get('status')}`",
        f"- target: `{target}`",
        f"- n_applicable: {metrics.get('coverage', {}).get('n_applicable')}",
        "",
    ]
    for tname, block in (metrics.get("by_target") or {}).items():
        lines += [
            f"## Target `{tname}`",
            "",
            f"- pairwise_agreement_rate: `{block.get('pairwise_agreement_rate')}`",
            f"- all_agree_rate: `{block.get('all_agree_rate')}`",
            f"- majority_accuracy: `{block.get('majority_accuracy')}`",
            "",
        ]
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
    write_manifest(
        out_dir / "manifest.json",
        RunManifest(
            run_id=cfg.get("run_id") or f"consistency_{uuid4().hex[:8]}",
            config_path=str(config_path.resolve()),
            packs=list(cfg.get("packs") or ["consistency"]),
            provenance=provenance,
            notes={"mode": "offline_consistency", "status": metrics.get("status")},
            artifact_index={
                "consistency_records": str(records_path),
                "consistency_metrics": str(out_dir / "consistency_metrics.json"),
                "report": str(report_path),
            },
        ),
    )
    return out_dir
