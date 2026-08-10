"""Offline metamorphic robustness evaluation (P2-A)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.jsonl_samples import load_predictions_jsonl, load_samples_jsonl
from linguaeval.core.fingerprint import build_provenance
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import (
    MetamorphicRelationSpec,
    OutputSpec,
    RunManifest,
    TaskSpec,
    VariantRecord,
)
from linguaeval.parse.pipeline import apply_output_spec
from linguaeval.robustness.aggregate import aggregate_robustness, build_robustness_records
from linguaeval.robustness.registry import ensure_builtin_perturbation_specs, list_perturbations


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


def _load_variants(path: Path) -> List[VariantRecord]:
    rows: List[VariantRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(VariantRecord.from_dict(json.loads(line)))
    return rows


def run_offline_robustness(config_path: Path) -> Path:
    ensure_builtin_perturbation_specs()
    cfg = _load_yaml(config_path)
    root = config_path.parent

    task_path = _resolve(root, cfg.get("task_spec") or cfg.get("task"))
    if not task_path or not task_path.is_file():
        raise FileNotFoundError(f"task_spec not found: {task_path}")
    task = TaskSpec.from_dict(_load_yaml(task_path))

    rel_cfg = dict(cfg.get("relation") or cfg.get("metamorphic") or {})
    if not rel_cfg.get("targets"):
        rel_cfg["targets"] = [t.name for t in task.targets]
    relation = MetamorphicRelationSpec.from_dict(rel_cfg)

    source = dict(cfg.get("source") or {})
    samples_path = _resolve(root, source.get("samples"))
    clean_path = _resolve(root, source.get("predictions") or source.get("clean_predictions"))
    variants_path = _resolve(root, source.get("variants"))
    var_pred_path = _resolve(
        root, source.get("variant_predictions") or source.get("perturbed_predictions")
    )
    for label, p in (
        ("samples", samples_path),
        ("predictions", clean_path),
        ("variants", variants_path),
        ("variant_predictions", var_pred_path),
    ):
        if not p or not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    samples = load_samples_jsonl(samples_path)
    clean_preds = load_predictions_jsonl(clean_path)
    variants = _load_variants(variants_path)
    variant_preds = load_predictions_jsonl(var_pred_path)

    output_path = _resolve(root, cfg.get("output_spec") or cfg.get("output"))
    output_spec = OutputSpec.from_dict(
        _load_yaml(output_path) if output_path and output_path.is_file() else {}
    )
    parse_mode = (cfg.get("parse") or {}).get("mode") or "from_parsed"
    clean_preds = apply_output_spec(clean_preds, output_spec, mode=parse_mode)
    variant_preds = apply_output_spec(variant_preds, output_spec, mode=parse_mode)

    records = build_robustness_records(
        samples=samples,
        clean_preds=clean_preds,
        variants=variants,
        variant_preds=variant_preds,
        task=task,
        relation=relation,
    )
    metrics = aggregate_robustness(records, bootstrap=dict(cfg.get("bootstrap") or {}))
    metrics["relation"] = {"type": relation.type, "targets": relation.targets}
    metrics["registry_perturbations"] = list_perturbations()

    out_dir = _resolve_out_dir(config_path, cfg.get("output_dir") or "results/15_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "robustness_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    viol_path = out_dir / "violations.jsonl"
    with viol_path.open("w", encoding="utf-8") as f:
        for r in records:
            if r.applicable and r.relation_satisfied is False:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    metrics_path = out_dir / "robustness_metrics.json"
    write_json(metrics_path, metrics)

    lines = [
        f"# LinguaEval Robustness — `{relation.type}`",
        "",
        f"- status: `{metrics.get('status')}`",
        f"- targets: `{relation.targets}`",
        f"- n_records: {metrics.get('coverage', {}).get('n_records')}",
        f"- n_applicable: {metrics.get('coverage', {}).get('n_applicable')}",
        "",
    ]
    for tname, block in (metrics.get("by_target") or {}).items():
        lines += [
            f"## Target `{tname}`",
            "",
            f"- accuracy_clean: `{block.get('accuracy_clean')}`",
            f"- accuracy_perturbed: `{block.get('accuracy_perturbed')}`",
            f"- delta_accuracy: `{block.get('delta_accuracy')}`",
            f"- flip_rate: `{block.get('flip_rate')}`",
            f"- metamorphic_violation_rate: `{block.get('metamorphic_violation_rate')}`",
            f"- robust_success_rate: `{block.get('robust_success_rate')}`",
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
        prediction_dicts=[p.to_dict() for p in clean_preds],
    )
    run_id = cfg.get("run_id") or f"robustness_{uuid4().hex[:8]}"
    manifest = RunManifest(
        run_id=run_id,
        config_path=str(config_path.resolve()),
        packs=list(cfg.get("packs") or ["robustness"]),
        provenance=provenance,
        notes={
            "mode": "offline_robustness",
            "relation_type": relation.type,
            "status": metrics.get("status"),
            "n_variants": len(variants),
        },
        artifact_index={
            "robustness_records": str(records_path),
            "violations": str(viol_path),
            "robustness_metrics": str(metrics_path),
            "report": str(report_path),
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return out_dir
