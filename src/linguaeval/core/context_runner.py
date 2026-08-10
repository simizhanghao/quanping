"""Offline context ablation evaluation (D6 / P2-E)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.jsonl_samples import load_predictions_jsonl, load_samples_jsonl
from linguaeval.context.aggregate import aggregate_context_ablation, build_context_ablation_records
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


def run_offline_context(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    root = config_path.parent

    task_path = _resolve(root, cfg.get("task_spec") or cfg.get("task"))
    if not task_path or not task_path.is_file():
        raise FileNotFoundError(f"task_spec not found: {task_path}")
    task = TaskSpec.from_dict(_load_yaml(task_path))

    ctx = dict(cfg.get("context") or {})
    target = ctx.get("target") or cfg.get("target")
    if not target:
        raise ValueError("context.target is required")
    denominator = str(ctx.get("denominator") or "semantic")

    source = dict(cfg.get("source") or {})
    samples_path = _resolve(root, source.get("samples"))
    without_path = _resolve(
        root,
        source.get("predictions_without_context")
        or (cfg.get("without_context") or {}).get("predictions")
        or (cfg.get("without_context") or {}).get("source", {}).get("predictions"),
    )
    with_path = _resolve(
        root,
        source.get("predictions_with_context")
        or (cfg.get("with_context") or {}).get("predictions")
        or (cfg.get("with_context") or {}).get("source", {}).get("predictions"),
    )
    for label, p in (
        ("samples", samples_path),
        ("predictions_without_context", without_path),
        ("predictions_with_context", with_path),
    ):
        if not p or not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    samples = load_samples_jsonl(samples_path)
    without_preds = load_predictions_jsonl(without_path)
    with_preds = load_predictions_jsonl(with_path)

    output_path = _resolve(root, cfg.get("output_spec") or cfg.get("output"))
    output_spec = OutputSpec.from_dict(
        _load_yaml(output_path) if output_path and output_path.is_file() else {}
    )
    parse_mode = (cfg.get("parse") or {}).get("mode") or "from_parsed"
    without_preds = apply_output_spec(without_preds, output_spec, mode=parse_mode)
    with_preds = apply_output_spec(with_preds, output_spec, mode=parse_mode)

    records = build_context_ablation_records(
        samples=samples,
        without_preds=without_preds,
        with_preds=with_preds,
        task=task,
        target=str(target),
        denominator=denominator,
    )
    metrics = aggregate_context_ablation(records)
    metrics["context"] = {"target": target, "denominator": denominator}

    out_dir = _resolve_out_dir(config_path, cfg.get("output_dir") or "results/20_context")
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "context_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out_dir / "context_metrics.json", metrics)

    for name, wanted in (
        ("context_gain_cases.jsonl", "gain"),
        ("context_regression_cases.jsonl", "regression"),
    ):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for r in records:
                if r.get("applicable") and r.get("transition") == wanted:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = [
        "# LinguaEval Context Ablation (D6)",
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
            f"- accuracy_without_context: `{block.get('accuracy_without_context')}`",
            f"- accuracy_with_context: `{block.get('accuracy_with_context')}`",
            f"- delta_accuracy: `{block.get('delta_accuracy')}`",
            f"- prediction_flip_rate: `{block.get('prediction_flip_rate')}`",
            f"- context_gain_rate: `{block.get('context_gain_rate')}`",
            f"- context_regression_rate: `{block.get('context_regression_rate')}`",
            f"- transitions: `{block.get('transitions')}`",
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
        prediction_dicts=[p.to_dict() for p in without_preds],
    )
    write_manifest(
        out_dir / "manifest.json",
        RunManifest(
            run_id=cfg.get("run_id") or f"context_{uuid4().hex[:8]}",
            config_path=str(config_path.resolve()),
            packs=list(cfg.get("packs") or ["context"]),
            provenance=provenance,
            notes={"mode": "offline_context_ablation", "status": metrics.get("status")},
            artifact_index={
                "context_records": str(records_path),
                "context_metrics": str(out_dir / "context_metrics.json"),
                "report": str(report_path),
            },
        ),
    )
    return out_dir
