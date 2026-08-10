"""Offline scoring runner (P0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.jsonl_samples import (
    load_predictions_jsonl,
    load_samples_jsonl,
    write_predictions_jsonl,
)
from linguaeval.adapters.dataset.n2s_dialogue import load_n2s_prediction_json
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import MetricSpec, OutputSpec, PredictionRecord, RunManifest, SampleRecord, TaskSpec
from linguaeval.metrics.aggregate import build_business_metrics
from linguaeval.metrics.classification import score_targets
from linguaeval.reports.markdown import write_report_md


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve(base: Path, maybe: Optional[str]) -> Optional[Path]:
    if not maybe:
        return None
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def load_bundle(config_path: Path) -> Tuple[Dict[str, Any], TaskSpec, OutputSpec, MetricSpec]:
    cfg = _load_yaml(config_path)
    root = config_path.parent
    task_path = _resolve(root, cfg.get("task_spec") or cfg.get("task"))
    output_path = _resolve(root, cfg.get("output_spec") or cfg.get("output"))
    metric_path = _resolve(root, cfg.get("metric_spec") or cfg.get("metrics"))
    if not task_path or not task_path.is_file():
        raise FileNotFoundError(f"task_spec not found: {task_path}")
    if not metric_path or not metric_path.is_file():
        raise FileNotFoundError(f"metric_spec not found: {metric_path}")
    task = TaskSpec.from_dict(_load_yaml(task_path))
    output = OutputSpec.from_dict(_load_yaml(output_path) if output_path and output_path.is_file() else {})
    metrics = MetricSpec.from_dict(_load_yaml(metric_path))
    return cfg, task, output, metrics


def load_samples_and_preds(
    cfg: Dict[str, Any],
    config_path: Path,
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    root = config_path.parent
    source = (cfg.get("source") or {}).get("type") or cfg.get("source_type") or "jsonl"

    if source == "n2s_dialogue_prediction":
        pred_path = _resolve(root, (cfg.get("source") or {}).get("path") or cfg.get("predictions"))
        if not pred_path or not pred_path.is_file():
            raise FileNotFoundError(f"N2S prediction JSON not found: {pred_path}")
        model_id = ((cfg.get("source") or {}).get("model_id")) or "sft"
        return load_n2s_prediction_json(pred_path, model_id=model_id)

    samples_path = _resolve(root, cfg.get("samples") or (cfg.get("source") or {}).get("samples"))
    preds_path = _resolve(root, cfg.get("predictions") or (cfg.get("source") or {}).get("predictions"))
    if not samples_path or not samples_path.is_file():
        raise FileNotFoundError(f"samples not found: {samples_path}")
    if not preds_path or not preds_path.is_file():
        raise FileNotFoundError(f"predictions not found: {preds_path}")
    return load_samples_jsonl(samples_path), load_predictions_jsonl(preds_path)


def run_offline_score(config_path: Path) -> Path:
    cfg, task, _output, metric_spec = load_bundle(config_path)
    samples, preds = load_samples_and_preds(cfg, config_path)

    scored = score_targets(samples, preds, task, metric_spec)
    business = build_business_metrics(scored)

    out_dir_raw = cfg.get("output_dir") or "results/01_eval_offline"
    out_dir = Path(out_dir_raw)
    if not out_dir.is_absolute():
        # prefer repo root (parents of configs/examples)
        repo = config_path.resolve()
        for parent in [repo.parent, *repo.parents]:
            if (parent / "pyproject.toml").exists() or (parent / "src" / "linguaeval").exists():
                out_dir = parent / out_dir_raw
                break
        else:
            out_dir = config_path.parent / out_dir_raw
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = cfg.get("run_id") or f"offline_{uuid4().hex[:8]}"
    manifest = RunManifest(
        run_id=run_id,
        config_path=str(config_path.resolve()),
        packs=list(cfg.get("packs") or ["business", "schema"]),
        artifact_index={},
        notes={
            "mode": "offline_score",
            "n_samples": len(samples),
            "n_predictions": len(preds),
            "task": task.name,
        },
    )

    pred_out = out_dir / "predictions.jsonl"
    write_predictions_jsonl(pred_out, preds)
    business_path = out_dir / "business_metrics.json"
    schema_path = out_dir / "schema_metrics.json"
    report_path = out_dir / "report.md"
    manifest_path = out_dir / "manifest.json"

    write_json(business_path, business)
    write_json(schema_path, business.get("schema") or {})
    write_report_md(report_path, business=business, manifest=manifest.to_dict())

    manifest.artifact_index = {
        "predictions": str(pred_out),
        "business_metrics": str(business_path),
        "schema_metrics": str(schema_path),
        "report": str(report_path),
    }
    write_manifest(manifest_path, manifest)

    # also dump raw score for debugging
    write_json(out_dir / "score_raw.json", scored)
    return out_dir
