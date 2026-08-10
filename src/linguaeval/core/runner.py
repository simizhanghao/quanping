"""Offline scoring runner — Kernel has no business-specific branches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.jsonl_samples import write_predictions_jsonl
from linguaeval.adapters.dataset.registry import get_adapter
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import (
    MetricSpec,
    OutputSpec,
    PredictionRecord,
    RunManifest,
    SampleRecord,
    ScoreRecord,
    TaskSpec,
)
from linguaeval.metrics.aggregate import build_business_metrics
from linguaeval.metrics.classification import score_targets
from linguaeval.metrics.score_records import build_score_records
from linguaeval.parse.pipeline import apply_output_spec
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
    """Dispatch via DatasetAdapter registry only (no business ifs)."""
    root = config_path.parent
    source = dict(cfg.get("source") or {})
    adapter_name = (
        source.get("adapter")
        or source.get("type")
        or cfg.get("source_type")
        or "jsonl"
    )
    adapter = get_adapter(str(adapter_name))
    return adapter(source, root, cfg)


def _write_scores_jsonl(path: Path, scores: List[ScoreRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def run_offline_score(config_path: Path) -> Path:
    cfg, task, output_spec, metric_spec = load_bundle(config_path)
    samples, preds = load_samples_and_preds(cfg, config_path)

    parse_mode = (
        (cfg.get("parse") or {}).get("mode")
        or cfg.get("prediction_mode")
        or "from_parsed"
    )
    if parse_mode not in {"from_raw", "from_parsed"}:
        raise ValueError(f"Unsupported parse.mode={parse_mode!r}; use from_raw|from_parsed")

    preds = apply_output_spec(preds, output_spec, mode=parse_mode)
    score_rows = build_score_records(samples, preds, task)
    scored = score_targets(samples, preds, task, metric_spec)
    business = build_business_metrics(scored, report_cfg=cfg.get("report") or {})
    business["parse"] = {
        "mode": parse_mode,
        "parser": output_spec.parser,
        "format": output_spec.format,
    }

    out_dir_raw = cfg.get("output_dir") or "results/01_eval_offline"
    out_dir = Path(out_dir_raw)
    if not out_dir.is_absolute():
        repo = config_path.resolve()
        for parent in [repo.parent, *repo.parents]:
            if (parent / "pyproject.toml").exists() or (parent / "src" / "linguaeval").exists():
                out_dir = parent / out_dir_raw
                break
        else:
            out_dir = config_path.parent / out_dir_raw
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = cfg.get("run_id") or f"offline_{uuid4().hex[:8]}"
    source = cfg.get("source") or {}
    adapter_name = source.get("adapter") or source.get("type") or "jsonl"
    manifest = RunManifest(
        run_id=run_id,
        config_path=str(config_path.resolve()),
        packs=list(cfg.get("packs") or ["business", "schema"]),
        artifact_index={},
        notes={
            "mode": "offline_score",
            "adapter": adapter_name,
            "parse_mode": parse_mode,
            "n_samples": len(samples),
            "n_predictions": len(preds),
            "n_score_records": len(score_rows),
            "task": task.name,
            "report": cfg.get("report") or {},
        },
    )

    pred_out = out_dir / "predictions.jsonl"
    scores_out = out_dir / "scores.jsonl"
    write_predictions_jsonl(pred_out, preds)
    _write_scores_jsonl(scores_out, score_rows)
    business_path = out_dir / "business_metrics.json"
    schema_path = out_dir / "schema_metrics.json"
    report_path = out_dir / "report.md"
    manifest_path = out_dir / "manifest.json"

    write_json(business_path, business)
    write_json(
        schema_path,
        {
            **(business.get("schema") or {}),
            "parse_mode": parse_mode,
            "parser": output_spec.parser,
        },
    )
    write_report_md(report_path, business=business, manifest=manifest.to_dict())

    manifest.artifact_index = {
        "predictions": str(pred_out),
        "scores": str(scores_out),
        "business_metrics": str(business_path),
        "schema_metrics": str(schema_path),
        "report": str(report_path),
    }
    write_manifest(manifest_path, manifest)
    write_json(out_dir / "score_raw.json", scored)
    return out_dir
