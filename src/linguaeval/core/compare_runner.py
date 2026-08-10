"""Offline paired compare runner — baseline vs candidate (no business branches)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.registry import get_adapter
from linguaeval.compare.aggregate import (
    metric_deltas,
    summarize_transitions,
    write_comparison_jsonl,
    write_transition_cases,
)
from linguaeval.compare.alignment import AlignmentError, index_by_id, require_strict_alignment
from linguaeval.compare.bootstrap import build_pair_rows, run_paired_bootstrap
from linguaeval.compare.gates import evaluate_gates
from linguaeval.compare.report import write_compare_report_md
from linguaeval.compare.slices import build_slice_comparison
from linguaeval.compare.transitions import build_comparison_records
from linguaeval.core.fingerprint import build_provenance, fingerprint_records
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import (
    MetricSpec,
    OutputSpec,
    PredictionRecord,
    RunManifest,
    SampleRecord,
    TaskSpec,
)
from linguaeval.metrics.aggregate import build_business_metrics
from linguaeval.metrics.classification import score_targets
from linguaeval.metrics.score_records import build_score_records
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


def _load_side(
    side_cfg: Dict[str, Any],
    config_path: Path,
    shared_cfg: Dict[str, Any],
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    source = dict(side_cfg.get("source") or side_cfg)
    # allow side to inherit adapter from top-level if omitted
    if "adapter" not in source and "type" not in source:
        top = shared_cfg.get("source") or {}
        if isinstance(top, dict):
            source.setdefault("adapter", top.get("adapter") or top.get("type"))
    adapter_name = source.get("adapter") or source.get("type") or "jsonl"
    adapter = get_adapter(str(adapter_name))
    # Merge side source over a shallow copy so samples path can live at top-level for jsonl
    merged = dict(shared_cfg)
    if "samples" in shared_cfg and "samples" not in source:
        source = {**source, "samples": shared_cfg.get("samples")}
    return adapter(source, config_path.parent, merged)


def run_offline_compare(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    root = config_path.parent
    compare_cfg = dict(cfg.get("compare") or {})
    target = compare_cfg.get("target")
    if not target:
        raise ValueError("compare.target is required")
    denominator = compare_cfg.get("denominator") or "semantic"
    if denominator not in {"semantic", "strict"}:
        raise ValueError("compare.denominator must be semantic|strict")

    task_path = _resolve(root, cfg.get("task_spec") or cfg.get("task"))
    output_path = _resolve(root, cfg.get("output_spec") or cfg.get("output"))
    metric_path = _resolve(root, cfg.get("metric_spec") or cfg.get("metrics"))
    if not task_path or not task_path.is_file():
        raise FileNotFoundError(f"task_spec not found: {task_path}")
    if not metric_path or not metric_path.is_file():
        raise FileNotFoundError(f"metric_spec not found: {metric_path}")

    task = TaskSpec.from_dict(_load_yaml(task_path))
    output_spec = OutputSpec.from_dict(
        _load_yaml(output_path) if output_path and output_path.is_file() else {}
    )
    metric_spec = MetricSpec.from_dict(_load_yaml(metric_path))
    if target not in {t.name for t in task.targets}:
        raise KeyError(f"compare.target={target!r} not in TaskSpec targets")

    baseline_cfg = dict(cfg.get("baseline") or {})
    candidate_cfg = dict(cfg.get("candidate") or {})
    if not baseline_cfg or not candidate_cfg:
        raise ValueError("config must define baseline: and candidate: blocks")

    samples_b, preds_b = _load_side(baseline_cfg, config_path, cfg)
    samples_c, preds_c = _load_side(candidate_cfg, config_path, cfg)

    # Prefer shared samples file when present (jsonl); else baseline samples as gold
    shared_samples_path = _resolve(root, cfg.get("samples"))
    if shared_samples_path and shared_samples_path.is_file():
        from linguaeval.adapters.dataset.jsonl_samples import load_samples_jsonl

        samples = load_samples_jsonl(shared_samples_path)
    else:
        samples = samples_b

    parse_mode = (
        (cfg.get("parse") or {}).get("mode")
        or cfg.get("prediction_mode")
        or "from_parsed"
    )
    preds_b = apply_output_spec(preds_b, output_spec, mode=parse_mode)
    preds_c = apply_output_spec(preds_c, output_spec, mode=parse_mode)

    try:
        align_audit = require_strict_alignment(preds_b, preds_c)
    except AlignmentError:
        # still write a minimal audit dir if output_dir known — re-raise after
        raise

    # Gold sample ids must cover prediction ids
    sample_ids = {s.sample_id for s in samples}
    pred_ids = {p.sample_id for p in preds_b}
    missing_gold = sorted(pred_ids - sample_ids)
    if missing_gold:
        raise AlignmentError(
            f"samples missing gold for sample_id={missing_gold[0]} "
            f"(count={len(missing_gold)})"
        )

    # Keep samples ordered by baseline prediction order
    sample_map = index_by_id(samples)
    ordered_ids = [p.sample_id for p in preds_b]
    samples_aligned = [sample_map[i] for i in ordered_ids]
    preds_b_map = index_by_id(preds_b)
    preds_c_map = index_by_id(preds_c)
    preds_b_ord = [preds_b_map[i] for i in ordered_ids]
    preds_c_ord = [preds_c_map[i] for i in ordered_ids]

    scores_b = build_score_records(samples_aligned, preds_b_ord, task)
    scores_c = build_score_records(samples_aligned, preds_c_ord, task)
    records, tcounts = build_comparison_records(
        scores_b,
        scores_c,
        target=str(target),
        denominator=denominator,
        ordered_ids=ordered_ids,
    )

    scored_b = score_targets(samples_aligned, preds_b_ord, task, metric_spec)
    scored_c = score_targets(samples_aligned, preds_c_ord, task, metric_spec)
    report_cfg = cfg.get("report") or {}
    biz_b = build_business_metrics(scored_b, report_cfg=report_cfg)
    biz_c = build_business_metrics(scored_c, report_cfg=report_cfg)
    deltas = metric_deltas(biz_b, biz_c, target=str(target))

    target_spec = next(t for t in task.targets if t.name == target)
    stats_cfg = dict(cfg.get("statistics") or {})
    slices_cfg = dict(cfg.get("slices") or {})
    bootstrap_unit = str(stats_cfg.get("bootstrap_unit") or "sample")
    metric_names = list(
        stats_cfg.get("metrics")
        or slices_cfg.get("metrics")
        or metric_spec.metrics.get(str(target))
        or []
    )
    metric_names = [
        m
        for m in metric_names
        if str(m).lower()
        in {
            "precision",
            "recall",
            "f1",
            "accuracy",
            "macro_f1",
            "exact_match",
        }
    ]
    if not metric_names and report_cfg.get("primary_metric"):
        metric_names = [str(report_cfg["primary_metric"])]

    pair_rows = build_pair_rows(
        samples_aligned,
        scores_b,
        scores_c,
        records,
        target=str(target),
        bootstrap_unit=bootstrap_unit,
        denominator=denominator,
    )

    statistics: Optional[Dict[str, Any]] = None
    if bool(stats_cfg.get("enabled", False)):
        statistics = {
            "enabled": True,
            "bootstrap_unit": bootstrap_unit,
            **run_paired_bootstrap(
                pair_rows,
                target_type=target_spec.type,
                metric_names=metric_names,
                n_bootstrap=int(stats_cfg.get("n_bootstrap") or 1000),
                confidence_level=float(
                    stats_cfg.get("confidence_level") or stats_cfg.get("ci") or 0.95
                ),
                seed=int(stats_cfg.get("seed") or 42),
                labels=target_spec.labels,
                round_digits=metric_spec.round_digits,
            ),
        }

    slice_comparison: Optional[Dict[str, Any]] = None
    if bool(slices_cfg.get("enabled", False)):
        slice_comparison = build_slice_comparison(
            samples_aligned,
            scores_b,
            scores_c,
            pair_rows,
            target=str(target),
            target_type=target_spec.type,
            metric_names=metric_names or [str(report_cfg.get("primary_metric") or "accuracy")],
            slice_specs=list(slices_cfg.get("specs") or []),
            labels=target_spec.labels,
            round_digits=metric_spec.round_digits,
            min_support=int(slices_cfg.get("min_support") or 1),
        )

    comparison_metrics: Dict[str, Any] = {
        "compare": {
            "target": target,
            "denominator": denominator,
            "display": compare_cfg.get("display")
            or {"baseline": "baseline", "candidate": "candidate"},
        },
        "transitions": summarize_transitions(tcounts),
        "metric_deltas": deltas,
        "statistics": statistics,
        "slices": slice_comparison,
        "baseline_business": {
            "primary": biz_b.get("primary"),
            "targets": {str(target): (biz_b.get("targets") or {}).get(target)},
            "schema": biz_b.get("schema"),
        },
        "candidate_business": {
            "primary": biz_c.get("primary"),
            "targets": {str(target): (biz_c.get("targets") or {}).get(target)},
            "schema": biz_c.get("schema"),
        },
    }

    gate_result: Optional[Dict[str, Any]] = None
    gate_specs = list(cfg.get("gates") or [])
    if gate_specs:
        gate_result = evaluate_gates(comparison_metrics, gate_specs)
        comparison_metrics["gate"] = gate_result

    out_dir = _resolve_out_dir(config_path, cfg.get("output_dir") or "results/05_compare_offline")
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = build_provenance(
        config_path=config_path,
        cfg=cfg,
        task_path=task_path,
        output_path=output_path,
        metric_path=metric_path,
        sample_dicts=[s.to_dict() for s in samples_aligned],
        prediction_dicts=[p.to_dict() for p in preds_b_ord],
    )
    provenance["candidate_prediction_fingerprint"] = fingerprint_records(
        [p.to_dict() for p in preds_c_ord]
    )
    provenance["baseline_prediction_path"] = str(
        _resolve(root, (baseline_cfg.get("source") or baseline_cfg).get("path")
                 or (baseline_cfg.get("source") or baseline_cfg).get("predictions"))
        or ""
    )
    provenance["candidate_prediction_path"] = str(
        _resolve(root, (candidate_cfg.get("source") or candidate_cfg).get("path")
                 or (candidate_cfg.get("source") or candidate_cfg).get("predictions"))
        or ""
    )

    run_id = cfg.get("run_id") or f"compare_{uuid4().hex[:8]}"
    manifest = RunManifest(
        run_id=run_id,
        config_path=str(config_path.resolve()),
        packs=list(cfg.get("packs") or ["compare"]),
        provenance=provenance,
        notes={
            "mode": "offline_compare",
            "target": target,
            "denominator": denominator,
            "n_aligned": len(ordered_ids),
            "parse_mode": parse_mode,
        },
    )

    records_path = out_dir / "comparison_records.jsonl"
    metrics_path = out_dir / "comparison_metrics.json"
    audit_path = out_dir / "alignment_audit.json"
    report_path = out_dir / "report.md"
    write_comparison_jsonl(records_path, records)
    case_paths = write_transition_cases(out_dir, records)
    write_json(
        audit_path,
        {
            **align_audit.to_dict(),
            "transitions": summarize_transitions(tcounts),
            "denominator": denominator,
            "target": target,
        },
    )
    write_json(metrics_path, comparison_metrics)
    stats_path = out_dir / "statistics.json"
    if statistics is not None:
        write_json(stats_path, statistics)
    slices_path = out_dir / "slice_comparison.json"
    if slice_comparison is not None:
        write_json(slices_path, slice_comparison)
    gate_path = out_dir / "gate.json"
    if gate_result is not None:
        write_json(gate_path, gate_result)
    write_compare_report_md(
        report_path, metrics=comparison_metrics, manifest=manifest.to_dict()
    )

    manifest.artifact_index = {
        "comparison_metrics": str(metrics_path),
        "comparison_records": str(records_path),
        "alignment_audit": str(audit_path),
        "report": str(report_path),
        **{f"{k}_cases": v for k, v in case_paths.items()},
    }
    if statistics is not None:
        manifest.artifact_index["statistics"] = str(stats_path)
    if slice_comparison is not None:
        manifest.artifact_index["slice_comparison"] = str(slices_path)
    if gate_result is not None:
        manifest.artifact_index["gate"] = str(gate_path)
    manifest.notes["statistics_enabled"] = bool(statistics)
    manifest.notes["slices_enabled"] = bool(slice_comparison)
    manifest.notes["gate_status"] = (gate_result or {}).get("status")
    write_manifest(out_dir / "manifest.json", manifest)
    # keep side business dumps for debugging
    write_json(out_dir / "baseline_business_metrics.json", biz_b)
    write_json(out_dir / "candidate_business_metrics.json", biz_c)
    return out_dir
