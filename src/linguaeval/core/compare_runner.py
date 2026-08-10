"""Offline paired compare runner — baseline vs candidate (no business branches)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.registry import get_adapter
from linguaeval.compare.aggregate import (
    write_comparison_jsonl,
    write_transition_cases,
)
from linguaeval.compare.alignment import AlignmentError
from linguaeval.compare.gates import evaluate_gates
from linguaeval.compare.paired import compute_paired_comparison
from linguaeval.compare.protocol import (
    evaluate_comparability,
    validate_comparison_protocol,
)
from linguaeval.compare.report import write_compare_report_md
from linguaeval.compare.slices import build_slice_comparison
from linguaeval.core.fingerprint import build_provenance, fingerprint_records, sha256_file
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import (
    MetricSpec,
    OutputSpec,
    PredictionRecord,
    RunManifest,
    SampleRecord,
    TaskSpec,
)
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

    report_cfg = dict(cfg.get("report") or {})
    stats_cfg = dict(cfg.get("statistics") or {})
    slices_cfg = dict(cfg.get("slices") or {})

    try:
        paired = compute_paired_comparison(
            samples,
            preds_b,
            preds_c,
            task=task,
            metric_spec=metric_spec,
            target=str(target),
            denominator=denominator,
            report_cfg=report_cfg,
            stats_cfg=stats_cfg,
        )
    except AlignmentError:
        raise

    samples_aligned = paired["samples_aligned"]
    preds_b_ord = paired["preds_baseline"]
    preds_c_ord = paired["preds_candidate"]
    scores_b = paired["scores_baseline"]
    scores_c = paired["scores_candidate"]
    records = paired["records"]
    pair_rows = paired["pair_rows"]
    ordered_ids = [p.sample_id for p in preds_b_ord]
    align_audit = paired["alignment"]
    statistics = paired["statistics"]
    biz_b_full = paired["baseline_business"]
    biz_c_full = paired["candidate_business"]

    target_spec = next(t for t in task.targets if t.name == target)
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
        "transitions": paired["transitions"],
        "metric_deltas": paired["metric_deltas"],
        "statistics": statistics,
        "slices": slice_comparison,
        "baseline_business": paired["baseline_business_view"],
        "candidate_business": paired["candidate_business_view"],
    }

    baseline_pred_path = _resolve(
        root,
        (baseline_cfg.get("source") or baseline_cfg).get("path")
        or (baseline_cfg.get("source") or baseline_cfg).get("predictions"),
    )
    candidate_pred_path = _resolve(
        root,
        (candidate_cfg.get("source") or candidate_cfg).get("path")
        or (candidate_cfg.get("source") or candidate_cfg).get("predictions"),
    )

    comparability_cfg = dict(cfg.get("comparability") or {})
    comparability = evaluate_comparability(
        comparability_cfg,
        baseline_side=dict(comparability_cfg.get("baseline") or {}),
        candidate_side=dict(comparability_cfg.get("candidate") or {}),
    )

    dataset_fingerprint = fingerprint_records([s.to_dict() for s in samples_aligned])
    protocol_audit = validate_comparison_protocol(
        dict(cfg.get("comparison_protocol") or {}),
        baseline_path=str(baseline_pred_path) if baseline_pred_path else None,
        candidate_path=str(candidate_pred_path) if candidate_pred_path else None,
        dataset_fingerprint=dataset_fingerprint,
        task_spec_hash=sha256_file(task_path) if task_path else None,
        output_spec_hash=sha256_file(output_path) if output_path else None,
        metric_spec_hash=sha256_file(metric_path) if metric_path else None,
        n_aligned=len(ordered_ids),
        comparability=comparability,
        require_semantic_comparable=bool(
            (cfg.get("comparison_protocol") or {}).get("require_semantic_comparable", True)
        )
        if cfg.get("comparison_protocol")
        else False,
    )

    comparison_metrics["comparability"] = comparability
    comparison_metrics["comparison_protocol"] = protocol_audit
    comparison_metrics["support"] = {
        "n_samples": len(pair_rows),
        "n_units": (statistics or {}).get("n_units") or len(pair_rows),
        "n_aligned": len(ordered_ids),
        "bootstrap_unit": bootstrap_unit,
        "cluster_mode": bool((statistics or {}).get("cluster_mode")),
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
    provenance["baseline_prediction_path"] = str(baseline_pred_path or "")
    provenance["candidate_prediction_path"] = str(candidate_pred_path or "")
    provenance["comparability"] = comparability
    provenance["comparison_protocol"] = {
        "protocol_id": protocol_audit.get("protocol_id"),
        "allowed_pair_matched": protocol_audit.get("allowed_pair_matched"),
    }

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
            "transitions": paired["transitions"],
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
    protocol_path = out_dir / "comparison_protocol.json"
    write_json(protocol_path, protocol_audit)
    write_compare_report_md(
        report_path, metrics=comparison_metrics, manifest=manifest.to_dict()
    )

    manifest.artifact_index = {
        "comparison_metrics": str(metrics_path),
        "comparison_records": str(records_path),
        "alignment_audit": str(audit_path),
        "report": str(report_path),
        "comparison_protocol": str(protocol_path),
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
    write_json(out_dir / "baseline_business_metrics.json", biz_b_full)
    write_json(out_dir / "candidate_business_metrics.json", biz_c_full)
    return out_dir
