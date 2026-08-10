"""Offline baseline↔candidate robustness compare (P2-D)."""

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
    MetricSpec,
    OutputSpec,
    RunManifest,
    TaskSpec,
    VariantRecord,
)
from linguaeval.parse.pipeline import apply_output_spec
from linguaeval.robustness.aggregate import aggregate_robustness, build_robustness_records
from linguaeval.robustness.compare import (
    VariantFingerprintError,
    compare_robustness_metrics,
    pair_robustness_records,
    require_shared_variant_fingerprint,
    summarize_compare,
)
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


def _read_side_fingerprint(manifest_path: Optional[Path]) -> Optional[str]:
    if not manifest_path or not manifest_path.is_file():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data.get("variant_fingerprint") or (data.get("notes") or {}).get("variant_fingerprint")


def run_offline_robustness_compare(config_path: Path) -> Path:
    ensure_builtin_perturbation_specs()
    cfg = _load_yaml(config_path)
    root = config_path.parent

    task_path = _resolve(root, cfg.get("task_spec") or cfg.get("task"))
    if not task_path or not task_path.is_file():
        raise FileNotFoundError(f"task_spec not found: {task_path}")
    task = TaskSpec.from_dict(_load_yaml(task_path))

    metric_path = _resolve(root, cfg.get("metric_spec") or cfg.get("metrics"))
    if metric_path and metric_path.is_file():
        metric_spec = MetricSpec.from_dict(_load_yaml(metric_path))
    else:
        default_metrics = {
            t.name: (["accuracy", "macro_f1"] if t.type == "multiclass" else ["accuracy"])
            for t in task.targets
        }
        metric_spec = MetricSpec(metrics=default_metrics, round_digits=4)

    rel_cfg = dict(cfg.get("relation") or cfg.get("metamorphic") or {})
    if not rel_cfg.get("targets"):
        rel_cfg["targets"] = [t.name for t in task.targets]
    relation = MetamorphicRelationSpec.from_dict(rel_cfg)

    shared = dict(cfg.get("source") or {})
    samples_path = _resolve(root, shared.get("samples"))
    variants_path = _resolve(root, shared.get("variants"))
    if not samples_path or not samples_path.is_file():
        raise FileNotFoundError(f"samples not found: {samples_path}")
    if not variants_path or not variants_path.is_file():
        raise FileNotFoundError(f"variants not found: {variants_path}")

    baseline_cfg = dict(cfg.get("baseline") or {})
    candidate_cfg = dict(cfg.get("candidate") or {})
    if not baseline_cfg or not candidate_cfg:
        raise ValueError("config must define baseline: and candidate: blocks")

    def _side_paths(side: Dict[str, Any]) -> tuple:
        src = dict(side.get("source") or side)
        clean = _resolve(root, src.get("predictions") or src.get("clean_predictions"))
        vpred = _resolve(root, src.get("variant_predictions") or src.get("perturbed_predictions"))
        man = _resolve(root, src.get("variant_manifest") or src.get("manifest"))
        return clean, vpred, man

    b_clean_p, b_var_p, b_man_p = _side_paths(baseline_cfg)
    c_clean_p, c_var_p, c_man_p = _side_paths(candidate_cfg)
    for label, p in (
        ("baseline.predictions", b_clean_p),
        ("baseline.variant_predictions", b_var_p),
        ("candidate.predictions", c_clean_p),
        ("candidate.variant_predictions", c_var_p),
    ):
        if not p or not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    samples = load_samples_jsonl(samples_path)
    variants = _load_variants(variants_path)
    expected_fp = shared.get("variant_fingerprint") or cfg.get("variant_fingerprint")
    fp = require_shared_variant_fingerprint(
        variants=variants,
        expected_fingerprint=str(expected_fp) if expected_fp else None,
        baseline_fingerprint=_read_side_fingerprint(b_man_p),
        candidate_fingerprint=_read_side_fingerprint(c_man_p),
    )

    output_path = _resolve(root, cfg.get("output_spec") or cfg.get("output"))
    output_spec = OutputSpec.from_dict(
        _load_yaml(output_path) if output_path and output_path.is_file() else {}
    )
    parse_mode = (cfg.get("parse") or {}).get("mode") or "from_parsed"

    def _load_preds(path: Path):
        return apply_output_spec(load_predictions_jsonl(path), output_spec, mode=parse_mode)

    b_clean = _load_preds(b_clean_p)
    b_vpred = _load_preds(b_var_p)
    c_clean = _load_preds(c_clean_p)
    c_vpred = _load_preds(c_var_p)

    b_records = build_robustness_records(
        samples=samples,
        clean_preds=b_clean,
        variants=variants,
        variant_preds=b_vpred,
        task=task,
        relation=relation,
    )
    c_records = build_robustness_records(
        samples=samples,
        clean_preds=c_clean,
        variants=variants,
        variant_preds=c_vpred,
        task=task,
        relation=relation,
    )

    boot = dict(cfg.get("bootstrap") or {})
    b_metrics = aggregate_robustness(
        b_records,
        samples=samples,
        clean_preds=b_clean,
        variants=variants,
        variant_preds=b_vpred,
        task=task,
        metric_spec=metric_spec,
        bootstrap=boot,
    )
    c_metrics = aggregate_robustness(
        c_records,
        samples=samples,
        clean_preds=c_clean,
        variants=variants,
        variant_preds=c_vpred,
        task=task,
        metric_spec=metric_spec,
        bootstrap=boot,
    )
    b_metrics["variant_fingerprint"] = fp
    c_metrics["variant_fingerprint"] = fp
    b_metrics["relation"] = {"type": relation.type, "targets": relation.targets}
    c_metrics["relation"] = {"type": relation.type, "targets": relation.targets}

    paired_rows, trans_counts, align_audit = pair_robustness_records(b_records, c_records)
    metric_cmp = compare_robustness_metrics(b_metrics, c_metrics)
    summary = summarize_compare(
        fingerprint=fp,
        metric_compare=metric_cmp,
        transition_counts=trans_counts,
        alignment_audit=align_audit,
    )
    summary["relation"] = {"type": relation.type, "targets": relation.targets}
    summary["registry_perturbations"] = list_perturbations()
    summary["baseline_metrics"] = b_metrics
    summary["candidate_metrics"] = c_metrics

    out_dir = _resolve_out_dir(config_path, cfg.get("output_dir") or "results/18_robustness_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "baseline_robustness_metrics.json", b_metrics)
    write_json(out_dir / "candidate_robustness_metrics.json", c_metrics)
    write_json(out_dir / "robustness_compare_metrics.json", summary)

    paired_path = out_dir / "robustness_compare_records.jsonl"
    with paired_path.open("w", encoding="utf-8") as f:
        for row in paired_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for name, wanted in (
        ("robustness_gain_cases.jsonl", "robustness_gain"),
        ("robustness_regression_cases.jsonl", "robustness_regression"),
        ("both_fragile_cases.jsonl", "both_fragile"),
        ("stable_robust_cases.jsonl", "stable_robust"),
    ):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for row in paired_rows:
                if row.get("transition") == wanted:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# LinguaEval Robustness Compare (P2-D)",
        "",
        f"- status: `{summary.get('status')}`",
        f"- variant_fingerprint: `{fp}`",
        f"- relation: `{relation.type}` targets=`{relation.targets}`",
        f"- transitions: `{trans_counts}`",
        f"- n_transition_eligible: `{summary.get('n_transition_eligible')}`",
        "",
    ]
    for tname, block in (metric_cmp.get("by_target") or {}).items():
        d = block.get("delta") or {}
        lines += [
            f"## Target `{tname}` (candidate − baseline)",
            "",
            f"- Δ flip_rate: `{d.get('flip_rate')}`",
            f"- Δ metamorphic_violation_rate: `{d.get('metamorphic_violation_rate')}`",
            f"- Δ end_to_end_robust_success_rate: `{d.get('end_to_end_robust_success_rate')}`",
            f"- baseline flip_rate: `{(block.get('baseline') or {}).get('flip_rate')}`",
            f"- candidate flip_rate: `{(block.get('candidate') or {}).get('flip_rate')}`",
            "",
        ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    provenance = build_provenance(
        config_path=config_path,
        cfg=cfg,
        task_path=task_path,
        output_path=output_path,
        metric_path=metric_path,
        sample_dicts=[s.to_dict() for s in samples],
        prediction_dicts=[p.to_dict() for p in b_clean],
    )
    run_id = cfg.get("run_id") or f"robustness_compare_{uuid4().hex[:8]}"
    write_manifest(
        out_dir / "manifest.json",
        RunManifest(
            run_id=run_id,
            config_path=str(config_path.resolve()),
            packs=list(cfg.get("packs") or ["robustness_compare"]),
            provenance=provenance,
            notes={
                "mode": "offline_robustness_compare",
                "variant_fingerprint": fp,
                "transitions": trans_counts,
            },
            artifact_index={
                "robustness_compare_metrics": str(out_dir / "robustness_compare_metrics.json"),
                "robustness_compare_records": str(paired_path),
                "report": str(report_path),
            },
        ),
    )
    return out_dir
