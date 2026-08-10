"""Offline operating-point selection runner (P1.5-C)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.registry import get_adapter
from linguaeval.confidence.extract import extract_confidence_records, summarize_confidence
from linguaeval.confidence.operating_point import OperatingPointError, select_operating_point
from linguaeval.core.fingerprint import build_provenance
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import ConfidenceSpec, OperatingPointSpec, OutputSpec, RunManifest, TaskSpec
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


def run_offline_operating_point(config_path: Path) -> Path:
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

    op_path = _resolve(root, cfg.get("operating_point_spec"))
    if op_path and op_path.is_file():
        op_cfg = _load_yaml(op_path)
    else:
        op_cfg = dict(cfg.get("operating_point") or {})
    op_spec = OperatingPointSpec.from_dict(op_cfg)

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

    try:
        result = select_operating_point(records, samples, op_spec)
        leakage_exc: Optional[OperatingPointError] = None
    except OperatingPointError as e:
        leakage_exc = e
        result = {
            "status": e.reason,
            "reason": e.reason,
            "detail": str(e),
            "target": op_spec.target,
            "positive_class": op_spec.positive_class,
            "optimize_on": op_spec.optimize_on,
            "evaluate_on": op_spec.evaluate_on,
            "mode": op_spec.mode,
            "selected": None,
            "test_evaluation": None,
            "threshold_curve": [],
        }

    out_dir = _resolve_out_dir(
        config_path, cfg.get("output_dir") or "results/09_operating_point"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "confidence_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    audit_path = out_dir / "confidence_audit.json"
    write_json(audit_path, audit)

    # curve separate; operating_points without full curve dump duplication optional
    curve = result.get("threshold_curve") or []
    curve_path = out_dir / "threshold_curve.json"
    write_json(curve_path, {"n": len(curve), "points": curve})

    op_slim = {k: v for k, v in result.items() if k != "threshold_curve"}
    op_path_out = out_dir / "operating_points.json"
    write_json(op_path_out, op_slim)

    sel = result.get("selected") or {}
    te = result.get("test_evaluation") or {}
    lines = [
        f"# LinguaEval Operating Point — `{op_spec.target}` / `{op_spec.positive_class}`",
        "",
        f"- status: `{result.get('status')}`",
        f"- mode: `{op_spec.mode}`",
        f"- optimize_on: `{op_spec.optimize_on}`",
        f"- evaluate_on: `{op_spec.evaluate_on}`",
        f"- confidence AVAILABLE: {audit['counts'].get('AVAILABLE', 0)} / {audit['n_records']}",
        "",
    ]
    if result.get("status") == "TEST_LEAKAGE":
        lines += ["**FAIL: test_leakage** — threshold must not be optimized on test.", ""]
    elif result.get("status") == "NO_FEASIBLE_OPERATING_POINT":
        lines += ["**NO_FEASIBLE_OPERATING_POINT** — no threshold meets constraints.", ""]
    elif result.get("status") == "NOT_AVAILABLE":
        lines += [
            f"**NOT_AVAILABLE** — {result.get('reason')}",
            "",
        ]
    elif sel:
        lines += [
            "## Selected (optimize split)",
            "",
            f"- threshold: `{sel.get('threshold')}`",
            f"- precision: `{sel.get('precision')}`",
            f"- recall: `{sel.get('recall')}`",
            f"- f1: `{sel.get('f1')}`",
            "",
            "## Frozen evaluation (evaluate split)",
            "",
        ]
        if isinstance(te, dict) and "precision" in te:
            lines += [
                f"- precision: `{te.get('precision')}`",
                f"- recall: `{te.get('recall')}`",
                f"- f1: `{te.get('f1')}`",
                "",
            ]
        else:
            lines += [f"- {te}", ""]

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
    run_id = cfg.get("run_id") or f"operating_point_{uuid4().hex[:8]}"
    manifest = RunManifest(
        run_id=run_id,
        config_path=str(config_path.resolve()),
        packs=list(cfg.get("packs") or ["operating_point"]),
        provenance=provenance,
        notes={
            "mode": "offline_operating_point",
            "status": result.get("status"),
            "selected_threshold": (sel or {}).get("threshold"),
        },
        artifact_index={
            "confidence_records": str(records_path),
            "confidence_audit": str(audit_path),
            "threshold_curve": str(curve_path),
            "operating_points": str(op_path_out),
            "report": str(report_path),
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    if leakage_exc is not None:
        raise leakage_exc
    return out_dir
