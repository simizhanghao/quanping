"""Multi-language / multi-capability score + Base↔SFT matrix (P3-B/C).

Reuses TaskScorer / MetricSpec / ScoreRecord — does not invent a second score world.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.registry import get_adapter
from linguaeval.compare.gates import evaluate_gates
from linguaeval.core.fingerprint import build_provenance
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import MetricSpec, OutputSpec, RunManifest, TaskSpec
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


def _primary_accuracy(business: Dict[str, Any], target: str) -> Optional[float]:
    block = ((business.get("targets") or {}).get(target) or {})
    v = block.get("accuracy")
    return float(v) if isinstance(v, (int, float)) else None


def _score_side(
    *,
    source: Dict[str, Any],
    config_path: Path,
    cfg: Dict[str, Any],
    task: TaskSpec,
    output_spec: OutputSpec,
    metric_spec: MetricSpec,
    parse_mode: str,
    default_adapter: str,
) -> Dict[str, Any]:
    adapter_name = source.get("adapter") or default_adapter
    adapter = get_adapter(str(adapter_name))
    samples, preds = adapter(source, config_path.parent, cfg)
    preds = apply_output_spec(preds, output_spec, mode=parse_mode)
    prov = None
    if samples and isinstance(samples[0].meta, dict):
        prov = samples[0].meta.get("provenance")
    if not preds:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "missing_predictions",
            "n_samples": len(samples),
            "provenance": prov,
        }
    scored = score_targets(samples, preds, task, metric_spec)
    business = build_business_metrics(scored, report_cfg=cfg.get("report") or {})
    score_rows = build_score_records(samples, preds, task)
    target = str((cfg.get("report") or {}).get("primary_target") or task.targets[0].name)
    return {
        "status": "AVAILABLE",
        "n_samples": len(samples),
        "n_predictions": len(preds),
        "n_score_records": len(score_rows),
        "accuracy": _primary_accuracy(business, target),
        "business": business,
        "target": target,
        "language": source.get("language"),
        "benchmark_id": source.get("benchmark_id"),
        "provenance": prov,
    }


def _eval_languages(
    *,
    languages_cfg: Dict[str, Any],
    capability: str,
    config_path: Path,
    cfg: Dict[str, Any],
    task: TaskSpec,
    output_spec: OutputSpec,
    metric_spec: MetricSpec,
    parse_mode: str,
    default_adapter: str,
    default_benchmark_prefix: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metrics_by_lang: Dict[str, Any] = {}
    reg_by_lang: Dict[str, Any] = {}
    for lang, block in languages_cfg.items():
        block = dict(block or {})
        shared = dict(block.get("source") or {})
        shared.setdefault("adapter", block.get("adapter") or default_adapter)
        shared.setdefault("language", lang)
        shared.setdefault("capability", capability)
        shared.setdefault(
            "benchmark_id",
            block.get("benchmark_id")
            or cfg.get("benchmark_id")
            or f"{default_benchmark_prefix}_{lang}",
        )
        single_src = dict(shared)
        if block.get("predictions"):
            single_src["predictions"] = block["predictions"]
        if block.get("samples"):
            single_src["samples"] = block["samples"]
        elif not shared.get("samples"):
            raise ValueError(f"languages.{lang}.samples (or source.samples) is required")

        single = None
        if single_src.get("predictions"):
            single = _score_side(
                source=single_src,
                config_path=config_path,
                cfg=cfg,
                task=task,
                output_spec=output_spec,
                metric_spec=metric_spec,
                parse_mode=parse_mode,
                default_adapter=default_adapter,
            )

        baseline_cfg = dict(block.get("baseline") or {})
        candidate_cfg = dict(block.get("candidate") or {})
        reg_block: Dict[str, Any] = {"status": "NOT_AVAILABLE", "reason": "no_baseline_candidate"}
        if baseline_cfg and candidate_cfg:
            b_src = {**shared, **dict(baseline_cfg.get("source") or baseline_cfg)}
            c_src = {**shared, **dict(candidate_cfg.get("source") or candidate_cfg)}
            b_src.setdefault("language", lang)
            c_src.setdefault("language", lang)
            b_src.setdefault("samples", single_src.get("samples"))
            c_src.setdefault("samples", single_src.get("samples"))
            b_res = _score_side(
                source=b_src,
                config_path=config_path,
                cfg=cfg,
                task=task,
                output_spec=output_spec,
                metric_spec=metric_spec,
                parse_mode=parse_mode,
                default_adapter=default_adapter,
            )
            c_res = _score_side(
                source=c_src,
                config_path=config_path,
                cfg=cfg,
                task=task,
                output_spec=output_spec,
                metric_spec=metric_spec,
                parse_mode=parse_mode,
                default_adapter=default_adapter,
            )
            b_acc, c_acc = b_res.get("accuracy"), c_res.get("accuracy")
            delta = (
                float(c_acc) - float(b_acc)
                if isinstance(b_acc, (int, float)) and isinstance(c_acc, (int, float))
                else None
            )
            reg_block = {
                "status": "AVAILABLE",
                "baseline_accuracy": b_acc,
                "candidate_accuracy": c_acc,
                "delta_accuracy": delta,
                "baseline": {k: b_res[k] for k in ("status", "n_samples", "accuracy", "benchmark_id")},
                "candidate": {k: c_res[k] for k in ("status", "n_samples", "accuracy", "benchmark_id")},
            }
            if single is None:
                single = c_res

        prov = (single or {}).get("provenance") or {
            "origin": "unknown",
            "translation": "unknown",
        }
        metrics_by_lang[lang] = {
            "capability": capability,
            "benchmark_id": shared.get("benchmark_id"),
            "status": (single or {}).get("status") or "NOT_AVAILABLE",
            "accuracy": (single or {}).get("accuracy"),
            "n_samples": (single or {}).get("n_samples"),
            "provenance": prov,
            "native_authored": bool((prov or {}).get("native_authored")),
        }
        reg_by_lang[lang] = reg_block
    return metrics_by_lang, reg_by_lang


def run_offline_language_matrix(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    root = config_path.parent

    task_path = _resolve(root, cfg.get("task_spec") or cfg.get("task"))
    metric_path = _resolve(root, cfg.get("metric_spec") or cfg.get("metrics"))
    output_path = _resolve(root, cfg.get("output_spec") or cfg.get("output"))
    if not task_path or not task_path.is_file():
        raise FileNotFoundError(f"task_spec not found: {task_path}")
    if not metric_path or not metric_path.is_file():
        raise FileNotFoundError(f"metric_spec not found: {metric_path}")

    task = TaskSpec.from_dict(_load_yaml(task_path))
    metric_spec = MetricSpec.from_dict(_load_yaml(metric_path))
    output_spec = OutputSpec.from_dict(
        _load_yaml(output_path) if output_path and output_path.is_file() else {}
    )
    parse_mode = (cfg.get("parse") or {}).get("mode") or "from_parsed"
    default_adapter = str(cfg.get("adapter") or "belebele_jsonl")

    # P3-C: capabilities: {cap: {languages: {...}, adapter?: ...}}
    # P3-B: languages: {...} + capability: ...
    caps_cfg = dict(cfg.get("capabilities") or {})
    if not caps_cfg:
        languages_cfg = dict(cfg.get("languages") or {})
        if not languages_cfg:
            raise ValueError("languages: or capabilities: map is required")
        caps_cfg = {
            str(cfg.get("capability") or "reading_comprehension"): {
                "languages": languages_cfg,
                "adapter": default_adapter,
            }
        }

    by_capability_metrics: Dict[str, Any] = {}
    by_capability_reg: Dict[str, Any] = {}
    for cap, cap_block in caps_cfg.items():
        cap_block = dict(cap_block or {})
        languages_cfg = dict(cap_block.get("languages") or {})
        if not languages_cfg:
            raise ValueError(f"capabilities.{cap}.languages is required")
        adapter = str(cap_block.get("adapter") or default_adapter)
        prefix = str(cap_block.get("benchmark_prefix") or cap)
        m_lang, r_lang = _eval_languages(
            languages_cfg=languages_cfg,
            capability=str(cap),
            config_path=config_path,
            cfg=cfg,
            task=task,
            output_spec=output_spec,
            metric_spec=metric_spec,
            parse_mode=parse_mode,
            default_adapter=adapter,
            default_benchmark_prefix=prefix,
        )
        by_capability_metrics[str(cap)] = {"by_language": m_lang}
        by_capability_reg[str(cap)] = {"by_language": r_lang}

    # Backward-compatible flat view when exactly one capability
    flat_cap = next(iter(by_capability_metrics)) if len(by_capability_metrics) == 1 else None
    language_metrics: Dict[str, Any] = {
        "status": "AVAILABLE",
        "task": task.name,
        "by_capability": by_capability_metrics,
    }
    language_regression: Dict[str, Any] = {
        "status": "AVAILABLE",
        "by_capability": by_capability_reg,
    }
    if flat_cap:
        language_metrics["capability"] = flat_cap
        language_metrics["by_language"] = by_capability_metrics[flat_cap]["by_language"]
        language_regression["capability"] = flat_cap
        language_regression["by_language"] = by_capability_reg[flat_cap]["by_language"]

    # P3-E: flat capability × language rows (no multilingual total score)
    report_rows: List[Dict[str, Any]] = []
    for cap, block in by_capability_reg.items():
        for lang, row in (block.get("by_language") or {}).items():
            mrow = (by_capability_metrics.get(cap) or {}).get("by_language", {}).get(lang) or {}
            report_rows.append(
                {
                    "language": lang,
                    "capability": cap,
                    "benchmark_id": mrow.get("benchmark_id"),
                    "native_authored": mrow.get("native_authored"),
                    "baseline_accuracy": row.get("baseline_accuracy"),
                    "candidate_accuracy": row.get("candidate_accuracy"),
                    "delta_accuracy": row.get("delta_accuracy"),
                    "status": row.get("status"),
                    "n_samples": mrow.get("n_samples"),
                }
            )

    capability_report: Dict[str, Any] = {
        "status": "AVAILABLE",
        "task": task.name,
        "rows": report_rows,
        "note": "No multilingual total score — compare by language × capability only.",
    }

    gate_context = {
        "language_metrics": language_metrics,
        "language_regression": language_regression,
        "capability_report": capability_report,
        "support": {"n_samples": sum(int(r.get("n_samples") or 0) for r in report_rows)},
    }
    gate_specs = list(cfg.get("gates") or [])
    gates = evaluate_gates(gate_context, gate_specs) if gate_specs else {
        "status": "NOT_APPLICABLE",
        "reason": "no_gates_configured",
        "gates": [],
    }
    capability_report["gates"] = gates

    out_dir = _resolve_out_dir(config_path, cfg.get("output_dir") or "results/22_language_matrix")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "language_metrics.json", language_metrics)
    write_json(out_dir / "language_regression.json", language_regression)
    write_json(out_dir / "language_capability_report.json", capability_report)
    write_json(out_dir / "gate.json", gates)

    lines = [
        "# LinguaEval Language Capability Report (P3-E)",
        "",
        f"- task: `{task.name}`",
        f"- capabilities: `{list(by_capability_metrics)}`",
        f"- gates: `{gates.get('status')}`",
        "",
        "## Matrix (candidate − baseline)",
        "",
        "| Language | Capability | Base | Candidate | Δ | Native |",
        "|----------|------------|-----:|----------:|--:|:------:|",
    ]
    for r in report_rows:
        lines.append(
            f"| `{r.get('language')}` | `{r.get('capability')}` | "
            f"{r.get('baseline_accuracy')} | {r.get('candidate_accuracy')} | "
            f"{r.get('delta_accuracy')} | {r.get('native_authored')} |"
        )
    lines += ["", "## Gate details", ""]
    for g in gates.get("gates") or []:
        lines.append(
            f"- `{g.get('id')}`: status=`{g.get('status')}` "
            f"path=`{g.get('path')}` observed=`{g.get('observed')}`"
        )
    if not gate_specs:
        lines.append("- (no gates configured)")
    lines.append("")
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    provenance = build_provenance(
        config_path=config_path,
        cfg=cfg,
        task_path=task_path,
        output_path=output_path,
        metric_path=metric_path,
        sample_dicts=[],
        prediction_dicts=[],
    )
    write_manifest(
        out_dir / "manifest.json",
        RunManifest(
            run_id=cfg.get("run_id") or f"language_matrix_{uuid4().hex[:8]}",
            config_path=str(config_path.resolve()),
            packs=list(cfg.get("packs") or ["language"]),
            provenance=provenance,
            notes={
                "mode": "offline_language_matrix",
                "capabilities": list(by_capability_metrics),
                "gate_status": gates.get("status"),
            },
            artifact_index={
                "language_metrics": str(out_dir / "language_metrics.json"),
                "language_regression": str(out_dir / "language_regression.json"),
                "language_capability_report": str(out_dir / "language_capability_report.json"),
                "gate": str(out_dir / "gate.json"),
                "report": str(report_path),
            },
        ),
    )
    return out_dir
