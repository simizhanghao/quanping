"""Offline perturbation generation (P2-B) — variants only; no model inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.jsonl_samples import load_samples_jsonl
from linguaeval.core.fingerprint import build_provenance
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import RunManifest
from linguaeval.robustness.generate import coverage_audit, generate_variants, variant_fingerprint
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


def _perturbation_ids(cfg: Dict[str, Any]) -> List[str]:
    raw = cfg.get("perturbations") or cfg.get("perturbation_ids") or []
    ids: List[str] = []
    for item in raw:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    if not ids:
        ids = ["case_lower", "strip_punctuation", "collapse_whitespace"]
    return ids


def run_offline_perturb(config_path: Path) -> Path:
    ensure_builtin_perturbation_specs()
    cfg = _load_yaml(config_path)
    root = config_path.parent

    source = dict(cfg.get("source") or {})
    samples_path = _resolve(root, source.get("samples") or cfg.get("samples"))
    if not samples_path or not samples_path.is_file():
        raise FileNotFoundError(f"samples not found: {samples_path}")
    samples = load_samples_jsonl(samples_path)

    pids = _perturbation_ids(cfg)
    seed = int(cfg.get("seed") or 42)
    validity = str(cfg.get("semantic_validity") or "AUTO_VALIDATED")
    variants = generate_variants(
        samples,
        pids,
        seed=seed,
        semantic_validity=validity,
    )
    fp = variant_fingerprint(variants)
    audit = coverage_audit(samples, variants, requested_perturbations=pids)
    audit["variant_fingerprint"] = fp
    audit["seed"] = seed
    audit["perturbation_ids"] = pids
    audit["registry"] = list_perturbations()

    out_dir = _resolve_out_dir(config_path, cfg.get("output_dir") or "results/16_perturb")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants_path = out_dir / "variants.jsonl"
    with variants_path.open("w", encoding="utf-8") as f:
        for v in variants:
            f.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")

    manifest_path = out_dir / "variant_manifest.json"
    write_json(manifest_path, audit)

    report = "\n".join(
        [
            "# LinguaEval Perturb (P2-B)",
            "",
            f"- n_parents: {audit['n_parents']}",
            f"- perturbations: `{pids}`",
            f"- n_generated: {audit['n_generated']}",
            f"- n_valid: {audit['n_valid']}",
            f"- variant_fingerprint: `{fp}`",
            f"- seed: {seed}",
            "",
            "Next: run model on variants.jsonl → variant_predictions.jsonl, then robustness-offline.",
            "",
        ]
    )
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    provenance = build_provenance(
        config_path=config_path,
        cfg=cfg,
        task_path=None,
        output_path=None,
        metric_path=None,
        sample_dicts=[s.to_dict() for s in samples],
        prediction_dicts=[],
    )
    run_id = cfg.get("run_id") or f"perturb_{uuid4().hex[:8]}"
    write_manifest(
        out_dir / "manifest.json",
        RunManifest(
            run_id=run_id,
            config_path=str(config_path.resolve()),
            packs=list(cfg.get("packs") or ["perturb"]),
            provenance=provenance,
            notes={
                "mode": "offline_perturb",
                "variant_fingerprint": fp,
                "n_generated": audit["n_generated"],
            },
            artifact_index={
                "variants": str(variants_path),
                "variant_manifest": str(manifest_path),
                "report": str(report_path),
            },
        ),
    )
    return out_dir
