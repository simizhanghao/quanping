"""Offline perturbation generation (P2-B/C0) — variants only; no model inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from linguaeval.adapters.dataset.jsonl_samples import load_samples_jsonl
from linguaeval.core.fingerprint import build_provenance
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import PerturbationSpec, RunManifest
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


def _perturbation_specs(cfg: Dict[str, Any]) -> List[PerturbationSpec]:
    raw = cfg.get("perturbations") or cfg.get("perturbation_ids") or []
    specs: List[PerturbationSpec] = []
    for item in raw:
        if isinstance(item, str):
            specs.append(PerturbationSpec(id=item))
        elif isinstance(item, dict):
            specs.append(PerturbationSpec.from_dict(item))
    if not specs:
        specs = [
            PerturbationSpec(id="case_lower"),
            PerturbationSpec(id="strip_punctuation"),
            PerturbationSpec(id="collapse_whitespace"),
        ]
    return specs


def run_offline_perturb(config_path: Path) -> Path:
    ensure_builtin_perturbation_specs()
    cfg = _load_yaml(config_path)
    root = config_path.parent

    source = dict(cfg.get("source") or {})
    samples_path = _resolve(root, source.get("samples") or cfg.get("samples"))
    if not samples_path or not samples_path.is_file():
        raise FileNotFoundError(f"samples not found: {samples_path}")
    samples = load_samples_jsonl(samples_path)

    specs = _perturbation_specs(cfg)
    # resolve lexicon_path etc. relative to config directory
    for i, sp in enumerate(specs):
        params = dict(sp.params or {})
        lp = params.get("lexicon_path")
        if lp:
            resolved = _resolve(root, str(lp))
            if resolved is not None:
                params["lexicon_path"] = str(resolved)
        specs[i] = PerturbationSpec(
            id=sp.id,
            category=sp.category,
            severity=sp.severity,
            seed=sp.seed,
            semantic_policy=sp.semantic_policy,
            transform_version=sp.transform_version,
            params=params,
            applies_to=sp.applies_to,
        )
    seed = int(cfg.get("seed") or 42)
    # Only used when transform changes input; NO-OP / N/A set per-variant.
    validity_if_changed = str(cfg.get("semantic_validity_if_changed") or "AUTO_VALIDATED")
    variants = generate_variants(
        samples,
        specs,
        seed=seed,
        semantic_validity_if_changed=validity_if_changed,
    )
    fp = variant_fingerprint(variants)
    pids = [s.id for s in specs]
    audit = coverage_audit(samples, variants, requested_perturbations=pids)
    audit["variant_fingerprint"] = fp
    audit["seed"] = seed
    audit["perturbation_specs"] = [s.to_dict() for s in specs]
    audit["registry"] = list_perturbations()

    out_dir = _resolve_out_dir(config_path, cfg.get("output_dir") or "results/16_perturb")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants_path = out_dir / "variants.jsonl"
    with variants_path.open("w", encoding="utf-8") as f:
        for v in variants:
            f.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")

    write_json(out_dir / "variant_manifest.json", audit)

    report = "\n".join(
        [
            "# LinguaEval Perturb (P2-B/C0)",
            "",
            f"- n_parents: {audit['n_parents']}",
            f"- perturbations: `{pids}`",
            f"- n_generated: {audit['n_generated']}",
            f"- n_valid: {audit['n_valid']}",
            f"- n_noop: {audit['n_noop']}",
            f"- n_not_applicable: {audit['n_not_applicable']}",
            f"- variant_fingerprint: `{fp}`",
            f"- seed: {seed}",
            "",
            "Next: run model on variants.jsonl → variant_predictions.jsonl, then robustness-offline.",
            "",
        ]
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")

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
                "n_valid": audit["n_valid"],
                "n_noop": audit["n_noop"],
            },
            artifact_index={
                "variants": str(variants_path),
                "variant_manifest": str(out_dir / "variant_manifest.json"),
                "report": str(out_dir / "report.md"),
            },
        ),
    )
    return out_dir
