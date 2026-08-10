"""Load Language / Benchmark / Pack YAML into registries; write availability audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from linguaeval.core.fingerprint import build_provenance
from linguaeval.core.manifest import write_json, write_manifest
from linguaeval.core.schema import RunManifest
from linguaeval.language.registry import (
    LanguageRegistryError,
    clear_registries,
    list_benchmarks,
    list_languages,
    list_packs,
    register_benchmark,
    register_language,
    register_pack,
    resolve_pack_availability,
)
from linguaeval.language.spec import BenchmarkSpec, LanguagePackSpec, LanguageSpec


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


def _load_many(dir_or_files: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in dir_or_files:
        if p.is_dir():
            for f in sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")):
                data = _load_yaml(f)
                if isinstance(data, list):
                    rows.extend(data)
                else:
                    rows.append(data)
        elif p.is_file():
            data = _load_yaml(p)
            if isinstance(data, list):
                rows.extend(data)
            else:
                rows.append(data)
    return rows


def _paths_from_cfg(root: Path, cfg: Dict[str, Any], key: str) -> List[Path]:
    raw = cfg.get(key)
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    out: List[Path] = []
    for item in raw:
        p = _resolve(root, str(item))
        if p is not None:
            out.append(p)
    return out


def load_language_ecosystem(config_path: Path, *, reset: bool = True) -> Dict[str, Any]:
    """Register languages/benchmarks/packs from YAML paths in config."""
    if reset:
        clear_registries()
    cfg = _load_yaml(config_path)
    root = config_path.parent

    for row in _load_many(_paths_from_cfg(root, cfg, "languages")):
        # allow {languages: [...]} wrapper
        if "iso639_3" in row:
            register_language(LanguageSpec.from_dict(row))
        elif "languages" in row:
            for item in row["languages"]:
                register_language(LanguageSpec.from_dict(item))

    for row in _load_many(_paths_from_cfg(root, cfg, "benchmarks")):
        if "id" in row and "capability" in row:
            register_benchmark(BenchmarkSpec.from_dict(row))
        elif "benchmarks" in row:
            for item in row["benchmarks"]:
                register_benchmark(BenchmarkSpec.from_dict(item))

    pack_paths = _paths_from_cfg(root, cfg, "packs")
    # also allow packs: [{id: ...}] inline? prefer files
    for row in _load_many(pack_paths):
        if "id" in row and "language" in row:
            register_pack(LanguagePackSpec.from_dict(row.get("pack") or row))
        elif "pack" in row:
            register_pack(LanguagePackSpec.from_dict(row["pack"]))

    return cfg


def run_offline_language_inspect(config_path: Path) -> Path:
    cfg = load_language_ecosystem(config_path, reset=True)
    root = config_path.parent

    pack_ids = list(cfg.get("inspect_packs") or list_packs())
    resolved = []
    errors = []
    for pid in pack_ids:
        try:
            resolved.append(resolve_pack_availability(str(pid)))
        except LanguageRegistryError as e:
            errors.append({"pack_id": pid, "status": "NOT_AVAILABLE", "reason": e.reason, "message": str(e)})

    unknown = cfg.get("probe_unknown_language")
    unknown_probe = None
    if unknown:
        from linguaeval.language.registry import get_language

        try:
            get_language(str(unknown))
            unknown_probe = {"language": unknown, "status": "UNEXPECTED_AVAILABLE"}
        except LanguageRegistryError as e:
            unknown_probe = {
                "language": unknown,
                "status": "NOT_AVAILABLE",
                "reason": e.reason,
                "message": str(e),
            }

    report = {
        "status": "AVAILABLE" if resolved and not errors else ("PARTIAL" if resolved else "NOT_AVAILABLE"),
        "languages": list_languages(),
        "benchmarks": list_benchmarks(),
        "packs": list_packs(),
        "resolved_packs": resolved,
        "errors": errors,
        "unknown_language_probe": unknown_probe,
    }

    out_dir = _resolve_out_dir(config_path, cfg.get("output_dir") or "results/21_language_pack")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "language_pack_audit.json", report)

    lines = [
        "# LinguaEval LanguagePack Inspect (P3-A)",
        "",
        f"- status: `{report['status']}`",
        f"- languages: `{report['languages']}`",
        f"- benchmarks: `{report['benchmarks']}`",
        f"- packs: `{report['packs']}`",
        "",
    ]
    for block in resolved:
        lang = (block.get("language") or {}).get("iso639_3")
        lines.append(f"## Pack `{block.get('pack_id')}` ({lang})")
        lines.append("")
        for cap, rows in (block.get("capabilities") or {}).items():
            lines.append(f"### capability `{cap}`")
            for r in rows:
                lines.append(
                    f"- `{r.get('benchmark_id')}`: status=`{r.get('status')}` "
                    f"reason=`{r.get('reason')}` native=`{r.get('native_authored')}`"
                )
            lines.append("")
    if unknown_probe:
        lines += [
            "## Unknown language probe",
            "",
            f"- `{unknown_probe}`",
            "",
        ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    provenance = build_provenance(
        config_path=config_path,
        cfg=cfg,
        task_path=None,
        output_path=None,
        metric_path=None,
        sample_dicts=[],
        prediction_dicts=[],
    )
    write_manifest(
        out_dir / "manifest.json",
        RunManifest(
            run_id=cfg.get("run_id") or f"language_inspect_{uuid4().hex[:8]}",
            config_path=str(config_path.resolve()),
            packs=list(cfg.get("packs_meta") or ["language"]),
            provenance=provenance,
            notes={"mode": "offline_language_inspect", "status": report["status"]},
            artifact_index={
                "language_pack_audit": str(out_dir / "language_pack_audit.json"),
                "report": str(report_path),
            },
        ),
    )
    return out_dir
