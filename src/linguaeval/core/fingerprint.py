"""Content fingerprints for reproducibility (no business logic)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> Optional[str]:
    if not path or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def git_sha(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def fingerprint_records(records: Iterable[Dict[str, Any]]) -> str:
    return sha256_json(list(records))


def build_provenance(
    *,
    config_path: Path,
    cfg: Dict[str, Any],
    task_path: Optional[Path],
    output_path: Optional[Path],
    metric_path: Optional[Path],
    sample_dicts: List[Dict[str, Any]],
    prediction_dicts: List[Dict[str, Any]],
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = repo_root
    if root is None:
        for parent in [config_path.parent, *config_path.parents]:
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                root = parent
                break
    return {
        "git_sha": git_sha(root) if root else None,
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_file(config_path),
        "task_spec_hash": sha256_file(task_path) if task_path else None,
        "output_spec_hash": sha256_file(output_path) if output_path else None,
        "metric_spec_hash": sha256_file(metric_path) if metric_path else None,
        "dataset_fingerprint": fingerprint_records(sample_dicts),
        "prediction_fingerprint": fingerprint_records(prediction_dicts),
        "eligible_samples": len(sample_dicts),
        "prediction_rows": len(prediction_dicts),
        "adapter": (cfg.get("source") or {}).get("adapter")
        or (cfg.get("source") or {}).get("type"),
        "parse_mode": (cfg.get("parse") or {}).get("mode") or cfg.get("prediction_mode") or "from_parsed",
    }
