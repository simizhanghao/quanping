from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from linguaeval.core.schema import PredictionRecord, SampleRecord


def load_samples_jsonl(path: Path) -> List[SampleRecord]:
    rows: List[SampleRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(SampleRecord.from_dict(json.loads(line)))
    return rows


def load_predictions_jsonl(path: Path) -> List[PredictionRecord]:
    rows: List[PredictionRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(PredictionRecord.from_dict(json.loads(line)))
    return rows


def write_predictions_jsonl(path: Path, preds: List[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def _resolve(base: Path, maybe: Optional[str]) -> Optional[Path]:
    if not maybe:
        return None
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def load_from_config(
    source: Dict[str, Any],
    config_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    """Generic paired jsonl adapter: SampleRecord + PredictionRecord files."""
    samples_path = _resolve(
        config_dir, source.get("samples") or cfg.get("samples")
    )
    preds_path = _resolve(
        config_dir, source.get("predictions") or cfg.get("predictions")
    )
    if not samples_path or not samples_path.is_file():
        raise FileNotFoundError(f"samples not found: {samples_path}")
    if not preds_path or not preds_path.is_file():
        raise FileNotFoundError(f"predictions not found: {preds_path}")
    return load_samples_jsonl(samples_path), load_predictions_jsonl(preds_path)
