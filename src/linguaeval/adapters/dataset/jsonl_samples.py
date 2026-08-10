from __future__ import annotations

import json
from pathlib import Path
from typing import List

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
