"""Adapter: existing N2S dialogue prediction JSON → SampleRecord + PredictionRecord.

Kernel remains unaware of N2S; this adapter lives under examples/adapters mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from linguaeval.core.schema import FormatStatus, PredictionRecord, SampleInput, SampleRecord


def _resolve(base: Path, maybe: Optional[str]) -> Optional[Path]:
    if not maybe:
        return None
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def _n2s_knowledge_to_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().upper() == "TRUE"


def _is_nested_dialogue_block(outer: Dict[str, Any]) -> bool:
    return isinstance(outer.get("turns"), list)


def flatten_dialogue_json(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block_idx, outer in enumerate(data):
        if not isinstance(outer, dict):
            continue
        if _is_nested_dialogue_block(outer):
            dialogue_id = outer.get("dialogue_id", block_idx)
            file_path = outer.get("file_path", "")
            for t in outer["turns"]:
                if not isinstance(t, dict):
                    continue
                turn_no = int(t.get("turn", 0))
                row = dict(t)
                row["dialogue_id"] = dialogue_id
                row["file_path"] = file_path
                row.setdefault("id", f"{dialogue_id}_{turn_no}")
                out.append(row)
        else:
            out.append(outer)
    return out


def _prediction_was_skipped(item: Dict[str, Any]) -> bool:
    steps = item.get("steps")
    if steps == "" or steps is None:
        return True
    if not isinstance(steps, dict):
        return True
    step = steps.get("n2s_prediction")
    if not isinstance(step, dict):
        return True
    return bool(step.get("skipped"))


def _n2s_prediction_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    steps = item.get("steps")
    if isinstance(steps, dict):
        pred = steps.get("n2s_prediction")
        if isinstance(pred, dict):
            inner = pred.get("result")
            if isinstance(inner, dict):
                return inner
            return pred
    return {}


def load_n2s_prediction_json(
    path: Path,
    *,
    model_id: str = "sft",
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    """Load nested/flat N2S result JSON; keep only rows that actually ran N2S."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"N2S prediction JSON must be a list: {path}")

    samples: List[SampleRecord] = []
    preds: List[PredictionRecord] = []
    for item in flatten_dialogue_json(data):
        if _prediction_was_skipped(item):
            continue
        pred = _n2s_prediction_from_item(item)
        sample_id = str(item.get("id") or f"{item.get('dialogue_id')}_{item.get('turn')}")
        gold_n2s = _n2s_knowledge_to_bool(item.get("n2s_knowledge"))
        samples.append(
            SampleRecord(
                sample_id=sample_id,
                input=SampleInput(text=str(item.get("content") or item.get("n2s_model_input") or "")),
                gold={
                    "n2s": gold_n2s,
                    "routing_skill": item.get("skill") or None,
                    "primary_intent": None,
                },
                meta={
                    "source": "n2s_dialogue_prediction",
                    "role": item.get("role"),
                    "language": "ind",
                },
                conversation={
                    "dialogue_id": item.get("dialogue_id"),
                    "turn_id": item.get("turn"),
                    "role": item.get("role"),
                    "context_mode": None,
                },
            )
        )
        format_ok = bool(pred.get("format_ok", True))
        latency_s = pred.get("time_cost")
        preds.append(
            PredictionRecord(
                sample_id=sample_id,
                model_id=model_id,
                raw_output=None,
                parsed={
                    "n2s": bool(pred.get("n2s", False)),
                    "routing_skill": pred.get("routing_skill"),
                    "primary_intent": pred.get("primary_intent", ""),
                },
                format=FormatStatus(
                    parse_ok=format_ok,
                    schema_ok=format_ok,
                    details={"source_format_ok": format_ok},
                ),
                timing={
                    "latency_ms": (float(latency_s) * 1000.0) if latency_s is not None else None,
                },
                meta={"adapter": "n2s_dialogue"},
            )
        )
    return samples, preds


def load_from_config(
    source: Dict[str, Any],
    config_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    """Registry entry for adapter name ``n2s_dialogue_prediction``."""
    pred_path = _resolve(
        config_dir, source.get("path") or source.get("predictions") or cfg.get("predictions")
    )
    if not pred_path or not pred_path.is_file():
        raise FileNotFoundError(f"N2S prediction JSON not found: {pred_path}")
    model_id = str(source.get("model_id") or "sft")
    return load_n2s_prediction_json(pred_path, model_id=model_id)
