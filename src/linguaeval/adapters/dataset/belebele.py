"""Belebele-format JSONL → SampleRecord (+ optional PredictionRecord).

Official-ish fields (FLORES/Belebele style):
  link, flores_passage, question, mc_answer1..4, correct_answer_num

Kernel never branches on language codes — language comes from config/meta only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from linguaeval.core.schema import FormatStatus, PredictionRecord, SampleInput, SampleRecord

_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}


def _resolve(base: Path, maybe: Optional[str]) -> Optional[Path]:
    if not maybe:
        return None
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def _correct_letter(row: Dict[str, Any]) -> str:
    raw = row.get("correct_answer_num")
    if raw is None:
        raw = row.get("correct_answer")
    if isinstance(raw, str) and raw.strip().upper() in {"A", "B", "C", "D"}:
        return raw.strip().upper()
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid correct_answer_num={raw!r}") from e
    if n not in _LETTER:
        raise ValueError(f"correct_answer_num out of range: {n}")
    return _LETTER[n]


def _build_text(row: Dict[str, Any]) -> str:
    passage = str(row.get("flores_passage") or row.get("passage") or "").strip()
    question = str(row.get("question") or "").strip()
    opts = []
    for i, letter in enumerate(["A", "B", "C", "D"], start=1):
        key = f"mc_answer{i}"
        val = row.get(key)
        if val is None:
            val = row.get(f"option_{letter.lower()}") or row.get(letter)
        opts.append(f"{letter}. {val}")
    body = "\n".join(opts)
    parts = []
    if passage:
        parts.append(passage)
    if question:
        parts.append(f"Question: {question}")
    parts.append(body)
    return "\n\n".join(parts)


def row_to_sample(
    row: Dict[str, Any],
    *,
    language: str,
    benchmark_id: str,
    default_id_prefix: str = "belebele",
) -> SampleRecord:
    link = str(row.get("link") or row.get("id") or row.get("sample_id") or "")
    if not link:
        raise ValueError("belebele row requires link|id|sample_id")
    sample_id = str(row.get("sample_id") or f"{default_id_prefix}:{language}:{link}")
    gold_letter = _correct_letter(row)
    return SampleRecord(
        sample_id=sample_id,
        gold={"answer": gold_letter},
        input=SampleInput(text=_build_text(row)),
        meta={
            "language": language,
            "benchmark_id": benchmark_id,
            "capability": "reading_comprehension",
            "link": link,
            "correct_answer_num": row.get("correct_answer_num"),
            "provenance": {"origin": "parallel", "translation": "parallel"},
        },
    )


def load_belebele_samples_jsonl(
    path: Path,
    *,
    language: str,
    benchmark_id: str,
) -> List[SampleRecord]:
    rows: List[SampleRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(
                row_to_sample(json.loads(line), language=language, benchmark_id=benchmark_id)
            )
    return rows


def load_from_config(
    source: Dict[str, Any],
    config_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    """Adapter entry: samples required; predictions optional (empty list if absent)."""
    samples_path = _resolve(config_dir, source.get("samples") or cfg.get("samples"))
    if not samples_path or not samples_path.is_file():
        raise FileNotFoundError(f"belebele samples not found: {samples_path}")
    language = str(source.get("language") or cfg.get("language") or "").strip().lower()
    if not language:
        raise ValueError("belebele adapter requires source.language (iso639_3)")
    benchmark_id = str(
        source.get("benchmark_id") or cfg.get("benchmark_id") or f"belebele_{language}"
    )
    samples = load_belebele_samples_jsonl(
        samples_path, language=language, benchmark_id=benchmark_id
    )

    preds_path = _resolve(
        config_dir, source.get("predictions") or cfg.get("predictions")
    )
    preds: List[PredictionRecord] = []
    if preds_path and preds_path.is_file():
        with preds_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # allow letter in parsed.answer or top-level answer
                parsed = dict(d.get("parsed") or {})
                if "answer" not in parsed and d.get("answer") is not None:
                    parsed["answer"] = d["answer"]
                fmt = d.get("format") or {}
                preds.append(
                    PredictionRecord(
                        sample_id=str(d["sample_id"]),
                        model_id=str(d.get("model_id") or "default"),
                        raw_output=d.get("raw_output"),
                        parsed=parsed,
                        scores=dict(d.get("scores") or {}),
                        format=FormatStatus(
                            parse_ok=bool(fmt.get("parse_ok", True)),
                            schema_ok=bool(fmt.get("schema_ok", True)),
                            details=dict(fmt.get("details") or {}),
                        ),
                        meta=dict(d.get("meta") or {}),
                    )
                )
    return samples, preds
