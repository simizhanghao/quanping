"""Native-authored MC adapters (IndoMMLU-style / COPAL-style).

Provenance is taken from config — Kernel does not hardcode Indonesian business.
Gold labels require explicit ``answer_encoding`` (no 0/1-based guessing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from linguaeval.adapters.dataset.answer_encoding import (
    AnswerEncodingError,
    as_mc_letter,
    require_answer_encoding,
)
from linguaeval.core.schema import FormatStatus, PredictionRecord, SampleInput, SampleRecord


def _resolve(base: Path, maybe: Optional[str]) -> Optional[Path]:
    if not maybe:
        return None
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def _load_preds(path: Path) -> List[PredictionRecord]:
    preds: List[PredictionRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
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
    return preds


def indommlu_row_to_sample(
    row: Dict[str, Any],
    *,
    language: str,
    benchmark_id: str,
    capability: str,
    answer_encoding: str,
) -> SampleRecord:
    """IndoMMLU-ish: question + options A-D + answer letter/index."""
    sid = str(row.get("sample_id") or row.get("id") or row.get("link") or "")
    if not sid:
        raise ValueError("indommlu row requires sample_id|id|link")
    sample_id = sid if ":" in sid else f"indommlu:{language}:{sid}"
    question = str(row.get("question") or row.get("prompt") or "").strip()
    choices = row.get("choices") or row.get("options") or {}
    if isinstance(choices, dict) and choices:
        body = "\n".join(f"{k}. {choices[k]}" for k in sorted(choices.keys()))
    else:
        opts = []
        for i, letter in enumerate(["A", "B", "C", "D"], start=1):
            val = row.get(f"option_{letter}") or row.get(f"mc_answer{i}") or row.get(letter)
            if val is not None:
                opts.append(f"{letter}. {val}")
        body = "\n".join(opts)
    raw_gold = row.get("answer") if row.get("answer") is not None else row.get("correct_answer")
    gold = as_mc_letter(raw_gold, encoding=answer_encoding)
    text = f"{question}\n\n{body}".strip() if body else question
    return SampleRecord(
        sample_id=sample_id,
        gold={"answer": gold},
        input=SampleInput(text=text),
        meta={
            "language": language,
            "benchmark_id": benchmark_id,
            "capability": capability,
            "answer_encoding": answer_encoding,
            "subject": row.get("subject") or row.get("topic"),
            "provenance": {
                "origin": "native_authored",
                "translation": "native",
                "native_authored": True,
                "culture_sensitive": bool(row.get("culture_sensitive", False)),
            },
        },
    )


def copal_row_to_sample(
    row: Dict[str, Any],
    *,
    language: str,
    benchmark_id: str,
    capability: str,
    answer_encoding: str,
) -> SampleRecord:
    """COPAL-ID-ish: premise + choice1/choice2 + label index → A/B."""
    sid = str(row.get("sample_id") or row.get("id") or "")
    if not sid:
        raise ValueError("copal row requires sample_id|id")
    sample_id = sid if ":" in sid else f"copal:{language}:{sid}"
    premise = str(row.get("premise") or row.get("context") or "").strip()
    c1 = str(row.get("choice1") or row.get("option_A") or "").strip()
    c2 = str(row.get("choice2") or row.get("option_B") or "").strip()
    label = row.get("label")
    if label is None:
        label = row.get("answer")
    gold = as_mc_letter(label, encoding=answer_encoding)
    if gold not in {"A", "B"}:
        raise AnswerEncodingError(f"copal expects binary label mapped to A/B, got {gold}")
    text = f"{premise}\n\nA. {c1}\nB. {c2}".strip()
    return SampleRecord(
        sample_id=sample_id,
        gold={"answer": gold},
        input=SampleInput(text=text),
        meta={
            "language": language,
            "benchmark_id": benchmark_id,
            "capability": capability,
            "answer_encoding": answer_encoding,
            "variant": row.get("variant") or row.get("dialect"),
            "provenance": {
                "origin": "native_authored",
                "translation": "native",
                "native_authored": True,
                "culture_sensitive": True,
            },
        },
    )


def _resolve_encoding(source: Dict[str, Any], cfg: Dict[str, Any], *, where: str) -> str:
    return require_answer_encoding(
        source.get("answer_encoding")
        if source.get("answer_encoding") is not None
        else cfg.get("answer_encoding"),
        where=where,
    )


def _load_indommlu(
    source: Dict[str, Any],
    config_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    samples_path = _resolve(config_dir, source.get("samples") or cfg.get("samples"))
    if not samples_path or not samples_path.is_file():
        raise FileNotFoundError(f"indommlu samples not found: {samples_path}")
    encoding = _resolve_encoding(source, cfg, where="indommlu answer_encoding")
    language = str(source.get("language") or cfg.get("language") or "ind").strip().lower()
    benchmark_id = str(source.get("benchmark_id") or cfg.get("benchmark_id") or "indommlu")
    capability = str(source.get("capability") or cfg.get("capability") or "local_knowledge")
    samples: List[SampleRecord] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(
                indommlu_row_to_sample(
                    json.loads(line),
                    language=language,
                    benchmark_id=benchmark_id,
                    capability=capability,
                    answer_encoding=encoding,
                )
            )
    preds_path = _resolve(config_dir, source.get("predictions") or cfg.get("predictions"))
    preds = _load_preds(preds_path) if preds_path and preds_path.is_file() else []
    return samples, preds


def _load_copal(
    source: Dict[str, Any],
    config_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    samples_path = _resolve(config_dir, source.get("samples") or cfg.get("samples"))
    if not samples_path or not samples_path.is_file():
        raise FileNotFoundError(f"copal samples not found: {samples_path}")
    encoding = _resolve_encoding(source, cfg, where="copal answer_encoding")
    language = str(source.get("language") or cfg.get("language") or "ind").strip().lower()
    benchmark_id = str(source.get("benchmark_id") or cfg.get("benchmark_id") or "copal_id")
    capability = str(source.get("capability") or cfg.get("capability") or "cultural_reasoning")
    samples: List[SampleRecord] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(
                copal_row_to_sample(
                    json.loads(line),
                    language=language,
                    benchmark_id=benchmark_id,
                    capability=capability,
                    answer_encoding=encoding,
                )
            )
    preds_path = _resolve(config_dir, source.get("predictions") or cfg.get("predictions"))
    preds = _load_preds(preds_path) if preds_path and preds_path.is_file() else []
    return samples, preds


def load_indommlu_from_config(
    source: Dict[str, Any], config_dir: Path, cfg: Dict[str, Any]
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    return _load_indommlu(source, config_dir, cfg)


def load_copal_from_config(
    source: Dict[str, Any], config_dir: Path, cfg: Dict[str, Any]
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    return _load_copal(source, config_dir, cfg)
