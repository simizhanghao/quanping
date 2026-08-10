"""lm-eval **multiple-choice** `--log_samples` → SampleRecord + PredictionRecord.

Scope (P3-D / P3-F-C): **MC tasks only**. Generation / free-form lm-eval dumps are
out of scope — use a future ``lm_eval_generation_samples`` adapter.

lm-eval remains an external executor. This adapter only converts MC sample dumps
into LinguaEval contracts so Kernel scorers/regression can be reused.
Does not import or depend on the ``lm_eval`` package.

Requires explicit ``answer_encoding``: ``letter`` | ``zero_based_index`` |
``one_based_index``.
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

ADAPTER_KIND = "multiple_choice"
ADAPTER_NAMES = ("lm_eval_samples", "lm_eval_mc_samples")


def _resolve(base: Path, maybe: Optional[str]) -> Optional[Path]:
    if not maybe:
        return None
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def _prompt_text(sample: Dict[str, Any]) -> str:
    args = sample.get("arguments")
    if isinstance(args, list) and args:
        first = args[0]
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        if isinstance(first, str):
            return first
    doc = sample.get("doc") or {}
    if isinstance(doc, dict):
        for k in ("question", "query", "input", "prompt", "text"):
            if doc.get(k):
                return str(doc[k])
    return json.dumps(doc, ensure_ascii=False) if doc else ""


def _looks_like_generation(raw: Any) -> bool:
    if not isinstance(raw, str):
        return False
    s = raw.strip()
    if not s:
        return False
    if len(s) <= 2:
        return False
    # "A. foo" style still MC; long free text is generation
    head = s[0].upper()
    if head in {"A", "B", "C", "D", "E"} and s[1] in {".", ")", ":", " "}:
        return False
    if s.upper() in {"A", "B", "C", "D", "E"}:
        return False
    return len(s.split()) >= 3 or len(s) > 8


def _pred_from_sample(sample: Dict[str, Any], *, encoding: str) -> str:
    raw = None
    if "filtered_resps" in sample and sample["filtered_resps"] is not None:
        raw = sample["filtered_resps"]
    elif "resps" in sample and sample["resps"] is not None:
        raw = sample["resps"]
    elif sample.get("pred") is not None:
        raw = sample["pred"]
    else:
        raise AnswerEncodingError("lm-eval MC sample missing filtered_resps/resps/pred")
    if _looks_like_generation(raw if not isinstance(raw, (list, tuple)) else (raw[0] if raw else None)):
        raise AnswerEncodingError(
            "lm_eval_samples / lm_eval_mc_samples is MC-only; "
            "generation-style filtered_resps detected. "
            "Use a future lm_eval_generation_samples adapter."
        )
    return as_mc_letter(raw, encoding=encoding)


def _gold_from_sample(sample: Dict[str, Any], *, encoding: str) -> str:
    if sample.get("target") is not None:
        return as_mc_letter(sample["target"], encoding=encoding)
    doc = sample.get("doc") or {}
    if isinstance(doc, dict):
        for k in ("gold", "label", "answer", "correct_answer"):
            if doc.get(k) is not None:
                return as_mc_letter(doc[k], encoding=encoding)
    raise AnswerEncodingError("lm-eval MC sample missing target/gold")


def load_lm_eval_samples_blob(path: Path, *, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load either JSONL rows or lm-eval samples JSON ({task: [rows]} / {samples: {...}})."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if "\n" in text and not text.lstrip().startswith("["):
        first = text.splitlines()[0].lstrip()
        if first.startswith("{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                rows = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
                return rows
            else:
                return _extract_task_rows(obj, task_name=task_name)
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return _extract_task_rows(obj, task_name=task_name)
    raise ValueError(f"unsupported lm-eval samples payload type: {type(obj)}")


def _extract_task_rows(obj: Dict[str, Any], *, task_name: Optional[str]) -> List[Dict[str, Any]]:
    samples = obj.get("samples") if isinstance(obj.get("samples"), dict) else obj
    if not isinstance(samples, dict):
        raise ValueError("expected samples mapping task_name → list")
    if task_name:
        if task_name not in samples:
            raise KeyError(f"task {task_name!r} not in lm-eval samples; have {sorted(samples)}")
        rows = samples[task_name]
    elif len(samples) == 1:
        rows = next(iter(samples.values()))
    else:
        raise ValueError(
            f"multiple tasks in lm-eval samples ({sorted(samples)}); set source.task_name"
        )
    if not isinstance(rows, list):
        raise ValueError("samples[task] must be a list")
    return rows


def sample_to_records(
    sample: Dict[str, Any],
    *,
    language: str,
    benchmark_id: str,
    task_name: str,
    model_id: str,
    capability: str,
    sample_index: int,
    answer_encoding: str,
) -> Tuple[SampleRecord, PredictionRecord]:
    doc_id = sample.get("doc_id")
    if doc_id is None:
        doc_id = sample_index
    sample_id = str(sample.get("sample_id") or f"lmeval:{task_name}:{doc_id}")
    gold = _gold_from_sample(sample, encoding=answer_encoding)
    pred = _pred_from_sample(sample, encoding=answer_encoding)
    srec = SampleRecord(
        sample_id=sample_id,
        gold={"answer": gold},
        input=SampleInput(text=_prompt_text(sample)),
        meta={
            "language": language,
            "benchmark_id": benchmark_id,
            "capability": capability,
            "lm_eval_task": task_name,
            "adapter_kind": ADAPTER_KIND,
            "answer_encoding": answer_encoding,
            "doc_id": doc_id,
            "provenance": {
                "origin": "external_executor",
                "translation": "unknown",
                "executor": "lm-eval",
                "adapter": "lm_eval_mc_samples",
            },
        },
    )
    prec = PredictionRecord(
        sample_id=sample_id,
        model_id=model_id,
        raw_output=json.dumps(sample.get("filtered_resps"), ensure_ascii=False),
        parsed={"answer": pred},
        format=FormatStatus(parse_ok=True, schema_ok=True),
        meta={
            "lm_eval_metrics": {
                k: sample[k]
                for k in sample.keys()
                if k
                not in {
                    "doc_id",
                    "doc",
                    "target",
                    "arguments",
                    "resps",
                    "filtered_resps",
                    "filter",
                    "metrics",
                    "doc_hash",
                    "prompt_hash",
                    "target_hash",
                }
                and not isinstance(sample[k], (dict, list))
            }
        },
    )
    return srec, prec


def load_from_config(
    source: Dict[str, Any],
    config_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[List[SampleRecord], List[PredictionRecord]]:
    path = _resolve(
        config_dir,
        source.get("samples")
        or source.get("log_samples")
        or source.get("lm_eval_samples")
        or cfg.get("samples"),
    )
    if not path or not path.is_file():
        raise FileNotFoundError(f"lm-eval samples not found: {path}")
    encoding = require_answer_encoding(
        source.get("answer_encoding")
        if source.get("answer_encoding") is not None
        else cfg.get("answer_encoding"),
        where="lm_eval_mc_samples answer_encoding",
    )
    task_name = str(source.get("task_name") or cfg.get("task_name") or "").strip() or None
    rows = load_lm_eval_samples_blob(path, task_name=task_name)
    language = str(source.get("language") or cfg.get("language") or "und").strip().lower()
    benchmark_id = str(source.get("benchmark_id") or cfg.get("benchmark_id") or "lm_eval_mc")
    capability = str(source.get("capability") or cfg.get("capability") or "external_benchmark")
    model_id = str(source.get("model_id") or cfg.get("model_id") or "lm_eval")
    resolved_task = task_name or "default"
    samples: List[SampleRecord] = []
    preds: List[PredictionRecord] = []
    for i, row in enumerate(rows):
        s, p = sample_to_records(
            row,
            language=language,
            benchmark_id=benchmark_id,
            task_name=resolved_task,
            model_id=model_id,
            capability=capability,
            sample_index=i,
            answer_encoding=encoding,
        )
        samples.append(s)
        preds.append(p)
    return samples, preds
