"""lm-eval `--log_samples` → SampleRecord + PredictionRecord.

lm-eval remains an external executor. This adapter only converts sample dumps
into LinguaEval contracts so Kernel scorers/regression can be reused.
Does not import or depend on the `lm_eval` package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from linguaeval.core.schema import FormatStatus, PredictionRecord, SampleInput, SampleRecord

_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def _resolve(base: Path, maybe: Optional[str]) -> Optional[Path]:
    if not maybe:
        return None
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def _as_letter(raw: Any) -> str:
    if raw is None:
        raise ValueError("empty answer/target")
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise ValueError("empty answer string")
        # take first char if "A. something"
        head = s[0].upper()
        if head in {"A", "B", "C", "D", "E"}:
            return head
        if s.upper() in {"A", "B", "C", "D", "E"}:
            return s.upper()
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError) as e:
        raise ValueError(f"cannot map answer={raw!r} to letter") from e
    if n in _LETTER:
        return _LETTER[n]
    if 1 <= n <= 5:
        return _LETTER[n - 1]
    raise ValueError(f"answer index out of range: {n}")


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


def _pred_from_sample(sample: Dict[str, Any]) -> str:
    if "filtered_resps" in sample and sample["filtered_resps"] is not None:
        return _as_letter(sample["filtered_resps"])
    if "resps" in sample and sample["resps"] is not None:
        return _as_letter(sample["resps"])
    if sample.get("pred") is not None:
        return _as_letter(sample["pred"])
    raise ValueError("lm-eval sample missing filtered_resps/resps/pred")


def _gold_from_sample(sample: Dict[str, Any]) -> str:
    if sample.get("target") is not None:
        return _as_letter(sample["target"])
    doc = sample.get("doc") or {}
    if isinstance(doc, dict):
        for k in ("gold", "label", "answer", "correct_answer"):
            if doc.get(k) is not None:
                return _as_letter(doc[k])
    raise ValueError("lm-eval sample missing target/gold")


def load_lm_eval_samples_blob(path: Path, *, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load either JSONL rows or lm-eval samples JSON ({task: [rows]} / {samples: {...}})."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # JSONL if first non-empty line is object and file has multiple lines without wrapping array
    if "\n" in text and not text.lstrip().startswith("["):
        first = text.splitlines()[0].lstrip()
        if first.startswith("{"):
            # could still be pretty JSON object — detect by trying full parse
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
) -> Tuple[SampleRecord, PredictionRecord]:
    doc_id = sample.get("doc_id")
    if doc_id is None:
        doc_id = sample_index
    sample_id = str(sample.get("sample_id") or f"lmeval:{task_name}:{doc_id}")
    gold = _gold_from_sample(sample)
    pred = _pred_from_sample(sample)
    srec = SampleRecord(
        sample_id=sample_id,
        gold={"answer": gold},
        input=SampleInput(text=_prompt_text(sample)),
        meta={
            "language": language,
            "benchmark_id": benchmark_id,
            "capability": capability,
            "lm_eval_task": task_name,
            "doc_id": doc_id,
            "provenance": {
                "origin": "external_executor",
                "translation": "unknown",
                "executor": "lm-eval",
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
    task_name = str(source.get("task_name") or cfg.get("task_name") or "").strip() or None
    rows = load_lm_eval_samples_blob(path, task_name=task_name)
    language = str(source.get("language") or cfg.get("language") or "und").strip().lower()
    benchmark_id = str(source.get("benchmark_id") or cfg.get("benchmark_id") or "lm_eval")
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
        )
        samples.append(s)
        preds.append(p)
    return samples, preds
